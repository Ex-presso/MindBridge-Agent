"""Retrieval IR benchmark runner.

Sweeps (chunk_size × chunk_overlap) configurations and reports standard
information-retrieval metrics (Recall@k, Hit@k, MRR, NDCG@k, MAP) against
a curated benchmark of MentalChat16K (input → output) pairs.

Gold construction:
    For each benchmark query, gold relevance is "cluster gold" — the set of
    all chunks produced by splitting the source example's `output` field
    with the same chunk_size/chunk_overlap as the index. Chunk identity is
    the composite (source_example_idx, chunk_idx).

Usage:
    cd evaluation/
    uv run --project ../backend python eval_retrieval.py
    uv run --project ../backend python eval_retrieval.py --no-build
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from itertools import product
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(EVAL_ROOT))

load_dotenv(ROOT / "backend" / ".env")

from config.settings import settings  # noqa: E402
from metrics.ir_metrics import (  # noqa: E402
    average_precision,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = EVAL_DIR / "configs"
DATASETS_DIR = EVAL_DIR / "datasets"
RESULTS_DIR = EVAL_DIR / "results"


def load_config() -> dict:
    with (CONFIGS_DIR / "retrieval_eval.yaml").open() as f:
        return yaml.safe_load(f)


def load_benchmark(path: Path) -> list[dict]:
    with path.open() as f:
        data = json.load(f)
    return data["queries"]


def gold_chunk_ids(reference_response: str, source_example_idx: int,
                   chunk_size: int, chunk_overlap: int) -> set[tuple[int, int]]:
    """Re-chunk the reference output to recover the same chunk IDs the index has."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(reference_response)
    return {(source_example_idx, i) for i in range(len(chunks))}


def list_existing_collections() -> set[str]:
    """Return non-empty collection names already in pgvector.

    Empty collections (e.g. partial inits left behind by a failed build) are
    excluded so the next run rebuilds them rather than treating them as cached.
    """
    try:
        import psycopg

        with psycopg.connect(settings.DATABASE_URL.replace("+psycopg", "")) as conn:
            cur = conn.execute(
                """
                SELECT c.name
                FROM langchain_pg_collection c
                WHERE EXISTS (
                    SELECT 1 FROM langchain_pg_embedding e
                    WHERE e.collection_id = c.uuid
                )
                """
            )
            return {row[0] for row in cur.fetchall()}
    except Exception as e:
        logger.warning("Could not query existing collections: %s", e)
        return set()


def ensure_index(chunk_size: int, chunk_overlap: int, collection: str,
                 max_samples: int | None, existing: set[str]) -> None:
    """Build the pgvector collection if it doesn't already exist."""
    if collection in existing:
        print(f"  Reusing existing index: {collection}")
        return
    from app.core.rag.vector_store import VectorStore

    print(f"  Building: cs={chunk_size}, co={chunk_overlap} -> {collection}")
    store = VectorStore(
        backend="pgvector",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        collection_name=collection,
        max_samples=max_samples,
    )
    store.build()


def get_retriever(chunk_size: int, chunk_overlap: int, collection: str, k: int):
    from app.core.rag.vector_store import VectorStore

    store = VectorStore(
        backend="pgvector",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        collection_name=collection,
    )
    store.load()
    return store.get_retriever(k=k)


def run_retrieval_eval(skip_build: bool = False) -> pd.DataFrame:
    config = load_config()
    benchmark_path = EVAL_DIR / config["benchmark_path"]
    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"Benchmark not found at {benchmark_path}. "
            "Run `uv run --project ../backend python build_ir_benchmark.py` first."
        )
    queries = load_benchmark(benchmark_path)
    print(f"Loaded {len(queries)} queries from {benchmark_path}")

    chunk_sizes = config["chunk_sizes"]
    chunk_overlaps = config["chunk_overlaps"]
    top_k_eval = sorted(config["top_k_eval"])
    max_k = max(top_k_eval)
    prefix = config.get("collection_prefix", "ir_eval")
    max_samples = config.get("max_dataset_samples")

    chunking_configs = [
        (cs, co)
        for cs, co in product(chunk_sizes, chunk_overlaps)
        if co < cs
    ]
    print(f"\n{'=' * 60}")
    print("Retrieval IR Evaluation")
    print(f"{'=' * 60}")
    print(f"Chunk sizes: {chunk_sizes}")
    print(f"Chunk overlaps: {chunk_overlaps}")
    print(f"top_k values: {top_k_eval}")
    print(f"Configs: {len(chunking_configs)}")
    print(f"Total query evaluations: {len(chunking_configs) * len(queries)}")
    print(f"{'=' * 60}\n")

    # Step 1: ensure all required indexes exist
    existing = set() if skip_build else list_existing_collections()
    if not skip_build:
        for cs, co in tqdm(chunking_configs, desc="Building indexes"):
            collection = f"{prefix}_cs{cs}_co{co}"
            ensure_index(cs, co, collection, max_samples, existing)

    # Step 2: evaluate each config
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_query_path = RESULTS_DIR / "retrieval_eval_results.csv"
    summary_path = RESULTS_DIR / "retrieval_eval_summary.csv"

    rows: list[dict] = []
    for cs, co in chunking_configs:
        collection = f"{prefix}_cs{cs}_co{co}"
        print(f"\n[cs={cs}, co={co}] collection={collection}")

        retriever = get_retriever(cs, co, collection, max_k)

        for q in tqdm(queries, desc=f"  cs={cs} co={co}", leave=False):
            gold = gold_chunk_ids(
                q["reference_response"], q["source_example_idx"], cs, co
            )
            t0 = time.time()
            docs = retriever.invoke(q["query"])
            retrieval_time = time.time() - t0

            retrieved_ids = [
                (
                    int(d.metadata.get("source_example_idx", -1)),
                    int(d.metadata.get("chunk_idx", -1)),
                )
                for d in docs
            ]

            for k in top_k_eval:
                rows.append(
                    {
                        "query_id": q["query_id"],
                        "source_example_idx": q["source_example_idx"],
                        "chunk_size": cs,
                        "chunk_overlap": co,
                        "top_k": k,
                        "n_gold_chunks": len(gold),
                        "recall_at_k": round(recall_at_k(retrieved_ids, gold, k), 4),
                        "precision_at_k": round(
                            precision_at_k(retrieved_ids, gold, k), 4
                        ),
                        "hit_at_k": round(hit_at_k(retrieved_ids, gold, k), 4),
                        "mrr_at_k": round(reciprocal_rank(retrieved_ids[:k], gold), 4),
                        "ndcg_at_k": round(ndcg_at_k(retrieved_ids, gold, k), 4),
                        "ap_at_k": round(average_precision(retrieved_ids[:k], gold), 4),
                        "retrieval_time_s": round(retrieval_time, 4),
                    }
                )

        # Incremental save after each config completes
        pd.DataFrame(rows).to_csv(per_query_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(per_query_path, index=False)
    print(f"\nPer-query results: {per_query_path} ({len(df)} rows)")

    summary = (
        df.groupby(["chunk_size", "chunk_overlap", "top_k"])
        .agg(
            recall_at_k=("recall_at_k", "mean"),
            precision_at_k=("precision_at_k", "mean"),
            hit_at_k=("hit_at_k", "mean"),
            mrr_at_k=("mrr_at_k", "mean"),
            ndcg_at_k=("ndcg_at_k", "mean"),
            map_at_k=("ap_at_k", "mean"),
            retrieval_time_s=("retrieval_time_s", "mean"),
            n_queries=("query_id", "nunique"),
        )
        .round(4)
    )
    summary.to_csv(summary_path)
    print(f"Summary: {summary_path}")
    print("\nTop configs by NDCG@5:")
    if 5 in top_k_eval:
        print(
            summary.xs(5, level="top_k")
            .sort_values("ndcg_at_k", ascending=False)
            .head(10)
            .to_string()
        )
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval IR benchmark runner")
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip index building; assume all required collections exist",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_retrieval_eval(skip_build=args.no_build)
