"""
RAG Parameter Evaluation Runner.

Tests different combinations of chunk_size, chunk_overlap, and top_k
to find the optimal RAG configuration.

Usage:
    cd evaluation/
    uv run python -m evaluation.eval_rag [--max-queries N]
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
from tqdm import tqdm

# Add parent dirs to path for imports
ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(EVAL_ROOT))

# Load backend .env so API keys are available
from dotenv import load_dotenv
load_dotenv(ROOT / "backend" / ".env")

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from metrics.judge import JudgeScores, LLMJudge, RetrievalRelevanceJudge, _create_llm

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = EVAL_DIR / "configs"
DATASETS_DIR = EVAL_DIR / "datasets"
RESULTS_DIR = EVAL_DIR / "results"


def load_config() -> dict:
    with open(CONFIGS_DIR / "rag_params.yaml") as f:
        return yaml.safe_load(f)


def load_queries(max_queries: int | None = None) -> list[dict]:
    with open(DATASETS_DIR / "eval_queries.json") as f:
        queries = json.load(f)
    if max_queries:
        queries = queries[:max_queries]
    return queries


def build_vectorstore_for_config(
    chunk_size: int,
    chunk_overlap: int,
    collection_name: str,
    max_samples: int | None = None,
) -> None:
    """Build a pgvector collection with specific chunking parameters."""
    from app.core.rag.vector_store import VectorStore

    store = VectorStore(
        backend="pgvector",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        collection_name=collection_name,
        max_samples=max_samples,
    )
    store.build()


def get_retriever_for_config(
    chunk_size: int,
    chunk_overlap: int,
    collection_name: str,
    top_k: int,
):
    """Get a retriever for a specific pgvector collection."""
    from app.core.rag.vector_store import VectorStore

    store = VectorStore(
        backend="pgvector",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        collection_name=collection_name,
    )
    store.load()
    return store.get_retriever(k=top_k)


def generate_response(
    query: str,
    retrieved_docs: list[str],
    llm,
) -> str:
    """Generate a therapy response using retrieved context."""
    system_prompt = (
        "You are a compassionate mental health assistant practicing Rogerian therapy principles. "
        "Show unconditional positive regard and genuine empathy. Reflect users' feelings. "
        "Be non-directive: explore rather than advise. "
        "Never provide medical diagnoses or prescribe medication."
    )

    context = ""
    if retrieved_docs:
        examples = "\n\n".join(
            f"[Example {i}]\n{doc}" for i, doc in enumerate(retrieved_docs, 1)
        )
        context = (
            f"\n\nRelevant counselor examples for reference:\n{examples}\n\n"
            "Use these examples to guide your response style, but paraphrase rather than copy."
        )

    messages = [
        SystemMessage(content=system_prompt + context),
        HumanMessage(content=query),
    ]

    result = llm.invoke(messages)
    return result.content


def run_rag_evaluation(max_queries: int | None = None) -> pd.DataFrame:
    """Run the full RAG parameter grid search evaluation."""
    config = load_config()
    queries = load_queries(max_queries or config.get("num_eval_queries", 50))

    chunk_sizes = config["chunk_sizes"]
    chunk_overlaps = config["chunk_overlaps"]
    top_k_values = config["top_k_values"]

    provider = config.get("eval_provider", "local")
    base_url = config.get("eval_base_url", "http://localhost:1234/v1")

    judge = LLMJudge(
        model=config["eval_model"],
        temperature=config["judge_temperature"],
        provider=provider,
        base_url=base_url,
    )
    relevance_judge = RetrievalRelevanceJudge(
        model=config["eval_model"],
        temperature=config["judge_temperature"],
        provider=provider,
        base_url=base_url,
    )
    response_llm = _create_llm(
        model=config["eval_model"],
        temperature=0.3,
        provider=provider,
        base_url=base_url,
    )

    # Step 1: Build vector stores for each unique (chunk_size, chunk_overlap) pair
    built_collections: dict[tuple[int, int], str] = {}
    chunking_configs = list(product(chunk_sizes, chunk_overlaps))

    print(f"\n{'='*60}")
    print(f"RAG Parameter Evaluation")
    print(f"{'='*60}")
    print(f"Chunk sizes: {chunk_sizes}")
    print(f"Chunk overlaps: {chunk_overlaps}")
    print(f"Top-k values: {top_k_values}")
    print(f"Queries: {len(queries)}")
    print(f"Total configs: {len(chunking_configs) * len(top_k_values)}")
    print(f"Total evaluations: {len(chunking_configs) * len(top_k_values) * len(queries)}")
    print(f"{'='*60}\n")

    # Check which collections already exist in pgvector
    existing_collections = set()
    try:
        import psycopg
        conn = psycopg.connect(settings.DATABASE_URL.replace("+psycopg", ""))
        cur = conn.execute("SELECT name FROM langchain_pg_collection")
        existing_collections = {row[0] for row in cur.fetchall()}
        conn.close()
    except Exception:
        pass

    for chunk_size, chunk_overlap in tqdm(chunking_configs, desc="Building indexes"):
        if chunk_overlap >= chunk_size:
            print(f"  Skipping overlap={chunk_overlap} >= chunk_size={chunk_size}")
            continue

        collection = f"eval_cs{chunk_size}_co{chunk_overlap}"
        if collection in existing_collections:
            print(f"  Reusing existing: {collection}")
        else:
            print(f"  Building: chunk_size={chunk_size}, overlap={chunk_overlap} -> {collection}")
            max_samples = config.get("max_dataset_samples")
            build_vectorstore_for_config(chunk_size, chunk_overlap, collection, max_samples)
        built_collections[(chunk_size, chunk_overlap)] = collection

    # Step 2: Evaluate each (chunk_size, chunk_overlap, top_k) combination
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "rag_eval_results.csv"
    results = []
    total_combos = len(built_collections) * len(top_k_values)
    combo_idx = 0

    for (chunk_size, chunk_overlap), collection in built_collections.items():
        for top_k in top_k_values:
            combo_idx += 1
            print(
                f"\n[{combo_idx}/{total_combos}] Evaluating: "
                f"chunk_size={chunk_size}, overlap={chunk_overlap}, top_k={top_k}"
            )

            retriever = get_retriever_for_config(
                chunk_size, chunk_overlap, collection, top_k
            )

            for q in tqdm(queries, desc=f"  cs={chunk_size} co={chunk_overlap} k={top_k}", leave=False):
                query_text = q["query"]

                # Retrieve
                t0 = time.time()
                docs = retriever.invoke(query_text)
                retrieval_time = time.time() - t0
                doc_texts = [d.page_content for d in docs]

                # Score retrieval relevance
                relevance = relevance_judge.score(query_text, doc_texts)

                # Generate response
                t0 = time.time()
                response = generate_response(query_text, doc_texts, response_llm)
                response_time = time.time() - t0

                # Judge response quality
                scores = judge.score(query_text, response)

                results.append(
                    {
                        "query_id": q["id"],
                        "category": q["category"],
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "top_k": top_k,
                        "retrieval_relevance": round(relevance, 3),
                        "retrieval_time_s": round(retrieval_time, 3),
                        "response_time_s": round(response_time, 3),
                        **scores.to_dict(),
                    }
                )

            # Save incrementally after each config completes
            pd.DataFrame(results).to_csv(output_path, index=False)
            print(f"  (incremental save: {len(results)} rows -> {output_path})")

    df = pd.DataFrame(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "rag_eval_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    print(f"Total evaluations: {len(df)}")

    # Print summary
    summary = (
        df.groupby(["chunk_size", "chunk_overlap", "top_k"])
        .agg(
            {
                "average": "mean",
                "retrieval_relevance": "mean",
                "retrieval_time_s": "mean",
                "response_time_s": "mean",
                "empathy": "mean",
                "helpfulness": "mean",
                "safety": "mean",
            }
        )
        .round(3)
    )
    summary_path = RESULTS_DIR / "rag_eval_summary.csv"
    summary.to_csv(summary_path)
    print(f"Summary saved to {summary_path}")
    print(f"\nTop 5 configurations by average score:")
    print(summary.sort_values("average", ascending=False).head(5))

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG parameter evaluation")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit number of eval queries (for quick testing)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_rag_evaluation(max_queries=args.max_queries)
