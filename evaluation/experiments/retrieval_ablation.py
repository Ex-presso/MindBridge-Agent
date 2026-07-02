"""Retrieval ablation: does any standard IR technique beat dense-only?

Compares, on the 30-query IR benchmark (cluster gold), the dense bi-encoder
baseline against: Qwen3 query-instruction prefix, BM25, hybrid BM25+dense
(RRF), and a cross-encoder reranker over dense candidates.

Finding (see docs/EVALUATION.md): none beat dense-only, because the task
matches a problem to its *complementary* counselor response, not to a similar
passage. Documents come from MentalChat16K `output`; BM25 corpus is re-derived
with the same params so it matches the pgvector index.

Usage:
    cd evaluation
    uv run python experiments/retrieval_ablation.py            # build if needed
    uv run python experiments/retrieval_ablation.py --no-build
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "evaluation"
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(EVAL))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / "backend" / ".env")

from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402
from metrics.ir_metrics import hit_at_k, ndcg_at_k, recall_at_k  # noqa: E402

CS, CO, COLL = 1000, 100, "ablation_cs1000_co100"
GENERIC_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
BENCH = json.load((EVAL / "datasets" / "ir_benchmark.json").open())["queries"]


def tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def gold_ids(ref: str, idx: int) -> set[tuple[int, int]]:
    sp = RecursiveCharacterTextSplitter(chunk_size=CS, chunk_overlap=CO)
    return {(idx, i) for i in range(len(sp.split_text(ref)))}


def doc_id(d) -> tuple[int, int]:
    return (int(d.metadata.get("source_example_idx", -1)), int(d.metadata.get("chunk_idx", -1)))


def rrf(rankings: list[list], k: int = 60) -> list:
    scores: dict = {}
    for ranking in rankings:
        for rank, d in enumerate(ranking, 1):
            scores[d] = scores.get(d, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda d: -scores[d])


def build_if_needed(skip_build: bool) -> None:
    import config.settings as sm
    import psycopg

    conn = psycopg.connect(sm.settings.DATABASE_URL.replace("+psycopg", ""))
    exists = conn.execute(
        "SELECT 1 FROM langchain_pg_collection c JOIN langchain_pg_embedding e "
        "ON e.collection_id=c.uuid WHERE c.name=%s LIMIT 1",
        (COLL,),
    ).fetchone()
    conn.close()
    if exists or skip_build:
        print(f"{'reusing' if exists else 'skip-build (missing!)'} collection {COLL}")
        return
    from app.core.rag.vector_store import VectorStore

    print(f"building {COLL} (cs={CS}, co={CO}, max_samples=3000)...")
    VectorStore(backend="pgvector", chunk_size=CS, chunk_overlap=CO,
                collection_name=COLL, max_samples=3000).build()


def dense_retriever(instruction: str):
    import config.settings as sm
    sm.settings.EMBEDDING_QUERY_INSTRUCTION = instruction
    from app.core.rag.vector_store import VectorStore

    store = VectorStore(backend="pgvector", chunk_size=CS, chunk_overlap=CO, collection_name=COLL)
    store.load()
    return store.get_retriever(k=20)


def build_bm25():
    from datasets import load_dataset
    from rank_bm25 import BM25Okapi

    ds = load_dataset("ShenLab/MentalChat16K", split="train").select(range(3000))
    sp = RecursiveCharacterTextSplitter(chunk_size=CS, chunk_overlap=CO)
    ids, toks = [], []
    for ex_idx, ex in enumerate(ds):
        for ci, ch in enumerate(sp.split_text(ex.get("output") or "")):
            ids.append((ex_idx, ci))
            toks.append(tok(ch))
    return BM25Okapi(toks), ids


def mean_metrics(rows):
    n = len(rows)
    return tuple(sum(r[i] for r in rows) / n for i in range(3))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()
    build_if_needed(args.no_build)

    dense = dense_retriever("")
    dense_instr = dense_retriever(GENERIC_INSTRUCTION)
    bm25, bm25_ids = build_bm25()
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

    results: dict[str, list] = {k: [] for k in ["dense", "dense+instr", "bm25", "hybrid", "dense+rerank"]}
    for q in BENCH:
        gold = gold_ids(q["reference_response"], q["source_example_idx"])
        d_docs = dense.invoke(q["query"])
        d = [doc_id(x) for x in d_docs]
        di = [doc_id(x) for x in dense_instr.invoke(q["query"])]
        bscores = bm25.get_scores(tok(q["query"]))
        b = [bm25_ids[i] for i in np.argsort(bscores)[::-1][:20]]
        h = rrf([d, b])
        ce_scores = ce.predict([(q["query"], x.page_content) for x in d_docs])
        rr = [doc_id(x) for x, _ in sorted(zip(d_docs, ce_scores), key=lambda t: -t[1])]
        for name, ids in [("dense", d), ("dense+instr", di), ("bm25", b), ("hybrid", h), ("dense+rerank", rr)]:
            results[name].append((ndcg_at_k(ids, gold, 5), recall_at_k(ids, gold, 5), hit_at_k(ids, gold, 5)))

    print(f"\n{'method':<14} {'NDCG@5':>8} {'Recall@5':>9} {'Hit@5':>7}")
    for name, rows in results.items():
        nd, rc, ht = mean_metrics(rows)
        print(f"{name:<14} {nd:>8.4f} {rc:>9.4f} {ht:>7.4f}")


if __name__ == "__main__":
    main()
