"""Retrieval ranking benchmark for the question/answer knowledge base.

Searches the production index with each benchmark query and scores where the
gold example ranks. Two views are reported:

- Ranking quality without the similarity floor: Hit@k, MRR, NDCG@k.
- What the tool actually serves with the production top-k and floor: how often
  the gold example is served, and how often nothing is.

Each query has one gold example, so Recall@k equals Hit@k. The dataset holds
many near-identical concerns, and ranking another equally relevant example
above the source counts as a miss, so read the scores as a lower bound.

Usage:
    cd evaluation/
    uv run --project ../backend python build_ir_benchmark.py   # once
    uv run --project ../backend python eval_retrieval.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(EVAL_DIR))
load_dotenv(ROOT / "backend" / ".env")

from app.core.rag.retriever import search_examples  # noqa: E402
from config.settings import settings  # noqa: E402
from metrics.ir_metrics import hit_at_k, ndcg_at_k, reciprocal_rank  # noqa: E402

NO_FLOOR = -1.0


async def evaluate(queries: list[dict], top_ks: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_k = max(top_ks)
    rows = []
    for item in queries:
        gold = {item["gold_id"]}
        ranked = await search_examples(item["query"], k=max_k, min_similarity=NO_FLOOR)
        ranked_ids = [example.id for example in ranked]
        served = [
            example
            for example in ranked[: settings.RAG_TOP_K]
            if example.similarity >= settings.RAG_MIN_SIMILARITY
        ]
        row = {
            "query_id": item["query_id"],
            "gold_rank": ranked_ids.index(item["gold_id"]) + 1 if item["gold_id"] in ranked_ids else None,
            "top1_similarity": round(ranked[0].similarity, 4) if ranked else None,
            "mrr": reciprocal_rank(ranked_ids, gold),
            "served_gold": float(any(example.id == item["gold_id"] for example in served)),
            "served_nothing": float(not served),
        }
        for k in top_ks:
            row[f"hit@{k}"] = hit_at_k(ranked_ids, gold, k)
            row[f"ndcg@{k}"] = ndcg_at_k(ranked_ids, gold, k)
        rows.append(row)

    results = pd.DataFrame(rows)
    metric_columns = [c for c in results.columns if c not in ("query_id", "gold_rank", "top1_similarity")]
    summary = pd.DataFrame(
        [{"metric": column, "n": len(results), "value": round(results[column].mean(), 4)} for column in metric_columns]
    )
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the retrieval benchmark")
    parser.add_argument("--benchmark", type=Path, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()

    with (EVAL_DIR / "configs" / "retrieval_eval.yaml").open() as f:
        cfg = yaml.safe_load(f)
    benchmark_path = args.benchmark or EVAL_DIR / cfg["benchmark_path"]
    if not benchmark_path.exists():
        raise SystemExit(f"{benchmark_path} not found. Run build_ir_benchmark.py first.")
    with benchmark_path.open() as f:
        queries = json.load(f)["queries"]
    if args.max_queries:
        queries = queries[: args.max_queries]

    results, summary = asyncio.run(evaluate(queries, cfg["top_k_eval"]))
    RESULTS_DIR.mkdir(exist_ok=True)
    results.to_csv(RESULTS_DIR / "retrieval_eval_results.csv", index=False)
    summary.to_csv(RESULTS_DIR / "retrieval_eval_summary.csv", index=False)
    print(
        f"collection={settings.RAG_COLLECTION} embedding={settings.EMBEDDING_MODEL} "
        f"top_k={settings.RAG_TOP_K} floor={settings.RAG_MIN_SIMILARITY}"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
