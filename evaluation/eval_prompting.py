"""
Prompting Strategy Evaluation Runner.

Compares Rogerian, CBT, and Baseline prompting strategies,
each tested with and without RAG augmentation.

Usage:
    cd evaluation/
    uv run python -m evaluation.eval_prompting [--max-queries N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(EVAL_ROOT))

# Load backend .env so API keys are available
from dotenv import load_dotenv
load_dotenv(ROOT / "backend" / ".env")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from metrics.judge import LLMJudge

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = EVAL_DIR / "configs"
DATASETS_DIR = EVAL_DIR / "datasets"
RESULTS_DIR = EVAL_DIR / "results"


def load_config() -> dict:
    with open(CONFIGS_DIR / "prompting_strategies.yaml") as f:
        return yaml.safe_load(f)


def load_queries(max_queries: int | None = None) -> list[dict]:
    with open(DATASETS_DIR / "eval_queries.json") as f:
        queries = json.load(f)
    if max_queries:
        queries = queries[:max_queries]
    return queries


def get_rag_context(query: str, config: dict) -> list[str]:
    """Retrieve RAG context using the configured parameters."""
    from app.core.rag.vector_store import VectorStore

    store = VectorStore(
        backend="pgvector",
        chunk_size=config["rag_chunk_size"],
        chunk_overlap=config["rag_chunk_overlap"],
    )
    store.load()
    retriever = store.get_retriever(k=config["rag_top_k"])
    docs = retriever.invoke(query)
    return [d.page_content for d in docs]


def generate_response(
    query: str,
    system_prompt: str,
    augmentation: str,
    rag_docs: list[str] | None,
    llm: ChatGoogleGenerativeAI,
) -> str:
    """Generate a response using a specific strategy."""
    full_system = system_prompt

    if rag_docs:
        examples = "\n\n".join(
            f"[Example {i}]\n{doc}" for i, doc in enumerate(rag_docs, 1)
        )
        full_system += (
            f"\n\nRelevant counselor examples for reference:\n{examples}\n\n"
            "Use these examples to guide your response style, but paraphrase rather than copy."
        )

    user_content = query
    if augmentation:
        user_content = f"{augmentation}\n\nUser: {query}"

    messages = [
        SystemMessage(content=full_system),
        HumanMessage(content=user_content),
    ]

    result = llm.invoke(messages)
    return result.content


def run_prompting_evaluation(max_queries: int | None = None) -> pd.DataFrame:
    """Run the full prompting strategy comparison."""
    config = load_config()
    queries = load_queries(max_queries or config.get("num_eval_queries", 50))
    strategies = config["strategies"]

    judge = LLMJudge(
        model=config["eval_model"],
        temperature=config["judge_temperature"],
    )
    response_llm = ChatGoogleGenerativeAI(
        model=config["eval_model"],
        temperature=0.3,
    )

    # Determine conditions
    conditions = []
    for key, strategy in strategies.items():
        if config.get("test_with_rag", True):
            conditions.append((key, strategy, True))
        if config.get("test_without_rag", True):
            conditions.append((key, strategy, False))

    print(f"\n{'='*60}")
    print(f"Prompting Strategy Evaluation")
    print(f"{'='*60}")
    print(f"Strategies: {[s['name'] for s in strategies.values()]}")
    print(f"With RAG: {config.get('test_with_rag', True)}")
    print(f"Without RAG: {config.get('test_without_rag', True)}")
    print(f"Conditions: {len(conditions)}")
    print(f"Queries: {len(queries)}")
    print(f"Total evaluations: {len(conditions) * len(queries)}")
    print(f"{'='*60}\n")

    results = []

    for cond_idx, (strategy_key, strategy, use_rag) in enumerate(conditions, 1):
        rag_label = "with_rag" if use_rag else "no_rag"
        label = f"{strategy['name']} ({rag_label})"
        print(f"\n[{cond_idx}/{len(conditions)}] {label}")

        for q in tqdm(queries, desc=f"  {label}", leave=False):
            query_text = q["query"]

            rag_docs = None
            if use_rag:
                try:
                    rag_docs = get_rag_context(query_text, config)
                except Exception as e:
                    logger.warning("RAG retrieval failed: %s", e)
                    rag_docs = None

            t0 = time.time()
            response = generate_response(
                query_text,
                strategy["system_prompt"],
                strategy.get("augmentation", ""),
                rag_docs,
                response_llm,
            )
            response_time = time.time() - t0

            scores = judge.score(query_text, response)

            results.append(
                {
                    "query_id": q["id"],
                    "category": q["category"],
                    "strategy": strategy_key,
                    "strategy_name": strategy["name"],
                    "use_rag": use_rag,
                    "condition": f"{strategy_key}_{'rag' if use_rag else 'norag'}",
                    "response_time_s": round(response_time, 3),
                    **scores.to_dict(),
                    "response_text": response,
                }
            )

    df = pd.DataFrame(results)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "prompting_eval_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

    # Print summary
    summary = (
        df.groupby(["strategy_name", "use_rag"])
        .agg(
            {
                "empathy": "mean",
                "therapeutic_alliance": "mean",
                "safety": "mean",
                "coherence": "mean",
                "helpfulness": "mean",
                "average": "mean",
                "response_time_s": "mean",
            }
        )
        .round(3)
    )
    summary_path = RESULTS_DIR / "prompting_eval_summary.csv"
    summary.to_csv(summary_path)
    print(f"Summary saved to {summary_path}")
    print(f"\nResults by strategy:")
    print(summary.to_string())

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompting strategy evaluation")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit number of eval queries (for quick testing)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_prompting_evaluation(max_queries=args.max_queries)
