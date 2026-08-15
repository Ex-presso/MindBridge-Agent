"""Agent routing benchmark runner.

Drives the LangGraph agent through 50 hand-labeled queries and records
whether it invoked `fetch_mental_health_examples`. Compares against the
ground-truth `should_call_rag` label to compute Precision / Recall / F1
on the binary tool-invocation decision.

Ambiguous queries (`should_call_rag: null`) are recorded but excluded from
the scored metrics.

Usage:
    cd evaluation/
    uv run --project ../backend python eval_routing.py
    uv run --project ../backend python eval_routing.py --max-queries 10
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(EVAL_ROOT))

load_dotenv(ROOT / "backend" / ".env")


def _apply_eval_overrides() -> None:
    """Install host-side eval settings before backend imports settings."""
    cfg_path = Path(__file__).parent / "configs" / "routing_eval.yaml"
    if not cfg_path.exists():
        return
    with cfg_path.open() as f:
        cfg = yaml.safe_load(f) or {}
    coll = cfg.get("pgvector_collection")
    if coll:
        os.environ["PGVECTOR_COLLECTION"] = coll
    embedding_provider = cfg.get("embedding_provider")
    if embedding_provider:
        os.environ["EMBEDDING_PROVIDER"] = embedding_provider
    embedding_model = cfg.get("embedding_model")
    if embedding_model:
        os.environ["EMBEDDING_MODEL"] = embedding_model
    embedding_base_url = cfg.get("embedding_base_url")
    if embedding_base_url:
        os.environ["EMBEDDING_BASE_URL"] = embedding_base_url


_apply_eval_overrides()

from app.core.agent.agent import Agent, AgentRunContext  # noqa: E402
from app.core.llm.provider import get_llm  # noqa: E402
from config.settings import settings  # noqa: E402

# Belt-and-suspenders: confirm the override won the resolution race against
# any transitive .env-based settings load that may have happened above.
import os as _os  # noqa: E402

if _os.environ.get("PGVECTOR_COLLECTION"):
    assert settings.PGVECTOR_COLLECTION == _os.environ["PGVECTOR_COLLECTION"], (
        f"Collection override leaked: settings={settings.PGVECTOR_COLLECTION!r} "
        f"but env={_os.environ['PGVECTOR_COLLECTION']!r}. A backend module was "
        "imported before _apply_collection_override() ran."
    )

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = EVAL_DIR / "configs"
RESULTS_DIR = EVAL_DIR / "results"
TOOL_NAME = "fetch_mental_health_examples"


def load_config() -> dict:
    with (CONFIGS_DIR / "routing_eval.yaml").open() as f:
        return yaml.safe_load(f)


def load_benchmark(path: Path) -> list[dict]:
    with path.open() as f:
        data = json.load(f)
    return data["queries"]


def build_agent(cfg: dict) -> Agent:
    api_key_env = cfg["llm_api_key_env"]
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"{api_key_env} is required for the routing evaluation.")
    target_temp = cfg.get("temperature", 0.0)
    llm = get_llm(
        provider=cfg["llm_provider"],
        api_key=api_key,
        base_url=cfg.get("llm_base_url"),
        model=cfg.get("llm_model"),
        temperature=target_temp,
    )
    return Agent(llm, checkpointer=None)


def detect_tool_call(messages) -> tuple[bool, list[str]]:
    """Return (called_target_tool, list_of_invoked_tool_names)."""
    invoked = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for call in m.tool_calls:
                name = (
                    call.get("name")
                    if isinstance(call, dict)
                    else getattr(call, "name", "")
                )
                if name:
                    invoked.append(name)
    return TOOL_NAME in invoked, invoked


def count_target_calls(messages) -> int:
    """How many times the target tool was invoked (across iterations)."""
    n = 0
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for call in m.tool_calls:
                name = (
                    call.get("name")
                    if isinstance(call, dict)
                    else getattr(call, "name", "")
                )
                if name == TOOL_NAME:
                    n += 1
    return n


async def run_one(agent: Agent, query: str) -> dict:
    state = {"messages": [HumanMessage(content=query)], "tool_iterations": 0}
    t0 = time.time()
    result = await agent.app.ainvoke(state, context=AgentRunContext())
    elapsed = time.time() - t0

    called, invoked = detect_tool_call(result["messages"])
    n_target = count_target_calls(result["messages"])
    final_ai = next(
        (
            m
            for m in reversed(result["messages"])
            if isinstance(m, AIMessage) and not m.tool_calls
        ),
        None,
    )
    response_text = final_ai.content if final_ai is not None else ""

    return {
        "called_target_tool": called,
        "invoked_tools": ",".join(invoked),
        "n_target_tool_calls": n_target,
        "n_total_tool_invocations": len(invoked),
        "response_time_s": round(elapsed, 3),
        "response_excerpt": (str(response_text)[:280] + "…")
        if len(str(response_text)) > 280
        else str(response_text),
    }


def score(per_query: pd.DataFrame, score_ambiguous: bool) -> pd.DataFrame:
    """Compute Precision/Recall/F1 over labeled queries.

    Ambiguous queries (`should_call_rag == null`) have no ground truth, so they
    are dropped here regardless of the flag — there is no defensible way to
    fold them into a binary confusion matrix. The `score_ambiguous` flag is
    retained for callers that may surface ambiguous-coverage stats elsewhere
    but it is intentionally NOT honored as "treat ambiguous as negative".
    """
    if score_ambiguous:
        logger.warning(
            "score_ambiguous=True has no defined semantics for binary P/R/F1; "
            "ambiguous rows are still excluded. Use the per-query CSV to "
            "inspect ambiguous decisions separately."
        )
    df = per_query[per_query["should_call_rag"].isin([True, False])].copy()

    tp = (df["should_call_rag"].eq(True) & df["called_target_tool"].eq(True)).sum()
    fp = (df["should_call_rag"].eq(False) & df["called_target_tool"].eq(True)).sum()
    fn = (df["should_call_rag"].eq(True) & df["called_target_tool"].eq(False)).sum()
    tn = (df["should_call_rag"].eq(False) & df["called_target_tool"].eq(False)).sum()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(df) if len(df) else 0.0

    return pd.DataFrame(
        [
            {
                "n_scored": int(len(df)),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "accuracy": round(accuracy, 4),
            }
        ]
    )


async def main_async(max_queries: int | None) -> None:
    cfg = load_config()
    benchmark_path = EVAL_DIR / cfg["benchmark_path"]
    queries = load_benchmark(benchmark_path)
    if max_queries:
        queries = queries[:max_queries]

    print(f"\n{'=' * 60}")
    print("Agent Routing Benchmark")
    print(f"{'=' * 60}")
    print(f"Provider: {cfg['llm_provider']}")
    print(f"Model:    {cfg.get('llm_model')}")
    print(f"Queries:  {len(queries)}")
    print(f"{'=' * 60}\n")

    agent = build_agent(cfg)

    rows: list[dict] = []
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_query_path = RESULTS_DIR / "routing_eval_results.csv"

    for q in tqdm(queries, desc="Routing"):
        try:
            outcome = await run_one(agent, q["query"])
        except Exception as exc:
            logger.exception("Agent failed on %s", q["id"])
            outcome = {
                "called_target_tool": False,
                "invoked_tools": "",
                "n_target_tool_calls": 0,
                "n_total_tool_invocations": 0,
                "response_time_s": 0.0,
                "response_excerpt": f"<ERROR: {exc!s}>",
                "error": str(exc),
            }
        else:
            outcome["error"] = ""

        rows.append(
            {
                "query_id": q["id"],
                "query": q["query"],
                "category": q["category"],
                "should_call_rag": q["should_call_rag"],
                **outcome,
                "notes": q.get("notes", ""),
            }
        )
        # Incremental save so partial runs are recoverable
        pd.DataFrame(rows).to_csv(per_query_path, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(per_query_path, index=False)
    print(f"\nPer-query results: {per_query_path}")

    failed = df["error"].astype(bool)
    if failed.any():
        raise RuntimeError(
            f"Routing evaluation failed for {int(failed.sum())}/{len(df)} queries; "
            "refusing to publish a misleading F1. Inspect the per-query CSV."
        )

    summary = score(df, cfg.get("score_ambiguous", False))
    summary_path = RESULTS_DIR / "routing_eval_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary:           {summary_path}")
    print(
        "\nMetrics (excluding ambiguous queries):"
        if not cfg.get("score_ambiguous", False)
        else "\nMetrics:"
    )
    print(summary.to_string(index=False))

    # Per-category breakdown
    cat_summary = (
        df[df["should_call_rag"].isin([True, False])]
        .groupby("category")
        .agg(
            n=("query_id", "count"),
            should_call=("should_call_rag", "mean"),
            did_call=("called_target_tool", "mean"),
        )
        .round(3)
    )
    cat_path = RESULTS_DIR / "routing_eval_by_category.csv"
    cat_summary.to_csv(cat_path)
    print(f"\nPer-category breakdown: {cat_path}")
    print(cat_summary.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent routing benchmark")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit number of queries (for smoke testing)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main_async(args.max_queries))


if __name__ == "__main__":
    main()
