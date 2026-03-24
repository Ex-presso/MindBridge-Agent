"""
Run all evaluations end-to-end.

Usage:
    # Full evaluation (slow - all 50 queries × all configs)
    uv run python -m evaluation.run_all

    # Quick smoke test (3 queries only)
    uv run python -m evaluation.run_all --quick

    # Custom query count
    uv run python -m evaluation.run_all --max-queries 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(EVAL_ROOT))


def main():
    parser = argparse.ArgumentParser(description="Run all MindBridge evaluations")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: only 3 queries for smoke testing",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Max number of evaluation queries",
    )
    parser.add_argument(
        "--skip-rag",
        action="store_true",
        help="Skip RAG parameter evaluation",
    )
    parser.add_argument(
        "--skip-prompting",
        action="store_true",
        help="Skip prompting strategy evaluation",
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help="Skip figure generation (only collect data)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    max_queries = args.max_queries
    if args.quick:
        max_queries = 3

    if not args.skip_rag:
        print("\n" + "=" * 60)
        print("PHASE 1: RAG Parameter Evaluation")
        print("=" * 60)
        from eval_rag import run_rag_evaluation

        run_rag_evaluation(max_queries=max_queries)

    if not args.skip_prompting:
        print("\n" + "=" * 60)
        print("PHASE 2: Prompting Strategy Evaluation")
        print("=" * 60)
        from eval_prompting import run_prompting_evaluation

        run_prompting_evaluation(max_queries=max_queries)

    if not args.skip_analysis:
        print("\n" + "=" * 60)
        print("PHASE 3: Analysis & Figure Generation")
        print("=" * 60)
        from analyze_results import main as analyze

        analyze()

    print("\n" + "=" * 60)
    print("ALL EVALUATIONS COMPLETE")
    print("=" * 60)
    print(f"Results: {ROOT / 'evaluation' / 'results'}")
    print(f"Figures: {ROOT / 'evaluation' / 'results' / 'figures'}")


if __name__ == "__main__":
    main()
