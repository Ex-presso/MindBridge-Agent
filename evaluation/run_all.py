"""Run the full five-layer evaluation suite end-to-end.

Each layer is a standalone script; this driver just sequences them and then
regenerates figures. Layers are independent — a failure in one is reported and
the rest still run.

Prerequisites: PostgreSQL (pgvector) up, LM Studio serving the generator (and
the judge model for the prompting layer). See docs/EVALUATION.md.

Usage:
    uv run python run_all.py                 # all layers, full size
    uv run python run_all.py --quick         # 3 queries per layer (smoke test)
    uv run python run_all.py --build         # (re)build IR indexes first
    uv run python run_all.py --skip prompting safety
    uv run python run_all.py --skip-analysis
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ANALYZE = EVAL_DIR.parent / "analysis" / "analyze.py"

# name -> (script, supports --max-queries)
LAYERS: dict[str, tuple[str, bool]] = {
    "retrieval": ("eval_retrieval.py", False),
    "routing": ("eval_routing.py", True),
    "reference": ("eval_reference.py", True),
    "prompting": ("eval_prompting.py", True),
    "safety": ("eval_safety.py", True),
}


def run_script(script: str, args: list[str]) -> bool:
    cmd = [sys.executable, str(EVAL_DIR / script), *args]
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=EVAL_DIR).returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all MindBridge evaluations")
    parser.add_argument("--quick", action="store_true", help="3 queries per layer (smoke test)")
    parser.add_argument("--max-queries", type=int, default=None, help="Cap queries for layers that support it")
    parser.add_argument("--build", action="store_true", help="Build IR indexes (retrieval layer) instead of reusing them")
    parser.add_argument("--skip", nargs="*", default=[], choices=list(LAYERS), help="Layers to skip")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip figure generation")
    args = parser.parse_args()

    max_queries = 3 if args.quick else args.max_queries

    failures: list[str] = []
    for name, (script, supports_max) in LAYERS.items():
        if name in args.skip:
            print(f"\n=== SKIP {name} ===")
            continue
        print(f"\n{'=' * 60}\n{name.upper()}\n{'=' * 60}")

        layer_args: list[str] = []
        if name == "retrieval" and not args.build:
            layer_args.append("--no-build")  # reuse existing pgvector collections
        if supports_max and max_queries:
            layer_args += ["--max-queries", str(max_queries)]

        if not run_script(script, layer_args):
            failures.append(name)
            print(f"!! {name} failed — continuing with remaining layers")

    if not args.skip_analysis:
        print(f"\n{'=' * 60}\nANALYSIS\n{'=' * 60}")
        if subprocess.run([sys.executable, str(ANALYZE)]).returncode != 0:
            failures.append("analysis")

    print(f"\n{'=' * 60}")
    if failures:
        print(f"DONE with failures: {', '.join(failures)}")
        sys.exit(1)
    print("ALL EVALUATIONS COMPLETE")
    print(f"Results: {EVAL_DIR / 'results'}   Figures: {ANALYZE.parent / 'pics'}")


if __name__ == "__main__":
    main()
