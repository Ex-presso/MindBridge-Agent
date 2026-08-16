"""Run the current offline release evaluations end-to-end.

Each layer is a standalone script. Layers are independent: a failure in one is
reported and the remaining layer still runs.

Prerequisites: PostgreSQL/pgvector, the configured retrieval collection, and a
DeepSeek key in backend/.env. The live memory acceptance runner remains separate
because it also requires the API, worker, and embedding server.

Usage:
    uv run --project ../backend python run_all.py
    uv run --project ../backend python run_all.py --quick
    uv run --project ../backend python run_all.py --skip safety
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent

# name -> (script, supports --max-queries)
LAYERS: dict[str, tuple[str, bool]] = {
    "routing": ("eval_routing.py", True),
    "safety": ("eval_safety.py", True),
}


def run_script(script: str, args: list[str]) -> bool:
    cmd = [sys.executable, str(EVAL_DIR / script), *args]
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=EVAL_DIR).returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all MindBridge evaluations")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run three routing and safety cases as a smoke check",
    )
    parser.add_argument("--max-queries", type=int, default=None, help="Cap queries for layers that support it")
    parser.add_argument("--skip", nargs="*", default=[], choices=list(LAYERS), help="Layers to skip")
    args = parser.parse_args()

    max_queries = 3 if args.quick else args.max_queries

    failures: list[str] = []
    for name, (script, supports_max) in LAYERS.items():
        if name in args.skip:
            print(f"\n=== SKIP {name} ===")
            continue
        print(f"\n{'=' * 60}\n{name.upper()}\n{'=' * 60}")

        layer_args: list[str] = []
        if supports_max and max_queries:
            layer_args += ["--max-queries", str(max_queries)]

        if not run_script(script, layer_args):
            failures.append(name)
            print(f"!! {name} failed — continuing with remaining layers")

    print(f"\n{'=' * 60}")
    if failures:
        print(f"DONE with failures: {', '.join(failures)}")
        sys.exit(1)
    print("ALL EVALUATIONS COMPLETE")
    print(f"Results: {EVAL_DIR / 'results'}")


if __name__ == "__main__":
    main()
