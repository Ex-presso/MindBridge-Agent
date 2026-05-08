"""Build the retrieval IR benchmark from MentalChat16K.

Samples N (input, source_example_idx) pairs from the indexed portion of the
MentalChat16K corpus. The user-side `input` becomes the eval query; gold
relevance is defined as "any chunk produced from the same source example."

Saves to `evaluation/datasets/ir_benchmark.json`. Re-running with the same
`--seed` and `--num-queries` is deterministic.

Usage:
    cd evaluation/
    uv run python build_ir_benchmark.py --num-queries 30 --seed 42 \
        --pool-size 3000 --min-input-words 8
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

EVAL_DIR = Path(__file__).resolve().parent
DATASETS_DIR = EVAL_DIR / "datasets"


def build_benchmark(
    num_queries: int,
    pool_size: int,
    seed: int,
    min_input_words: int,
    max_input_words: int,
) -> list[dict]:
    """Sample queries and record source_example_idx for each.

    pool_size must match the `max_dataset_samples` used to build the eval
    indexes — otherwise gold examples won't be retrievable.
    """
    ds = load_dataset("ShenLab/MentalChat16K", split="train")
    pool = ds.select(range(min(pool_size, len(ds))))

    candidates = []
    for idx, ex in enumerate(pool):
        text = (ex.get("input") or "").strip()
        n_words = len(text.split())
        if min_input_words <= n_words <= max_input_words:
            # Output is stored verbatim so re-chunking it at eval time
            # produces the same chunks the index has — no .strip().
            candidates.append((idx, text, ex.get("output") or ""))

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:num_queries]

    return [
        {
            "query_id": f"ir_{i:03d}",
            "query": text,
            "source_example_idx": src_idx,
            "reference_response": ref,
        }
        for i, (src_idx, text, ref) in enumerate(selected, start=1)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retrieval IR benchmark")
    parser.add_argument("--num-queries", type=int, default=30)
    parser.add_argument("--pool-size", type=int, default=3000,
                        help="Must match max_dataset_samples used at index build time")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-input-words", type=int, default=8)
    parser.add_argument("--max-input-words", type=int, default=120)
    parser.add_argument("--output", type=Path,
                        default=DATASETS_DIR / "ir_benchmark.json")
    args = parser.parse_args()

    items = build_benchmark(
        num_queries=args.num_queries,
        pool_size=args.pool_size,
        seed=args.seed,
        min_input_words=args.min_input_words,
        max_input_words=args.max_input_words,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(
            {
                "config": {
                    "num_queries": args.num_queries,
                    "pool_size": args.pool_size,
                    "seed": args.seed,
                    "min_input_words": args.min_input_words,
                    "max_input_words": args.max_input_words,
                    "dataset": "ShenLab/MentalChat16K",
                },
                "queries": items,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Wrote {len(items)} queries to {args.output}")


if __name__ == "__main__":
    main()
