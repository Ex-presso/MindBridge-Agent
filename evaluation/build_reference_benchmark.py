"""Build the reference-based evaluation benchmark.

Selects N MentalChat16K examples from a held-out range that is *outside*
the indexed pool, so the gold counselor response is not retrievable.
For each example, the user-side `input` becomes the eval query and the
counselor-side `output` is the reference response we measure against.

Usage:
    cd evaluation/
    uv run python build_reference_benchmark.py --num-queries 100 \
        --indexed-pool-size 3000 --num-holdout 200 --seed 42
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
    indexed_pool_size: int,
    num_holdout: int,
    seed: int,
    min_input_words: int,
    max_input_words: int,
) -> list[dict]:
    """Pick `num_queries` from the held-out region [indexed_pool_size, indexed_pool_size + num_holdout)."""
    ds = load_dataset("ShenLab/MentalChat16K", split="train")

    holdout_start = indexed_pool_size
    holdout_end = min(indexed_pool_size + num_holdout, len(ds))
    if holdout_end - holdout_start < num_queries:
        raise ValueError(
            f"Holdout window [{holdout_start}, {holdout_end}) has only "
            f"{holdout_end - holdout_start} examples; cannot draw {num_queries}."
        )

    holdout = ds.select(range(holdout_start, holdout_end))
    candidates = []
    for offset, ex in enumerate(holdout):
        text = (ex.get("input") or "").strip()
        n_words = len(text.split())
        if min_input_words <= n_words <= max_input_words:
            # Output is stored verbatim (no .strip()) so anyone reproducing
            # the BERTScore against this reference gets the exact same string.
            candidates.append((holdout_start + offset, text, ex.get("output") or ""))

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:num_queries]
    if len(selected) < num_queries:
        raise ValueError(
            f"Only {len(selected)} candidates passed the word filter; "
            f"requested {num_queries}. Increase --num-holdout or relax filters."
        )

    return [
        {
            "query_id": f"ref_{i:03d}",
            "query": text,
            "source_example_idx": src_idx,
            "reference_response": ref,
        }
        for i, (src_idx, text, ref) in enumerate(selected, start=1)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reference-based eval benchmark")
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--indexed-pool-size", type=int, default=3000,
                        help="Examples [0, indexed_pool_size) are in the IR index. "
                             "Holdout starts at this index.")
    parser.add_argument("--num-holdout", type=int, default=300,
                        help="Window size for sampling, starting at indexed_pool_size.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-input-words", type=int, default=8)
    parser.add_argument("--max-input-words", type=int, default=120)
    parser.add_argument("--output", type=Path,
                        default=DATASETS_DIR / "reference_benchmark.json")
    args = parser.parse_args()

    items = build_benchmark(
        num_queries=args.num_queries,
        indexed_pool_size=args.indexed_pool_size,
        num_holdout=args.num_holdout,
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
                    "indexed_pool_size": args.indexed_pool_size,
                    "num_holdout": args.num_holdout,
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
    print(f"Wrote {len(items)} hold-out queries to {args.output}")


if __name__ == "__main__":
    main()
