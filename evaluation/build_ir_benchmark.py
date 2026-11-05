"""Build the retrieval benchmark for the question/answer knowledge base.

Samples indexed client questions and has an LLM rewrite each one as the search
query the agent would pass to the retrieval tool. The gold label is the id of
the source example. The rewrite is what makes the benchmark meaningful: the
index stores the questions themselves, so a verbatim question would always
retrieve itself at rank 1.

Usage:
    cd evaluation/
    uv run --project ../backend python build_ir_benchmark.py --num-queries 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
load_dotenv(ROOT / "backend" / ".env")

from app.core.llm.provider import get_llm  # noqa: E402
from app.core.rag.knowledge_base import load_examples  # noqa: E402

REWRITE_PROMPT = (
    "You write search queries for a counseling knowledge base. Rewrite the "
    "client's message as a short first-person description of the same concern, "
    "in one or two sentences and in different words. Keep the specific "
    "situation and feelings. Do not add advice or details that are not in the "
    "message. Output only the rewritten text."
)


def build_benchmark(cfg: dict, num_queries: int, seed: int, min_words: int, max_words: int) -> list[dict]:
    api_key = os.environ.get(cfg["llm_api_key_env"])
    if not api_key:
        raise ValueError(f"{cfg['llm_api_key_env']} is required to rewrite queries.")
    llm = get_llm(
        cfg["llm_provider"],
        api_key=api_key,
        base_url=cfg.get("llm_base_url"),
        model=cfg.get("llm_model"),
        temperature=0.0,
    )

    candidates = [
        example
        for example in load_examples()
        if min_words <= len(example.question.split()) <= max_words
    ]
    selected = random.Random(seed).sample(candidates, num_queries)

    queries = []
    for index, example in enumerate(tqdm(selected, desc="Rewriting"), start=1):
        reply = llm.invoke(
            [SystemMessage(content=REWRITE_PROMPT), HumanMessage(content=example.question)]
        )
        queries.append(
            {
                "query_id": f"ir_{index:03d}",
                "query": str(reply.content).strip(),
                "gold_id": example.id,
                "source_question": example.question,
            }
        )
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the retrieval benchmark")
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-words", type=int, default=8)
    parser.add_argument("--max-words", type=int, default=120)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    with (EVAL_DIR / "configs" / "retrieval_eval.yaml").open() as f:
        cfg = yaml.safe_load(f)
    output = args.output or EVAL_DIR / cfg["benchmark_path"]

    queries = build_benchmark(cfg, args.num_queries, args.seed, args.min_words, args.max_words)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(
            {
                "config": {
                    "num_queries": args.num_queries,
                    "seed": args.seed,
                    "rewrite_model": cfg.get("llm_model"),
                },
                "queries": queries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Wrote {len(queries)} queries to {output}")


if __name__ == "__main__":
    main()
