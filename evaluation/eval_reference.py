"""Reference-based evaluation against MentalChat16K hold-out.

For each (query, counselor_reference) pair drawn from a held-out region of
MentalChat16K (outside the indexed pool, so retrieval cannot directly cheat),
runs the LangGraph agent on the query and computes:
    - BERTScore precision/recall/F1 vs the counselor reference
    - Cosine similarity in Qwen3-Embedding space

These metrics complement LLM-as-judge by giving model-free, reproducible
signal on how well the agent's response aligns with real counselor output.

Usage:
    cd evaluation/
    uv run python eval_reference.py
    uv run python eval_reference.py --max-queries 10  # smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from langchain_core.messages import AIMessage, HumanMessage
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(EVAL_ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / "backend" / ".env")


def _patch_bertscore_tokenizer() -> None:
    """Restore `build_inputs_with_special_tokens` for bert_score under transformers>=5.

    transformers >= 5.0 dropped this method from `PreTrainedTokenizerBase`, but
    bert_score 0.3.13 still calls it (only on the empty-string fallback path,
    `token_ids_1=None`). The shim reconstructs the canonical
    `[BOS] tokens [EOS]` single-segment pattern using the tokenizer's own
    special-token IDs — accurate for RoBERTa and DeBERTa (which is all
    bert_score actually exercises). The paired-segment branch is a
    best-effort RoBERTa-style `[BOS] a [EOS][EOS] b [EOS]` and is NOT used
    by bert_score; do not rely on it for BERT-style tokenizers, which use
    `[CLS] a [SEP] b [SEP]` (a single SEP separator).
    """
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if hasattr(PreTrainedTokenizerBase, "build_inputs_with_special_tokens"):
        return

    def _shim(self, token_ids_0, token_ids_1=None):
        bos = self.bos_token_id if self.bos_token_id is not None else self.cls_token_id
        eos = self.eos_token_id if self.eos_token_id is not None else self.sep_token_id
        if token_ids_1 is None:
            return [bos] + list(token_ids_0) + [eos]
        # RoBERTa/DeBERTa-style; bert_score never invokes this branch.
        return [bos] + list(token_ids_0) + [eos, eos] + list(token_ids_1) + [eos]

    PreTrainedTokenizerBase.build_inputs_with_special_tokens = _shim


_patch_bertscore_tokenizer()


def _apply_collection_override() -> None:
    cfg_path = Path(__file__).parent / "configs" / "reference_eval.yaml"
    if not cfg_path.exists():
        return
    import os

    with cfg_path.open() as f:
        cfg = yaml.safe_load(f) or {}
    coll = cfg.get("pgvector_collection")
    if coll:
        os.environ["PGVECTOR_COLLECTION"] = coll


_apply_collection_override()

from app.core.agent.agent import Agent, AgentRunContext  # noqa: E402
from app.core.llm.provider import get_llm  # noqa: E402
from config.settings import settings  # noqa: E402

import os as _os  # noqa: E402

if _os.environ.get("PGVECTOR_COLLECTION"):
    assert settings.PGVECTOR_COLLECTION == _os.environ["PGVECTOR_COLLECTION"], (
        "Collection override leaked — a backend module was imported before "
        "_apply_collection_override() ran."
    )

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = EVAL_DIR / "configs"
RESULTS_DIR = EVAL_DIR / "results"


def load_config() -> dict:
    with (CONFIGS_DIR / "reference_eval.yaml").open() as f:
        return yaml.safe_load(f)


def load_benchmark(path: Path) -> list[dict]:
    with path.open() as f:
        data = json.load(f)
    return data["queries"]


def build_agent(cfg: dict) -> Agent:
    llm = get_llm(
        provider=cfg["llm_provider"],
        api_key=cfg.get("llm_api_key"),
        base_url=cfg.get("llm_base_url"),
        model=cfg.get("llm_model"),
    )
    target_temp = cfg.get("temperature", 0.3)
    if hasattr(llm, "temperature"):
        try:
            llm.temperature = target_temp
        except Exception as exc:
            logger.warning("Could not set llm.temperature=%s: %s", target_temp, exc)
    return Agent(llm, checkpointer=None)


async def run_one(agent: Agent, query: str) -> tuple[str, float]:
    """Returns (response_text, latency_s)."""
    state = {"messages": [HumanMessage(content=query)], "tool_iterations": 0}
    t0 = time.time()
    result = await agent.app.ainvoke(state, context=AgentRunContext())
    elapsed = time.time() - t0
    final_ai = next(
        (m for m in reversed(result["messages"]) if isinstance(m, AIMessage) and not m.tool_calls),
        None,
    )
    response_text = str(final_ai.content) if final_ai is not None else ""
    return response_text, elapsed


def cosine_similarity_pairs(refs: list[str], hyps: list[str], embed_model_name: str) -> np.ndarray:
    """Cosine similarity for paired (ref, hyp) lists."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(embed_model_name)
    embed_refs = model.encode(refs, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    embed_hyps = model.encode(hyps, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return (embed_refs * embed_hyps).sum(axis=1)


def bertscore_pairs(
    refs: list[str], hyps: list[str], model_name: str, lang: str,
    rescale_with_baseline: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (precision, recall, f1) numpy arrays.

    With rescale_with_baseline=True, scores are subtracted by a corpus-derived
    baseline (random-pair similarity for the model/lang) so the values are
    bounded near [0, 1] and meaningfully interpretable. Raw BERTScore F1 for
    English roberta-large floats around 0.85 even on random pairs and
    *cannot* be read as "85% similar"; rescaled F1 typically lands in
    [0.0, 0.5] and *can* be read as alignment strength relative to baseline.
    """
    from bert_score import score as bert_score_fn

    p, r, f1 = bert_score_fn(
        cands=hyps, refs=refs, model_type=model_name, lang=lang,
        verbose=False, batch_size=8,
        rescale_with_baseline=rescale_with_baseline,
    )
    return p.numpy(), r.numpy(), f1.numpy()


async def generate_responses(cfg: dict, queries: list[dict], output_path: Path) -> pd.DataFrame:
    """Phase 1: drive the agent on each query and save responses to CSV."""
    agent = build_agent(cfg)
    rows: list[dict] = []
    print("Phase 1/2: generating agent responses ...")
    for q in tqdm(queries, desc="Generating"):
        try:
            response, elapsed = await run_one(agent, q["query"])
        except Exception as exc:
            logger.exception("Agent failed on %s", q["query_id"])
            raise RuntimeError(
                f"Reference evaluation failed on {q['query_id']}; refusing to "
                "score an incomplete run."
            ) from exc
        rows.append(
            {
                "query_id": q["query_id"],
                "source_example_idx": q["source_example_idx"],
                "query": q["query"],
                "reference": q["reference_response"],
                "response": response,
                "response_time_s": round(elapsed, 3),
            }
        )
        pd.DataFrame(rows).to_csv(output_path, index=False)
    return pd.DataFrame(rows)


def add_similarity_scores(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Phase 2: add cosine + BERTScore columns to the response DataFrame.

    Operates idempotently — safe to re-run on a CSV that already has scores
    or partial generation output. Empty/NaN responses are coerced to "" so
    BERTScore returns 0 for them rather than crashing.
    """
    df = df.copy()
    df["response"] = df["response"].fillna("").astype(str)
    df["reference"] = df["reference"].fillna("").astype(str)
    refs = df["reference"].tolist()
    hyps = df["response"].tolist()

    print("Phase 2a/2: embedding cosine similarity ...")
    cos_sims = cosine_similarity_pairs(refs, hyps, cfg["embedding_model"])
    df["cosine_similarity"] = [round(float(c), 4) for c in cos_sims]

    print("Phase 2b/2: BERTScore ...")
    bs_p, bs_r, bs_f1 = bertscore_pairs(
        refs, hyps, cfg["bertscore_model"], cfg["bertscore_lang"]
    )
    df["bertscore_p"] = [round(float(x), 4) for x in bs_p]
    df["bertscore_r"] = [round(float(x), 4) for x in bs_r]
    df["bertscore_f1"] = [round(float(x), 4) for x in bs_f1]
    return df


def write_summary(df: pd.DataFrame, summary_path: Path) -> pd.DataFrame:
    """Aggregate over non-empty responses; report empty count separately.

    Empty/error responses produce extreme negative rescaled BERTScores
    (~-5 from cosine of an essentially empty embedding) which would
    badly distort the mean if included. They're tracked via
    n_empty_responses.
    """
    response_lengths = df["response"].fillna("").str.len()
    nonempty = df[response_lengths > 0]
    summary = pd.DataFrame(
        [
            {
                "n_total": len(df),
                "n_scored": len(nonempty),
                "n_empty_responses": int(len(df) - len(nonempty)),
                "bertscore_f1_mean": round(nonempty["bertscore_f1"].mean(), 4),
                "bertscore_f1_std": round(nonempty["bertscore_f1"].std(), 4),
                "bertscore_p_mean": round(nonempty["bertscore_p"].mean(), 4),
                "bertscore_r_mean": round(nonempty["bertscore_r"].mean(), 4),
                "cosine_similarity_mean": round(nonempty["cosine_similarity"].mean(), 4),
                "cosine_similarity_std": round(nonempty["cosine_similarity"].std(), 4),
                "response_time_s_mean": round(df["response_time_s"].mean(), 3),
            }
        ]
    )
    summary.to_csv(summary_path, index=False)
    return summary


async def main_async(max_queries: int | None, score_only: bool) -> None:
    cfg = load_config()
    benchmark_path = EVAL_DIR / cfg["benchmark_path"]
    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"Benchmark not found at {benchmark_path}. "
            "Run `uv run python build_reference_benchmark.py` first."
        )
    queries = load_benchmark(benchmark_path)
    if max_queries:
        queries = queries[:max_queries]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_query_path = RESULTS_DIR / "reference_eval_results.csv"

    print(f"\n{'=' * 60}")
    print("Reference-Based Eval (vs MentalChat16K hold-out)")
    print(f"{'=' * 60}")
    print(f"Provider: {cfg['llm_provider']}")
    print(f"Model:    {cfg.get('llm_model')}")
    print(f"BERTScore model: {cfg['bertscore_model']}")
    print(f"Queries:  {len(queries)}")
    print(f"Mode:     {'score-only' if score_only else 'full (generate + score)'}")
    print(f"{'=' * 60}\n")

    if score_only:
        if not per_query_path.exists():
            raise FileNotFoundError(
                f"No existing CSV at {per_query_path}; run without --score-only first."
            )
        df = pd.read_csv(per_query_path)
        print(f"Loaded {len(df)} pre-computed responses from {per_query_path}")
    else:
        df = await generate_responses(cfg, queries, per_query_path)

    df = add_similarity_scores(df, cfg)
    df.to_csv(per_query_path, index=False)
    print(f"\nPer-query results: {per_query_path}")

    summary = write_summary(df, RESULTS_DIR / "reference_eval_summary.csv")
    print(f"Summary:           {RESULTS_DIR / 'reference_eval_summary.csv'}")
    print("\nMetrics:")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference-based eval")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Limit number of queries (for smoke testing)")
    parser.add_argument("--score-only", action="store_true",
                        help="Skip generation; load existing per-query CSV and "
                             "(re)compute cosine + BERTScore. Useful when "
                             "BERTScore breaks but the slow generation phase "
                             "already finished.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main_async(args.max_queries, args.score_only))


if __name__ == "__main__":
    main()
