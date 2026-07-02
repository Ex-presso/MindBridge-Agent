"""Cross-judge calibration.

Re-scores the existing prompting-eval generations with a second judge from a
different model lineage and reports inter-judge agreement (Spearman rank
correlation + within-1 agreement rate) per dimension. This is the most direct
check on the documented limitation that the primary judge is Claude-distilled
and may carry a Claude-style scoring prior.

No generation is run — it reuses `response_text` already in the source CSV, so
it's cheap (one judge call per row) and can use an API judge.

Usage:
    cd evaluation
    uv run python eval_cross_judge.py --dry-run     # validate data plumbing, no judge
    uv run python eval_cross_judge.py               # re-judge + correlate
    uv run python eval_cross_judge.py --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from scipy import stats

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from metrics.judge import LLMJudge  # noqa: E402

CONFIGS_DIR = EVAL_DIR / "configs"
RESULTS_DIR = EVAL_DIR / "results"


def load_config() -> dict:
    with (CONFIGS_DIR / "cross_judge_eval.yaml").open() as f:
        return yaml.safe_load(f)


def load_query_map(path: Path) -> dict[str, str]:
    with path.open() as f:
        return {q["id"]: q["query"] for q in json.load(f)}


def prepare_rows(cfg: dict) -> pd.DataFrame:
    """Load generations, attach query text, keep rows we can re-judge."""
    df = pd.read_csv(EVAL_DIR / cfg["source_csv"])
    qmap = load_query_map(EVAL_DIR / cfg["queries_path"])
    df["query"] = df["query_id"].map(qmap)
    df["response_text"] = df["response_text"].fillna("").astype(str)
    keep = df["query"].notna() & (df["response_text"].str.len() > 0)
    dropped = int((~keep).sum())
    if dropped:
        print(f"  dropped {dropped} rows (missing query text or empty response)")
    return df[keep].reset_index(drop=True)


def correlate(df: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    rows = []
    for dim in dims:
        orig, cross = df[dim], df[f"cross_{dim}"]
        rho, p = stats.spearmanr(orig, cross)
        within1 = float((abs(orig - cross) <= 1).mean())
        rows.append({
            "dimension": dim,
            "spearman_rho": round(float(rho), 4),
            "p_value": round(float(p), 4),
            "within_1_agreement": round(within1, 4),
            "mean_abs_diff": round(float((orig - cross).abs().mean()), 4),
            "primary_mean": round(float(orig.mean()), 3),
            "cross_mean": round(float(cross.mean()), 3),
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-judge calibration")
    ap.add_argument("--limit", type=int, default=None, help="Cap rows (for a cheap check)")
    ap.add_argument("--dry-run", action="store_true", help="Validate data plumbing without calling the judge")
    args = ap.parse_args()

    cfg = load_config()
    dims = list(cfg["judge_dimensions"])
    df = prepare_rows(cfg)
    if args.limit:
        df = df.head(args.limit)

    print(f"Rows to judge: {len(df)}   dimensions: {dims}")
    print(f"Cross-judge: {cfg['judge_model']} (provider={cfg['judge_provider']})")

    if args.dry_run:
        print("--dry-run: data plumbing OK, skipping judge calls.")
        return

    judge = LLMJudge(
        model=cfg["judge_model"],
        temperature=cfg["judge_temperature"],
        provider=cfg["judge_provider"],
        base_url=cfg["judge_base_url"],
        dimensions=tuple(dims),
    )

    from tqdm import tqdm

    scored = {f"cross_{d}": [] for d in dims}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cross-judging"):
        s = judge.score(row["query"], row["response_text"]).to_dict()
        for d in dims:
            scored[f"cross_{d}"].append(s[d])
    for col, vals in scored.items():
        df[col] = vals

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_path = RESULTS_DIR / "cross_judge_results.csv"
    df.to_csv(per_path, index=False)

    summary = correlate(df, dims)
    summary_path = RESULTS_DIR / "cross_judge_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"\nPer-row: {per_path}\nSummary: {summary_path}\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
