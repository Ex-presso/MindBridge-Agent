"""Human-calibration scaffolding for the LLM judge.

`sample` mode draws a stratified subset of the existing generations and writes a
CSV template with blank human-rating columns. A human fills human_empathy /
human_safety (1-5), then `score` mode reports Spearman + quadratic-weighted
Cohen's kappa between the human and the LLM judge — closing the loop on judge
trustworthiness (the "no human calibration" limitation in EVALUATION.md).

Usage:
    cd evaluation
    uv run python build_calibration_sample.py sample --n 30
    # ... a human fills human_empathy / human_safety in the template ...
    uv run python build_calibration_sample.py score
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
SOURCE_CSV = RESULTS_DIR / "prompting_eval_results.csv"
QUERIES = EVAL_DIR / "datasets" / "eval_queries.json"
TEMPLATE = RESULTS_DIR / "human_calibration_template.csv"
DIMS = ["empathy", "safety"]


def _query_map() -> dict[str, str]:
    with QUERIES.open() as f:
        return {q["id"]: q["query"] for q in json.load(f)}


def sample(n: int, seed: int) -> None:
    df = pd.read_csv(SOURCE_CSV)
    df["query"] = df["query_id"].map(_query_map())
    df["response_text"] = df["response_text"].fillna("").astype(str)
    df = df[df["query"].notna() & (df["response_text"].str.len() > 0)]

    # Stratify across conditions so all strategies/RAG settings are represented.
    strata = "condition" if "condition" in df.columns else "strategy"
    per = max(1, n // max(1, df[strata].nunique()))
    parts = [g.sample(min(len(g), per), random_state=seed) for _, g in df.groupby(strata)]
    picked = pd.concat(parts).head(n).reset_index(drop=True)

    out = picked[["query_id", "condition", "query", "response_text"] + DIMS].copy()
    out = out.rename(columns={d: f"judge_{d}" for d in DIMS})
    for d in DIMS:
        out[f"human_{d}"] = ""  # blank for the human rater
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(TEMPLATE, index=False)
    print(f"Wrote {len(out)} rows to {TEMPLATE}")
    print(f"Fill human_{'/human_'.join(DIMS)} (1-5), then: build_calibration_sample.py score")


def score() -> None:
    from scipy import stats
    from sklearn.metrics import cohen_kappa_score

    if not TEMPLATE.exists():
        raise FileNotFoundError(f"No template at {TEMPLATE}; run `sample` first.")
    df = pd.read_csv(TEMPLATE)

    rows = []
    for d in DIMS:
        sub = df[[f"judge_{d}", f"human_{d}"]].dropna()
        sub = sub[sub[f"human_{d}"].astype(str).str.strip() != ""]
        if len(sub) < 3:
            print(f"[{d}] only {len(sub)} rated rows — fill more of the template.")
            continue
        judge = sub[f"judge_{d}"].astype(int)
        human = sub[f"human_{d}"].astype(int)
        rho, p = stats.spearmanr(judge, human)
        kappa = cohen_kappa_score(judge, human, weights="quadratic")
        rows.append({
            "dimension": d,
            "n": len(sub),
            "spearman_rho": round(float(rho), 4),
            "p_value": round(float(p), 4),
            "weighted_kappa": round(float(kappa), 4),
            "judge_mean": round(float(judge.mean()), 3),
            "human_mean": round(float(human.mean()), 3),
        })

    if rows:
        summary = pd.DataFrame(rows)
        out = RESULTS_DIR / "human_calibration_summary.csv"
        summary.to_csv(out, index=False)
        print(summary.to_string(index=False))
        print(f"\nSaved: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Human-calibration scaffolding")
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("sample", help="Write a blank rating template")
    s.add_argument("--n", type=int, default=30)
    s.add_argument("--seed", type=int, default=42)
    sub.add_parser("score", help="Score a filled template (Spearman + weighted kappa)")
    args = ap.parse_args()

    if args.mode == "sample":
        sample(args.n, args.seed)
    else:
        score()


if __name__ == "__main__":
    main()
