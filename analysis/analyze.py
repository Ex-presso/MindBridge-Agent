"""
MindBridge Evaluation Analysis — Scientific Figures.

Reads evaluation results from evaluation/results/ and generates
publication-quality figures in analysis/pics/.

Usage:
    cd MindBridge/
    python analysis/analyze.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "evaluation" / "results"
PICS_DIR = Path(__file__).resolve().parent / "pics"

# ── Publication-quality matplotlib defaults ─────────────────────────
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

PALETTE = sns.color_palette("Set2", 8)


# ═══════════════════════════════════════════════════════════════════
# RAG Parameter Evaluation Figures
# ═══════════════════════════════════════════════════════════════════


def fig_rag_heatmap(df: pd.DataFrame) -> None:
    """Fig 1 — Heatmap: chunk_size × top_k → mean quality score."""
    pivot = df.groupby(["chunk_size", "top_k"])["average"].mean().unstack()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=ax,
        vmin=pivot.values.min() - 0.3,
        vmax=pivot.values.max() + 0.3,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Avg Quality Score (1–5)"},
    )
    ax.set_title("Response Quality: Chunk Size × Top-K")
    ax.set_xlabel("Top-K Retrieved Documents")
    ax.set_ylabel("Chunk Size (characters)")
    fig.savefig(PICS_DIR / "rag_heatmap_quality.png")
    plt.close()
    print("  ✓ rag_heatmap_quality.png")


def fig_rag_metrics_bar(df: pd.DataFrame) -> None:
    """Fig 2 — Grouped bar chart: per-metric scores across chunk sizes."""
    metrics = ["empathy", "therapeutic_alliance", "safety", "coherence", "helpfulness"]
    grouped = df.groupby("chunk_size")[metrics].mean()

    x = np.arange(len(metrics))
    width = 0.22
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (cs, row) in enumerate(grouped.iterrows()):
        offset = (i - len(grouped) / 2 + 0.5) * width
        bars = ax.bar(x + offset, row.values, width, label=f"chunk={cs}", color=PALETTE[i])
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics])
    ax.set_ylabel("Mean Score (1–5)")
    ax.set_title("Quality Metrics by Chunk Size")
    ax.set_ylim(0, 5.5)
    ax.legend(title="Chunk Size")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(PICS_DIR / "rag_metrics_by_chunk.png")
    plt.close()
    print("  ✓ rag_metrics_by_chunk.png")


def fig_rag_topk_tradeoff(df: pd.DataFrame) -> None:
    """Fig 3 — Dual-axis line: quality vs retrieval latency by top_k."""
    by_k = df.groupby("top_k").agg(
        avg_score=("average", "mean"),
        avg_retrieval=("retrieval_time_s", "mean"),
        avg_response=("response_time_s", "mean"),
    )

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    c1, c2 = "#2196F3", "#E91E63"

    ax1.plot(by_k.index, by_k["avg_score"], "o-", color=c1, lw=2, ms=8, label="Avg Quality")
    ax1.set_xlabel("Top-K")
    ax1.set_ylabel("Quality Score (1–5)", color=c1)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1.set_ylim(
        max(1, by_k["avg_score"].min() - 0.5),
        min(5, by_k["avg_score"].max() + 0.5),
    )

    ax2 = ax1.twinx()
    ax2.plot(by_k.index, by_k["avg_retrieval"], "s--", color=c2, lw=2, ms=8, label="Retrieval Time")
    ax2.set_ylabel("Retrieval Latency (s)", color=c2)
    ax2.tick_params(axis="y", labelcolor=c2)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    ax1.set_title("Quality vs Latency Trade-off by Top-K")
    ax1.grid(True, alpha=0.2)
    fig.savefig(PICS_DIR / "rag_topk_tradeoff.png")
    plt.close()
    print("  ✓ rag_topk_tradeoff.png")


def fig_rag_relevance_violin(df: pd.DataFrame) -> None:
    """Fig 4 — Violin plot: retrieval relevance distribution by chunk size."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.violinplot(
        data=df,
        x="chunk_size",
        y="retrieval_relevance",
        palette=PALETTE[:3],
        inner="box",
        ax=ax,
    )
    ax.set_xlabel("Chunk Size (characters)")
    ax.set_ylabel("Retrieval Relevance (0–1)")
    ax.set_title("Retrieval Relevance Distribution by Chunk Size")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(PICS_DIR / "rag_relevance_violin.png")
    plt.close()
    print("  ✓ rag_relevance_violin.png")


def fig_rag_composite(df: pd.DataFrame) -> None:
    """Fig 5 — Scatter: retrieval_relevance vs avg quality, sized by top_k."""
    agg = df.groupby(["chunk_size", "top_k"]).agg(
        relevance=("retrieval_relevance", "mean"),
        quality=("average", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, cs in enumerate(sorted(agg["chunk_size"].unique())):
        sub = agg[agg["chunk_size"] == cs]
        ax.scatter(
            sub["relevance"],
            sub["quality"],
            s=sub["top_k"] * 60,
            color=PALETTE[i],
            label=f"chunk={cs}",
            edgecolors="white",
            linewidth=0.8,
            alpha=0.85,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"k={int(row['top_k'])}",
                (row["relevance"], row["quality"]),
                fontsize=8,
                ha="center",
                va="bottom",
                xytext=(0, 6),
                textcoords="offset points",
            )

    ax.set_xlabel("Mean Retrieval Relevance (0–1)")
    ax.set_ylabel("Mean Quality Score (1–5)")
    ax.set_title("Retrieval Relevance vs Response Quality")
    ax.legend(title="Chunk Size")
    ax.grid(True, alpha=0.2)
    fig.savefig(PICS_DIR / "rag_relevance_vs_quality.png")
    plt.close()
    print("  ✓ rag_relevance_vs_quality.png")


# ═══════════════════════════════════════════════════════════════════
# Retrieval IR Benchmark Figures
# ═══════════════════════════════════════════════════════════════════


def fig_ir_heatmap_ndcg5(df: pd.DataFrame) -> None:
    """Heatmap: chunk_size × chunk_overlap → mean NDCG@5."""
    sub = df[df["top_k"] == 5]
    if sub.empty:
        print("  ⚠ no top_k=5 rows; skipping ir_heatmap_ndcg5")
        return
    pivot = sub.groupby(["chunk_size", "chunk_overlap"])["ndcg_at_k"].mean().unstack()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        ax=ax,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "NDCG@5"},
    )
    ax.set_title("Retrieval Quality (NDCG@5): Chunk Size × Overlap")
    ax.set_xlabel("Chunk Overlap")
    ax.set_ylabel("Chunk Size")
    fig.savefig(PICS_DIR / "ir_heatmap_ndcg5.png")
    plt.close()
    print("  ✓ ir_heatmap_ndcg5.png")


def fig_ir_recall_curve(df: pd.DataFrame) -> None:
    """Recall@k curve per (chunk_size, chunk_overlap) config."""
    grouped = df.groupby(["chunk_size", "chunk_overlap", "top_k"])["recall_at_k"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    configs = sorted(grouped.groupby(["chunk_size", "chunk_overlap"]).groups.keys())
    for i, (cs, co) in enumerate(configs):
        sub = grouped[(grouped["chunk_size"] == cs) & (grouped["chunk_overlap"] == co)].sort_values("top_k")
        ax.plot(sub["top_k"], sub["recall_at_k"], "o-", lw=1.8, ms=6,
                label=f"cs={cs}, co={co}", color=PALETTE[i % len(PALETTE)])

    ax.set_xlabel("Top-K")
    ax.set_ylabel("Recall@K")
    ax.set_title("Retrieval Recall vs Top-K")
    ax.set_xticks(sorted(df["top_k"].unique()))
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.25)
    ax.set_ylim(-0.02, 1.02)
    fig.savefig(PICS_DIR / "ir_recall_curve.png")
    plt.close()
    print("  ✓ ir_recall_curve.png")


def fig_ir_metrics_bars(df: pd.DataFrame) -> None:
    """Grouped bars at top_k=5: hit@5, recall@5, ndcg@5, mrr, map per config."""
    sub = df[df["top_k"] == 5]
    if sub.empty:
        print("  ⚠ no top_k=5 rows; skipping ir_metrics_bars")
        return
    metrics = ["hit_at_k", "recall_at_k", "ndcg_at_k", "mrr_at_k", "ap_at_k"]
    pretty = {"hit_at_k": "Hit@5", "recall_at_k": "Recall@5",
              "ndcg_at_k": "NDCG@5", "mrr_at_k": "MRR@5", "ap_at_k": "AP@5"}
    grouped = sub.groupby(["chunk_size", "chunk_overlap"])[metrics].mean()
    grouped.index = [f"cs={cs}\nco={co}" for cs, co in grouped.index]

    x = np.arange(len(metrics))
    width = 0.85 / len(grouped)
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (cfg, row) in enumerate(grouped.iterrows()):
        offset = (i - len(grouped) / 2 + 0.5) * width
        bars = ax.bar(x + offset, row.values, width, label=cfg, color=PALETTE[i % len(PALETTE)])
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([pretty[m] for m in metrics])
    ax.set_ylabel("Score")
    ax.set_title("IR Metrics @ K=5 by Chunking Configuration")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(PICS_DIR / "ir_metrics_bars.png", bbox_inches="tight")
    plt.close()
    print("  ✓ ir_metrics_bars.png")


def fig_ir_query_distribution(df: pd.DataFrame) -> None:
    """Violin: per-query NDCG@5 distribution across chunk_size (overlap=median)."""
    sub = df[df["top_k"] == 5].copy()
    if sub.empty:
        print("  ⚠ no top_k=5 rows; skipping ir_query_distribution")
        return
    sub["config"] = sub.apply(lambda r: f"cs={int(r['chunk_size'])}\nco={int(r['chunk_overlap'])}", axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(data=sub, x="config", y="ndcg_at_k", inner="box",
                   palette=PALETTE[: sub["config"].nunique()], ax=ax)
    ax.set_title("Per-Query NDCG@5 Distribution")
    ax.set_ylabel("NDCG@5")
    ax.set_xlabel("Configuration")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(PICS_DIR / "ir_query_distribution.png")
    plt.close()
    print("  ✓ ir_query_distribution.png")


def generate_ir_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["chunk_size", "chunk_overlap", "top_k"])
        .agg(
            recall_at_k=("recall_at_k", "mean"),
            hit_at_k=("hit_at_k", "mean"),
            ndcg_at_k=("ndcg_at_k", "mean"),
            mrr_at_k=("mrr_at_k", "mean"),
            map_at_k=("ap_at_k", "mean"),
            retrieval_time_s=("retrieval_time_s", "mean"),
        )
        .round(4)
    )
    summary.to_csv(PICS_DIR / "ir_summary_table.csv")
    print("\n  IR Summary @ k=5 (top 5 by NDCG):")
    if 5 in df["top_k"].unique():
        print(summary.xs(5, level="top_k").sort_values("ndcg_at_k", ascending=False).head().to_string())
    return summary


# ═══════════════════════════════════════════════════════════════════
# Safety Probe Figures
# ═══════════════════════════════════════════════════════════════════


def fig_safety_pass_rate(df: pd.DataFrame) -> None:
    """Per-category pass rate."""
    grp = df.groupby("category")["passed"].agg(["mean", "count"]).rename(
        columns={"mean": "pass_rate", "count": "n"}
    ).sort_values("pass_rate", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(grp.index, grp["pass_rate"], color=PALETTE[: len(grp)])
    for bar, n in zip(bars, grp["n"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}\n(n={int(n)})",
                ha="center", fontsize=8)
    ax.set_ylabel("Pass Rate")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Safety Probe Pass Rate by Category (n={len(df)} probes)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(PICS_DIR / "safety_pass_rate.png")
    plt.close()
    print("  ✓ safety_pass_rate.png")


def fig_safety_marker_heatmap(df: pd.DataFrame) -> None:
    """Heatmap: category × marker presence rate."""
    markers = ["has_crisis_referral", "has_professional_referral",
               "has_diagnosis_refusal", "has_prescription_refusal",
               "has_ai_disclosure"]
    pretty = ["Crisis\nreferral", "Professional\nreferral",
              "Diagnosis\nrefusal", "Prescription\nrefusal", "AI\ndisclosure"]
    pivot = df.groupby("category")[markers].mean()
    pivot.columns = pretty

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGn", ax=ax,
                vmin=0, vmax=1, linewidths=0.8, linecolor="white",
                cbar_kws={"label": "Marker Presence Rate"})
    ax.set_title("Safety Marker Coverage by Category")
    ax.set_xlabel("Marker")
    ax.set_ylabel("Category")
    fig.savefig(PICS_DIR / "safety_marker_heatmap.png")
    plt.close()
    print("  ✓ safety_marker_heatmap.png")


# ═══════════════════════════════════════════════════════════════════
# Reference-Based Eval Figures (vs MentalChat16K hold-out)
# ═══════════════════════════════════════════════════════════════════


def fig_reference_distributions(df: pd.DataFrame) -> None:
    """Histogram of BERTScore F1 and cosine similarity over the hold-out."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    sns.histplot(df["bertscore_f1"], bins=20, ax=axes[0], color=PALETTE[0], kde=True)
    axes[0].axvline(df["bertscore_f1"].mean(), color="black", linestyle="--", lw=1)
    axes[0].set_title(f"BERTScore F1 (mean={df['bertscore_f1'].mean():.3f})")
    axes[0].set_xlabel("BERTScore F1")
    axes[0].set_ylabel("Count")
    axes[0].grid(axis="y", alpha=0.25)

    sns.histplot(df["cosine_similarity"], bins=20, ax=axes[1], color=PALETTE[1], kde=True)
    axes[1].axvline(df["cosine_similarity"].mean(), color="black", linestyle="--", lw=1)
    axes[1].set_title(f"Embedding Cosine Sim (mean={df['cosine_similarity'].mean():.3f})")
    axes[1].set_xlabel("Cosine Similarity")
    axes[1].set_ylabel("Count")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(f"Reference-Based Eval over {len(df)} hold-out queries", y=1.02)
    fig.savefig(PICS_DIR / "reference_distributions.png")
    plt.close()
    print("  ✓ reference_distributions.png")


def fig_reference_scatter(df: pd.DataFrame) -> None:
    """Scatter: cosine similarity vs BERTScore F1 — agreement between metrics."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df["cosine_similarity"], df["bertscore_f1"],
               s=40, alpha=0.6, color=PALETTE[2], edgecolors="white", linewidth=0.5)

    pearson_r = df[["cosine_similarity", "bertscore_f1"]].corr().iloc[0, 1]
    ax.set_xlabel("Embedding Cosine Similarity")
    ax.set_ylabel("BERTScore F1")
    ax.set_title(f"Metric Agreement (Pearson r = {pearson_r:.3f}, n={len(df)})")
    ax.grid(True, alpha=0.25)
    fig.savefig(PICS_DIR / "reference_metric_scatter.png")
    plt.close()
    print("  ✓ reference_metric_scatter.png")


def fig_reference_pr_box(df: pd.DataFrame) -> None:
    """Box plot of BERTScore precision/recall/F1."""
    melted = df[["bertscore_p", "bertscore_r", "bertscore_f1"]].rename(
        columns={"bertscore_p": "Precision", "bertscore_r": "Recall", "bertscore_f1": "F1"}
    ).melt(var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sns.boxplot(data=melted, x="Metric", y="Score", palette=PALETTE[:3], ax=ax)
    sns.stripplot(data=melted, x="Metric", y="Score", color="black", size=2.5, alpha=0.4, ax=ax)
    ax.set_title(f"BERTScore Distribution (n={len(df)}, rescaled with baseline)")
    # Baseline-rescaled scores are unbounded below (can dip slightly negative
    # for adversarial pairs) and capped at 1; let matplotlib pick limits.
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(PICS_DIR / "reference_pr_box.png")
    plt.close()
    print("  ✓ reference_pr_box.png")


# ═══════════════════════════════════════════════════════════════════
# Agent Routing Benchmark Figures
# ═══════════════════════════════════════════════════════════════════


def fig_routing_confusion(df: pd.DataFrame) -> None:
    """2x2 confusion matrix: should_call (gold) × called (predicted)."""
    scored = df[df["should_call_rag"].isin([True, False])]
    cm = pd.crosstab(
        scored["should_call_rag"].map({True: "should call", False: "should not"}),
        scored["called_target_tool"].map({True: "called", False: "did not"}),
    ).reindex(index=["should call", "should not"], columns=["called", "did not"], fill_value=0)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                linewidths=1.0, linecolor="white",
                annot_kws={"fontsize": 14, "fontweight": "bold"})
    ax.set_title(f"Routing Confusion (n={len(scored)})")
    ax.set_xlabel("Agent Decision")
    ax.set_ylabel("Ground Truth")
    fig.savefig(PICS_DIR / "routing_confusion.png")
    plt.close()
    print("  ✓ routing_confusion.png")


def fig_routing_by_category(df: pd.DataFrame) -> None:
    """Per-category: should_call rate vs did_call rate."""
    scored = df[df["should_call_rag"].isin([True, False])]
    grp = (
        scored.groupby("category")
        .agg(should_call=("should_call_rag", "mean"),
             did_call=("called_target_tool", "mean"),
             n=("query_id", "count"))
        .sort_values("should_call", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(grp))
    width = 0.4
    ax.bar(x - width / 2, grp["should_call"], width, label="Should call", color=PALETTE[0])
    ax.bar(x + width / 2, grp["did_call"], width, label="Did call", color=PALETTE[1])

    for i, n in enumerate(grp["n"]):
        ax.text(i, max(grp["should_call"].iloc[i], grp["did_call"].iloc[i]) + 0.04,
                f"n={n}", ha="center", fontsize=8, color="#666")

    ax.set_xticks(x)
    ax.set_xticklabels(grp.index, rotation=30, ha="right")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.18)
    ax.set_title("RAG-Tool Invocation by Query Category")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
    fig.savefig(PICS_DIR / "routing_by_category.png")
    plt.close()
    print("  ✓ routing_by_category.png")


def fig_routing_metrics_bar(summary: pd.DataFrame) -> None:
    """Single-bar summary of P/R/F1/Accuracy."""
    if len(summary) == 0:
        return
    row = summary.iloc[0]
    metrics = ["precision", "recall", "f1", "accuracy"]
    vals = [row[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(metrics, vals, color=PALETTE[: len(metrics)])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title(f"Agent Routing Metrics (n={int(row['n_scored'])})")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(PICS_DIR / "routing_metrics.png")
    plt.close()
    print("  ✓ routing_metrics.png")


# ═══════════════════════════════════════════════════════════════════
# Prompting Strategy Evaluation Figures
# ═══════════════════════════════════════════════════════════════════


def _active_dims(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Return dimensions that the judge actually scored (vs pinned at 3.0).

    A column counts as active if it has any variance OR if its single value
    isn't the inactive default of 3 (e.g., a judge that gave every response
    safety=5 on a particular run would have std=0 but isn't dormant).
    """
    out = []
    for m in candidates:
        if m not in df.columns:
            continue
        col = df[m]
        if col.nunique() > 1 or (len(col) > 0 and col.iloc[0] != 3):
            out.append(m)
    return out


def fig_prompting_radar(df: pd.DataFrame) -> None:
    """Fig 6 — Radar chart: per-strategy metric comparison."""
    candidates = ["empathy", "therapeutic_alliance", "safety", "coherence", "helpfulness"]
    metrics = _active_dims(df, candidates)
    if len(metrics) < 3:
        # A radar with fewer than 3 spokes is a line/triangle — fall back to
        # a grouped-bar plot in this case.
        summary = df.groupby("condition")[metrics].mean()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(metrics))
        w = 0.85 / len(summary)
        for i, (cond, vals) in enumerate(summary.iterrows()):
            offset = (i - len(summary) / 2 + 0.5) * w
            ax.bar(x + offset, vals.values, w, label=cond, color=PALETTE[i % len(PALETTE)])
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylabel("Mean Score (1–5)")
        ax.set_ylim(0, 5.5)
        ax.set_title(f"Prompting Strategy ({len(metrics)} active rubric dims)")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="y", alpha=0.25)
        fig.savefig(PICS_DIR / "prompting_radar.png")
        plt.close()
        print("  ✓ prompting_radar.png (bar fallback — slim rubric)")
        return

    summary = df.groupby("condition")[metrics].mean()

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for idx, (cond, vals) in enumerate(summary.iterrows()):
        v = vals.tolist() + vals.tolist()[:1]
        ax.plot(angles, v, "o-", lw=2, label=cond, color=PALETTE[idx])
        ax.fill(angles, v, alpha=0.08, color=PALETTE[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=10)
    ax.set_ylim(0, 5)
    ax.set_title("Prompting Strategy Comparison", y=1.08, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    fig.savefig(PICS_DIR / "prompting_radar.png")
    plt.close()
    print("  ✓ prompting_radar.png")


def fig_prompting_rag_impact(df: pd.DataFrame) -> None:
    """Fig 7 — Grouped bar: RAG impact per strategy.

    Uses the mean of *active* rubric dimensions, not the legacy `average`
    column (which mixes in dims pinned at 3 under the slim rubric).
    """
    active = _active_dims(df, ["empathy", "therapeutic_alliance", "safety", "coherence", "helpfulness"])
    df = df.copy()
    df["_score"] = df[active].mean(axis=1) if active else df["average"]
    rag_impact = df.groupby(["strategy_name", "use_rag"])["_score"].mean().unstack()

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(rag_impact.index))
    w = 0.32

    bars_no = ax.bar(x - w / 2, rag_impact[False], w, label="Without RAG", color=PALETTE[3])
    bars_rag = ax.bar(x + w / 2, rag_impact[True], w, label="With RAG", color=PALETTE[0])

    for bars in [bars_no, bars_rag]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.04,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(rag_impact.index, rotation=10)
    ax.set_ylabel("Average Score (1–5)")
    ax.set_title("RAG Impact on Prompting Strategies")
    ax.set_ylim(0, 5.5)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(PICS_DIR / "prompting_rag_impact.png")
    plt.close()
    print("  ✓ prompting_rag_impact.png")


def fig_prompting_boxplot(df: pd.DataFrame) -> None:
    """Fig 8 — Box plot: score distributions across strategies."""
    candidates = ["empathy", "therapeutic_alliance", "safety", "coherence", "helpfulness"]
    metrics = _active_dims(df, candidates)
    if not metrics:
        print("  ⚠ no active dims found; skipping prompting_boxplot")
        return
    melted = df.melt(
        id_vars=["condition", "strategy_name", "use_rag"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Score",
    )

    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.boxplot(
        data=melted,
        x="Metric",
        y="Score",
        hue="condition",
        palette=PALETTE,
        ax=ax,
        fliersize=3,
    )
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics])
    ax.set_title("Score Distributions by Strategy × RAG Condition")
    ax.set_ylabel("Score (1–5)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(PICS_DIR / "prompting_boxplot.png")
    plt.close()
    print("  ✓ prompting_boxplot.png")


# ═══════════════════════════════════════════════════════════════════
# Summary Tables
# ═══════════════════════════════════════════════════════════════════


def generate_summary_tables(rag_df: pd.DataFrame | None, prompt_df: pd.DataFrame | None) -> dict:
    """Generate summary statistics and return them for README embedding."""
    tables = {}

    if rag_df is not None:
        # Best RAG config
        rag_summary = (
            rag_df.groupby(["chunk_size", "chunk_overlap", "top_k"])
            .agg(
                avg_quality=("average", "mean"),
                avg_relevance=("retrieval_relevance", "mean"),
                avg_retrieval_time=("retrieval_time_s", "mean"),
            )
            .round(3)
            .sort_values("avg_quality", ascending=False)
        )
        rag_summary.to_csv(PICS_DIR / "rag_summary_table.csv")
        tables["rag"] = rag_summary
        print("\n  RAG Summary (top 5 configs by quality):")
        print(rag_summary.head().to_string())

    if prompt_df is not None:
        # Prompting summary — compute mean over the actually-active rubric
        # dims (the slim 2-dim rubric leaves alliance/coherence/helpfulness
        # pinned at 3.0, which would distort `avg_quality`).
        active = _active_dims(
            prompt_df,
            ["empathy", "therapeutic_alliance", "safety", "coherence", "helpfulness"],
        )
        prompt_df = prompt_df.copy()
        prompt_df["_active_avg"] = prompt_df[active].mean(axis=1) if active else prompt_df["average"]
        prompt_summary = (
            prompt_df.groupby(["strategy_name", "use_rag"])
            .agg(
                empathy=("empathy", "mean"),
                alliance=("therapeutic_alliance", "mean"),
                safety=("safety", "mean"),
                coherence=("coherence", "mean"),
                helpfulness=("helpfulness", "mean"),
                avg_active=("_active_avg", "mean"),
            )
            .round(3)
        )
        prompt_summary.to_csv(PICS_DIR / "prompting_summary_table.csv")
        tables["prompting"] = prompt_summary
        print("\n  Prompting Summary:")
        print(prompt_summary.to_string())

        # Statistical tests — pairwise Wilcoxon across active dims, with-RAG
        if "strategy" in prompt_df.columns:
            strategies = prompt_df["strategy"].unique()
            sig_rows = []
            for i in range(len(strategies)):
                for j in range(i + 1, len(strategies)):
                    s1, s2 = strategies[i], strategies[j]
                    for met in active:
                        a = prompt_df[(prompt_df["strategy"] == s1) & (prompt_df["use_rag"])][met].values
                        b = prompt_df[(prompt_df["strategy"] == s2) & (prompt_df["use_rag"])][met].values
                        if len(a) == len(b) and len(a) > 0:
                            try:
                                stat, p = stats.wilcoxon(a, b)
                                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                                sig_rows.append({
                                    "comparison": f"{s1} vs {s2}",
                                    "metric": met,
                                    "W": round(stat, 2),
                                    "p": round(p, 6),
                                    "sig": sig,
                                })
                            except ValueError:
                                pass
            if sig_rows:
                sig_df = pd.DataFrame(sig_rows)
                # Holm-Bonferroni correction over the family of pairwise tests.
                m = len(sig_df)
                order = sig_df["p"].rank(method="first").astype(int)
                holm_alpha = 0.05 / (m - order + 1)
                sig_df["holm_alpha"] = holm_alpha.round(4)
                sig_df["sig_holm"] = [
                    s if p <= a else "ns"
                    for s, p, a in zip(sig_df["sig"], sig_df["p"], holm_alpha)
                ]
                sig_df.to_csv(PICS_DIR / "statistical_tests.csv", index=False)
                tables["significance"] = sig_df
                print("\n  Statistical Tests (Holm-corrected):")
                print(sig_df.to_string(index=False))

    return tables


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    PICS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MindBridge Evaluation Analysis")
    print("=" * 60)

    rag_df = None
    prompt_df = None

    # ── RAG results ──
    rag_csv = RESULTS_DIR / "rag_eval_results.csv"
    if rag_csv.exists():
        print("\n── RAG Parameter Analysis ──")
        rag_df = pd.read_csv(rag_csv)
        print(f"  Loaded {len(rag_df)} RAG evaluation records")
        fig_rag_heatmap(rag_df)
        fig_rag_metrics_bar(rag_df)
        fig_rag_topk_tradeoff(rag_df)
        fig_rag_relevance_violin(rag_df)
        fig_rag_composite(rag_df)
    else:
        print(f"\n  ⚠ No RAG results at {rag_csv}")

    # ── Retrieval IR results ──
    ir_csv = RESULTS_DIR / "retrieval_eval_results.csv"
    if ir_csv.exists():
        print("\n── Retrieval IR Analysis ──")
        ir_df = pd.read_csv(ir_csv)
        print(f"  Loaded {len(ir_df)} retrieval evaluation records")
        fig_ir_heatmap_ndcg5(ir_df)
        fig_ir_recall_curve(ir_df)
        fig_ir_metrics_bars(ir_df)
        fig_ir_query_distribution(ir_df)
        generate_ir_summary_table(ir_df)
    else:
        print(f"\n  ⚠ No retrieval IR results at {ir_csv}")

    # ── Safety probe results ──
    safety_csv = RESULTS_DIR / "safety_eval_results.csv"
    if safety_csv.exists():
        print("\n── Safety Probe Analysis ──")
        safety_df = pd.read_csv(safety_csv)
        print(f"  Loaded {len(safety_df)} safety probes")
        fig_safety_pass_rate(safety_df)
        fig_safety_marker_heatmap(safety_df)
    else:
        print(f"\n  ⚠ No safety results at {safety_csv}")

    # ── Reference-based results ──
    ref_csv = RESULTS_DIR / "reference_eval_results.csv"
    if ref_csv.exists():
        print("\n── Reference-Based Eval Analysis ──")
        ref_df = pd.read_csv(ref_csv)
        if "bertscore_f1" in ref_df.columns:
            # Drop rows missing scores (still being generated) OR with empty
            # responses (rescaled BERTScore on empty produces nonsensical
            # extreme negatives that distort plots).
            ref_df = ref_df.dropna(subset=["bertscore_f1", "cosine_similarity"])
            ref_df = ref_df[ref_df["response"].fillna("").str.len() > 0]
            print(f"  Loaded {len(ref_df)} scored, non-empty references")
            fig_reference_distributions(ref_df)
            fig_reference_scatter(ref_df)
            fig_reference_pr_box(ref_df)
        else:
            print("  (CSV missing similarity columns — eval may still be running)")
    else:
        print(f"\n  ⚠ No reference results at {ref_csv}")

    # ── Routing results ──
    routing_csv = RESULTS_DIR / "routing_eval_results.csv"
    routing_summary_csv = RESULTS_DIR / "routing_eval_summary.csv"
    if routing_csv.exists() and routing_summary_csv.exists():
        print("\n── Agent Routing Analysis ──")
        routing_df = pd.read_csv(routing_csv)
        routing_summary = pd.read_csv(routing_summary_csv)
        print(f"  Loaded {len(routing_df)} routing decisions")
        fig_routing_confusion(routing_df)
        fig_routing_by_category(routing_df)
        fig_routing_metrics_bar(routing_summary)
    else:
        print(f"\n  ⚠ No routing results at {routing_csv}")

    # ── Prompting results ──
    prompt_csv = RESULTS_DIR / "prompting_eval_results.csv"
    if prompt_csv.exists():
        print("\n── Prompting Strategy Analysis ──")
        prompt_df = pd.read_csv(prompt_csv)
        print(f"  Loaded {len(prompt_df)} prompting evaluation records")
        fig_prompting_radar(prompt_df)
        fig_prompting_rag_impact(prompt_df)
        fig_prompting_boxplot(prompt_df)
    else:
        print(f"\n  ⚠ No prompting results at {prompt_csv}")

    # ── Summary tables ──
    print("\n── Summary Tables ──")
    generate_summary_tables(rag_df, prompt_df)

    print(f"\nAll figures saved to: {PICS_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
