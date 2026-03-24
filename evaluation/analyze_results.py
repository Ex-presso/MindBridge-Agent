"""
Generate figures and tables from evaluation results.

Usage:
    uv run python -m evaluation.analyze_results

Outputs figures to evaluation/results/figures/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
    }
)


def analyze_rag_results():
    """Generate RAG parameter comparison figures."""
    csv_path = RESULTS_DIR / "rag_eval_results.csv"
    if not csv_path.exists():
        print(f"No RAG results found at {csv_path}. Run eval_rag.py first.")
        return

    df = pd.read_csv(csv_path)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: Heatmap - chunk_size × top_k → average quality score ──
    pivot = df.groupby(["chunk_size", "top_k"])["average"].mean().unstack()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=ax,
        vmin=1,
        vmax=5,
        linewidths=0.5,
    )
    ax.set_title("Average Quality Score: Chunk Size × Top-K")
    ax.set_xlabel("Top-K Retrieved Documents")
    ax.set_ylabel("Chunk Size (characters)")
    fig.savefig(FIGURES_DIR / "rag_heatmap_chunksize_topk.png")
    plt.close()
    print("  Saved: rag_heatmap_chunksize_topk.png")

    # ── Figure 2: Heatmap - chunk_size × chunk_overlap → average score ──
    pivot2 = df.groupby(["chunk_size", "chunk_overlap"])["average"].mean().unstack()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        pivot2,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        ax=ax,
        vmin=1,
        vmax=5,
        linewidths=0.5,
    )
    ax.set_title("Average Quality Score: Chunk Size × Chunk Overlap")
    ax.set_xlabel("Chunk Overlap (characters)")
    ax.set_ylabel("Chunk Size (characters)")
    fig.savefig(FIGURES_DIR / "rag_heatmap_chunksize_overlap.png")
    plt.close()
    print("  Saved: rag_heatmap_chunksize_overlap.png")

    # ── Figure 3: Line chart - top_k vs latency tradeoff ──
    latency_by_k = df.groupby("top_k").agg(
        avg_score=("average", "mean"),
        avg_retrieval_time=("retrieval_time_s", "mean"),
        avg_response_time=("response_time_s", "mean"),
    )

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color1 = "#2196F3"
    color2 = "#FF5722"

    ax1.plot(
        latency_by_k.index,
        latency_by_k["avg_score"],
        "o-",
        color=color1,
        linewidth=2,
        markersize=8,
        label="Avg Quality Score",
    )
    ax1.set_xlabel("Top-K")
    ax1.set_ylabel("Average Quality Score", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(1, 5)

    ax2 = ax1.twinx()
    ax2.plot(
        latency_by_k.index,
        latency_by_k["avg_retrieval_time"],
        "s--",
        color=color2,
        linewidth=2,
        markersize=8,
        label="Avg Retrieval Time",
    )
    ax2.set_ylabel("Retrieval Time (s)", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Quality vs Latency Tradeoff by Top-K")
    ax1.grid(True, alpha=0.3)
    fig.savefig(FIGURES_DIR / "rag_topk_tradeoff.png")
    plt.close()
    print("  Saved: rag_topk_tradeoff.png")

    # ── Figure 4: Retrieval relevance distribution ──
    fig, ax = plt.subplots(figsize=(10, 6))
    for cs in sorted(df["chunk_size"].unique()):
        subset = df[df["chunk_size"] == cs]
        ax.hist(
            subset["retrieval_relevance"],
            bins=20,
            alpha=0.5,
            label=f"chunk_size={cs}",
        )
    ax.set_xlabel("Retrieval Relevance Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Retrieval Relevance Distribution by Chunk Size")
    ax.legend()
    fig.savefig(FIGURES_DIR / "rag_relevance_distribution.png")
    plt.close()
    print("  Saved: rag_relevance_distribution.png")

    # ── Table: Best config per metric ──
    metrics = ["average", "empathy", "helpfulness", "safety", "retrieval_relevance"]
    best_configs = []
    for metric in metrics:
        best = (
            df.groupby(["chunk_size", "chunk_overlap", "top_k"])[metric]
            .mean()
            .idxmax()
        )
        best_score = (
            df.groupby(["chunk_size", "chunk_overlap", "top_k"])[metric]
            .mean()
            .max()
        )
        best_configs.append(
            {
                "metric": metric,
                "chunk_size": best[0],
                "chunk_overlap": best[1],
                "top_k": best[2],
                "score": round(best_score, 3),
            }
        )

    best_df = pd.DataFrame(best_configs)
    best_df.to_csv(RESULTS_DIR / "rag_best_configs.csv", index=False)
    print("\n  Best configuration per metric:")
    print(best_df.to_string(index=False))


def analyze_prompting_results():
    """Generate prompting strategy comparison figures."""
    csv_path = RESULTS_DIR / "prompting_eval_results.csv"
    if not csv_path.exists():
        print(f"No prompting results found at {csv_path}. Run eval_prompting.py first.")
        return

    df = pd.read_csv(csv_path)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    metrics = ["empathy", "therapeutic_alliance", "safety", "coherence", "helpfulness"]

    # ── Figure 1: Radar chart - strategy comparison ──
    summary = df.groupby("condition")[metrics].mean()

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0", "#00BCD4"]

    for idx, (condition, values) in enumerate(summary.iterrows()):
        vals = values.tolist()
        vals += vals[:1]
        ax.plot(
            angles,
            vals,
            "o-",
            linewidth=2,
            label=condition,
            color=colors[idx % len(colors)],
        )
        ax.fill(angles, vals, alpha=0.1, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics])
    ax.set_ylim(0, 5)
    ax.set_title("Prompting Strategy Comparison", y=1.08, fontsize=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.savefig(FIGURES_DIR / "prompting_radar_chart.png")
    plt.close()
    print("  Saved: prompting_radar_chart.png")

    # ── Figure 2: Box plot - score distributions ──
    df_melted = df.melt(
        id_vars=["condition", "strategy_name", "use_rag"],
        value_vars=metrics,
        var_name="metric",
        value_name="score",
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(
        data=df_melted,
        x="metric",
        y="score",
        hue="condition",
        ax=ax,
    )
    ax.set_title("Score Distributions by Strategy and Condition")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score (1-5)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.savefig(FIGURES_DIR / "prompting_boxplot.png")
    plt.close()
    print("  Saved: prompting_boxplot.png")

    # ── Figure 3: Grouped bar chart - RAG impact ──
    rag_impact = df.groupby(["strategy_name", "use_rag"])["average"].mean().unstack()

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(rag_impact.index))
    width = 0.35

    bars1 = ax.bar(x - width / 2, rag_impact[False], width, label="Without RAG", color="#FF9800")
    bars2 = ax.bar(x + width / 2, rag_impact[True], width, label="With RAG", color="#2196F3")

    ax.set_xlabel("Strategy")
    ax.set_ylabel("Average Score")
    ax.set_title("RAG Impact on Each Prompting Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels(rag_impact.index, rotation=15)
    ax.legend()
    ax.set_ylim(0, 5)
    ax.grid(axis="y", alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    fig.savefig(FIGURES_DIR / "prompting_rag_impact.png")
    plt.close()
    print("  Saved: prompting_rag_impact.png")

    # ── Statistical significance tests ──
    print("\n  Statistical Significance Tests (Wilcoxon signed-rank):")
    strategy_keys = df["strategy"].unique()

    sig_results = []
    for i in range(len(strategy_keys)):
        for j in range(i + 1, len(strategy_keys)):
            s1, s2 = strategy_keys[i], strategy_keys[j]
            # Compare with RAG
            scores1 = df[(df["strategy"] == s1) & (df["use_rag"] == True)]["average"].values
            scores2 = df[(df["strategy"] == s2) & (df["use_rag"] == True)]["average"].values

            if len(scores1) == len(scores2) and len(scores1) > 0:
                try:
                    stat, p_value = stats.wilcoxon(scores1, scores2)
                    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                    sig_results.append(
                        {
                            "comparison": f"{s1} vs {s2}",
                            "condition": "with_rag",
                            "statistic": round(stat, 3),
                            "p_value": round(p_value, 6),
                            "significance": sig,
                        }
                    )
                    print(f"    {s1} vs {s2} (with RAG): p={p_value:.6f} {sig}")
                except ValueError:
                    print(f"    {s1} vs {s2}: could not compute (identical distributions)")

    if sig_results:
        sig_df = pd.DataFrame(sig_results)
        sig_df.to_csv(RESULTS_DIR / "statistical_tests.csv", index=False)

    # ── Qualitative examples table ──
    # Pick 3 diverse queries and show responses from each strategy
    example_ids = [1, 5, 15]  # anxiety, depression, anger
    examples_df = df[
        (df["query_id"].isin(example_ids)) & (df["use_rag"] == True)
    ][["query_id", "strategy_name", "response_text", "average"]].copy()
    examples_df.to_csv(RESULTS_DIR / "qualitative_examples.csv", index=False)
    print("\n  Qualitative examples saved to qualitative_examples.csv")


def main():
    print("=" * 60)
    print("MindBridge Evaluation Analysis")
    print("=" * 60)

    print("\n--- RAG Parameter Analysis ---")
    analyze_rag_results()

    print("\n--- Prompting Strategy Analysis ---")
    analyze_prompting_results()

    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
