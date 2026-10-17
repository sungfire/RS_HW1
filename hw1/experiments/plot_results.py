"""Draw the HW1 figures from hw1/results/*.csv (run after hw1/experiments/run_experiments.py).

    python hw1/experiments/plot_results.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
METRICS = {
    "recall@10": "Recall@10",
    "precision@10": "Precision@10",
    "ndcg@10": "NDCG@10",
    "mrr@10": "MRR@10",
    "hit@10": "Hit@10",
}
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "font.size": 10,
    "text.color": "#0b0b0b",
    "axes.titlecolor": "#0b0b0b",
    "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "axes.edgecolor": "#c3c2b7",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "lines.markeredgecolor": "white",
    "lines.markeredgewidth": 1.0,
})


def q1():
    df = pd.read_csv(os.path.join(RESULT_DIR, "q1_knn.csv"))
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, (metric, title) in zip(axes.flat, METRICS.items()):
        for method, label, color, marker in [("item", "ItemKNN", BLUE, "o"), ("user", "UserKNN", ORANGE, "s")]:
            d = df[df.knn_method == method].sort_values("k")
            ax.plot(d.k, d[metric], color=color, marker=marker, label=label)
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 100, 500], ["1", "5", "10", "100", "500"])
        ax.minorticks_off()
        ax.set_title(title)
        ax.set_xlabel("k (number of neighbors)")
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    legend_ax.legend(*axes.flat[0].get_legend_handles_labels(), loc="center", frameon=False, fontsize=11)
    fig.suptitle("ItemKNN vs UserKNN on ml-100k: metrics vs k")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "q1_knn.png"), dpi=200)


def q2():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)  # same metric -> same scale
    panels = [
        ("q2_embedding.csv", "embedding_size", "Embedding size", "BPR: NDCG@10 vs embedding size (reg_lambda = 0.1)"),
        ("q2_reg.csv", "reg_lambda", "Regularization lambda", "BPR: NDCG@10 vs reg_lambda (embedding size = 64)"),
    ]
    for ax, (filename, column, xlabel, title) in zip(axes, panels):
        d = pd.read_csv(os.path.join(RESULT_DIR, filename)).sort_values(column)
        ax.plot(d[column], d["ndcg@10"], color=BLUE, marker="o")
        ax.set_xticks(d[column])
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("NDCG@10")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "q2_bpr.png"), dpi=200)


def q3():
    df = pd.read_csv(os.path.join(RESULT_DIR, "q3_item2item.csv")).set_index("model")
    print(df[list(METRICS)].rename(columns=METRICS).to_string())


if __name__ == "__main__":
    q1()
    q2()
    q3()
    print("saved:", os.path.join(RESULT_DIR, "q1_knn.png"), os.path.join(RESULT_DIR, "q2_bpr.png"))
