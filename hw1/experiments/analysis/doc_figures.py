"""Figures for the report, drawn at their printed size (6.5 in wide) so text stays legible.

usage: python doc_figures.py <results_dir> <out_dir>
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

results_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

METRICS = {"recall@10": "Recall@10", "precision@10": "Precision@10", "ndcg@10": "NDCG@10",
           "mrr@10": "MRR@10", "hit@10": "Hit@10"}
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 7.5,
    "axes.titlesize": 8,
    "text.color": "#0b0b0b",
    "axes.titlecolor": "#0b0b0b",
    "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.edgecolor": "#c3c2b7",
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "lines.markeredgecolor": "white",
    "lines.markeredgewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

# (1) KNN: one panel per metric, ItemKNN vs UserKNN
df = pd.read_csv(os.path.join(results_dir, "q1_knn.csv"))
fig, axes = plt.subplots(2, 3, figsize=(6.5, 3.45))
for ax, (metric, title) in zip(axes.flat, METRICS.items()):
    for method, label, color, marker in [("item", "ItemKNN", BLUE, "o"), ("user", "UserKNN", ORANGE, "s")]:
        d = df[df.knn_method == method].sort_values("k")
        ax.plot(d.k, d[metric], color=color, marker=marker, label=label)
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 100, 500], ["1", "5", "10", "100", "500"])
    ax.minorticks_off()
    ax.set_title(title, pad=3)
    ax.set_xlabel("k (number of neighbors)", labelpad=1)
legend_ax = axes.flat[-1]
legend_ax.axis("off")
legend_ax.legend(*axes.flat[0].get_legend_handles_labels(), loc="center", frameon=False, fontsize=8.5)
fig.tight_layout(pad=0.4, h_pad=0.8, w_pad=0.8)
fig.savefig(os.path.join(out_dir, "q1_knn_doc.png"), dpi=300)

# (2) BPR: NDCG@10 vs embedding size / reg_lambda, shared y so effect sizes compare
fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.35), sharey=True)
panels = [
    ("q2_embedding.csv", "embedding_size", "Embedding size", "(i) NDCG@10 vs embedding size (reg_lambda = 0.1)"),
    ("q2_reg.csv", "reg_lambda", "reg_lambda", "(ii) NDCG@10 vs reg_lambda (embedding size = 64)"),
]
for ax, (filename, column, xlabel, title) in zip(axes, panels):
    d = pd.read_csv(os.path.join(results_dir, filename)).sort_values(column)
    ax.plot(d[column], d["ndcg@10"], color=BLUE, marker="o")
    ax.set_xticks(d[column])
    ax.set_xlabel(xlabel, labelpad=1)
    ax.set_title(title, pad=3)
axes[0].set_ylabel("NDCG@10")
fig.tight_layout(pad=0.4, w_pad=1.0)
fig.savefig(os.path.join(out_dir, "q2_bpr_doc.png"), dpi=300)
print("saved to", out_dir)
