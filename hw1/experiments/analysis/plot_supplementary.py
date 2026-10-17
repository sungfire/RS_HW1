"""Supplementary BPR figure from the 5-seed runs (mean ± std).

usage: python plot_supplementary.py <seed_csv_dir> <out_png>
"""
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

seed_dir, out_png = sys.argv[1], sys.argv[2]
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#898781"

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

runs = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(seed_dir, "seed_*.csv"))])
bpr = runs[runs.model == "BPR"]
pop = runs[runs.model == "Pop"]["ndcg@10"].mean()
n_seeds = runs.seed.nunique()

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)

ax = axes[0]
for reg, color, marker in [(0.01, ORANGE, "s"), (0.1, BLUE, "o")]:
    g = bpr[bpr.reg_lambda == reg].groupby("embedding_size")["ndcg@10"].agg(["mean", "std"]).reset_index()
    ax.errorbar(g.embedding_size, g["mean"], yerr=g["std"], color=color, marker=marker,
                capsize=3, elinewidth=1, label=f"reg_lambda = {reg}")
ax.set_xticks([32, 64, 128, 256])
ax.set_xlabel("Embedding size")
ax.set_ylabel("NDCG@10")
ax.set_title("Embedding size matters only under weak regularization", fontsize=10)
ax.legend(frameon=False, loc="lower right")

ax = axes[1]
g = bpr[bpr.embedding_size == 64].groupby("reg_lambda")["ndcg@10"].agg(["mean", "std"]).reset_index()
pos = range(len(g))
ax.errorbar(pos, g["mean"], yerr=g["std"], color=BLUE, marker="o", capsize=3, elinewidth=1)
ax.set_xticks(list(pos), [f"{r:g}" for r in g.reg_lambda])
ax.axhline(pop, color=MUTED, linewidth=1)
ax.annotate("Pop (most-popular baseline)", (0, pop), xytext=(2, 4), textcoords="offset points",
            color="#52514e", fontsize=9)
ax.set_xlabel("reg_lambda (embedding size = 64)")
ax.set_title("Stronger L2 penalty -> lower NDCG; 0.5 falls to the Pop level", fontsize=10)

fig.suptitle(f"BPR on ml-100k, mean ± std over {n_seeds} seeds", fontsize=11)
fig.tight_layout()
fig.savefig(out_png, dpi=200)
print("saved", out_png)
