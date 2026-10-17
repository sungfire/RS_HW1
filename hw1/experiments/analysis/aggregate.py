"""Summarise the official (seed 2020) results and the 5-seed runs.

usage: python aggregate.py <official_results_dir> <seed_csv_dir>
"""
import glob
import os
import sys

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

official_dir, seed_dir = sys.argv[1], sys.argv[2]
ACC = ["recall@10", "precision@10", "ndcg@10", "mrr@10", "hit@10"]
EXTRA = ["itemcoverage@10", "averagepopularity@10", "giniindex@10"]
KEYS = ["model", "knn_method", "k", "embedding_size", "reg_lambda"]


def mean_std(df, cols):
    g = df.groupby(KEYS, dropna=False)[cols]
    m, s = g.mean(), g.std()
    out = m.round(4).astype(str) + " ± " + s.round(4).astype(str)
    return out.reset_index()


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


# ---------------- official results (what the student's scripts produce) ----------------
section("OFFICIAL seed 2020 (hw1/results)")
for f in ["q1_knn.csv", "q2_embedding.csv", "q2_reg.csv", "q3_item2item.csv"]:
    print(f"\n-- {f}")
    print(pd.read_csv(os.path.join(official_dir, f)).to_string(index=False))

# ---------------- 5-seed runs ----------------
runs = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(seed_dir, "seed_*.csv")))])
print(f"\nseeds: {sorted(runs.seed.unique())}, rows: {len(runs)}")

# consistency check: seed 2020 from the analysis runner vs the official run
off = pd.read_csv(os.path.join(official_dir, "q1_knn.csv"))
chk = runs[(runs.seed == 2020) & (runs.model == "ItemKNN")].merge(off, on=["knn_method", "k"], suffixes=("", "_off"))
print("max |diff| seed2020 runner vs official (Q1 ndcg):", (chk["ndcg@10"] - chk["ndcg@10_off"]).abs().max())
off2 = pd.concat([pd.read_csv(os.path.join(official_dir, f)) for f in ["q2_embedding.csv", "q2_reg.csv"]]).drop_duplicates()
chk2 = runs[(runs.seed == 2020) & (runs.model == "BPR")].merge(off2, on=["embedding_size", "reg_lambda"], suffixes=("", "_off"))
print("max |diff| seed2020 runner vs official (Q2 ndcg):", (chk2["ndcg@10"] - chk2["ndcg@10_off"]).abs().max())

section("Q1 KNN: mean ± std over seeds")
q1 = runs[runs.exp == "q1"]
print(mean_std(q1, ACC).drop(columns=["embedding_size", "reg_lambda"]).to_string(index=False))
print()
print(mean_std(q1, EXTRA).drop(columns=["embedding_size", "reg_lambda"]).to_string(index=False))

section("Q1 item - user (per seed), NDCG@10 / Recall@10")
piv = q1.pivot_table(index=["seed", "k"], columns="knn_method", values=["ndcg@10", "recall@10", "mrr@10"])
for metric in ["ndcg@10", "recall@10", "mrr@10"]:
    d = (piv[(metric, "item")] - piv[(metric, "user")]).unstack("k")
    print(f"\n{metric}: item - user")
    print(d.round(4).to_string())
    print("mean:", d.mean().round(4).to_dict(), "| item wins (of 5):", (d > 0).sum().to_dict())

section("Q1 best k per seed (NDCG@10)")
best = q1.sort_values("ndcg@10").groupby(["seed", "knn_method"]).tail(1)
print(best[["seed", "knn_method", "k", "ndcg@10"]].sort_values(["knn_method", "seed"]).to_string(index=False))

section("Q2 BPR: mean ± std over seeds (incl. supplementary)")
bpr = runs[runs.model == "BPR"]
print(mean_std(bpr, ["ndcg@10", "recall@10", "mrr@10"] + EXTRA).drop(columns=["knn_method", "k"]).sort_values(["reg_lambda", "embedding_size"]).to_string(index=False))

section("Q2 embedding sweep: spread of means vs seed std")
for reg in [0.1, 0.01]:
    d = bpr[bpr.reg_lambda == reg].groupby("embedding_size")["ndcg@10"].agg(["mean", "std"])
    print(f"reg={reg}: means {d['mean'].round(4).to_dict()} | range of means {d['mean'].max() - d['mean'].min():.4f} | avg seed std {d['std'].mean():.4f}")

section("Q3 + Pop: mean ± std over seeds")
q3 = runs[(runs.exp == "q3") | (runs.model == "Pop") | ((runs.model == "ItemKNN") & (runs.knn_method == "item") & (runs.k == 100))]
print(mean_std(q3, ACC + EXTRA).drop(columns=["knn_method", "k", "embedding_size", "reg_lambda"]).to_string(index=False))

section("Q3 pairwise wins over 5 seeds (NDCG@10)")
p = q3.assign(name=q3.model).pivot_table(index="seed", columns="name", values="ndcg@10")
for a, b in [("EASE", "SLIMElastic"), ("SLIMElastic", "ItemKNN"), ("EASE", "ItemKNN")]:
    diff = p[a] - p[b]
    print(f"{a} - {b}: mean {diff.mean():.4f}, min {diff.min():.4f}, wins {(diff > 0).sum()}/5")

section("runtime (seconds, single thread, 5 parallel processes)")
print(runs.groupby(["model"])["seconds"].describe()[["mean", "min", "max"]].round(1).to_string())
