"""RS HW1 experiments on ml-100k.

Run from the repository root:
    python hw1/experiments/run_experiments.py           # q1, q2, q3
    python hw1/experiments/run_experiments.py q2        # only some of them
Results are saved to hw1/results/*.csv
"""
import csv
import os
import sys

from recbole.quick_start import run_recbole

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def run(model, **params):
    result = run_recbole(
        model=model,
        dataset="ml-100k",
        config_dict={**params, "show_progress": False, "state": "WARNING"},
    )
    row = {"model": model, **params, **result["test_result"]}
    print(">>>", row, flush=True)
    return row


def save(rows, filename):
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(os.path.join(RESULT_DIR, filename), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def q1():
    """ItemKNN / UserKNN with different numbers of neighbors k."""
    rows = [
        run("ItemKNN", knn_method=method, k=k)
        for method in ["item", "user"]
        for k in [1, 5, 10, 100, 500]
    ]
    save(rows, "q1_knn.csv")


def q2():
    """BPR: embedding size sweep (reg_lambda=0.1), reg_lambda sweep (embedding_size=64)."""
    rows = [run("BPR", embedding_size=e, reg_lambda=0.1) for e in [32, 64, 128, 256]]
    save(rows, "q2_embedding.csv")
    rows = [run("BPR", embedding_size=64, reg_lambda=r) for r in [0.01, 0.1, 0.5]]
    save(rows, "q2_reg.csv")


def q3():
    """Item-to-item models vs ItemKNN, all with default hyperparameters."""
    rows = [run(model) for model in ["ItemKNN", "SLIMElastic", "EASE"]]
    save(rows, "q3_item2item.csv")


if __name__ == "__main__":
    targets = sys.argv[1:] or ["q1", "q2", "q3"]
    sys.argv = sys.argv[:1]  # RecBole would otherwise try to parse "q1" etc. as its own options
    for name in targets:
        {"q1": q1, "q2": q2, "q3": q3}[name]()
