"""Multi-seed runs with beyond-accuracy metrics, for analysing the HW1 results.

usage:
    python runner.py <seed> <out_dir> <dataset_parent_dir>
Each seed runs in its own working directory (out_dir/run_<seed>) so that
RecBole's ./log and ./saved directories are never shared between processes.
"""
import copy
import csv
import os
import sys
import time

import torch

torch.set_num_threads(1)

from recbole.evaluator.collector import Collector
from recbole.quick_start import run_recbole


def _get_data_struct(self):
    # RecBole calls .cpu() on every collected value, but ItemCoverage/GiniIndex
    # store an int and AveragePopularity a Counter -> AttributeError. Same logic, tensors only.
    for key, value in self.data_struct._data_dict.items():
        if isinstance(value, torch.Tensor):
            self.data_struct._data_dict[key] = value.cpu()
    returned_struct = copy.deepcopy(self.data_struct)
    for key in ["rec.topk", "rec.meanrank", "rec.score", "rec.items", "data.label"]:
        if key in self.data_struct:
            del self.data_struct[key]
    return returned_struct


Collector.get_data_struct = _get_data_struct

METRICS = ["Recall", "MRR", "NDCG", "Hit", "Precision", "ItemCoverage", "AveragePopularity", "GiniIndex"]

CONFIGS = (
    # Q1
    [("q1", "ItemKNN", {"knn_method": m, "k": k}) for m in ["item", "user"] for k in [1, 5, 10, 100, 500]]
    # Q2 (embedding_size=64, reg_lambda=0.1 is shared by both sweeps)
    + [("q2", "BPR", {"embedding_size": e, "reg_lambda": 0.1}) for e in [32, 64, 128, 256]]
    + [("q2", "BPR", {"embedding_size": 64, "reg_lambda": r}) for r in [0.01, 0.5]]
    # Q3 (ItemKNN default = item, k=100, already above)
    + [("q3", m, {}) for m in ["SLIMElastic", "EASE"]]
    # supplementary references
    + [("extra", "Pop", {})]
    + [("extra", "BPR", {"embedding_size": 64, "reg_lambda": 0.0})]
    + [("extra", "BPR", {"embedding_size": e, "reg_lambda": 0.01}) for e in [32, 128, 256]]
)

FIELDS = ["exp", "model", "knn_method", "k", "embedding_size", "reg_lambda", "seed", "seconds"]


def main(seed, out_dir, data_path):
    out_dir, data_path = os.path.abspath(out_dir), os.path.abspath(data_path)
    work_dir = os.path.join(out_dir, f"run_{seed}")
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)
    out_file = os.path.join(out_dir, f"seed_{seed}.csv")
    rows = []
    for exp, model, params in CONFIGS:
        start = time.time()
        result = run_recbole(
            model=model,
            dataset="ml-100k",
            config_dict={
                **params,
                "seed": seed,
                "metrics": METRICS,
                "topk": [10],
                "show_progress": False,
                "state": "WARNING",
                "data_path": data_path,
            },
        )
        row = {"exp": exp, "model": model, **params, "seed": seed,
               "seconds": round(time.time() - start, 1), **result["test_result"]}
        rows.append(row)
        print(">>>", row, flush=True)

        fields = FIELDS + [k for k in row if k not in FIELDS]
        with open(out_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    seed, out_dir, data_path = int(sys.argv[1]), sys.argv[2], sys.argv[3]
    sys.argv = sys.argv[:1]
    main(seed, out_dir, data_path)
