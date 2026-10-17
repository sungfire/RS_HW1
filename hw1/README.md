# HW1: Recommender Systems on ml-100k (RecBole)

Every number and figure in the report was produced by the code in this folder, run on this
repository (RecBole master @ `7b02be5` plus the two changes below). All commands are run from
the repository root.

```
hw1/
├── experiments/            # code
│   ├── run_experiments.py  # all runs for (1)-(3)
│   ├── plot_results.py     # figures + Table 1
│   └── analysis/           # supplementary analysis used in the explanations
└── results/                # outputs
    ├── q1_knn.csv, q2_embedding.csv, q2_reg.csv, q3_item2item.csv
    ├── q1_knn.png, q2_bpr.png
    └── analysis/           # 5-seed runs, diagnostics, CLI re-check, report figures
```

## Changes to RecBole

- `recbole/trainer/trainer.py`: `torch.load(..., weights_only=False)`. PyTorch >= 2.6 otherwise
  refuses to reload RecBole's own checkpoints before the test evaluation.
- `recbole/model/general_recommender/bpr.py`, `recbole/properties/model/BPR.yaml`: new `reg_lambda`
  (default `0.0`). RecBole's BPR has no regularization term, so `reg_lambda` weights an L2 penalty on the
  user / positive / negative embeddings of each batch:
  `loss = BPR + reg_lambda * 1/2 * mean(||u||^2 + ||i||^2 + ||j||^2)`.

## Environment

Python 3.10 or 3.11 (RecBole needs `ray<=2.6.3`, which has no wheels for Python >= 3.12):

```bash
pip install -e . "numpy<2" "setuptools<81" "pandas<3" matplotlib
```

## Main experiments (report Figures 1-2 and Table 1)

RecBole defaults: per-user random 8:1:1 split, full ranking, metrics@10, seed 2020.

```bash
python hw1/experiments/run_experiments.py    # 20 runs, about 5 minutes on a 4-core CPU -> hw1/results/*.csv
python hw1/experiments/plot_results.py       # hw1/results/q1_knn.png, q2_bpr.png, and prints Table 1
```

UserKNN is `ItemKNN` with `knn_method=user`. The `--method=user` flag from the handout is ignored by
RecBole and silently gives item-based results.

The handout's CLI form gives the same numbers, for example
`python run_recbole.py --dataset=ml-100k --model=ItemKNN --k=10 --knn_method=user`.
`bash hw1/experiments/analysis/cli_check.sh` re-runs all 20 settings that way. Its output,
`hw1/results/analysis/cli_check.txt`, matches `hw1/results` exactly.

## Supplementary analysis (claims made in the report's explanations)

Scripts are in `hw1/experiments/analysis/`, outputs in `hw1/results/analysis/`.

| Claim | Command | Output |
|---|---|---|
| Results hold over 5 random seeds (2020-2024): ItemKNN vs UserKNN, EASE ~ SLIM > ItemKNN, Pop baseline, popularity and coverage of the top-10 | `python hw1/experiments/analysis/runner.py <seed> hw1/results/analysis/seeds dataset` for each seed, then `python hw1/experiments/analysis/aggregate.py hw1/results hw1/results/analysis/seeds` | `seeds/seed_*.csv`, `summary.txt` |
| UserKNN with k = 1: 89% of users get ten tied scores | `python hw1/experiments/analysis/knn_ties.py` | `knn_ties.txt` |
| BPR embedding norms shrink with reg_lambda and collapse at 0.5 | `python hw1/experiments/analysis/bpr_inspect.py` | `bpr_inspect.txt` |
| With reg_lambda = 0.01, NDCG@10 grows with embedding size | `runner.py` (above), `python hw1/experiments/analysis/plot_supplementary.py hw1/results/analysis/seeds hw1/results/analysis/q2_bpr_5seeds.png` | `q2_bpr_5seeds.png` |
| Figures as embedded in the report | `python hw1/experiments/analysis/doc_figures.py hw1/results hw1/results/analysis/figs` | `figs/*.png` |

`runner.py` patches RecBole's evaluation collector in-process: it calls `.cpu()` on every collected
value, which crashes the ItemCoverage / AveragePopularity / GiniIndex metrics (they store an int or a
Counter). Accuracy metrics are unaffected.
