"""Why does KNN with tiny k do badly? Inspect the score matrix ItemKNN builds.

For each test user (history = train + valid items, as in RecBole's test evaluation):
  - candidates: unseen items with a non-zero score
  - tie@10: the 10th and 11th best scores are equal, so top-10 membership is
            decided by tie-breaking (item id order), not by the model
usage (from the repo root): python knn_ties.py
"""
import numpy as np
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import ItemKNN
from recbole.utils import init_seed

config = Config(model="ItemKNN", dataset="ml-100k", config_dict={"state": "WARNING"})
init_seed(config["seed"], config["reproducibility"])
dataset = create_dataset(config)
train_data, valid_data, test_data = data_preparation(config, dataset)

test_users = [u for u in range(1, dataset.user_num) if test_data.uid2positive_item[u] is not None]
print(f"test users: {len(test_users)}")
print(f"{'method':6} {'k':>4} | {'median cand':>11} {'<10 cand':>9} {'tie@10':>7} {'all 10 tied':>11}")

for method in ["item", "user"]:
    for k in [1, 5, 10, 100, 500]:
        config["knn_method"], config["k"] = method, k
        pred = ItemKNN(config, train_data._dataset).pred_mat.tocsr()
        n_cand, few, tie, all_tied = [], 0, 0, 0
        for u in test_users:
            scores = pred[u].toarray().ravel()
            scores[0] = -np.inf  # padding item
            scores[test_data.uid2history_item[u].numpy()] = -np.inf
            cand = int((scores > 0).sum())
            n_cand.append(cand)
            top = np.sort(scores)[::-1][:11]
            few += cand < 10
            tie += top[9] == top[10]
            all_tied += top[0] == top[9]
        n = len(test_users)
        print(f"{method:6} {k:>4} | {np.median(n_cand):>11.0f} {few / n:>9.1%} {tie / n:>7.1%} {all_tied / n:>11.1%}")
