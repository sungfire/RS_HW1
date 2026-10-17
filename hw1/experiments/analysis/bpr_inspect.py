"""How does reg_lambda change what BPR learns? (seed 2020, embedding_size 64)

Prints, for the best checkpoint (the one RecBole evaluates on test):
  best epoch, mean L2 norm of user / item embeddings, spread of the scores
  a user gives to all items, and test NDCG@10.
usage (from the repo root): python bpr_inspect.py
"""
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import BPR
from recbole.utils import get_trainer, init_logger, init_seed

print(f"{'reg':>5} | {'best epoch':>10} | {'|user emb|':>10} {'|item emb|':>10} | {'score std/user':>14} | {'NDCG@10':>7}")
for reg in [0.0, 0.01, 0.1, 0.5]:
    config = Config(model="BPR", dataset="ml-100k",
                    config_dict={"reg_lambda": reg, "state": "WARNING", "show_progress": False,
                                 "checkpoint_dir": f"saved_inspect_{reg}"})
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = BPR(config, train_data._dataset)
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    trainer.fit(train_data, valid_data, verbose=False, show_progress=False)
    result = trainer.evaluate(test_data, load_best_model=True, show_progress=False)
    best_epoch = torch.load(trainer.saved_model_file, weights_only=False)["epoch"]

    with torch.no_grad():
        u = model.user_embedding.weight[1:]
        i = model.item_embedding.weight[1:]
        scores = u @ i.T
    print(f"{reg:>5} | {best_epoch:>10} | {u.norm(dim=1).mean():>10.3f} {i.norm(dim=1).mean():>10.3f} | "
          f"{scores.std(dim=1).mean():>14.4f} | {result['ndcg@10']:>7.4f}")
