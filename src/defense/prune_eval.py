"""
Inference-time similarity pruning of GCN-Skip, evaluated against random
camouflage (5 sampling seeds), nearest-licit, and a defence-aware attacker
that knows the pruning threshold and the model. Test-set metrics.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.attacks.target_selection import get_attack_targets
from src.attacks.camouflage_attack import get_licit_pool, apply_camouflage_attack
from src.attacks.nearest_attack import apply_nearest_attack
from src.attacks.adaptive_attack import apply_adaptive_attack
from src.attacks.train_and_cache import get_or_train_model
from src.defense.eval_utils import predict, metrics
from src.defense.similarity_prune import prune, edge_sim

OUT_PATH = Path(__file__).resolve().parents[2] / "results" / "prune_eval.csv"
QS, BUDGETS, SEEDS = [0.0, 0.10, 0.25, 0.50], [1, 3, 5], [0, 1, 2, 3, 4]


def main():
    data = load_graph()
    train_mask, test_mask, _ = get_split_masks()
    data = standardize_features(data, train_mask)
    model = get_or_train_model("GCN-Skip", data, train_mask, test_mask)
    targets, _, _ = get_attack_targets(model, data, test_mask)
    licit_pool = get_licit_pool(data, train_mask)

    test_idx = test_mask.nonzero(as_tuple=True)[0]
    y = data.y[test_idx].numpy()
    real = edge_sim(F.normalize(data.x, dim=1), data.edge_index).numpy()

    def score(graph, thr):
        pruned, _ = prune(graph, thr)
        p, pr = predict(model, pruned)
        return metrics(y, p[test_idx].numpy(), pr[test_idx].numpy())

    rows = []
    for q in QS:
        thr = float(np.quantile(real, q)) if q > 0 else -2.0  # -2: keep every edge
        add = lambda attack, b, m: rows.append({"q": q, "thr": thr, "attack": attack, "budget": b, **m})

        add("none", 0, score(data, thr))
        for b in BUDGETS:
            ms = [score(apply_camouflage_attack(data, targets, licit_pool, budget=b, seed=s), thr)
                  for s in SEEDS]
            add("random", b, {k: float(np.mean([m[k] for m in ms])) for k in ms[0]})
            add("nearest", b, score(apply_nearest_attack(data, targets, licit_pool, budget=b), thr))
            add("adaptive", b, score(apply_adaptive_attack(data, model, targets, licit_pool,
                                                           thr=thr, budget=b), thr))
        print(f"q={q:.2f} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(df.round(3)[["q", "attack", "budget", "f1", "recall", "fp", "auc_pr"]].to_string(index=False))


if __name__ == "__main__":
    main()