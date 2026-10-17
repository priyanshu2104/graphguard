"""Repeats the camouflage attack over several sampling seeds (inference only)."""
import numpy as np
import torch

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.attacks.target_selection import get_attack_targets
from src.attacks.camouflage_attack import get_licit_pool, apply_camouflage_attack
from src.attacks.train_and_cache import get_or_train_model

SEEDS = [0, 1, 2, 3, 4]
BUDGETS = [1, 3, 5]


@torch.no_grad()
def evasion_rate(model, data, targets, licit_pool, budget, seed):
    attacked = apply_camouflage_attack(data, targets, licit_pool, budget=budget, seed=seed)
    preds = model(attacked.x, attacked.edge_index).argmax(dim=1)
    return 1 - (preds[targets] == 1).float().mean().item()


def main():
    data = load_graph()
    train_mask, test_mask, _ = get_split_masks()
    data = standardize_features(data, train_mask)
    licit_pool = get_licit_pool(data, train_mask)

    for name in ["GCN-Skip", "GraphSAGE", "GAT-Skip"]:
        model = get_or_train_model(name, data, train_mask, test_mask)
        targets, _, _ = get_attack_targets(model, data, test_mask)
        for b in BUDGETS:
            vals = [evasion_rate(model, data, targets, licit_pool, b, s) for s in SEEDS]
            print(f"{name} | b={b} | evasion {np.mean(vals):.2%} ± {np.std(vals):.2%}")


if __name__ == "__main__":
    main()