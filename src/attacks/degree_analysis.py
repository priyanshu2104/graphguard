"""Evasion rate by the target's ORIGINAL in-degree (budget 3, 5 sampling seeds)."""
import numpy as np
import torch
from torch_geometric.utils import degree

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.attacks.target_selection import get_attack_targets
from src.attacks.camouflage_attack import get_licit_pool, apply_camouflage_attack
from src.attacks.train_and_cache import get_or_train_model

SEEDS, BUDGET = [0, 1, 2, 3, 4], 3
BINS = [(0, 0), (1, 2), (3, 5), (6, 10**9)]


def main():
    data = load_graph()
    train_mask, test_mask, _ = get_split_masks()
    data = standardize_features(data, train_mask)
    licit_pool = get_licit_pool(data, train_mask)
    indeg = degree(data.edge_index[1], num_nodes=data.num_nodes)

    for name in ["GCN-Skip", "GraphSAGE", "GAT-Skip"]:
        model = get_or_train_model(name, data, train_mask, test_mask)
        targets, _, _ = get_attack_targets(model, data, test_mask)
        d = indeg[targets].numpy()
        ev = np.zeros(len(targets))
        for s in SEEDS:
            att = apply_camouflage_attack(data, targets, licit_pool, budget=BUDGET, seed=s)
            with torch.no_grad():
                preds = model(att.x, att.edge_index).argmax(1)
            ev += (preds[targets] != 1).float().numpy()
        ev /= len(SEEDS)
        for lo, hi in BINS:
            m = (d >= lo) & (d <= hi)
            if m.sum():
                print(f"{name} | in-degree {lo}-{hi if hi < 10**9 else '+'} | n={m.sum()} | evasion {ev[m].mean():.1%}")


if __name__ == "__main__":
    main()