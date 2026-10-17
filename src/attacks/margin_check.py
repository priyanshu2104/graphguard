"""Is GAT-Skip's robustness architecture, or just large clean margins?"""
import numpy as np
import torch

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.attacks.target_selection import get_attack_targets
from src.attacks.camouflage_attack import get_licit_pool, apply_camouflage_attack
from src.attacks.train_and_cache import get_or_train_model

SEEDS, BUDGETS = [0, 1, 2], [1, 5, 10, 20]


def main():
    data = load_graph()
    train_mask, test_mask, _ = get_split_masks()
    data = standardize_features(data, train_mask)
    licit_pool = get_licit_pool(data, train_mask)

    for name in ["GCN-Skip", "GraphSAGE", "GAT-Skip"]:
        model = get_or_train_model(name, data, train_mask, test_mask)
        targets, _, _ = get_attack_targets(model, data, test_mask)
        with torch.no_grad():
            out = model(data.x, data.edge_index)[targets]
        margin = (out[:, 1] - out[:, 0]).numpy()  # logit margin, illicit minus licit
        q = np.percentile(margin, [25, 50, 75])
        print(f"{name} | clean margin on targets: 25%={q[0]:.2f} median={q[1]:.2f} 75%={q[2]:.2f}")
        for b in BUDGETS:
            ev = []
            for s in SEEDS:
                att = apply_camouflage_attack(data, targets, licit_pool, budget=b, seed=s)
                with torch.no_grad():
                    preds = model(att.x, att.edge_index).argmax(1)
                ev.append((preds[targets] != 1).float().mean().item())
            print(f"{name} | b={b} | evasion {np.mean(ev):.2%} ± {np.std(ev):.2%}")


if __name__ == "__main__":
    main()