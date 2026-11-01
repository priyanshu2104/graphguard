"""
Does Week 10's picture (GCN-Skip fragile, GAT-Skip resistant) survive a
white-box, diversified adaptive attacker? Undefended models, no pruning
constraint, averaged over 3 sampling seeds.
"""
import numpy as np
import torch

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.attacks.target_selection import get_attack_targets
from src.attacks.camouflage_attack import get_licit_pool
from src.attacks.adaptive_attack import apply_adaptive_attack
from src.attacks.train_and_cache import get_or_train_model
from src.defense.eval_utils import predict, metrics

SEEDS, BUDGETS, POOLS = [0, 1, 2], [1, 3, 5], [200, 2000]


def main():
    data = load_graph()
    train_mask, test_mask, _ = get_split_masks()
    data = standardize_features(data, train_mask)
    licit_pool = get_licit_pool(data, train_mask)
    test_idx = test_mask.nonzero(as_tuple=True)[0]
    y = data.y[test_idx].numpy()

    for arch in ["GCN-Skip", "GraphSAGE", "GAT-Skip"]:
        model = get_or_train_model(arch, data, train_mask, test_mask)
        targets, _, _ = get_attack_targets(model, data, test_mask)
        p, pr = predict(model, data)
        c = metrics(y, p[test_idx].numpy(), pr[test_idx].numpy())
        print(f"\n{arch}: clean F1 {c['f1']:.3f} recall {c['recall']:.3f}")
        for b in BUDGETS:
            for m in POOLS:
                ev, f1, rc = [], [], []
                for s in SEEDS:
                    att = apply_adaptive_attack(data, model, targets, licit_pool, thr=-2.0,
                                                budget=b, pool_size=m, seed=s)
                    p, pr = predict(model, att)
                    r = metrics(y, p[test_idx].numpy(), pr[test_idx].numpy())
                    ev.append((p[targets] != 1).float().mean().item())
                    f1.append(r["f1"]); rc.append(r["recall"])
                print(f"  b={b} M={m:<4} | evasion {np.mean(ev):.1%} ± {np.std(ev):.1%} | "
                      f"F1 {np.mean(f1):.3f} | recall {np.mean(rc):.3f}", flush=True)


if __name__ == "__main__":
    main()