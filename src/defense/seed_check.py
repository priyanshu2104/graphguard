"""
Training-seed check. The architecture differences rest on ONE trained model
each (seed 42). Train two more per architecture and repeat the diversified
adaptive attack (b = 1, 3, 5; M = 200 and 2000; strongest variant per budget).
"""
import numpy as np
import torch

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.attacks.model_registry import MODEL_CONFIGS, build_model
from src.attacks.train_and_cache import get_or_train_model
from src.attacks.target_selection import get_attack_targets
from src.attacks.camouflage_attack import get_licit_pool
from src.attacks.adaptive_attack import apply_adaptive_attack
from src.models.train_utils import run_training
from src.defense.eval_utils import predict, metrics

SEEDS, BUDGETS, POOLS = [42, 0, 1], [1, 3, 5], [200, 2000]


def get_model(arch, seed, data, train_mask, test_mask):
    if seed == 42:  # the cached model used everywhere else
        return get_or_train_model(arch, data, train_mask, test_mask)
    torch.manual_seed(seed)
    model = build_model(arch, data.num_node_features)
    run_training(model, data, train_mask, test_mask, epochs=100, lr=MODEL_CONFIGS[arch]["lr"])
    model.eval()
    return model


def main():
    data = load_graph()
    train_mask, test_mask, _ = get_split_masks()
    data = standardize_features(data, train_mask)
    licit_pool = get_licit_pool(data, train_mask)
    test_idx = test_mask.nonzero(as_tuple=True)[0]
    y = data.y[test_idx].numpy()

    summary = {}
    for arch in ["GCN-Skip", "GraphSAGE", "GAT-Skip"]:
        clean, strongest = [], {b: [] for b in BUDGETS}
        for seed in SEEDS:
            model = get_model(arch, seed, data, train_mask, test_mask)
            targets, _, _ = get_attack_targets(model, data, test_mask)
            p, pr = predict(model, data)
            clean.append(metrics(y, p[test_idx].numpy(), pr[test_idx].numpy())["f1"])
            line = f"{arch} seed {seed}: clean F1 {clean[-1]:.3f} | strongest-variant evasion"
            for b in BUDGETS:
                evs = []
                for m in POOLS:
                    att = apply_adaptive_attack(data, model, targets, licit_pool, thr=-2.0,
                                                budget=b, pool_size=m, seed=0)
                    pa, _ = predict(model, att)
                    evs.append((pa[targets] != 1).float().mean().item())
                strongest[b].append(max(evs))
                line += f" b={b}: {max(evs):.1%}"
            print(line, flush=True)
        summary[arch] = (clean, strongest)

    print("\nMean ± std over training seeds")
    for arch, (clean, strongest) in summary.items():
        row = f"{arch:<10} clean F1 {np.mean(clean):.3f} ± {np.std(clean):.3f}"
        for b in BUDGETS:
            row += f" | b={b} evasion {np.mean(strongest[b]):.1%} ± {np.std(strongest[b]):.1%}"
        print(row)


if __name__ == "__main__":
    main()