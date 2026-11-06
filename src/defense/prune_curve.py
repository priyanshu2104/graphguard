"""
Pruning curve for GCN-Skip against the time-consistent adaptive attacker, over 3 training seeds.
For each pruning level reports clean F1, clean false positives, and the worst case over budgets
(b = 3, 5) and partner-selection variants (M = 10, 25, 50, 100, 200).
The whole curve is reported; the selection rule at the end is applied to TEST results (there is no
validation split), so say so in the paper.
"""
import numpy as np
import torch.nn.functional as F

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.attacks.target_selection import get_attack_targets
from src.attacks.adaptive_attack import apply_adaptive_attack
from src.defense.eval_utils import predict, metrics
from src.defense.similarity_prune import prune, edge_sim
from src.defense.seed_check import get_model

SEEDS = [42, 0, 1]
QS = [0.0, 0.05, 0.10, 0.25, 0.50, 0.75]
BUDGETS, POOLS = [3, 5], [10, 25, 50, 100, 200]


def main():
    data = load_graph()
    train_mask, test_mask, _ = get_split_masks()
    data = standardize_features(data, train_mask)
    licit_all = (data.y == 0).nonzero(as_tuple=True)[0]
    test_idx = test_mask.nonzero(as_tuple=True)[0]
    y = data.y[test_idx].numpy()
    real = edge_sim(F.normalize(data.x, dim=1), data.edge_index).numpy()

    res = {q: {"clean": [], "clean_fp": [], "worst": [], "worst_fp": []} for q in QS}
    for seed in SEEDS:
        model = get_model("GCN-Skip", seed, data, train_mask, test_mask)
        targets, _, _ = get_attack_targets(model, data, test_mask)

        def score(graph, thr):
            pruned, _ = prune(graph, thr)
            p, pr = predict(model, pruned)
            return metrics(y, p[test_idx].numpy(), pr[test_idx].numpy())

        for q in QS:
            thr = float(np.quantile(real, q)) if q > 0 else -2.0
            c = score(data, thr)
            worst = None
            for b in BUDGETS:
                for m_pool in POOLS:
                    att = apply_adaptive_attack(data, model, targets, licit_all, thr=thr, budget=b,
                                                pool_size=m_pool, seed=0, same_time=True)
                    r = score(att, thr)
                    if worst is None or r["f1"] < worst["f1"]:
                        worst = r
            res[q]["clean"].append(c["f1"]); res[q]["clean_fp"].append(c["fp"])
            res[q]["worst"].append(worst["f1"]); res[q]["worst_fp"].append(worst["fp"])
            print(f"seed {seed} q={q:.2f}: clean F1 {c['f1']:.3f} (FP {c['fp']}) | "
                  f"worst adaptive F1 {worst['f1']:.3f} (FP {worst['fp']})", flush=True)

    print("\nMean ± sample std over training seeds")
    for q in QS:
        r = res[q]
        print(f"q={q:.2f}: clean F1 {np.mean(r['clean']):.3f} ± {np.std(r['clean'], ddof=1):.3f} | "
              f"clean FP {np.mean(r['clean_fp']):.0f} | "
              f"worst adaptive F1 {np.mean(r['worst']):.3f} ± {np.std(r['worst'], ddof=1):.3f}")

    base = np.mean(res[0.0]["clean"])
    eligible = [q for q in QS if np.mean(res[q]["clean"]) >= base - 0.01]
    best = max(eligible, key=lambda q: np.mean(res[q]["worst"]))
    print(f"\nRule: highest mean worst-case F1 among levels whose clean F1 is within 0.01 of "
          f"unpruned -> q = {best:.2f}")


if __name__ == "__main__":
    main()