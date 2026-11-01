"""
Re-evaluates the attacks with TIME-CONSISTENT partners, plus a time-consistency
filter that drops edges between different time steps.
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.attacks.target_selection import get_attack_targets
from src.attacks.camouflage_attack import get_licit_pool
from src.attacks.sametime_attack import apply_random_sametime
from src.attacks.adaptive_attack import apply_adaptive_attack
from src.attacks.train_and_cache import get_or_train_model
from src.defense.eval_utils import predict, metrics
from src.defense.similarity_prune import prune, edge_sim

BUDGETS, POOLS, SEEDS = [1, 3, 5], [10, 50, 200], [0, 1, 2, 3, 4]


def drop_cross_time(graph):
    ei = graph.edge_index
    keep = graph.time_step[ei[0]] == graph.time_step[ei[1]]
    out = copy.copy(graph)
    out.edge_index = ei[:, keep]
    return out


def main():
    data = load_graph()
    train_mask, test_mask, _ = get_split_masks()
    data = standardize_features(data, train_mask)
    ei, ts = data.edge_index, data.time_step
    print(f"1) real edges across time steps: {(ts[ei[0]] != ts[ei[1]]).float().mean().item():.4%}")

    train_licit = get_licit_pool(data, train_mask)
    licit_all = (data.y == 0).nonzero(as_tuple=True)[0]
    test_idx = test_mask.nonzero(as_tuple=True)[0]
    y = data.y[test_idx].numpy()
    real = edge_sim(F.normalize(data.x, dim=1), ei).numpy()

    def evaluate(model, graph, targets, thr=-2.0):
        pruned, _ = prune(graph, thr)
        p, pr = predict(model, pruned)
        m = metrics(y, p[test_idx].numpy(), pr[test_idx].numpy())
        return m["f1"], m["recall"], (p[targets] != 1).float().mean().item()

    # 2) the earlier (cross-time) attack against a time-consistency filter
    model = get_or_train_model("GCN-Skip", data, train_mask, test_mask)
    targets, _, _ = get_attack_targets(model, data, test_mask)
    old = apply_adaptive_attack(data, model, targets, train_licit, thr=-2.0,
                                budget=3, pool_size=2000, seed=0)
    f_old, f_filt = evaluate(model, old, targets), evaluate(model, drop_cross_time(old), targets)
    print(f"2) GCN-Skip, earlier cross-time adaptive attack b=3: F1 {f_old[0]:.3f} evasion {f_old[2]:.1%} | "
          f"after dropping cross-time edges: F1 {f_filt[0]:.3f} evasion {f_filt[2]:.1%}")

    # 3) time-consistent attackers against the three undefended architectures
    print("\n3) Time-consistent attacks (undefended)")
    for arch in ["GCN-Skip", "GraphSAGE", "GAT-Skip"]:
        model = get_or_train_model(arch, data, train_mask, test_mask)
        targets, _, _ = get_attack_targets(model, data, test_mask)
        c = evaluate(model, data, targets)
        print(f"{arch}: clean F1 {c[0]:.3f} recall {c[1]:.3f}")
        for b in BUDGETS:
            r = [evaluate(model, apply_random_sametime(data, targets, budget=b, seed=s), targets) for s in SEEDS]
            a = [evaluate(model, apply_adaptive_attack(data, model, targets, licit_all, thr=-2.0, budget=b,
                                                       pool_size=m, seed=0, same_time=True), targets)
                 for m in POOLS]
            best = max(a, key=lambda v: v[2])
            print(f"   b={b} | random: evasion {np.mean([v[2] for v in r]):.1%} F1 {np.mean([v[0] for v in r]):.3f}"
                  f" | adaptive (strongest of M={POOLS}): evasion {best[2]:.1%} F1 {best[0]:.3f} recall {best[1]:.3f}",
                  flush=True)

    # 4) pruning against the time-consistent adaptive attacker (GCN-Skip)
    print("\n4) Pruned GCN-Skip vs time-consistent adaptive attacker (strongest of M)")
    model = get_or_train_model("GCN-Skip", data, train_mask, test_mask)
    targets, _, _ = get_attack_targets(model, data, test_mask)
    for q in [0.0, 0.25, 0.50]:
        thr = float(np.quantile(real, q)) if q > 0 else -2.0
        clean = evaluate(model, data, targets, thr)
        for b in [3, 5]:
            a = [evaluate(model, apply_adaptive_attack(data, model, targets, licit_all, thr=thr, budget=b,
                                                       pool_size=m, seed=0, same_time=True), targets, thr)
                 for m in POOLS]
            worst = min(a, key=lambda v: v[0])
            print(f"   q={q:.2f} b={b}: clean F1 {clean[0]:.3f} | worst adaptive F1 {worst[0]:.3f} "
                  f"recall {worst[1]:.3f}", flush=True)


if __name__ == "__main__":
    main()