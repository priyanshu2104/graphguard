"""
Same experiment as explain_stability.py, but each explanation runs on the
node's 3-hop subgraph instead of the full 203k-node graph. Verifies the
subgraph reproduces the full-graph logits before trusting it.
"""
import time
import numpy as np
import torch
from scipy.stats import spearmanr
from torch_geometric.utils import k_hop_subgraph

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.attacks.evaluate_attack import load_frozen_model
from src.attacks.target_selection import get_attack_targets
from src.attacks.camouflage_attack import get_licit_pool, apply_camouflage_attack
from src.explain.explain_attacked_nodes import build_explainer

N_NODES, BUDGET, HOPS = 150, 3, 3


def sub_explain(explainer, model, x, edge_index, idx, seed, n):
    subset, ei, mapping, _ = k_hop_subgraph(idx, HOPS, edge_index,
                                            relabel_nodes=True, num_nodes=n)
    xs, local = x[subset], int(mapping)
    with torch.no_grad():
        ok = torch.allclose(model(xs, ei)[local], model(x, edge_index)[idx], atol=1e-4)
    torch.manual_seed(seed)  # GNNExplainer initialises masks randomly
    exp = explainer(xs, ei, index=local)
    return exp.node_mask[local].abs().detach().numpy(), ok, len(subset)


def overlap(a, b):
    return len(set(np.argsort(-a)[:5]) & set(np.argsort(-b)[:5]))


def main():
    data = load_graph()
    train_mask, test_mask, _ = get_split_masks()
    data = standardize_features(data, train_mask)
    model = load_frozen_model(data)
    targets, _, _ = get_attack_targets(model, data, test_mask)

    g = torch.Generator().manual_seed(0)
    sample = targets[torch.randperm(len(targets), generator=g)[:N_NODES]].tolist()

    attacked = apply_camouflage_attack(data, targets, get_licit_pool(data, train_mask), budget=BUDGET)
    with torch.no_grad():
        evaded = (model(attacked.x, attacked.edge_index).argmax(1) != 1).numpy()

    explainer = build_explainer(model)
    n, rows, bad = data.num_nodes, [], 0
    for i, idx in enumerate(sample, 1):
        t0 = time.time()
        a, ok1, s1 = sub_explain(explainer, model, data.x, data.edge_index, idx, 0, n)
        b, ok2, _ = sub_explain(explainer, model, data.x, data.edge_index, idx, 1, n)
        c, ok3, s3 = sub_explain(explainer, model, attacked.x, attacked.edge_index, idx, 0, n)
        bad += (not (ok1 and ok2 and ok3))
        rows.append((evaded[idx], overlap(a, b), overlap(a, c),
                     spearmanr(a, b)[0], spearmanr(a, c)[0]))
        print(f"[{i}/{N_NODES}] node {idx} | evaded={bool(evaded[idx])} | "
              f"subgraph {s1}->{s3} nodes | {time.time()-t0:.1f}s", flush=True)

    r = np.array(rows, dtype=float)
    if bad:
        print(f"\nWARNING: {bad} node(s) failed the subgraph-equivalence check; treat results with caution.")
    print(f"\nNoise floor (clean vs clean): overlap {r[:,1].mean():.2f}/5, rho {r[:,3].mean():.2f}")
    for label, m in [("evaded", r[:, 0] == 1), ("not evaded", r[:, 0] == 0)]:
        if m.sum():
            print(f"{label} (n={int(m.sum())}): clean-vs-attacked overlap "
                  f"{r[m,2].mean():.2f}/5, rho {r[m,4].mean():.2f}")

    from scipy.stats import mannwhitneyu
    ev, ne = r[r[:, 0] == 1, 4], r[r[:, 0] == 0, 4]
    if len(ev) > 1 and len(ne) > 1:
        u, p = mannwhitneyu(ev, ne, alternative="less")
        print(f"Mann-Whitney (evaded rho < not-evaded rho): U={u:.0f}, p={p:.4f}")


if __name__ == "__main__":
    main()