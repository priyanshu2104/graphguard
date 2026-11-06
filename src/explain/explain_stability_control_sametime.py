"""
Control for the explanation-shift result. The first stability run used
explanation_type="model", which explains the model's PREDICTED class. For an
evaded node that class changes from illicit to licit, so its before/after
explanations targeted different classes. Here the explainer always targets
the ILLICIT class ("phenomenon" mode). It also measures the explainer's
run-to-run floor on the ATTACKED graph, not just the clean one.
Uses the same 150 sampled nodes as explain_stability.py.
"""
import numpy as np
import torch
from src.attacks.sametime_attack import apply_random_sametime
from scipy.stats import mannwhitneyu, spearmanr
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.attacks.evaluate_attack import load_frozen_model
from src.attacks.target_selection import get_attack_targets
from src.attacks.camouflage_attack import get_licit_pool, apply_camouflage_attack

N_NODES, BUDGET, HOPS = 150, 3, 3


def build_explainer(model):
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type="phenomenon",  # explain a FIXED target class
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="multiclass_classification", task_level="node", return_type="raw"),
    )


def explain(explainer, x, edge_index, idx, seed, n):
    subset, ei, mapping, _ = k_hop_subgraph(idx, HOPS, edge_index,
                                            relabel_nodes=True, num_nodes=n)
    local = int(mapping)
    target = torch.ones(len(subset), dtype=torch.long)  # class 1 = illicit
    torch.manual_seed(seed)  # GNNExplainer initialises masks randomly
    exp = explainer(x[subset], ei, target=target, index=local)
    return exp.node_mask[local].abs().detach().numpy(), len(subset)


def rho(a, b):
    return spearmanr(a, b)[0]


def drop_nan(v):
    v = np.asarray(v, dtype=float)
    return v[~np.isnan(v)]


def main():
    data = load_graph()
    train_mask, test_mask, _ = get_split_masks()
    data = standardize_features(data, train_mask)
    model = load_frozen_model(data)
    targets, _, _ = get_attack_targets(model, data, test_mask)

    # same generator seed as explain_stability.py -> the same 150 nodes
    g = torch.Generator().manual_seed(0)
    sample = targets[torch.randperm(len(targets), generator=g)[:N_NODES]].tolist()

    attacked = apply_random_sametime(data, targets, budget=BUDGET)
    with torch.no_grad():
        evaded = (model(attacked.x, attacked.edge_index).argmax(1) != 1).numpy()

    explainer = build_explainer(model)
    n, rows = data.num_nodes, []
    for i, idx in enumerate(sample, 1):
        a0, size_clean = explain(explainer, data.x, data.edge_index, idx, 0, n)
        a1, _ = explain(explainer, data.x, data.edge_index, idx, 1, n)
        b0, _ = explain(explainer, attacked.x, attacked.edge_index, idx, 0, n)
        b1, _ = explain(explainer, attacked.x, attacked.edge_index, idx, 1, n)
        rows.append((evaded[idx], size_clean, rho(a0, a1), rho(b0, b1), rho(a0, b0)))
        if i % 25 == 0:
            print(f"[{i}/{N_NODES}]", flush=True)

    # columns: evaded, pre-attack subgraph size, clean floor, attacked floor, clean-vs-attacked
    r = np.array(rows, dtype=float)
    ev = r[:, 0] == 1

    print(f"\nExplainer run-to-run floor, clean graph:    rho {np.nanmean(r[:, 2]):.2f}")
    print(f"Explainer run-to-run floor, attacked graph: evaded {np.nanmean(r[ev, 3]):.2f} "
          f"| not evaded {np.nanmean(r[~ev, 3]):.2f}")
    print(f"Clean vs attacked, FIXED target (illicit): evaded (n={ev.sum()}) "
          f"rho {np.nanmean(r[ev, 4]):.2f} | not evaded (n={(~ev).sum()}) rho {np.nanmean(r[~ev, 4]):.2f}")

    x, y = drop_nan(r[ev, 4]), drop_nan(r[~ev, 4])
    u, p = mannwhitneyu(x, y, alternative="less")
    print(f"Mann-Whitney (evaded rho < not-evaded rho): U={u:.0f}, p={p:.2g}, "
          f"P(evaded lower) = {1 - u / (len(x) * len(y)):.2f}")

    small = r[:, 1] <= 4
    x, y = drop_nan(r[ev & small, 4]), drop_nan(r[~ev & small, 4])
    if len(x) > 5 and len(y) > 5:
        u, p = mannwhitneyu(x, y, alternative="less")
        print(f"Only nodes with pre-attack subgraph <= 4 nodes "
              f"(evaded n={len(x)}, not evaded n={len(y)}): U={u:.0f}, p={p:.2g}")


if __name__ == "__main__":
    main()