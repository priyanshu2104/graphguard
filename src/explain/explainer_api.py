"""
Reusable explanation function for the demo: input a node, get back the prediction and the features
and edges that drove it.

GNNExplainer is stochastic (its run-to-run rank correlation was about 0.84 in our tests), so the
masks are averaged over `n_runs` seeds and a stability score is returned. It runs on the node's
k-hop subgraph, which is checked to reproduce the full-graph prediction.
Feature indices are 0-based columns of the standardised feature matrix; the 'local'/'aggregated'
labels follow the dataset's published layout (93 local, then 72 aggregated), which we have not
verified independently.
"""
import itertools

import numpy as np
import torch
from scipy.stats import spearmanr
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph

N_LOCAL = 93


def feature_group(index):
    return "local" if index < N_LOCAL else "aggregated"


def _explainer(model, target_class, epochs):
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        # None -> explain the model's predicted class; 1 -> always explain the illicit class
        explanation_type="model" if target_class is None else "phenomenon",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="multiclass_classification", task_level="node", return_type="raw"),
    )


def explain_node(model, data, node_idx, hops=3, epochs=100, n_runs=3, top_k=5,
                 target_class=None, check=True):
    """Returns a dict with the prediction, top features, top edges (global node ids) and a stability score."""
    model.eval()
    subset, ei, mapping, _ = k_hop_subgraph(node_idx, hops, data.edge_index,
                                            relabel_nodes=True, num_nodes=data.num_nodes)
    local = int(mapping)
    xs = data.x[subset]

    with torch.no_grad():
        sub_logits = model(xs, ei)[local]
        if check and not torch.allclose(sub_logits, model(data.x, data.edge_index)[node_idx], atol=1e-4):
            raise RuntimeError("The subgraph does not reproduce the full-graph prediction; "
                               "increase `hops`.")
        prob = torch.softmax(sub_logits, dim=0)
    predicted = int(prob.argmax())

    explainer = _explainer(model, target_class, epochs)
    node_masks, edge_masks = [], []
    for seed in range(n_runs):
        torch.manual_seed(seed)
        if target_class is None:
            exp = explainer(xs, ei, index=local)
        else:
            target = torch.full((len(subset),), target_class, dtype=torch.long)
            exp = explainer(xs, ei, target=target, index=local)
        node_masks.append(exp.node_mask[local].abs().detach().numpy())
        edge_masks.append(exp.edge_mask.detach().numpy())

    node_mask = np.mean(node_masks, axis=0)
    edge_mask = np.mean(edge_masks, axis=0)

    stability = None
    if n_runs >= 2:
        pairs = [spearmanr(a, b)[0] for a, b in itertools.combinations(node_masks, 2)]
        stability = float(np.nanmean(pairs))

    src_global = subset[ei[0]].numpy()
    dst_global = subset[ei[1]].numpy()
    top_edges = [(int(src_global[i]), int(dst_global[i]), float(edge_mask[i]))
                 for i in np.argsort(-edge_mask)[:top_k]] if len(edge_mask) else []
    top_features = [(int(i), float(node_mask[i]), feature_group(int(i)))
                    for i in np.argsort(-node_mask)[:top_k]]

    return {
        "node": int(node_idx),
        "predicted_class": predicted,
        "p_illicit": float(prob[1]),
        "explained_class": "predicted" if target_class is None else target_class,
        "subgraph_nodes": int(len(subset)),
        "top_features": top_features,   # (column index, importance, group)
        "top_edges": top_edges,         # (source node, target node, importance)
        "stability": stability,         # mean pairwise Spearman rho between runs; lower = less trustworthy
    }


def main():
    from src.utils.splits import load_graph, get_split_masks, standardize_features
    from src.attacks.evaluate_attack import load_frozen_model
    from src.attacks.target_selection import get_attack_targets

    data = load_graph()
    train_mask, test_mask, _ = get_split_masks()
    data = standardize_features(data, train_mask)
    model = load_frozen_model(data)
    targets, _, _ = get_attack_targets(model, data, test_mask)
    for node in targets[:3].tolist():
        result = explain_node(model, data, node)
        print(f"\nnode {result['node']}: predicted class {result['predicted_class']} "
              f"(p_illicit {result['p_illicit']:.3f}), subgraph of {result['subgraph_nodes']} nodes, "
              f"stability rho {result['stability']:.2f}")
        for idx, score, group in result["top_features"]:
            print(f"   feature {idx:>3} ({group}): {score:.3f}")
        for s, d, score in result["top_edges"]:
            print(f"   edge {s} -> {d}: {score:.3f}")


if __name__ == "__main__":
    main()