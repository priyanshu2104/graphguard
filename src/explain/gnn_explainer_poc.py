"""
Proof of concept: run GNNExplainer on a few correctly-classified illicit
nodes using the frozen GCN-Skip model. Confirms the explainability
pipeline works end-to-end before Week 13's full integration -- this is
deliberately minimal, just 3 nodes, printed output only.
"""
import torch
from torch_geometric.explain import Explainer, GNNExplainer

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.attacks.evaluate_attack import load_frozen_model
from src.attacks.target_selection import get_attack_targets


def main():
    data = load_graph()
    train_mask, test_mask, labeled_mask = get_split_masks()
    data = standardize_features(data, train_mask)

    model = load_frozen_model(data)

    target_node_indices, _, _ = get_attack_targets(model, data, test_mask)
    sample_nodes = target_node_indices[:3]  # just 3 nodes for this proof of concept

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        ),
    )

    for node_idx in sample_nodes.tolist():
        explanation = explainer(data.x, data.edge_index, index=node_idx)

        feat_importance = explanation.node_mask[node_idx].abs()
        top_feats = feat_importance.topk(5)

        print(f"\n=== Node {node_idx} ===")
        print(f"Top 5 contributing feature indices: {top_feats.indices.tolist()}")
        print(f"Importance scores: {[round(v, 4) for v in top_feats.values.tolist()]}")

        edge_mask_sum = explanation.edge_mask.sum().item()
        print(f"Edge mask sum (raw): {edge_mask_sum:.4f}")

        top_edge_scores, top_edge_idx = explanation.edge_mask.topk(5)
        top_edges = data.edge_index[:, top_edge_idx]
        print("Top 5 edges (source -> target, importance):")
        for i in range(len(top_edge_idx)):
            src, dst = top_edges[0, i].item(), top_edges[1, i].item()
            print(f"  {src} -> {dst}  (importance: {top_edge_scores[i].item():.4f})")


if __name__ == "__main__":
    main()