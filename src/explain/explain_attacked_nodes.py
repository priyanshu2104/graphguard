"""
Compares GNNExplainer output BEFORE and AFTER the camouflage attack on the
same sample nodes, using the frozen GCN-Skip model. A node whose explanation
shifts sharply after a few added edges shows the attack changing what the
model looks at, not just tipping a borderline score.
"""
import torch
from torch_geometric.explain import Explainer, GNNExplainer

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.attacks.evaluate_attack import load_frozen_model
from src.attacks.target_selection import get_attack_targets
from src.attacks.camouflage_attack import get_licit_pool, apply_camouflage_attack

ATTACK_BUDGET = 3  # mid-range budget from Week 9


def build_explainer(model):
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(mode="multiclass_classification", task_level="node", return_type="raw"),
    )


def summarize_explanation(node_idx, explanation):
    feat_importance = explanation.node_mask[node_idx].abs()
    top_feats = feat_importance.topk(5)
    return {
        "top_features": top_feats.indices.tolist(),
        # raw sum, NOT int(): the mask holds fractional scores, and int()
        # truncates anything below 1.0 to 0 (the Week 9 display bug)
        "edge_mask_sum": round(explanation.edge_mask.sum().item(), 4),
    }


def main():
    data = load_graph()
    train_mask, test_mask, labeled_mask = get_split_masks()
    data = standardize_features(data, train_mask)

    model = load_frozen_model(data)
    target_node_indices, _, _ = get_attack_targets(model, data, test_mask)
    sample_nodes = target_node_indices[:3].tolist()

    licit_pool = get_licit_pool(data, train_mask)
    attacked_data = apply_camouflage_attack(data, target_node_indices, licit_pool, budget=ATTACK_BUDGET)

    with torch.no_grad():
        preds_attacked = model(attacked_data.x, attacked_data.edge_index).argmax(dim=1)

    explainer = build_explainer(model)

    for node_idx in sample_nodes:
        print(f"\n{'='*50}\nNode {node_idx}\n{'='*50}")

        still_flagged = preds_attacked[node_idx].item() == 1
        print(f"Prediction after attack (budget={ATTACK_BUDGET}): "
              f"{'still ILLICIT' if still_flagged else 'FLIPPED to LICIT (evaded)'}")

        exp_clean = explainer(data.x, data.edge_index, index=node_idx)
        exp_attacked = explainer(attacked_data.x, attacked_data.edge_index, index=node_idx)

        s_clean = summarize_explanation(node_idx, exp_clean)
        s_att = summarize_explanation(node_idx, exp_attacked)

        print(f"BEFORE — top features: {s_clean['top_features']}, edge mask sum: {s_clean['edge_mask_sum']}")
        print(f"AFTER  — top features: {s_att['top_features']}, edge mask sum: {s_att['edge_mask_sum']}")

        overlap = set(s_clean["top_features"]) & set(s_att["top_features"])
        print(f"Feature overlap before/after: {len(overlap)}/5")


if __name__ == "__main__":
    main()