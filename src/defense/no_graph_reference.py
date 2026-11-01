"""
Reference: GCN-Skip trained and evaluated with an EMPTY edge set. Immune to
edge attacks by construction, so this is the bar a defence that merely
ignores the graph would reach.
"""
import copy
import torch

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.attacks.model_registry import build_model
from src.models.train_utils import run_training
from src.utils.metrics import evaluate_illicit, log_results


def main():
    data = load_graph()
    train_mask, test_mask, _ = get_split_masks()
    data = standardize_features(data, train_mask)

    no_graph = copy.copy(data)
    no_graph.edge_index = torch.empty((2, 0), dtype=torch.long)

    torch.manual_seed(42)
    model = build_model("GCN-Skip", in_channels=data.num_node_features)
    preds, probs, y_true = run_training(model, no_graph, train_mask, test_mask, epochs=100, lr=0.01)

    results = evaluate_illicit(y_true, preds, probs, model_name="GCN-Skip [no graph]")
    log_results(results, split_name="defense_reference")


if __name__ == "__main__":
    main()