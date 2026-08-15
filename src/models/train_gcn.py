"""
Trains GCN on the Elliptic graph using the temporal split, evaluates on the
illicit class, and logs results to results/results_log.csv.
"""
import torch

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.utils.metrics import evaluate_illicit, log_results
from src.models.gnn_models import GCNSkip
from src.models.train_utils import run_training


def main():
    data = load_graph()
    train_mask, test_mask, labeled_mask = get_split_masks()
    data = standardize_features(data, train_mask)

    torch.manual_seed(42)
    model = GCNSkip(in_channels=data.num_node_features, hidden_channels=64, out_channels=2)

    print(f"Training GCN on {train_mask.sum().item()} nodes, "
          f"testing on {test_mask.sum().item()} nodes\n")

    preds, probs, y_true = run_training(
        model, data, train_mask, test_mask, epochs=100, lr=0.01
    )

    results = evaluate_illicit(y_true, preds, probs, model_name="GCN-Skip")
    log_results(results)

    torch.save(model.state_dict(), "results/gcn_weights.pt")
    print("\nModel weights saved to results/gcn_weights.pt")


if __name__ == "__main__":
    main()