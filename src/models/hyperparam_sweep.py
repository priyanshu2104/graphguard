"""
Small hyperparameter sweep across GCN-Skip, GraphSAGE, and GAT-Skip.
Deliberately kept small (a handful of configs per model, not an exhaustive
grid) -- the goal this week is "meaningfully better than the untuned
Week 4 numbers," not a perfect optimum. Every run is logged, so even
"failed" configs are useful data for the paper's methodology section.
"""
import torch
import itertools

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.utils.metrics import evaluate_illicit, log_results
from src.models.gnn_models import GCNSkip, GraphSAGE, GATSkip
from src.models.train_utils import run_training

# Keep this small on purpose. 3 models x 3 configs = 9 runs, each a few
# minutes on CPU -- manageable in one sitting. Expand later only if time allows.
SWEEP_CONFIGS = [
    {"hidden_channels": 32, "lr": 0.01, "dropout": 0.3},
    {"hidden_channels": 64, "lr": 0.01, "dropout": 0.3},
    {"hidden_channels": 64, "lr": 0.005, "dropout": 0.5},
]


def build_model(model_name, in_channels, config):
    if model_name == "GCN-Skip":
        return GCNSkip(in_channels, hidden_channels=config["hidden_channels"], dropout=config["dropout"])
    elif model_name == "GraphSAGE":
        return GraphSAGE(in_channels, hidden_channels=config["hidden_channels"], dropout=config["dropout"])
    elif model_name == "GAT-Skip":
        return GATSkip(in_channels, hidden_channels=config["hidden_channels"], heads=4, dropout=config["dropout"])
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    data = load_graph()
    train_mask, test_mask, labeled_mask = get_split_masks()
    data = standardize_features(data, train_mask)

    model_names = ["GCN-Skip", "GraphSAGE", "GAT-Skip"]
    best_per_model = {}

    for model_name, config in itertools.product(model_names, SWEEP_CONFIGS):
        torch.manual_seed(42)
        model = build_model(model_name, data.num_node_features, config)

        config_str = f"hc{config['hidden_channels']}_lr{config['lr']}_do{config['dropout']}"
        run_label = f"{model_name} [{config_str}]"
        print(f"\n=== {run_label} ===")

        preds, probs, y_true = run_training(
            model, data, train_mask, test_mask,
            epochs=100, lr=config["lr"], weight_decay=5e-4,
        )

        results = evaluate_illicit(y_true, preds, probs, model_name=run_label)
        log_results(results, split_name="hyperparam_sweep")

        f1 = results["f1_illicit"]
        if model_name not in best_per_model or f1 > best_per_model[model_name][1]:
            best_per_model[model_name] = (run_label, f1, config)

    print("\n\n=== Best config per model ===")
    for model_name, (run_label, f1, config) in best_per_model.items():
        print(f"{model_name}: F1={f1:.4f} | {config}")


if __name__ == "__main__":
    main()