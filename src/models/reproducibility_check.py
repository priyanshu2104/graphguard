"""
Reproducibility check: reruns the best GCN-Skip config across multiple seeds
to confirm the Week 5 result (F1=0.6005) is stable, not a lucky seed.
"""
import torch
import numpy as np

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.utils.metrics import evaluate_illicit, log_results
from src.models.gnn_models import GCNSkip
from src.models.train_utils import run_training

SEEDS = [42, 0, 1, 7, 123]
BEST_CONFIG = {"hidden_channels": 64, "lr": 0.01, "dropout": 0.3}


def main():
    data = load_graph()
    train_mask, test_mask, labeled_mask = get_split_masks()
    data = standardize_features(data, train_mask)

    f1_scores = []
    all_results = []

    for seed in SEEDS:
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = GCNSkip(
            in_channels=data.num_node_features,
            hidden_channels=BEST_CONFIG["hidden_channels"],
            dropout=BEST_CONFIG["dropout"],
        )

        print(f"\n=== GCN-Skip, seed={seed} ===")
        preds, probs, y_true = run_training(
            model, data, train_mask, test_mask,
            epochs=100, lr=BEST_CONFIG["lr"], weight_decay=5e-4,
        )

        results = evaluate_illicit(y_true, preds, probs, model_name=f"GCN-Skip [seed={seed}]")
        log_results(results, split_name="reproducibility_check")

        f1_scores.append(results["f1_illicit"])
        all_results.append(results)

    f1_scores = np.array(f1_scores)
    print("\n\n=== Reproducibility Summary ===")
    print(f"F1 scores across {len(SEEDS)} seeds: {f1_scores.tolist()}")
    print(f"Mean F1: {f1_scores.mean():.4f}")
    print(f"Std Dev: {f1_scores.std():.4f}")
    print(f"Min/Max: {f1_scores.min():.4f} / {f1_scores.max():.4f}")

    if f1_scores.std() > 0.05:
        print("\n⚠️  High variance across seeds (std > 0.05) — worth noting as a "
              "limitation in the paper, and consider averaging multiple runs "
              "for the frozen model rather than picking a single seed.")
    else:
        print("\n✅ Low variance across seeds — result looks stable.")


if __name__ == "__main__":
    main()