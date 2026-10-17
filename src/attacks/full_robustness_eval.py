"""
Full GNN robustness evaluation: runs the camouflage attack across all
three GNN architectures and all budgets. Baselines (Random Forest /
XGBoost) are handled in baseline_immunity_eval.py, a separate process,
to avoid a PyTorch/XGBoost OpenMP conflict that segfaults on macOS.
Produces results/robustness_table.csv.
"""
import torch
import pandas as pd
from pathlib import Path

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.attacks.target_selection import get_attack_targets
from src.attacks.camouflage_attack import get_licit_pool, apply_camouflage_attack
from src.attacks.train_and_cache import get_or_train_model
from src.utils.metrics import evaluate_illicit

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBUSTNESS_TABLE_PATH = REPO_ROOT / "results" / "robustness_table.csv"

GNN_MODELS = ["GCN-Skip", "GraphSAGE", "GAT-Skip"]
BUDGETS = [1, 3, 5]


@torch.no_grad()
def get_predictions(model, data, mask):
    out = model(data.x, data.edge_index)
    probs = torch.softmax(out, dim=1)[:, 1]
    preds = out.argmax(dim=1)
    return preds[mask], probs[mask]


def evaluate_gnn_robustness(name, data, train_mask, test_mask, licit_pool, rows):
    model = get_or_train_model(name, data, train_mask, test_mask)

    target_node_indices, clean_preds, clean_probs = get_attack_targets(model, data, test_mask)
    test_indices = test_mask.nonzero(as_tuple=True)[0]
    y_test = data.y[test_indices]
    total_targets = len(target_node_indices)

    clean_results = evaluate_illicit(
        y_test.numpy(), clean_preds[test_indices].numpy(), clean_probs[test_indices].numpy(),
        model_name=f"{name} [clean]"
    )
    rows.append({"architecture": name, "budget": 0, "evasion_rate": 0.0, **clean_results})

    for budget in BUDGETS:
        attacked_data = apply_camouflage_attack(data, target_node_indices, licit_pool, budget=budget)
        preds_a, probs_a = get_predictions(model, attacked_data, test_mask)

        target_pos = torch.isin(test_indices, target_node_indices)
        still_detected = (preds_a[target_pos] == 1).sum().item()
        evasion_rate = 1 - (still_detected / total_targets) if total_targets else 0.0

        attacked_results = evaluate_illicit(
            y_test.numpy(), preds_a.numpy(), probs_a.numpy(),
            model_name=f"{name} [attacked b={budget}]"
        )
        rows.append({"architecture": name, "budget": budget,
                     "evasion_rate": round(evasion_rate, 4), **attacked_results})

        print(f"{name} | budget={budget} | evasion_rate={evasion_rate:.2%} | "
              f"F1={attacked_results['f1_illicit']:.4f}")


def main():
    data = load_graph()
    train_mask, test_mask, labeled_mask = get_split_masks()
    data = standardize_features(data, train_mask)
    licit_pool = get_licit_pool(data, train_mask)

    rows = []
    for name in GNN_MODELS:
        print(f"\n{'='*50}\n{name}\n{'='*50}")
        evaluate_gnn_robustness(name, data, train_mask, test_mask, licit_pool, rows)

    df = pd.DataFrame(rows)
    df.to_csv(ROBUSTNESS_TABLE_PATH, index=False)
    print(f"\nGNN robustness table saved to {ROBUSTNESS_TABLE_PATH}")
    print(df[["architecture", "budget", "evasion_rate", "f1_illicit", "recall_illicit"]].to_string(index=False))


if __name__ == "__main__":
    main()