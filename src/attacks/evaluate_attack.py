"""
Ties together target selection + the camouflage attack: runs the attack
at several budgets against the frozen GCN-Skip model, measures how much
detection degrades, and logs everything to results/attack_results_log.csv.
"""
import torch
import json
import pandas as pd
from pathlib import Path

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.models.gnn_models import GCNSkip
from src.attacks.target_selection import get_attack_targets
from src.attacks.camouflage_attack import get_licit_pool, apply_camouflage_attack
from src.utils.metrics import evaluate_illicit, log_results

REPO_ROOT = Path(__file__).resolve().parents[2]
ATTACK_LOG_PATH = REPO_ROOT / "results" / "attack_results_log.csv"

BUDGETS = [1, 3, 5]


def load_frozen_model(data):
    with open(REPO_ROOT / "results" / "final_model_config.json") as f:
        config = json.load(f)
    model = GCNSkip(
        in_channels=config["in_channels"],
        hidden_channels=config["hidden_channels"],
        dropout=config["dropout"],
    )
    model.load_state_dict(torch.load(REPO_ROOT / "results" / "final_model.pt", weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def get_predictions(model, data, mask):
    out = model(data.x, data.edge_index)
    probs = torch.softmax(out, dim=1)[:, 1]
    preds = out.argmax(dim=1)
    return preds[mask], probs[mask]


def log_attack_result(row: dict):
    """Separate CSV from results_log.csv -- attack results have extra
    columns (budget, evasion_rate) that don't fit the standard schema."""
    df_row = pd.DataFrame([row])
    if ATTACK_LOG_PATH.exists():
        df_row.to_csv(ATTACK_LOG_PATH, mode="a", header=False, index=False)
    else:
        df_row.to_csv(ATTACK_LOG_PATH, mode="w", header=True, index=False)


def main():
    data = load_graph()
    train_mask, test_mask, labeled_mask = get_split_masks()
    data = standardize_features(data, train_mask)

    model = load_frozen_model(data)

    # Step 1: find valid attack targets (correctly-classified illicit nodes)
    target_node_indices, clean_preds, clean_probs = get_attack_targets(model, data, test_mask)

    test_indices = test_mask.nonzero(as_tuple=True)[0]
    y_test = data.y[test_indices]

    # Log the clean (unattacked) baseline for reference
    clean_results = evaluate_illicit(
        y_test.numpy(), clean_preds[test_indices].numpy(), clean_probs[test_indices].numpy(),
        model_name="GCN-Skip [clean]"
    )
    log_results(clean_results, split_name="attack_eval")

    licit_pool = get_licit_pool(data, train_mask)
    total_targets = len(target_node_indices)

    print(f"\n{'='*50}\nRunning camouflage attack across budgets: {BUDGETS}\n{'='*50}")

    for budget in BUDGETS:
        attacked_data = apply_camouflage_attack(data, target_node_indices, licit_pool, budget=budget)
        preds_attacked, probs_attacked = get_predictions(model, attacked_data, test_mask)

        # Of the originally-correctly-flagged targets, how many are STILL flagged illicit?
        target_positions_in_test = torch.isin(test_indices, target_node_indices)
        still_detected = (preds_attacked[target_positions_in_test] == 1).sum().item()
        evasion_rate = 1 - (still_detected / total_targets)

        print(f"\n--- Budget={budget} camouflage edges/node ---")
        print(f"Targets still detected: {still_detected}/{total_targets}")
        print(f"Evasion rate: {evasion_rate:.2%}")

        # Overall model performance on the FULL test set under attack
        # (not just the targeted nodes -- confirms the attack doesn't
        # accidentally break predictions on untouched nodes too)
        attacked_results = evaluate_illicit(
            y_test.numpy(), preds_attacked.numpy(), probs_attacked.numpy(),
            model_name=f"GCN-Skip [attacked budget={budget}]"
        )
        log_results(attacked_results, split_name="attack_eval")

        log_attack_result({
            "budget": budget,
            "total_targets": total_targets,
            "still_detected": still_detected,
            "evasion_rate": round(evasion_rate, 4),
            "overall_f1_under_attack": attacked_results["f1_illicit"],
            "overall_recall_under_attack": attacked_results["recall_illicit"],
        })

    print(f"\nAttack results logged to {ATTACK_LOG_PATH}")


if __name__ == "__main__":
    main()