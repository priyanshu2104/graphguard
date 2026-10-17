"""
Runs separately from the GNN robustness eval to avoid a PyTorch/XGBoost
OpenMP conflict (segfault on macOS) when both load in one process.
Appends Random Forest / XGBoost rows to results/robustness_table.csv.
"""
import sklearn.ensemble, xgboost  # noqa: F401  -- must load BEFORE torch (macOS OpenMP crash)
import pandas as pd
from pathlib import Path

from src.utils.splits import get_baseline_splits
from src.models.baselines import train_random_forest, train_xgboost

REPO_ROOT = Path(__file__).resolve().parents[2]
ROBUSTNESS_TABLE_PATH = REPO_ROOT / "results" / "robustness_table.csv"
BUDGETS = [1, 3, 5]


def main():
    X_train, y_train, X_test, y_test = get_baseline_splits()
    _, rf_results = train_random_forest(X_train, y_train, X_test, y_test)
    _, xgb_results = train_xgboost(X_train, y_train, X_test, y_test)

    rows = []
    for name, results in [("Random Forest", rf_results), ("XGBoost", xgb_results)]:
        for budget in [0] + BUDGETS:
            row = {"architecture": name, "budget": budget, "evasion_rate": 0.0, **results}
            row["model"] = (f"{name} [clean]" if budget == 0
                            else f"{name} [attacked b={budget}] (structurally immune)")
            rows.append(row)

    df_new = pd.DataFrame(rows)

    if ROBUSTNESS_TABLE_PATH.exists():
        existing = pd.read_csv(ROBUSTNESS_TABLE_PATH)
        existing = existing[~existing["architecture"].isin(["Random Forest", "XGBoost"])]
        df_combined = pd.concat([existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(ROBUSTNESS_TABLE_PATH, index=False)
    print(f"Baseline immunity rows added to {ROBUSTNESS_TABLE_PATH}")


if __name__ == "__main__":
    main()