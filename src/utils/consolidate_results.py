"""
Reads results/results_log.csv, finds the best (highest F1) run for each
GNN model from the hyperparameter sweep, and combines with the Week 3
baseline numbers into one final comparison table.
"""
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = REPO_ROOT / "results" / "results_log.csv"


def get_best_per_model(df, model_prefix):
    """Among rows whose 'model' column starts with model_prefix, return the
    one with the highest F1."""
    subset = df[df["model"].str.startswith(model_prefix)]
    if subset.empty:
        return None
    return subset.loc[subset["f1_illicit"].idxmax()]


def main():
    df = pd.read_csv(RESULTS_PATH)

    baseline_rf = df[df["model"] == "Random Forest"].iloc[-1]
    baseline_xgb = df[df["model"] == "XGBoost"].iloc[-1]

    best_gcn = get_best_per_model(df, "GCN-Skip")
    best_sage = get_best_per_model(df, "GraphSAGE")
    best_gat = get_best_per_model(df, "GAT-Skip")

    final_table = pd.DataFrame([
        baseline_rf, baseline_xgb, best_gcn, best_sage, best_gat
    ])[["model", "precision_illicit", "recall_illicit", "f1_illicit", "auc_pr_illicit"]]

    print("\n=== Final Comparison Table ===")
    print(final_table.to_string(index=False))

    # Also export as a markdown table for direct paste into paper/draft.md
    md_lines = ["| Model | Precision | Recall | F1 | AUC-PR |", "|---|---|---|---|---|"]
    for _, row in final_table.iterrows():
        md_lines.append(
            f"| {row['model']} | {row['precision_illicit']:.4f} | "
            f"{row['recall_illicit']:.4f} | {row['f1_illicit']:.4f} | "
            f"{row['auc_pr_illicit']:.4f} |"
        )
    md_table = "\n".join(md_lines)

    output_path = REPO_ROOT / "results" / "final_comparison_table.md"
    output_path.write_text(md_table)
    print(f"\nMarkdown table saved to {output_path}")


if __name__ == "__main__":
    main()