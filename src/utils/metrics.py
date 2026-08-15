"""
Shared evaluation harness. Accuracy is deliberately NOT included as a headline
metric -- with ~2-11% illicit prevalence, a model that predicts "licit" for
everything scores 90%+ accuracy while catching zero fraud. Precision/Recall/F1/
AUC-PR on the illicit class are what actually matter here.
"""
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix,
)
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = REPO_ROOT / "results" / "results_log.csv"


def evaluate_illicit(y_true, y_pred, y_proba=None, model_name="unnamed"):
    """
    y_true, y_pred: arrays of 0 (licit) / 1 (illicit)
    y_proba: predicted probability of illicit class (needed for AUC-PR). Optional.
    """
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    auc_pr = average_precision_score(y_true, y_proba) if y_proba is not None else None

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    results = {
        "model": model_name,
        "precision_illicit": round(precision, 4),
        "recall_illicit": round(recall, 4),
        "f1_illicit": round(f1, 4),
        "auc_pr_illicit": round(auc_pr, 4) if auc_pr is not None else None,
    }

    print(f"\n--- {model_name} ---")
    print(f"Precision (illicit): {precision:.4f}")
    print(f"Recall    (illicit): {recall:.4f}")
    print(f"F1        (illicit): {f1:.4f}")
    if auc_pr is not None:
        print(f"AUC-PR    (illicit): {auc_pr:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    return results


def log_results(results: dict, split_name="baseline_temporal"):
    """Appends a results dict to results/results_log.csv, creating it if needed."""
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    row = {**results, "split": split_name, "timestamp": datetime.now().isoformat(timespec="seconds")}
    df_row = pd.DataFrame([row])

    if RESULTS_PATH.exists():
        df_row.to_csv(RESULTS_PATH, mode="a", header=False, index=False)
    else:
        df_row.to_csv(RESULTS_PATH, mode="w", header=True, index=False)

    print(f"Logged to {RESULTS_PATH}")