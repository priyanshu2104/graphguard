"""
Classical ML baselines: Random Forest and XGBoost, trained on node features
only (no graph structure). These numbers establish the bar the GNNs (Week 4+)
need to beat -- if a GNN can't outperform these, the graph structure isn't
actually adding value and that's worth knowing early, not in Week 12.
"""
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.utils.splits import get_baseline_splits
from src.utils.metrics import evaluate_illicit, log_results


def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        class_weight="balanced",  # important given ~2% illicit prevalence
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probability of illicit class

    results = evaluate_illicit(y_test, y_pred, y_proba, model_name="Random Forest")
    log_results(results)
    return model, results


def train_xgboost(X_train, y_train, X_test, y_test):
    # scale_pos_weight approximates class balancing for XGBoost
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results = evaluate_illicit(y_test, y_pred, y_proba, model_name="XGBoost")
    log_results(results)
    return model, results


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_baseline_splits()
    print(f"Training on {X_train.shape[0]} nodes, testing on {X_test.shape[0]} nodes\n")

    rf_model, rf_results = train_random_forest(X_train, y_train, X_test, y_test)
    xgb_model, xgb_results = train_xgboost(X_train, y_train, X_test, y_test)

    print("\n=== Summary ===")
    print(f"{'Model':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC-PR':<12}")
    for r in [rf_results, xgb_results]:
        print(f"{r['model']:<15} {r['precision_illicit']:<12} {r['recall_illicit']:<12} "
              f"{r['f1_illicit']:<12} {r['auc_pr_illicit']:<12}")