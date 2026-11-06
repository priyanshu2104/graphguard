"""
Checks the assumption behind the local-features experiment: that the first 93 of the 165 columns are
local features and the last 72 are aggregated ones (the layout in the dataset's published description).
If the assumption holds, a Random Forest on the local columns alone should score clearly below one on
all columns, as in Weber et al. (2019). This cannot prove the layout; it can contradict it.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

from src.utils.splits import get_baseline_splits

N_LOCAL = 93


def main():
    X_train, y_train, X_test, y_test = get_baseline_splits()
    for name, cols in [("all 165 columns", slice(None)),
                       ("first 93 (local?)", slice(0, N_LOCAL)),
                       ("last 72 (aggregated?)", slice(N_LOCAL, None))]:
        rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                    random_state=42, n_jobs=-1)
        rf.fit(X_train[:, cols], y_train)
        pred = rf.predict(X_test[:, cols])
        print(f"RF on {name:<24} precision {precision_score(y_test, pred):.3f} "
              f"recall {recall_score(y_test, pred):.3f} F1 {f1_score(y_test, pred):.3f}")


if __name__ == "__main__":
    main()