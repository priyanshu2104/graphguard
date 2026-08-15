"""
Loads the raw Elliptic CSVs and merges them into one clean DataFrame.
Import this from notebooks/scripts rather than re-writing loading logic everywhere.
"""
import pandas as pd
from pathlib import Path

# Anchor DATA_DIR to the repo root regardless of the caller's working directory
# (a notebook in notebooks/ has a different cwd than a script run from the repo root).
# __file__ = .../graphguard/src/utils/load_data.py
# .parents[2] climbs: utils -> src -> graphguard (repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "raw" / "elliptic_bitcoin_dataset"


def load_raw():
    """Returns (features_df, classes_df, edges_df) exactly as they are on disk."""
    features = pd.read_csv(DATA_DIR / "elliptic_txs_features.csv", header=None)
    classes = pd.read_csv(DATA_DIR / "elliptic_txs_classes.csv")
    edges = pd.read_csv(DATA_DIR / "elliptic_txs_edgelist.csv")

    # Name the feature columns explicitly so nothing downstream relies on bare integer indices
    feature_cols = ["txId", "time_step"] + [f"feat_{i}" for i in range(1, 166)]
    features.columns = feature_cols

    return features, classes, edges


def build_merged_table():
    """
    Merges features + classes into one table indexed by txId.
    class label is mapped to an int: 1 = illicit, 0 = licit, -1 = unknown.
    """
    features, classes, edges = load_raw()

    merged = features.merge(classes, on="txId", how="left")

    label_map = {"1": 1, "2": 0, "unknown": -1}
    merged["label"] = merged["class"].map(label_map)
    merged = merged.drop(columns=["class"])

    return merged, edges


if __name__ == "__main__":
    merged, edges = build_merged_table()
    print("Merged table shape:", merged.shape)
    print(merged["label"].value_counts())
    print("\nEdges shape:", edges.shape)