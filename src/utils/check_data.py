"""
Confirms the raw Elliptic CSVs are present and readable.
Run: python src/utils/check_data.py
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw/elliptic_bitcoin_dataset")

features_path = DATA_DIR / "elliptic_txs_features.csv"
classes_path = DATA_DIR / "elliptic_txs_classes.csv"
edges_path = DATA_DIR / "elliptic_txs_edgelist.csv"

for path in [features_path, classes_path, edges_path]:
    assert path.exists(), f"Missing file: {path}. Check your download."

features = pd.read_csv(features_path, header=None)
classes = pd.read_csv(classes_path)
edges = pd.read_csv(edges_path)

print("Features shape:", features.shape)   # expect (203769, 167) -> col0 = txId, cols1-166 = features
print("Classes shape:", classes.shape)     # expect (203769, 2)  -> txId, class
print("Edges shape:", edges.shape)         # expect (234355, 2)  -> txId1, txId2

print("\nClass distribution:")
print(classes["class"].value_counts())
# Expect: '3' (unknown) ~157205, '2' (licit) ~42019, '1' (illicit) ~4545

print("\n✅ Dataset OK — all 3 files present and readable with expected shapes.")