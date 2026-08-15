"""
Converts the merged Elliptic table + edge list into a single PyG Data object.
"""
import torch
import pandas as pd
from pathlib import Path
from torch_geometric.data import Data
from src.utils.load_data import build_merged_table

# Same repo-root anchoring as load_data.py -- makes the save path work
# identically whether this is run from the repo root, from src/utils/,
# or imported from a notebook in notebooks/.
REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def build_pyg_graph():
    merged, edges = build_merged_table()

    # Map each txId to a contiguous node index [0, N-1] -- PyG requires this,
    # raw txIds are large sparse integers and won't work directly as indices.
    txid_to_idx = {txid: i for i, txid in enumerate(merged["txId"])}

    # Build edge_index: shape [2, num_edges], filtering out any edge whose
    # endpoint isn't in our node set (shouldn't happen, but check defensively)
    src = edges["txId1"].map(txid_to_idx)
    dst = edges["txId2"].map(txid_to_idx)
    valid = src.notna() & dst.notna()
    dropped = (~valid).sum()
    if dropped > 0:
        print(f"Warning: dropped {dropped} edges with unknown endpoints")

    import numpy as np

    edge_index_np = np.array(
        [src[valid].astype(int).values, dst[valid].astype(int).values]
    )
    edge_index = torch.from_numpy(edge_index_np).long()

    # Feature matrix: everything except txId, time_step, label
    feature_cols = [c for c in merged.columns if c.startswith("feat_")]
    x = torch.tensor(merged[feature_cols].values, dtype=torch.float)

    y = torch.tensor(merged["label"].values, dtype=torch.long)
    time_step = torch.tensor(merged["time_step"].values, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y, time_step=time_step)

    # Keep the txId <-> index mapping attached to the object (useful later for
    # the demo app, where a user will look up a wallet by its real txId).
    data.txid_to_idx = txid_to_idx

    return data


def summarize(data):
    """Readable one-line summary instead of print(data), which dumps the
    entire txid_to_idx dictionary and floods the terminal."""
    print(
        f"Nodes: {data.num_nodes} | Edges: {data.num_edges} | "
        f"Features: {data.num_node_features} | Directed: {data.is_directed()} | "
        f"Isolated nodes: {data.has_isolated_nodes()}"
    )


if __name__ == "__main__":
    data = build_pyg_graph()

    summarize(data)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PROCESSED_DIR / "elliptic_graph.pt"
    torch.save(data, save_path)
    print(f"Saved to {save_path}")