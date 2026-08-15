"""
Temporal train/test split for the Elliptic dataset.
Train: time steps 1-34. Test: time steps 35-49.
Only LABELED nodes (illicit/licit) are used for classical ML baselines --
unlabeled nodes have no ground truth to train or evaluate against.
"""
import torch
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GRAPH_PATH = REPO_ROOT / "data" / "processed" / "elliptic_graph.pt"

TRAIN_CUTOFF = 34  # time steps 1-34 = train, 35-49 = test


def load_graph():
    return torch.load(GRAPH_PATH, weights_only=False)


def get_baseline_splits():
    """
    Returns X_train, y_train, X_test, y_test as numpy arrays,
    restricted to labeled nodes only, split by time step.
    """
    data = load_graph()

    x = data.x.numpy()
    y = data.y.numpy()
    time_step = data.time_step.numpy()

    labeled_mask = y != -1
    train_mask = labeled_mask & (time_step <= TRAIN_CUTOFF)
    test_mask = labeled_mask & (time_step > TRAIN_CUTOFF)

    X_train, y_train = x[train_mask], y[train_mask]
    X_test, y_test = x[test_mask], y[test_mask]

    return X_train, y_train, X_test, y_test


def get_split_masks():
    """
    Returns boolean masks (train_mask, test_mask, labeled_mask) over ALL nodes
    (including unlabeled). GNN models from Week 4 onward use these directly on
    the full graph rather than extracting a separate feature matrix.
    """
    data = load_graph()
    y = data.y.numpy()
    time_step = data.time_step.numpy()

    labeled_mask = y != -1
    train_mask = labeled_mask & (time_step <= TRAIN_CUTOFF)
    test_mask = labeled_mask & (time_step > TRAIN_CUTOFF)

    return (
        torch.tensor(train_mask),
        torch.tensor(test_mask),
        torch.tensor(labeled_mask),
    )


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_baseline_splits()

    print(f"Train: {X_train.shape[0]} nodes | illicit: {(y_train == 1).sum()} "
          f"({(y_train == 1).mean():.2%})")
    print(f"Test:  {X_test.shape[0]} nodes | illicit: {(y_test == 1).sum()} "
          f"({(y_test == 1).mean():.2%})")