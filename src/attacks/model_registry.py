"""
Registry of each GNN architecture's BEST hyperparameter configuration,
taken directly from the Week 5 sweep results. Used to rebuild consistent,
known-good models for the Week 10 robustness evaluation.
"""
from src.models.gnn_models import GCNSkip, GraphSAGE, GATSkip

MODEL_CONFIGS = {
    "GCN-Skip":  {"cls": GCNSkip,   "hidden_channels": 64, "dropout": 0.3, "lr": 0.01, "extra": {}},
    "GraphSAGE": {"cls": GraphSAGE, "hidden_channels": 64, "dropout": 0.3, "lr": 0.01, "extra": {}},
    "GAT-Skip":  {"cls": GATSkip,   "hidden_channels": 32, "dropout": 0.3, "lr": 0.01, "extra": {"heads": 4}},
}


def build_model(name, in_channels):
    cfg = MODEL_CONFIGS[name]
    return cfg["cls"](
        in_channels=in_channels,
        hidden_channels=cfg["hidden_channels"],
        dropout=cfg["dropout"],
        **cfg["extra"],
    )