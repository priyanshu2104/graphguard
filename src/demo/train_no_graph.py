"""Trains the no-graph GCN-Skip used by the demo (seed 42, the configuration used in the paper) and saves it."""
import copy

import torch

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.attacks.model_registry import build_model
from src.models.train_utils import run_training
from src.defense.eval_utils import predict, metrics
from src.demo.service import NO_GRAPH_PATH


def main():
    data = load_graph()
    train_mask, test_mask, _ = get_split_masks()
    data = standardize_features(data, train_mask)
    no_graph = copy.copy(data)
    no_graph.edge_index = torch.empty((2, 0), dtype=torch.long)

    torch.manual_seed(42)
    model = build_model("GCN-Skip", data.num_node_features)
    run_training(model, no_graph, train_mask, test_mask, epochs=100, lr=0.01)
    model.eval()

    NO_GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), NO_GRAPH_PATH)

    test_idx = test_mask.nonzero(as_tuple=True)[0]
    pred, prob = predict(model, no_graph)
    m = metrics(data.y[test_idx].numpy(), pred[test_idx].numpy(), prob[test_idx].numpy())
    print(f"\nSaved {NO_GRAPH_PATH}")
    print(f"Test F1 {m['f1']:.4f} (paper, seed 42: 0.6296) | false positives {m['fp']} (paper: 199)")


if __name__ == "__main__":
    main()