"""Is AUC-PR a reliable ranking comparison here? Adds ROC-AUC and checks ties/saturation."""
import copy
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.attacks.model_registry import build_model
from src.attacks.train_and_cache import get_or_train_model
from src.models.train_utils import run_training


@torch.no_grad()
def report(name, model, graph, test_idx, y):
    model.eval()
    p = torch.softmax(model(graph.x, graph.edge_index), dim=1)[test_idx, 1].numpy()
    print(f"{name:<20} AP {average_precision_score(y, p):.4f} | ROC-AUC {roc_auc_score(y, p):.4f} | "
          f"distinct probs {len(np.unique(p))} | p>0.999 {np.mean(p > 0.999):.1%} | p<0.001 {np.mean(p < 0.001):.1%}")


def main():
    data = load_graph()
    train_mask, test_mask, _ = get_split_masks()
    data = standardize_features(data, train_mask)
    test_idx = test_mask.nonzero(as_tuple=True)[0]
    y = data.y[test_idx].numpy()

    for arch in ["GCN-Skip", "GraphSAGE", "GAT-Skip"]:
        report(arch, get_or_train_model(arch, data, train_mask, test_mask), data, test_idx, y)

    no_graph = copy.copy(data)
    no_graph.edge_index = torch.empty((2, 0), dtype=torch.long)
    torch.manual_seed(42)
    model = build_model("GCN-Skip", in_channels=data.num_node_features)
    run_training(model, no_graph, train_mask, test_mask, epochs=100, lr=0.01)
    report("GCN-Skip [no graph]", model, no_graph, test_idx, y)


if __name__ == "__main__":
    main()