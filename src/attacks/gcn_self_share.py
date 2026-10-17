"""Share of GCN's (structural) aggregation weight that stays on the target itself."""
import torch

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.attacks.target_selection import get_attack_targets
from src.attacks.camouflage_attack import get_licit_pool, apply_camouflage_attack
from src.attacks.train_and_cache import get_or_train_model


def self_share(edge_index, n, targets):
    deg = torch.bincount(edge_index[1], minlength=n).float() + 1  # in-degree + self-loop, as gcn_norm
    w = deg[edge_index[0]].pow(-0.5) * deg[edge_index[1]].pow(-0.5)
    self_w = 1.0 / deg
    total = torch.zeros(n).index_add_(0, edge_index[1], w) + self_w
    return (self_w / total)[targets].mean().item()


def main():
    data = load_graph()
    train_mask, test_mask, _ = get_split_masks()
    data = standardize_features(data, train_mask)
    gat = get_or_train_model("GAT-Skip", data, train_mask, test_mask)
    targets, _, _ = get_attack_targets(gat, data, test_mask)  # same targets as the attention check
    licit_pool = get_licit_pool(data, train_mask)

    print(f"b=0  | GCN self-share {self_share(data.edge_index, data.num_nodes, targets):.1%}")
    for b in [1, 5, 20]:
        att = apply_camouflage_attack(data, targets, licit_pool, budget=b, seed=0)
        print(f"b={b:<2} | GCN self-share {self_share(att.edge_index, att.num_nodes, targets):.1%}")


if __name__ == "__main__":
    main()