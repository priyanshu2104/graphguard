"""Does GAT-Skip down-weight camouflage edges? (layer 1 only)
attention/uniform == 1.0 means the edge gets an average share."""
import torch

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.attacks.target_selection import get_attack_targets
from src.attacks.camouflage_attack import get_licit_pool, apply_camouflage_attack
from src.attacks.train_and_cache import get_or_train_model


@torch.no_grad()
def main():
    data = load_graph()
    train_mask, test_mask, _ = get_split_masks()
    data = standardize_features(data, train_mask)
    model = get_or_train_model("GAT-Skip", data, train_mask, test_mask)
    targets, _, _ = get_attack_targets(model, data, test_mask)
    licit_pool = get_licit_pool(data, train_mask)

    n, E0 = data.num_nodes, data.edge_index.shape[1]
    is_target = torch.zeros(n, dtype=torch.bool)
    is_target[targets] = True

    for b in [1, 5, 20]:
        att = apply_camouflage_attack(data, targets, licit_pool, budget=b, seed=0)
        cam = att.edge_index[:, E0:]
        cam_keys = cam[0] * n + cam[1]

        _, (ei, alpha) = model.conv1(att.x, att.edge_index, return_attention_weights=True)
        src, dst = ei[0], ei[1]
        ratio = alpha.mean(dim=1) * torch.bincount(dst, minlength=n)[dst].float()

        is_cam = torch.isin(src * n + dst, cam_keys)
        is_self = src == dst
        into_t = is_target[dst]
        for label, m in [("camouflage edges", is_cam & into_t & ~is_self),
                         ("original edges", ~is_cam & ~is_self & into_t),
                         ("self-loops", is_self & into_t)]:
            print(f"b={b} | {label}: n={int(m.sum())}, attention/uniform = {ratio[m].mean():.2f}")


if __name__ == "__main__":
    main()