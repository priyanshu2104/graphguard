"""
1) Is the greedy adaptive attacker crippled by hub concentration?
2) Does pruning hold against a diversified adaptive attacker?
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import degree

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.attacks.target_selection import get_attack_targets
from src.attacks.camouflage_attack import get_licit_pool
from src.attacks.adaptive_attack import apply_adaptive_attack
from src.attacks.train_and_cache import get_or_train_model
from src.defense.eval_utils import predict, metrics
from src.defense.similarity_prune import prune, edge_sim


def main():
    data = load_graph()
    train_mask, test_mask, _ = get_split_masks()
    data = standardize_features(data, train_mask)
    model = get_or_train_model("GCN-Skip", data, train_mask, test_mask)
    targets, _, _ = get_attack_targets(model, data, test_mask)
    licit_pool = get_licit_pool(data, train_mask)

    test_idx = test_mask.nonzero(as_tuple=True)[0]
    y = data.y[test_idx].numpy()
    real = edge_sim(F.normalize(data.x, dim=1), data.edge_index).numpy()
    E0 = data.edge_index.shape[1]

    def score(graph, thr):
        pruned, _ = prune(graph, thr)
        p, pr = predict(model, pruned)
        return metrics(y, p[test_idx].numpy(), pr[test_idx].numpy())

    print("1) Hub diagnostic (greedy attacker, no pruning constraint)")
    for b in [1, 3, 5]:
        att = apply_adaptive_attack(data, model, targets, licit_pool, thr=-2.0, budget=b)
        partners = torch.unique(att.edge_index[0, E0::2])   # licit end of each added edge pair
        indeg = degree(att.edge_index[1], num_nodes=att.num_nodes)
        print(f"   b={b}: {len(partners)} distinct partners for {len(targets)} targets; "
              f"mean in-degree of those partners after attack {indeg[partners].mean():.0f}")

    print("\n2) Greedy vs diversified adaptive attacker (F1 / recall)")
    for q in [0.0, 0.10, 0.25, 0.50]:
        thr = float(np.quantile(real, q)) if q > 0 else -2.0
        for b in [3, 5]:
            g = score(apply_adaptive_attack(data, model, targets, licit_pool, thr=thr, budget=b), thr)
            line = f"   q={q:.2f} b={b} | greedy {g['f1']:.3f}/{g['recall']:.3f}"
            for m in [200, 2000]:
                a = apply_adaptive_attack(data, model, targets, licit_pool, thr=thr,
                                          budget=b, pool_size=m, seed=0)
                d = score(a, thr)
                line += f" | diversified M={m} {d['f1']:.3f}/{d['recall']:.3f}"
            print(line, flush=True)


if __name__ == "__main__":
    main()