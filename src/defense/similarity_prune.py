"""
Defence B (optional): inference-time feature-similarity pruning of GCN-Skip.
Step 1 diagnostic: are camouflage edges less similar than real edges?
Step 2: prune edges below a similarity threshold and re-score.
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.attacks.target_selection import get_attack_targets
from src.attacks.camouflage_attack import get_licit_pool, apply_camouflage_attack
from src.attacks.nearest_attack import apply_nearest_attack
from src.attacks.train_and_cache import get_or_train_model
from src.defense.eval_utils import predict, metrics


def edge_sim(xn, ei):
    return (xn[ei[0]] * xn[ei[1]]).sum(1)


def prune(data, thr):
    keep = edge_sim(F.normalize(data.x, dim=1), data.edge_index) >= thr
    out = copy.copy(data)
    out.edge_index = data.edge_index[:, keep]
    return out, keep.float().mean().item()


def main():
    data = load_graph()
    train_mask, test_mask, _ = get_split_masks()
    data = standardize_features(data, train_mask)
    model = get_or_train_model("GCN-Skip", data, train_mask, test_mask)

    targets, _, _ = get_attack_targets(model, data, test_mask)
    licit_pool = get_licit_pool(data, train_mask)
    E0 = data.edge_index.shape[1]
    xn = F.normalize(data.x, dim=1)

    rand_g = apply_camouflage_attack(data, targets, licit_pool, budget=3, seed=0)
    near_g = apply_nearest_attack(data, targets, licit_pool, budget=3)

    real = edge_sim(xn, data.edge_index).numpy()
    sim_rand = edge_sim(xn, rand_g.edge_index[:, E0:]).numpy()
    sim_near = edge_sim(xn, near_g.edge_index[:, E0:]).numpy()
    print("Cosine similarity of edge endpoints (median):")
    print(f"  real edges {np.median(real):.2f} | random camouflage {np.median(sim_rand):.2f} "
          f"| nearest-licit camouflage {np.median(sim_near):.2f}\n")

    test_idx = test_mask.nonzero(as_tuple=True)[0]
    y = data.y[test_idx].numpy()

    def score(g):
        p, pr = predict(model, g)
        return metrics(y, p[test_idx].numpy(), pr[test_idx].numpy())

    # thresholds chosen as quantiles of REAL-edge similarity, so q=0.10
    # means "would remove the 10% least-similar real edges"
    for q in [0.0, 0.05, 0.10, 0.25]:
        thr = float(np.quantile(real, q))
        clean_g, kept = prune(data, thr)
        c = score(clean_g)
        r = score(prune(rand_g, thr)[0])
        n = score(prune(near_g, thr)[0])
        print(f"q={q:.2f} thr={thr:.3f} real edges kept {kept:.1%} | "
              f"clean F1 {c['f1']:.3f} | random b=3 F1 {r['f1']:.3f} recall {r['recall']:.3f} "
              f"| nearest b=3 F1 {n['f1']:.3f} recall {n['recall']:.3f}")


if __name__ == "__main__":
    main()