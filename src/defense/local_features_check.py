"""
Is the graph unhelpful because Elliptic's 72 aggregated features already encode
one-hop neighbourhoods? Train GCN-Skip on the LOCAL features only (first 93
columns; assumes 93 local then 72 aggregated), with and without edges, over 3
training seeds. Also attack the with-graph model (adaptive, b=3, M=2000).
"""
import copy
import numpy as np
import torch

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.attacks.model_registry import build_model
from src.attacks.target_selection import get_attack_targets
from src.attacks.camouflage_attack import get_licit_pool
from src.attacks.adaptive_attack import apply_adaptive_attack
from src.models.train_utils import run_training
from src.defense.eval_utils import predict, metrics

SEEDS, N_LOCAL = [42, 0, 1], 93


def main():
    data = load_graph()
    train_mask, test_mask, _ = get_split_masks()
    data = standardize_features(data, train_mask)

    local = copy.copy(data)
    local.x = data.x[:, :N_LOCAL]
    no_graph = copy.copy(local)
    no_graph.edge_index = torch.empty((2, 0), dtype=torch.long)

    licit_pool = get_licit_pool(local, train_mask)
    test_idx = test_mask.nonzero(as_tuple=True)[0]
    y = data.y[test_idx].numpy()

    out = {"graph": [], "no graph": [], "evasion": []}
    for seed in SEEDS:
        for name, g in [("graph", local), ("no graph", no_graph)]:
            torch.manual_seed(seed)
            model = build_model("GCN-Skip", N_LOCAL)
            run_training(model, g, train_mask, test_mask, epochs=100, lr=0.01)
            p, pr = predict(model, g)
            m = metrics(y, p[test_idx].numpy(), pr[test_idx].numpy())
            out[name].append((m["f1"], m["auc_pr"]))
            print(f"seed {seed} | local features, {name}: F1 {m['f1']:.3f}  AUC-PR {m['auc_pr']:.3f}",
                  flush=True)
            if name == "graph":
                targets, _, _ = get_attack_targets(model, local, test_mask)
                att = apply_adaptive_attack(local, model, targets, licit_pool, thr=-2.0,
                                            budget=3, pool_size=2000, seed=0)
                pa, _ = predict(model, att)
                out["evasion"].append((pa[targets] != 1).float().mean().item())
                print(f"         adaptive attack b=3: evasion {out['evasion'][-1]:.1%}", flush=True)

    print("\nMean ± sample std over training seeds (local features only)")
    for name in ["graph", "no graph"]:
        f1 = [v[0] for v in out[name]]
        ap = [v[1] for v in out[name]]
        print(f"  {name:<9} F1 {np.mean(f1):.3f} ± {np.std(f1, ddof=1):.3f} | "
              f"AUC-PR {np.mean(ap):.3f} ± {np.std(ap, ddof=1):.3f}")
    print(f"  evasion of the with-graph model, adaptive b=3: "
          f"{np.mean(out['evasion']):.1%} ± {np.std(out['evasion'], ddof=1):.1%}")


if __name__ == "__main__":
    main()