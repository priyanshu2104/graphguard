"""
Seed check for the two defence-related claims:
 A) no-graph model clean F1 / AUC-PR over training seeds
 B) pruning GCN-Skip: clean F1 and worst-case adaptive-attack F1 over training seeds
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.attacks.model_registry import build_model
from src.attacks.target_selection import get_attack_targets
from src.attacks.camouflage_attack import get_licit_pool
from src.attacks.adaptive_attack import apply_adaptive_attack
from src.models.train_utils import run_training
from src.defense.eval_utils import predict, metrics
from src.defense.similarity_prune import prune, edge_sim
from src.defense.seed_check import get_model

SEEDS, QS, BUDGETS, POOLS = [42, 0, 1], [0.0, 0.25, 0.50], [3, 5], [200, 2000]


def main():
    data = load_graph()
    train_mask, test_mask, _ = get_split_masks()
    data = standardize_features(data, train_mask)
    licit_pool = get_licit_pool(data, train_mask)
    test_idx = test_mask.nonzero(as_tuple=True)[0]
    y = data.y[test_idx].numpy()
    real = edge_sim(F.normalize(data.x, dim=1), data.edge_index).numpy()

    print("A) No-graph model")
    ng_f1, ng_ap = [], []
    no_graph = copy.copy(data)
    no_graph.edge_index = torch.empty((2, 0), dtype=torch.long)
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = build_model("GCN-Skip", data.num_node_features)
        run_training(model, no_graph, train_mask, test_mask, epochs=100, lr=0.01)
        p, pr = predict(model, no_graph)
        m = metrics(y, p[test_idx].numpy(), pr[test_idx].numpy())
        ng_f1.append(m["f1"]); ng_ap.append(m["auc_pr"])
        print(f"   seed {seed}: F1 {m['f1']:.3f}  AUC-PR {m['auc_pr']:.3f}", flush=True)
    print(f"   no-graph F1 {np.mean(ng_f1):.3f} ± {np.std(ng_f1, ddof=1):.3f} | "
          f"AUC-PR {np.mean(ng_ap):.3f} ± {np.std(ng_ap, ddof=1):.3f}\n")

    print("B) Pruned GCN-Skip")
    res = {q: {"clean": [], "worst_f1": [], "worst_recall": []} for q in QS}
    for seed in SEEDS:
        model = get_model("GCN-Skip", seed, data, train_mask, test_mask)
        targets, _, _ = get_attack_targets(model, data, test_mask)

        def score(graph, thr):
            pruned, _ = prune(graph, thr)
            p, pr = predict(model, pruned)
            return metrics(y, p[test_idx].numpy(), pr[test_idx].numpy())

        for q in QS:
            thr = float(np.quantile(real, q)) if q > 0 else -2.0
            res[q]["clean"].append(score(data, thr)["f1"])
            worst = None
            for b in BUDGETS:
                for m_pool in POOLS:
                    att = apply_adaptive_attack(data, model, targets, licit_pool, thr=thr,
                                                budget=b, pool_size=m_pool, seed=0)
                    r = score(att, thr)
                    if worst is None or r["f1"] < worst["f1"]:
                        worst = r
            res[q]["worst_f1"].append(worst["f1"]); res[q]["worst_recall"].append(worst["recall"])
            print(f"   seed {seed} q={q:.2f}: clean F1 {res[q]['clean'][-1]:.3f} | "
                  f"worst adaptive F1 {worst['f1']:.3f} recall {worst['recall']:.3f}", flush=True)

    print("\n   Mean ± sample std over training seeds")
    for q in QS:
        c, w, r = res[q]["clean"], res[q]["worst_f1"], res[q]["worst_recall"]
        print(f"   q={q:.2f}: clean F1 {np.mean(c):.3f} ± {np.std(c, ddof=1):.3f} | "
              f"worst adaptive F1 {np.mean(w):.3f} ± {np.std(w, ddof=1):.3f} | "
              f"worst recall {np.mean(r):.3f} ± {np.std(r, ddof=1):.3f}")


if __name__ == "__main__":
    main()