"""
Final comparison on the time-consistent threat model, over three training seeds.
Per model and seed: clean F1 / AUC-PR / false positives; random camouflage at b=5 (mean over 3
sampling seeds); worst case over the adaptive attacker (b = 3, 5; M = 10, 50, 200).
GCN-Skip + pruning uses PRUNE_Q: set it from the curve in prune_curve.py before running.
The no-graph model cannot be touched by edge attacks in this setting (its neighbourhood features
are frozen), so its attack columns equal its clean ones.
"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.attacks.model_registry import build_model
from src.attacks.target_selection import get_attack_targets
from src.attacks.sametime_attack import apply_random_sametime
from src.attacks.adaptive_attack import apply_adaptive_attack
from src.models.train_utils import run_training
from src.defense.eval_utils import predict, metrics
from src.defense.similarity_prune import prune, edge_sim
from src.defense.seed_check import get_model

SEEDS, BUDGETS, POOLS = [42, 0, 1], [3, 5], [10, 50, 200]
PRUNE_Q = 0.25  # set from the pruning curve
RESULTS = Path(__file__).resolve().parents[2] / "results"
COLUMNS = ["clean_f1", "clean_auc_pr", "clean_fp", "rand_b5_f1", "rand_b5_fp",
           "worst_f1", "worst_recall", "worst_fp", "adaptive_b5_evasion"]


def attack_suite(model, data, targets, licit_all, y, test_idx, thr=-2.0):
    def run(graph):
        pruned, _ = prune(graph, thr)
        p, pr = predict(model, pruned)
        m = metrics(y, p[test_idx].numpy(), pr[test_idx].numpy())
        m["evasion"] = (p[targets] != 1).float().mean().item()
        return m

    clean = run(data)
    rand = [run(apply_random_sametime(data, targets, budget=5, seed=s)) for s in (0, 1, 2)]
    worst, best_ev = None, 0.0
    for b in BUDGETS:
        for m_pool in POOLS:
            r = run(apply_adaptive_attack(data, model, targets, licit_all, thr=thr, budget=b,
                                          pool_size=m_pool, seed=0, same_time=True))
            if worst is None or r["f1"] < worst["f1"]:
                worst = r
            if b == 5:
                best_ev = max(best_ev, r["evasion"])
    return {
        "clean_f1": clean["f1"], "clean_auc_pr": clean["auc_pr"], "clean_fp": clean["fp"],
        "rand_b5_f1": float(np.mean([r["f1"] for r in rand])),
        "rand_b5_fp": float(np.mean([r["fp"] for r in rand])),
        "worst_f1": worst["f1"], "worst_recall": worst["recall"], "worst_fp": worst["fp"],
        "adaptive_b5_evasion": best_ev,
    }


def no_graph_row(seed, data, train_mask, test_mask, y, test_idx):
    no_graph = copy.copy(data)
    no_graph.edge_index = torch.empty((2, 0), dtype=torch.long)
    torch.manual_seed(seed)
    model = build_model("GCN-Skip", data.num_node_features)
    run_training(model, no_graph, train_mask, test_mask, epochs=100, lr=0.01)
    p, pr = predict(model, no_graph)
    m = metrics(y, p[test_idx].numpy(), pr[test_idx].numpy())
    return {"clean_f1": m["f1"], "clean_auc_pr": m["auc_pr"], "clean_fp": m["fp"],
            "rand_b5_f1": m["f1"], "rand_b5_fp": m["fp"], "worst_f1": m["f1"],
            "worst_recall": m["recall"], "worst_fp": m["fp"], "adaptive_b5_evasion": 0.0}


def main():
    data = load_graph()
    train_mask, test_mask, _ = get_split_masks()
    data = standardize_features(data, train_mask)
    licit_all = (data.y == 0).nonzero(as_tuple=True)[0]
    test_idx = test_mask.nonzero(as_tuple=True)[0]
    y = data.y[test_idx].numpy()
    real = edge_sim(F.normalize(data.x, dim=1), data.edge_index).numpy()
    thr_prune = float(np.quantile(real, PRUNE_Q))
    pruned_name = f"GCN-Skip + pruning ({PRUNE_Q:.0%})"

    rows = []
    for seed in SEEDS:
        for arch in ["GCN-Skip", "GraphSAGE", "GAT-Skip"]:
            model = get_model(arch, seed, data, train_mask, test_mask)
            targets, _, _ = get_attack_targets(model, data, test_mask)
            rows.append({"model": arch, "seed": seed,
                         **attack_suite(model, data, targets, licit_all, y, test_idx)})
            if arch == "GCN-Skip":
                rows.append({"model": pruned_name, "seed": seed,
                             **attack_suite(model, data, targets, licit_all, y, test_idx, thr=thr_prune)})
        rows.append({"model": "GCN-Skip, no graph", "seed": seed,
                     **no_graph_row(seed, data, train_mask, test_mask, y, test_idx)})
        print(f"seed {seed} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "final_comparison_sametime.csv", index=False)
    summary = df.groupby("model")[COLUMNS].agg(["mean", "std"]).round(3)
    print("\nMean and sample std over three training seeds")
    print(summary.to_string())

    log = pd.read_csv(RESULTS / "results_log.csv")
    print("\nTree models (single run; edge attacks cannot change their inputs in this setting)")
    for name in ["Random Forest", "XGBoost"]:
        r = log[(log["model"] == name) & (log["split"] == "baseline_temporal")].iloc[0]
        print(f"  {name}: clean F1 {r['f1_illicit']:.3f}  AUC-PR {r['auc_pr_illicit']:.3f}")


if __name__ == "__main__":
    main()