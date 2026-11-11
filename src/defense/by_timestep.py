"""
Evasion by time step for the frozen GCN-Skip (seed 42), adaptive attacker with b = 3.
All conditions use the same targets (correctly detected illicit test transactions):
  A  undefended model, UNCONSTRAINED attacker  (what the undefended rows of the tables used)
  B  undefended model, similarity-CONSTRAINED attacker (the kind used against the pruned model)
  C  pruned model, constrained attacker         (what the pruned rows used)
  D  pruned model, unconstrained attacker
The figure shown for each time step is the larger of the rates for M = 50 and M = 200.
"""
import numpy as np
import torch.nn.functional as F

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.attacks.evaluate_attack import load_frozen_model
from src.attacks.target_selection import get_attack_targets
from src.attacks.adaptive_attack import apply_adaptive_attack
from src.defense.eval_utils import predict
from src.defense.similarity_prune import prune, edge_sim

BUDGET, POOLS, PRUNE_Q = 3, [50, 200], 0.25


def main():
    data = load_graph()
    train_mask, test_mask, _ = get_split_masks()
    data = standardize_features(data, train_mask)
    model = load_frozen_model(data)
    targets, _, _ = get_attack_targets(model, data, test_mask)
    licit_all = (data.y == 0).nonzero(as_tuple=True)[0]
    thr = float(np.quantile(edge_sim(F.normalize(data.x, dim=1), data.edge_index).numpy(), PRUNE_Q))
    steps = data.time_step[targets].numpy()

    def evaded(graph, defend_thr):
        g = graph if defend_thr is None else prune(graph, defend_thr)[0]
        pred, _ = predict(model, g)
        return (pred[targets] != 1).numpy()

    conditions = {"A": (-2.0, None), "B": (thr, None), "C": (thr, thr), "D": (-2.0, thr)}
    masks = {}
    for name, (attack_thr, defend_thr) in conditions.items():
        masks[name] = [evaded(apply_adaptive_attack(data, model, targets, licit_all, thr=attack_thr,
                                                    budget=BUDGET, pool_size=m, seed=0, same_time=True),
                              defend_thr) for m in POOLS]
        print(f"condition {name} done", flush=True)

    def rate(name, sel):
        return 100 * max(float(m[sel].mean()) for m in masks[name])

    print("\nEvasion (%) of detected illicit test transactions, b=3, strongest of M = 50 / 200")
    print("step |   n |     A     B     C     D")
    for s in sorted(set(steps.tolist())):
        sel = steps == s
        print(f"{s:>4} | {int(sel.sum()):>3} | " + " ".join(f"{rate(n, sel):5.1f}" for n in "ABCD"))
    allsel = np.ones(len(steps), dtype=bool)
    print(f" all | {len(steps):>3} | " + " ".join(f"{rate(n, allsel):5.1f}" for n in "ABCD"))
    print("\nA undefended/unconstrained | B undefended/constrained | C pruned/constrained | D pruned/unconstrained")


if __name__ == "__main__":
    main()