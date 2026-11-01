"""
Scores a model on the clean graph, under random camouflage (averaged over
sampling seeds), and under the feature-aware nearest-licit attack.
"""
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

from src.attacks.target_selection import get_attack_targets
from src.attacks.camouflage_attack import get_licit_pool, apply_camouflage_attack
from src.attacks.nearest_attack import apply_nearest_attack


@torch.no_grad()
def predict(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    return out.argmax(1), torch.softmax(out, 1)[:, 1]


def metrics(y, pred, prob):
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc_pr": average_precision_score(y, prob),
        "fp": int(((pred == 1) & (y == 0)).sum()),
    }


def evaluate_under_attacks(model, data, train_mask, test_mask, name,
                           budgets=(1, 3, 5), seeds=(0, 1, 2, 3, 4)):
    test_idx = test_mask.nonzero(as_tuple=True)[0]
    y = data.y[test_idx].numpy()
    licit_pool = get_licit_pool(data, train_mask)
    targets, _, _ = get_attack_targets(model, data, test_mask)

    def score(graph):
        p, pr = predict(model, graph)
        m = metrics(y, p[test_idx].numpy(), pr[test_idx].numpy())
        return m, (p[targets] != 1).float().mean().item()

    rows = []
    m, _ = score(data)
    rows.append({"model": name, "attack": "none", "budget": 0, "evasion": 0.0, **m})

    for b in budgets:
        ms, evs = [], []
        for s in seeds:
            m, ev = score(apply_camouflage_attack(data, targets, licit_pool, budget=b, seed=s))
            ms.append(m); evs.append(ev)
        avg = {k: float(np.mean([x[k] for x in ms])) for k in ms[0]}
        rows.append({"model": name, "attack": "random", "budget": b,
                     "evasion": float(np.mean(evs)), **avg})

        m, ev = score(apply_nearest_attack(data, targets, licit_pool, budget=b))
        rows.append({"model": name, "attack": "nearest", "budget": b, "evasion": ev, **m})
    return rows