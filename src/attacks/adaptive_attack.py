"""
Defence-aware (white-box) attacker. Knows the pruning threshold `thr` and the
model. For each target, considers licit pool nodes whose cosine similarity to
the target is >= thr (so the edge survives pruning), ranked by how licit the
model finds them (lowest illicit probability). Edges are added in both
directions. thr = -2 means no pruning constraint.

pool_size=None : greedy (top-`budget`); many targets pick the SAME partners,
                 which become hubs with little weight under GCN normalisation.
pool_size=M    : pick `budget` partners at random from the top-M candidates.
same_time=True : only partners from the target's own time step are allowed.
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F


def apply_adaptive_attack(data, model, target_node_indices, licit_pool, thr,
                          budget=3, pool_size=None, seed=0, chunk=128, same_time=False):
    rng = np.random.default_rng(seed)
    xn = F.normalize(data.x, dim=1)
    model.eval()
    with torch.no_grad():
        p_illicit = torch.softmax(model(data.x, data.edge_index), dim=1)[:, 1]
    pool_x = xn[licit_pool]
    licit_score = -p_illicit[licit_pool]  # higher = looks more licit to the model
    pool_ts = data.time_step[licit_pool] if same_time else None

    k = budget if pool_size is None else max(budget, pool_size)
    k = min(k, len(licit_pool))

    src, dst = [], []
    for start in range(0, len(target_node_indices), chunk):
        t = target_node_indices[start:start + chunk]
        sim = xn[t] @ pool_x.T                                  # [B, pool]
        score = licit_score.unsqueeze(0).expand_as(sim).clone()
        score[sim < thr] = float("-inf")                        # edge would be pruned
        if same_time:
            score[pool_ts.unsqueeze(0) != data.time_step[t].unsqueeze(1)] = float("-inf")
        vals, top = score.topk(k, dim=1)
        chosen = licit_pool[top]
        for row, target in enumerate(t.tolist()):
            ok = torch.isfinite(vals[row]).nonzero(as_tuple=True)[0].tolist()
            if pool_size is None:
                pick = ok[:budget]
            else:
                pick = rng.choice(ok, size=min(budget, len(ok)), replace=False).tolist() if ok else []
            for kk in pick:
                licit = int(chosen[row, kk])
                src += [licit, target]
                dst += [target, licit]

    attacked = copy.copy(data)
    new_edges = torch.tensor([src, dst], dtype=torch.long)
    attacked.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
    return attacked