"""
Feature-aware camouflage attack: connect each target to the `budget` licit
nodes (from the training pool) with the highest cosine similarity to it in
standardized feature space. Edges are added in both directions, matching the
random attack. Deterministic, so there is no sampling seed.
"""
import copy
import torch
import torch.nn.functional as F


def apply_nearest_attack(data, target_node_indices, licit_pool, budget=3, chunk=256):
    xn = F.normalize(data.x, dim=1)
    pool_x = xn[licit_pool]

    src, dst = [], []
    for start in range(0, len(target_node_indices), chunk):
        t = target_node_indices[start:start + chunk]
        top = (xn[t] @ pool_x.T).topk(budget, dim=1).indices
        chosen = licit_pool[top]
        for row, target in enumerate(t.tolist()):
            for licit in chosen[row].tolist():
                src += [licit, target]
                dst += [target, licit]

    attacked = copy.copy(data)
    new_edges = torch.tensor([src, dst], dtype=torch.long)
    attacked.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
    return attacked