"""Random camouflage with TIME-CONSISTENT partners (same time step as the target)."""
import copy
import numpy as np
import torch


def licit_by_time(data):
    licit = (data.y == 0).nonzero(as_tuple=True)[0]
    ts = data.time_step[licit]
    return {int(t): licit[ts == t].numpy() for t in torch.unique(ts).tolist()}


def apply_random_sametime(data, targets, budget=3, seed=42):
    rng = np.random.default_rng(seed)
    pools = licit_by_time(data)
    src, dst = [], []
    for t in targets.tolist():
        pool = pools[int(data.time_step[t])]
        for licit in rng.choice(pool, size=min(budget, len(pool)), replace=False):
            src += [int(licit), t]
            dst += [t, int(licit)]
    attacked = copy.copy(data)
    new_edges = torch.tensor([src, dst], dtype=torch.long)
    attacked.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
    return attacked