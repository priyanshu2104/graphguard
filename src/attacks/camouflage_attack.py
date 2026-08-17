"""
Camouflage-edge evasion attack: adds edges connecting targeted illicit
nodes to randomly sampled licit nodes, in both directions. This dilutes
the illicit node's aggregated neighborhood representation without
changing its own local features at all.
"""
import copy
import torch
import numpy as np


def get_licit_pool(data, train_mask):
    """
    Pool of node indices labeled licit within the TRAINING set only.
    Using only training-set licit nodes (not test-set ones) keeps this
    attack realistic -- an attacker plausibly has access to known
    legitimate wallets, not knowledge of which specific nodes are in
    your held-out test split.
    """
    licit_mask = (data.y == 0) & train_mask
    return licit_mask.nonzero(as_tuple=True)[0]


def apply_camouflage_attack(data, target_node_indices, licit_pool, budget=3, seed=42):
    """
    For each target node, sample `budget` licit nodes and add edges in
    both directions (licit -> target and target -> licit). Returns a NEW
    Data object with the augmented edge_index -- the original `data` is
    never mutated, so the clean graph stays available for comparison.
    """
    rng = np.random.default_rng(seed)
    licit_pool_np = licit_pool.numpy()

    new_src, new_dst = [], []
    for node_idx in target_node_indices.tolist():
        sampled = rng.choice(licit_pool_np, size=budget, replace=False)
        for licit_node in sampled:
            new_src.append(int(licit_node)); new_dst.append(node_idx)   # licit -> target
            new_src.append(node_idx); new_dst.append(int(licit_node))   # target -> licit

    new_edges = torch.tensor([new_src, new_dst], dtype=torch.long)

    attacked_data = copy.copy(data)
    attacked_data.edge_index = torch.cat([data.edge_index, new_edges], dim=1)
    return attacked_data