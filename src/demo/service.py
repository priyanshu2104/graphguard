"""
Demo back end: loads the graph and the models once, scores transactions under three configurations,
simulates a camouflage attack on one transaction, and produces explanations.

Configurations (all seed-42 models, as in the paper):
  no_graph : GCN-Skip trained without edges (edge attacks cannot change its input)
  graph    : the frozen GCN-Skip, undefended
  pruned   : the same frozen GCN-Skip with the least similar 25% of edges removed at inference

The nodes of the Elliptic graph are TRANSACTIONS (anonymised), not wallets.
"""
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.attacks.evaluate_attack import load_frozen_model
from src.attacks.model_registry import build_model
from src.attacks.sametime_attack import apply_random_sametime
from src.attacks.adaptive_attack import apply_adaptive_attack
from src.defense.similarity_prune import prune, edge_sim
from src.explain.explainer_api import explain_node

REPO_ROOT = Path(__file__).resolve().parents[2]
NO_GRAPH_PATH = REPO_ROOT / "results" / "demo_models" / "no_graph_gcnskip_seed42.pt"
PRUNE_Q = 0.25
TEST_FIRST_STEP = 35
LABELS = {"no_graph": "No graph", "graph": "Graph (GCN-Skip)", "pruned": "Graph + pruning (25%)"}
LABEL_NAMES = {1: "illicit", 0: "licit", -1: "unlabelled"}
CANDIDATE_KINDS = [
    "Illicit, flagged by the graph model (test period)",
    "Illicit, missed by the graph model (test period)",
    "Licit (test period)",
]


class DemoService:
    def __init__(self):
        data = load_graph()
        self.train_mask, self.test_mask, _ = get_split_masks()
        self.data = standardize_features(data, self.train_mask)
        self.graph_model = load_frozen_model(self.data)
        self.no_graph_model = self._load_no_graph()

        self.no_graph_data = copy.copy(self.data)
        self.no_graph_data.edge_index = torch.empty((2, 0), dtype=torch.long)

        real = edge_sim(F.normalize(self.data.x, dim=1), self.data.edge_index).numpy()
        self.prune_thr = float(np.quantile(real, PRUNE_Q))
        self.pruned_data, _ = prune(self.data, self.prune_thr)

        self.licit_all = (self.data.y == 0).nonzero(as_tuple=True)[0]
        self.idx_to_txid = {int(i): int(t) for t, i in self.data.txid_to_idx.items()}
        self.clean_probs = {k: self.probs(m, g) for k, (m, g) in self.variants().items()}

    def _load_no_graph(self):
        if not NO_GRAPH_PATH.exists():
            raise FileNotFoundError(f"{NO_GRAPH_PATH} not found. Run: python -m src.demo.train_no_graph")
        model = build_model("GCN-Skip", self.data.num_node_features)
        model.load_state_dict(torch.load(NO_GRAPH_PATH, weights_only=True))
        model.eval()
        return model

    def variants(self):
        return {"no_graph": (self.no_graph_model, self.no_graph_data),
                "graph": (self.graph_model, self.data),
                "pruned": (self.graph_model, self.pruned_data)}

    @staticmethod
    def probs(model, graph):
        model.eval()
        with torch.no_grad():
            return torch.softmax(model(graph.x, graph.edge_index), dim=1)[:, 1].numpy()

    # ---- transactions -------------------------------------------------------------------------
    def txid(self, idx):
        return self.idx_to_txid[int(idx)]

    def idx(self, txid):
        i = self.data.txid_to_idx.get(int(txid))
        return None if i is None else int(i)

    def info(self, idx):
        step = int(self.data.time_step[idx])
        return {"txid": self.txid(idx), "time_step": step,
                "label": LABEL_NAMES[int(self.data.y[idx])],
                "test_period": step >= TEST_FIRST_STEP,
                "in_degree": int((self.data.edge_index[1] == int(idx)).sum())}

    def candidates(self, kind, n=150, seed=0):
        y = self.data.y
        flagged = torch.from_numpy(self.clean_probs["graph"] >= 0.5)
        if kind == CANDIDATE_KINDS[0]:
            mask = self.test_mask & (y == 1) & flagged
        elif kind == CANDIDATE_KINDS[1]:
            mask = self.test_mask & (y == 1) & ~flagged
        else:
            mask = self.test_mask & (y == 0)
        idx = mask.nonzero(as_tuple=True)[0].numpy()
        if len(idx) > n:
            idx = np.random.default_rng(seed).choice(idx, size=n, replace=False)
        return sorted(int(i) for i in idx)

    # ---- attack simulation --------------------------------------------------------------------
    def _attack_graph(self, kind, target, budget, seed, thr, pool_size=50):
        if kind == "random":
            return apply_random_sametime(self.data, target, budget=budget, seed=seed)
        return apply_adaptive_attack(self.data, self.graph_model, target, self.licit_all, thr=thr,
                                     budget=budget, pool_size=pool_size, seed=seed, same_time=True)

    def _strongest(self, kind, target, budget, seed, attack_thr, defend_thr):
        """The strongest attack (lowest score for the target) over the pool sizes used in the paper."""
        best = None
        for m in ((50,) if kind == "random" else (50, 200)):
            att = self._attack_graph(kind, target, budget, seed, attack_thr, pool_size=m)
            scored = att if defend_thr is None else prune(att, defend_thr)[0]
            p = float(self.probs(self.graph_model, scored)[int(target[0])])
            if best is None or p < best[0]:
                best = (p, att)
        return best

    def simulate_attack(self, idx, kind="adaptive", budget=3, seed=0):
        """Attacks ONE transaction with time-consistent partners. The no-graph model cannot be
        affected (its input has no edges); the pruned model faces an attacker that knows the threshold.
        For the adaptive attacker the stronger of two pool sizes is used, as in the paper."""
        target = torch.tensor([int(idx)])
        e0 = self.data.edge_index.shape[1]
        p0 = float(self.clean_probs["no_graph"][idx])
        out = {"no_graph": {"before": p0, "after": p0, "partners": []}}

        p_graph, att = self._strongest(kind, target, budget, seed, -2.0, None)
        out["graph"] = {"before": float(self.clean_probs["graph"][idx]), "after": p_graph,
                        "partners": [self.txid(p) for p in att.edge_index[0, e0::2].tolist()]}

        p_pruned, att_p = self._strongest(kind, target, budget, seed, self.prune_thr, self.prune_thr)
        out["pruned"] = {"before": float(self.clean_probs["pruned"][idx]), "after": p_pruned,
                         "partners": [self.txid(p) for p in att_p.edge_index[0, e0::2].tolist()]}
        return out

    # ---- explanations -------------------------------------------------------------------------
    def explain(self, variant, idx, **kwargs):
        model, graph = self.variants()[variant]
        return explain_node(model, graph, int(idx), **kwargs)