"""
Defence A: adversarial training of GCN-Skip. Each epoch the training graph is
perturbed with random camouflage edges applied to the SAME FRACTION of illicit
and licit training nodes (so the perturbation itself carries no label
information), then one training step is taken on the perturbed graph.
Evaluated with the shared harness against both attacks, over 3 training seeds.
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.attacks.model_registry import build_model
from src.attacks.camouflage_attack import get_licit_pool, apply_camouflage_attack
from src.attacks.train_and_cache import get_or_train_model
from src.models.train_utils import compute_class_weights, train_one_epoch
from src.defense.eval_utils import evaluate_under_attacks

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "results" / "defense_eval.csv"

TRAIN_SEEDS = [42, 0, 1]
EPOCHS, FRAC, TRAIN_BUDGETS = 100, 0.5, [1, 3, 5]


def train_adversarial(data, train_mask, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = build_model("GCN-Skip", data.num_node_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    class_weights = compute_class_weights(data, train_mask)

    licit_pool = get_licit_pool(data, train_mask)  # licit training nodes
    illicit_train = ((data.y == 1) & train_mask).nonzero(as_tuple=True)[0]

    def sample(idx):
        return idx[torch.from_numpy(rng.random(len(idx)) < FRAC)]

    for epoch in range(1, EPOCHS + 1):
        victims = torch.cat([sample(illicit_train), sample(licit_pool)])
        budget = int(rng.choice(TRAIN_BUDGETS))
        perturbed = apply_camouflage_attack(
            data, victims, licit_pool, budget=budget,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        loss = train_one_epoch(model, perturbed, optimizer, train_mask, class_weights)
        if epoch % 20 == 0:
            print(f"seed {seed} | epoch {epoch} | loss {loss:.4f}", flush=True)

    model.eval()
    return model


def main():
    data = load_graph()
    train_mask, test_mask, _ = get_split_masks()
    data = standardize_features(data, train_mask)

    rows = []
    # Undefended references through the SAME harness. The cheapest alternative
    # to a defence is simply choosing a more robust architecture.
    for arch in ["GCN-Skip", "GraphSAGE", "GAT-Skip"]:
        ref = get_or_train_model(arch, data, train_mask, test_mask)
        rows += evaluate_under_attacks(ref, data, train_mask, test_mask, f"undefended {arch}")

    for seed in TRAIN_SEEDS:
        model = train_adversarial(data, train_mask, seed)
        rows += evaluate_under_attacks(model, data, train_mask, test_mask, f"adv-train s{seed}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)

    df["group"] = df["model"].str.replace(r" s\d+$", "", regex=True)
    summary = (df.groupby(["group", "attack", "budget"])[["f1", "recall", "fp", "evasion"]]
                 .agg(["mean", "std"]).round(3))
    print(summary.to_string())
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()