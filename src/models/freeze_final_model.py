"""
Trains and saves the FINAL frozen GCN-Skip model. This is the single
canonical checkpoint used for all adversarial attack/defense/explainability
work in Weeks 9-13. Do not retrain or swap architectures after this point --
if something needs to change, that's a team decision, not a solo one.
"""
import torch
import json
from pathlib import Path
from datetime import datetime

from src.utils.splits import load_graph, get_split_masks, standardize_features
from src.utils.metrics import evaluate_illicit, log_results
from src.models.gnn_models import GCNSkip
from src.models.train_utils import run_training

REPO_ROOT = Path(__file__).resolve().parents[2]
FINAL_MODEL_PATH = REPO_ROOT / "results" / "final_model.pt"
MODEL_CARD_PATH = REPO_ROOT / "results" / "model_card.md"

# Use the seed that produced the median (not necessarily the maximum) F1 in
# your reproducibility check -- picking the single best-of-5 seed would be
# a subtle form of cherry-picking. Update FINAL_SEED after reviewing your
# repro_log.txt results.
FINAL_SEED = 42
FINAL_CONFIG = {"hidden_channels": 64, "lr": 0.01, "dropout": 0.3, "weight_decay": 5e-4, "epochs": 100}


def main():
    data = load_graph()
    train_mask, test_mask, labeled_mask = get_split_masks()
    data = standardize_features(data, train_mask)

    torch.manual_seed(FINAL_SEED)
    model = GCNSkip(
        in_channels=data.num_node_features,
        hidden_channels=FINAL_CONFIG["hidden_channels"],
        dropout=FINAL_CONFIG["dropout"],
    )

    print(f"Training FINAL frozen model (seed={FINAL_SEED})...")
    preds, probs, y_true = run_training(
        model, data, train_mask, test_mask,
        epochs=FINAL_CONFIG["epochs"], lr=FINAL_CONFIG["lr"],
        weight_decay=FINAL_CONFIG["weight_decay"],
    )

    results = evaluate_illicit(y_true, preds, probs, model_name="GCN-Skip-FINAL")
    log_results(results, split_name="frozen_final")

    # Save weights
    torch.save(model.state_dict(), FINAL_MODEL_PATH)

    # Save the exact config alongside the weights, so loading it later
    # (Week 9+) doesn't rely on anyone remembering the right hyperparameters
    config_path = REPO_ROOT / "results" / "final_model_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "architecture": "GCNSkip",
            "in_channels": data.num_node_features,
            "seed": FINAL_SEED,
            **FINAL_CONFIG,
        }, f, indent=2)

    # Human-readable model card -- this is what you paste into the paper
    # and what any teammate reads before touching this model in Week 9+
    model_card = f"""# Final Model Card — GraphGuard

**Frozen on:** {datetime.now().strftime('%Y-%m-%d')}
**Architecture:** GCN with skip connection (GCNSkip)
**Seed:** {FINAL_SEED}

## Hyperparameters
- hidden_channels: {FINAL_CONFIG['hidden_channels']}
- dropout: {FINAL_CONFIG['dropout']}
- learning_rate: {FINAL_CONFIG['lr']}
- weight_decay: {FINAL_CONFIG['weight_decay']}
- epochs: {FINAL_CONFIG['epochs']}
- loss: class-weighted cross-entropy
- feature preprocessing: z-score standardization (fit on train split only)

## Performance (temporal test split, time steps 35-49)
- Precision (illicit): {results['precision_illicit']}
- Recall (illicit): {results['recall_illicit']}
- F1 (illicit): {results['f1_illicit']}
- AUC-PR (illicit): {results['auc_pr_illicit']}

## Files
- Weights: `results/final_model.pt`
- Config: `results/final_model_config.json`

## Usage (Weeks 9+)
```python
import torch, json
from src.models.gnn_models import GCNSkip

with open("results/final_model_config.json") as f:
    config = json.load(f)

model = GCNSkip(
    in_channels=config["in_channels"],
    hidden_channels=config["hidden_channels"],
    dropout=config["dropout"],
)
model.load_state_dict(torch.load("results/final_model.pt", weights_only=True))
model.eval()
```

**This model is frozen. Do not retrain, change architecture, or swap
hyperparameters without a team discussion — all attack/defense/explainability
results from Week 9 onward assume this exact model.**
"""

    with open(MODEL_CARD_PATH, "w") as f:
        f.write(model_card)

    print(f"\nFinal model saved to {FINAL_MODEL_PATH}")
    print(f"Config saved to {config_path}")
    print(f"Model card written to {MODEL_CARD_PATH}")


if __name__ == "__main__":
    main()