# Final Model Card — GraphGuard

**Frozen on:** 2026-08-16
**Architecture:** GCN with skip connection (GCNSkip)
**Seed:** 42

## Hyperparameters
- hidden_channels: 64
- dropout: 0.3
- learning_rate: 0.01
- weight_decay: 0.0005
- epochs: 100
- loss: class-weighted cross-entropy
- feature preprocessing: z-score standardization (fit on train split only)

## Performance (temporal test split, time steps 35-49)
- Precision (illicit): 0.5893
- Recall (illicit): 0.6122
- F1 (illicit): 0.6005
- AUC-PR (illicit): 0.6111

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
