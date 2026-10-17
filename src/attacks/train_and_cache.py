"""
Trains (or loads a cached checkpoint for) each GNN architecture using its
registered best config, so repeated runs of the robustness evaluation
don't retrain from scratch every time.
"""
import torch
from pathlib import Path

from src.attacks.model_registry import MODEL_CONFIGS, build_model
from src.models.train_utils import run_training

REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "results" / "attack_models"


def get_or_train_model(name, data, train_mask, test_mask, seed=42, force_retrain=False):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{name.replace(' ', '_')}.pt"

    if cache_path.exists() and not force_retrain:
        model = build_model(name, in_channels=data.num_node_features)
        model.load_state_dict(torch.load(cache_path, weights_only=True))
        model.eval()
        print(f"Loaded cached {name} from {cache_path}")
        return model

    print(f"Training {name} fresh (seed={seed})...")
    torch.manual_seed(seed)
    model = build_model(name, in_channels=data.num_node_features)
    run_training(model, data, train_mask, test_mask, epochs=100, lr=MODEL_CONFIGS[name]["lr"])
    model.eval()

    torch.save(model.state_dict(), cache_path)
    return model