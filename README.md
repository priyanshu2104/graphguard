# GraphGuard

Explainable & Adversarially-Robust Fraud Detection in Cryptocurrency Networks using Graph Neural Networks.

Minor project — B.Tech CSE, final year. 3 credits.

## Overview

Cryptocurrency transaction graphs are pseudonymous, public, and permanent.
Illicit wallets (ransomware, darknet markets, mixers) reveal themselves
through network structure, not isolated features — which motivates a
graph-based approach over classical tabular ML. This project:

1. Benchmarks classical ML (Random Forest, XGBoost) against GNN architectures
   (GCN, GraphSAGE, GAT) on the Elliptic Bitcoin dataset
2. Designs and evaluates an adversarial graph-evasion attack against the
   best-performing GNN, and a defense against it
3. Adds an explainability layer (GNNExplainer) so every flagged wallet comes
   with a reason, not just a score
4. Packages the result as a Streamlit demo with a tamper-evident (Merkle-tree)
   audit log

## Results So Far

| Model | Precision | Recall | F1 | AUC-PR |
|---|---|---|---|---|
| Random Forest | 0.9908 | 0.6934 | 0.8159 | 0.7897 |
| XGBoost | 0.8422 | 0.7341 | 0.7844 | 0.8023 |
| GCN-Skip (frozen model) | 0.5893 | 0.6122 | 0.6005 | 0.6111 |
| GraphSAGE | 0.5402 | 0.6260 | 0.5800 | 0.6108 |
| GAT-Skip | 0.3930 | 0.7054 | 0.5048 | 0.5695 |

See `results/model_card.md` for the exact frozen model specification.

## Setup

```bash
git clone <repo-url>
cd graphguard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .          # enables `import src...` from anywhere
```

On macOS, XGBoost additionally requires OpenMP:
```bash
brew install libomp
```

Download the dataset:
```bash
kaggle datasets download -d ellipticco/elliptic-data-set -p data/raw --unzip
```

## Reproducing Results

```bash
python -m src.utils.build_graph          # builds + saves the graph object
python -m src.models.baselines            # Random Forest + XGBoost
python -m src.models.train_gcn            # GCN-Skip
python -m src.models.train_graphsage      # GraphSAGE
python -m src.models.train_gat            # GAT-Skip
python -m src.utils.consolidate_results   # final comparison table
```

## Project Structure

```
graphguard/
├── data/               # raw + processed data (gitignored, not redistributed)
├── notebooks/          # exploratory analysis
├── src/
│   ├── models/         # model definitions, training scripts
│   ├── attacks/        # adversarial attack + defense (Week 9+)
│   ├── explain/         # explainability layer (Week 13+)
│   └── utils/           # data loading, splits, metrics, shared helpers
├── demo/                # Streamlit app (Week 13+)
├── paper/               # research paper draft + figures
├── results/             # logged experiment results, frozen model, model card
└── docs/                 # setup guides, reading notes
```

## Team

- Rishi Rajkumar — Data pipeline, baselines, evaluation
- Priyanshu Shekhar — GNN modeling, adversarial robustness
- Satya Prakash — Explainability, demo, paper writing

## Dataset

[Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
— Weber et al., "Anti-Money Laundering in Bitcoin: Experimenting with GCNs
for Financial Forensics" (arXiv:1908.02591)