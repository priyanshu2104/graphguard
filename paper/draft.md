# GraphGuard: Explainable & Adversarially-Robust Fraud Detection in Cryptocurrency Networks using GNNs

## Abstract
(write last)

## 1. Introduction

## 2. Related Work
### 2.1 Fraud detection on blockchain data
### 2.2 Graph Neural Networks
### 2.3 Adversarial robustness of GNNs
### 2.4 Explainability in GNNs

## 3. Dataset
(Elliptic dataset description, class distribution, temporal structure -- pull numbers from this week's EDA)
- Elliptic Bitcoin dataset: 203,769 nodes, 234,355 edges, 165 features/node
- Class distribution: ~2% illicit, ~21% licit, ~77% unknown (overall dataset)
- Temporal split: train on time steps 1-34, test on 35-49 (following Weber et al.)
- Train set: 29,894 nodes, 3,462 illicit (11.58%)
- Test set: 16,670 nodes, 1,083 illicit (6.50%)

## 4. Methodology
### 4.1 Graph construction & temporal split
### 4.2 Baseline models
### 4.3 GNN architectures
- GraphSAGE (Hamilton et al.): 2-layer, hidden dim 64, ReLU, dropout 0.3,
  mean aggregation (PyG default), class-weighted cross-entropy loss,
  standardized input features (z-score, fit on train split only).
- GCN (Kipf & Welling), with skip connection: 2-layer GCN backbone
  (hidden dim 64) whose learned representation is concatenated with the
  original raw node features before a final linear classification layer.
  Same training configuration as GraphSAGE (class-weighted loss,
  standardized features). The skip connection was necessary — a vanilla
  GCN without it underperformed GraphSAGE by a wide margin (F1 0.44 vs 0.58).
- Both trained transductively on the full graph (203,769 nodes); loss
  computed only on labeled training nodes (time steps 1-34); evaluated on
  labeled test nodes (time steps 35-49).
### 4.4 Adversarial attack design
### 4.5 Defense mechanism
### 4.6 Explainability

## 5. Experiments & Results
### 5.1 Baseline vs GNN comparison

Main comparison (final numbers used in headline results):

| Model | Precision | Recall | F1 | AUC-PR |
|---|---|---|---|---|
| Random Forest | 0.9908 | 0.6934 | 0.8159 | 0.7897 |
| XGBoost | 0.8422 | 0.7341 | 0.7844 | 0.8023 |
| GraphSAGE | 0.5402 | 0.6260 | 0.5800 | 0.6108 |
| GCN (with skip connection) | 0.5893 | 0.6122 | 0.6005 | 0.6111 |

Tree-based baselines outperform vanilla GNNs on this dataset, consistent with
the original Elliptic paper (Weber et al.). This is attributed to the strong
local-feature signal in the 165 engineered features, combined with the
~77% unlabeled-neighbor noise that dilutes pure graph-averaging.

### Ablation: what closed the gap for GCN

| Variant | F1 |
|---|---|
| GCN, raw (unstandardized) features | 0.4621 |
| GCN, standardized features, no skip connection | 0.4386 |
| GCN, standardized features + skip connection | 0.6005 |

Feature standardization alone did not help GCN in isolation; the skip
connection (concatenating raw node features with the learned graph
representation before classification) was the change that meaningfully
closed the gap to GraphSAGE and the classical baselines.

### 5.2 Robustness evaluation
### 5.3 Defense evaluation
### 5.4 Explainability examples

## 6. Discussion & Limitations

## 7. Conclusion & Future Work

## References