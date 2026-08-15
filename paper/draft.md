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
### 4.4 Adversarial attack design
### 4.5 Defense mechanism
### 4.6 Explainability

## 5. Experiments & Results
### 5.1 Baseline vs GNN comparison
- Random Forest: Precision 0.9908, Recall 0.6934, F1 0.8159, AUC-PR 0.7897
- XGBoost: Precision 0.8422, Recall 0.7341, F1 0.7844, AUC-PR 0.8023
- (GCN/GraphSAGE/GAT rows added Week 5)

### 5.2 Robustness evaluation
### 5.3 Defense evaluation
### 5.4 Explainability examples

## 6. Discussion & Limitations

## 7. Conclusion & Future Work

## References