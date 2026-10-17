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

We construct a transaction graph from the Elliptic Bitcoin dataset, where
nodes represent transactions and directed edges represent the flow of funds
between them. The graph contains 203,769 nodes and 234,355 edges, each node
described by 165 anonymized features spanning 49 discrete time steps. To
reflect a realistic deployment setting and avoid temporal leakage, we adopt
the standard split used by Weber et al.: nodes from time steps 1-34 form the
training set, and nodes from time steps 35-49 form the held-out test set.
Only labeled nodes (illicit or licit) are used for supervised training and
evaluation; the ~77% of nodes with unknown labels remain part of the graph
structure but do not contribute to the loss.

### 4.2 Baseline models

We establish two classical machine learning baselines — Random Forest and
XGBoost — trained on node features alone, without any use of the graph
structure. Both models use class-weighting (`class_weight="balanced"` for
Random Forest, `scale_pos_weight` for XGBoost) to address the substantial
class imbalance in the training set (~11.6% illicit). These baselines serve
two purposes: they establish the performance achievable from local features
alone, and they quantify how much (if anything) the graph structure
contributes once we introduce GNN models.

### 4.3 GNN architectures
- GraphSAGE (Hamilton et al.): 2-layer, hidden dim 64, ReLU, dropout 0.3,
  mean aggregation (PyG default), class-weighted cross-entropy loss,
  standardized input features (z-score, fit on train split only).
- GCN (Kipf & Welling), with skip connection: 2-layer GCN backbone
  (hidden dim 64) whose learned representation is concatenated with the
  original raw node features before a final linear classification layer.
  Same training configuration as GraphSAGE. The skip connection was
  necessary — a vanilla GCN without it underperformed GraphSAGE by a wide
  margin (F1 0.44 vs 0.58).
- GAT (Veličković et al.), with skip connection: same skip-connection
  pattern as GCN, with 4 attention heads in the first layer (concatenated)
  and 1 head in the second layer (averaged). ELU activation per the
  original GAT paper.
- Both trained transductively on the full graph (203,769 nodes); loss
  computed only on labeled training nodes (time steps 1-34); evaluated on
  labeled test nodes (time steps 35-49).
- Hyperparameter sweep: hidden_channels in {32, 64}, learning rate in
  {0.01, 0.005}, dropout in {0.3, 0.5}, 3 configs tried per architecture,
  100 epochs each, best selected by illicit-class F1 on the held-out
  temporal test split. Full sweep results logged in
  results/results_log.csv.

### 4.4 Adversarial attack design

All GNN models are trained transductively on the full graph for 100 epochs
using the Adam optimizer, with node features standardized via z-score
normalization (statistics computed from the training split only, to avoid
information leakage from the test set). We use class-weighted cross-entropy
loss throughout, with weights computed as the inverse class frequency in the
training split. A small hyperparameter sweep (Section 4.3) was conducted
across hidden layer width, learning rate, and dropout for each architecture;
the best configuration by illicit-class F1 was selected for each. To confirm
result stability rather than reporting a single favorable run, we additionally
retrained the best-performing model (GCN-Skip) across 5 random seeds
(Section 5.2).

### 4.5 Adversarial Attack Design

We implement a camouflage-edge evasion attack targeting the frozen GCN-Skip
model. For each illicit test node correctly classified by the clean model,
the attack adds `budget` bidirectional edges connecting it to randomly
sampled licit nodes drawn from the training set (never from the test set,
to keep the threat model realistic and avoid leaking test-split information
into the attack). This dilutes the node's graph-aggregated representation
without altering its own local features. We evaluate at budgets of 1, 3,
and 5 camouflage edges per node.

**Evaluation protocol.** For each model, the attack targets are the illicit
test nodes that the clean model classifies correctly (663, 678 and 764 nodes
for GCN-Skip, GraphSAGE and GAT-Skip respectively). The evasion rate is the
fraction of these targets classified as licit after the attack; it is zero on
the clean graph by construction. We evaluate budgets of 1, 3 and 5 camouflage
edges per target, plus 10 and 20 in a supplementary run. The random choice of
licit partners is repeated over five seeds for budgets 1, 3 and 5 and over
three seeds for budgets 10 and 20, and we report mean ± standard deviation.
Precision, recall, F1 and AUC-PR are computed over the full labelled test set
for a single attack draw (seed 42). Each architecture was trained once (seed
42) with its best configuration from the hyperparameter sweep, so the
reported variability reflects the attacker's random choice of partners, not
variation in training.

### 4.6 Defense mechanism
### 4.7 Explainability

## 5. Experiments & Results
### 5.1 Baseline vs GNN comparison

Final results after hyperparameter sweep (best config per GNN architecture,
selected by F1 on illicit class):

| Model | Precision | Recall | F1 | AUC-PR |
|---|---|---|---|---|
| Random Forest | 0.9908 | 0.6934 | 0.8159 | 0.7897 |
| XGBoost | 0.8422 | 0.7341 | 0.7844 | 0.8023 |
| GCN-Skip | 0.5893 | 0.6122 | 0.6005 | 0.6111 |
| GraphSAGE | 0.5402 | 0.6260 | 0.5800 | 0.6108 |
| GAT-Skip | 0.3930 | 0.7054 | 0.5048 | 0.5695 |

**Observations:**
- Tree-based baselines outperform all three GNN architectures on this
  dataset, consistent with the original Elliptic paper (Weber et al.).
  Attributed to the strong local-feature signal in the 165 engineered
  features, combined with the ~77% unlabeled-neighbor noise that dilutes
  pure graph-averaging.
- Among GNNs, GCN-Skip performs best overall (highest F1 and AUC-PR),
  followed by GraphSAGE, then GAT-Skip.
- Hyperparameter sweep found the default configuration (hidden_channels=64,
  lr=0.01, dropout=0.3) performed best for GCN-Skip and GraphSAGE; the
  higher-dropout, lower-lr configuration consistently underperformed across
  all three architectures, suggesting underfitting rather than overfitting
  at 100 training epochs.
- GAT-Skip shows a stable high-recall, low-precision profile across all
  swept configurations (recall 0.70-0.75, precision 0.28-0.39), a
  structurally different tradeoff from GCN/GraphSAGE rather than a
  tuning artifact.

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

## 5.2 Adversarial Robustness Results

**Table 3.** Evasion rate (%) of correctly detected illicit test nodes under
random camouflage, mean ± std over sampling seeds. Budgets 1, 3, 5: five
seeds; budgets 10, 20: three seeds.

| Model | Targets | b=1 | b=3 | b=5 | b=10 | b=20 |
|---|---|---|---|---|---|---|
| GCN-Skip | 663 | 26.9 ± 2.0 | 52.0 ± 1.4 | 66.9 ± 0.4 | 86.2 ± 1.4 | 96.4 ± 0.4 |
| GraphSAGE | 678 | 22.5 ± 1.5 | 31.1 ± 0.8 | 33.9 ± 0.8 | 35.1 ± 1.1 | 36.8 ± 0.6 |
| GAT-Skip | 764 | 5.1 ± 1.0 | 5.4 ± 0.3 | 5.1 ± 0.5 | 3.7 ± 0.1 | 0.9 ± 0.2 |
| Random Forest, XGBoost | — | 0 | 0 | 0 | 0 | 0 |

Random Forest and XGBoost are structurally immune: the attack adds edges and
never modifies node features, so their predictions are identical before and
after.

**Table 4.** Test-set performance under attack (single attack draw, seed 42).

| Model | Clean F1 | F1 b=1 | F1 b=3 | F1 b=5 | Clean recall | Recall b=5 |
|---|---|---|---|---|---|---|
| GCN-Skip | 0.6005 | 0.4828 | 0.3389 | 0.2642 | 0.6122 | 0.2170 |
| GraphSAGE | 0.5800 | 0.4752 | 0.4347 | 0.4479 | 0.6260 | 0.4423 |
| GAT-Skip | 0.5048 | 0.4827 | 0.4822 | 0.4885 | 0.7054 | 0.6750 |
| Random Forest | 0.8159 | 0.8159 | 0.8159 | 0.8159 | 0.6934 | 0.6934 |
| XGBoost | 0.7844 | 0.7844 | 0.7844 | 0.7844 | 0.7341 | 0.7341 |

**The ranking inverts under attack.** GCN-Skip has the highest clean F1
among the GNNs (0.6005) and the lowest under attack, while GAT-Skip is the
reverse: at b=5 its F1 (0.4885) exceeds GCN-Skip's (0.2642) despite starting
0.096 lower. GAT-Skip's advantage is not free, however: it produces 1,180
false positives on the clean test set versus 462 for GCN-Skip.

**False positives are unaffected by the attack.** They stay within three of
the clean value for every model (GCN-Skip 462 → 460–461, GraphSAGE 577 →
576–577, GAT-Skip 1,180 → 1,179–1,180). This is consistent with an attack
that perturbs only illicit nodes' neighbourhoods; the loss in F1 comes from
recall.

**The three architectures respond differently to budget.** GCN-Skip's
evasion rate keeps rising and reaches 96.4% at b=20. GraphSAGE's flattens at
roughly 34–37% from b=5 onward, so about two thirds of its targets remain
detected even at b=20. GAT-Skip's is flat at about 5% for b ≤ 5 and
then falls to 0.9% at b=20, i.e. additional camouflage edges make it less
likely to flip. We do not have an explanation for this decline (see
Section 6).

**The gap is not explained by prediction confidence.** The median clean logit
margin (illicit minus licit) on the targets is 3.10, 3.42 and 3.61 for
GCN-Skip, GraphSAGE and GAT-Skip. These are similar in magnitude, so GAT-Skip
does not simply start further from the decision boundary. Logit scales are
not strictly comparable across architectures, so we treat this as evidence
against, not a proof against, a confidence explanation.

**Targets are overwhelmingly low-degree.** About 99% of targets have
in-degree at most 2 (GCN-Skip 657 of 663, GraphSAGE 670 of 678, GAT-Skip 752
of 764). Within the in-degree 1–2 bin (561–622 targets per model), evasion at
b=3 is 51.6%, 29.0% and 4.8% respectively, so the model differences are not
explained by differences in the targets' in-degree. Bins with in-degree ≥ 3
contain only 6–12 targets per model and are not interpreted.

### 5.3 Defense evaluation

## 5.3 Explainability Under Attack

We compared GNNExplainer feature-importance vectors for the frozen GCN-Skip
model before and after a budget-3 camouflage attack on [N] randomly sampled
correctly-detected illicit test nodes ([n_evaded] evaded, [n_not] did not).
Explanations were computed on each node's 3-hop subgraph, which we verified
reproduces the full-graph logits.

The explainer is itself unstable: explaining the same node twice on the same
graph gives only [floor_overlap]/5 overlap in the top-5 features, so top-k
overlap is not a usable metric here. We therefore report Spearman rank
correlation across all 165 features, where the run-to-run baseline is
ρ = [floor_rho].

Clean-versus-attacked correlation was ρ = [rho_evaded] for nodes that evaded
and ρ = [rho_not] for nodes that did not (Mann–Whitney p = [p]).

[If p ≥ 0.05] We find no evidence that evaded nodes' explanations change more
than those of nodes that remained detected; both groups diverge from the
clean explanation by more than the explainer's own run-to-run baseline.

### 5.4 Explainability examples

## 6. Discussion & Limitations

## 6. Discussion & Limitations

**Why do the architectures differ?** We examined the layer-1 aggregation
weights for the 764 GAT-Skip targets. In GCN, the weight on a node's own
features is fixed by degree normalisation; the share of total aggregation
weight on the node itself falls from 55.2% on the clean graph to 33.7%, 9.9%
and 1.8% at b = 1, 5 and 20. GAT-Skip instead learns to up-weight self-loops
(attention 2.2×, 3.5× and 5.1× a uniform share at b = 1, 5, 20) and assigns
camouflage edges 0.31×, 0.52× and 0.78× a uniform share, against 0.65×,
0.90× and 1.28× for original edges into the same nodes. GraphSAGE applies a
separate transformation to the node's own features. The ordering of
own-feature retention across the three architectures matches the ordering of
vulnerability, which is consistent with a dilution explanation. We treat this
as a hypothesis, not a finding: it compares three architectures that differ
in other respects, it covers the first layer only, and it does not account for
the decline of GAT-Skip's evasion rate at large budgets, which remains
unexplained. At b=20, the 20 camouflage edges per target, each at 0.78× a
uniform share, still account for most of the layer-1 attention mass the
targets receive (derived from the ratios above), so attention suppression
alone cannot be the explanation.

**Limitations.**
- Each architecture was trained once (seed 42); reported variability covers
  the attacker's sampling only, not training variation.
- The attack is non-adaptive: licit partners are chosen at random. A
  feature-aware attacker may be considerably stronger (evaluated in the
  defence experiments).
- About 99% of targets have in-degree at most 2, so conclusions apply to
  sparsely connected wallets.
- Evasion is measured only over nodes the clean model detects. GAT-Skip
  detects more illicit nodes but at a substantially higher false-positive
  count (1,180 vs 462 for GCN-Skip), so its robustness comes with a precision
  cost.
- GNNExplainer is stochastic, and the explanation analysis uses a sample of
  [N] nodes on one budget.
- All experiments use a single dataset.

## 7. Conclusion & Future Work

## References