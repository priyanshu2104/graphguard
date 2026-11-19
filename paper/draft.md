# GraphGuard: Explainable & Adversarially-Robust Fraud Detection in Cryptocurrency Networks using GNNs

## Abstract
Graph neural networks (GNNs) are widely proposed for detecting illicit Bitcoin
transactions. We evaluate them on the Elliptic dataset with a temporal split
and ask three questions: do they beat simpler models, how easily can an
attacker who adds edges evade them, and can a cheap defence help? Random Forest
(F1 0.816) and XGBoost (0.784) outperform every GNN we trained (GCN-Skip 0.554,
GraphSAGE 0.573, GAT-Skip 0.495, mean over three seeds), and a GCN trained
without edges scores higher than the same GCN with edges (0.642 against 0.554).
We study a camouflage attack that connects illicit transactions to licit ones.
A first, unconstrained attacker evaded about 99% of GCN-Skip's detections, but
its edges crossed time steps, which the dataset never contains, and a one-line
time-consistency check removes it. Against a time-consistent white-box attacker
adding one to five edges per transaction, evasion is 20-32% for GAT-Skip, 54-62%
for GCN-Skip and 71-89% for GraphSAGE, and the order of GCN-Skip and GraphSAGE
depends on the attacker. Removing the least similar edges at inference recovers
about half of the F1 that GCN-Skip loses, without retraining, but its effect
varies by time step, and our adversarial-training attempt failed. Models
without edges are unaffected by edge attacks in our setting because their
neighbourhood features are precomputed. We also provide a demonstration that
explains scores with GNNExplainer and keeps a Merkle-tree log of flag reports.
Our results concern one dataset, an untuned attacker and three training seeds.

## 1. Introduction

Bitcoin transactions are public but pseudonymous, which makes the network
attractive for illicit use and makes its transaction graph a natural object for
financial forensics. The Elliptic dataset [1] labels a subset of Bitcoin
transactions as licit or illicit and provides the graph of coin flows between
them, and graph neural networks (GNNs) [3, 4, 5] are a natural candidate model.
Weber et al. [1] reported, however, that a Random Forest on the provided
features was competitive with a graph convolutional network [check against
Table 1 of [1] before submission]. Whether the graph helps, and what it costs,
is the starting point of this work.

The cost matters because fraudsters adapt. A model that reads a transaction's
neighbourhood can be influenced by changing that neighbourhood, and adversarial
attacks on graph models are well documented for other domains [7, 8, 9]. For a
fraud detector the question is concrete: can an illicit transaction avoid
detection by acquiring licit neighbours, and what can a defender do?

We evaluate GNNs on Elliptic with a temporal split, three training seeds and
illicit-class metrics, and make five contributions:

1. **A temporal benchmark.** Tree ensembles outperform every GNN we trained, and
   a GCN without edges outperforms the same GCN with them, with all features and
   with local features only (Sections 5.1-5.2).
2. **An attack-realism result.** Our first attacker evaded about 99% of
   GCN-Skip's detections, but its edges connected transactions from different
   time steps, which never occurs in the data, and a time-consistency check
   removes them at no cost. Against a time-consistent white-box attacker, evasion
   is 20-32% (GAT-Skip), 54-62% (GCN-Skip) and 71-89% (GraphSAGE); the order of
   GCN-Skip and GraphSAGE depends on the attacker, and attack success varies
   strongly with the target's time step (Section 5.3).
3. **A partial defence and a negative result.** Pruning the least similar edges
   at inference recovers about half of the lost F1 for GCN-Skip without
   retraining, but increases evasion at some time steps; our adversarial-training
   implementation failed (Section 5.4).
4. **A consolidated comparison.** Clean and attacked performance, with false
   positives, for every configuration (Section 5.5). A model without edges is
   best on both counts in our setting, because its neighbourhood features are
   precomputed.
5. **A demonstration.** A Streamlit application that scores transactions,
   explains them, simulates the attack and keeps a tamper-evident log of flag
   reports (Section 6).

We do not claim that GNNs are unsuitable for fraud detection in general. Our
results concern one dataset, one family of attacks that we did not tune, one
defence, and three training seeds; Section 7 lists the limitations.

## 2. Related Work

**Illicit-transaction detection on Bitcoin.** Weber et al. [1] released the
Elliptic dataset, in which nodes are Bitcoin transactions and edges are the flow
of coins between them, and compared logistic regression, Random Forest, an MLP
and a GCN. Elmougy and Liu [2] extended it to Elliptic++, which adds an
address-level graph and further features; we use only the original
transaction graph.

**GNN architectures.** We use the graph convolutional network of Kipf and
Welling [3], GraphSAGE [4] and graph attention networks [5]. Tree ensembles [12,
13] are our non-graph baselines. Grinsztajn et al. [6] find that tree-based
models remain state-of-the-art on medium-sized tabular data and that neural
networks struggle with uninformative features; this is consistent with, but does
not demonstrate, the gap we observe on Elliptic's tabular features.

**Adversarial attacks on graph models.** Zügner et al. [7] introduced targeted
attacks on GCNs that perturb a node's features and graph structure under
unnoticeability constraints. Dai et al. [8] attack graph classifiers and node
classifiers by modifying structure under different levels of attacker knowledge.
Wu et al. [9] observe that adding edges between nodes with different features
plays the key role in the attacks they study and propose defending by removing
such edges. Our similarity pruning applies this idea to Elliptic and examines
it against a defence-aware attacker and across time steps. Unlike these works,
our attack setting is constrained by the data's time structure: a transaction
can only be connected to transactions from its own time step.

**Explanations.** GNNExplainer [10] learns edge and feature masks that explain a
GNN's prediction. We find it stochastic on this task, so we average masks over
runs and report their agreement.

**Tamper-evident logs.** Merkle trees with an inclusion-proof construction are
used for Certificate Transparency [11]. Our report log uses the same
construction; it detects edits relative to a saved root and does not prevent
them.
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

## 3. Dataset and Experimental Setup

**Data.** The Elliptic dataset [1] contains 203,769 transactions (nodes) and
234,355 directed edges over 49 time steps. Of the transactions, 4,545 are
labelled illicit, 42,019 licit and 157,205 unlabelled (about 2%, 21% and 77%).
We verified that none of the 234,355 edges connects transactions from different
time steps. Each transaction has 166 columns including its time step; we use the
other 165 as features, standardised with training-set statistics. By the
dataset's published description the first 93 of these describe the transaction
itself and the last 72 aggregate its one-hop neighbourhood; we assume this
layout and could not verify it independently (a Random Forest scores F1 0.799 on
the first 93 columns and 0.816 on all 165).

**Split and metrics.** Labelled transactions from time steps 1-34 form the
training set (29,894 transactions, 3,462 illicit, 11.58%) and those from steps
35-49 the test set (16,670 transactions, 1,083 illicit, 6.50%). We report
precision, recall, F1 and AUC-PR on the illicit class, with predictions at the
default threshold, and never accuracy.

**Models.** Random Forest and XGBoost: [fill in hyperparameters from
src/models/baselines.py]. GCN, GraphSAGE and GAT are two-layer networks; the
"Skip" variants concatenate the raw input features with the learned
representation before the classifier. Hidden size 64 (32 for GAT-Skip, from our
sweep), dropout 0.3, Adam with learning rate 0.01 [and weight decay: verify in
src/models/train_utils.py], 100 epochs, class-weighted cross-entropy. The
skip connection raised GCN's F1 from 0.439 (standardised features, no skip) to
0.601 (Table 2). A "no-graph" model is the same GCN-Skip trained and evaluated
with an empty edge set.

**Seeds and reporting.** All GNN results use three training seeds (42, 0, 1).
We report mean ± sample standard deviation. The frozen model used in the
single-model experiments is GCN-Skip with seed 42, which has the highest clean
F1 of its three seeds (0.601 against a mean of 0.554). All code, logs and the
scripts that regenerate every table are in the project repository
(github.com/priyanshu2104/graphguard).

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

**Threat model.** Elliptic contains no edges between time steps (none of its
234,355 edges connects nodes from different time steps), so an attacker can
only connect a target to wallets from the same time step. Our main attacks
therefore draw partners from the target's own time step. We assume the
attacker knows which wallets from that period are legitimate and, for the
adaptive variants, has white-box access to the model's output probabilities
and to the pruning threshold. Edges are added in both directions. Edges into
the target require licit transactions to feed the target's outputs, which may
be difficult to arrange in practice, so results measure the effect of an
attacker with this capability, not real-world exposure.

**Attackers.** *Random camouflage* connects each target to b partners sampled
uniformly from the licit wallets of its time step. The *adaptive attacker*
ranks those wallets by the model's illicit probability (lowest first), keeps
those whose feature cosine similarity to the target passes the pruning
threshold (none, for undefended models), and connects the target to b partners
drawn at random from the top M candidates (M = 10, 50 or 200; we report the
strongest variant for each setting). A greedy variant that always takes the
top b is ineffective: every target then shares the same partners, which become
hubs (in-degree about 665) and carry little weight under degree-normalised
aggregation. The attacker is not tuned, so evasion rates are lower bounds.

**Unconstrained variant.** As an upper bound we also report attackers that
draw partners from the training period's licit wallets, so every added edge
crosses time steps. Dropping edges whose endpoints lie in different time steps
removes such an attack entirely at no cost (GCN-Skip F1 0.009 → 0.601,
evasion 98.9% → 0%), so we do not treat it as a realistic threat.

### 4.6 Defense mechanism

**Similarity pruning.** At inference, edges whose endpoints' cosine similarity
(on standardised features) falls below a threshold are removed. The threshold
is a quantile of the similarity distribution of the real edges, so a value of
25% removes the 25% least similar real edges. No retraining is needed.

**Adversarial training (attempted).** We trained GCN-Skip for 100 epochs on
graphs perturbed at every epoch with random camouflage edges (budget 1, 3 or
5), applied to the same fraction (50%) of illicit and licit training nodes so
that the perturbation carries no label information. The partners were drawn
from the training period's licit wallets, so these training edges crossed time
steps.

### 4.7 Explainability

On three correctly detected test nodes the run-to-run rank correlation between
explainer runs was 0.82-0.94, but the five highest feature scores of a node
differed by 0.02 or less, so the output should be read as a set of important
features and not as a strict ranking.

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

### 5.2.1 Time-consistent attacks

**Table 6.** Evasion rate (%) under random camouflage with time-consistent
partners (seed-42 models; mean over five sampling seeds).

| Model | b=1 | b=3 | b=5 |
|---|---|---|---|
| GCN-Skip | 19.3 | 32.6 | 42.0 |
| GraphSAGE | 15.0 | 18.3 | 20.1 |
| GAT-Skip | 5.8 | 5.5 | 3.7 |

**Table 7.** Evasion rate (%) under the time-consistent adaptive attacker
(strongest of three variants per cell), mean ± sample std over three training
seeds, one attack-sampling seed each.

| Model | Clean F1 | b=1 | b=3 | b=5 |
|---|---|---|---|---|
| GCN-Skip | 0.554 ± 0.045 | 54.0 ± 5.8 | 62.4 ± 8.0 | 60.2 ± 7.9 |
| GraphSAGE | 0.573 ± 0.014 | 71.2 ± 2.7 | 86.2 ± 3.3 | 89.1 ± 2.5 |
| GAT-Skip | 0.495 ± 0.011 | 19.7 ± 4.0 | 26.8 ± 9.6 | 32.2 ± 11.2 |

The adaptive attacker is far stronger than random camouflage (for example,
GraphSAGE at b=5: 89% against 20%). The ordering GAT-Skip < GCN-Skip <
GraphSAGE holds at every budget in all three training seeds, with disjoint
ranges. Under the unconstrained attacker GCN-Skip was at least as fragile as
GraphSAGE, so we do not claim an architecture ranking independent of the
attacker. The frozen GAT-Skip model (seed 42) is the most resistant of its
three seeds (19.4% evasion at b=5, against 37.6% and 39.7%). No model is
robust: even the most resistant loses about a third of its detected targets at
b=5.

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

## 5.3 Explainability Under Attack

We compared GNNExplainer feature-importance vectors for the frozen GCN-Skip
model before and after a budget-3 random camouflage attack with time-consistent
partners, on 150 randomly sampled correctly detected illicit test nodes
(55 evaded, 95 did not). Explanations were computed on each node's 3-hop
subgraph, which we verified reproduces the full-graph logits. The explainer is
stochastic: explaining the same node twice on the clean graph gives only
1.40/5 overlap in the top-5 features, so we report Spearman rank correlation
across all 165 features instead. Its run-to-run baseline is ρ = 0.84 on the
clean graph and ρ = 0.81 (evaded) and 0.84 (not evaded) on the attacked graph.

Because the predicted class of an evaded node changes, we explain the illicit
class on both graphs. Clean-versus-attacked correlation was ρ = 0.63 for evaded
nodes and ρ = 0.67 for nodes that stayed detected (Mann–Whitney p = 0.0087; for
62% of evaded/not-evaded pairs the evaded node had the lower correlation, where
50% would mean no difference). The difference also holds among nodes with small
pre-attack neighbourhoods (p = 0.019; 55 evaded and 88 not evaded). For both
groups the correlation is well below the explainer's own baseline, so the attack
changes the explanations whether or not it changes the prediction, and evaded
nodes change slightly more. Evaded nodes are the ones whose output moved most,
so this does not show that the attack alters the model's reasoning specifically.
Under the unconstrained attacker the corresponding values were 0.61 and 0.65
(p = 0.0017). We did not test whether non-adversarial perturbations of the same
size produce the same shift.

## 5.4 Defence Evaluation

**Table 8b.** Pruning level against the time-consistent adaptive attacker,
GCN-Skip, mean ± sample std over three training seeds. Worst-case F1 is the
minimum over budgets 3 and 5 and five partner-selection variants
(M = 10, 25, 50, 100, 200).

| Real edges pruned | Clean F1 | Clean false positives | Worst-case F1 |
|---|---|---|---|
| 0% | 0.554 ± 0.045 | 675 | 0.251 ± 0.064 |
| 5% | 0.550 ± 0.046 | 678 | 0.247 ± 0.068 |
| 10% | 0.550 ± 0.046 | 677 | 0.257 ± 0.055 |
| 25% | 0.555 ± 0.043 | 648 | 0.397 ± 0.042 |
| 50% | 0.548 ± 0.040 | 621 | 0.357 ± 0.049 |
| 75% | 0.542 ± 0.034 | 638 | 0.437 ± 0.014 |

Pruning below 25% had no effect. At 25% and above, worst-case F1 rose to
0.36-0.44 with clean F1 within 0.012 of the unpruned model. Differences among
25%, 50% and 75% are within about one standard deviation across three seeds, so
the best level is not sharply determined. We used 25%, the smallest effective
level and the one selected by our rule (highest mean worst-case F1 among levels
whose clean F1 is within 0.01 of unpruned; 75% scored higher but its clean F1
was 0.012 below). The selection used test results because we had no validation
split. At 25%, pruning recovers about half of the F1 lost to the attack
(0.251 → 0.397, against 0.554 clean) but does not neutralise it.

**Adversarial training (negative result).** As implemented, adversarial
training failed. Over three training seeds, clean F1 fell to 0.302 ± 0.020
(undefended 0.601) with 3,953 ± 787 false positives (undefended 462), and
clean AUC-PR fell to 0.356 (undefended, seed 42: 0.611). Evasion under random
unconstrained camouflage was lower (4-13% at budgets 1-5, against 27-67%
undefended) but only because the model flagged many licit nodes: its AUC-PR
under attack remained below the undefended model's (0.169 against 0.226 at
b=5). We report this as a negative result of this implementation (100 epochs,
a new random graph at every step, final training loss about 0.20 against 0.087
undefended, training edges that crossed time steps), not as evidence about
adversarial training in general.

## 5.5 Does the Graph Help?

To check whether message passing contributes anything on this dataset, we
trained GCN-Skip with and without edges, using either all 165 features or only
the 93 local features (the remaining 72 are aggregated over each transaction's
one-hop neighbours; Weber et al., 2019). Table 9 reports clean test
performance over three training seeds.

**Table 9.**

| Features | Edges | F1 | AUC-PR |
|---|---|---|---|
| All 165 | with graph | 0.554 ± 0.045 | 0.611 (seed 42 only) |
| All 165 | without | 0.642 ± 0.023 | 0.626 ± 0.033 |
| Local 93 | with graph | 0.350 ± 0.018 | 0.400 ± 0.016 |
| Local 93 | without | 0.442 ± 0.019 | 0.582 ± 0.052 |

The model without edges scored higher than the same model with edges in both
settings, and the per-seed F1 ranges do not overlap. With all features the
AUC-PR difference is small (0.580 ± 0.034 with the graph, 0.626 ± 0.033
without; about 1.4 standard deviations), while the false-positive count at the
default threshold differs by a factor of three (675 ± 268 against 219 ± 44), so
much of the F1 gap reflects the operating point and not ranking quality. With
local features only, the AUC-PR gap is large (0.400 against 0.582).

Removing the 72 columns that, by the dataset's published layout, are aggregated
neighbourhood features lowered the no-graph model's F1 by 0.20 (0.642 to
0.442), so GCN-Skip relies on them. A Random Forest does not: it scores F1
0.799 on the first 93 columns and 0.816 on all 165 (0.584 on the last 72 alone),
so the dependence is specific to the neural model. We could not verify the
column layout independently. Message passing over the local features did not
recover the information in the aggregated columns: the graph model on local
features was worse than the model without edges. We therefore find no evidence
that the graph helps this architecture on this dataset, in either feature
setting. The hyperparameters were selected on graph-based models, so the
comparison does not favour the no-graph variants. GCN-Skip without edges is
itself well below Random Forest (0.642 against 0.816 on all features, 0.442
against 0.799 on local features), so the comparison is within a weaker model
family and may not carry over to stronger tabular learners. With the graph, the
local-features model remained highly vulnerable (99.0 ± 0.4% evasion at b=3
under the unconstrained adaptive attacker). We tested only GCN-Skip in this
comparison.

**Label structure.** Of the 36,624 edges whose endpoints are both labelled, 93%
connect two licit nodes. Among illicit nodes with at least one labelled
neighbour, 51% of those neighbours are illicit, against a base rate of 9.8%
among labelled nodes, so illicit nodes cluster. We did not establish why
message passing nevertheless fails to help.

## 5.6 Final Comparison

**Table 10.** Clean and attacked performance on the time-consistent threat
model, mean ± sample std over three training seeds (tree models: single run).
Worst-case F1 is the minimum over the adaptive attacker's settings
(b = 3, 5; M = 10, 50, 200); the random-camouflage column is b=5, with
false positives in brackets.

| Model | Clean F1 | Clean AUC-PR | Clean FP | Random b=5 F1 (FP) | Worst-case adaptive F1 | Evasion b=5 (%) |
|---|---|---|---|---|---|---|
| Random Forest | 0.816 | 0.790 | 7 | = clean | = clean | — |
| XGBoost | 0.784 | 0.802 | 149 | = clean | = clean | — |
| GCN-Skip | 0.554 ± 0.045 | 0.580 ± 0.034 | 675 ± 268 | 0.344 ± 0.039 (834 ± 307) | 0.254 ± 0.066 | 60.2 ± 7.9 |
| GraphSAGE | 0.573 ± 0.014 | 0.593 ± 0.045 | 604 ± 30 | 0.459 ± 0.027 (630 ± 24) | 0.085 ± 0.022 | 89.1 ± 2.5 |
| GAT-Skip | 0.495 ± 0.011 | 0.545 ± 0.030 | 1,221 ± 160 | 0.379 ± 0.006 (1,940 ± 179) | 0.363 ± 0.055 | 32.2 ± 11.2 |
| GCN-Skip + pruning (25%) | 0.555 ± 0.043 | 0.570 ± 0.034 | 648 ± 259 | 0.399 ± 0.041 (793 ± 294) | 0.397 ± 0.042 | 36.9 ± 2.6 |
| GCN-Skip, no graph | 0.642 ± 0.023 | 0.626 ± 0.033 | 219 ± 44 | = clean | = clean | — |

A model trained without edges has higher clean F1 (0.642 ± 0.023) and higher
attacked F1 than every graph-based configuration we tested, with about a third
of the false positives (219 ± 44 against 675 ± 268 for GCN-Skip). Random
Forest and XGBoost are unaffected by edge attacks only because their
neighbourhood features are precomputed and fixed (Section 6). The most
resistant graph model, GAT-Skip, still loses 32.2 ± 11.2% of its detected
targets at b=5 to the adaptive attacker. Pruned GCN-Skip has a worst-case F1
(0.397 ± 0.042) comparable to GAT-Skip's (0.363 ± 0.055; the spreads overlap),
with higher clean F1 (0.555 against 0.495) and about half the false positives
(648 against 1,221), but lower recall under attack (0.396 against 0.476).

Random camouflage with time-consistent partners also raised false positives
for GAT-Skip (from 1,221 ± 160 to 1,940 ± 179 at b=5) and GCN-Skip (675 ± 268
to 834 ± 307), but not materially for GraphSAGE (604 ± 30 to 630 ± 24). This
explains why GAT-Skip's F1 falls from 0.495 to 0.379 under random camouflage
although only 3.7% of the seed-42 model's targets evade (Table 6). The partners
are test-period wallets, so connecting them to illicit targets can cause them
to be flagged; we did not verify this node by node. We therefore report evasion
and these collateral false positives as separate effects. At the worst-case
adaptive setting, which uses the most licit-looking partners, false positives
were within 3.2% of the clean count for every model.

**Table 11.** Evasion rate (%) by time step of the target, frozen GCN-Skip
(seed 42), adaptive attacker with b=3. A: undefended model, unconstrained
attacker. B: undefended model, similarity-constrained attacker. C: pruned model
(25%), constrained attacker. D: pruned model, unconstrained attacker. Each row
uses the stronger of M = 50 and M = 200 for that step.

| Time step | Targets | A | B | C | D |
|---|---|---|---|---|---|
| 35 | 159 | 48.4 | 42.8 | 42.1 | 17.6 |
| 36 | 27 | 100.0 | 96.3 | 0.0 | 0.0 |
| 37 | 22 | 100.0 | 90.9 | 9.1 | 4.5 |
| 38 | 91 | 100.0 | 95.6 | 6.6 | 13.2 |
| 39 | 66 | 54.5 | 51.5 | 13.6 | 7.6 |
| 40 | 58 | 74.1 | 56.9 | 13.8 | 5.2 |
| 41 | 79 | 41.8 | 46.8 | 57.0 | 36.7 |
| 42 | 158 | 43.7 | 58.2 | 61.4 | 38.6 |
| 47-49 | 3 | 100.0 | 100.0 | 100.0 | 100.0 |
| All targets | 663 | 53.4 | 52.0 | 35.7 | 21.3 |

The all-targets row uses the single M that is strongest across all targets, so
it is lower than the target-weighted mean of the rows for A and B (60.5% and
60.3%).

**Dependence on time step.** Evasion varied strongly with the target's time
step (Table 11). The undefended model was evaded in all targets at steps 36-38
(140 transactions) under the unconstrained attacker, but in 42-74% of targets
at the other steps with more than three targets. Pruning cut evasion sharply at
steps 36-40 (264 targets, pooled: 82.9% to 9.5%, comparing A with C) but raised
it at steps 41-42 (237 targets: 43.1% to 59.9%); at step 35 the difference was
small (48.4% to 42.1%). The average improvement from pruning (53.4% to 35.7%
over all targets) is therefore driven by steps 36-40. Against the undefended
model the two attacker families were about equally strong overall (A 53.4%,
B 52.0%), but the constrained one was stronger at steps 41-42 and the
unconstrained one at steps 35-40. There are no targets in steps 43-46. These
results are for one model and one budget, and we do not have an explanation for
the dependence on time step.


### 6. Discussion & Limitations

## 6. Discussion & Limitations

**Why do the architectures differ?** We do not have an explanation we can
support. Under the time-consistent adaptive attacker, GAT-Skip is the most
resistant (20-32% evasion at b = 1-5), followed by GCN-Skip (54-62%) and
GraphSAGE (71-89%); the order of GCN-Skip and GraphSAGE reverses under the
unconstrained attacker. We examined whether the share of aggregation weight
kept on a node's own features accounts for the differences. With unconstrained
camouflage edges, GCN's structural self-share falls from 55% on the clean
graph to 2% at b=20, while GAT-Skip learns to weight its self-loops 2-5 times
a uniform share and gives camouflage edges about 0.5-0.6 times the weight of
real edges. That is consistent with GAT-Skip's relative resistance, but it does
not explain why GraphSAGE, which has a separate transformation for a node's own
features, is more fragile than GCN-Skip under the time-consistent attacker, so
we do not offer it as the mechanism.

**Immunity of feature-only models depends on frozen features.** Elliptic's 165
features include 72 aggregated features that summarise each transaction's
one-hop neighbourhood (Weber et al., 2019). These are precomputed and fixed in
our experiments, so inserting edges changes the GNN's message passing but not
the inputs of Random Forest, XGBoost or the no-graph model. In a deployment
that recomputed neighbourhood features from the live graph, an attacker's edges
would also change those features, and the immunity we report would not hold. We
did not test this.

**Limitations.**
- The attacker is not tuned (the best partner-selection variant differed
  across models), so evasion rates are lower bounds. It also assumes white-box
  access and knowledge of which same-period wallets are legitimate.
- Most main experiments use one trained model per architecture (seed 42); the
  adaptive-attack ranking, the pruning evaluation and the graph/no-graph
  comparison were checked over three training seeds, which is only a crude
  estimate of spread.
- The frozen GCN-Skip model (seed 42) is the best of its three seeds on clean
  F1 (0.601, against a mean of 0.554), and the frozen GAT-Skip model is the most
  resistant of its three seeds.
- The explanation analysis (Section 5.3) was measured under the unconstrained
  attacker, and it changes the explainer's target class for evaded nodes
  (addressed by a fixed-target control). We did not test whether
  non-adversarial perturbations of the same size produce the same shift.
- Pruning and the graph/no-graph comparison were tested on GCN-Skip only.
- Evasion is measured only over nodes the clean model detects. GAT-Skip detects more illicit nodes but at a higher false-positive count on clean data (1,221 ± 160 against 675 ± 268 for GCN-Skip).
- Adversarial training was tested in one implementation, with cross-time
  training edges.
- All experiments use a single dataset.
- Clean false-positive counts vary widely across training seeds (GCN-Skip: 462, 586, 976), and the frozen seed-42 model has the fewest, so operating-point comparisons based on one model are unreliable.
- Neural models without edges are well below tree ensembles on this data (GCN-Skip without edges 0.642 against Random Forest 0.816), so conclusions about the graph apply to this model family.
- Attack success and the effect of pruning depend strongly on the time step
  (Table 11); pruning increased evasion at steps 41-42. We examined one model
  and one budget (b=3) and cannot explain the dependence.
- The undefended and pruned models were attacked by different attacker families
  (unconstrained and similarity-constrained). At b=3 the families were about
  equally strong against the undefended model overall, but we did not compare
  them at other budgets.

## 7. Demonstration System

We provide a local Streamlit application that scores transactions from the
Elliptic dataset under three configurations (a GCN-Skip model trained without
edges, the undefended GCN-Skip, and GCN-Skip with similarity pruning), explains
a score with GNNExplainer, simulates a time-consistent camouflage attack on a
single transaction, and keeps a log of flag reports. The application browses
the dataset: it cannot score transactions outside it, because the features are
anonymised and precomputed, and it does not use live blockchain data.

Explanations are averaged over three explainer runs and returned with their
rank correlation, since the explainer is stochastic; the top feature scores of
a transaction are typically within 0.02 of each other, so the application shows
them as a set. Flag reports are appended to a log whose Merkle root (SHA-256,
RFC 6962 construction) commits to every record and its order; an edit, removal
or reordering of an earlier record changes the root. This makes tampering
evident relative to a root saved elsewhere. It is not tamper-proof, and we did
not implement anchoring of the root on a public ledger.

## 8. Conclusion

On the Elliptic dataset, tree ensembles outperform every GNN we trained, and a
GCN trained without edges outperforms the same GCN with them. A first attacker
that evaded almost all of GCN-Skip's detections was infeasible: its edges crossed
time steps, and a time-consistency check removes it. Against a time-consistent
white-box attacker, GAT-Skip is the most resistant of the three GNNs and
GraphSAGE the least, though none is robust and the order of GCN-Skip and
GraphSAGE depends on the attacker. Similarity pruning recovers about half of the
F1 that GCN-Skip loses, without retraining, but raises evasion at some time
steps, and adversarial training failed in our implementation. A model without
edges was best on clean data and under attack in our setting, an advantage that
depends on its neighbourhood features being precomputed. Future work includes
attackers optimised by gradient or search, recomputing neighbourhood features
from the attacked graph, pruning for other architectures, validation-based
selection of the pruning level, and other datasets.

## References
[1] M. Weber, G. Domeniconi, J. Chen, D. K. I. Weidele, C. Bellei, T. Robinson, C. E. Leiserson. Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics. KDD '19 Workshop on Anomaly Detection in Finance, 2019. arXiv:1908.02591.

[2] Y. Elmougy, L. Liu. Demystifying Fraudulent Transactions and Illicit Nodes in the Bitcoin Network for Financial Forensics. KDD '23, 2023. doi:10.1145/3580305.3599803.

[3] T. N. Kipf, M. Welling. Semi-Supervised Classification with Graph Convolutional Networks. ICLR 2017.

[4] W. L. Hamilton, R. Ying, J. Leskovec. Inductive Representation Learning on Large Graphs. NeurIPS 2017.

[5] P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, Y. Bengio. Graph Attention Networks. ICLR 2018.

[6] L. Grinsztajn, E. Oyallon, G. Varoquaux. Why do tree-based models still outperform deep learning on typical tabular data? NeurIPS 2022 Datasets and Benchmarks. arXiv:2207.08815.

[7] D. Zügner, A. Akbarnejad, S. Günnemann. Adversarial Attacks on Neural Networks for Graph Data. KDD 2018, pp. 2847-2856. doi:10.1145/3219819.3220078.

[8] H. Dai, H. Li, T. Tian, X. Huang, L. Wang, J. Zhu, L. Song. Adversarial Attack on Graph Structured Data. ICML 2018, PMLR 80:1115-1124.

[9] H. Wu, C. Wang, Y. Tyshetskiy, A. Docherty, K. Lu, L. Zhu. Adversarial Examples for Graph Data: Deep Insights into Attack and Defense. IJCAI 2019, pp. 4816-4823. doi:10.24963/ijcai.2019/669.

[10] 	R. Ying, D. Bourgeois, J. You, M. Zitnik, J. Leskovec. GNNExplainer: Generating Explanations for Graph Neural Networks. NeurIPS 2019.

[11] B. Laurie, A. Langley, E. Kasper. Certificate Transparency. RFC 6962, 2013.

[12] L. Breiman. Random Forests. Machine Learning 45(1):5-32, 2001.

[13] 	T. Chen, C. Guestrin. XGBoost: A Scalable Tree Boosting System. KDD 2016.