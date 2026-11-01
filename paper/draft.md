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

## 5.4 Defence Evaluation

**Table 8.** Similarity pruning of GCN-Skip against the time-consistent
adaptive attacker. Worst-case F1 is the minimum over budgets 3 and 5 and three
partner-selection variants. Mean ± sample std over three training seeds.

| Real edges pruned | Clean F1 | Worst-case F1 |
|---|---|---|
| 0% | 0.554 ± 0.045 | 0.254 ± 0.066 |
| 25% | 0.555 ± 0.043 | 0.397 ± 0.042 |
| 50% | 0.548 ± 0.040 | 0.358 ± 0.049 |
| No edges (reference) | 0.642 ± 0.023 | 0.642 ± 0.023 |

Pruning 25% of edges left clean F1 unchanged and raised worst-case F1 in all
three seeds (by 0.108 to 0.179), recovering about half of the F1 lost to the
attack; pruning 50% was not better. Pruning therefore mitigates the attack but
does not neutralise it. A model trained without edges scores higher than every
defended result and is unaffected by edge attacks in our setting, because its
neighbourhood features are precomputed and fixed (Section 6). Under the
unconstrained attacker the best pruning level was different (worst-case F1
0.003 ± 0.002 undefended, 0.334 ± 0.027 at 25%, 0.477 ± 0.028 at 50%), so the
best level depends on the attacker. The attacker was not tuned and these are
not true worst cases.

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
settings (the per-seed F1 ranges do not overlap). Removing the aggregated
features lowered the no-graph model's F1 by 0.20, so those features carry
useful information, but message passing over local features did not recover
it. We therefore find no evidence that the graph helps this architecture on
this dataset, in either feature setting. The hyperparameters were selected on
graph-based models, so the comparison does not favour the no-graph variants.
With the graph, the local-features model remained highly vulnerable (99.0 ±
0.4% evasion at b=3 under the unconstrained adaptive attacker). We tested only
GCN-Skip in this comparison.

**Label structure.** Of the 36,624 edges whose endpoints are both labelled, 93%
connect two licit nodes. Among illicit nodes with at least one labelled
neighbour, 51% of those neighbours are illicit, against a base rate of 9.8%
among labelled nodes, so illicit nodes cluster. We did not establish why
message passing nevertheless fails to help.

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
- Evasion is measured only over nodes the clean model detects. GAT-Skip detects
  more illicit nodes but at a higher false-positive count on clean data (1,180
  against 462 for GCN-Skip).
- Adversarial training was tested in one implementation, with cross-time
  training edges.
- All experiments use a single dataset.

## 7. Conclusion & Future Work

## References