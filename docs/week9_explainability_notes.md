# Week 9 — GNNExplainer Proof-of-Concept Findings

- Feature importance varies meaningfully per node (different top-5 indices
  across all 3 sample nodes) -- explainer is producing node-specific signal,
  not a generic default.
- Edge importance varies dramatically by node degree:
  - Low-degree nodes (e.g. node 136279, edge mask sum ~0) rely almost
    entirely on raw node features via the skip connection -- there's
    little graph structure to explain.
  - Higher-degree nodes (e.g. node 136312, edge mask sum ~12) show
    strong, directionally consistent edge importance -- specifically,
    INCOMING edges (other wallets sending funds TO this node) dominate,
    consistent with PyG's message-passing convention (aggregation happens
    at the target of each edge).
- Implication for Week 13: the explainability layer should report BOTH
  feature importance and edge importance per node, since which one
  "explains" a prediction depends on the node's degree -- a single
  explanation format won't fit all nodes well.