"""
Environment sanity check for GraphGuard.
Run: python src/utils/check_env.py
"""
import torch
import torch_geometric
from torch_geometric.data import Data
import sklearn
import xgboost
import networkx as nx
import pandas as pd

print("torch:", torch.__version__)
print("torch_geometric:", torch_geometric.__version__)
print("scikit-learn:", sklearn.__version__)
print("xgboost:", xgboost.__version__)
print("networkx:", nx.__version__)
print("pandas:", pd.__version__)

# Build a tiny 4-node toy graph to confirm PyG actually works end-to-end
edge_index = torch.tensor([[0, 1, 2, 3],
                            [1, 0, 3, 2]], dtype=torch.long)
x = torch.rand((4, 8))  # 4 nodes, 8 features each
data = Data(x=x, edge_index=edge_index)

print("\nToy graph created successfully:")
print(data)
assert data.num_nodes == 4
assert data.num_edges == 4
print("\n✅ Environment OK — all imports working, PyG Data object builds correctly.")