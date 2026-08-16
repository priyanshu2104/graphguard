"""
GCN and GraphSAGE model definitions.
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import GATConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=2, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # raw logits -- softmax/argmax applied at eval time


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=2, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GCNSkip(torch.nn.Module):
    """
    GCN with a skip connection: concatenates the original node features with
    the GCN's learned representation before the final classification layer.
    This is the variant Weber et al. found closes most of the gap to RF on
    Elliptic -- pure graph-averaging alone tends to dilute the strong local
    feature signal this dataset has.
    """
    def __init__(self, in_channels, hidden_channels=64, out_channels=2, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels + in_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)

        combined = torch.cat([h, x], dim=1)  # skip connection: raw features + learned graph features
        out = self.classifier(combined)
        return out

class GATSkip(torch.nn.Module):
    """
    GAT with a skip connection (raw features concatenated with the learned
    representation before classification), following the same pattern that
    helped GCN in Week 4. Attention heads let the model learn to weight
    neighbors differently -- useful later in Week 13 when you visualize
    attention weights as part of the explainability layer.
    """
    def __init__(self, in_channels, hidden_channels=64, out_channels=2, heads=4, dropout=0.3):
        super().__init__()
        # First layer: multiple attention heads, concatenated -> hidden_channels * heads output
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # Second layer: average the heads instead of concatenating (concat=False), output hidden_channels
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout)
        self.classifier = torch.nn.Linear(hidden_channels + in_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.elu(h)  # GAT papers conventionally use ELU, not ReLU
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.elu(h)

        combined = torch.cat([h, x], dim=1)
        out = self.classifier(combined)
        return out