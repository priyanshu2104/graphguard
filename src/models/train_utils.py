"""
Reusable training loop for GNN models (GCN, GraphSAGE, GAT).
"""
import torch
import torch.nn.functional as F


def compute_class_weights(data, train_mask):
    """
    Inverse-frequency class weights from the TRAIN split only (never look at
    test labels when computing anything used during training).
    """
    y_train = data.y[train_mask]
    n_licit = (y_train == 0).sum().item()
    n_illicit = (y_train == 1).sum().item()
    total = n_licit + n_illicit

    # Inverse frequency: rarer class gets a higher weight
    weight_licit = total / (2 * n_licit)
    weight_illicit = total / (2 * n_illicit)

    return torch.tensor([weight_licit, weight_illicit], dtype=torch.float)


def train_one_epoch(model, data, optimizer, train_mask, class_weights=None):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask], weight=class_weights)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    probs = F.softmax(out, dim=1)[:, 1]  # probability of illicit class
    preds = out.argmax(dim=1)
    return preds[mask].numpy(), probs[mask].numpy(), data.y[mask].numpy()


def run_training(model, data, train_mask, test_mask, epochs=100, lr=0.01, weight_decay=5e-4, use_class_weights=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_weights = compute_class_weights(data, train_mask) if use_class_weights else None

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, data, optimizer, train_mask, class_weights)
        if epoch % 20 == 0 or epoch == epochs:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f}")

    preds, probs, y_true = evaluate(model, data, test_mask)
    return preds, probs, y_true