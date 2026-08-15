"""
Reusable training loop skeleton for GNN models (used from Week 4 onward).
Not run this week -- just scaffolded so Week 4 starts from a working harness.
"""
import torch
import torch.nn.functional as F


def train_one_epoch(model, data, optimizer, train_mask):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
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


def run_training(model, data, train_mask, test_mask, epochs=100, lr=0.01, weight_decay=5e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, data, optimizer, train_mask)
        if epoch % 20 == 0 or epoch == epochs:
            print(f"Epoch {epoch:3d} | Loss: {loss:.4f}")

    preds, probs, y_true = evaluate(model, data, test_mask)
    return preds, probs, y_true