"""
Identifies which illicit test nodes are valid attack targets: nodes the
CLEAN (unattacked) frozen model currently classifies correctly as illicit.
Attacking a node the model already gets wrong is meaningless -- there's
nothing to evade.
"""
import torch


@torch.no_grad()
def get_attack_targets(model, data, test_mask):
    """
    Returns:
        target_node_indices: global node indices of illicit test nodes
                              correctly classified by the clean model
        clean_preds: model predictions on all nodes (for reuse, avoids
                     re-running inference)
        clean_probs: predicted illicit-class probabilities on all nodes
    """
    model.eval()
    out = model(data.x, data.edge_index)
    probs = torch.softmax(out, dim=1)[:, 1]
    preds = out.argmax(dim=1)

    test_indices = test_mask.nonzero(as_tuple=True)[0]
    y_test = data.y[test_indices]
    preds_test = preds[test_indices]

    correctly_flagged = (y_test == 1) & (preds_test == 1)
    target_node_indices = test_indices[correctly_flagged]

    print(f"Illicit nodes in test set: {(y_test == 1).sum().item()}")
    print(f"Correctly flagged by clean model: {len(target_node_indices)} "
          f"({len(target_node_indices) / (y_test == 1).sum().item():.1%} of illicit test nodes)")

    return target_node_indices, preds, probs