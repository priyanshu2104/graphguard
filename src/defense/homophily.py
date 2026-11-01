"""Label homophily on Elliptic: what do the labelled neighbours of illicit nodes look like?"""
import torch
from src.utils.splits import load_graph


def main():
    data = load_graph()
    y, (src, dst) = data.y, data.edge_index
    n = data.num_nodes

    both = (y[src] != -1) & (y[dst] != -1)
    a, b = y[src][both], y[dst][both]
    print(f"edges with both endpoints labelled: {int(both.sum())} of {data.edge_index.shape[1]}")
    for name, m in [("illicit -> illicit", (a == 1) & (b == 1)), ("illicit -> licit", (a == 1) & (b == 0)),
                    ("licit -> illicit", (a == 0) & (b == 1)), ("licit -> licit", (a == 0) & (b == 0))]:
        print(f"  {name}: {int(m.sum())} ({m.float().mean():.1%})")

    ill_nb, lic_nb = torch.zeros(n), torch.zeros(n)
    for s, d in [(src, dst), (dst, src)]:              # neighbours in either direction
        ill_nb.index_add_(0, d, (y[s] == 1).float())
        lic_nb.index_add_(0, d, (y[s] == 0).float())
    m = y == 1
    tot = (ill_nb + lic_nb)[m]
    has = tot > 0
    base = (y == 1).sum().item() / ((y == 1).sum().item() + (y == 0).sum().item())
    print(f"\nillicit nodes with at least one labelled neighbour: {int(has.sum())} of {int(m.sum())}")
    print(f"mean share of those neighbours that are illicit: {(ill_nb[m][has] / tot[has]).mean():.1%} "
          f"(base rate of illicit among labelled nodes: {base:.1%})")


if __name__ == "__main__":
    main()