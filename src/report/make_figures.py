"""
Builds the paper's figures from the logs and CSVs in results/ (matplotlib only).

    python -m src.report.make_figures                 # writes paper/figures/fig*.png and .pdf
    python -m src.report.make_figures --results DIR --out DIR

A figure is skipped (with a message) if its input is missing. Every number plotted is parsed from a
log or CSV, so the figures cannot drift away from the tables.
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
COLORS = {"GCN-Skip": "#1f77b4", "GraphSAGE": "#ff7f0e", "GAT-Skip": "#2ca02c",
          "GCN-Skip + pruning (25%)": "#9467bd", "GCN-Skip, no graph": "#7f7f7f",
          "Random Forest": "#8c564b", "XGBoost": "#bcbd22"}
ARCHS = ["GCN-Skip", "GraphSAGE", "GAT-Skip"]

SEED_CHECK = re.compile(r"^(GCN-Skip|GraphSAGE|GAT-Skip) seed (\d+): clean F1 ([\d.]+) \| strongest-variant evasion "
                        r"b=1: ([\d.]+)% b=3: ([\d.]+)% b=5: ([\d.]+)%", re.M)
PRUNE = re.compile(r"seed (\d+) q=([\d.]+): clean F1 ([\d.]+) \(FP (\d+)\) \| worst adaptive F1 ([\d.]+) \(FP (\d+)\)")
STEP = re.compile(r"^\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$", re.M)


def read(folder, name):
    path = Path(folder) / name
    if not path.exists():
        print(f"skipped: {name} not found")
        return None
    return path.read_text(encoding="utf-8")


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}")


def mean_std(values):
    v = np.asarray(values, dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


# ----------------------------------------------------------------------------------------------------
def fig_overview(out):
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=fc, ec="#444444", lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color="#444444", lw=1.2))

    top, bottom = 2.35, 0.45
    box(0.1, top, 2.4, 1.2, "Elliptic data\n203,769 transactions\n234,355 edges\n49 time steps", "#e8f0fa")
    box(3.1, top, 2.7, 1.2, "Temporal split\ntrain: steps 1-34\ntest: steps 35-49\nstandardise on train", "#e8f0fa")
    box(6.4, top, 4.0, 1.2, "Models (3 training seeds)\nRF, XGBoost | GCN, GraphSAGE, GAT (+skip)\nGCN-Skip without edges", "#e8f0fa")
    arrow(2.5, top + 0.6, 3.1, top + 0.6)
    arrow(5.8, top + 0.6, 6.4, top + 0.6)

    box(0.1, bottom, 2.4, 1.2, "Camouflage attack\nadd edges to licit\ntransactions of the\nsame time step", "#fdecea")
    box(3.1, bottom, 2.7, 1.2, "Defence\nsimilarity pruning at\ninference (adversarial\ntraining: negative result)", "#eaf5ea")
    box(6.4, bottom, 1.9, 1.2, "Explanation\nGNNExplainer on\n3-hop subgraph", "#fff6e0")
    box(8.5, bottom, 1.9, 1.2, "Demo + log\nStreamlit, Merkle\ntree of reports", "#fff6e0")
    arrow(2.5, bottom + 0.6, 3.1, bottom + 0.6)
    arrow(5.8, bottom + 0.6, 6.4, bottom + 0.6)
    arrow(8.3, bottom + 0.6, 8.5, bottom + 0.6)
    # elbow connector: models -> attack (the trained models are what gets attacked, defended and explained)
    ax.plot([8.4, 8.4, 1.3], [top, 1.95, 1.95], color="#444444", lw=1.2)
    arrow(1.3, 1.95, 1.3, bottom + 1.2)
    ax.text(4.8, 2.08, "trained models are attacked, defended and explained", ha="center", fontsize=8,
            style="italic", color="#444444")
    save(fig, out, "fig1_overview")


def fig_attack_budget(folder, out):
    seeds = read(folder, "seed_check_sametime_log.txt")
    single = read(folder, "sametime_eval_log.txt")
    if seeds is None:
        return
    adaptive = {}
    for arch, _, _, e1, e3, e5 in SEED_CHECK.findall(seeds):
        adaptive.setdefault(arch, {1: [], 3: [], 5: []})
        for b, e in ((1, e1), (3, e3), (5, e5)):
            adaptive[arch][b].append(float(e))
    rand, current = {}, None
    for line in (single or "").splitlines():
        head = re.match(r"^(GCN-Skip|GraphSAGE|GAT-Skip): clean F1", line)
        if head:
            current = head.group(1)
        row = re.match(r"^\s+b=(\d) \| random: evasion ([\d.]+)%", line)
        if row and current:
            rand.setdefault(current, {})[int(row.group(1))] = float(row.group(2))

    budgets = [1, 3, 5]
    fig, (ax_a, ax_r) = plt.subplots(1, 2, figsize=(8.2, 3.5), sharey=True)
    for arch in ARCHS:
        if arch in adaptive:
            stats = [mean_std(adaptive[arch][b]) for b in budgets]
            ax_a.errorbar(budgets, [s[0] for s in stats], yerr=[s[1] for s in stats], marker="o", capsize=3,
                          color=COLORS[arch], label=arch)
        if arch in rand:
            ax_r.plot(budgets, [rand[arch][b] for b in budgets], marker="s", color=COLORS[arch], label=arch)
    ax_a.set_title("Adaptive attacker (3 training seeds, mean ± std)", fontsize=9)
    ax_r.set_title("Random camouflage (seed-42 models)", fontsize=9)
    for ax in (ax_a, ax_r):
        ax.set_xlabel("Edges added per transaction (b)")
        ax.set_xticks(budgets)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
    ax_a.set_ylabel("Evasion of detected illicit transactions (%)")
    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.17))
    save(fig, out, "fig2_attack_budget")


def fig_time_step(folder, out):
    text = read(folder, "by_timestep_log.txt")
    if text is None:
        return
    rows = [(int(s), int(n), float(a), float(b), float(c)) for s, n, a, b, c, _ in STEP.findall(text) if int(n) > 3]
    if not rows:
        print("skipped: no per-step rows found")
        return
    steps = [f"{s}\n(n={n})" for s, n, *_ in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    width = 0.27
    ax.bar(x - width, [r[2] for r in rows], width, label="Undefended, unconstrained attacker", color="#1f77b4")
    ax.bar(x, [r[3] for r in rows], width, label="Undefended, constrained attacker", color="#9ecae1")
    ax.bar(x + width, [r[4] for r in rows], width, label="Pruned (25%), constrained attacker", color="#9467bd")
    ax.set_xticks(x)
    ax.set_xticklabels(steps, fontsize=8)
    ax.set_xlabel("Time step of the target transaction")
    ax.set_ylabel("Evasion (%), b=3")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
    save(fig, out, "fig3_time_step")


def fig_pruning_curve(folder, out):
    text = read(folder, "prune_curve_log.txt")
    if text is None:
        return
    by_q = {}
    for _, q, clean, _, worst, _ in PRUNE.findall(text):
        by_q.setdefault(float(q), {"clean": [], "worst": []})
        by_q[float(q)]["clean"].append(float(clean))
        by_q[float(q)]["worst"].append(float(worst))
    qs = sorted(by_q)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for key, label, color in (("clean", "Clean F1", "#7f7f7f"),
                              ("worst", "Worst-case F1 under adaptive attack", "#9467bd")):
        stats = [mean_std(by_q[q][key]) for q in qs]
        ax.errorbar([100 * q for q in qs], [s[0] for s in stats], yerr=[s[1] for s in stats], marker="o",
                    capsize=3, color=color, label=label)
    ax.set_xlabel("Real edges pruned (%)")
    ax.set_ylabel("F1 on the illicit class")
    ax.set_ylim(0, 0.8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    save(fig, out, "fig4_pruning_curve")


def fig_final_comparison(folder, out):
    path = Path(folder) / "final_comparison_sametime.csv"
    if not path.exists():
        print("skipped: final_comparison_sametime.csv not found")
        return
    df = pd.read_csv(path)
    order = ["GCN-Skip", "GraphSAGE", "GAT-Skip", "GCN-Skip + pruning (25%)", "GCN-Skip, no graph"]
    order = [m for m in order if m in set(df["model"])]
    trees = {"Random Forest": 0.816, "XGBoost": 0.784}   # single runs, from the Week 3 log

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, col, title in ((axes[0], "clean_f1", "Clean test set"),
                           (axes[1], "worst_f1", "Worst case under the adaptive attacker")):
        labels, means, stds, colors = [], [], [], []
        for name, val in trees.items():
            labels.append(name); means.append(val); stds.append(0.0); colors.append(COLORS[name])
        for m in order:
            vals = df.loc[df["model"] == m, col]
            labels.append(m.replace(" + pruning (25%)", "\n+ pruning").replace(", no graph", "\nno graph"))
            means.append(vals.mean()); stds.append(vals.std(ddof=1)); colors.append(COLORS[m])
        x = np.arange(len(labels))
        bars = ax.bar(x, means, color=colors)
        ax.errorbar(x[2:], means[2:], yerr=stds[2:], fmt="none", ecolor="black", capsize=3)   # trees: single runs, no bar
        for i in (0, 1):                      # tree models: not attacked in this setting
            bars[i].set_hatch("//" if col == "worst_f1" else "")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("F1 on the illicit class")
    save(fig, out, "fig5_final_comparison")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(REPO / "results"))
    parser.add_argument("--out", default=str(REPO / "paper" / "figures"))
    args = parser.parse_args()
    out = Path(args.out)
    fig_overview(out)
    fig_attack_budget(args.results, out)
    fig_time_step(args.results, out)
    fig_pruning_curve(args.results, out)
    fig_final_comparison(args.results, out)


if __name__ == "__main__":
    main()