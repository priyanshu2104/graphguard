"""
Recomputes the paper's multi-seed numbers from the raw per-seed lines in the logs and compares them
with the values written in the paper. Run it before pasting numbers; wherever it prints DIFF or
MISSING, fix the paper (or the log) before going on.

    python -m src.defense.audit_numbers            # reads results/
    python -m src.defense.audit_numbers --dir PATH # reads another folder
"""
import argparse
import re
from pathlib import Path

import numpy as np

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "results"
ARCHS = ("GCN-Skip", "GraphSAGE", "GAT-Skip")
F1_TOL, EV_TOL = 0.0025, 0.06   # tolerances for F1-type numbers and for evasion percentages

SEED_CHECK = re.compile(
    r"^(GCN-Skip|GraphSAGE|GAT-Skip) seed (\d+): clean F1 ([\d.]+) \| strongest-variant evasion "
    r"b=1: ([\d.]+)% b=3: ([\d.]+)% b=5: ([\d.]+)%", re.M)
NO_GRAPH = re.compile(r"^\s+seed (\d+): F1 ([\d.]+)\s+AUC-PR ([\d.]+)", re.M)
PRUNED = re.compile(
    r"^\s+seed (\d+) q=([\d.]+): clean F1 ([\d.]+) \| worst adaptive F1 ([\d.]+) recall ([\d.]+)", re.M)
LOCAL = re.compile(r"^seed (\d+) \| local features, (graph|no graph): F1 ([\d.]+)\s+AUC-PR ([\d.]+)", re.M)
FILTER = re.compile(
    r"F1 ([\d.]+) evasion ([\d.]+)% \| after dropping cross-time edges: F1 ([\d.]+) evasion ([\d.]+)%")

results = {"ok": 0, "bad": 0}


def mean_std(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def compare(label, values, expected, tol):
    """expected = (mean, std) as written in the paper."""
    if not values:
        print(f"MISSING  {label}")
        results["bad"] += 1
        return
    m, s = mean_std(values)
    good = abs(m - expected[0]) <= tol and abs(s - expected[1]) <= tol
    print(f"{'ok      ' if good else 'DIFF    '}{label}: log {m:.3f} ± {s:.3f} (n={len(values)}) | paper {expected[0]} ± {expected[1]}")
    results["ok" if good else "bad"] += 1


def compare_single(label, got, expected, tol):
    if got is None:
        print(f"MISSING  {label}")
        results["bad"] += 1
        return
    good = abs(got - expected) <= tol
    print(f"{'ok      ' if good else 'DIFF    '}{label}: log {got} | paper {expected}")
    results["ok" if good else "bad"] += 1


def read(folder, name):
    path = folder / name
    return path.read_text(encoding="utf-8") if path.exists() else None


def check_seed_table(folder, fname, title, clean_exp, ev_exp):
    print(f"\n{title}  ({fname})")
    text = read(folder, fname)
    if text is None:
        print(f"MISSING  {fname}")
        results["bad"] += 1
        return
    rows = {}
    for arch, _, f1, e1, e3, e5 in SEED_CHECK.findall(text):
        r = rows.setdefault(arch, {"clean": [], 1: [], 3: [], 5: []})
        r["clean"].append(float(f1)); r[1].append(float(e1)); r[3].append(float(e3)); r[5].append(float(e5))
    for arch in ARCHS:
        r = rows.get(arch, {})
        compare(f"{arch} clean F1", r.get("clean"), clean_exp[arch], F1_TOL)
        for b in (1, 3, 5):
            compare(f"{arch} evasion b={b}", r.get(b), ev_exp[arch][b], EV_TOL)


def check_defense_seeds(folder, fname, title, pruned_exp, check_no_graph):
    print(f"\n{title}  ({fname})")
    text = read(folder, fname)
    if text is None:
        print(f"MISSING  {fname}")
        results["bad"] += 1
        return
    if check_no_graph:
        ng = NO_GRAPH.findall(text)
        compare("no-graph F1", [float(f) for _, f, _ in ng], (0.642, 0.023), F1_TOL)
        compare("no-graph AUC-PR", [float(a) for _, _, a in ng], (0.626, 0.033), F1_TOL)
    by_q = {}
    for _, q, clean, worst, rec in PRUNED.findall(text):
        r = by_q.setdefault(round(float(q), 2), {"clean": [], "worst": []})
        r["clean"].append(float(clean)); r["worst"].append(float(worst))
    for q, (clean_exp, worst_exp) in pruned_exp.items():
        r = by_q.get(q, {})
        compare(f"pruned q={q:.2f} clean F1", r.get("clean"), clean_exp, F1_TOL)
        compare(f"pruned q={q:.2f} worst-case F1", r.get("worst"), worst_exp, F1_TOL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(DEFAULT_DIR))
    folder = Path(parser.parse_args().dir)

    # Table 5 (unconstrained / cross-time adaptive attacker, three training seeds)
    check_seed_table(
        folder, "seed_check_log.txt", "Unconstrained (cross-time) adaptive attacker",
        {"GCN-Skip": (0.554, 0.045), "GraphSAGE": (0.573, 0.014), "GAT-Skip": (0.495, 0.011)},
        {"GCN-Skip": {1: (87.9, 3.0), 3: (99.0, 0.1), 5: (99.6, 0.2)},
         "GraphSAGE": {1: (70.9, 1.1), 3: (91.5, 0.7), 5: (96.7, 1.6)},
         "GAT-Skip": {1: (23.7, 3.1), 3: (46.6, 1.8), 5: (58.3, 3.5)}})

    # Table 7 (time-consistent adaptive attacker, three training seeds)
    check_seed_table(
        folder, "seed_check_sametime_log.txt", "Time-consistent adaptive attacker",
        {"GCN-Skip": (0.554, 0.045), "GraphSAGE": (0.573, 0.014), "GAT-Skip": (0.495, 0.011)},
        {"GCN-Skip": {1: (54.0, 5.8), 3: (62.4, 8.0), 5: (60.2, 7.9)},
         "GraphSAGE": {1: (71.2, 2.7), 3: (86.2, 3.3), 5: (89.1, 2.5)},
         "GAT-Skip": {1: (19.7, 4.0), 3: (26.8, 9.6), 5: (32.2, 11.2)}})

    # Table 8 (pruning, time-consistent) and the no-graph reference
    check_defense_seeds(
        folder, "defense_seeds_sametime_log.txt", "Pruning, time-consistent attacker",
        {0.0: ((0.554, 0.045), (0.254, 0.066)), 0.25: ((0.555, 0.043), (0.397, 0.042)),
         0.5: ((0.548, 0.040), (0.358, 0.049))}, True)

    # Pruning, unconstrained attacker (ablation)
    check_defense_seeds(
        folder, "defense_seeds_log.txt", "Pruning, unconstrained attacker",
        {0.0: ((0.554, 0.045), (0.003, 0.002)), 0.25: ((0.555, 0.043), (0.334, 0.027)),
         0.5: ((0.548, 0.040), (0.477, 0.028))}, False)

    # Table 9 (local-features rows)
    print("\nGraph vs no graph, local features  (local_features_log.txt)")
    text = read(folder, "local_features_log.txt")
    if text is None:
        print("MISSING  local_features_log.txt")
        results["bad"] += 1
    else:
        rows = {"graph": {"f1": [], "ap": []}, "no graph": {"f1": [], "ap": []}}
        for _, kind, f1, ap in LOCAL.findall(text):
            rows[kind]["f1"].append(float(f1)); rows[kind]["ap"].append(float(ap))
        compare("local graph F1", rows["graph"]["f1"], (0.350, 0.018), F1_TOL)
        compare("local graph AUC-PR", rows["graph"]["ap"], (0.400, 0.016), F1_TOL)
        compare("local no-graph F1", rows["no graph"]["f1"], (0.442, 0.019), F1_TOL)
        compare("local no-graph AUC-PR", rows["no graph"]["ap"], (0.582, 0.053), F1_TOL)

    # Table 6 and the time-consistency filter (single runs)
    print("\nTime-consistent random camouflage and the time filter  (sametime_eval_log.txt)")
    text = read(folder, "sametime_eval_log.txt")
    if text is None:
        print("MISSING  sametime_eval_log.txt")
        results["bad"] += 1
    else:
        expected = {"GCN-Skip": {1: 19.3, 3: 32.6, 5: 42.0}, "GraphSAGE": {1: 15.0, 3: 18.3, 5: 20.1},
                    "GAT-Skip": {1: 5.8, 3: 5.5, 5: 3.7}}
        current, got = None, {}
        for line in text.splitlines():
            head = re.match(r"^(GCN-Skip|GraphSAGE|GAT-Skip): clean F1", line)
            if head:
                current = head.group(1)
            row = re.match(r"^\s+b=(\d) \| random: evasion ([\d.]+)%", line)
            if row and current:
                got.setdefault(current, {})[int(row.group(1))] = float(row.group(2))
        for arch in ARCHS:
            for b in (1, 3, 5):
                compare_single(f"{arch} random same-time evasion b={b}", got.get(arch, {}).get(b),
                               expected[arch][b], 0.05)
        m = FILTER.search(text)
        if m:
            for label, got_v, exp_v in zip(
                    ("filter: F1 before", "filter: evasion before", "filter: F1 after", "filter: evasion after"),
                    (float(x) for x in m.groups()), (0.009, 98.9, 0.601, 0.0)):
                compare_single(label, got_v, exp_v, 0.06)
        else:
            print("MISSING  time-consistency filter line")
            results["bad"] += 1

    print(f"\n{results['ok']} ok, {results['bad']} DIFF/MISSING")


if __name__ == "__main__":
    main()