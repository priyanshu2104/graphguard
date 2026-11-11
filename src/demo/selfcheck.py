"""
QA for the demo back end. The demo must reproduce the numbers in the paper, and transaction ids must
round-trip. With --examples it also looks for transactions that make a good live walkthrough.

    python -m src.demo.selfcheck
    python -m src.demo.selfcheck --examples
"""
import argparse

import numpy as np

from src.demo.service import DemoService, CANDIDATE_KINDS, LABELS
from src.defense.eval_utils import metrics

# seed-42 models, clean test set: (F1, false positives), from the logs used in the paper
EXPECTED = {"no_graph": (0.6296, 199), "graph": (0.6005, 462), "pruned": (0.597, 454)}


def find_examples(svc, n=40):
    print(f"\nAdaptive attack, b=3, on {n} detected illicit test transactions (this takes a few minutes)")
    rows = []
    for idx in svc.candidates(CANDIDATE_KINDS[0], n=n, seed=1):
        r = svc.simulate_attack(idx, kind="adaptive", budget=3, seed=0)
        rows.append((idx, svc.info(idx), r))
    show = lambda r, k: f"{r[k]['before']:.2f}->{r[k]['after']:.2f}"
    print("txid        step in-deg | no graph   | graph      | pruned")
    for idx, info, r in rows:
        print(f"{info['txid']:<11} {info['time_step']:>4} {info['in_degree']:>6} | "
              f"{show(r, 'no_graph'):<10} | {show(r, 'graph'):<10} | {show(r, 'pruned')}")
    helps = [i for i, info, r in rows if r["graph"]["after"] < 0.5 <= r["pruned"]["after"]]
    hurts = [i for i, info, r in rows
             if r["graph"]["after"] >= 0.5 > r["pruned"]["after"] and r["pruned"]["before"] >= 0.5]
    print(f"\nPruning helps (graph model fooled, pruned model still flags): "
          f"{[(svc.txid(i), svc.info(i)['time_step']) for i in helps[:6]]}")
    print(f"Pruning hurts (graph model still flags, pruned model fooled): "
          f"{[(svc.txid(i), svc.info(i)['time_step']) for i in hurts[:6]]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", action="store_true")
    args = parser.parse_args()

    svc = DemoService()
    test_idx = svc.test_mask.nonzero(as_tuple=True)[0].numpy()
    y = svc.data.y[test_idx].numpy()
    failures = 0

    print("Clean test-set scores from the demo's back end against the paper's seed-42 numbers")
    for key, (f1_exp, fp_exp) in EXPECTED.items():
        p = svc.clean_probs[key][test_idx]
        m = metrics(y, (p >= 0.5).astype(int), p)
        ok = abs(m["f1"] - f1_exp) <= 0.002 and abs(m["fp"] - fp_exp) <= 3
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {LABELS[key]:<24} F1 {m['f1']:.4f} (expected {f1_exp}) | "
              f"FP {m['fp']} (expected {fp_exp})")

    rng = np.random.default_rng(0)
    sample = rng.choice(test_idx, size=5, replace=False)
    round_trip = all(svc.idx(svc.txid(i)) == int(i) for i in sample)
    kept = svc.pruned_data.edge_index.shape[1] / svc.data.edge_index.shape[1]
    steps_ok = bool((svc.data.time_step[test_idx] >= 35).all())
    for label, ok in [("transaction id round-trip", round_trip),
                      (f"pruned graph keeps {kept:.1%} of edges (expected about 75%)", abs(kept - 0.75) < 0.01),
                      ("all labelled test nodes are in time steps >= 35", steps_ok)]:
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {label}")

    print(f"\n{'All checks passed.' if not failures else str(failures) + ' check(s) FAILED.'}")
    if args.examples:
        find_examples(svc)


if __name__ == "__main__":
    main()