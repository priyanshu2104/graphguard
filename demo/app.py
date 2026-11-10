"""
GraphGuard demo: score an Elliptic transaction under three configurations, explain the score,
simulate a camouflage attack on it, and keep a tamper-evident log of flag reports.

    streamlit run demo/app.py        (from the repository root, after: python -m src.demo.train_no_graph)
"""
import copy
from pathlib import Path

import pandas as pd
import streamlit as st

from src.demo.merkle_log import MerkleLog, leaf_hash, canonical, verify_record_hex
from src.demo.service import DemoService, LABELS, CANDIDATE_KINDS, PRUNE_Q

LOG_PATH = Path(__file__).resolve().parents[1] / "results" / "demo_log" / "reports.jsonl"
VARIANTS = list(LABELS)

st.set_page_config(page_title="GraphGuard demo", layout="wide")


@st.cache_resource(show_spinner="Loading the graph and models (about a minute the first time)...")
def get_service():
    return DemoService()


@st.cache_resource
def get_log():
    return MerkleLog(LOG_PATH)


svc = get_service()
log = get_log()

# ---------------------------------------------------------------------------------------------
# Sidebar: choose a transaction
# ---------------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Choose a transaction")
    kind = st.selectbox("Pick from", CANDIDATE_KINDS, key="kind")
    options = svc.candidates(kind)
    choice = st.selectbox("Transaction ID", [svc.txid(i) for i in options], key="choice")
    typed = st.text_input("...or type a transaction ID", key="typed")

idx = None
if typed.strip():
    try:
        idx = svc.idx(int(typed.strip()))
    except ValueError:
        idx = None
    if idx is None:
        st.sidebar.error("That is not a transaction ID in the Elliptic dataset; using the list instead.")
if idx is None and options:
    idx = svc.idx(choice)

st.title("GraphGuard")
st.caption("Fraud detection on Bitcoin transactions: scores, explanations, a camouflage-attack simulator "
           "and a tamper-evident report log.")

with st.expander("What this demo is, and what it is not", expanded=False):
    st.markdown(
        "- It browses the **Elliptic dataset**: anonymised Bitcoin *transactions* (the graph's nodes), "
        "not wallets, and not live blockchain data. You can only score transactions that are in the dataset.\n"
        "- Models are the seed-42 models from the paper. They were trained on time steps 1-34; "
        "the transactions offered here are from time steps 35-49, which they did not see.\n"
        "- The dataset label is shown for reference only; the models never see it.\n"
        "- Feature columns are anonymised, so explanations say *which columns* mattered and which group "
        "they belong to, not what they mean. The local/aggregated grouping follows the dataset's published "
        "layout, which we could not verify independently.\n"
        "- The attack simulator attacks one transaction. Population-level results are in the paper; a "
        "single transaction may or may not flip.")

if idx is None:
    st.warning("No transaction available.")
    st.stop()

info = svc.info(idx)
with st.sidebar:
    st.divider()
    st.write(f"**Transaction {info['txid']}**")
    st.write(f"Time step {info['time_step']} | in-degree {info['in_degree']}")
    st.caption(f"Dataset label: {info['label']} (not available to the models)")
    if not info["test_period"]:
        st.warning("Time step <= 34: the models were trained on this period, so its score is not a fair test.")

tab_score, tab_attack, tab_log = st.tabs(["Score and explanation", "Attack simulator", "Integrity log"])

# ---------------------------------------------------------------------------------------------
# Tab 1: scores and explanation
# ---------------------------------------------------------------------------------------------
with tab_score:
    cols = st.columns(len(VARIANTS))
    for col, key in zip(cols, VARIANTS):
        p = float(svc.clean_probs[key][idx])
        col.subheader(LABELS[key])
        col.metric("P(illicit)", f"{p:.3f}")
        col.write("**FLAGGED**" if p >= 0.5 else "not flagged")

    st.divider()
    variant = st.radio("Explain which model?", VARIANTS, format_func=LABELS.get, horizontal=True, key="variant")
    if st.button("Explain this score", key="btn_explain"):
        with st.spinner("Running the explainer three times (a few seconds)..."):
            st.session_state["expl"] = {"idx": idx, "variant": variant, "result": svc.explain(variant, idx)}

    expl = st.session_state.get("expl")
    if expl and expl["idx"] == idx and expl["variant"] == variant:
        res = expl["result"]
        stability = res.get("stability")
        st.write(f"Explained on a neighbourhood of {res['subgraph_nodes']} transaction(s). "
                 + (f"Agreement between explainer runs (rank correlation): **{stability:.2f}**."
                    if stability is not None else ""))
        if stability is not None and stability < 0.7:
            st.warning("The explainer runs disagree; treat this explanation with caution.")
        features = pd.DataFrame([{"feature": f"feat_{i + 1}", "group": g, "importance": round(s, 4)}
                                 for i, s, g in res["top_features"]])
        st.markdown("**Most important feature columns**")
        st.bar_chart(features.set_index("feature")["importance"])
        st.dataframe(features, hide_index=True)
        st.caption("The top scores are usually within about 0.02 of each other, so read this as a set of "
                   "important columns, not a strict ranking.")
        if res["top_edges"]:
            edges = pd.DataFrame([{"from transaction": svc.txid(s), "to transaction": svc.txid(d),
                                   "importance": round(sc, 3)} for s, d, sc in res["top_edges"]])
            st.markdown("**Most important edges in the neighbourhood**")
            st.dataframe(edges, hide_index=True)
        else:
            st.caption("This transaction has no edges in the graph this model sees, so only features matter.")

    st.divider()
    if st.button("Log this flag report", key="btn_log"):
        p = float(svc.clean_probs[variant][idx])
        record = {"txid": info["txid"], "time_step": info["time_step"], "model": LABELS[variant],
                  "p_illicit": round(p, 6), "flagged": bool(p >= 0.5),
                  "prune_q": PRUNE_Q if variant == "pruned" else None}
        if expl and expl["idx"] == idx and expl["variant"] == variant:
            record["top_features"] = [[f"feat_{i + 1}", round(s, 4)] for i, s, _ in expl["result"]["top_features"]]
            record["explanation_stability"] = (round(expl["result"]["stability"], 3)
                                               if expl["result"].get("stability") is not None else None)
        n = log.append(record)
        st.success(f"Logged as record {n}. See the Integrity log tab.")

# ---------------------------------------------------------------------------------------------
# Tab 2: attack simulator
# ---------------------------------------------------------------------------------------------
with tab_attack:
    st.write("Add camouflage edges between this transaction and licit transactions from the **same time step**, "
             "then rescore it. The no-graph model cannot change (its input has no edges, and its neighbourhood "
             "features are fixed).")
    attacker = st.radio("Attacker", ["adaptive", "random"], horizontal=True, key="atk_kind",
                        format_func={"adaptive": "Adaptive (white-box: picks the most licit-looking partners)",
                                     "random": "Random partners"}.get)
    budget = st.slider("Camouflage edges added (budget b)", 1, 5, 3, key="atk_b")
    seed = st.number_input("Random seed", min_value=0, max_value=99, value=0, key="atk_seed")

    if st.button("Simulate attack on this transaction", key="btn_attack"):
        with st.spinner("Building the attack and rescoring (a few seconds)..."):
            st.session_state["attack"] = {"idx": idx, "kind": attacker, "b": budget,
                                          "res": svc.simulate_attack(idx, attacker, budget, int(seed))}
    atk = st.session_state.get("attack")
    if atk and atk["idx"] == idx:
        rows = []
        for key in VARIANTS:
            r = atk["res"][key]
            before, after = r["before"], r["after"]
            outcome = ("EVADED" if before >= 0.5 > after else
                       "newly flagged" if before < 0.5 <= after else "unchanged decision")
            rows.append({"model": LABELS[key], "P(illicit) before": round(before, 3),
                         "P(illicit) after": round(after, 3), "outcome": outcome})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        partners = atk["res"]["graph"]["partners"]
        if partners:
            st.caption(f"Partner transactions used against the undefended graph model: "
                       f"{', '.join(str(p) for p in partners[:10])}")
    st.info("For context, across all detected illicit test transactions at b=5, the adaptive attacker evades "
            "about 60% of the undefended graph model's detections and about 37% once pruning is applied "
            "(three training seeds). The no-graph model is unaffected in this setting. The attack assumes the "
            "attacker can get licit transactions to feed the target's outputs, which may be hard to arrange.")

# ---------------------------------------------------------------------------------------------
# Tab 3: integrity log
# ---------------------------------------------------------------------------------------------
with tab_log:
    n = len(log.records)
    st.write(f"Records: **{n}**. Current Merkle root:")
    st.code(log.root_hex(), language=None)
    if n:
        table = pd.DataFrame([{"index": i, "txid": r["txid"], "model": r["model"], "P(illicit)": r["p_illicit"],
                               "flagged": r["flagged"], "logged at": r["logged_at"],
                               "leaf hash": leaf_hash(canonical(r)).hex()[:12]}
                              for i, r in enumerate(log.records)])
        st.dataframe(table, hide_index=True)

    c1, c2 = st.columns(2)
    if c1.button("Save a checkpoint of the current root", key="btn_checkpoint"):
        cp = log.checkpoint()
        st.success(f"Saved root for {cp['size']} record(s).")
    if c2.button("Audit the log against saved checkpoints", key="btn_audit"):
        results = log.audit()
        if not results:
            st.info("No checkpoints saved yet.")
        for r in results:
            (st.success if r["ok"] else st.error)(
                f"Checkpoint of {r['size']} record(s): {'matches' if r['ok'] else 'DOES NOT MATCH (' + r['why'] + ')'}")

    if n:
        st.divider()
        i = st.number_input("Record index", min_value=0, max_value=n - 1, value=0, key="rec_idx")
        proof = log.proof_hex(int(i))
        ok = verify_record_hex(log.records[int(i)], int(i), n, proof, log.root_hex())
        st.write(f"Inclusion proof for record {int(i)}: {len(proof)} hash(es); "
                 f"verifies against the current root: **{ok}**")
        if st.button("Simulate editing this record (in memory only)", key="btn_tamper"):
            edited = copy.deepcopy(log.records)
            edited[int(i)]["p_illicit"] = round(edited[int(i)]["p_illicit"] * 0.2, 6)
            still_ok = verify_record_hex(edited[int(i)], int(i), n, proof, log.root_hex())
            st.error(f"The edited record's proof verifies against the published root: {still_ok}")
            for r in log.audit(edited):
                (st.success if r["ok"] else st.error)(
                    f"Checkpoint of {r['size']} record(s): {'matches' if r['ok'] else 'DOES NOT MATCH'}")

    st.caption("This log is tamper-EVIDENT, not tamper-proof. An edit, removal or reordering of an earlier "
               "record changes the root, but only a root saved somewhere the log's owner cannot rewrite "
               "(printed, emailed, committed elsewhere, or anchored on a public ledger) makes that visible. "
               "Anchoring on a ledger is not implemented.")