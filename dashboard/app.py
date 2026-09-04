"""Risk desk — live view over the case audit log.

Reads the append-only SQLite log the agents write to. Nothing here computes a
decision: the dashboard's whole job is to make the evidence behind a decision
readable, which is the point the project is arguing.

    .venv/bin/streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "cases.db"
RESULTS_DIR = ROOT / "eval" / "results"

DECISION_COLOR = {"block": "#c62828", "escalate": "#ef6c00", "allow": "#2e7d32"}

st.set_page_config(page_title="AI-Native Fraud Defense — Risk Desk",
                   page_icon="🛡️", layout="wide")


@st.cache_data(ttl=3)
def load_cases() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM cases ORDER BY timestamp DESC", conn)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
        df["n_evidence"] = df["evidence"].map(lambda e: len(json.loads(e)))
    return df


@st.cache_data(ttl=10)
def load_metrics() -> dict:
    out = {}
    for path in sorted(RESULTS_DIR.glob("*_metrics.json")):
        try:
            out[path.stem.replace("_metrics", "")] = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
    return out


def evidence_table(raw: str) -> pd.DataFrame:
    items = json.loads(raw)
    if not items:
        return pd.DataFrame(columns=["signal", "value", "weight"])
    return pd.DataFrame(items)[["signal", "value", "weight"]]


df = load_cases()
metrics = load_metrics()

st.title("🛡️ AI-Native Fraud Defense — Risk Desk")
if df.empty:
    st.warning(
        "No cases in the audit log yet. Populate it by running an agent:\n\n"
        "```\n.venv/bin/python -m agents.spike_sentinel\n"
        ".venv/bin/python -m agents.ring_detector\n"
        ".venv/bin/python -m agents.content_forensics\n"
        ".venv/bin/python -m agents.adjudicator\n```")
    st.stop()

st.caption(
    f"{len(df):,} cases in the append-only audit log · "
    f"latest {df['timestamp'].max():%Y-%m-%d %H:%M} UTC · "
    f"source: `data/cases.db`")

# --- headline counters ------------------------------------------------------
cols = st.columns(5)
cols[0].metric("Cases", f"{len(df):,}")
cols[1].metric("Agents", df["source_agent"].nunique())
for i, decision in enumerate(("block", "escalate", "allow"), start=2):
    n = int((df["decision"] == decision).sum())
    cols[i].metric(decision.capitalize(), f"{n:,}",
                   f"{n/len(df):.1%} of cases", delta_color="off")

tab_cases, tab_metrics, tab_replay = st.tabs(
    ["Case files", "Held-out metrics", "Replay"])

# ---------------------------------------------------------------------------
# Case files
# ---------------------------------------------------------------------------
with tab_cases:
    left, right = st.columns([1, 2.2])

    with left:
        st.subheader("Filter")
        agents = st.multiselect(
            "Agent", sorted(df["source_agent"].unique()),
            default=sorted(df["source_agent"].unique()))
        decisions = st.multiselect(
            "Decision", ["block", "escalate", "allow"],
            default=["block", "escalate", "allow"])
        min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
        search = st.text_input("Search cases", "",
                               help="Matches entity id, case id, or reasoning "
                                    "text — e.g. a cluster id, a session id, "
                                    "or an amount")

        view = df[
            df["source_agent"].isin(agents)
            & df["decision"].isin(decisions)
            & (df["confidence"] >= min_conf)
        ]
        if search:
            # Search the identifiers as well as the narrative: looking up a
            # known entity is the first thing an analyst does, and the entity
            # id is often the only handle they have (a cluster id, a session
            # id from an alert). Text-only search silently returned nothing
            # for those.
            haystack = (view["entity_id"].fillna("") + " "
                        + view["case_id"].fillna("") + " "
                        + view["reasoning_text"].fillna(""))
            view = view[haystack.str.contains(search, case=False, na=False,
                                              regex=False)]

        st.caption(f"{len(view):,} matching cases")
        st.dataframe(
            view["source_agent"].value_counts().rename("cases"),
            width="stretch")

    with right:
        st.subheader("Cases")
        if view.empty:
            st.info("No cases match these filters.")
        else:
            display = view[["timestamp", "source_agent", "entity_type",
                            "decision", "confidence", "n_evidence",
                            "cost_estimate", "case_id"]].head(300)
            event = st.dataframe(
                display, width="stretch", hide_index=True,
                on_select="rerun", selection_mode="single-row",
                column_config={
                    "confidence": st.column_config.ProgressColumn(
                        "confidence", min_value=0.0, max_value=1.0,
                        format="%.3f"),
                    "cost_estimate": st.column_config.NumberColumn(
                        "cost if wrong", format="%.2f"),
                    "case_id": st.column_config.TextColumn("case", width="small"),
                })

            rows = event.selection.rows if event and event.selection else []
            idx = display.index[rows[0]] if rows else view.index[0]
            case = view.loc[idx]

            st.divider()
            colour = DECISION_COLOR.get(case["decision"], "#555")
            st.markdown(
                f"### Case `{case['case_id'][:8]}` "
                f"<span style='color:{colour}'>[{case['decision'].upper()}]</span>",
                unsafe_allow_html=True)
            meta = st.columns(4)
            meta[0].metric("Agent", case["source_agent"])
            meta[1].metric("Confidence", f"{case['confidence']:.3f}")
            meta[2].metric("Entity type", case["entity_type"])
            meta[3].metric("Cost if wrong", f"{case['cost_estimate']:,.2f}")
            st.caption(f"entity `{case['entity_id']}`")

            st.markdown("**Evidence**")
            st.dataframe(evidence_table(case["evidence"]),
                         width="stretch", hide_index=True)

            st.markdown("**Reasoning**")
            st.info(case["reasoning_text"] or "_(none recorded)_")

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
with tab_metrics:
    if not metrics:
        st.info("No metrics JSON found in eval/results/ yet.")
    for name, report in metrics.items():
        st.subheader(name)
        results = report.get("results", {})

        # Ring Detector nests results per split; flatten for display.
        flat = {}
        for key, block in results.items():
            if isinstance(block, dict) and "precision" in block:
                flat[key] = block
            elif isinstance(block, dict):
                for sub, inner in block.items():
                    if isinstance(inner, dict) and "precision" in inner:
                        flat[f"{key} / {sub}"] = inner
        if flat:
            st.dataframe(
                pd.DataFrame(flat).T[
                    [c for c in ("precision", "recall", "f1", "pr_auc",
                                 "tp", "fp", "fn", "tn")
                     if c in next(iter(flat.values()))]],
                width="stretch")
        with st.expander("Full report JSON"):
            st.json(report)

    cost_path = RESULTS_DIR / "cost_table.json"
    if cost_path.exists():
        st.subheader("Cost table")
        cost = json.loads(cost_path.read_text())
        st.caption(cost["disclaimer"])
        st.dataframe(pd.DataFrame(cost["table"]).T[
            ["entity", "false_positive_inr", "false_negative_inr",
             "fn_to_fp_ratio", "policy"]], width="stretch")

# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
with tab_replay:
    st.write(
        "Replays cases out of the audit log at demo speed, newest last, so "
        "the desk visibly fills the way it would under a live feed. This is a "
        "replay of decisions already made and recorded — not a live scoring "
        "pipeline, and the README says so too.")
    speed = st.slider("Cases per second", 1, 20, 4)
    n = st.slider("How many cases", 5, 100, 25)
    if st.button("▶ Replay", type="primary"):
        stream = df.sort_values("timestamp").tail(n)
        placeholder = st.empty()
        bar = st.progress(0.0)
        shown = []
        for i, (_, row) in enumerate(stream.iterrows(), 1):
            colour = DECISION_COLOR.get(row["decision"], "#555")
            shown.insert(0, (
                f"<div style='border-left:4px solid {colour};padding:6px 10px;"
                f"margin-bottom:6px;background:rgba(128,128,128,0.06)'>"
                f"<b>{row['source_agent']}</b> · "
                f"<span style='color:{colour}'><b>{row['decision'].upper()}</b>"
                f"</span> · confidence {row['confidence']:.3f}<br>"
                f"<small>{row['reasoning_text'][:240]}</small></div>"))
            placeholder.markdown("".join(shown[:12]), unsafe_allow_html=True)
            bar.progress(i / len(stream))
            time.sleep(1.0 / speed)
        st.success(f"Replayed {len(stream)} cases.")
