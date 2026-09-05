"""Risk desk — live view over the case audit log.

Reads the append-only SQLite log the agents write to. Nothing here computes a
decision: the dashboard's whole job is to make the evidence behind a decision
readable, which is the point the project is arguing.

"Readable" was doing a lot of work in that sentence. The first version of this
page was readable to someone who already knew what precision, P(fraud) and a
held-out split were. A risk desk is read by support leads, ops managers and
founders as well as analysts, so every technical term is now shown next to a
plain-English twin from dashboard/plain_english.py — with the original value
still on screen, because a translation you cannot check is just a claim.

    .venv/bin/streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
# `streamlit run dashboard/app.py` puts dashboard/ on sys.path but not the
# project root, so the sibling module is imported by package path only after
# the root is added. Explicit beats depending on the launch directory.
sys.path.insert(0, str(ROOT))

from dashboard.plain_english import (AGENTS, DECISIONS, GLOSSARY,  # noqa: E402
                                     NOTES, cost_in_words,
                                     score_in_words, signal_plain)

DB_PATH = ROOT / "data" / "cases.db"
RESULTS_DIR = ROOT / "eval" / "results"

DECISION_COLOR = {"block": "#c62828", "escalate": "#ef6c00", "allow": "#2e7d32"}
DATA_BADGE = {
    "real": "🟢 measured on real data",
    "simulated": "🟡 measured on SIMULATED data — not a real-world result",
    "synthetic": "🟡 measured on SYNTHETIC specimens — not real documents",
    "n/a": "⚪ no detection score — this one arbitrates, it does not detect",
}

# Four pre-located cases that carry the story. Wiring them to buttons means a
# reader who does not know what to search for can still see the good parts.
TOUR = [
    ("🕸️ A fraud ring", "cluster:122640",
     "Eleven accounts, one device, one internet connection, all opened within "
     "the same hour."),
    ("🚫 A score with no explanation", "398.89",
     "The model is 99.8% sure this is fraud and cannot say why. Watch what the "
     "system refuses to do about it."),
    ("✅ An AI agent, allowed", "S000021",
     "Almost certainly a shopping bot — and let through, because it stayed "
     "inside the rules its owner set."),
    ("⛔ An AI agent, blocked", "S000397",
     "The same robotic behaviour, but this one went outside its owner's rules."),
]

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


def scored_blocks(obj, prefix: str = ""):
    """Walk a metrics report and yield every (path, block) holding a score.

    The six reports nest their results differently — Ring Detector splits by
    train/test strategy, Checkout Guard separates its simulated detector from
    its deterministic policy engine. Walking for the shape rather than a fixed
    key means a new agent's report renders without touching this file.
    """
    if not isinstance(obj, dict):
        return
    if "precision" in obj and "recall" in obj:
        yield prefix or "results", obj
        return
    for key, value in obj.items():
        # rule_diagnostics_validation scores each individual rule, not a
        # configuration of the detector; listing it as one would imply the
        # agent had been run that way.
        if key in ("thresholds", "rule_thresholds", "lr_coefficients",
                   "rule_diagnostics_validation"):
            continue
        yield from scored_blocks(value, f"{prefix} / {key}" if prefix else key)


df = load_cases()
metrics = load_metrics()

st.title("🛡️ AI-Native Fraud Defense — Risk Desk")
st.markdown(
    "**Six programs watch for six kinds of fraud. Every time one of them makes "
    "a decision, it writes down what it saw, how sure it was, and what that "
    "decision costs if it is wrong. This page is that pile of paperwork.**")

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
    f"source: `data/cases.db` · nothing here is ever edited or deleted, "
    f"including the decisions that turned out to be wrong")

# --- headline counters ------------------------------------------------------
cols = st.columns(5)
cols[0].metric("Decisions made", f"{len(df):,}")
cols[1].metric("Programs watching", df["source_agent"].nunique())
for i, decision in enumerate(("block", "escalate", "allow"), start=2):
    n = int((df["decision"] == decision).sum())
    label, _ = DECISIONS[decision]
    cols[i].metric(label, f"{n:,}", f"{n/len(df):.1%} of decisions",
                   delta_color="off")
st.caption(" · ".join(f"**{DECISIONS[d][0]}** — {DECISIONS[d][1].split('.')[0]}."
                      for d in ("block", "escalate", "allow")))

tab_start, tab_cases, tab_metrics, tab_replay = st.tabs(
    ["👋 Start here", "📁 Case files", "📊 How well does it work?", "▶️ Replay"])

# ---------------------------------------------------------------------------
# Start here — the page for someone who has never seen this before
# ---------------------------------------------------------------------------
with tab_start:
    st.header("What problem is this solving?")
    st.markdown(
        "Fraud systems were built on two assumptions:\n\n"
        "1. The person trying to cheat you is **a human being clicking through "
        "a checkout**.\n"
        "2. The reviews, disputes and documents people send you were "
        "**written by people**.\n\n"
        "Both are now false. Software can shop on your behalf, and text that "
        "reads like a furious customer can be produced by the thousand for "
        "nothing. A fraud desk built on those two assumptions is defending a "
        "door that attackers no longer use.")

    st.info(
        "**The one rule this whole system is built around**\n\n"
        "A decision that cannot be explained in words a human can read is "
        "never allowed to stop a customer — no matter how certain the software "
        "is. It goes to a person instead.\n\n"
        "That is not a slogan. It is enforced in code, and it changed "
        "**461 real decisions in this log** from *stopped* to *sent to a "
        "person*. You can read every one of them in the Case files tab.")

    st.header("The six programs")
    st.caption("Each one watches a different thing and writes into the same "
               "shared file. None of them can see another's decision except "
               "the last one, whose whole job is to referee.")
    st.dataframe(
        pd.DataFrame([{
            "": info["title"],
            "What it watches": info["watches"],
            "What it does, in one sentence": info["job"],
            "Are its scores from real data?": DATA_BADGE[info["data"]],
            "Decisions in this log": int((df["source_agent"] == name).sum()),
        } for name, info in AGENTS.items()]),
        width="stretch", hide_index=True)

    st.header("What happens to one case")
    flow = st.columns(5)
    steps = [
        ("1 · Something happens",
         "A payment, a sign-up, a review, a checkout, an uploaded ID."),
        ("2 · Signals are gathered",
         "Plain observations: *this is 12× their usual spend*, *these eleven "
         "accounts share one device*."),
        ("3 · A score is formed",
         "Rules and a statistical model combine into one number between 0 and "
         "1. The number alone is never enough."),
        ("4 · Two gates",
         "**Is there readable evidence?** If not — a person decides. **Is a "
         "false alarm nearly as costly as a miss?** If so — a person decides."),
        ("5 · A case file is written",
         "Stopped, sent to a person, or let through — with the evidence, the "
         "cost, and the reason, permanently."),
    ]
    for col, (head, body) in zip(flow, steps):
        col.markdown(f"**{head}**")
        col.caption(body)

    st.header("Where the numbers come from, and where they don't")
    st.markdown(
        "This matters more than any score on this page, so it is said first "
        "rather than buried.")
    real = [n for n, i in AGENTS.items() if i["data"] == "real"]
    fake = [n for n, i in AGENTS.items() if i["data"] in ("simulated", "synthetic")]
    honest = st.columns(2)
    honest[0].success(
        "**Measured on real data**\n\n"
        + "\n".join(f"- **{AGENTS[n]['title']}** — {AGENTS[n]['data_note']}"
                    for n in real))
    honest[1].warning(
        "**Measured on data we generated ourselves**\n\n"
        + "\n".join(f"- **{AGENTS[n]['title']}** — {AGENTS[n]['data_note']}"
                    for n in fake)
        + "\n\nNeither attack has a public record to test against yet. Their "
          "scores show that the code works on the problem as posed — they are "
          "**not** evidence about real traffic, and are labelled that way "
          "everywhere they appear.")

    with st.expander("📖 Glossary — every term on this page, in one line each"):
        for term, meaning in GLOSSARY:
            st.markdown(f"**{term}** — {meaning}")

# ---------------------------------------------------------------------------
# Case files
# ---------------------------------------------------------------------------
with tab_cases:
    st.caption("Every decision the system has made. Start with one of these "
               "four if you don't know what to look for:")
    tour_cols = st.columns(len(TOUR) + 1)
    for col, (label, query, blurb) in zip(tour_cols, TOUR):
        if col.button(label, width="stretch", help=blurb):
            st.session_state["case_search"] = query
            st.rerun()
    if tour_cols[-1].button("↩️ Show everything", width="stretch"):
        st.session_state["case_search"] = ""
        st.rerun()

    active = next((t for t in TOUR if t[1] == st.session_state.get("case_search")),
                  None)
    if active:
        st.info(f"**{active[0]}** — {active[2]}")

    left, right = st.columns([1, 2.2])

    with left:
        st.subheader("Filter")
        agents = st.multiselect(
            "Which program", sorted(df["source_agent"].unique()),
            default=sorted(df["source_agent"].unique()),
            format_func=lambda n: AGENTS.get(n, {}).get("title", n))
        decisions = st.multiselect(
            "What it decided", ["block", "escalate", "allow"],
            default=["block", "escalate", "allow"],
            format_func=lambda d: DECISIONS[d][0])
        min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05,
                             help="How sure the system was, 0 to 1.")
        search = st.text_input(
            "Search", key="case_search",
            help="Matches entity id, case id, or reasoning text — e.g. a "
                 "cluster id, a session id, or an amount")

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
                    "source_agent": st.column_config.TextColumn("program"),
                    "entity_type": st.column_config.TextColumn("looked at"),
                    "decision": st.column_config.TextColumn("decided"),
                    "n_evidence": st.column_config.NumberColumn("reasons"),
                    "confidence": st.column_config.ProgressColumn(
                        "how sure", min_value=0.0, max_value=1.0,
                        format="%.3f"),
                    "cost_estimate": st.column_config.NumberColumn(
                        "cost if wrong", format="%.2f"),
                    "case_id": st.column_config.TextColumn("case", width="small"),
                })

            rows = event.selection.rows if event and event.selection else []
            idx = display.index[rows[0]] if rows else view.index[0]
            case = view.loc[idx]
            info = AGENTS.get(case["source_agent"], {})
            label, meaning = DECISIONS.get(case["decision"],
                                           (case["decision"], ""))

            st.divider()
            colour = DECISION_COLOR.get(case["decision"], "#555")
            st.markdown(
                f"### Case `{case['case_id'][:8]}` "
                f"<span style='color:{colour}'>[{case['decision'].upper()}]</span>",
                unsafe_allow_html=True)

            # The one-paragraph version, for a reader who will not decode a
            # table of signals and weights.
            st.markdown(
                f"**In plain English:** *{info.get('title', case['source_agent'])}* "
                f"— {info.get('job', '')} It looked at one "
                f"**{case['entity_type']}** and decided: "
                f"<span style='color:{colour}'><b>{label.lower()}</b></span>. "
                f"It was **{case['confidence']:.0%} sure**. If that call is "
                f"wrong it costs roughly **₹{case['cost_estimate']:,.0f}**. "
                f"<br><small>{meaning}</small>",
                unsafe_allow_html=True)

            meta = st.columns(4)
            meta[0].metric("Program", info.get("title", case["source_agent"]))
            meta[1].metric("How sure", f"{case['confidence']:.3f}")
            meta[2].metric("Looked at", case["entity_type"])
            meta[3].metric("Cost if wrong", f"₹{case['cost_estimate']:,.2f}")
            st.caption(f"entity `{case['entity_id']}`")

            ev = evidence_table(case["evidence"])
            st.markdown("**Why — the reasons behind this decision**")
            if ev.empty:
                st.warning(
                    "No readable evidence at all. Under this system's one rule, "
                    "a case like this can never be stopped automatically.")
            else:
                for _, row in ev.iterrows():
                    st.markdown(
                        f"- **{signal_plain(row['signal'])}**  \n"
                        f"  <small><code>{row['signal']}</code> · "
                        f"{row['value']} · weight {row['weight']}</small>",
                        unsafe_allow_html=True)

            with st.expander("The same evidence as the system stores it"):
                st.dataframe(ev, width="stretch", hide_index=True)

            st.markdown("**What the system wrote in the file**")
            st.info(case["reasoning_text"] or "_(none recorded)_")

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
with tab_metrics:
    st.markdown(
        "Each program is scored **separately**, on data it never saw while it "
        "was being built. They are never averaged into one headline number, "
        "because they run on different data with different amounts of fraud in "
        "it — an average would hide exactly what you need to see.")
    st.caption("Two numbers matter, and they pull against each other. "
               "**Precision**: of everything it flagged, how much really was "
               "bad — low precision means annoying innocent customers. "
               "**Recall**: of everything really bad, how much it caught — low "
               "recall means fraud gets through.")

    cost_path = RESULTS_DIR / "cost_table.json"
    costs = json.loads(cost_path.read_text())["table"] if cost_path.exists() else {}

    if not metrics:
        st.info("No metrics JSON found in eval/results/ yet.")

    for name, report in metrics.items():
        info = AGENTS.get(name, {})
        st.divider()
        st.subheader(info.get("title", name))
        st.caption(f"{info.get('job', '')}  ·  {DATA_BADGE[info.get('data', 'n/a')]}")

        flat = dict(scored_blocks(report))
        headline = next(
            (k for k in flat if any(h in k for h in (
                "component / graph_rules_only", "ensemble",
                "results / logistic_regression",
                "agent_detection_SIMULATED / logistic_regression",
                "ela_logistic_regression"))), None)

        if headline:
            block = flat[headline]
            if info.get("data") != "real":
                st.warning(f"These numbers come from data we generated. "
                           f"{info.get('data_note', '')}")
            st.markdown(score_in_words(block["precision"], block["recall"],
                                       info.get("finds", "what it looks for")))
            st.caption(f"Headline configuration: `{headline}`. Every other "
                       f"variant tried — including the ones that scored "
                       f"worse — is in the table below, unedited.")
        elif name == "adjudicator":
            found = report.get("conflicts_found", {})
            st.markdown(
                f"This one has no precision or recall, because it does not "
                f"detect anything — it settles arguments. It read "
                f"**{report.get('cases_read', 0):,} case files**, found "
                f"**{sum(found.values()):,} disagreements**, ruled on "
                f"**{report.get('conflicts_arbitrated', 0)}** of them and "
                f"**overturned {report.get('decisions_overturned', 0)}** "
                f"decisions its colleagues had already made.")

        if name in NOTES:
            st.markdown(NOTES[name])

        if name in costs:
            c = costs[name]
            st.markdown("**What a mistake costs**")
            st.markdown(cost_in_words(c["false_positive_inr"],
                                      c["false_negative_inr"],
                                      c["fn_to_fp_ratio"], c["policy"]))
            st.caption(f"A false alarm here means: {c['false_positive_effect']}. "
                       f"A miss means: {c['false_negative_effect']}.")

        if flat:
            with st.expander("Every configuration tested, with the raw numbers"):
                cols_wanted = ("precision", "recall", "f1", "pr_auc",
                               "tp", "fp", "fn", "tn")
                table = pd.DataFrame(flat).T
                st.dataframe(table[[c for c in cols_wanted if c in table.columns]],
                             width="stretch")
        with st.expander("Full report JSON"):
            st.json(report)

    if costs:
        st.divider()
        st.subheader("The cost table")
        st.caption(json.loads(cost_path.read_text())["disclaimer"])
        st.markdown(
            "This is the table that turns a score into an action. A program is "
            "only allowed to stop a customer where missing the fraud is *much* "
            "more expensive than a false alarm. Where the two costs are close, "
            "it can only ever ask a human.")
        st.dataframe(pd.DataFrame(costs).T[
            ["entity", "false_positive_inr", "false_negative_inr",
             "fn_to_fp_ratio", "policy"]], width="stretch",
            column_config={
                "entity": st.column_config.TextColumn("what it decides about"),
                "false_positive_inr": st.column_config.NumberColumn(
                    "cost of a false alarm", format="₹%.0f"),
                "false_negative_inr": st.column_config.NumberColumn(
                    "cost of a miss", format="₹%.0f"),
                "fn_to_fp_ratio": st.column_config.NumberColumn(
                    "how much worse a miss is", format="%.2f×"),
                "policy": st.column_config.TextColumn("so it may…"),
            })

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
            title = AGENTS.get(row["source_agent"], {}).get(
                "title", row["source_agent"])
            verdict = DECISIONS.get(row["decision"], (row["decision"], ""))[0]
            shown.insert(0, (
                f"<div style='border-left:4px solid {colour};padding:6px 10px;"
                f"margin-bottom:6px;background:rgba(128,128,128,0.06)'>"
                f"<b>{title}</b> · "
                f"<span style='color:{colour}'><b>{verdict.upper()}</b>"
                f"</span> · {row['confidence']:.0%} sure<br>"
                f"<small>{row['reasoning_text'][:240]}</small></div>"))
            placeholder.markdown("".join(shown[:12]), unsafe_allow_html=True)
            bar.progress(i / len(stream))
            time.sleep(1.0 / speed)
        st.success(f"Replayed {len(stream)} cases.")
