"""Guardrails for the things that silently broke during the build.

Every test here corresponds to a bug that actually shipped and produced
plausible-looking but wrong numbers. They are cheap, need no data download,
and make no API calls.

    .venv/bin/python -m pytest tests/ -q
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.content_forensics import TemplateReuse, compute_signals, group_split
from eval.cost_table import COSTS, decide
from eval.metrics import binary_metrics, best_f1_threshold
from spine.db import connect, count_cases, fetch_cases, insert_cases
from spine.schema import Case, Evidence


# ---------------------------------------------------------------------------
# The split bug: a prefix-colliding hash put every AI text in held-out and
# every human text in train, so held-out had no negatives and precision was
# trivially 1.0.
# ---------------------------------------------------------------------------

def _toy_corpus() -> pd.DataFrame:
    rows = []
    for domain in ("review", "dispute"):
        for topic in range(12):
            for j in range(3):
                rows.append({"text": f"ai text {domain} {topic} {j}",
                             "is_ai_generated": 1, "domain": domain,
                             "prompt_group": f"{domain}_topic{topic:02d}"})
        for k in range(36):
            rows.append({"text": f"human text {domain} {k}",
                         "is_ai_generated": 0, "domain": domain,
                         "prompt_group": f"corpus_id_{domain}_{k}"})
    return pd.DataFrame(rows)


def test_split_contains_both_classes_in_both_halves():
    split = group_split(_toy_corpus())
    for domain in ("review", "dispute"):
        for half in ("train", "heldout"):
            block = split[(split["domain"] == domain) & (split["split"] == half)]
            classes = set(block["is_ai_generated"])
            assert classes == {0, 1}, (
                f"{domain}/{half} contains only {classes} — a split that "
                f"correlates with the label makes precision meaningless")


def test_split_keeps_ai_groups_whole():
    """Held-out AI topics must be unseen in training, or template reuse just
    memorizes our own generation phrasing."""
    split = group_split(_toy_corpus())
    ai = split[split["is_ai_generated"] == 1]
    for group, block in ai.groupby("prompt_group"):
        assert block["split"].nunique() == 1, f"group {group} straddles the split"


def test_split_is_deterministic():
    a = group_split(_toy_corpus())["split"].tolist()
    b = group_split(_toy_corpus())["split"].tolist()
    assert a == b, "split must not depend on a salted hash"


# ---------------------------------------------------------------------------
# The self-match bug: train AI texts were scored against a matrix containing
# themselves, so every one scored 1.0 and the derived threshold was
# unreachable on held-out.
# ---------------------------------------------------------------------------

def test_template_reuse_excludes_self_match():
    # Shared vocabulary matters: the vectorizer uses min_df=2, so a corpus of
    # wholly unrelated sentences would prune every term.
    texts = ["the charge on my account was never refunded by support",
             "the charge on my account was never reversed by support",
             "my account shows a charge that support never refunded"]
    reuse = TemplateReuse().fit(texts)
    with_self = reuse.score_many(texts, exclude_self=False)
    without_self = reuse.score_many(texts, exclude_self=True)
    assert np.allclose(with_self, 1.0), "a text should match itself perfectly"
    assert (without_self < 0.99).all(), (
        "training rows must be scored with their own column masked")


def test_template_reuse_scores_are_bounded():
    reuse = TemplateReuse().fit(["alpha beta gamma delta",
                                 "alpha beta epsilon zeta"])
    scores = reuse.score_many(["completely unrelated wording here"])
    assert (scores >= 0).all() and (scores <= 1).all()


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def test_burstiness_separates_uniform_from_varied_rhythm():
    uniform = "This is a sentence. This is a sentence. This is a sentence."
    varied = ("Short. This one runs considerably longer than the last one did, "
              "carrying several clauses along with it. Tiny.")
    assert compute_signals(uniform)["burstiness"] < \
           compute_signals(varied)["burstiness"]


def test_signals_survive_degenerate_input():
    for text in ("", ".", "word", "!!!"):
        sig = compute_signals(text)
        assert np.isfinite(sig["burstiness"])
        assert np.isfinite(sig["mattr"])


# ---------------------------------------------------------------------------
# Cost policy — the gate that stopped 461 evidence-free auto-blocks.
# ---------------------------------------------------------------------------

def test_model_score_alone_never_blocks():
    decision, why = decide("spike_sentinel", confidence=1.0,
                           has_interpretable_evidence=False)
    assert decision == "escalate"
    assert "human-readable" in why


def test_block_allowed_with_evidence_and_favourable_ratio():
    decision, _ = decide("spike_sentinel", confidence=0.99,
                         has_interpretable_evidence=True)
    assert decision == "block"


def test_content_forensics_never_auto_blocks():
    """FN:FP is 2.33 — below the policy bar, so no confidence justifies a
    block on a customer's dispute."""
    decision, why = decide("content_forensics", confidence=1.0,
                           has_interpretable_evidence=True)
    assert decision == "escalate"
    assert "FN:FP" in why


def test_llm_veto_downgrades_to_escalate():
    decision, _ = decide("ring_detector", confidence=1.0,
                         has_interpretable_evidence=True, vetoed=True)
    assert decision == "escalate"


def test_every_agent_has_a_cost_entry():
    for agent in ("spike_sentinel", "ring_detector", "content_forensics"):
        spec = COSTS[agent]
        assert spec["false_positive"]["cost"] > 0
        assert spec["false_negative"]["cost"] > 0
        assert spec["false_positive"]["assumption"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_binary_metrics_match_hand_computed_values():
    y_true = [1, 1, 1, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 0, 1, 0, 0, 0, 0]
    m = binary_metrics(y_true, y_pred)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 4)
    assert m["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["recall"] == pytest.approx(2 / 3, abs=1e-4)


def test_binary_metrics_handles_all_negative_predictions():
    m = binary_metrics([1, 0, 0], [0, 0, 0])
    assert m["precision"] == 0.0 and m["recall"] == 0.0


def test_best_f1_threshold_is_within_score_range():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 200)
    s = rng.rand(200)
    assert s.min() <= best_f1_threshold(y, s) <= s.max()


# ---------------------------------------------------------------------------
# Audit log — append-only, and round-trippable.
# ---------------------------------------------------------------------------

def _case(**kw) -> Case:
    base = dict(source_agent="test_agent", entity_id="E1",
                entity_type="transaction", confidence=0.8, decision="escalate",
                reasoning_text="because",
                evidence=[Evidence("sig", "value", 0.5)])
    base.update(kw)
    return Case(**base)


def test_cases_round_trip_through_the_log(tmp_path):
    conn = connect(tmp_path / "t.db")
    original = _case()
    insert_cases(conn, [original])
    (back,) = fetch_cases(conn)
    assert back.case_id == original.case_id
    assert back.evidence[0].signal == "sig"
    # A fetched case must be re-insertable: insert_cases calls .isoformat()
    # on the timestamp, which fails if it came back as a string.
    insert_cases(conn, [back])
    assert count_cases(conn) == 1, "re-inserting the same case must not duplicate"


def test_log_is_append_only_on_duplicate_ids(tmp_path):
    conn = connect(tmp_path / "t.db")
    c = _case()
    insert_cases(conn, [c])
    insert_cases(conn, [_case(case_id=c.case_id, decision="block")])
    (row,) = conn.execute("SELECT decision FROM cases").fetchall()
    assert row[0] == "escalate", "an existing audit row must never be rewritten"


def test_fetch_filters_by_agent(tmp_path):
    conn = connect(tmp_path / "t.db")
    insert_cases(conn, [_case(source_agent="a"), _case(source_agent="b")])
    assert len(fetch_cases(conn, source_agent="a")) == 1
    assert count_cases(conn, "b") == 1


# ---------------------------------------------------------------------------
# Adjudicator conflict detection
# ---------------------------------------------------------------------------

def test_cross_agent_conflict_is_detected():
    from agents.adjudicator import find_conflicts
    cases = [_case(source_agent="spike_sentinel", entity_id="X"),
             _case(source_agent="ring_detector", entity_id="X")]
    types = [c["type"] for c in find_conflicts(cases)]
    assert "cross_agent" in types


def test_model_only_flag_is_detected_as_a_conflict():
    from agents.adjudicator import find_conflicts
    bare = _case(source_agent="spike_sentinel",
                 evidence=[Evidence("model_probability", "P=0.99", 0.35)])
    types = [c["type"] for c in find_conflicts([bare])]
    assert "model_without_evidence" in types


def test_rule_backed_flag_is_not_a_conflict():
    from agents.adjudicator import find_conflicts
    backed = _case(source_agent="spike_sentinel",
                   evidence=[Evidence("high_amount", "$900", 0.2),
                             Evidence("model_probability", "P=0.99", 0.35)])
    types = [c["type"] for c in find_conflicts([backed])]
    assert "model_without_evidence" not in types
