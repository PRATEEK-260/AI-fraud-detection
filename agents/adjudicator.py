"""Adjudicator-in-Chief — arbitrates conflicting evidence in the audit log.

This is a rule-TRIGGERED LLM arbitration step, not an independent model. It
reads the append-only case log, finds cases where the evidence argues with
itself, sends both arguments to Claude verbatim, and writes a new Case
recording the resolution. It never mutates the original cases — the audit
trail keeps the disagreement and the ruling side by side.

Conflict types, and how honestly each one fires on this project's data:

  model_without_evidence   FIRES (461 cases). Spike Sentinel flagged the
                           transaction on model probability alone, with zero
                           interpretable rule signals behind it — and marked
                           several `block`. This is the sharpest conflict in
                           the system because it contradicts the project's own
                           thesis: a decision with no human-readable evidence
                           is exactly the bare score this system exists to
                           replace. The adjudicator's job is to decide whether
                           model confidence alone is ever enough to block.

  detector_split           FIRES when Content Forensics' interpretable signal
                           rules and its LLM judgment reach opposite verdicts
                           on the same text.

  llm_veto                 FIRES when Ring Detector's graph rules flagged a
                           cluster and the LLM verifier called it a legitimate
                           shared-device pattern (a family, an office).

  cross_agent              DOES NOT FIRE on this data, and saying so matters.
                           The three agents run on three different datasets
                           (a deliberate choice — see IMPLEMENTATION_PLAN.md
                           §4 — because it avoids fabricating device/IP
                           signal), so their entity universes are disjoint and
                           no entity_id can be flagged by two agents. The code
                           path is implemented and unit-checked with a
                           synthetic pair, because in a production deployment
                           all agents observe one entity universe and this
                           becomes the common case. It is reported as 0 rather
                           than manufactured by joining unrelated datasets.

Usage:
    .venv/bin/python -m agents.adjudicator [--limit N] [--no-llm]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from eval.cost_table import COSTS, ESCALATE_WHEN_RATIO_BELOW
from spine.db import DEFAULT_DB_PATH, connect, count_cases, fetch_cases, insert_cases
from spine.llm import REASONING_MODEL, chat_json
from spine.schema import Case, Evidence

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "eval" / "results"

ARBITER_MODEL = REASONING_MODEL

# Signals that constitute human-readable evidence, per agent. A case built
# only from signals OUTSIDE these sets is a bare score wearing a case file.
INTERPRETABLE_SIGNALS = {
    "spike_sentinel": {
        "velocity_burst", "amount_spike", "high_amount", "night_high_amount",
    },
    "content_forensics": {
        "low_burstiness", "high_readability", "template_reuse",
        "low_lexical_diversity",
    },
    "ring_detector": {
        "shared_device_ring", "shared_ip_ring", "signup_burst",
        "instant_purchase",
    },
}


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

def _signals(case: Case) -> set[str]:
    return {e.signal for e in case.evidence}


def find_conflicts(cases: list[Case]) -> list[dict]:
    """Locate cases whose evidence argues with itself, or with another agent."""
    conflicts: list[dict] = []

    # 1. Cross-agent: one entity, two or more agents.
    by_entity: dict[str, list[Case]] = defaultdict(list)
    for c in cases:
        by_entity[c.entity_id].append(c)
    for entity_id, group in by_entity.items():
        agents = {c.source_agent for c in group}
        if len(agents) > 1:
            conflicts.append({
                "type": "cross_agent",
                "entity_id": entity_id,
                "cases": group,
                "question": (
                    f"{len(agents)} agents independently flagged entity "
                    f"{entity_id}. Do their evidence sets corroborate each "
                    f"other or contradict each other?"
                ),
            })

    for c in cases:
        sigs = _signals(c)
        interpretable = INTERPRETABLE_SIGNALS.get(c.source_agent, set())

        # 2. Model-only flag: a decision with no human-readable evidence.
        if c.source_agent == "spike_sentinel" and not (sigs & interpretable):
            conflicts.append({
                "type": "model_without_evidence",
                "entity_id": c.entity_id,
                "cases": [c],
                "question": (
                    "The gradient-boosted model flagged this transaction with "
                    "high confidence, but not one interpretable velocity or "
                    "amount rule fired — there is no human-readable reason "
                    "for the flag. Is model confidence alone sufficient to "
                    "BLOCK a payment, or must this be escalated to a human?"
                ),
            })

        # 3. Detector split inside Content Forensics: the signal rules and the
        #    LLM reached opposite conclusions about the same text.
        if c.source_agent == "content_forensics":
            has_llm = any(s.startswith("llm_reason") for s in sigs)
            has_signals = bool(sigs & interpretable)
            if has_llm != has_signals:
                conflicts.append({
                    "type": "detector_split",
                    "entity_id": c.entity_id,
                    "cases": [c],
                    "question": (
                        "The interpretable linguistic signals and the LLM "
                        "forensic judgment do not agree on this text. Which "
                        "argument is stronger, and does the disagreement "
                        "warrant denying a customer's dispute?"
                    ),
                })

        # 4. LLM veto on a rule-flagged ring.
        if c.source_agent == "ring_detector":
            # Match the verdict marker the Ring Detector writes, not the bare
            # word: the verifier's own prose says things like "unusual for
            # legitimate shared family use" while ruling the cluster a RING,
            # and a substring test turned those into phantom vetoes.
            if any(s.startswith("llm_verification") for s in sigs) and \
                    "LLM verification: legitimate" in c.reasoning_text:
                conflicts.append({
                    "type": "llm_veto",
                    "entity_id": c.entity_id,
                    "cases": [c],
                    "question": (
                        "The graph rules flagged this cluster as a coordinated "
                        "ring, but the LLM verifier judged it a legitimate "
                        "shared-device pattern. Whose call stands?"
                    ),
                })
    return conflicts


# ---------------------------------------------------------------------------
# Arbitration
# ---------------------------------------------------------------------------

ARBITER_SYSTEM_PROMPT = (
    "You are the adjudicator on a payments risk desk. Two detection layers "
    "have produced conflicting or incomplete evidence about one entity. Your "
    "job is NOT to re-score the entity — you cannot see the raw data. Your "
    "job is to weigh the ARGUMENTS as presented and choose the action.\n\n"
    "Governing principles:\n"
    "1. A decision to BLOCK must rest on evidence a human reviewer could "
    "read and check. A high model score with no interpretable signal behind "
    "it is not that, however confident the model is.\n"
    "2. Blocking costs real money and real customer trust, and the cost is "
    "asymmetric per agent — you are given the cost ratio.\n"
    "3. ESCALATE is not a failure state. It is the correct answer when the "
    "evidence is genuinely ambiguous.\n"
    "4. Never invent evidence that was not presented to you.\n\n"
    'Respond with JSON only: {"decision": "allow" | "escalate" | "block", '
    '"winning_argument": "<which layer you found more persuasive>", '
    '"rationale": "<2-3 sentences a human reviewer would act on>", '
    '"confidence": 0-1}'
)


def _format_arguments(conflict: dict) -> str:
    """Both sides of the disagreement, verbatim — the PDF promises the human
    reviewer sees the arguments as they were made, not a summary of them."""
    parts = [f"CONFLICT TYPE: {conflict['type']}",
             f"QUESTION: {conflict['question']}", ""]
    for i, case in enumerate(conflict["cases"], 1):
        agent = case.source_agent
        cost = COSTS.get(agent)
        parts.append(f"--- ARGUMENT {i}: agent `{agent}` ---")
        parts.append(f"entity: {case.entity_id} ({case.entity_type})")
        parts.append(f"its decision: {case.decision}  "
                     f"confidence: {case.confidence:.3f}")
        if cost:
            ratio = (cost["false_negative"]["cost"]
                     / cost["false_positive"]["cost"])
            parts.append(
                f"cost asymmetry for this agent: a false positive costs "
                f"Rs {cost['false_positive']['cost']:,.0f} "
                f"({cost['false_positive']['what_breaks']}); a false negative "
                f"costs Rs {cost['false_negative']['cost']:,.0f} "
                f"({cost['false_negative']['what_breaks']}). "
                f"FN:FP ratio {ratio:.2f}.")
        parts.append("evidence presented:")
        if case.evidence:
            for e in case.evidence:
                parts.append(f"  - [{e.signal}] {e.value}  (weight {e.weight})")
        else:
            parts.append("  (none)")
        parts.append(f"stated reasoning: {case.reasoning_text}")
        parts.append("")
    return "\n".join(parts)


def arbitrate(conflict: dict, use_llm: bool = True) -> dict:
    """Ask the arbiter to rule. Falls back to a conservative rule offline."""
    argument_text = _format_arguments(conflict)
    if not use_llm:
        return {
            "decision": "escalate",
            "winning_argument": "n/a (offline fallback)",
            "rationale": "LLM arbitration disabled; conservative default is to "
                         "escalate rather than auto-action a contested case.",
            "confidence": 0.5,
            "arguments": argument_text,
        }
    out = chat_json(
        [{"role": "system", "content": ARBITER_SYSTEM_PROMPT},
         {"role": "user", "content": argument_text + "\nJSON ruling:"}],
        model=ARBITER_MODEL, temperature=0.0, max_tokens=400,
    )
    decision = out.get("decision", "escalate")
    if decision not in ("allow", "escalate", "block"):
        decision = "escalate"
    return {
        "decision": decision,
        "winning_argument": str(out.get("winning_argument", ""))[:200],
        "rationale": str(out.get("rationale", ""))[:600],
        "confidence": float(out.get("confidence", 0.5)),
        "arguments": argument_text,
    }


def make_case(conflict: dict, ruling: dict) -> Case:
    source_cases = conflict["cases"]
    evidence = [
        Evidence("conflict_type", conflict["type"], 0.20),
        Evidence("question_put_to_arbiter", conflict["question"], 0.10),
    ]
    for case in source_cases:
        evidence.append(Evidence(
            f"source_case:{case.source_agent}",
            f"case {case.case_id[:8]} decided `{case.decision}` at confidence "
            f"{case.confidence:.3f} on {len(case.evidence)} evidence items",
            0.20))
    evidence.append(Evidence(
        "arbiter_ruling",
        f"{ruling['decision']} — {ruling['rationale']}",
        round(0.40 * ruling["confidence"], 4)))
    if ruling["winning_argument"]:
        evidence.append(Evidence(
            "winning_argument", ruling["winning_argument"], 0.10))

    original = source_cases[0].decision
    overturned = ruling["decision"] != original
    return Case(
        source_agent="adjudicator",
        entity_id=conflict["entity_id"],
        entity_type=source_cases[0].entity_type,
        evidence=evidence,
        confidence=float(ruling["confidence"]),
        cost_estimate=max(c.cost_estimate for c in source_cases),
        decision=ruling["decision"],
        reasoning_text=(
            f"[{conflict['type']}] Original decision by "
            f"{source_cases[0].source_agent}: `{original}`. "
            f"Adjudicator ruling: `{ruling['decision']}`"
            f"{' (OVERTURNED)' if overturned else ' (upheld)'}. "
            f"{ruling['rationale']} "
            f"Winning argument: {ruling['winning_argument']}."
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=25,
                    help="max conflicts to arbitrate this run")
    ap.add_argument("--no-llm", action="store_true",
                    help="skip arbitration calls (offline runs)")
    ap.add_argument("--fetch", type=int, default=5000,
                    help="how many cases to read from the audit log")
    return ap.parse_args()


def self_check_cross_agent() -> bool:
    """The cross_agent path cannot fire on this project's data (disjoint
    datasets), so it is verified here against a synthetic pair instead of
    being left silently untested."""
    a = Case(source_agent="spike_sentinel", entity_id="ENT-1",
             entity_type="account", decision="block", confidence=0.9,
             evidence=[Evidence("high_amount", "Rs 90,000", 0.2)])
    b = Case(source_agent="ring_detector", entity_id="ENT-1",
             entity_type="account", decision="escalate", confidence=0.6,
             evidence=[Evidence("shared_device_ring", "4 accounts", 0.35)])
    found = [c for c in find_conflicts([a, b]) if c["type"] == "cross_agent"]
    return len(found) == 1 and len(found[0]["cases"]) == 2


def main() -> None:
    args = parse_args()

    ok = self_check_cross_agent()
    print(f"cross_agent path self-check: {'PASS' if ok else 'FAIL'} "
          f"(verified on a synthetic pair; see module docstring for why it "
          f"cannot fire on this project's data)")

    conn = connect(DEFAULT_DB_PATH)
    cases = fetch_cases(conn, limit=args.fetch)
    print(f"Read {len(cases):,} cases from the audit log "
          f"({count_cases(conn):,} total)")

    conflicts = find_conflicts(cases)
    by_type: dict[str, int] = defaultdict(int)
    for c in conflicts:
        by_type[c["type"]] += 1
    print(f"\nConflicts found: {len(conflicts):,}")
    for t in ("cross_agent", "model_without_evidence", "detector_split",
              "llm_veto"):
        n = by_type.get(t, 0)
        note = "  <- does not fire on this data, by construction" if (
            t == "cross_agent" and n == 0) else ""
        print(f"  {t:<26} {n:>6,}{note}")

    # Arbitrate the most consequential conflicts first: those whose source
    # case would otherwise auto-block.
    conflicts.sort(key=lambda c: (
        0 if any(x.decision == "block" for x in c["cases"]) else 1,
        -max(x.confidence for x in c["cases"]),
    ))
    todo = conflicts[: args.limit]

    rulings, t0 = [], time.perf_counter()
    for i, conflict in enumerate(todo, 1):
        ruling = arbitrate(conflict, use_llm=not args.no_llm)
        rulings.append((conflict, ruling))
        if i % 10 == 0:
            print(f"  arbitrated {i}/{len(todo)}")
    elapsed = time.perf_counter() - t0

    cases_out = [make_case(c, r) for c, r in rulings]
    insert_cases(conn, cases_out)

    overturned = sum(
        1 for c, r in rulings if r["decision"] != c["cases"][0].decision)
    outcome: dict[str, int] = defaultdict(int)
    for _, r in rulings:
        outcome[r["decision"]] += 1

    print(f"\nArbitrated {len(rulings)} conflicts in {elapsed:.1f}s "
          f"({elapsed/max(len(rulings),1):.2f}s each, disk-cached on re-runs)")
    print(f"  overturned the original decision: {overturned}/{len(rulings)}")
    print(f"  rulings: {dict(outcome)}")
    print(f"\nWrote {len(cases_out)} adjudicator cases; audit log now holds "
          f"{count_cases(conn, 'adjudicator')} adjudicator / "
          f"{count_cases(conn)} total")

    if rulings:
        conflict, ruling = rulings[0]
        print("\n" + "=" * 72)
        print("HERO CASE — the disagreement to walk through in the pitch")
        print("=" * 72)
        print(ruling["arguments"])
        print(f"RULING: {ruling['decision'].upper()} "
              f"(confidence {ruling['confidence']:.2f})")
        print(f"WINNING ARGUMENT: {ruling['winning_argument']}")
        print(f"RATIONALE: {ruling['rationale']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "agent": "adjudicator",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "design": "rule-triggered LLM arbitration over the audit log; never "
                  "mutates source cases, appends a ruling case instead",
        "arbiter_model": ARBITER_MODEL,
        "cross_agent_selfcheck_passed": ok,
        "cases_read": len(cases),
        "conflicts_found": dict(by_type),
        "cross_agent_zero_reason": (
            "the three agents run on three different datasets, so no entity_id "
            "is observable by more than one agent; the path is implemented and "
            "self-checked rather than manufactured by joining unrelated data"
        ),
        "conflicts_arbitrated": len(rulings),
        "decisions_overturned": overturned,
        "ruling_distribution": dict(outcome),
        "seconds_per_arbitration": round(elapsed / max(len(rulings), 1), 2),
        "escalate_when_ratio_below": ESCALATE_WHEN_RATIO_BELOW,
        "hero_case": {
            "conflict_type": rulings[0][0]["type"],
            "entity_id": rulings[0][0]["entity_id"],
            "arguments": rulings[0][1]["arguments"],
            "ruling": rulings[0][1]["decision"],
            "rationale": rulings[0][1]["rationale"],
        } if rulings else None,
    }
    out_path = RESULTS_DIR / "adjudicator_metrics.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved metrics to {out_path}")


if __name__ == "__main__":
    main()
