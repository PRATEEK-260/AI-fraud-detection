"""Agentic Checkout Guard — is this checkout a person, and is it in bounds?

The attack this addresses (project doc §1): an AI shopping agent transacting
on a user's behalf that exceeds what its owner actually authorised — wrong
amount, wrong merchant category, wrong frequency — or that is running on a
stolen card while presenting itself as a browser.

Two halves, and they are NOT of equal evidential quality. Say so out loud:

  1. MANDATE POLICY ENGINE — real. "Did this transaction breach the spend cap,
     leave the allowed categories, or exceed the velocity limit its owner
     granted?" is deterministic logic evaluated against a mandate the user
     sets. It is not learned, it cannot be wrong about its own inputs, and it
     would ship into production unchanged. This is the half that matters, and
     it is the half the industry will actually need first: agent commerce
     needs a machine-readable mandate before it needs a behavioural detector.

  2. BEHAVIOURAL FINGERPRINT — speculative. Timing cadence, response floor,
     passive-event density, fingerprint completeness. There is no public
     corpus of real AI-agent checkout sessions, so it is evaluated against
     scripts/simulate_sessions.py. **Every precision/recall figure this agent
     reports about agent-detection describes how separable that simulation is,
     not how it would perform on real traffic.** A simulation cannot validate
     a detector against an adversary that does not exist yet.

Why both are still worth building: the policy engine degrades gracefully. If
the behavioural half is wrong about whether a session is an agent, a declared
agent breaching its mandate is still caught by the policy engine, which needs
no inference at all. The design deliberately puts the load-bearing decision on
the deterministic half.

Decision logic:
    human                        -> allow  (not this agent's jurisdiction)
    agent, inside its mandate    -> allow  (authorised agents must NOT be
                                            punished for being agents)
    agent, outside its mandate   -> cost-table policy gate
    undeclared agent, high conf  -> escalate (identity, not spend, is the issue)

Usage:
    .venv/bin/python -m agents.checkout_guard [--fresh]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eval.cost_table import decide
from eval.metrics import binary_metrics, format_report, pr_auc
from spine.db import DEFAULT_DB_PATH, connect, count_cases, insert_cases
from spine.schema import Case, Evidence

ROOT = Path(__file__).resolve().parents[1]
SESSIONS_PATH = ROOT / "data" / "processed" / "checkout_sessions.parquet"
RESULTS_DIR = ROOT / "eval" / "results"

HELDOUT_FRACTION = 0.30

# Interpretable behavioural rules — these become the evidence in a Case.
RULES: dict[str, dict] = {
    "metronomic_cadence": {
        "weight": 0.30,
        "desc": "inter-action intervals too regular for human hesitation",
    },
    "inhuman_response": {
        "weight": 0.25,
        "desc": "acted faster than human perception-plus-motor response",
    },
    "no_passive_events": {
        "weight": 0.20,
        "desc": "almost no scrolling or pointer movement — nothing was read",
    },
    "thin_fingerprint": {
        "weight": 0.25,
        "desc": "browser fingerprint surface missing or automation user-agent",
    },
}

# Below this, a person cannot see a page change and respond to it.
HUMAN_RESPONSE_FLOOR_MS = 250.0

FEATURE_COLS = [
    "gap_cv", "gap_median", "gap_min", "min_response_ms",
    "passive_per_action", "n_actions", "fingerprint_score",
    "is_automation_ua", "session_duration_s",
]


# ---------------------------------------------------------------------------
# Behavioural signals
# ---------------------------------------------------------------------------

def compute_signals(row: pd.Series) -> dict:
    gaps = np.asarray(json.loads(row["gaps_json"]), dtype=float)
    mean = float(gaps.mean()) if len(gaps) else 0.0
    # Coefficient of variation: human hesitation is irregular, a loop is not.
    cv = float(gaps.std() / mean) if mean > 0 else 0.0
    fingerprint = sum((
        bool(row["has_accept_language"]),
        bool(row["has_canvas_fingerprint"]),
        bool(row["has_webgl"]),
        bool(row["cookies_enabled"]),
    )) / 4.0
    ua = str(row["user_agent"])
    return {
        "gap_cv": cv,
        "gap_median": float(np.median(gaps)) if len(gaps) else 0.0,
        "gap_min": float(gaps.min()) if len(gaps) else 0.0,
        "min_response_ms": float(row["min_response_ms"]),
        "passive_per_action": float(row["passive_events"]) / max(int(row["n_actions"]), 1),
        "n_actions": float(row["n_actions"]),
        "fingerprint_score": fingerprint,
        "is_automation_ua": float(not ua.startswith("Mozilla/5.0")),
        "session_duration_s": float(row["session_duration_s"]),
    }


def apply_rules(sig: pd.DataFrame) -> pd.DataFrame:
    hits = pd.DataFrame(index=sig.index)
    hits["metronomic_cadence"] = sig["gap_cv"] < 0.35
    hits["inhuman_response"] = sig["min_response_ms"] < HUMAN_RESPONSE_FLOOR_MS
    hits["no_passive_events"] = sig["passive_per_action"] < 1.0
    hits["thin_fingerprint"] = (sig["fingerprint_score"] < 0.75) | \
                               (sig["is_automation_ua"] > 0)
    out = sig.copy()
    for name, series in hits.items():
        out[f"rule_{name}"] = series
    out["rule_score"] = sum(
        hits[name] * RULES[name]["weight"] for name in RULES)
    return out


# ---------------------------------------------------------------------------
# Mandate policy engine — deterministic, the load-bearing half
# ---------------------------------------------------------------------------

def evaluate_mandate(row: pd.Series) -> dict:
    """Check a transaction against the authority its owner granted.

    No inference, no thresholds, no training. Either the transaction is inside
    the bounds the user set or it is not. A session with no mandate (a human,
    or an agent that never registered one) returns `has_mandate=False` — which
    is itself a finding for a session the behavioural layer thinks is an agent.
    """
    raw = row.get("mandate_json") or ""
    if not raw:
        return {"has_mandate": False, "breaches": [], "within_bounds": None}

    mandate = json.loads(raw)
    breaches = []
    if float(row["amount"]) > float(mandate["max_amount"]):
        breaches.append({
            "bound": "max_amount",
            "authorised": f"Rs {mandate['max_amount']:,.0f}",
            "observed": f"Rs {float(row['amount']):,.2f}",
            "detail": f"exceeded its spend cap by "
                      f"{float(row['amount']) / float(mandate['max_amount']) - 1:.0%}",
        })
    if row["merchant_category"] not in mandate["allowed_categories"]:
        breaches.append({
            "bound": "allowed_categories",
            "authorised": ", ".join(mandate["allowed_categories"]),
            "observed": str(row["merchant_category"]),
            "detail": f"bought from `{row['merchant_category']}`, which the "
                      f"owner never authorised",
        })
    if int(row["txns_last_hour"]) > int(mandate["max_txns_per_hour"]):
        breaches.append({
            "bound": "max_txns_per_hour",
            "authorised": str(mandate["max_txns_per_hour"]),
            "observed": str(row["txns_last_hour"]),
            "detail": f"{row['txns_last_hour']} transactions in the last hour "
                      f"against a limit of {mandate['max_txns_per_hour']}",
        })
    return {"has_mandate": True, "breaches": breaches,
            "within_bounds": len(breaches) == 0, "mandate": mandate}


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------

def make_case(row: pd.Series, sig: dict, p_agent: float, policy: dict) -> Case:
    evidence: list[Evidence] = []

    if sig["rule_metronomic_cadence"]:
        evidence.append(Evidence(
            "metronomic_cadence",
            f"inter-action interval variation {sig['gap_cv']:.2f} "
            f"(human sessions rarely below 0.35)",
            RULES["metronomic_cadence"]["weight"]))
    if sig["rule_inhuman_response"]:
        evidence.append(Evidence(
            "inhuman_response",
            f"responded in {sig['min_response_ms']:.0f}ms, below the "
            f"{HUMAN_RESPONSE_FLOOR_MS:.0f}ms human perception-motor floor",
            RULES["inhuman_response"]["weight"]))
    if sig["rule_no_passive_events"]:
        evidence.append(Evidence(
            "no_passive_events",
            f"{sig['passive_per_action']:.1f} scroll/pointer events per action "
            f"— the page was not read",
            RULES["no_passive_events"]["weight"]))
    if sig["rule_thin_fingerprint"]:
        evidence.append(Evidence(
            "thin_fingerprint",
            f"fingerprint completeness {sig['fingerprint_score']:.2f}, "
            f"user-agent `{str(row['user_agent'])[:40]}`",
            RULES["thin_fingerprint"]["weight"]))

    # Mandate breaches are the strongest evidence available — deterministic.
    for breach in policy["breaches"]:
        evidence.append(Evidence(
            f"mandate_breach:{breach['bound']}",
            f"{breach['detail']} (authorised: {breach['authorised']}; "
            f"observed: {breach['observed']})",
            0.40))

    if row["declared_agent"]:
        evidence.append(Evidence(
            "declared_agent", "client identified itself as an automated agent",
            0.15))
    elif p_agent >= 0.5:
        evidence.append(Evidence(
            "undeclared_agent",
            "behavioural signals indicate an automated client that did not "
            "declare itself", 0.20))

    evidence.append(Evidence(
        "agent_probability", f"P(automated client) = {p_agent:.3f}", 0.10))

    breached = bool(policy["breaches"])
    # Confidence in the ACTION, which is about the mandate breach when there is
    # one — that is deterministic, so it does not inherit the detector's doubt.
    confidence = 1.0 if breached else float(p_agent)

    if breached:
        has_readable = True          # a breach is inherently human-readable
        decision, why = decide("checkout_guard", confidence, has_readable)
    elif policy["has_mandate"]:
        decision, why = "allow", (
            "authorised agent operating inside the mandate its owner granted "
            "— being an agent is not itself a violation")
    elif p_agent >= 0.80:
        decision, why = "escalate", (
            "probable automated client with no registered mandate: the issue "
            "is unverified authority, not spend")
    else:
        decision, why = "allow", "session consistent with a human checkout"

    return Case(
        source_agent="checkout_guard",
        entity_id=str(row["session_id"]),
        entity_type="session",
        evidence=evidence,
        confidence=round(confidence, 4),
        cost_estimate=round(float(row["amount"]), 2),
        decision=decision,
        reasoning_text=(
            f"Checkout session {row['session_id']}: "
            f"Rs {float(row['amount']):,.2f} at `{row['merchant_category']}`. "
            f"P(automated client) {p_agent:.3f}"
            f"{', declared' if row['declared_agent'] else ''}. "
            + (f"MANDATE BREACH: "
               + "; ".join(b["detail"] for b in policy["breaches"]) + ". "
               if breached else
               ("Within mandate bounds. " if policy["has_mandate"]
                else "No mandate registered. "))
            + f"Action `{decision}`: {why}."
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _stable_hash(value: str) -> int:
    return int(hashlib.md5(str(value).encode()).hexdigest()[:8], 16)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-cases", type=int, default=400,
                    help="cap on cases written to the audit log this run")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if not SESSIONS_PATH.exists():
        raise SystemExit(
            f"{SESSIONS_PATH} missing — run:\n"
            f"    .venv/bin/python scripts/simulate_sessions.py")

    df = pd.read_parquet(SESSIONS_PATH)
    df["split"] = np.where(
        df["session_id"].map(lambda s: _stable_hash(s) % 100)
        < HELDOUT_FRACTION * 100, "heldout", "train")

    print("=" * 72)
    print("SIMULATED SESSIONS. The agent-detection metrics below measure how "
          "separable\nthis simulation is — NOT real-world performance. No "
          "public corpus of AI\nshopping-agent checkouts exists. The mandate "
          "policy engine is deterministic\nand is the half that would ship "
          "unchanged.")
    print("=" * 72)

    t0 = time.perf_counter()
    sig = pd.DataFrame([compute_signals(r) for _, r in df.iterrows()],
                       index=df.index)
    sig = apply_rules(sig)
    signal_ms = (time.perf_counter() - t0) * 1000 / max(len(df), 1)

    train_mask = df["split"] == "train"
    held = df[~train_mask]
    sig_held = sig[~train_mask]

    print(f"\nsessions {len(df):,}  (train {int(train_mask.sum()):,} / "
          f"heldout {len(held):,})   agent rate "
          f"{df['is_agent'].mean():.3f}   hard cases "
          f"{df['is_hard_case'].mean():.0%}")

    # --- agent detection ----------------------------------------------------
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, C=1.0))
    model.fit(sig[train_mask][FEATURE_COLS], df.loc[train_mask, "is_agent"])
    p_held = model.predict_proba(sig_held[FEATURE_COLS])[:, 1]

    results = {
        "rules_only": binary_metrics(held["is_agent"], sig_held["rule_score"] >= 0.50),
        "logistic_regression": {
            **binary_metrics(held["is_agent"], p_held >= 0.5),
            "pr_auc": pr_auc(held["is_agent"], p_held),
        },
    }
    for name, m in results.items():
        print(format_report(f"[held-out, SIMULATED] agent detection / {name}", m))

    # Where the detector actually errs — the honest part of a simulated eval.
    hard = held["is_hard_case"] == 1
    pred = p_held >= 0.5
    results["by_difficulty"] = {
        "easy_cases": binary_metrics(held.loc[~hard, "is_agent"], pred[~hard.to_numpy()]),
        "hard_cases": binary_metrics(held.loc[hard, "is_agent"], pred[hard.to_numpy()]),
    }
    print(format_report("[held-out] EASY cases (naive bots, unhurried humans)",
                        results["by_difficulty"]["easy_cases"]))
    print(format_report("[held-out] HARD cases (jittered agents, rushed humans)",
                        results["by_difficulty"]["hard_cases"]))

    # --- mandate policy engine ---------------------------------------------
    policies = [evaluate_mandate(r) for _, r in held.iterrows()]
    with_mandate = [p for p in policies if p["has_mandate"]]
    breached = [p for p in with_mandate if p["breaches"]]
    rogue_truth = held["is_rogue"].to_numpy()
    breach_pred = np.array([1 if p["breaches"] else 0 for p in policies])
    policy_metrics = binary_metrics(rogue_truth, breach_pred)
    print(format_report(
        "[held-out] mandate policy engine (DETERMINISTIC, not learned)",
        policy_metrics))
    print("  ^ TAUTOLOGICAL on simulated data, and must be presented that "
          "way: the\n    simulator DEFINES a rogue agent as one that breaches "
          "a bound, and this\n    engine checks bounds. A perfect score here "
          "confirms the implementation\n    is correct — it is not evidence "
          "the approach catches real rogue agents.\n    What it does show is "
          "that when a mandate exists, catching a breach needs\n    no "
          "inference at all.")

    breach_kinds: dict[str, int] = {}
    for p in breached:
        for b in p["breaches"]:
            breach_kinds[b["bound"]] = breach_kinds.get(b["bound"], 0) + 1
    print(f"\nMandates evaluated: {len(with_mandate):,}; breaching: "
          f"{len(breached):,}  {breach_kinds}")

    # --- cases --------------------------------------------------------------
    cases = []
    n_allowed_agents = 0
    for i, ((idx, row), policy) in enumerate(zip(held.iterrows(), policies)):
        if len(cases) >= args.max_cases:
            break
        s = sig_held.loc[idx].to_dict()
        case = make_case(row, s, float(p_held[i]), policy)
        if case.decision != "allow":
            cases.append(case)
        elif (policy["has_mandate"] and policy["within_bounds"]
              and float(p_held[i]) >= 0.80 and n_allowed_agents < 40):
            # Deliberate allows worth auditing: the behavioural layer is
            # confident this was an automated client and the system let it
            # through because its owner authorised the spend. A risk system
            # that cannot show why it did NOT act is as opaque as one that
            # cannot show why it did. Ordinary human allows are not recorded —
            # they would swamp the log to no purpose.
            cases.append(case)
            n_allowed_agents += 1

    conn = connect(DEFAULT_DB_PATH)
    insert_cases(conn, cases)
    print(f"\nWrote {len(cases)} checkout_guard cases "
          f"({n_allowed_agents} of them deliberate allows of authorised "
          f"agents); "
          f"audit log now holds {count_cases(conn, 'checkout_guard')} "
          f"checkout_guard / {count_cases(conn)} total")
    if cases:
        c = max(cases, key=lambda x: len(x.evidence))
        print(f"\nExample case {c.case_id[:8]} [{c.decision}]:\n  "
              f"{c.reasoning_text}")

    # --- save ---------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "agent": "checkout_guard",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "DATA_WARNING": (
            "Sessions are SIMULATED by scripts/simulate_sessions.py. No public "
            "corpus of AI shopping-agent checkouts exists. The agent-detection "
            "metrics below measure the separability of that simulation and "
            "must NOT be read as real-world detection performance. The mandate "
            "policy engine is deterministic and carries no such caveat."
        ),
        "dataset": {
            "n_sessions": len(df),
            "n_heldout": len(held),
            "agent_rate": round(float(df["is_agent"].mean()), 4),
            "rogue_rate": round(float(df["is_rogue"].mean()), 4),
            "hard_case_fraction": round(float(df["is_hard_case"].mean()), 3),
            "simulator_seed": 1337,
        },
        "agent_detection_SIMULATED": results,
        "mandate_policy_engine": {
            "nature": "deterministic bounds check, not a learned model",
            "why_the_perfect_score_is_not_a_result": (
                "The simulator defines a rogue agent as one breaching at least "
                "one mandate bound, and this engine checks those bounds, so "
                "1.00/1.00 is tautological. It verifies the implementation, "
                "not the approach. The transferable claim is narrower and "
                "still useful: where a machine-readable mandate exists, a "
                "breach is detectable deterministically, with no model risk."
            ),
            "metrics_vs_rogue_label": policy_metrics,
            "mandates_evaluated": len(with_mandate),
            "breaching": len(breached),
            "breach_kinds": breach_kinds,
        },
        "rules": {k: v["desc"] for k, v in RULES.items()},
        "human_response_floor_ms": HUMAN_RESPONSE_FLOOR_MS,
        "signal_ms_per_session": round(signal_ms, 3),
        "cases_written": len(cases),
        "decisions": {
            d: int(sum(c.decision == d for c in cases))
            for d in ("block", "escalate", "allow")
        },
    }
    out_path = RESULTS_DIR / "checkout_guard_metrics.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
