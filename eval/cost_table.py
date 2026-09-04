"""False-positive / false-negative cost model, shared across agents.

The point of this table is that a wrongful block does not cost the same
amount depending on which agent triggered it, so a single global threshold is
the wrong design. Each agent carries its own asymmetry, and that asymmetry is
what decides whether a flag becomes `block` or `escalate`.

Every number here is a hand-reasoned estimate for a mid-size Indian merchant
portfolio, not a measured figure — they are ORDERS OF MAGNITUDE meant to make
the relative weighting explicit and arguable, and they are labelled as such
wherever they are reported. The structure (which error dominates, and by how
much) is the claim; the absolute rupee values are not.

Usage:
    .venv/bin/python -m eval.cost_table
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "eval" / "results"

# INR. Assumptions are stated per agent so a reviewer can argue with them
# rather than having to reverse-engineer them.
COSTS: dict[str, dict] = {
    "spike_sentinel": {
        "entity": "transaction",
        "false_positive": {
            "cost": 850.0,
            "what_breaks": "a legitimate high-value transaction is declined",
            "assumption": "~10% of a declined basket's value is lost to "
                          "abandonment and merchant trust damage; mean flagged "
                          "basket in the held-out set is roughly Rs 8,500. "
                          "A declined customer retries elsewhere ~40% of the "
                          "time and a share never returns.",
        },
        "false_negative": {
            "cost": 9200.0,
            "what_breaks": "fraud settles, then arrives as a chargeback",
            "assumption": "full transaction value written off plus a "
                          "Rs 1,500-2,500 chargeback handling fee; card "
                          "networks also penalise merchants whose chargeback "
                          "ratio crosses ~0.9%, which this ignores.",
        },
    },
    "ring_detector": {
        "entity": "account cluster",
        "false_positive": {
            "cost": 4200.0,
            "what_breaks": "a family, office, or shared-device household is "
                           "locked out together",
            "assumption": "the whole cluster is actioned at once, so one wrong "
                          "call multiplies across every member: ~3-6 accounts "
                          "x support handling + lost lifetime value. This is "
                          "the agent where a false positive is least "
                          "recoverable, because the affected users are "
                          "unrelated to each other and all get punished.",
        },
        "false_negative": {
            "cost": 31000.0,
            "what_breaks": "a coordinated ring keeps operating",
            "assumption": "a ring is not one loss but a stream: a missed "
                          "cluster keeps transacting until caught by another "
                          "control. Sized as ~5 accounts x several "
                          "transactions before independent detection.",
        },
    },
    "checkout_guard": {
        "entity": "checkout session",
        "false_positive": {
            "cost": 1600.0,
            "what_breaks": "a legitimate agent-assisted or hurried human "
                           "checkout is interrupted",
            "assumption": "a step-up challenge on a good session costs "
                          "conversion plus support contact. Deliberately "
                          "priced ABOVE a Spike Sentinel false positive: "
                          "wrongly blocking an agent a customer authorised "
                          "breaks a delegated workflow they set up on "
                          "purpose, and agent commerce is exactly the "
                          "behaviour a payments company wants to encourage.",
        },
        "false_negative": {
            "cost": 11000.0,
            "what_breaks": "an agent transacts outside the authority its "
                           "owner granted",
            "assumption": "the owner disputes a charge they never authorised, "
                          "so the loss is the transaction plus a chargeback "
                          "plus the trust cost of a mandate that did not hold. "
                          "A breached mandate also tends to repeat until "
                          "revoked.",
        },
    },
    "document_forensics": {
        "entity": "KYC document image",
        "false_positive": {
            "cost": 3100.0,
            "what_breaks": "a genuine customer's identity document is "
                           "rejected as forged",
            "assumption": "onboarding abandonment plus manual re-review, and "
                          "the customer is accused of forgery on the evidence "
                          "of a compression artifact. Priced high on purpose: "
                          "ELA false-positives on legitimately re-saved or "
                          "phone-compressed documents are common, so this "
                          "agent must never auto-reject.",
        },
        "false_negative": {
            "cost": 6200.0,
            "what_breaks": "a tampered identity document passes KYC",
            "assumption": "downstream account-takeover and mule-account risk "
                          "rather than a direct loss at onboarding; a single "
                          "forged KYC can open a channel used repeatedly.",
        },
    },
    "content_forensics": {
        "entity": "document / dispute narrative",
        "false_positive": {
            "cost": 2400.0,
            "what_breaks": "a genuine customer's dispute is wrongly denied",
            "assumption": "regulatory and reputational exposure dominates the "
                          "direct amount: a wrongly denied complaint can "
                          "escalate to the banking ombudsman, and denying a "
                          "real victim of fraud a second time is the single "
                          "worst customer outcome in this system.",
        },
        "false_negative": {
            "cost": 5600.0,
            "what_breaks": "a fabricated dispute is paid out",
            "assumption": "refund value plus handling; AI-generated dispute "
                          "narratives are cheap to mass-produce, so the real "
                          "risk is volume rather than any single payout.",
        },
    },
}

# Cost ratio above which a flag must NOT auto-block, because being wrong is
# too expensive relative to being right. This is the rule that turns the table
# into behaviour instead of decoration.
ESCALATE_WHEN_RATIO_BELOW = 2.5


def summarize() -> dict:
    out = {}
    for agent, spec in COSTS.items():
        fp = spec["false_positive"]["cost"]
        fn = spec["false_negative"]["cost"]
        ratio = fn / fp
        out[agent] = {
            "entity": spec["entity"],
            "false_positive_inr": fp,
            "false_negative_inr": fn,
            "fn_to_fp_ratio": round(ratio, 2),
            "policy": (
                "auto-block permitted at high confidence"
                if ratio >= ESCALATE_WHEN_RATIO_BELOW
                else "escalate to a human rather than auto-block — a wrong "
                     "block costs too much relative to a miss"
            ),
            "false_positive_effect": spec["false_positive"]["what_breaks"],
            "false_negative_effect": spec["false_negative"]["what_breaks"],
            "assumptions": {
                "false_positive": spec["false_positive"]["assumption"],
                "false_negative": spec["false_negative"]["assumption"],
            },
        }
    return out


def decide(agent: str, confidence: float, has_interpretable_evidence: bool,
           vetoed: bool = False, block_at: float = 0.95) -> tuple[str, str]:
    """The single place a flag becomes an action. Returns (decision, why).

    Two gates, in order:

    1. EVIDENCE GATE. A block must rest on something a human reviewer can
       read and check. A model probability is not that, no matter how high —
       a case built only from a score is the bare score this system exists to
       replace. Auditing the log with agents/adjudicator.py found 461 Spike
       Sentinel cases that had been auto-blocked on model confidence with
       zero interpretable signals behind them; this gate is the fix.
    2. COST GATE. Where a false positive is nearly as expensive as a false
       negative (FN:FP below ESCALATE_WHEN_RATIO_BELOW), no confidence level
       justifies an automatic block — a human decides.
    """
    spec = COSTS[agent]
    ratio = spec["false_negative"]["cost"] / spec["false_positive"]["cost"]

    if vetoed:
        return "escalate", "LLM verification vetoed the rule flag"
    if not has_interpretable_evidence:
        return "escalate", (
            "no human-readable evidence behind the flag — a model score alone "
            "never blocks")
    if ratio < ESCALATE_WHEN_RATIO_BELOW:
        return "escalate", (
            f"FN:FP ratio {ratio:.2f} is below {ESCALATE_WHEN_RATIO_BELOW}: a "
            f"wrong block costs too much relative to a miss for this agent")
    if confidence >= block_at:
        return "block", (
            f"confidence {confidence:.2f} >= {block_at} with interpretable "
            f"evidence, and FN:FP {ratio:.2f} justifies auto-blocking")
    return "escalate", f"confidence {confidence:.2f} below the block bar {block_at}"


def expected_cost(agent: str, fp_count: int, fn_count: int) -> float:
    """Rupee cost of one agent's held-out confusion matrix."""
    spec = COSTS[agent]
    return (fp_count * spec["false_positive"]["cost"]
            + fn_count * spec["false_negative"]["cost"])


def render_table() -> str:
    rows = summarize()
    width = max(len(a) for a in rows) + 2
    lines = [
        f"{'agent':<{width}} {'FP cost':>10} {'FN cost':>10} {'FN:FP':>7}  policy",
        "-" * (width + 32 + 40),
    ]
    for agent, r in rows.items():
        lines.append(
            f"{agent:<{width}} {r['false_positive_inr']:>10,.0f} "
            f"{r['false_negative_inr']:>10,.0f} {r['fn_to_fp_ratio']:>7.2f}  "
            f"{r['policy'][:44]}"
        )
    return "\n".join(lines)


def score_against_results() -> dict:
    """Apply the table to whatever held-out results have been produced.

    Reads each agent's metrics JSON so the cost column is driven by real
    confusion matrices rather than being asserted separately from them.
    """
    scored = {}

    spike = RESULTS_DIR / "spike_sentinel_metrics.json"
    if spike.exists():
        r = json.loads(spike.read_text())["results"]["ensemble"]
        scored["spike_sentinel"] = {
            "fp": r["fp"], "fn": r["fn"],
            "expected_cost_inr": expected_cost("spike_sentinel", r["fp"], r["fn"]),
            "detector": "ensemble (0.7 model + 0.3 rules)",
        }

    ring = RESULTS_DIR / "ring_detector_metrics.json"
    if ring.exists():
        report = json.loads(ring.read_text())
        headline = report["split"]["headline"]
        r = report["results"][headline]["graph_rules_llm_verified"]
        verified = report.get("llm_verification", {}).get("clusters_reviewed", 0)
        label = ("graph rules + LLM verification" if verified
                 else "graph rules only (no LLM verification in this run)")
        scored["ring_detector"] = {
            "fp": r["fp"], "fn": r["fn"],
            "expected_cost_inr": expected_cost("ring_detector", r["fp"], r["fn"]),
            "detector": f"{label} ({headline} split)",
        }

    content = RESULTS_DIR / "content_forensics_metrics.json"
    if content.exists():
        report = json.loads(content.read_text())
        # Price the detector that actually decides what gets flagged. That is
        # the logistic-regression layer: the LLM is the reasoning layer here
        # (it explains a flag) and is measurably the weaker detector, so
        # costing the system on its confusion matrix would misstate the
        # operating point.
        r, label = (report["results"]["logistic_regression"],
                    "logistic regression (operating detector)")
        scored["content_forensics"] = {
            "fp": r["fp"], "fn": r["fn"],
            "expected_cost_inr": expected_cost("content_forensics", r["fp"], r["fn"]),
            "detector": label,
        }
    guard = RESULTS_DIR / "checkout_guard_metrics.json"
    if guard.exists():
        report = json.loads(guard.read_text())
        r = report["agent_detection_SIMULATED"]["logistic_regression"]
        scored["checkout_guard"] = {
            "fp": r["fp"], "fn": r["fn"],
            "expected_cost_inr": expected_cost("checkout_guard", r["fp"], r["fn"]),
            "detector": "behavioural LR — SIMULATED SESSIONS, not real traffic",
        }

    docs = RESULTS_DIR / "document_forensics_metrics.json"
    if docs.exists():
        report = json.loads(docs.read_text())
        r = report["results"]["ela_logistic_regression"]
        scored["document_forensics"] = {
            "fp": r["fp"], "fn": r["fn"],
            "expected_cost_inr": expected_cost("document_forensics", r["fp"], r["fn"]),
            "detector": "ELA LR — SYNTHETIC SPECIMENS, not real documents",
        }

    return scored


def main() -> None:
    print(__doc__.strip().split("\n\n")[0])
    print()
    print(render_table())
    print(f"\nEscalate rather than auto-block when FN:FP < "
          f"{ESCALATE_WHEN_RATIO_BELOW}.")

    scored = score_against_results()
    if scored:
        print("\nApplied to held-out confusion matrices:")
        for agent, s in scored.items():
            print(f"  {agent:<20} FP {s['fp']:>6,}  FN {s['fn']:>6,}  "
                  f"-> Rs {s['expected_cost_inr']:>14,.0f}   [{s['detector']}]")
        total = sum(s["expected_cost_inr"] for s in scored.values())
        print(f"  {'TOTAL':<20} {'':>26}-> Rs {total:>14,.0f}")
        print("\nThese totals are not comparable across agents — each is "
              "measured on a different held-out set with a different base "
              "rate and a different population size.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "cost_table.json"
    with open(out_path, "w") as f:
        json.dump({
            "currency": "INR",
            "disclaimer": "hand-reasoned order-of-magnitude estimates for a "
                          "mid-size Indian merchant portfolio, not measured "
                          "figures; the relative weighting is the claim, not "
                          "the absolute values",
            "escalate_when_ratio_below": ESCALATE_WHEN_RATIO_BELOW,
            "table": summarize(),
            "applied_to_heldout": scored,
        }, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
