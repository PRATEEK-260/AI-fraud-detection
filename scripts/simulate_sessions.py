"""Simulate checkout sessions: human, authorised AI agent, rogue AI agent.

READ THIS BEFORE BELIEVING ANY NUMBER PRODUCED FROM THIS FILE.

There is no public dataset of AI-shopping-agent checkout sessions, because the
attack is new. IMPLEMENTATION_PLAN.md §6.4 anticipated that and said the input
would have to be fabricated. So it is fabricated — and that means any
precision/recall measured against it describes **how separable this simulation
is**, not how well the detector would work on real traffic. Every report this
feeds is labelled that way, and the README repeats it.

What is NOT fabricated, and is worth more than the detector metrics: the
mandate policy engine in agents/checkout_guard.py. "Did this transaction
exceed the spend cap / leave the allowed categories / breach the velocity
limit its owner authorised?" is deterministic business logic evaluated against
a mandate the user grants. That logic would ship unchanged into production.
The behavioural fingerprint is the speculative half; the policy engine is not.

Two design rules keep the simulation from being self-fulfilling:

1. **The simulator never imports the detector.** Its parameters come from
   behavioural assumptions stated below, not from the thresholds the detector
   uses. If the two were tuned together the evaluation would be circular.
2. **Hard cases are generated on purpose.** A simulation containing only
   metronome-regular bots and leisurely humans would be trivially separable and
   the reported numbers would be meaningless. So it emits rushed humans
   (checkout in seconds, low variance), and jittered agents that deliberately
   randomise their cadence and carry a full browser fingerprint to imitate a
   person. Those two groups overlap, and they are where the detector's real
   errors come from.

Behavioural assumptions (stated so a reviewer can dispute them):
  - Humans pause irregularly to read, compare, hunt for a card. Inter-action
    gaps are heavy-tailed — log-normal, high coefficient of variation.
  - Humans cannot act on a page faster than roughly 250 ms (perception plus
    motor response); agents routinely do.
  - Scripted clients emit fewer passive events (scroll, mouse move, focus
    change) because they do not read or hesitate.
  - An automated client's headers and fingerprint surface are usually thinner
    (missing Accept-Language, no WebGL/canvas entropy, headless UA hints).

Usage:
    .venv/bin/python scripts/simulate_sessions.py [--n 3000]
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "processed" / "checkout_sessions.parquet"

SEED = 1337

# Population mix. Rogue agents are rare, as the real thing would be.
MIX = {"human": 0.62, "agent_authorised": 0.24, "agent_rogue": 0.14}

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "fashion", "travel", "gaming",
    "gift_cards", "crypto", "pharmacy", "home", "entertainment",
]

# Categories a user rarely pre-authorises an agent to buy from: the ones that
# liquidate. A rogue agent drifts into these.
HIGH_LIQUIDITY = {"gift_cards", "crypto", "electronics", "gaming"}

BROWSER_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/141.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) Safari/18.1",
    "Mozilla/5.0 (X11; Linux x86_64) Firefox/136.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_2) Mobile/15E148",
]
AUTOMATION_UAS = [
    "python-requests/2.34.2",
    "Mozilla/5.0 (X11; Linux x86_64) HeadlessChrome/141.0",
    "node-fetch/3.3",
    "ShopAgent/1.4 (+https://example.invalid/agent)",
]


def _lognormal_gaps(rng: random.Random, n: int, median_s: float,
                    sigma: float) -> list[float]:
    """Heavy-tailed inter-action gaps — the shape human hesitation takes."""
    mu = math.log(max(median_s, 1e-3))
    return [float(math.exp(rng.gauss(mu, sigma))) for _ in range(n)]


def simulate_human(rng: random.Random, rushed: bool) -> dict:
    """A person checking out. `rushed` produces the hard cases: someone who
    knows exactly what they want and completes in seconds with little
    variance — the humans a naive regularity test would wrongly flag."""
    n_actions = rng.randint(4, 8) if rushed else rng.randint(7, 22)
    gaps = _lognormal_gaps(
        rng, n_actions,
        median_s=rng.uniform(0.7, 1.6) if rushed else rng.uniform(2.0, 6.5),
        sigma=rng.uniform(0.35, 0.6) if rushed else rng.uniform(0.8, 1.35))
    return {
        "actor_type": "human",
        "gaps": gaps,
        # Even a hurried person needs a beat to see the page and move a hand.
        "min_response_ms": rng.uniform(260, 900),
        "passive_events": rng.randint(6, 40) if not rushed else rng.randint(3, 14),
        "user_agent": rng.choice(BROWSER_UAS),
        "has_accept_language": True,
        "has_canvas_fingerprint": rng.random() > 0.06,   # a few block it
        "has_webgl": rng.random() > 0.10,
        "cookies_enabled": True,
        "declared_agent": False,
        "mandate": None,
    }


def _mandate(rng: random.Random, obedient: bool) -> dict:
    """The spending authority a user grants an agent. This is the part that is
    real: a mandate is configuration a user sets, not observed data."""
    cap = rng.choice([2000.0, 5000.0, 10000.0, 25000.0])
    allowed = rng.sample(
        [c for c in MERCHANT_CATEGORIES if c not in HIGH_LIQUIDITY],
        k=rng.randint(2, 4))
    if obedient and rng.random() < 0.35:
        allowed.append(rng.choice(sorted(HIGH_LIQUIDITY)))
    return {
        "max_amount": cap,
        "allowed_categories": sorted(set(allowed)),
        "max_txns_per_hour": rng.choice([2, 3, 5]),
    }


def simulate_agent(rng: random.Random, rogue: bool, jittered: bool) -> dict:
    """An AI shopping agent. `jittered` is the hard case: an agent that
    deliberately randomises its cadence and presents a full browser
    fingerprint to pass as a person."""
    n_actions = rng.randint(3, 9)
    if jittered:
        gaps = _lognormal_gaps(rng, n_actions, median_s=rng.uniform(1.2, 3.0),
                               sigma=rng.uniform(0.55, 0.95))
    else:
        base = rng.uniform(0.05, 0.9)
        gaps = [base + rng.gauss(0, base * rng.uniform(0.02, 0.12))
                for _ in range(n_actions)]
        gaps = [max(g, 0.01) for g in gaps]

    fingerprinted = jittered and rng.random() < 0.7
    return {
        "actor_type": "agent_rogue" if rogue else "agent_authorised",
        "gaps": gaps,
        "min_response_ms": (rng.uniform(120, 400) if jittered
                            else rng.uniform(8, 150)),
        "passive_events": rng.randint(0, 9) if not fingerprinted else rng.randint(2, 18),
        "user_agent": (rng.choice(BROWSER_UAS) if fingerprinted
                       else rng.choice(AUTOMATION_UAS)),
        "has_accept_language": fingerprinted or rng.random() < 0.25,
        "has_canvas_fingerprint": fingerprinted and rng.random() < 0.6,
        "has_webgl": fingerprinted and rng.random() < 0.5,
        "cookies_enabled": rng.random() < 0.8,
        # An honest agent identifies itself; a rogue one usually will not.
        "declared_agent": (rng.random() < 0.80) if not rogue else (rng.random() < 0.15),
        "mandate": _mandate(rng, obedient=not rogue),
    }


def finalize(session: dict, rng: random.Random, idx: int) -> dict:
    """Turn behaviour into the row the detector sees."""
    gaps = session.pop("gaps")
    mandate = session.pop("mandate")
    rogue = session["actor_type"] == "agent_rogue"

    amount = round(rng.lognormvariate(math.log(1800), 0.9), 2)
    category = rng.choice(MERCHANT_CATEGORIES)
    txns_last_hour = rng.randint(1, 2)

    if mandate:
        if rogue:
            # A rogue agent breaches at least one bound — that is what makes
            # it rogue. Which bound varies, so no single rule catches them all.
            breach = rng.choice(["amount", "category", "velocity", "mixed"])
            if breach in ("amount", "mixed"):
                amount = round(mandate["max_amount"] * rng.uniform(1.15, 4.0), 2)
            if breach in ("category", "mixed"):
                category = rng.choice(sorted(HIGH_LIQUIDITY - set(
                    mandate["allowed_categories"])) or ["crypto"])
            if breach in ("velocity", "mixed"):
                txns_last_hour = mandate["max_txns_per_hour"] + rng.randint(1, 6)
        else:
            amount = round(min(amount, mandate["max_amount"] * rng.uniform(0.1, 0.92)), 2)
            category = rng.choice(mandate["allowed_categories"])
            txns_last_hour = rng.randint(1, max(mandate["max_txns_per_hour"], 1))

    arr = np.asarray(gaps, dtype=float)
    return {
        "session_id": f"S{idx:06d}",
        **session,
        "n_actions": len(arr),
        "gaps_json": json.dumps([round(g, 4) for g in arr.tolist()]),
        "session_duration_s": round(float(arr.sum()), 3),
        "amount": amount,
        "merchant_category": category,
        "txns_last_hour": txns_last_hour,
        "mandate_json": json.dumps(mandate) if mandate else "",
        # Labels — evaluation only, never a feature.
        "is_agent": int(session["actor_type"] != "human"),
        "is_rogue": int(rogue),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=3000, help="sessions to simulate")
    ap.add_argument("--hard-fraction", type=float, default=0.40,
                    help="share of each class generated as a HARD case "
                         "(rushed humans, jittered agents) — raising this "
                         "makes the task harder and the numbers lower")
    args = ap.parse_args()

    rng = random.Random(SEED)
    rows = []
    for i in range(args.n):
        roll = rng.random()
        cum = 0.0
        for actor, share in MIX.items():
            cum += share
            if roll <= cum:
                break
        hard = rng.random() < args.hard_fraction
        if actor == "human":
            s = simulate_human(rng, rushed=hard)
        else:
            s = simulate_agent(rng, rogue=(actor == "agent_rogue"), jittered=hard)
        s["is_hard_case"] = int(hard)
        rows.append(finalize(s, rng, i))

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print(f"Wrote {len(df):,} simulated sessions -> {OUT_PATH}")
    print("\nSIMULATED DATA. Metrics computed on it measure how separable this "
          "simulation is, not real-world detection performance.\n")
    print(df["actor_type"].value_counts().to_string())
    print(f"\nhard cases: {int(df['is_hard_case'].sum()):,} "
          f"({df['is_hard_case'].mean():.0%})")
    print("\nmedian inter-action gap by actor (s):")
    med = df.assign(g=df["gaps_json"].map(lambda j: float(np.median(json.loads(j)))))
    print(med.groupby(["actor_type", "is_hard_case"])["g"].median().round(3).to_string())


if __name__ == "__main__":
    main()
