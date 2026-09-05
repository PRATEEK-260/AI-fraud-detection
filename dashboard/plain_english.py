"""Translation layer: the vocabulary of the system, in words a non-specialist reads.

The dashboard exists to make evidence readable. That claim only holds if the
reader is an analyst who already knows what "precision" and "P(fraud)" mean.
A risk desk is looked at by support leads, ops managers and founders too, so
every technical term the system emits is given a plain-English twin here.

Nothing in this module computes or alters a decision. It renames things.
The technical value is always still on screen next to its translation, because
a translation the reader cannot check against the original is just a claim.
"""

from __future__ import annotations

# --- agents ----------------------------------------------------------------
# (one-line job, what it watches, whether its numbers come from real data)
AGENTS = {
    "spike_sentinel": {
        "title": "Spike Sentinel",
        "job": "Spots a card payment that does not fit how this cardholder "
               "normally spends.",
        "watches": "Card transactions",
        "data": "real",
        "data_note": "1.85 million real card transactions",
        "finds": "fraudulent payments",
    },
    "ring_detector": {
        "title": "Ring Detector",
        "job": "Spots groups of accounts that look independent but are really "
               "one person, or one crew, operating together.",
        "watches": "Account sign-ups",
        "data": "real",
        "data_note": "151,000 real e-commerce accounts with genuine device "
                     "and IP columns",
        "finds": "accounts inside a fraud ring",
    },
    "content_forensics": {
        "title": "Content Forensics",
        "job": "Spots reviews and payment disputes that were written by a "
               "machine rather than by a customer.",
        "watches": "Written text",
        "data": "real",
        "data_note": "real Amazon reviews and real consumer finance complaints",
        "finds": "machine-written text",
    },
    "checkout_guard": {
        "title": "Checkout Guard",
        "job": "Spots AI shopping agents at checkout, then checks each one "
               "against the spending rules its owner actually granted it.",
        "watches": "Checkout sessions",
        "data": "simulated",
        "data_note": "SIMULATED sessions — no public record of AI-agent "
                     "checkouts exists yet, so these were generated",
        "finds": "automated checkout clients",
    },
    "document_forensics": {
        "title": "Document Forensics",
        "job": "Spots ID documents where part of the picture has been edited "
               "and saved again.",
        "watches": "Identity documents",
        "data": "synthetic",
        "data_note": "SYNTHETIC specimen cards — no real identity documents "
                     "were used, and none should be",
        "finds": "edited documents",
    },
    "adjudicator": {
        "title": "Adjudicator",
        "job": "The referee. When two agents reach opposite conclusions, it "
               "reads both arguments, rules, and records why.",
        "watches": "The other agents' case files",
        "data": "n/a",
        "data_note": "reads the audit log the other five agents write",
        "finds": "disagreements between agents",
    },
}

# --- decisions -------------------------------------------------------------
DECISIONS = {
    "block": ("Stopped",
              "The payment, sign-up or checkout was refused. This is the only "
              "action a customer feels immediately, so the bar for it is high."),
    "escalate": ("Sent to a person",
                 "Nothing was stopped. A human reviewer gets the case file and "
                 "makes the call. This is what happens whenever the system is "
                 "confident but cannot explain itself."),
    "allow": ("Let through, on the record",
              "Deliberately permitted — and the reason is written down. Being "
              "able to show why you did NOT act matters as much as the blocks."),
}

# --- evidence signals ------------------------------------------------------
# The name each agent stamps on a piece of evidence, said in ordinary words.
SIGNALS = {
    # spike_sentinel
    "high_amount": "The amount is unusually large.",
    "amount_spike": "Far bigger than this cardholder's normal purchase.",
    "velocity_burst": "Several purchases fired off in quick succession.",
    "night_high_amount": "A large purchase in the middle of the night.",
    "model_probability": "The statistical model's own score. On its own this "
                         "explains nothing — which is exactly why it is not "
                         "allowed to block anything by itself.",
    "ensemble_score": "The rules and the model combined into one score.",
    "decision_policy": "The safety rule that turned that score into an action.",
    # ring_detector
    "shared_device_ring": "Several accounts were opened from the same device.",
    "shared_ip_ring": "Several accounts were opened from the same internet "
                      "connection.",
    "signup_burst": "The accounts were all created within a short window.",
    "instant_purchase": "Bought almost immediately after signing up — nobody "
                        "browsed, compared or hesitated.",
    "graph_density": "How tightly this group of accounts is wired together.",
    "llm_verification_1": "Claude's read of the cluster, in its own words.",
    "llm_verification_2": "Claude's read of the cluster, in its own words.",
    "llm_verification_3": "Claude's read of the cluster, in its own words.",
    "llm_verification_4": "Claude's read of the cluster, in its own words.",
    # content_forensics
    "low_burstiness": "Sentence lengths are unnaturally even. People vary far "
                      "more than this when they write.",
    "template_reuse": "The phrasing overlaps heavily with other submissions — "
                      "the same text, lightly reworded.",
    "llm_reason_1": "Claude's explanation of the flag.",
    "llm_reason_2": "Claude's explanation of the flag.",
    "llm_reason_3": "Claude's explanation of the flag.",
    "llm_reason_4": "Claude's explanation of the flag.",
    # checkout_guard
    "metronomic_cadence": "Clicks arrived at machine-perfect intervals.",
    "inhuman_response": "Reacted to the page faster than a person physically can.",
    "thin_fingerprint": "The browser is missing things every real browser has.",
    "declared_agent": "The client openly identified itself as software. Honest.",
    "undeclared_agent": "Behaves like software while presenting as a person.",
    "no_passive_events": "No scrolling, no mouse movement — nobody was reading.",
    "agent_probability": "The model's score that this was software, not a person.",
    "mandate_breach:max_amount": "Spent more than the owner's own limit allowed.",
    "mandate_breach:allowed_categories": "Bought from a kind of shop the owner "
                                         "never approved.",
    "mandate_breach:max_txns_per_hour": "Made more purchases per hour than the "
                                        "owner allowed.",
    # document_forensics
    "ela_hotspot": "The strongest trace of editing sits on a data field.",
    "ela_model_probability": "The model's score that this image was edited.",
    "residual_outlier": "One patch of the image compresses unlike the rest.",
    "uneven_error_surface": "The image's compression fingerprint is patchy, as "
                            "it is when part of a picture is pasted in.",
    # adjudicator
    "conflict_type": "The kind of disagreement that triggered a ruling.",
    "question_put_to_arbiter": "The question the referee was asked.",
    "winning_argument": "Which agent's argument carried.",
    "arbiter_ruling": "The referee's decision and its reason.",
}


def signal_plain(name: str) -> str:
    """Plain-English gloss for an evidence signal name."""
    if name in SIGNALS:
        return SIGNALS[name]
    for prefix in ("source_case:", "mandate_breach:", "llm_verification_",
                   "llm_reason_"):
        if name.startswith(prefix):
            return SIGNALS.get(prefix.rstrip("_:"), {
                "source_case": "The original case file this ruling reviewed.",
                "mandate_breach": "The agent went outside its owner's rules.",
                "llm_verification": "Claude's read of the case, in its own words.",
                "llm_reason": "Claude's explanation of the flag.",
            }.get(prefix.rstrip("_:"), name.replace("_", " ")))
    return name.replace("_", " ")


# --- metrics ---------------------------------------------------------------
def score_in_words(precision: float, recall: float, finds: str) -> str:
    """Precision and recall as two sentences anyone can check.

    "Precision 0.914" is a number. "Of every 100 it flagged, 91 really were
    fraud" is the same number, and it is the form in which a non-specialist
    can tell whether it is good.
    """
    p = round(precision * 100)
    r = round(recall * 100)
    return (f"**Of every 100 it flagged, about {p} really were {finds}** — "
            f"the other {100 - p} were false alarms.  \n"
            f"**It caught about {r} out of every 100** that were really there — "
            f"the other {100 - r} slipped past.")


def cost_in_words(fp: float, fn: float, ratio: float, policy: str) -> str:
    """The cost asymmetry, said out loud."""
    lead = (f"Getting it wrong in one direction costs about **₹{fp:,.0f}**; "
            f"in the other, about **₹{fn:,.0f}**. Missing real fraud is "
            f"**{ratio:.1f}× more expensive** than a false alarm")
    if "never" in policy or ratio < 2.5:
        return (lead + " — which is not a wide enough gap to justify stopping "
                "a real customer, so this agent is **never allowed to block "
                "on its own**, at any confidence.")
    return (lead + " — a wide enough gap that stopping the payment is the "
            "cheaper mistake, so this agent **may block** when the evidence "
            "is readable and the confidence is high.")


GLOSSARY = [
    ("Case file", "One decision, written down: what was seen, what was "
                  "decided, how sure the system was, and what it costs if "
                  "that was wrong. Everything here is case files."),
    ("Evidence", "The individual reasons behind one decision. If a decision "
                 "has no evidence a person can read, this system is not "
                 "allowed to stop anything with it."),
    ("Confidence", "How sure the system is, from 0 to 1. High confidence is "
                   "not permission to act — the evidence and the cost decide "
                   "that."),
    ("Precision", "Out of everything it flagged, how much really was bad. Low "
                  "precision means annoying innocent customers."),
    ("Recall", "Out of everything really bad, how much it caught. Low recall "
               "means fraud gets through."),
    ("Held-out", "Data the system never saw while it was being built, kept "
                 "back to test it honestly. Scores on data it studied are "
                 "worthless."),
    ("False positive", "A false alarm: a real customer treated as a fraudster."),
    ("False negative", "A miss: real fraud treated as a real customer."),
    ("Mandate", "The spending rules a person grants their AI shopping agent — "
                "a spend cap, which shops it may use, how often. Checking them "
                "is arithmetic, not AI."),
    ("Audit log", "The append-only file every decision is written to. Nothing "
                  "in it is ever edited or deleted, including the wrong ones."),
    ("Agent", "Two meanings here, unfortunately. Our six *detection* agents "
              "are the programs doing the watching. An *AI shopping agent* is "
              "the thing Checkout Guard watches for."),
]


# --- per-agent footnotes ---------------------------------------------------
# Each of these explains a row in that agent's results table that would
# otherwise read as a failure, or as a success it does not deserve. Leaving
# them out would break the promise this module makes exactly where it matters.
NOTES = {
    "ring_detector":
        "**Why one row below scores near zero.** The table shows this agent "
        "tested two ways. Splitting the accounts by *ring* gives the headline "
        "number. Splitting them by *date* scores almost zero — because in this "
        "public dataset the shared-device behaviour is almost entirely in the "
        "first month (31% of January accounts share a device with 3+ people, "
        "against 0.2% every month after). A date split therefore leaves nothing "
        "to find and measures how the dataset was assembled, not the detector. "
        "Both numbers are shown rather than the flattering one.",
    "content_forensics":
        "**The row that changed the design.** `llm_zeroshot` is what happens "
        "when you simply ask a large language model whether a piece of text was "
        "machine-written: it found **2 out of 112**, and the same thing happened "
        "on three different model families. So the language model was taken off "
        "this decision entirely. A far simpler statistical method makes the "
        "call, and the language model does what it is genuinely good at — "
        "explaining a flag once it has been raised.",
    "document_forensics":
        "**What this does not prove.** The detector tells edited documents from "
        "clean ones — but when asked to point at *which part* was edited, it is "
        "right about as often as random guessing. So it is a cheap first filter "
        "that puts a document in front of a person, never a control that can "
        "reject one on its own. The agent says so in its own output too.",
    "checkout_guard":
        "**Two halves, only one of them speculative.** Guessing whether a "
        "checkout was driven by software is the learned, simulated half. "
        "Checking whether it stayed inside the spend cap, shop types and "
        "frequency its owner authorised is plain arithmetic against a rule the "
        "customer set — no model, no threshold, no simulation. That half would "
        "ship into production tomorrow. It scores a perfect 1.00, which is "
        "**not a result**: the simulator defines a rogue agent as one that "
        "breaks a rule, and this checks the rules, so it could hardly do "
        "otherwise. It shows the code is correct, nothing more.",
    "spike_sentinel":
        "**Why the rules-only row looks terrible.** Hand-written rules alone "
        "flag 30,000 innocent payments to catch 1,500 fraudulent ones. They are "
        "kept anyway — not for accuracy, but because they are the part a human "
        "can read. The model supplies the accuracy; the rules supply the "
        "explanation, and without an explanation nothing here is allowed to "
        "block.",
}
