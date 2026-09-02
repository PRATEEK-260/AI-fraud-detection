# AI-Native Fraud Defense System — Implementation Plan

**Track:** Razorpay AI Buildathon — AI Risk Manager
**Deadline:** 5 September (3 build days from kickoff)
**Author:** Prateek

---

## 1. Goal

Build a multi-agent risk system that catches both classic fraud (transaction spikes,
coordinated rings) and AI-native fraud (misbehaving shopping agents, AI-generated
KYC/reviews/disputes), with every decision backed by an evidence case file — not a bare score.

**Non-negotiable priority order if time runs short:**
Spike Sentinel → Content Forensics → Ring Detector → Dashboard/README → Adjudicator → Agentic Checkout Guard (cut first).

---

## 2. Repo Structure

```
fraud-defense-system/
├── data/
│   ├── raw/
│   │   ├── sparkov/            # kartik2112/fraud-detection
│   │   └── ecommerce/          # vbinh002/fraud-ecommerce
│   └── processed/              # time-split train/held-out, never re-touched
├── spine/
│   ├── __init__.py
│   ├── schema.py                # Case dataclass
│   ├── event_bus.py             # async queue connecting agents
│   └── db.py                    # SQLite audit log
├── agents/
│   ├── __init__.py
│   ├── spike_sentinel.py
│   ├── ring_detector.py
│   ├── content_forensics.py
│   ├── checkout_guard.py        # stretch, build last
│   └── adjudicator.py
├── eval/
│   ├── __init__.py
│   ├── metrics.py               # precision/recall/F1 per agent
│   └── cost_table.py            # false-positive cost table
├── dashboard/
│   └── app.py                   # Streamlit
├── notebooks/                   # exploration only, not shipped logic
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 3. Environment Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pandas numpy scikit-learn xgboost networkx python-louvain \
            fastapi uvicorn streamlit anthropic textstat scikit-image \
            opencv-python-headless pytest
```

`requirements.txt` should pin versions once finalized — do this on Day 3, not Day 0
(don't waste time pinning while APIs are still in flux).

---

## 4. Data Pipeline

Two datasets, one per purpose — don't force a single dataset to do everything.

| Dataset | Used by | Why |
|---|---|---|
| `kartik2112/fraud-detection` (Sparkov) | Spike Sentinel | Readable features, sequential per-customer transactions, geo lat/long for velocity |
| `vbinh002/fraud-ecommerce` (Amazon Fraud) | Ring Detector | Real `device_id`/`ip_address` columns — no fabricated signal |

```bash
kaggle datasets download -d kartik2112/fraud-detection -p data/raw/sparkov --unzip
kaggle datasets download -d vbinh002/fraud-ecommerce -p data/raw/ecommerce --unzip
```

**Splitting rule:** time-based split, not random (`sort by timestamp, first 80% = train,
last 20% = held-out`). Write the held-out set to `data/processed/` immediately and never
re-open it until final evaluation — this is what makes your reported metrics defensible.

**Content Forensics dataset (self-built):** generate ~200-300 paired samples of
human-written vs. LLM-generated KYC descriptions, reviews, and dispute narratives.
Use the Claude API for the AI side; hand-write or pull real reviews for the human side.
Save as `data/processed/content_forensics_labeled.csv` with columns `text, is_ai_generated, source`.

---

## 5. Shared Spine

### 5.1 `spine/schema.py`

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Optional
import uuid

Decision = Literal["allow", "escalate", "block"]

@dataclass
class Evidence:
    signal: str          # e.g. "velocity_spike", "shared_device_id"
    value: str
    weight: float         # contribution to confidence, 0-1

@dataclass
class Case:
    case_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_agent: str = ""
    entity_id: str = ""              # user_id / card_id / device_id / dispute_id
    entity_type: str = ""            # "transaction" | "account" | "document" | "dispute"
    evidence: list[Evidence] = field(default_factory=list)
    confidence: float = 0.0          # 0-1
    cost_estimate: float = 0.0       # expected cost if wrongly blocked
    decision: Decision = "allow"
    reasoning_text: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
```

### 5.2 `spine/event_bus.py`
Simple `asyncio.Queue` wrapping a pub/sub pattern — agents publish `Case` objects,
the dashboard and adjudicator subscribe. No need for Kafka/Redis at this scale;
don't over-engineer infra that doesn't affect the score.

### 5.3 `spine/db.py`
SQLite table `cases` mirroring the `Case` schema, one row per case, append-only
(this is your audit trail — never update or delete rows).

---

## 6. Agent Specs

### 6.1 Spike Sentinel (`agents/spike_sentinel.py`)
- **Input:** Sparkov transaction stream
- **Features:** rolling tx-count per user per minute, amount deviation from user's
  rolling mean, merchant-category risk score, hour-of-day anomaly
- **Model:** velocity rule (hard threshold) OR XGBoost trained on engineered features,
  combined via weighted ensemble
- **Output:** `Case` with `entity_type="transaction"`, evidence list of triggered signals

### 6.2 Ring Detector (`agents/ring_detector.py`)
- **Input:** Amazon Fraud e-commerce dataset (`device_id`, `ip_address`, `user_id`)
- **Graph:** nodes = accounts, edges = shared device_id or IP
- **Clustering:** `python-louvain` community detection on the graph
- **Verification pass:** for each flagged cluster, send account summaries to Claude
  with the prompt: "does this cluster look like coordinated fraud or a legitimate
  shared pattern (family, office, marketplace)?" — this LLM check is what prevents
  false positives from tanking precision
- **Output:** one `Case` per flagged cluster, `entity_type="account"`

### 6.3 Content Forensics (`agents/content_forensics.py`)
- **Input:** self-built labeled text dataset
- **Signals:**
  - Perplexity/burstiness proxy via `textstat` (sentence-length variance, lexical diversity)
  - Template reuse: TF-IDF cosine similarity against other flagged texts
  - For document images: error-level analysis (`opencv`/`PIL`) for tampering artifacts
- **LLM reasoning call:** ask Claude to explain *why* a text looks AI-generated
  (uniform sentence length, generic phrasing, reused structure) — this becomes
  `reasoning_text`
- **Output:** `Case` with `entity_type="document"` or `"dispute"`

### 6.4 Agentic Checkout Guard (`agents/checkout_guard.py`) — build last
- **Input:** simulated checkout session logs (you'll need to fabricate these —
  request timing, header patterns, decision latency)
- **Signals:** inhuman timing regularity, non-browser headers, spend outside a
  declared policy bound
- **Scope down if short on time:** a rules-based fingerprint (e.g. sub-200ms
  decision latency + missing browser fingerprint = likely non-human) is an
  acceptable minimum viable version — don't attempt a full ML model here unless
  everything else is done early.

### 6.5 Adjudicator-in-Chief (`agents/adjudicator.py`)
- **Trigger:** two agents flag the same `entity_id` (or linked entities via shared
  device/IP) within a time window
- **Logic:** pull both `Case` objects, send both to Claude with a prompt to weigh
  the evidence and either resolve (pick a decision) or escalate to human with
  both arguments shown verbatim
- Keep this simple — a rule-triggered LLM arbitration call, not an independent
  ML model. Over-engineering this wastes time that should go to Agents 1-3.

---

## 7. Evaluation (`eval/`)

- **`metrics.py`:** precision, recall, F1 per agent on its held-out set — report
  individually, never averaged across agents
- **`cost_table.py`:** a small hand-reasoned table, e.g.

| Agent | False-positive cost | False-negative cost |
|---|---|---|
| Spike Sentinel | Blocked legitimate high-value transaction, merchant trust hit | Fraud loss, chargeback |
| Ring Detector | Legitimate shared-device users wrongly locked out | Ring continues undetected |
| Content Forensics | Genuine customer's dispute wrongly denied | Fraudulent refund paid out |

- **Failure case:** pick one real false positive from your held-out results and
  walk through why it was escalated (not blocked) — this is a required part of
  the pitch, not optional polish.
- **Latency:** measure per-agent processing time on a batch of N transactions,
  report as "X ms/transaction, path to production streaming would require Y."

---

## 8. Dashboard (`dashboard/app.py`)

Streamlit, single page:
- Live table of `Case` records (poll SQLite every few seconds)
- Filter by `source_agent`
- Click a row → show full evidence list + `reasoning_text`
- A "replay" button that feeds the held-out set through the pipeline at demo
  speed (e.g. 1 transaction/second) so the dashboard visibly updates live

---

## 9. Day-by-Day Timeline

### Day 0 (today)
1. Repo + folder structure (done)
2. Download both datasets, do the time-based split
3. Build `Case` schema + SQLite db
4. Build Spike Sentinel end-to-end
5. **Checkpoint:** precision/recall printed and saved — this alone is a submittable project

### Day 1
1. Build Content Forensics dataset (Claude-generated AI side + human side)
2. Build signal extraction + LLM reasoning wrapper
3. **Checkpoint:** precision/recall on content forensics held-out set saved —
   two working agents now, a strong stopping point

### Day 2 (morning)
1. Build Ring Detector: graph → Louvain → LLM verification pass
2. **If behind schedule, stop here and move to dashboard/pitch prep**

### Day 2 (afternoon, only if ahead of schedule)
1. Adjudicator-in-Chief (rule-triggered, not a full agent)
2. Dashboard build + wire to SQLite
3. Skip Checkout Guard unless there's real time left over

### Day 3
1. Morning: freeze code, run final eval on all held-out sets, finalize cost table
2. README: architecture diagram, setup instructions, honest metrics, explicit
   out-of-scope note (no production streaming, no real KYC data)
3. Write and rehearse the 5-min pitch script (problem → per-agent demo →
   cross-agent disagreement case → metrics → limitations)
4. Rehearse on a **fresh clone** of the repo, not your dev machine, to catch
   environment issues
5. Submit early, not at the deadline

---

## 10. Pitch Structure (5 minutes)

1. **Problem (30s):** fraud systems assume human attackers and human-written
   documents — both assumptions are breaking
2. **Live demo (2.5 min):** one flagged case per working agent, showing the
   evidence and reasoning text, not just a score
3. **Cross-agent disagreement (1 min):** walk through the one case where two
   agents disagreed and the adjudicator resolved it
4. **Metrics + honest limitations (1 min):** precision/recall numbers, the cost
   table, and an explicit statement of what's out of scope and why

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Ring Detector graph too sparse on the chosen dataset | Fall back to a synthetic device/IP overlay on Sparkov if Amazon Fraud data proves too small |
| LLM calls too slow for "live" demo feel | Pre-cache a handful of representative case reasoning calls, call live only for 1-2 hero cases |
| Running out of time before Content Forensics dataset is built | Cap it at 100 paired samples instead of 300 — still enough for a defensible eval |
| Judges question fabricated data realism | Lead with the two-dataset choice as a deliberate engineering decision (already in project doc) |

---

## 12. Definition of Done (minimum viable submission)

- [ ] Spike Sentinel working with real precision/recall
- [ ] Content Forensics working with real precision/recall
- [ ] `Case` schema + audit log populated by both agents
- [ ] README with architecture diagram and honest metrics
- [ ] 5-min pitch rehearsed at least twice
- [ ] Repo is public and runs from a fresh clone

Everything beyond this (Ring Detector, Adjudicator, Dashboard, Checkout Guard) is
upside, not required for a legitimate submission.
