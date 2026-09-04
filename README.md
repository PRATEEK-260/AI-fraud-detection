# AI-Native Fraud Defense System

A multi-agent risk desk for fraud in the AI era. Every decision produces an
**evidence-backed case file**, not a bare score — and a decision with no
human-readable evidence behind it is never allowed to auto-block.

**Razorpay AI Buildathon · AI Risk Manager track · Prateek**

---

## The problem

Fraud detection assumes the attacker is a human clicking through checkout, and
that identity documents, reviews, and dispute narratives are written by real
people. Both assumptions are breaking:

- **Generative AI is industrialising fraud content.** Fake KYC text, fake
  reviews, and fake dispute narratives ("my order never arrived") can be
  mass-produced by an LLM, cheaply and convincingly.
- **Agentic commerce is arriving.** AI shopping agents will transact on a
  user's behalf, and an agent exceeding its owner's intent is a new attack
  surface.

Existing systems are not built to ask *"was this even a human?"* That question
is the gap this project addresses.

---

## Architecture

One shared spine — a `Case` schema, an append-only SQLite audit log, and an
event bus — that four specialist agents write into, and one dashboard reads.

```
                    ┌──────────────────────────────────────────┐
   Sparkov  ──────► │ Spike Sentinel      (transactions)       │
   1.85M tx         │   velocity + amount rules → evidence     │
                    │   XGBoost → probability                  │
                    │   0.7·model + 0.3·rules → score          │
                    └──────────────┬───────────────────────────┘
                                   │
   Amazon Fraud ───► ┌─────────────▼────────────────────────────┐
   151k accounts     │ Ring Detector       (account clusters)   │
                     │   shared device/IP graph → Louvain       │
                     │   cluster rules → evidence               │
                     │   LLM verification can VETO a flag       │
                     └─────────────┬────────────────────────────┘
                                   │
   Self-built ─────► ┌─────────────▼────────────────────────────┐
   775 texts         │ Content Forensics   (docs / disputes)    │
                     │   burstiness, MATTR, readability,        │
                     │   TF-IDF template reuse → evidence       │
                     │   logistic regression + LLM judgment     │
                     └─────────────┬────────────────────────────┘
                                   │
                     ┌─────────────▼────────────────────────────┐
                     │ Case  {evidence[], confidence,           │
                     │        cost_estimate, decision,          │
                     │        reasoning_text}                   │
                     └─────────────┬────────────────────────────┘
                                   │
                     ┌─────────────▼────────────────────────────┐
                     │ eval/cost_table.decide()                 │
                     │   GATE 1  no readable evidence → never   │
                     │           block, whatever the score      │
                     │   GATE 2  FN:FP below bar → never block  │
                     └─────────────┬────────────────────────────┘
                                   │
              ┌────────────────────▼───────────────────┐
              │ append-only SQLite audit log           │
              └────────┬──────────────────────┬────────┘
                       │                      │
         ┌─────────────▼──────────┐   ┌───────▼─────────────┐
         │ Adjudicator-in-Chief   │   │ Streamlit risk desk │
         │  finds contradictions, │   │  cases, evidence,   │
         │  arbitrates, appends   │   │  metrics, replay    │
         └────────────────────────┘   └─────────────────────┘
```

The audit log is **append-only**. The adjudicator never rewrites a case; it
appends its ruling next to the original, so the disagreement stays visible.

---

## Results (held-out)

Reported per agent, never averaged — each runs on a different dataset with a
different base rate, so a cross-agent average would be meaningless.

### Spike Sentinel — 555,719 held-out transactions, 0.386% fraud

| detector | precision | recall | F1 | PR-AUC |
|---|---|---|---|---|
| rules only | 0.048 | 0.720 | 0.090 | 0.046 |
| XGBoost | 0.949 | 0.890 | 0.919 | 0.963 |
| **ensemble (0.7/0.3)** | **0.914** | **0.905** | **0.909** | 0.884 |

Held-out is the Kaggle `fraudTest` period (Jun–Dec 2020), verified
chronologically disjoint from train. Thresholds were tuned on the last 10% of
the *train* period, never on held-out.

### Ring Detector — 45,507 held-out accounts, 9.58% fraud

| detector | precision | recall | F1 |
|---|---|---|---|
| **graph rules** | **0.907** | **0.548** | **0.683** |

Recall is capped by design: only ~71% of fraud in this dataset sits on a
shared identifier at all, and clusters below 3 accounts are not treated as
rings.

### Content Forensics — 230 held-out texts, 48% AI-generated

| detector | precision | recall | F1 | PR-AUC |
|---|---|---|---|---|
| rule thresholds | 0.723 | 0.736 | 0.730 | — |
| **logistic regression** | **0.890** | **0.809** | **0.848** | 0.949 |

Held-out AI texts come from **topics never seen in training**, so the
template-reuse signal has to generalise rather than memorise our own
generation phrasing.

### Latency

| stage | measured |
|---|---|
| Spike Sentinel (rules + model) | 0.0008 ms/transaction |
| Ring Detector graph build | 3.8 s for 151k accounts |
| Louvain clustering | 0.7 s for 848 clusters |
| Content Forensics signals | 0.6 ms/text |

These are **batch, in-memory** figures. Production streaming would add
per-event I/O, feature-store reads, and network latency, and would need the
rolling per-user windows maintained incrementally rather than recomputed. This
project does not claim production streaming.

---

## What the measurements changed

Three findings from this build changed the design rather than being papered
over. They are the honest core of the submission.

**1. The ring signal lives in one cohort.** In the e-commerce dataset, 31% of
January-2015 accounts sit on a device shared by 3+ users (31.5% fraud rate);
every later month sits at ~0.2% and ~4.6%. The coordinated-ring behaviour was
injected into a single cohort. A time-based split therefore puts 100% of the
phenomenon in train and leaves held-out with nothing to detect — it would
score the dataset's construction, not the detector. Both splits are computed
and reported (`ring_detector_metrics.json`); the headline uses a **component
split** where whole rings go to one side, and the time split is shown next to
it at P 0.200 / R 0.001 with the cohort table as evidence for why.

**2. 461 cases were auto-blocking on a bare score.** Auditing the log found
461 Spike Sentinel cases flagged with a high model probability and **zero**
interpretable rule signals — marked `block`. That is exactly the black-box
score this project argues against, shipped by this project. `cost_table.decide()`
is the fix: no readable evidence, no auto-block, whatever the model says.

**3. The cost table decides the action.** Content Forensics has an FN:FP
ratio of 2.33 — wrongly denying a real customer's dispute costs nearly as much
as paying out a fake one, and is far worse for the customer. So that agent
**never auto-blocks**, at any confidence. Ring Detector (7.38) and Spike
Sentinel (10.82) may, but only with evidence behind them.

| agent | FP cost | FN cost | FN:FP | policy |
|---|---|---|---|---|
| spike_sentinel | ₹850 | ₹9,200 | 10.82 | auto-block permitted at high confidence |
| ring_detector | ₹4,200 | ₹31,000 | 7.38 | auto-block permitted at high confidence |
| content_forensics | ₹2,400 | ₹5,600 | 2.33 | **escalate only — never auto-block** |

Rupee values are hand-reasoned orders of magnitude, not measured figures; the
relative weighting is the claim. Assumptions per line are in
`eval/cost_table.py`.

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # add OPENROUTER_API_KEY for the LLM layers
```

Data (Kaggle CLI required):

```bash
kaggle datasets download -d kartik2112/fraud-detection -p data/raw/sparkov --unzip
kaggle datasets download -d vbinh002/fraud-ecommerce -p data/raw/ecommerce --unzip
python scripts/prepare_data.py          # freezes the time-based split
```

Run the agents:

```bash
python -m agents.spike_sentinel --fresh-db
python -m agents.ring_detector --no-llm      # add --limit-llm N for verification
python -m agents.content_forensics --no-llm  # add --limit-llm N for the LLM detector
python -m agents.adjudicator --no-llm
python -m eval.cost_table
streamlit run dashboard/app.py
```

Every agent runs **without an API key** using `--no-llm`; the statistical and
graph layers are complete on their own, and the metrics above need no LLM
calls. The LLM layers add the reasoning and verification passes.

```bash
python -m pytest tests/ -q     # 21 tests, no data or API needed
```

---

## Data

| dataset | agent | why |
|---|---|---|
| `kartik2112/fraud-detection` (Sparkov) | Spike Sentinel | readable non-PCA columns, sequential per-card transactions |
| `vbinh002/fraud-ecommerce` | Ring Detector | **real** `device_id` / `ip_address` columns |
| self-built, 775 texts | Content Forensics | no public corpus of paired human/AI fraud text exists |

**Deviation from the project doc:** the original write-up proposed
*synthesising* device/IP onto Sparkov. That was replaced with a second real
dataset, because a detector evaluated on fabricated signal measures the
fabrication. This was a deliberate trade: the ring dataset turned out to be
temporally degenerate (finding 1 above), which a synthetic overlay would have
hidden.

**Content Forensics corpus** — human: real Amazon reviews and real CFPB
complaint narratives. AI: generated by `minimax-m3` and `mistral-small-24b`,
two families, **neither of them Claude**, so the detector has no
self-recognition advantage. Word budgets are drawn from the human word-count
distribution of the same domain and the same 80–800 character band is applied
to both classes: human 232 vs AI 226 chars for reviews, 469 vs 440 for
disputes. Without that matching the earlier corpus ran 1.9× longer on the AI
side, and any detector trained on it would have been measuring verbosity.

---

## Scope and limitations

Stated plainly, because the evaluation is the point of this submission.

- **Not production streaming.** Batch replay with measured per-stage latency.
- **No real KYC data.** KYC self-descriptions are out of eval scope — there is
  no public corpus of genuine human KYC text to pair against. Reviews and
  disputes are the two domains with real human corpora, so they are the two
  that get measured. Image forensics on ID documents (ELA) was scoped out for
  the same reason.
- **The Agentic Checkout Guard is not built.** It was the first cut in the
  plan's priority order. Building it would have required fabricating the
  session-timing data it detects, and finding 1 is the reason that was
  avoided.
- **Cross-agent adjudication does not fire on this data.** The three agents run
  on three different datasets, so no `entity_id` is observable by two agents.
  The code path is implemented and unit-tested against a synthetic pair rather
  than manufactured by joining unrelated datasets. The conflicts that *do*
  fire are real: 461 model-without-evidence cases.
- **Ring recall is 0.55**, capped by shared-identifier coverage in the data.
- **Content Forensics is evaluated on 230 held-out texts.** Small. The
  confidence interval on 0.890 precision is wide, and two generator families
  is not proof of generalisation to a third.
- **Cost figures are reasoned estimates**, not measured losses.

---

## Repository

```
spine/       Case schema, append-only SQLite log, event bus, LLM client
agents/      spike_sentinel, ring_detector, content_forensics, adjudicator
eval/        metrics, cost table + the decision policy, results/*.json
scripts/     prepare_data (split freeze), build_content_dataset
dashboard/   Streamlit risk desk
tests/       21 regression tests, one per bug that shipped
```
