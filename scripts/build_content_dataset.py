"""Build the Content Forensics labeled dataset (real human vs generated AI text).

Human side (genuine human-written text, no synthetic stand-ins):
    - product reviews: SetFit/amazon_reviews_multi_en (real Amazon reviews)
    - dispute narratives: CFPB consumer complaint narratives
AI side (fraud-style generated text):
    - same two domains, generated via OpenRouter by two models from
      different families (minimax-m3, mistral-small-24b), neither of them
      the DETECTOR family (Claude) — no self-recognition bias, and two
      generators rather than one so the detector cannot pass by memorizing
      a single model's phrasing.

Diversity matrix per generated sample (seeded RNG, reproducible):
    topic (group tag) x persona x tone x instruction template x word budget.
The word budget is DRAWN FROM the human word-count distribution of the same
domain, and the same 80-800 character band is applied to both classes, so
length cannot act as a free giveaway signal. The generator is asked for a
word count rather than a sentence count: prescribing a sentence count would
manufacture the uniform rhythm that Agent 4 is supposed to measure.
The topic tag enables a GROUP-BASED held-out split: held-out AI texts come
from topics never seen in training, so the template-reuse signal must
generalize rather than memorize our own generation phrasing.

Idempotent: every LLM call goes through spine.llm's disk cache, so re-running
after a partial failure resumes for free.

Usage:
    .venv/bin/python scripts/build_content_dataset.py [--scale full|plan]
"""

from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))   # allow `python scripts/...` from anywhere

from spine.llm import chat  # noqa: E402
CORPUS_DIR = ROOT / "data" / "raw" / "corpus"
OUT_PATH = ROOT / "data" / "processed" / "content_forensics_labeled.csv"

SEED = 42
N_PER_DOMAIN = 200          # human AND AI, per domain ("full" scale)

# Two DIFFERENT generator families, neither of them the detector's family
# (Claude). Two matters: with a single generator the template-reuse signal
# can memorize one model's phrasing, and the reported numbers would not
# survive a new attacker model.
#   - minimax-m3 is free-tier, rate-limited but usable.
#   - The gemma / glm / nemotron free tiers were all rate-limited to
#     unusability at build time, and nemotron leaks its reasoning preamble
#     into the output ("We need a 30-word review...") — a detectable artifact
#     rather than realistic fraud text. The second slot is therefore a paid
#     model, ~$0.01 for the whole corpus.
GENERATOR_MODELS = [
    "minimax/minimax-m3:free",
    "mistralai/mistral-small-24b-instruct-2501",
]

# 40 topic groups per domain; held-out groups are chosen by hash at split
# time (agents/content_forensics.py), never here.
REVIEW_TOPICS = [
    "wireless mouse", "blender", "running shoes", "bluetooth headphones",
    "phone case", "coffee maker", "desk lamp", "laptop backpack",
    "mechanical keyboard", "yoga mat", "insulated water bottle", "phone charger",
    "memory foam pillow", "non-stick cookware set", "garden hose", "external SSD",
    "monitor stand", "air fryer", "hair dryer", "board game",
    "electric toothbrush", "baby stroller", "car vacuum", "electric kettle",
    "sunscreen", "face moisturizer", "protein powder", "dog chew toys",
    "cat litter box", "winter jacket", "hiking boots", "travel suitcase",
    "smart light bulbs", "robot vacuum", "cast iron skillet", "office chair",
    "standing desk", "usb-c hub", "wireless earbuds", "bookshelf",
]
DISPUTE_TOPICS = [
    "an unauthorized charge on your credit card",
    "a refund that never arrived after returning an item",
    "a package marked delivered but never received",
    "a subscription still charging after cancellation",
    "being double-billed for one purchase",
    "a late fee charged despite paying on time",
    "a fraudulent charge after card skimming",
    "an item that arrived damaged and the seller ignoring you",
    "a duplicate subscription charge in the same month",
    "a payment that failed but still deducted from your account",
    "an overcharge at a restaurant on your card",
    "a cash withdrawal you never made from an ATM",
    "a hotel charging a deposit that was never returned",
    "a cancelled flight refund still pending after months",
    "a warranty claim denied for a covered defect",
    "a price higher at checkout than advertised",
    "a gift card balance that vanished",
    "a recurring donation that will not stop after cancelling",
    "a wire transfer that never reached the recipient",
    "a merchant charging you twice for one meal",
    "a phone bill with services you never activated",
    "an insurance premium charged after policy cancellation",
    "a returned rental car billed for damage you did not cause",
    "a delivery service charging a tip you did not authorize",
    "a credit card annual fee on a card you closed",
    "a utility bill with someone else's charges merged in",
    "a bank transfer flagged then fees charged anyway",
    "a marketplace purchase that never shipped and the seller vanished",
    "a subscription free trial that charged immediately",
    "a package stolen from your doorstep the seller refuses to refund",
    "a charge in a foreign currency you never made",
    "a gym membership charging after you moved cities",
    "a double ATM withdrawal for one cash amount",
    "a streaming service charging three times in one month",
    "an online order cancelled by the seller but still billed",
    "a taxi ride charged at surge pricing you never accepted",
    "a rent deposit never returned after moving out",
    "a course purchased but never given access to",
    "a fraudulent in-app purchase on your account",
    "a checkout that saved your card and charged it later without consent",
]

PERSONAS = [
    "a college student", "a busy parent of two", "a retiree",
    "an office worker", "a small business owner", "a gamer",
    "a teacher", "a nurse",
]
REVIEW_TONES = ["positive", "negative", "mixed — some praise, some complaints"]
DISPUTE_TONES = ["angry and frustrated", "polite but firm", "urgent and stressed"]

REVIEW_TEMPLATES = [
    "Write a product review for {topic}, written by {persona}. Tone: {tone}. "
    "Roughly {w} words. Output only the review text, nothing else.",
    "You are {persona} leaving a review on an e-commerce site for {topic}. "
    "Your feelings are {tone}. Write roughly {w} words, exactly as a real "
    "shopper would type them. Output only the review text.",
    "Pretend to be {persona}. Leave a review of about {w} words for {topic}. "
    "Overall impression: {tone}. Output only the review text.",
    "Review of {topic}, in the voice of {persona}: about {w} words, {tone}. "
    "Output only the review text.",
    "Write what {persona} would post as a customer review for {topic}. "
    "Make it {tone}, about {w} words. Output only the review text.",
    "Product review request. Product: {topic}. Writer: {persona}. "
    "Sentiment: {tone}. Length: about {w} words. Output only the review text.",
]
DISPUTE_TEMPLATES = [
    "Write a dispute narrative that {persona} would send to their payment "
    "provider about {topic}. Tone: {tone}. About {w} words. "
    "Output only the dispute text.",
    "You are {persona} disputing {topic}. Write the message you would submit "
    "to your bank's dispute form. Be {tone}. About {w} words. "
    "Output only the dispute text.",
    "Pretend to be {persona} writing a formal complaint about {topic} to a "
    "payments company. Sound {tone}. About {w} words. "
    "Output only the complaint text.",
    "Dispute narrative: {persona}, {topic}, {tone}, about {w} words. "
    "Output only the text of the narrative.",
    "Write the message {persona} submits to support to dispute {topic}. "
    "Keep it {tone}, around {w} words. Output only the message text.",
    "A dispute letter from {persona} regarding {topic}, {tone} in tone, "
    "about {w} words. Output only the letter text.",
]


def n_sentences(text: str) -> int:
    import re
    return max(len([s for s in re.split(r"[.!?]+", text) if s.strip()]), 1)


def load_human_reviews(n: int) -> pd.DataFrame:
    df = pd.read_parquet(CORPUS_DIR / "reviews.parquet", columns=["id", "text"])
    df = df.dropna(subset=["text"])
    # Length band: >=2 sentences, 80-800 chars — must overlap the AI band or
    # length becomes a giveaway signal.
    mask = (df["text"].str.len() >= 80) & (df["text"].str.len() <= 800)
    df = df[mask].drop_duplicates("text").sample(frac=1.0, random_state=SEED)
    df = df[df["text"].map(n_sentences) >= 2].head(n)
    return pd.DataFrame({
        "text": df["text"].str.strip(),
        "is_ai_generated": 0,
        "source": "human_amazon_reviews",
        "domain": "review",
        "prompt_group": df["id"],
    })


# ChatGPT's public release. Text collected after this date cannot be assumed
# human-written, however it is labelled.
LLM_ERA_START = "2022-11-30"


def load_human_disputes(n: int) -> pd.DataFrame:
    df = pd.read_parquet(
        CORPUS_DIR / "complaints.parquet",
        columns=["Complaint ID", "Consumer complaint narrative", "Product",
                 "Date received"],
    )
    df = df.dropna(subset=["Consumer complaint narrative"])

    # PROVENANCE GATE. The human class needs a guarantee, not an assumption.
    # 62.9% of CFPB narratives in this dump were received after ChatGPT shipped,
    # so an unfiltered "human" sample may contain AI-written or AI-assisted
    # complaints — mislabelled negatives that make a detector's false positives
    # uninterpretable (some of them might be correct). Restricting to
    # pre-release filings is the only cheap way to be certain the human class
    # is human. 36,201 narratives survive this plus the length band, against
    # the 200 needed, so it costs nothing in sample size.
    before = len(df)
    df["Date received"] = pd.to_datetime(df["Date received"], errors="coerce")
    df = df[df["Date received"] < LLM_ERA_START]
    print(f"  provenance gate: {before:,} -> {len(df):,} narratives filed "
          f"before {LLM_ERA_START} (pre-LLM, so certainly human-written)")
    df = df.rename(columns={"Consumer complaint narrative": "narrative",
                            "Complaint ID": "cid"})
    # CFPB redacts PII as runs of X. The first version of this loader mapped
    # those to "REDACTED" in capitals — a token appearing in 81.5% of human
    # disputes and 0% of AI ones, which pushed human caps_ratio to 0.172 vs
    # 0.030 and let a classifier score well by detecting the redaction
    # convention rather than AI authorship. A lowercase placeholder keeps the
    # sentence intact without handing over a capitalisation giveaway.
    df["narrative"] = (
        df["narrative"]
        .str.replace(r"XX/XX/(?:XXXX|XX)", "a date", regex=True)
        .str.replace(r"\{?\$[\d,.]+\}?", "an amount", regex=True)
        .str.replace(r"XX+", "redacted", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    mask = (df["narrative"].str.len() >= 80) & (df["narrative"].str.len() <= 800)
    df = df[mask].drop_duplicates("narrative").sample(frac=1.0, random_state=SEED)
    df = df[df["narrative"].map(n_sentences) >= 2].head(n)
    return pd.DataFrame({
        "text": df["narrative"].str.strip(),
        "is_ai_generated": 0,
        "source": "human_cfpb_complaints",
        "domain": "dispute",
        "prompt_group": df["cid"].astype(str),
    })


def word_count(text: str) -> int:
    import re
    return max(len(re.findall(r"[A-Za-z\']+", str(text))), 1)


def build_generation_matrix(topics: list[str], templates: list[str],
                            tones: list[str], domain: str,
                            human_word_counts: list[int]) -> list[dict]:
    """One generation spec per sample; seeded RNG → reproducible + cache-friendly.

    The word budget for each sample is DRAWN FROM the human word-count
    distribution of the same domain, so the two classes are length-matched by
    construction rather than by discarding data afterwards. Length is the
    single easiest way to accidentally build a detector that only measures
    verbosity — this is what stops that.
    """
    rng = random.Random(SEED)
    samples_per_group = max(N_PER_DOMAIN // len(topics), 1)
    specs = []
    for gi, topic in enumerate(topics):
        for j in range(samples_per_group):
            specs.append({
                "domain": domain,
                "prompt_group": f"{domain}_topic{gi:02d}",
                "topic": topic,
                "persona": rng.choice(PERSONAS),
                "tone": rng.choice(tones),
                "template": rng.choice(templates),
                "w": int(rng.choice(human_word_counts)),
                "model": GENERATOR_MODELS[(gi + j) % len(GENERATOR_MODELS)],
            })
    return specs[:N_PER_DOMAIN]


def generate_one(spec: dict) -> dict | None:
    prompt = spec["template"].format(
        topic=spec["topic"], persona=spec["persona"],
        tone=spec["tone"], w=spec["w"],
    )
    messages = [{"role": "user", "content": prompt}]
    for model in [spec["model"]] + [m for m in GENERATOR_MODELS if m != spec["model"]]:
        try:
            out = chat(messages, model=model, temperature=0.9,
                       max_tokens=max(120, int(spec["w"] * 3)), use_cache=True)
            out = out.strip()
            if len(out) >= 40:            # reject empty/garbage generations
                return {**spec, "text": out}
        except Exception as e:
            print(f"    {model} failed ({type(e).__name__}), falling back")
    return None


def trim_to_length_parity(df: pd.DataFrame, tol: float = 0.12,
                          max_drop: float = 0.20) -> pd.DataFrame:
    """Trim the tails of whichever class is longer until the per-domain mean
    lengths agree within `tol`.

    Generators do not hit a word budget exactly — asked for the human dispute
    distribution they still came back ~18% short on average. Rather than
    discard half the corpus to get an exact 1:1 length match, this drops at
    most `max_drop` of each (domain, class) group from the offending tail,
    which is enough to close the gap while keeping most of the data. If the
    gap cannot be closed within that budget the rows stay and main() prints
    the warning, so the confound is reported rather than hidden.
    """
    keep = []
    for domain, block in df.groupby("domain"):
        human = block[block["is_ai_generated"] == 0].copy()
        ai = block[block["is_ai_generated"] == 1].copy()
        for _ in range(200):
            h_mean, a_mean = human["text"].str.len().mean(), ai["text"].str.len().mean()
            if abs(a_mean - h_mean) / max(h_mean, 1) <= tol:
                break
            longer, shorter = (human, ai) if h_mean > a_mean else (ai, human)
            if len(longer) <= (1 - max_drop) * len(block[
                    block["is_ai_generated"] == (0 if longer is human else 1)]):
                break
            # drop the single longest row of the longer class, and the
            # single shortest of the shorter class, keeping sizes comparable
            longer.drop(longer["text"].str.len().idxmax(), inplace=True)
            if len(shorter) > (1 - max_drop) * len(shorter):
                pass
            human, ai = (longer, shorter) if longer is human else (shorter, longer)
        keep.append(pd.concat([human, ai]))
    out = pd.concat(keep, ignore_index=True)
    print(f"Length-parity trim: {len(df) - len(out)} rows dropped from the "
          f"longer-class tails")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", choices=["full", "plan"], default="full",
                    help="'plan' = 100 per class per domain (risk-table fallback)")
    args = ap.parse_args()

    global N_PER_DOMAIN
    if args.scale == "plan":
        N_PER_DOMAIN = 100

    print(f"Scale: {args.scale} ({N_PER_DOMAIN} human + {N_PER_DOMAIN} AI per domain)")
    humans = pd.concat([load_human_reviews(N_PER_DOMAIN),
                        load_human_disputes(N_PER_DOMAIN)], ignore_index=True)
    print(f"Human samples: {len(humans):,} "
          f"({(humans['domain']=='review').sum()} reviews, "
          f"{(humans['domain']=='dispute').sum()} disputes)")

    # Word budgets are drawn from the human texts of the SAME domain.
    wc = {dom: humans.loc[humans["domain"] == dom, "text"].map(word_count).tolist()
          for dom in ("review", "dispute")}
    for dom, counts in wc.items():
        print(f"  human {dom} word counts: median {int(pd.Series(counts).median())}, "
              f"p10 {int(pd.Series(counts).quantile(0.1))}, "
              f"p90 {int(pd.Series(counts).quantile(0.9))}")

    specs = (build_generation_matrix(REVIEW_TOPICS, REVIEW_TEMPLATES,
                                     REVIEW_TONES, "review", wc["review"])
             + build_generation_matrix(DISPUTE_TOPICS, DISPUTE_TEMPLATES,
                                       DISPUTE_TONES, "dispute", wc["dispute"]))
    print(f"Generation specs: {len(specs)} "
          f"(models: {', '.join(GENERATOR_MODELS)})")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(generate_one, s): s for s in specs}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            if r is not None:
                results.append(r)
            if i % 25 == 0:
                print(f"  {i}/{len(specs)} done ({len(results)} kept)")

    if len(results) < 0.8 * len(specs):
        print(f"WARNING: only {len(results)}/{len(specs)} generations succeeded "
              f"— re-run to resume from cache, or use --scale plan")

    ai = pd.DataFrame([{
        "text": r["text"], "is_ai_generated": 1, "source": f"ai_{r['model'].split('/')[0]}",
        "domain": r["domain"], "prompt_group": r["prompt_group"],
    } for r in results])
    df = pd.concat([humans, ai], ignore_index=True)

    # Same character band on BOTH classes. Previously this was enforced on the
    # human side only, which let AI texts run to 1400 chars and turned length
    # into a free giveaway signal.
    n_before = len(df)
    ln = df["text"].str.len()
    df = df[(ln >= 80) & (ln <= 800)].reset_index(drop=True)
    print(f"\nLength band 80-800 chars applied to both classes: "
          f"{n_before - len(df)} rows dropped")

    df = trim_to_length_parity(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nWrote {len(df):,} rows -> {OUT_PATH}")
    print(df.groupby(["domain", "is_ai_generated"]).size().to_string())
    print("\nLength match check (chars) — these must be close, or the "
          "detector is partly a length detector:")
    stats = (df.assign(chars=df["text"].str.len(),
                       words=df["text"].map(word_count))
               .groupby(["domain", "is_ai_generated"])[["chars", "words"]]
               .mean().round(0))
    print(stats.to_string())
    for dom in ("review", "dispute"):
        h = stats.loc[(dom, 0), "chars"]
        a = stats.loc[(dom, 1), "chars"]
        gap = abs(a - h) / max(h, 1)
        flag = "OK" if gap <= 0.15 else "TOO WIDE — length is a confound"
        print(f"  {dom}: human {h:.0f} vs AI {a:.0f} chars "
              f"({gap:.1%} gap) -> {flag}")


if __name__ == "__main__":
    main()
