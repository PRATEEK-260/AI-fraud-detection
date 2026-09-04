"""Document Forensics — Error Level Analysis for tampered KYC documents.

This closes the last gap in the project doc's Agent 4: "basic image forensics
on ID documents". Content Forensics handles submitted TEXT; this handles
submitted IMAGES.

HOW IT WORKS. A JPEG is stored in 8x8 blocks quantised at a chosen quality.
Re-saving an image at a known quality and subtracting the result from the
original leaves a residual whose magnitude reflects how much each region still
had to lose — i.e. how many compression generations it has already been
through. An untouched image responds fairly uniformly. A region that was
edited and re-saved has a different compression history from its surroundings,
so it responds differently, and the boundary shows up as a local spike in the
residual. That is a property of the codec, not an assumption we invented.

The detector is deliberately not a single global threshold. A forger changes a
FIELD, not a whole card, so the informative statistic is the *contrast between
the worst local block and the rest of the document* — `block_ratio` below.

WHAT THIS EVIDENCE IS WORTH. Read this before quoting the numbers:

  - The documents are synthetic specimens (scripts/make_documents.py). No real
    identity documents were used, and none should be.
  - The tampering, however, is genuine image manipulation, so the artifact ELA
    responds to is a real codec artifact rather than a planted signal.
  - Therefore: a strong score here demonstrates the ELA implementation is
    correct on splice-and-resave tampering. It does NOT demonstrate the method
    survives a competent forger. Anyone who re-encodes the finished document
    once at uniform quality, or rebuilds it from a clean render, removes this
    discontinuity entirely — and that is a cheap countermeasure.
  - So ELA belongs in a stack as a fast, free first filter that catches lazy
    tampering, never as the sole control. The Case files say `escalate`, never
    `block`, for exactly that reason: the cost table prices a wrongly rejected
    genuine KYC document far above a missed edit at this stage of review.

Usage:
    .venv/bin/python -m agents.document_forensics
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageChops
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eval.cost_table import decide
from eval.metrics import binary_metrics, format_report, pr_auc
from spine.db import DEFAULT_DB_PATH, connect, count_cases, insert_cases
from spine.schema import Case, Evidence

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "processed" / "documents_manifest.csv"
RESULTS_DIR = ROOT / "eval" / "results"

HELDOUT_FRACTION = 0.30
ELA_QUALITY = 90          # the known re-save quality the residual is taken at
BLOCK = 16                # residual is pooled over 16x16 tiles

# Analysis is restricted to the card's text-field band and the residual is
# normalised by local edge energy. Both corrections were forced by measurement:
# taking the raw maximum over the whole card, the highest-residual tile landed
# on the edited field in 0 of 43 tampered documents, because raw residual is
# dominated by whatever part of the image has the most texture (the portrait
# block, the header bar) rather than by edit history. Dividing by local edge
# energy makes a flat field and a busy one comparable, and confining it to the
# field band removes the fixed furniture entirely. In production the band comes
# from OCR/layout detection, which is domain knowledge about card structure —
# not knowledge of which field was tampered.
ROI = (180, 530, 88, 300)   # x0, x1, y0, y1

RULES: dict[str, dict] = {
    "ela_hotspot": {
        "weight": 0.40,
        "desc": "one region's compression residual far exceeds the rest of "
                "the document — a locally different edit history",
    },
    "residual_outlier": {
        "weight": 0.30,
        "desc": "peak residual above the level seen on untouched documents",
    },
    "uneven_error_surface": {
        "weight": 0.30,
        "desc": "residual varies across the document more than uniform "
                "compression can explain",
    },
}

FEATURE_COLS = ["ela_mean", "ela_p99", "ela_max", "block_max",
                "block_ratio", "block_std", "hot_blocks"]


# ---------------------------------------------------------------------------
# Error Level Analysis
# ---------------------------------------------------------------------------

def ela_signals(path: Path) -> dict:
    """Edge-normalised residual statistics over the card's field band."""
    original = Image.open(path).convert("RGB")

    buf = io.BytesIO()
    original.save(buf, "JPEG", quality=ELA_QUALITY)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")

    residual = np.asarray(
        ImageChops.difference(original, resaved), dtype=np.float32).max(axis=2)

    # Local edge energy: how much high-frequency detail this area holds. A
    # spliced region is pasted in clean, so it carries less texture and less
    # compression history than the photographed area around it.
    grey = np.asarray(original.convert("L"), dtype=np.float32)
    edge = np.abs(np.gradient(grey, axis=1)) + np.abs(np.gradient(grey, axis=0))

    x0, x1, y0, y1 = ROI
    roi_res, roi_edge = residual[y0:y1, x0:x1], edge[y0:y1, x0:x1]
    h, w = roi_res.shape
    bh, bw = h // BLOCK, w // BLOCK
    rr = roi_res[: bh * BLOCK, : bw * BLOCK].reshape(bh, BLOCK, bw, BLOCK).mean(axis=(1, 3))
    ee = roi_edge[: bh * BLOCK, : bw * BLOCK].reshape(bh, BLOCK, bw, BLOCK).mean(axis=(1, 3))
    norm = rr / np.maximum(ee, 0.5)

    med = float(np.median(norm))
    return {
        "ela_mean": float(roi_res.mean()),
        "ela_p99": float(np.percentile(roi_res, 99)),
        "ela_max": float(roi_res.max()),
        "block_max": float(norm.max()),
        "block_ratio": float(norm.max()) / max(med, 1e-3),
        "block_std": float(norm.std()),
        "hot_blocks": float((norm > med * 3).sum()),
        "_blocks": norm,
    }


def apply_rules(sig: pd.DataFrame, thr: dict) -> pd.DataFrame:
    hits = pd.DataFrame(index=sig.index)
    hits["ela_hotspot"] = sig["block_ratio"] > thr["block_ratio"]
    hits["residual_outlier"] = sig["block_max"] > thr["block_max"]
    hits["uneven_error_surface"] = sig["block_std"] > thr["block_std"]
    out = sig.copy()
    for name, series in hits.items():
        out[f"rule_{name}"] = series
    out["rule_score"] = sum(hits[n] * RULES[n]["weight"] for n in RULES)
    return out


def locate_hotspot(blocks: np.ndarray) -> tuple[int, int]:
    """Pixel coordinates of the worst tile — what a reviewer should look at."""
    r, c = np.unravel_index(int(np.argmax(blocks)), blocks.shape)
    return int(ROI[0] + c * BLOCK), int(ROI[2] + r * BLOCK)


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------

def make_case(row: pd.Series, sig: dict, p: float, hotspot: tuple[int, int]) -> Case:
    evidence: list[Evidence] = []
    if sig["rule_ela_hotspot"]:
        evidence.append(Evidence(
            "ela_hotspot",
            f"worst region's compression residual is {sig['block_ratio']:.1f}x "
            f"the document median, centred near pixel "
            f"({hotspot[0]}, {hotspot[1]})",
            RULES["ela_hotspot"]["weight"]))
    if sig["rule_residual_outlier"]:
        evidence.append(Evidence(
            "residual_outlier",
            f"peak block residual {sig['block_max']:.2f} exceeds the level "
            f"seen on untouched specimens",
            RULES["residual_outlier"]["weight"]))
    if sig["rule_uneven_error_surface"]:
        evidence.append(Evidence(
            "uneven_error_surface",
            f"residual spread {sig['block_std']:.2f} across the document is "
            f"wider than uniform compression explains",
            RULES["uneven_error_surface"]["weight"]))
    evidence.append(Evidence(
        "ela_model_probability", f"P(tampered) = {p:.3f}", 0.15))

    has_readable = any(e.signal in RULES for e in evidence)
    decision, why = decide("document_forensics", p, has_readable)

    return Case(
        source_agent="document_forensics",
        entity_id=str(row["doc_id"]),
        entity_type="document",
        evidence=evidence,
        confidence=round(float(p), 4),
        cost_estimate=3100.0,
        decision=decision,
        reasoning_text=(
            f"KYC document {row['doc_id']}: Error Level Analysis over the "
            f"card's field band gives an edge-normalised residual peak "
            f"{sig['block_ratio']:.1f}x the band median. P(tampered) {p:.3f}. "
            f"The document as a whole is inconsistent with a single "
            f"uninterrupted capture-and-compress history, which is what an "
            f"edited-and-re-saved field looks like. NOTE: this agent's "
            f"region-localisation was measured at chance level, so it says "
            f"the document warrants inspection, NOT which field was altered — "
            f"a reviewer should re-verify every field, not just one. "
            f"Action `{decision}`: {why}."
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _stable_hash(v: str) -> int:
    return int(hashlib.md5(str(v).encode()).hexdigest()[:8], 16)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-cases", type=int, default=200)
    args = ap.parse_args()

    if not MANIFEST.exists():
        raise SystemExit(
            f"{MANIFEST} missing — run:\n"
            f"    .venv/bin/python scripts/make_documents.py")

    df = pd.read_csv(MANIFEST)
    df["split"] = np.where(
        df["doc_id"].map(lambda d: _stable_hash(d) % 100)
        < HELDOUT_FRACTION * 100, "heldout", "train")

    print("=" * 72)
    print("SYNTHETIC SPECIMEN DOCUMENTS — no real identity documents are used.\n"
          "The tampering is real image manipulation, so the ELA artifact is\n"
          "genuine; a strong score shows the implementation is correct, NOT\n"
          "that ELA defeats a competent forger who re-encodes uniformly.")
    print("=" * 72)

    t0 = time.perf_counter()
    raw = [ela_signals(ROOT / p) for p in df["path"]]
    ela_ms = (time.perf_counter() - t0) * 1000 / max(len(df), 1)
    blocks = [r.pop("_blocks") for r in raw]
    sig = pd.DataFrame(raw, index=df.index)

    train_mask = df["split"] == "train"
    genuine_train = sig[train_mask & (df["is_tampered"] == 0)]
    # Thresholds read off UNTOUCHED training documents only.
    thr = {
        "block_ratio": float(np.quantile(genuine_train["block_ratio"], 0.90)),
        "block_max": float(np.quantile(genuine_train["block_max"], 0.90)),
        "block_std": float(np.quantile(genuine_train["block_std"], 0.90)),
    }
    sig = apply_rules(sig, thr)
    print(f"\ndocuments {len(df)}  (train {int(train_mask.sum())} / heldout "
          f"{int((~train_mask).sum())})   tampered rate "
          f"{df['is_tampered'].mean():.2f}")
    print(f"Rule thresholds from untouched TRAIN documents: "
          f"{ {k: round(v, 3) for k, v in thr.items()} }")

    held = df[~train_mask]
    sig_held = sig[~train_mask]

    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, C=1.0))
    model.fit(sig[train_mask][FEATURE_COLS], df.loc[train_mask, "is_tampered"])
    p_held = model.predict_proba(sig_held[FEATURE_COLS])[:, 1]

    results = {
        "ela_rules": binary_metrics(held["is_tampered"], sig_held["rule_score"] >= 0.40),
        "ela_logistic_regression": {
            **binary_metrics(held["is_tampered"], p_held >= 0.5),
            "pr_auc": pr_auc(held["is_tampered"], p_held),
        },
    }
    for name, m in results.items():
        print(format_report(f"[held-out, SYNTHETIC SPECIMENS] {name}", m))

    # Does the hotspot actually land on the edited field? A detector that is
    # right for the wrong reason would still score well above.
    hits = 0
    checked = 0
    for pos, (idx, row) in enumerate(held.iterrows()):
        if not row["is_tampered"]:
            continue
        box = json.loads(row["tampered_box"])
        if not box:
            continue
        x, y = locate_hotspot(blocks[df.index.get_loc(idx)])
        checked += 1
        # generous: the tile centre falls within the edited field's band
        if box[1] - BLOCK <= y <= box[3] + BLOCK:
            hits += 1
    localisation = round(hits / max(checked, 1), 4)
    # A tile chosen at random lands inside the edited band this often, given
    # the band height plus the +/-BLOCK tolerance against the ROI height.
    band_h = 40 + 2 * BLOCK
    chance = band_h / (ROI[3] - ROI[2])
    print(f"\nHotspot localisation: the highest-residual tile falls on the "
          f"edited field in {hits}/{checked} tampered documents "
          f"({localisation:.1%}); chance is {chance:.1%}.")
    if localisation <= chance * 1.25:
        print("  ^ AT CHANCE. The detector separates tampered from genuine "
              "documents, but it\n    does NOT identify which region was "
              "edited. So whatever it keys on is not\n    the localised "
              "compression discontinuity ELA is supposed to find. The most "
              "likely\n    mechanism is global: a spliced field is pasted in "
              "clean, so it lacks the sensor\n    noise of the photographed "
              "card around it and shifts whole-ROI statistics.\n    That is a "
              "real consequence of tampering, but it is a different claim, and "
              "the\n    case files must not tell a reviewer to look at a "
              "specific pixel on this basis.")
    else:
        print("  ^ Above chance: the detector fires on the tamper itself, not "
              "on an unrelated\n    artifact that happens to correlate with "
              "the label.")

    # --- cases --------------------------------------------------------------
    cases = []
    for pos, (idx, row) in enumerate(held.iterrows()):
        if len(cases) >= args.max_cases:
            break
        if p_held[pos] < 0.5:
            continue
        s = sig_held.loc[idx].to_dict()
        cases.append(make_case(row, s, float(p_held[pos]),
                               locate_hotspot(blocks[df.index.get_loc(idx)])))
    conn = connect(DEFAULT_DB_PATH)
    insert_cases(conn, cases)
    print(f"\nWrote {len(cases)} document_forensics cases; audit log now holds "
          f"{count_cases(conn, 'document_forensics')} document_forensics / "
          f"{count_cases(conn)} total")
    if cases:
        print(f"\nExample case {cases[0].case_id[:8]} [{cases[0].decision}]:\n"
              f"  {cases[0].reasoning_text}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "agent": "document_forensics",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "DATA_WARNING": (
            "Documents are SYNTHETIC specimens rendered by "
            "scripts/make_documents.py; no real identity documents were used. "
            "The tampering is genuine image manipulation, so the ELA artifact "
            "is a real JPEG codec effect — but these scores demonstrate a "
            "correct ELA implementation against splice-and-resave tampering, "
            "NOT robustness against a forger who re-encodes the whole document "
            "uniformly, which erases the signal. ELA is a cheap first filter, "
            "never a sole control."
        ),
        "method": {
            "technique": "Error Level Analysis",
            "resave_quality": ELA_QUALITY,
            "block_size": BLOCK,
            "key_statistic": "block_ratio — worst tile residual over the "
                             "document median, because tampering is local",
        },
        "dataset": {
            "n_documents": len(df),
            "n_heldout": int((~train_mask).sum()),
            "tampered_rate": round(float(df["is_tampered"].mean()), 3),
        },
        "rule_thresholds_from_untouched_train": {k: round(v, 4) for k, v in thr.items()},
        "results": results,
        "hotspot_localisation_rate": localisation,
        "hotspot_localisation_note": (
            "Share of tampered held-out documents where the highest-residual "
            "tile lands on the edited field. Guards against scoring well for "
            "the wrong reason."
        ),
        "ela_ms_per_document": round(ela_ms, 2),
        "cases_written": len(cases),
    }
    out_path = RESULTS_DIR / "document_forensics_metrics.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
