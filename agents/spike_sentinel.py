"""Spike Sentinel — per-user transaction velocity & anomaly detection.

Pipeline:
    1. Feature engineering that uses ONLY each user's prior transactions
       (no leakage — shift(1) before every rolling stat, trailing-window
       counts exclude the current row):
         trailing 1h/24h tx counts, amount z-score and ratio vs the user's
         rolling history, minutes since previous tx, haversine distance from
       home, hour/night, category & merchant risk (target-encoded on train
       rows only).
    2. A hand-tuned rule layer producing interpretable velocity signals —
       these become the evidence in each Case file.
    3. XGBoost on the engineered features (early stopping on a time-slice
       validation set carved from TRAIN — never from held-out).
    4. Weighted ensemble (0.7 model + 0.3 rules) -> decision + Case objects
       written to the SQLite audit log.

Usage:
    .venv/bin/python -m agents.spike_sentinel [--users N] [--fresh-db]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from eval.cost_table import decide
from eval.metrics import best_f1_threshold, binary_metrics, format_report, pr_auc
from spine.db import connect, count_cases, insert_cases
from spine.schema import Case, Evidence

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "eval" / "results"
CASES_DB = ROOT / "data" / "cases.db"

HOUR_S = 3600
DAY_S = 86400

# Weighted ensemble: the model carries most of the discrimination; the rules
# keep every flag explainable and provide the hard evidence signals.
MODEL_WEIGHT = 0.7
RULE_WEIGHT = 0.3

# Validation = last 10% of the train period (time-slice, never held-out).
VALID_FRACTION = 0.10

FEATURE_COLS = [
    "amt_log", "hour", "dow", "is_night", "age_years",
    "city_pop_log", "minutes_since_prev_log", "tx_count_1h", "tx_count_24h",
    "amt_zscore", "amt_ratio", "category_te", "merchant_te",
]


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features using only each user's PRIOR transactions.

    Note: haversine distance home→merchant was measured on train and carries
    zero fraud signal in this dataset (fraud and legit distributions are
    identical), so it is deliberately excluded.
    """
    df = df.sort_values(["cc_num", "unix_time"], kind="mergesort").copy()
    unix = df["unix_time"].to_numpy()
    g = df.groupby("cc_num", sort=False)

    dt = df["trans_date_trans_time"]
    df["hour"] = dt.dt.hour.astype(float)
    df["dow"] = dt.dt.dayofweek.astype(float)
    df["is_night"] = (dt.dt.hour <= 5).astype(int)
    df["age_years"] = (dt - pd.to_datetime(df["dob"])).dt.days / 365.25
    df["city_pop_log"] = np.log1p(df["city_pop"].astype(float))
    df["amt_log"] = np.log1p(df["amt"].astype(float))

    # Gap to the user's previous transaction (first ever -> 24h sentinel).
    df["minutes_since_prev"] = (g["unix_time"].diff() / 60.0).fillna(24 * 60.0)
    df["minutes_since_prev_log"] = np.log1p(df["minutes_since_prev"])

    # Exact trailing-window counts per user (exclude the current row):
    # count of prior transactions within the trailing 1h / 24h.
    n = len(df)
    c1h = np.zeros(n, dtype=np.float64)
    c24h = np.zeros(n, dtype=np.float64)
    for positions in g.indices.values():
        t = unix[positions]  # time-sorted within user (frame is sorted)
        c1h[positions] = np.arange(len(t)) - np.searchsorted(t, t - HOUR_S, side="right")
        c24h[positions] = np.arange(len(t)) - np.searchsorted(t, t - DAY_S, side="right")
    df["tx_count_1h"] = c1h
    df["tx_count_24h"] = c24h

    # Amount vs the user's own rolling history (prior 10 tx only).
    df["amt_mean_prev10"] = g["amt"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    )
    std_prev10 = g["amt"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).std()
    )
    z = (df["amt"] - df["amt_mean_prev10"]) / std_prev10.where(std_prev10 > 0)
    df["amt_zscore"] = z.fillna(0.0).clip(-50, 50)
    df["amt_ratio"] = (df["amt"] / df["amt_mean_prev10"].clip(lower=1.0)).clip(0, 100)
    df["amt_ratio"] = df["amt_ratio"].fillna(1.0)

    return df


def fit_target_encodings(
    train_df: pd.DataFrame, cols: tuple[str, ...] = ("category", "merchant"), k: float = 50.0
) -> tuple[dict[str, dict], float]:
    """Smoothed fraud-rate encoding per category/merchant, fit on train rows only."""
    base = float(train_df["is_fraud"].mean())
    maps: dict[str, dict] = {}
    for col in cols:
        stats = train_df.groupby(col)["is_fraud"].agg(["sum", "count"])
        maps[col] = ((stats["sum"] + k * base) / (stats["count"] + k)).to_dict()
    return maps, base


# --------------------------------------------------------------------------
# Rule layer — interpretable velocity signals, these become Case evidence
# --------------------------------------------------------------------------

RULES: dict[str, dict] = {
    "velocity_burst": {
        "weight": 0.25,
        "desc": "2+ prior transactions by this card within the past hour",
    },
    "amount_spike": {
        "weight": 0.20,
        "desc": "amount >= 5x the user's rolling mean AND >= $200",
    },
    "high_amount": {
        "weight": 0.20,
        "desc": "amount >= $400 (train fraud rate at this level: ~17%)",
    },
    "night_high_amount": {
        "weight": 0.15,
        "desc": "night hours (00-05) AND amount >= $200",
    },
}


def apply_rules(fdf: pd.DataFrame) -> pd.DataFrame:
    """Add one boolean column per rule plus the aggregate rule_score."""
    hits = pd.DataFrame(index=fdf.index)
    hits["velocity_burst"] = fdf["tx_count_1h"] >= 2
    hits["amount_spike"] = (fdf["amt"] >= 5 * fdf["amt_mean_prev10"]) & (fdf["amt"] >= 200)
    hits["high_amount"] = fdf["amt"] >= 400
    hits["night_high_amount"] = (fdf["is_night"] == 1) & (fdf["amt"] >= 200)
    fdf = fdf.copy()
    for name, series in hits.items():
        fdf[f"rule_{name}"] = series
    fdf["rule_score"] = hits.mean(axis=1)
    return fdf


def rule_diagnostics(fdf: pd.DataFrame) -> dict:
    """Hit rate / precision / recall per rule — computed on validation only."""
    out = {}
    for name in RULES:
        hit = fdf[f"rule_{name}"]
        tp = int((hit & (fdf["is_fraud"] == 1)).sum())
        hits = int(hit.sum())
        out[name] = {
            "hits": hits,
            "hit_rate": round(hits / max(len(fdf), 1), 5),
            "precision": round(tp / hits, 4) if hits else 0.0,
            "recall": round(tp / max(int(fdf["is_fraud"].sum()), 1), 4),
        }
    return out


# --------------------------------------------------------------------------
# Case construction
# --------------------------------------------------------------------------

def make_case(row: pd.Series, p: float, s: float, threshold: float) -> Case:
    evidence: list[Evidence] = []
    if row["rule_velocity_burst"]:
        evidence.append(Evidence(
            "velocity_burst",
            f"{int(row['tx_count_1h'])} prior transactions within the past hour",
            RULES["velocity_burst"]["weight"]))
    if row["rule_amount_spike"]:
        mean = row["amt_mean_prev10"]
        mean_str = f"${mean:,.2f}" if pd.notna(mean) else "no history"
        evidence.append(Evidence(
            "amount_spike",
            f"${row['amt']:,.2f} vs user rolling mean {mean_str}",
            RULES["amount_spike"]["weight"]))
    if row["rule_high_amount"]:
        evidence.append(Evidence(
            "high_amount",
            f"${row['amt']:,.2f} exceeds the $400 high-risk band",
            RULES["high_amount"]["weight"]))
    if row["rule_night_high_amount"]:
        evidence.append(Evidence(
            "night_high_amount",
            f"transaction at {int(row['hour']):02d}:00 for ${row['amt']:,.2f}",
            RULES["night_high_amount"]["weight"]))
    evidence.append(Evidence(
        "model_probability", f"XGBoost P(fraud) = {p:.3f}", 0.35))
    evidence.append(Evidence(
        "ensemble_score", f"weighted score {s:.3f} vs threshold {threshold:.3f}", 0.10))

    reasons = "; ".join(e.value for e in evidence if e.signal in RULES) or "model-only signal"
    # The action comes from the shared cost policy, not from the score alone:
    # a flag with no interpretable rule behind it is never auto-blocked.
    has_evidence = any(e.signal in RULES for e in evidence)
    decision, policy_why = decide("spike_sentinel", p, has_evidence)
    evidence.append(Evidence("decision_policy", policy_why, 0.05))
    return Case(
        source_agent="spike_sentinel",
        entity_id=str(row["trans_num"]),
        entity_type="transaction",
        evidence=evidence,
        confidence=float(s),
        # Proxy: a wrongly-blocked legitimate transaction costs ~10% of its
        # value in merchant/consumer friction (documented in the cost table).
        cost_estimate=round(0.10 * float(row["amt"]), 2),
        decision=decision,
        reasoning_text=(
            f"Flagged ${row['amt']:,.2f} at {row['merchant']} "
            f"(category: {row['category']}) at {int(row['hour']):02d}:00. "
            f"Signals: {reasons}. Model probability {p:.3f}, ensemble score "
            f"{s:.3f} (threshold {threshold:.3f}). "
            f"Action `{decision}`: {policy_why}."
        ),
    )


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--users", type=int, default=0,
                    help="limit to the first N cards (quick iteration runs)")
    ap.add_argument("--max-cases", type=int, default=3000,
                    help="cap on cases written to the audit log this run")
    ap.add_argument("--fresh-db", action="store_true",
                    help="delete the cases DB before writing (demo cleanliness)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    train = pd.read_parquet(PROCESSED_DIR / "spike_train.parquet")
    heldout = pd.read_parquet(PROCESSED_DIR / "spike_heldout.parquet")
    print(f"Loaded train={len(train):,} heldout={len(heldout):,}")

    if args.users:
        cards = pd.unique(pd.concat([train["cc_num"], heldout["cc_num"]]))[: args.users]
        train = train[train["cc_num"].isin(cards)]
        heldout = heldout[heldout["cc_num"].isin(cards)]
        print(f"Limited to {args.users} cards: train={len(train):,} heldout={len(heldout):,}")

    # Features are built over the combined timeline so a user's held-out
    # transactions see their train-period history (as in a streaming system).
    # The model is FIT on train rows only — no label leakage.
    combined = pd.concat(
        [train.assign(_origin="train"), heldout.assign(_origin="heldout")],
        ignore_index=True,
    )
    feat = build_features(combined)

    enc_maps, base_rate = fit_target_encodings(feat[feat["_origin"] == "train"])
    for col, mapping in enc_maps.items():
        feat[f"{col}_te"] = feat[col].map(mapping).fillna(base_rate)

    train_f = feat[feat["_origin"] == "train"]
    held_f = feat[feat["_origin"] == "heldout"]

    # Time-slice validation: the last VALID_FRACTION of the train period.
    cut = train_f["trans_date_trans_time"].quantile(1 - VALID_FRACTION)
    fit_df = train_f[train_f["trans_date_trans_time"] <= cut]
    valid_df = train_f[train_f["trans_date_trans_time"] > cut]

    valid_df = apply_rules(valid_df)

    print(f"fit={len(fit_df):,}  valid={len(valid_df):,}  "
          f"heldout={len(held_f):,}  (heldout fraud rate "
          f"{held_f['is_fraud'].mean():.4%})")
    print("Rule diagnostics (validation slice, tuning only):")
    for name, d in rule_diagnostics(valid_df).items():
        print(f"  {name:<18} hits={d['hits']:>7,}  precision={d['precision']:.3f}  "
              f"recall={d['recall']:.3f}")

    # --- Model -------------------------------------------------------------
    pos = int(fit_df["is_fraud"].sum())
    scale_pos_weight = (len(fit_df) - pos) / max(pos, 1)
    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
        early_stopping_rounds=40, tree_method="hist", random_state=42,
    )
    t0 = time.perf_counter()
    model.fit(
        fit_df[FEATURE_COLS], fit_df["is_fraud"],
        eval_set=[(valid_df[FEATURE_COLS], valid_df["is_fraud"])], verbose=False,
    )
    print(f"\nModel trained in {time.perf_counter() - t0:.1f}s "
          f"(best_iteration={model.best_iteration})")

    p_valid = model.predict_proba(valid_df[FEATURE_COLS])[:, 1]

    # --- Thresholds tuned on VALIDATION only --------------------------------
    t_model = best_f1_threshold(valid_df["is_fraud"], p_valid)
    s_valid = MODEL_WEIGHT * p_valid + RULE_WEIGHT * valid_df["rule_score"]
    t_ens = best_f1_threshold(valid_df["is_fraud"], s_valid)
    print(f"Thresholds (tuned on validation): model={t_model:.3f}  ensemble={t_ens:.3f}")

    # --- Final held-out evaluation (single pass, thresholds frozen) ---------
    # Latency covers the full per-transaction decision path: rules + model.
    t0 = time.perf_counter()
    held_f = apply_rules(held_f)
    p_held = model.predict_proba(held_f[FEATURE_COLS])[:, 1]
    latency_ms = (time.perf_counter() - t0) * 1000 / len(held_f)

    s_held = MODEL_WEIGHT * p_held + RULE_WEIGHT * held_f["rule_score"]
    flags = {
        "rules_only": held_f["rule_score"] > 0,
        "xgboost": p_held >= t_model,
        "ensemble": s_held >= t_ens,
    }
    results = {}
    for name, flag, score in [
        ("rules_only", flags["rules_only"], held_f["rule_score"]),
        ("xgboost", flags["xgboost"], p_held),
        ("ensemble", flags["ensemble"], s_held),
    ]:
        results[name] = {
            **binary_metrics(held_f["is_fraud"], flag),
            "pr_auc": pr_auc(held_f["is_fraud"], score),
        }
    for name, m in results.items():
        print(format_report(f"[held-out] {name}", m))
    print(f"\nLatency: {latency_ms:.4f} ms/transaction (batch, in-memory; "
          f"production streaming would add I/O per event)")

    # --- Case files for flagged held-out transactions ------------------------
    flagged = held_f[flags["ensemble"]].copy()
    flagged["_p"] = p_held[flags["ensemble"]]
    flagged["_s"] = s_held[flags["ensemble"]]
    n_write = min(len(flagged), args.max_cases)
    cases = [
        make_case(row, row["_p"], row["_s"], t_ens)
        for _, row in flagged.head(n_write).iterrows()
    ]
    if args.fresh_db and CASES_DB.exists():
        CASES_DB.unlink()
    conn = connect(CASES_DB)
    insert_cases(conn, cases)
    total = count_cases(conn, "spike_sentinel")
    print(f"\nWrote {len(cases)} cases ({n_write} of {len(flagged)} flagged, "
          f"cap {args.max_cases}); audit log now holds {total} spike_sentinel cases")
    if cases:
        c = cases[0]
        print(f"\nExample case {c.case_id[:8]} [{c.decision}] "
              f"confidence={c.confidence:.3f}:\n  {c.reasoning_text}")

    # --- Save results --------------------------------------------------------
    importances = sorted(
        zip(FEATURE_COLS, model.feature_importances_.round(4).tolist()),
        key=lambda kv: -kv[1],
    )[:8]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "agent": "spike_sentinel",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data": {
            "n_fit": len(fit_df), "n_valid": len(valid_df), "n_heldout": len(held_f),
            "fraud_rate_heldout": round(float(held_f["is_fraud"].mean()), 5),
            "users_limited_to": args.users or None,
        },
        "thresholds": {"model": round(t_model, 4), "ensemble": round(t_ens, 4)},
        "results": results,
        "rule_diagnostics_validation": rule_diagnostics(valid_df),
        "latency_ms_per_tx": round(latency_ms, 6),
        "top_features": importances,
        "cases_written": len(cases),
        "decisions": {
            "block": int(sum(c.decision == "block" for c in cases)),
            "escalate": int(sum(c.decision == "escalate" for c in cases)),
        },
    }
    out_path = RESULTS_DIR / "spike_sentinel_metrics.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
