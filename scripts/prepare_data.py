"""Freeze the time-based train/held-out split for the Spike Sentinel.

Kaggle's fraudTrain/fraudTest split for kartik2112/fraud-detection is already
chronological with disjoint time ranges (every train transaction precedes every
test transaction), so we adopt it as this project's time-based split instead
of re-splitting 1.85M rows. This script verifies that assumption — it fails
loudly if the ranges overlap — then freezes parquet copies in data/processed/
along with a split_summary.json.

The held-out set is frozen after this script runs. Tuning (thresholds, rules,
hyperparameters) must use only the train period; Spike Sentinel carves its
validation slice from train, never from held-out.

Usage:
    .venv/bin/python scripts/prepare_data.py
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "sparkov"
OUT_DIR = ROOT / "data" / "processed"


def summarize(df: pd.DataFrame, name: str) -> dict:
    return {
        "rows": int(len(df)),
        "fraud_count": int(df["is_fraud"].sum()),
        "fraud_rate": round(float(df["is_fraud"].mean()), 5),
        "start": str(df["trans_date_trans_time"].min()),
        "end": str(df["trans_date_trans_time"].max()),
        "n_cards": int(df["cc_num"].nunique()),
        "n_merchants": int(df["merchant"].nunique()),
        "mean_amount": round(float(df["amt"].mean()), 2),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(RAW_DIR / "fraudTrain.csv", index_col=0)
    test = pd.read_csv(RAW_DIR / "fraudTest.csv", index_col=0)
    train["trans_date_trans_time"] = pd.to_datetime(train["trans_date_trans_time"])
    test["trans_date_trans_time"] = pd.to_datetime(test["trans_date_trans_time"])

    # Verify the chronology assumption that justifies adopting Kaggle's split.
    train_end = train["trans_date_trans_time"].max()
    test_start = test["trans_date_trans_time"].min()
    assert train_end < test_start, (
        f"Split is NOT chronological: train ends {train_end}, "
        f"test starts {test_start} — must re-split manually."
    )

    train.to_parquet(OUT_DIR / "spike_train.parquet", index=False)
    test.to_parquet(OUT_DIR / "spike_heldout.parquet", index=False)

    summary = {
        "dataset": "kartik2112/fraud-detection (Sparkov)",
        "split_strategy": "Kaggle-provided split adopted as time-based split "
        "(verified chronological, disjoint ranges)",
        "train": summarize(train, "train"),
        "heldout": summarize(test, "heldout"),
        "frozen_at": str(pd.Timestamp.now()),
    }
    with open(OUT_DIR / "split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT_DIR/'spike_train.parquet'} and "
          f"{OUT_DIR/'spike_heldout.parquet'} — held-out is now frozen.")


if __name__ == "__main__":
    main()
