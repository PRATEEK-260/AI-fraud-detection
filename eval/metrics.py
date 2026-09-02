"""Per-agent evaluation metrics.

Reported per agent, never averaged across agents — each agent works on a
different dataset with a different fraud base rate, so cross-agent averages
are meaningless.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def binary_metrics(y_true, y_pred) -> dict:
    """Precision / recall / F1 plus the confusion matrix for one agent."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def pr_auc(y_true, y_score) -> float:
    return round(float(average_precision_score(y_true, y_score)), 4)


def best_f1_threshold(y_true, y_score) -> float:
    """Score threshold that maximizes F1 — used for tuning, on train/validation only."""
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    order = np.argsort(-y_score)
    y_sorted = np.asarray(y_true)[order]
    s_sorted = y_score[order]

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    fn = tp[-1] - tp
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    return float(s_sorted[int(np.argmax(f1))])


def format_report(title: str, m: dict) -> str:
    return (
        f"\n{title}\n"
        f"  precision {m['precision']:.4f}   recall {m['recall']:.4f}   "
        f"f1 {m['f1']:.4f}\n"
        f"  TP {m['tp']:<7} FP {m['fp']:<7} FN {m['fn']:<7} TN {m['tn']}"
    )
