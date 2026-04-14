import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def find_best_threshold(y_true, y_probs, metric='f1', min_precision=0.5):
    """
    Find best threshold based on F1 or recall.
    For exfiltration detection, use metric='recall' to minimize missed attacks.
    """
    thresholds = np.linspace(0.1, 0.9, 50)
    best_score = 0
    best_t = 0.5

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        prec = precision_score(y_true, preds, zero_division=0)
        rec  = recall_score(y_true, preds, zero_division=0)
        f1   = f1_score(y_true, preds, zero_division=0)

        if metric == 'recall':
            score = rec if prec >= min_precision else 0.0
        else:
            score = f1

        if score > best_score:
            best_score = score
            best_t = t

    return best_t


def compute_metrics(y_true, y_pred_probs, threshold=0.5):
    """
    Compute classification metrics.
    Always pass threshold explicitly — find it on validation set,
    NOT on the same data you're evaluating (avoids data leakage).
    """
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)

    # Input validation
    assert len(y_true) == len(y_pred_probs), \
        "Length mismatch between labels and predictions"
    assert set(np.unique(y_true)).issubset({0, 1}), \
        "y_true must be binary (0 or 1)"

    y_pred = (y_pred_probs >= threshold).astype(int)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_pred_probs)
    except ValueError as e:
        print(f"[Warning] AUC could not be computed: {e}")
        auc = 0.0

    # labels=[0,1] ensures 2x2 matrix even if one class missing
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy":  acc,
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "auc":       auc,
        "threshold": threshold,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }