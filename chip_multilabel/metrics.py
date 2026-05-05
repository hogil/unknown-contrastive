"""Metrics: per-class F1-max threshold sweep, multi-hot metrics, 11-class confusion, ECE."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             hamming_loss, precision_recall_curve)

from .constants import ALL_CLASS_KEYS, TRAIN_CLASSES


def joint_macro_f1_threshold(probs: np.ndarray, y_true_multihot: np.ndarray,
                             class_names: List[str], n_iter: int = 4,
                             n_grid: int = 25) -> Dict[str, float]:
    """Coordinate descent on val macro-F1 directly (vs per-class binary F1).

    For each class in turn, try `n_grid` thresholds in [0.01, 0.99], keep the one
    that maximizes val macro-F1 holding others fixed. Cycle `n_iter` rounds.
    """
    C = probs.shape[1]
    if y_true_multihot.shape != probs.shape:
        raise ValueError(f"shape mismatch: probs={probs.shape} y_true={y_true_multihot.shape}")
    thr = np.full(C, 0.5, dtype=np.float64)
    grid = np.linspace(0.02, 0.98, n_grid)
    for _ in range(n_iter):
        for c in range(C):
            best_thr = thr[c]
            best_f1 = -1.0
            for t in grid:
                thr_t = thr.copy()
                thr_t[c] = float(t)
                pred = (probs >= thr_t[None, :]).astype(int)
                f1 = float(f1_score(y_true_multihot, pred, average="macro", zero_division=0))
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = float(t)
            thr[c] = best_thr
    return {n: float(thr[i]) for i, n in enumerate(class_names)}


def per_class_f1_max_threshold(probs: np.ndarray, y_true_multihot: np.ndarray,
                               class_names: List[str]) -> Dict[str, float]:
    """For each class, find threshold maximizing binary F1.

    probs:           (N, C) sigmoid prob
    y_true_multihot: (N, C) {0,1}
    class_names:     length C
    """
    out: Dict[str, float] = {}
    n_pos_total = int(y_true_multihot.sum())
    if n_pos_total == 0:
        return {c: 0.5 for c in class_names}
    for ci, cname in enumerate(class_names):
        y = y_true_multihot[:, ci]
        p = probs[:, ci]
        if y.sum() == 0 or y.sum() == len(y):
            out[cname] = 0.5
            continue
        prec, rec, thresh = precision_recall_curve(y, p)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
        if len(thresh) == 0:
            out[cname] = 0.5
            continue
        best_idx = int(np.argmax(f1[:-1])) if len(f1) > 1 else 0
        out[cname] = float(thresh[best_idx])
    return out


def multihot_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray,
                     class_names: List[str]) -> Dict[str, float]:
    """Multi-hot 4-dim metric pack.

    y_true, y_pred: (N, C) {0,1}
    probs:          (N, C) for mAP
    """
    out: Dict[str, float] = {}
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["micro_f1"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    out["hamming_loss"] = float(hamming_loss(y_true, y_pred))
    out["subset_accuracy"] = float((y_true == y_pred).all(axis=1).mean())
    if y_true.sum() > 0 and (y_true.sum(axis=0) > 0).all():
        out["mAP"] = float(average_precision_score(y_true, probs, average="macro"))
    else:
        per_ap = []
        for ci in range(y_true.shape[1]):
            if y_true[:, ci].sum() > 0:
                per_ap.append(float(average_precision_score(y_true[:, ci], probs[:, ci])))
        out["mAP"] = float(np.mean(per_ap)) if per_ap else 0.0
    return out


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray,
                      thresholds: Dict[str, float], class_names: List[str]) -> List[Dict]:
    rows: List[Dict] = []
    for ci, cname in enumerate(class_names):
        y = y_true[:, ci]
        p = y_pred[:, ci]
        prob = probs[:, ci]
        tp = int(((y == 1) & (p == 1)).sum())
        fp = int(((y == 0) & (p == 1)).sum())
        fn = int(((y == 1) & (p == 0)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        ap = float(average_precision_score(y, prob)) if y.sum() > 0 else 0.0
        rows.append({
            "class": cname,
            "threshold": float(thresholds.get(cname, 0.5)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(y.sum()),
            "ap": ap,
        })
    return rows


def confusion_11class(y_true: List[str], y_pred: List[str]) -> Tuple[np.ndarray, List[str]]:
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(y_pred)}")
    labels = list(ALL_CLASS_KEYS)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm, labels


def top1_11class_accuracy(y_true: List[str], y_pred: List[str]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(y_pred)}")
    return float(np.mean([t == p for t, p in zip(y_true, y_pred)]))


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    """ECE for multi-class softmax: avg |confidence - accuracy| per bin.

    probs: (N, C) softmax; y_true: (N,) class index.
    """
    if probs.ndim != 2:
        raise ValueError(f"probs must be (N,C), got {probs.shape}")
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float32)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if not in_bin.any():
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = conf[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)
