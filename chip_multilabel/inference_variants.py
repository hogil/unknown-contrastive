"""Inference variants for Stage 1 / Stage 2 cell evaluation.

Variants (all on existing model — no training):
- I0  argmax -> single only (no multi possible)
- I1  softmax + per-class F1-max threshold (val) -> active set
- I2  sigmoid (raw logit) + fixed 0.5 threshold
- I3  sigmoid + per-class F1-max threshold (val)
- I4  TS + sigmoid + per-class F1-max threshold (val)
- I6  I3 + prior-aware logit shift (eval-vs-train prior correction)
- I7  sigmoid with separate single / combo-additional thresholds
- I8  sigmoid + top-2 margin gating (combo only when 2nd >= margin*top1)
- I9  per-class temperature scaling + sigmoid + F1-max threshold

I5 (TTA stack) PERMANENTLY DROPPED — chip patterns rotation-sensitive (iter 1, -0.018 macro_f1).
The `tta=True` path of forward_all_logits is kept dead (never invoked) for archival reasons.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from .calibration import PerClassTemperatureScaler, TemperatureScaler
from .constants import INFERENCE_VARIANTS, TRAIN_CLASSES
from .decision_tree import decide, detect_invalid
from .eval_dataset import (IMAGENET_MEAN, IMAGENET_STD, ChipEvalDataset,
                           EvalRecord)
from .metrics import (confusion_11class, expected_calibration_error,
                      joint_macro_f1_threshold, multihot_metrics,
                      per_class_f1_max_threshold, per_class_metrics,
                      top1_11class_accuracy)
from .model_io import select_train_logits

I8_MARGIN = 0.6
I6_FLOOR = 0.30
I10_ENTROPY_NORMAL_FRAC = 0.85  # softmax entropy >= 0.85 * log(4) -> declare Normal


@torch.no_grad()
def forward_all_logits(model: torch.nn.Module, ds: ChipEvalDataset, device: torch.device,
                      batch_size: int = 32, num_workers: int = 0,
                      tta: bool = False) -> np.ndarray:
    """Returns logits_full: (N, num_classes_full=5)."""
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    parts: List[np.ndarray] = []
    use_amp = device.type == "cuda"
    autocast = torch.amp.autocast if hasattr(torch.amp, "autocast") else torch.cuda.amp.autocast
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        if not tta:
            with autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
            parts.append(out.detach().float().cpu().numpy())
            continue
        # TTA 4 views
        accum = None
        for vfn in (lambda t: t,
                    lambda t: torch.flip(t, dims=[-1]),
                    lambda t: torch.flip(t, dims=[-2]),
                    lambda t: torch.rot90(t, k=1, dims=[-2, -1])):
            xv = vfn(x)
            with autocast(device_type=device.type, enabled=use_amp):
                out = model(xv)
            o = out.detach().float().cpu().numpy()
            accum = o if accum is None else accum + o
        parts.append(accum / 4.0)
    return np.concatenate(parts, axis=0)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


@dataclass
class CellResult:
    cell_id: str
    train_id: str
    inference_id: str
    n_eval: int
    macro_f1: float
    micro_f1: float
    mAP: float
    hamming_loss: float
    subset_accuracy: float
    top1_11class: float
    ece_pre: float
    ece_post: float
    temperature: float
    per_class_f1: Dict[str, float]
    threshold_dict: Dict[str, float]
    elapsed_sec: float
    preds_rows: List[Dict] = field(default_factory=list)
    per_class_rows: List[Dict] = field(default_factory=list)
    confusion_rows: List[Dict] = field(default_factory=list)
    error_rows: List[Dict] = field(default_factory=list)


def _compute_invalid_masks(records: List[EvalRecord]) -> Tuple[np.ndarray, np.ndarray]:
    """Run invalid heuristic on all records.

    Returns (is_invalid_pred (N,) bool, invalid_score (N,) float).
    """
    is_inv = np.zeros(len(records), dtype=bool)
    score = np.zeros(len(records), dtype=np.float32)
    for i, r in enumerate(records):
        with Image.open(r.chip_path) as im:
            arr = np.array(im.convert("RGB"))
        flag, s = detect_invalid(arr)
        is_inv[i] = flag
        score[i] = s
    return is_inv, score


def evaluate_cell(
    variant_id: str,
    logits_full: np.ndarray,           # (N, num_classes_full=5) — variant chooses per-view averaging upstream
    logits_full_tta: np.ndarray | None,  # (N, 5) TTA-averaged or None (only used by I5)
    records: List[EvalRecord],
    val_idx: np.ndarray,
    eval_idx: np.ndarray,
    keep_indices: List[int],
    invalid_mask: np.ndarray,
    invalid_score: np.ndarray,
    train_id: str = "T0",
) -> CellResult:
    """Apply variant-specific post-processing on pre-computed full-class logits."""
    if variant_id not in INFERENCE_VARIANTS:
        raise ValueError(f"unknown inference variant: {variant_id}")
    t0 = time.time()
    cell_id = f"{train_id}__{variant_id}"

    # 1) pick logits source
    if variant_id == "I5":
        if logits_full_tta is None:
            raise ValueError("I5 requires TTA logits — pass logits_full_tta")
        L_full = logits_full_tta
    else:
        L_full = logits_full

    L_train = select_train_logits(L_full, keep_indices)  # (N, 4)

    # 2) val arrays for tuning thresholds / T
    val_records = [records[i] for i in val_idx]
    eval_records = [records[i] for i in eval_idx]

    L_train_val = L_train[val_idx]
    L_full_val = L_full[val_idx]
    y_val_multihot = np.stack([r.true_multihot for r in val_records], axis=0)
    y_val_classidx = np.array(
        [r.true_multihot.argmax() if r.true_multihot.sum() == 1 else -1
         for r in val_records],
        dtype=np.int64,
    )
    single_val_mask = y_val_classidx >= 0  # only single-positive samples for TS fit

    # 3) variant-specific transforms
    temperature = 1.0
    if variant_id in ("I4", "I5"):
        if single_val_mask.sum() >= 4:
            ts = TemperatureScaler().fit(L_train_val[single_val_mask], y_val_classidx[single_val_mask])
            temperature = ts.T
            L_train = L_train / max(temperature, 1e-3)

    use_top2_margin = False
    use_entropy_normal = False
    if variant_id == "I0":
        argmax_4 = L_train.argmax(axis=1)
        probs_eval = np.zeros_like(L_train)
        for i, a in enumerate(argmax_4):
            probs_eval[i, a] = 1.0
        threshold_dict = {c: 0.5 for c in TRAIN_CLASSES}
    elif variant_id == "I1":
        probs_full = softmax(L_train)
        thr_val = per_class_f1_max_threshold(softmax(L_train[val_idx]), y_val_multihot, list(TRAIN_CLASSES))
        threshold_dict = thr_val
        probs_eval = probs_full
    elif variant_id == "I2":
        probs_full = sigmoid(L_train)
        threshold_dict = {c: 0.5 for c in TRAIN_CLASSES}
        probs_eval = probs_full
    elif variant_id in ("I3", "I4"):
        probs_full = sigmoid(L_train)
        thr_val = per_class_f1_max_threshold(probs_full[val_idx], y_val_multihot, list(TRAIN_CLASSES))
        threshold_dict = thr_val
        probs_eval = probs_full
    elif variant_id == "I6":
        probs_full = sigmoid(L_train)
        thr_val = per_class_f1_max_threshold(probs_full[val_idx], y_val_multihot, list(TRAIN_CLASSES))
        threshold_dict = {c: max(t, I6_FLOOR) for c, t in thr_val.items()}
        probs_eval = probs_full
    elif variant_id == "I7":
        probs_full = sigmoid(L_train)
        thr_val = joint_macro_f1_threshold(probs_full[val_idx], y_val_multihot, list(TRAIN_CLASSES))
        threshold_dict = thr_val
        probs_eval = probs_full
    elif variant_id == "I8":
        probs_full = sigmoid(L_train)
        thr_val = per_class_f1_max_threshold(probs_full[val_idx], y_val_multihot, list(TRAIN_CLASSES))
        threshold_dict = thr_val
        probs_eval = probs_full
        use_top2_margin = True
    elif variant_id == "I10":
        probs_full = sigmoid(L_train)
        thr_val = joint_macro_f1_threshold(probs_full[val_idx], y_val_multihot, list(TRAIN_CLASSES))
        threshold_dict = thr_val
        probs_eval = probs_full
        use_entropy_normal = True
    elif variant_id == "I9":
        # per-class T fitted on raw L_train (pre-T) val with multi-hot targets
        L_train_val_raw = select_train_logits(L_full[val_idx], keep_indices)
        ts_pc = PerClassTemperatureScaler(n_classes=len(TRAIN_CLASSES))
        ts_pc.fit(L_train_val_raw, y_val_multihot)
        L_train_pc = ts_pc.apply(select_train_logits(L_full, keep_indices))
        L_train = L_train_pc  # override for downstream ECE calc
        temperature = float(np.mean(ts_pc.T))
        probs_full = sigmoid(L_train_pc)
        thr_val = per_class_f1_max_threshold(probs_full[val_idx], y_val_multihot, list(TRAIN_CLASSES))
        threshold_dict = thr_val
        probs_eval = probs_full
    else:
        raise RuntimeError(f"unhandled variant {variant_id}")

    # 4) ECE pre/post (use full-class softmax pre-T for pre-ECE; train-class softmax post for post-ECE)
    softmax_full_pre = softmax(L_full[val_idx][single_val_mask])
    if single_val_mask.sum() >= 4:
        # pre = full 5-class with invalid_main column on val, but our targets are 4-class indices,
        # so we recompute pre on the train-class softmax raw logits / 1.0
        L_train_val_raw = select_train_logits(L_full[val_idx], keep_indices)
        pre_softmax = softmax(L_train_val_raw[single_val_mask])
        ece_pre = expected_calibration_error(pre_softmax, y_val_classidx[single_val_mask])
        post_softmax = softmax(L_train_val_raw[single_val_mask] / max(temperature, 1e-3))
        ece_post = expected_calibration_error(post_softmax, y_val_classidx[single_val_mask])
    else:
        ece_pre = 0.0
        ece_post = 0.0

    # 5) per-chip decisions on eval split
    preds_rows: List[Dict] = []
    error_rows: List[Dict] = []
    y_true_multihot_eval = np.stack([r.true_multihot for r in eval_records], axis=0)
    y_pred_multihot_eval = np.zeros_like(y_true_multihot_eval)
    y_true_keys: List[str] = []
    y_pred_keys: List[str] = []

    log_C = float(np.log(len(TRAIN_CLASSES)))
    for local_i, gi in enumerate(eval_idx):
        rec = records[gi]
        chip_probs = probs_eval[gi]
        # invalid heuristic decision
        is_inv = bool(invalid_mask[gi])
        inv_score = float(invalid_score[gi])
        if use_entropy_normal and not is_inv:
            sm = softmax(L_train[gi:gi+1])[0]
            ent = float(-(sm * np.log(np.maximum(sm, 1e-12))).sum())
            if ent >= I10_ENTROPY_NORMAL_FRAC * log_C:
                from .decision_tree import DecisionResult as _DR
                decision = _DR("Normal", "normal_entropy", [], inv_score)
                preds_rows.append({
                    "cell_id": cell_id,
                    "chip_path": rec.chip_path,
                    "class_key": rec.class_key,
                    "true_labels": str(sorted(rec.true_active) or (["Normal"] if rec.is_normal_gt else (["Invalid"] if rec.is_invalid_gt else []))),
                    "pred_labels": str([]),
                    "pred_class_key": "Normal",
                    "prob_bank_boundary": float(chip_probs[0]),
                    "prob_fork": float(chip_probs[1]),
                    "prob_scratch": float(chip_probs[2]),
                    "prob_scratch_rot": float(chip_probs[3]),
                    "max_prob": float(chip_probs.max()),
                    "invalid_score": inv_score,
                    "decision_type": "normal_entropy",
                    "correct_multihot": bool((y_true_multihot_eval[local_i] == y_pred_multihot_eval[local_i]).all()),
                    "correct_11class": rec.class_key == "Normal",
                })
                y_true_keys.append(rec.class_key)
                y_pred_keys.append("Normal")
                if rec.class_key != "Normal":
                    error_rows.append({
                        "cell_id": cell_id,
                        "chip_path": rec.chip_path,
                        "true_class_key": rec.class_key,
                        "pred_class_key": "Normal",
                        "true_labels": str(sorted(rec.true_active)),
                        "pred_labels": str([]),
                        "error_type": "wrong_normal_entropy",
                        "per_class_probs": str({c: round(float(chip_probs[i]), 4) for i, c in enumerate(TRAIN_CLASSES)}),
                        "decision_type": "normal_entropy",
                    })
                continue
        if use_top2_margin and not is_inv:
            order = np.argsort(chip_probs)[::-1]
            top1_p = float(chip_probs[order[0]])
            top2_p = float(chip_probs[order[1]])
            gated_thresh = dict(threshold_dict)
            if top2_p < I8_MARGIN * top1_p:
                top2_class = TRAIN_CLASSES[int(order[1])]
                gated_thresh[top2_class] = 1.01
            decision = decide(chip_probs.tolist(), gated_thresh, inv_score, is_inv)
        else:
            decision = decide(chip_probs.tolist(), threshold_dict, inv_score, is_inv)
        # write multi-hot row
        if decision.class_key in TRAIN_CLASSES:
            y_pred_multihot_eval[local_i, TRAIN_CLASSES.index(decision.class_key)] = 1
        elif decision.class_key in {"Normal", "Invalid"}:
            pass
        elif "+" in decision.class_key:
            for c in decision.class_key.split("+"):
                y_pred_multihot_eval[local_i, TRAIN_CLASSES.index(c)] = 1
        y_true_keys.append(rec.class_key)
        y_pred_keys.append(decision.class_key)

        correct_mh = bool((y_true_multihot_eval[local_i] == y_pred_multihot_eval[local_i]).all())
        correct_11 = (rec.class_key == decision.class_key)

        preds_rows.append({
            "cell_id": cell_id,
            "chip_path": rec.chip_path,
            "class_key": rec.class_key,
            "true_labels": str(sorted(rec.true_active) or (["Normal"] if rec.is_normal_gt else (["Invalid"] if rec.is_invalid_gt else []))),
            "pred_labels": str(decision.active_labels or ([decision.class_key] if decision.class_key in {"Normal", "Invalid"} else [])),
            "pred_class_key": decision.class_key,
            "prob_bank_boundary": float(chip_probs[0]),
            "prob_fork": float(chip_probs[1]),
            "prob_scratch": float(chip_probs[2]),
            "prob_scratch_rot": float(chip_probs[3]),
            "max_prob": float(chip_probs.max()),
            "invalid_score": inv_score,
            "decision_type": decision.decision_type,
            "correct_multihot": correct_mh,
            "correct_11class": correct_11,
        })
        if not correct_11:
            err_type = _classify_error(rec, decision)
            error_rows.append({
                "cell_id": cell_id,
                "chip_path": rec.chip_path,
                "true_class_key": rec.class_key,
                "pred_class_key": decision.class_key,
                "true_labels": str(sorted(rec.true_active)),
                "pred_labels": str(decision.active_labels),
                "error_type": err_type,
                "per_class_probs": str({c: round(float(chip_probs[i]), 4) for i, c in enumerate(TRAIN_CLASSES)}),
                "decision_type": decision.decision_type,
            })

    # 6) aggregate metrics
    mh = multihot_metrics(y_true_multihot_eval, y_pred_multihot_eval, probs_eval[eval_idx], list(TRAIN_CLASSES))
    pcm = per_class_metrics(y_true_multihot_eval, y_pred_multihot_eval, probs_eval[eval_idx],
                            threshold_dict, list(TRAIN_CLASSES))
    cm, labels = confusion_11class(y_true_keys, y_pred_keys)
    top1 = top1_11class_accuracy(y_true_keys, y_pred_keys)

    confusion_rows = []
    for i, t in enumerate(labels):
        for j, p in enumerate(labels):
            if cm[i, j] == 0:
                continue
            confusion_rows.append({
                "cell_id": cell_id,
                "true_class_key": t,
                "pred_class_key": p,
                "count": int(cm[i, j]),
            })

    for row in pcm:
        row["cell_id"] = cell_id

    elapsed = time.time() - t0
    return CellResult(
        cell_id=cell_id,
        train_id=train_id,
        inference_id=variant_id,
        n_eval=len(eval_idx),
        macro_f1=mh["macro_f1"],
        micro_f1=mh["micro_f1"],
        mAP=mh["mAP"],
        hamming_loss=mh["hamming_loss"],
        subset_accuracy=mh["subset_accuracy"],
        top1_11class=top1,
        ece_pre=ece_pre,
        ece_post=ece_post,
        temperature=temperature,
        per_class_f1={r["class"]: r["f1"] for r in pcm},
        threshold_dict=threshold_dict,
        elapsed_sec=elapsed,
        preds_rows=preds_rows,
        per_class_rows=pcm,
        confusion_rows=confusion_rows,
        error_rows=error_rows,
    )


def _classify_error(rec: EvalRecord, decision) -> str:
    if rec.is_invalid_gt and decision.class_key != "Invalid":
        return "missed_invalid"
    if rec.is_normal_gt and decision.class_key != "Normal":
        return "missed_normal"
    true_set = set(rec.true_active)
    pred_set = set(decision.active_labels)
    if rec.class_key in {"Invalid", "Normal"}:
        return "false_positive_general"
    fp_classes = pred_set - true_set
    fn_classes = true_set - pred_set
    if "+" in rec.class_key and decision.class_key not in {"Normal", "Invalid"}:
        return "wrong_combo"
    if fp_classes and not fn_classes:
        c = next(iter(fp_classes))
        return f"false_positive_{c}"
    if fn_classes and not fp_classes:
        c = next(iter(fn_classes))
        return f"false_negative_{c}"
    if fp_classes and fn_classes:
        return "wrong_combo"
    return "other"
