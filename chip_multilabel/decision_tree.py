"""Decision tree: probs + thresholds + invalid heuristic -> 11-class label.

Pipeline per chip:
    chip image -> invalid heuristic check
        -> True : decision_type='invalid', class_key='Invalid'
        -> False: 4-prob + threshold dict -> active set
                  |active|==0 : decision_type='normal', class_key='Normal'
                  |active|==1 : decision_type='single', class_key=<class>
                  |active|==2 : decision_type='combo', class_key='<a>+<b>' (canonical)
                                if combo not in COMBO_KEYS -> truncate to top-1 (e.g. scratch+scratch_rot)
                  |active|>=3 : decision_type='truncated_3plus', keep top-2 by prob
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Sequence

import numpy as np

from .constants import COMBO_KEYS, TRAIN_CLASSES, labels_to_class_key

ORANGE_BORDER_RGB = np.array([240, 160, 0], dtype=np.int16)
ORANGE_TOLERANCE = 40
WHITE_RGB = np.array([255, 255, 255], dtype=np.int16)
WHITE_TOLERANCE = 10


@dataclass
class DecisionResult:
    class_key: str
    decision_type: str  # 'invalid' | 'normal' | 'single' | 'combo' | 'truncated_3plus'
    active_labels: List[str]
    invalid_score: float


def detect_invalid(img_rgb: np.ndarray, white_ratio_thresh: float = 0.95) -> tuple[bool, float]:
    """Heuristic: large white area + orange border = invalid chip.

    img_rgb: HxWx3 uint8.
    Returns (is_invalid, white_ratio).
    """
    if img_rgb.dtype != np.uint8 or img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"img_rgb must be HxWx3 uint8, got shape={img_rgb.shape} dtype={img_rgb.dtype}")
    diff = np.abs(img_rgb.astype(np.int16) - WHITE_RGB).max(axis=-1)
    white_mask = diff <= WHITE_TOLERANCE
    white_ratio = float(white_mask.mean())
    if white_ratio < white_ratio_thresh:
        return False, white_ratio
    h, w = img_rgb.shape[:2]
    border_thickness = max(1, min(h, w) // 50)
    borders = [
        img_rgb[:border_thickness, :, :].reshape(-1, 3),
        img_rgb[-border_thickness:, :, :].reshape(-1, 3),
        img_rgb[:, :border_thickness, :].reshape(-1, 3),
        img_rgb[:, -border_thickness:, :].reshape(-1, 3),
    ]
    orange_present = []
    for border in borders:
        d = np.abs(border.astype(np.int16) - ORANGE_BORDER_RGB).max(axis=-1)
        orange_present.append(bool((d <= ORANGE_TOLERANCE).any()))
    is_invalid = sum(orange_present) >= 3
    return is_invalid, white_ratio


def probs_to_active(probs: Sequence[float], thresholds: Sequence[float],
                    train_classes: Sequence[str] = TRAIN_CLASSES) -> FrozenSet[str]:
    if len(probs) != len(thresholds) or len(probs) != len(train_classes):
        raise ValueError(
            f"length mismatch: probs={len(probs)} thresh={len(thresholds)} classes={len(train_classes)}"
        )
    active = {c for p, t, c in zip(probs, thresholds, train_classes) if p >= t}
    return frozenset(active)


def decide(
    probs: Sequence[float],
    thresholds: Dict[str, float],
    invalid_score: float,
    is_invalid: bool,
    train_classes: Sequence[str] = TRAIN_CLASSES,
) -> DecisionResult:
    if is_invalid:
        return DecisionResult(
            class_key="Invalid",
            decision_type="invalid",
            active_labels=[],
            invalid_score=invalid_score,
        )
    thresh_arr = [thresholds[c] for c in train_classes]
    active = probs_to_active(probs, thresh_arr, train_classes)

    if len(active) == 0:
        return DecisionResult("Normal", "normal", [], invalid_score)
    if len(active) == 1:
        (k,) = active
        return DecisionResult(k, "single", [k], invalid_score)

    if len(active) >= 3:
        order = np.argsort(probs)[::-1]
        kept_idx = [int(i) for i in order if train_classes[int(i)] in active][:2]
        active = frozenset(train_classes[i] for i in kept_idx)
        decision_type = "truncated_3plus"
    else:
        decision_type = "combo"

    sorted_active = sorted(active)
    combo_key = "+".join(sorted_active)
    if combo_key not in COMBO_KEYS:
        order = np.argsort(probs)[::-1]
        for i in order:
            cname = train_classes[int(i)]
            if cname in active:
                return DecisionResult(cname, "combo_collapsed", [cname], invalid_score)
        raise RuntimeError(f"unreachable: active={active}")
    return DecisionResult(combo_key, decision_type, sorted_active, invalid_score)
