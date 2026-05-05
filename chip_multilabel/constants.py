"""Constant definitions for chip multi-label evaluation.

11 evaluation classes:
- 4 single defects (training set, sigmoid-active class slot)
- 5 combo (2-defect overlay; scratch+scratch_rot excluded — same defect family)
- Normal (no defect active)
- Invalid (heuristic-detected: white + orange border)

The model is trained on TRAIN_CLASSES only (4-way single-label CE/Focal/ASL/BCE).
"""
from __future__ import annotations

from typing import Tuple

TRAIN_CLASSES: Tuple[str, ...] = (
    "bank_boundary",
    "fork",
    "scratch",
    "scratch_rot",
)

SINGLE_KEYS: Tuple[str, ...] = TRAIN_CLASSES

COMBO_KEYS: Tuple[str, ...] = (
    "bank_boundary+fork",
    "bank_boundary+scratch",
    "bank_boundary+scratch_rot",
    "fork+scratch",
    "fork+scratch_rot",
)

SPECIAL_KEYS: Tuple[str, ...] = ("Normal", "Invalid")

ALL_CLASS_KEYS: Tuple[str, ...] = SINGLE_KEYS + COMBO_KEYS + SPECIAL_KEYS

NUM_TRAIN = len(TRAIN_CLASSES)
NUM_ALL = len(ALL_CLASS_KEYS)

INFERENCE_VARIANTS: Tuple[str, ...] = (
    "I0",  # argmax (single-only baseline)
    "I1",  # softmax + per-class F1-max threshold
    "I2",  # sigmoid + fixed 0.5 threshold
    "I3",  # sigmoid + per-class F1-max threshold
    "I4",  # TS + sigmoid + per-class F1-max threshold
    # I5 (TTA stack) PERMANENTLY DROPPED: see notes.md "Hard Rules"
    # iter1 measured -0.018 macro_f1 vs I4. chip patterns rotation-sensitive.
    "I6",  # sigmoid + F1-max threshold + floor 0.3 (anti-degenerate)
    "I7",  # sigmoid + joint macro-F1 coord descent threshold (vs per-class binary F1)
    "I8",  # sigmoid + top-2 margin gating (combo only when 2nd >= margin*top1)
    "I9",  # per-class T (vs scalar T) + sigmoid + F1-max threshold
    "I10",  # I7 + softmax-entropy Normal short-circuit (high entropy -> Normal)
)
TRAIN_VARIANTS: Tuple[str, ...] = ("T0", "T1", "T2", "T3", "T4", "T5", "T6")

DEFAULT_BACKBONE_CKPT = (
    "D:/project/known-cnn/outputs/logs_chip/"
    "chip5_round4_v14_260505_061558_running/best_model.pth"
)
DEFAULT_CLASSIFICATION_CHIPS = "D:/project/data/wm-811k/classification_chips"


def sort_label_pair(a: str, b: str) -> str:
    """Canonical 'a+b' combo key — alphabetic order."""
    if a == b:
        raise ValueError(f"combo of identical class: {a}")
    lo, hi = sorted([a, b])
    return f"{lo}+{hi}"


def _build_key_to_labels() -> dict[str, frozenset[str]]:
    out: dict[str, frozenset[str]] = {}
    for k in SINGLE_KEYS:
        out[k] = frozenset({k})
    for k in COMBO_KEYS:
        out[k] = frozenset(k.split("+"))
    out["Normal"] = frozenset()
    out["Invalid"] = frozenset({"__INVALID__"})
    return out


KEY_TO_LABELS: dict[str, frozenset[str]] = _build_key_to_labels()
LABELS_TO_KEY: dict[frozenset[str], str] = {v: k for k, v in KEY_TO_LABELS.items()}


def labels_to_class_key(active: frozenset[str], is_invalid: bool = False) -> str:
    """Map (active defect set, invalid flag) -> 11-class key.

    - is_invalid True -> 'Invalid'
    - active empty   -> 'Normal'
    - active size 1  -> single key
    - active size 2  -> combo key (canonical sort)
    - active size >=3 -> caller must truncate first; raises here
    """
    if is_invalid:
        return "Invalid"
    if len(active) == 0:
        return "Normal"
    if len(active) == 1:
        (k,) = active
        return k
    if len(active) == 2:
        a, b = sorted(active)
        key = f"{a}+{b}"
        if key not in COMBO_KEYS:
            raise KeyError(f"unknown combo {key}; excluded combo not allowed")
        return key
    raise ValueError(f"active set size {len(active)} > 2 — caller must truncate first")
