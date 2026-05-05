"""Eval set loader: walks chip_multilabel_eval_<TS>/ and emits typed records."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .constants import (ALL_CLASS_KEYS, COMBO_KEYS, KEY_TO_LABELS, SINGLE_KEYS,
                        TRAIN_CLASSES)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class EvalRecord:
    chip_path: str
    class_key: str
    true_multihot: np.ndarray  # (4,) {0,1}
    is_invalid_gt: bool
    is_normal_gt: bool

    @property
    def true_active(self) -> List[str]:
        return [c for c, v in zip(TRAIN_CLASSES, self.true_multihot) if v == 1]


def _multihot_for_key(class_key: str) -> Tuple[np.ndarray, bool, bool]:
    """Return (multihot 4-d, is_invalid, is_normal) for a class_key in ALL_CLASS_KEYS."""
    if class_key == "Invalid":
        return np.zeros(len(TRAIN_CLASSES), dtype=np.int64), True, False
    if class_key == "Normal":
        return np.zeros(len(TRAIN_CLASSES), dtype=np.int64), False, True
    if class_key in SINGLE_KEYS:
        m = np.zeros(len(TRAIN_CLASSES), dtype=np.int64)
        m[TRAIN_CLASSES.index(class_key)] = 1
        return m, False, False
    if class_key in COMBO_KEYS:
        labels = class_key.split("+")
        m = np.zeros(len(TRAIN_CLASSES), dtype=np.int64)
        for c in labels:
            m[TRAIN_CLASSES.index(c)] = 1
        return m, False, False
    raise KeyError(f"unknown class_key: {class_key}")


def discover_records(eval_root: str | Path) -> List[EvalRecord]:
    root = Path(eval_root)
    if not root.exists():
        raise FileNotFoundError(f"eval root not found: {root}")
    out: List[EvalRecord] = []
    for class_key in ALL_CLASS_KEYS:
        cdir = root / class_key
        if not cdir.exists():
            continue
        m, is_inv, is_norm = _multihot_for_key(class_key)
        for png in sorted(cdir.glob("*.png")):
            out.append(EvalRecord(
                chip_path=str(png),
                class_key=class_key,
                true_multihot=m,
                is_invalid_gt=is_inv,
                is_normal_gt=is_norm,
            ))
    return out


class ChipEvalDataset(Dataset):
    """Returns (img_tensor 3xHxW float32, idx int)."""
    def __init__(self, records: List[EvalRecord], img_size: int = 384):
        self.records = records
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rec = self.records[idx]
        img = Image.open(rec.chip_path).convert("RGB")
        x = self.tf(img)
        return x, idx


def stratified_val_eval_split(records: List[EvalRecord], val_ratio: float = 0.2,
                              seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Stratify by class_key. Returns (val_idx, eval_idx) into records list."""
    rng = np.random.default_rng(seed)
    by_cls: dict[str, List[int]] = {}
    for i, r in enumerate(records):
        by_cls.setdefault(r.class_key, []).append(i)
    val_idx: List[int] = []
    eval_idx: List[int] = []
    for cls, idxs in by_cls.items():
        idxs_arr = np.array(idxs)
        rng.shuffle(idxs_arr)
        n_val = max(1, int(round(len(idxs_arr) * val_ratio)))
        n_val = min(n_val, len(idxs_arr) - 1) if len(idxs_arr) > 1 else 0
        val_idx.extend(idxs_arr[:n_val].tolist())
        eval_idx.extend(idxs_arr[n_val:].tolist())
    return np.array(sorted(val_idx)), np.array(sorted(eval_idx))
