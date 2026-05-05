"""Load chip backbone checkpoint, drop invalid_main slot for 4-way inference."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import timm
import torch

from .constants import TRAIN_CLASSES


def load_chip_backbone(ckpt_path: str | Path, device: torch.device) -> Tuple[torch.nn.Module, Dict, List[int]]:
    """Load chip backbone, return (model, ckpt_meta, keep_indices_in_5).

    keep_indices: positional indices into the original 5-class logits that
    correspond to TRAIN_CLASSES (bank_boundary, fork, scratch, scratch_rot).
    Use these to drop invalid_main column before applying any inference variant.
    """
    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f"checkpoint not found: {p}")
    ckpt = torch.load(p, map_location="cpu", weights_only=False)

    classes_full: List[str] = ckpt["classes"]
    backbone: str = ckpt["backbone"]
    img_size: int = int(ckpt["img_size"])
    num_classes_full = len(classes_full)

    keep_indices = [classes_full.index(c) for c in TRAIN_CLASSES]
    if len(keep_indices) != len(TRAIN_CLASSES):
        raise RuntimeError(f"could not map TRAIN_CLASSES into ckpt classes={classes_full}")

    model = timm.create_model(backbone, pretrained=False, num_classes=num_classes_full)
    sd = ckpt["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        raise RuntimeError(f"missing keys when loading chip backbone: {missing}")
    model.eval().to(device)

    meta = {
        "ckpt_path": str(p),
        "classes_full": classes_full,
        "backbone": backbone,
        "img_size": img_size,
        "val_macro_f1": float(ckpt.get("val_macro_f1", 0.0)),
        "epoch": int(ckpt.get("epoch", -1)),
    }
    return model, meta, keep_indices


def select_train_logits(logits_full: np.ndarray, keep_indices: List[int]) -> np.ndarray:
    """logits_full: (N, num_classes_full) -> (N, len(TRAIN_CLASSES))."""
    return logits_full[:, keep_indices]
