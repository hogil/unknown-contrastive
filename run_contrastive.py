#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_contrastive.py — Windows env wrapper for contrastive.py.

contrastive.py 자체 수정 안 함. CFG override + cnn_train backbone state_dict 추출만.

사용:
    python run_contrastive.py                              # 기본 smoke (EPOCHS=2)
    EPOCHS=20 BATCH=32 python run_contrastive.py           # 본 학습
    BACKBONE_CKPT=logs_wafer/<run>/best_model.pth python run_contrastive.py
"""
from __future__ import annotations
import os
import sys
import tempfile
from pathlib import Path
import torch


def extract_state_dict(ckpt_path: str | Path) -> str:
    """cnn_train save format ({"model": state_dict, "classes": [...], ...}) 에서
    state_dict 만 추출해서 임시 .pth 로 저장. contrastive.py 가 직접 load 가능한 raw state.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"backbone ckpt not found: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        # try common wrap keys (cnn_train: "model"; some others: "state_dict" / "model_state")
        for k in ("model", "model_state", "state_dict"):
            if k in sd and isinstance(sd[k], dict):
                state = sd[k]
                break
        else:
            state = sd  # assume already raw state_dict
    else:
        state = sd
    out = Path(tempfile.gettempdir()) / f"contrastive_init_{ckpt_path.stem}.pth"
    torch.save(state, out)
    print(f"[init] extracted state_dict: {ckpt_path} → {out}  ({len(state)} keys)")
    return str(out)


def main():
    # Resolve backbone init (TAPT — wafer best_model.pth)
    backbone_ckpt = os.environ.get(
        "BACKBONE_CKPT",
        "D:/project/unknown-contrastive/logs_wafer/overall/best_model.pth",
    )
    backbone_state_path = extract_state_dict(backbone_ckpt)

    # Import contrastive (executes top-level CFG dict)
    sys.path.insert(0, str(Path(__file__).parent))
    import contrastive  # noqa: E402

    # Monkey-patch ImageFolder:
    # 1. classification/classification_chips 빈 폴더 제외 (cnn_train EXCLUDE_CLASSES 동일 정책)
    # 2. PER_CLASS_CAP 으로 각 class 의 sample 수 제한 (env var PER_CLASS_CAP)
    from torchvision.datasets import ImageFolder as _BaseImageFolder
    EXCLUDE = {"classification", "classification_chips"}
    PER_CLASS_CAP = int(os.environ.get("PER_CLASS_CAP", 0)) or None

    class FilteredImageFolder(_BaseImageFolder):
        def find_classes(self, directory):
            classes, _ = super().find_classes(directory)
            classes = [c for c in classes if c not in EXCLUDE]
            class_to_idx = {c: i for i, c in enumerate(classes)}
            return classes, class_to_idx

        def __init__(self, root, **kwargs):
            super().__init__(root, **kwargs)
            if PER_CLASS_CAP:
                from collections import defaultdict
                per_cls = defaultdict(list)
                for path, lbl in self.samples:
                    per_cls[lbl].append((path, lbl))
                capped = []
                for cls_idx, lst in sorted(per_cls.items()):
                    lst.sort(key=lambda x: x[0])  # deterministic
                    capped.extend(lst[:PER_CLASS_CAP])
                self.samples = capped
                self.targets = [lbl for _, lbl in capped]
                self.imgs = self.samples
                print(f"[CappedImageFolder] {root}: capped to {PER_CLASS_CAP} per class "
                      f"→ total {len(capped)} samples")
    contrastive.ImageFolder = FilteredImageFolder
    cap_msg = f"+ cap {PER_CLASS_CAP}/class" if PER_CLASS_CAP else "(no cap)"
    print(f"[patch] ImageFolder filter active: exclude {EXCLUDE}  {cap_msg}")

    # 데이터 디렉토리 — 원본 D:/project/data/wm-811k/unknown 절대 안 건드림.
    # 미리 _make_contrastive_cache.py 로 만든 384×384 RGB cache 사용 (disk 로드 100× 빠름).
    data_dir = os.environ.get("DATA_DIR", "D:/project/data/wm-811k/unknown_cache_384")

    # CFG override for Windows + current synthetic data
    contrastive.CFG.update({
        "TRAIN_DIR":  data_dir,
        "UNKNOWN_DIR": data_dir,
        "OVERLAY_DIR": data_dir,                                # no separate overlay
        "OUTPUT_DIR":  "D:/project/unknown-contrastive/outputs_contrastive",
        "LOCAL_BACKBONE_WEIGHTS": backbone_state_path,
        "IMAGE_SIZE": int(os.environ.get("IMAGE_SIZE", 384)),
        "BATCH":       int(os.environ.get("BATCH",       32)),  # 16GB GPU 안전
        "NUM_WORKERS": int(os.environ.get("NUM_WORKERS",  0)),  # Windows 기본 0
        "EPOCHS":      int(os.environ.get("EPOCHS",       2)),  # smoke default
        "WARMUP_EPOCHS": int(os.environ.get("WARMUP_EPOCHS", 1)),
        "TRAIN_SAMPLING_RATIO": float(os.environ.get("TRAIN_SAMPLING_RATIO", 0.25)),
        "USE_LOCAL":   os.environ.get("USE_LOCAL", "true").lower() == "true",
        "USE_QUEUE":   os.environ.get("USE_QUEUE", "true").lower() == "true",
        "CLUSTER_SELECTION_METHOD": os.environ.get("CLUSTER_SELECTION_METHOD", "leaf"),
        "CLUSTER_SELECTION_EPSILON": float(os.environ.get("CLUSTER_SELECTION_EPSILON", 0.06)),
        "MIN_CLUSTER_SIZE": int(os.environ.get("MIN_CLUSTER_SIZE", 12)),
        "MIN_SAMPLES": int(os.environ.get("MIN_SAMPLES", 4)),
        "PIN_MEMORY": False,
        "PERSISTENT": False,
    })

    print("[CFG override] paths + Windows defaults applied")
    print(f"  TRAIN_DIR={contrastive.CFG['TRAIN_DIR']}")
    print(f"  UNKNOWN_DIR={contrastive.CFG['UNKNOWN_DIR']}")
    print(f"  OUTPUT_DIR={contrastive.CFG['OUTPUT_DIR']}")
    print(f"  LOCAL_BACKBONE_WEIGHTS={contrastive.CFG['LOCAL_BACKBONE_WEIGHTS']}")
    print(f"  IMAGE_SIZE={contrastive.CFG['IMAGE_SIZE']}, BATCH={contrastive.CFG['BATCH']}, "
          f"NUM_WORKERS={contrastive.CFG['NUM_WORKERS']}, EPOCHS={contrastive.CFG['EPOCHS']}")

    contrastive.main()


if __name__ == "__main__":
    main()
