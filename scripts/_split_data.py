#!/usr/bin/env python3
"""Data reorganize — CNN train (ImageFolder) / Contrastive train (flat) / Contrastive eval (ImageFolder).

사용자 정책 (260527):
  "클래스를 보는건 치팅이니까 아예 이미지 생성때부터 따로"

산출 3 폴더 (모두 같은 SOURCE_ROOT 에서 wafer disjoint split):
  CNN_TRAIN_DIR/<class>/*.png         CNN 학습용 (Split A class) — class 보존 (supervised)
  CL_TRAIN_DIR/*.png                  Contrastive 학습용 (Split B class) — flat, class label 숨김
  CL_EVAL_DIR/<class>/*.png           Contrastive eval 용 (Split B class) — class 보존 (metric 계산)

wafer 단위 disjoint:
  CNN_TRAIN ↔ CL_TRAIN ↔ CL_EVAL  완전 disjoint (data leakage 0)
"""
from __future__ import annotations

# ===================================================================
# === CONFIG ===
# ===================================================================
SOURCE_ROOT         = "data/images/unknown"          # 프로젝트 상대 (없으면 generate_data.py 먼저)
EXCLUDE_CLASSES     = {"classification", "classification_chips"}

# Class split YAML — 어떤 class 가 CNN 또는 Contrastive 인지
CNN_ACTIVE_YAML        = "experiments/split_a_cnn_21.yaml"
CONTRASTIVE_ACTIVE_YAML = "experiments/split_b_contrastive_22.yaml"

# 출력 폴더 (프로젝트 상대)
CNN_TRAIN_DIR       = "data/images/cnn_train"           # ImageFolder (class subdir)
CL_TRAIN_DIR        = "data/images/contrastive_train"   # flat (no class subdir)
CL_EVAL_DIR         = "data/images/contrastive_eval"    # ImageFolder (class subdir)

# Contrastive train vs eval split (wafer disjoint)
CL_TRAIN_RATIO      = 0.8                                   # 80% train / 20% eval

# 이미지 처리
COPY_MODE           = "link"        # "link" (symlink, 빠름 + disk 절약) or "copy"
SEED                = 42
DRY_RUN             = False         # True 면 path 만 print
# ===================================================================

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from _common import resolve_path


def main():
    src = resolve_path(SOURCE_ROOT)
    if not src.exists():
        raise SystemExit(
            f"SOURCE_ROOT not found: {src}\n\n"
            f"먼저 wafer 이미지를 생성하거나 복사하세요:\n"
            f"  python scripts/generate_data.py          # 합성 demo 데이터\n"
            f"  또는 직접 SOURCE_ROOT 폴더에 <class>/*.png 배치\n"
        )

    cnn_classes = set(yaml.safe_load(open(resolve_path(CNN_ACTIVE_YAML))).get("classes", []))
    cl_classes = set(yaml.safe_load(open(resolve_path(CONTRASTIVE_ACTIVE_YAML))).get("classes", []))
    overlap = cnn_classes & cl_classes
    if overlap:
        raise SystemExit(f"[ERR] CNN ↔ Contrastive class OVERLAP detected: {sorted(overlap)}")
    print(f"[classes] CNN: {len(cnn_classes)}, Contrastive: {len(cl_classes)}, overlap: 0")

    cnn_train = resolve_path(CNN_TRAIN_DIR); cl_train = resolve_path(CL_TRAIN_DIR); cl_eval = resolve_path(CL_EVAL_DIR)
    if not DRY_RUN:
        for d in (cnn_train, cl_train, cl_eval):
            d.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)

    # ---- CNN classes ----
    cnn_summary = defaultdict(int)
    for cls_dir in sorted(src.iterdir()):
        if not cls_dir.is_dir(): continue
        cls = cls_dir.name
        if cls in EXCLUDE_CLASSES: continue
        if cls not in cnn_classes: continue
        out_cls = cnn_train / cls
        if not DRY_RUN: out_cls.mkdir(exist_ok=True)
        pngs = sorted(cls_dir.glob("*.png"))
        for src_png in pngs:
            dst = out_cls / src_png.name
            place_file(src_png, dst)
            cnn_summary[cls] += 1
    print(f"[CNN_TRAIN] {dict(cnn_summary)}")
    print(f"[CNN_TRAIN] total: {sum(cnn_summary.values())} images, {len(cnn_summary)} classes")

    # ---- Contrastive classes — wafer-level split ----
    cl_train_summary = 0
    cl_eval_summary = defaultdict(int)
    for cls_dir in sorted(src.iterdir()):
        if not cls_dir.is_dir(): continue
        cls = cls_dir.name
        if cls in EXCLUDE_CLASSES: continue
        if cls not in cl_classes: continue

        pngs = sorted(cls_dir.glob("*.png"))
        random.shuffle(pngs)
        n_train = int(len(pngs) * CL_TRAIN_RATIO)
        train_split = pngs[:n_train]
        eval_split = pngs[n_train:]

        # Contrastive train: flat (class label 숨김)
        # 파일명 prefix 로 원본 class 보존 (debug 용 — model 은 못 봄)
        for src_png in train_split:
            dst = cl_train / f"{cls}__{src_png.name}"
            place_file(src_png, dst)
            cl_train_summary += 1

        # Contrastive eval: ImageFolder (class subdir)
        out_cls = cl_eval / cls
        if not DRY_RUN: out_cls.mkdir(exist_ok=True)
        for src_png in eval_split:
            dst = out_cls / src_png.name
            place_file(src_png, dst)
            cl_eval_summary[cls] += 1

    print(f"[CL_TRAIN  ] flat: {cl_train_summary} images ({CL_TRAIN_RATIO*100:.0f}%)")
    print(f"[CL_EVAL   ] {dict(cl_eval_summary)}")
    print(f"[CL_EVAL   ] total: {sum(cl_eval_summary.values())} images, {len(cl_eval_summary)} classes")

    # write manifest
    manifest = {
        "source_root": SOURCE_ROOT,
        "cnn_classes": sorted(cnn_classes),
        "contrastive_classes": sorted(cl_classes),
        "cnn_train_dir": CNN_TRAIN_DIR,
        "cl_train_dir": CL_TRAIN_DIR,
        "cl_eval_dir": CL_EVAL_DIR,
        "cl_train_ratio": CL_TRAIN_RATIO,
        "seed": SEED,
        "summary": {
            "cnn_train": dict(cnn_summary),
            "cl_train": cl_train_summary,
            "cl_eval": dict(cl_eval_summary),
        },
    }
    if not DRY_RUN:
        for d in (cnn_train, cl_train, cl_eval):
            (d.parent / f"{d.name}_manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
    print("\n[OUT]")
    print(f"  {CNN_TRAIN_DIR}/<class>/*.png  (CNN supervised)")
    print(f"  {CL_TRAIN_DIR}/*.png            (Contrastive train, flat)")
    print(f"  {CL_EVAL_DIR}/<class>/*.png    (Contrastive eval)")


def place_file(src, dst):
    """COPY_MODE 따라 link 또는 copy."""
    if DRY_RUN:
        print(f"  {COPY_MODE}: {src} → {dst}")
        return
    if dst.exists(): return
    if COPY_MODE == "link":
        try:
            import os
            os.link(src, dst)         # hard link (Windows NTFS 지원)
        except Exception:
            import shutil
            shutil.copy2(src, dst)
    else:
        import shutil
        shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
