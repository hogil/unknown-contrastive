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
CNN_TRAIN_DIR       = "E:/data/images/cnn_train"           # ★ 절대규: 모든 이미지 E:/data/images/
CL_TRAIN_DIR        = "E:/data/images/contrastive_train"   # flat (no class subdir)
CL_EVAL_DIR         = "E:/data/images/contrastive_eval"    # ImageFolder (class subdir)

# Contrastive train vs eval split (wafer disjoint)
CL_TRAIN_RATIO      = 0.8                                   # 80% train / 20% eval

# 이미지 처리
COPY_MODE           = "link"        # "link" (symlink, 빠름 + disk 절약) or "copy"
SEED                = 42
DRY_RUN             = False         # True 면 path 만 print
# ===================================================================

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from _common import resolve_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=str, default=None,
                   help="원본 이미지 폴더. 예: E:/data/images/unknown_260601")
    p.add_argument("--cnn-train-dir", type=str, default=None,
                   help="CNN ImageFolder 출력 폴더.")
    p.add_argument("--cl-train-dir", type=str, default=None,
                   help="Contrastive train flat 출력 폴더.")
    p.add_argument("--cl-eval-dir", type=str, default=None,
                   help="Contrastive eval ImageFolder 출력 폴더.")
    p.add_argument("--copy-mode", type=str, choices=["link", "copy"], default=None,
                   help="link=hardlink(빠름), copy=복사.")
    p.add_argument("--seed", type=int, default=None, help="split seed override.")
    p.add_argument("--dry-run", action="store_true", help="파일 생성 없이 경로만 출력.")
    return p.parse_args()


def main():
    args = parse_args()
    source_root = args.source_root or SOURCE_ROOT
    cnn_train_dir = args.cnn_train_dir or CNN_TRAIN_DIR
    cl_train_dir = args.cl_train_dir or CL_TRAIN_DIR
    cl_eval_dir = args.cl_eval_dir or CL_EVAL_DIR
    copy_mode = args.copy_mode or COPY_MODE
    seed = SEED if args.seed is None else args.seed
    dry_run = DRY_RUN or args.dry_run

    src = resolve_path(source_root)
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

    cnn_train = resolve_path(cnn_train_dir); cl_train = resolve_path(cl_train_dir); cl_eval = resolve_path(cl_eval_dir)
    if not dry_run:
        for d in (cnn_train, cl_train, cl_eval):
            d.mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    # ---- CNN classes ----
    cnn_summary = defaultdict(int)
    for cls_dir in sorted(src.iterdir()):
        if not cls_dir.is_dir(): continue
        cls = cls_dir.name
        if cls in EXCLUDE_CLASSES: continue
        if cls not in cnn_classes: continue
        out_cls = cnn_train / cls
        if not dry_run: out_cls.mkdir(exist_ok=True)
        pngs = sorted(cls_dir.glob("*.png"))
        for src_png in pngs:
            dst = out_cls / src_png.name
            place_file(src_png, dst, copy_mode, dry_run)
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
            place_file(src_png, dst, copy_mode, dry_run)
            cl_train_summary += 1

        # Contrastive eval: ImageFolder (class subdir)
        out_cls = cl_eval / cls
        if not dry_run: out_cls.mkdir(exist_ok=True)
        for src_png in eval_split:
            dst = out_cls / src_png.name
            place_file(src_png, dst, copy_mode, dry_run)
            cl_eval_summary[cls] += 1

    print(f"[CL_TRAIN  ] flat: {cl_train_summary} images ({CL_TRAIN_RATIO*100:.0f}%)")
    print(f"[CL_EVAL   ] {dict(cl_eval_summary)}")
    print(f"[CL_EVAL   ] total: {sum(cl_eval_summary.values())} images, {len(cl_eval_summary)} classes")

    # write manifest
    manifest = {
        "source_root": source_root,
        "cnn_classes": sorted(cnn_classes),
        "contrastive_classes": sorted(cl_classes),
        "cnn_train_dir": cnn_train_dir,
        "cl_train_dir": cl_train_dir,
        "cl_eval_dir": cl_eval_dir,
        "cl_train_ratio": CL_TRAIN_RATIO,
        "copy_mode": copy_mode,
        "seed": seed,
        "summary": {
            "cnn_train": dict(cnn_summary),
            "cl_train": cl_train_summary,
            "cl_eval": dict(cl_eval_summary),
        },
    }
    if not dry_run:
        for d in (cnn_train, cl_train, cl_eval):
            (d.parent / f"{d.name}_manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
            )
    print("\n[OUT]")
    print(f"  {cnn_train_dir}/<class>/*.png  (CNN supervised)")
    print(f"  {cl_train_dir}/*.png            (Contrastive train, flat)")
    print(f"  {cl_eval_dir}/<class>/*.png    (Contrastive eval)")


def place_file(src, dst, copy_mode, dry_run):
    """COPY_MODE 따라 link 또는 copy."""
    if dry_run:
        print(f"  {copy_mode}: {src} → {dst}")
        return
    if dst.exists(): return
    if copy_mode == "link":
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
