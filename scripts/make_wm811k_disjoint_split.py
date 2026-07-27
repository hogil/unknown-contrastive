#!/usr/bin/env python3
"""Build WM-811K class-disjoint near-novel split.

Default protocol:
  A CNN supervised seen: Center, Edge-Ring, Near-full
  B contrastive unlabeled train: Loc, Scratch
  C novel eval: Donut, Edge-Loc, Random
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from _common import resolve_path


SRC_ALL = "E:/data/images/wm811k_500_train_eval_512/all"
OUT_ROOT = "E:/data/images/wm811k_novel_disjoint_v1"

CNN_CLASSES = ["Center", "Edge-Ring", "Near-full"]
CONTRASTIVE_CLASSES = ["Loc", "Scratch"]
EVAL_CLASSES = ["Donut", "Edge-Loc", "Random"]

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_classes(s: str | None, default: list[str]) -> list[str]:
    if not s:
        return list(default)
    return [x.strip() for x in s.split(",") if x.strip()]


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)


def class_images(src_all: Path, cls: str) -> list[Path]:
    root = src_all / cls
    if not root.is_dir():
        raise SystemExit(f"class folder not found: {root}")
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)


def write_imagefolder(src_all: Path, dst_root: Path, classes: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for cls in classes:
        imgs = class_images(src_all, cls)
        counts[cls] = len(imgs)
        for src in imgs:
            copy_file(src, dst_root / cls / src.name)
    return counts


def write_flat(src_all: Path, dst_root: Path, classes: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    dst_root.mkdir(parents=True, exist_ok=True)
    for cls in classes:
        imgs = class_images(src_all, cls)
        counts[cls] = len(imgs)
        for i, src in enumerate(imgs):
            dst_name = f"{cls}___{i:05d}___{src.name}"
            copy_file(src, dst_root / dst_name)
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-all", default=SRC_ALL, help="source ImageFolder root with all WM classes.")
    ap.add_argument("--out-root", default=OUT_ROOT, help="output split root.")
    ap.add_argument("--cnn-classes", default=None, help="comma-separated A/CNN seen classes.")
    ap.add_argument("--contrastive-classes", default=None, help="comma-separated B contrastive train classes.")
    ap.add_argument("--eval-classes", default=None, help="comma-separated C novel eval classes.")
    ap.add_argument("--clean", action="store_true", help="remove output root before writing.")
    args = ap.parse_args()

    src_all = resolve_path(args.src_all)
    out_root = resolve_path(args.out_root)
    cnn_classes = parse_classes(args.cnn_classes, CNN_CLASSES)
    contrastive_classes = parse_classes(args.contrastive_classes, CONTRASTIVE_CLASSES)
    eval_classes = parse_classes(args.eval_classes, EVAL_CLASSES)

    sets = {
        "cnn_classes": set(cnn_classes),
        "contrastive_classes": set(contrastive_classes),
        "eval_classes": set(eval_classes),
    }
    overlap = {
        "cnn_vs_contrastive": sorted(sets["cnn_classes"] & sets["contrastive_classes"]),
        "cnn_vs_eval": sorted(sets["cnn_classes"] & sets["eval_classes"]),
        "contrastive_vs_eval": sorted(sets["contrastive_classes"] & sets["eval_classes"]),
    }
    if any(overlap.values()):
        raise SystemExit(f"class split is not disjoint: {overlap}")
    if not src_all.is_dir():
        raise SystemExit(f"src-all not found: {src_all}")

    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cnn_dir = out_root / "cnn_seen_train"
    cl_train_dir = out_root / "contrastive_unlabeled_train"
    eval_dir = out_root / "novel_eval"

    counts = {
        "cnn_seen_train": write_imagefolder(src_all, cnn_dir, cnn_classes),
        "contrastive_unlabeled_train": write_flat(src_all, cl_train_dir, contrastive_classes),
        "novel_eval": write_imagefolder(src_all, eval_dir, eval_classes),
    }
    summary = {
        "src_all": str(src_all.resolve()),
        "out_root": str(out_root.resolve()),
        "cnn_seen_train": str(cnn_dir.resolve()),
        "contrastive_unlabeled_train": str(cl_train_dir.resolve()),
        "novel_eval": str(eval_dir.resolve()),
        "classes": {
            "cnn": cnn_classes,
            "contrastive": contrastive_classes,
            "eval": eval_classes,
        },
        "counts": counts,
        "totals": {k: int(sum(v.values())) for k, v in counts.items()},
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
