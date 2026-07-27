#!/usr/bin/env python3
"""Build a WM811K+MixedWM38 component-disjoint split.

Train uses C/D/ER/NF components. Eval uses EL/L/S components.
This keeps train and eval labels disjoint and also avoids base-component overlap.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


IMAGE_ROOT = Path("E:/data/images")
WM_ALL = IMAGE_ROOT / "wm811k_500_train_eval_512" / "all"
MIXED_ALL = IMAGE_ROOT / "mixedwm38" / "rendered" / "all"
OUT_ROOT = IMAGE_ROOT / "cross_wm_mixed_disjoint_cder_nf_vs_ell_s"
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

TRAIN_SPECS = [
    ("wm811k", "Center", "WM_Center"),
    ("wm811k", "Donut", "WM_Donut"),
    ("wm811k", "Edge-Ring", "WM_Edge-Ring"),
    ("wm811k", "Near-full", "WM_Near-full"),
    ("mixedwm38", "C", "MWM_C"),
    ("mixedwm38", "D", "MWM_D"),
    ("mixedwm38", "ER", "MWM_ER"),
    ("mixedwm38", "NF", "MWM_NF"),
    ("mixedwm38", "C+ER", "MWM_C+ER"),
    ("mixedwm38", "D+ER", "MWM_D+ER"),
]

EVAL_SPECS = [
    ("mixedwm38", "EL", "EL"),
    ("mixedwm38", "L", "L"),
    ("mixedwm38", "S", "S"),
    ("mixedwm38", "EL+L", "EL+L"),
    ("mixedwm38", "EL+S", "EL+S"),
    ("mixedwm38", "L+S", "L+S"),
    ("mixedwm38", "EL+L+S", "EL+L+S"),
]


def source_root(dataset: str) -> Path:
    if dataset == "wm811k":
        return WM_ALL
    if dataset == "mixedwm38":
        return MIXED_ALL
    raise ValueError(dataset)


def image_paths(dataset: str, cls: str) -> list[Path]:
    root = source_root(dataset) / cls
    if not root.is_dir():
        raise SystemExit(f"class folder not found: {root}")
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    shutil.copy2(src, dst)


def write_split(specs: list[tuple[str, str, str]], dst_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for dataset, cls, out_cls in specs:
        imgs = image_paths(dataset, cls)
        counts[out_cls] = len(imgs)
        for src in imgs:
            dst_name = f"{dataset}___{cls.replace('+', '_')}___{src.name}"
            copy_file(src, dst_root / out_cls / dst_name)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    train_dir = out_root / "train"
    eval_dir = out_root / "eval"
    train_counts = write_split(TRAIN_SPECS, train_dir)
    eval_counts = write_split(EVAL_SPECS, eval_dir)
    train_labels = {x[2] for x in TRAIN_SPECS}
    eval_labels = {x[2] for x in EVAL_SPECS}
    overlap = sorted(train_labels & eval_labels)
    if overlap:
        raise SystemExit(f"train/eval label overlap: {overlap}")

    summary = {
        "out_root": str(out_root),
        "train": str(train_dir.resolve()),
        "eval": str(eval_dir.resolve()),
        "train_components": ["C", "D", "ER", "NF"],
        "eval_components": ["EL", "L", "S"],
        "train_specs": TRAIN_SPECS,
        "eval_specs": EVAL_SPECS,
        "train_counts": train_counts,
        "eval_counts": eval_counts,
        "totals": {
            "train": sum(train_counts.values()),
            "eval": sum(eval_counts.values()),
        },
    }
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
