#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build hardlink subset matching the data anchor spec.

Anchor (★ 전제 — feedback_data_spec_anchor.md, DECISIONS.md D-15):
- defect class 별 평균 ≈ 30, random 분포 (seed=42, Uniform(15, 45))
- Normal_bank_boundary 전체 사용
- file_list.parquet 자동 저장 (재현 보장)

Usage:
    python _build_anchor_subset.py
    python _build_anchor_subset.py --seed 42 --low 15 --high 45 --tag avg30
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

EXCLUDE_DIRS = {"classification", "classification_chips"}
NORMAL_CLASS = "Normal_bank_boundary"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="D:/project/data/wm-811k/unknown", type=Path)
    p.add_argument("--dst-root", default="D:/project/data/contrastive_anchor", type=Path)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--low", type=int, default=15, help="defect per-class min")
    p.add_argument("--high", type=int, default=45, help="defect per-class max (inclusive)")
    p.add_argument("--tag", default="avg30")
    p.add_argument("--normal-class", default="Normal_bank_boundary",
                   help="Normal class folder name (Normal_bank_boundary 또는 Normal)")
    args = p.parse_args()
    NORMAL_CLASS_ARG = args.normal_class

    src = args.src.resolve()
    if not src.exists():
        print(f"src not found: {src}", file=sys.stderr)
        return 2

    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    dst = (args.dst_root / f"{args.tag}_{ts}").resolve()
    dst.mkdir(parents=True, exist_ok=True)
    print(f"[anchor] src={src}")
    print(f"[anchor] dst={dst}")
    print(f"[anchor] seed={args.seed}  defect ~ Uniform({args.low}, {args.high})  Normal=ALL")

    classes = sorted(d.name for d in src.iterdir() if d.is_dir() and d.name not in EXCLUDE_DIRS)
    print(f"[classes] {len(classes)} (defect + Normal)")

    rng = np.random.default_rng(args.seed)
    rows = []
    n_total = 0
    n_defect_classes = 0
    n_defect_samples = 0

    summary = []
    for cls in classes:
        cls_src = src / cls
        files = sorted(p.name for p in cls_src.iterdir() if p.suffix.lower() == ".png")
        if not files:
            print(f"  {cls:<35} 0 (empty, skip)")
            continue

        if cls == NORMAL_CLASS_ARG:
            chosen = files
            kind = "Normal-ALL"
        else:
            n_pick = int(rng.integers(args.low, args.high + 1))
            n_pick = min(n_pick, len(files))
            idx = rng.choice(len(files), size=n_pick, replace=False)
            chosen = [files[i] for i in sorted(idx)]
            kind = "defect-sample"
            n_defect_classes += 1
            n_defect_samples += len(chosen)

        cls_dst = dst / cls
        cls_dst.mkdir(parents=True, exist_ok=True)
        for fname in chosen:
            s = cls_src / fname
            d = cls_dst / fname
            if not d.exists():
                try:
                    os.link(s, d)
                except OSError as e:
                    print(f"  link fail {cls}/{fname}: {e}", file=sys.stderr)
                    continue
            rows.append({"class": cls, "basename": fname, "src": str(s), "dst": str(d), "kind": kind})

        summary.append((cls, len(chosen), kind))
        n_total += len(chosen)

    fl = pd.DataFrame(rows)
    fl_path = dst / "file_list.parquet"
    fl.to_parquet(fl_path, index=False)

    print()
    print("=== per-class count ===")
    for cls, n, kind in summary:
        marker = "**" if kind == "Normal-ALL" else "  "
        print(f"  {marker} {cls:<35} {n:>5}  [{kind}]")
    print(f"  {'TOTAL':<38} {n_total:>5}")
    print()
    print(f"[stats] defect classes={n_defect_classes}  defect samples={n_defect_samples}  "
          f"defect mean per class={n_defect_samples / max(n_defect_classes, 1):.1f}")
    print(f"[file_list] saved -> {fl_path}")
    print(f"[OK] anchor subset ready  ({n_total} files)")
    print(f"     dst: {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
