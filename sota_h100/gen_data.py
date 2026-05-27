#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data generation for the H100 single-SOTA reproduction (fully independent).

Reproduces the two reference folders without importing any existing generator:

  --mode train  ->  classification_chips/  (4 single defects, N/class, palette PNG)
                    used as the SOTA training source (single-defect only).

  --mode eval   ->  eval_set/  (the v15direct_n2000-equivalent evaluation set)
                    4 single + 6 two-combo + 4 OOD + Normal + Invalid = 16 classes,
                    N/class. Writes manifest.csv + _preview/<class>.png 16-grid.

Usage:
  python -m sota_h100.gen_data --mode train --per-class 200 --out data/images/classification_chips
  python -m sota_h100.gen_data --mode eval  --per-class 2000 --out data/images/eval_set
  python -m sota_h100.gen_data --mode eval  --per-class 5 --out /tmp/eval_smoke   # smoke
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from . import synth
except ImportError:  # allow direct `python sota_h100/gen_data.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from sota_h100 import synth

TRAIN_CLASSES = synth.SINGLE_DEFECTS                                   # 4
EVAL_CLASSES = (synth.SINGLE_DEFECTS + synth.COMBO_2 +
                synth.OOD_CLASSES + synth.SPECIAL)                     # 16


def _save(img, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, optimize=False, compress_level=1)


def _preview(chips_dir: Path, out_path: Path, n: int = 16):
    files = sorted(chips_dir.glob("*.png"))[:n]
    if not files:
        return
    cell = synth.CHIP
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    canvas = np.full((rows * cell, cols * cell, 3), 255, dtype=np.uint8)
    for i, f in enumerate(files):
        r, c = i // cols, i % cols
        with Image.open(f) as im:
            arr = np.array(im.convert("RGB"))
        canvas[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = arr
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)


def generate(mode: str, per_class: int, out_root: Path, seed: int, clean: bool):
    classes = TRAIN_CLASSES if mode == "train" else EVAL_CLASSES
    out_root.mkdir(parents=True, exist_ok=True)
    master = np.random.default_rng(seed)
    t0 = time.time()
    rows = []
    total = 0
    for cls in classes:
        cdir = out_root / cls
        cdir.mkdir(parents=True, exist_ok=True)
        if clean:
            for p in cdir.glob("*.png"):
                p.unlink()
        if cls in synth.OOD_CLASSES:
            # one wafer-pattern realization yields many chips (cells the pattern crosses)
            i = 0
            wafer = 0
            max_wafer = per_class * 5 + 50
            while i < per_class and wafer < max_wafer:
                pat_seed = int(master.integers(0, 2**31 - 1))
                for gy, gx, img in synth.iter_ood_chips(cls, pat_seed):
                    _save(img, cdir / f"{cls}_{i:05d}_g{gy}_{gx}.png")
                    i += 1
                    if i >= per_class:
                        break
                wafer += 1
        else:
            for i in range(per_class):
                chip_seed = int(master.integers(0, 2**31 - 1))
                rng = np.random.default_rng(chip_seed)
                img = synth.render_chip(cls, rng)
                _save(img, cdir / f"{cls}_{i:05d}_s{chip_seed}.png")
        n = len(list(cdir.glob("*.png")))
        _preview(cdir, out_root / "_preview" / f"{cls}.png")
        rows.append({"class_key": cls, "n": n})
        total += n
        print(f"[gen:{mode}] {cls}: {n}", flush=True)

    # NOTE: filename intentionally NOT "manifest.csv" — run_stage1 treats a
    # manifest.csv as a per-chip schema (chip_path/defect_pixel_ratio) and would
    # mis-discover the eval set. Without it run_stage1 walks subdirs (as it does
    # for the reference v15direct_n2000). This file is a human-readable summary.
    with (out_root / "gen_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["class_key", "n"])
        w.writeheader()
        w.writerows(rows)

    dt = time.time() - t0
    print(f"\n[OK] mode={mode} {total} chips ({len(classes)} classes x {per_class}) "
          f"in {dt:.1f}s ({total / max(0.1, dt):.1f} img/s)")
    print(f"[OUT] {out_root.resolve()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval"], required=True)
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=20260527)
    ap.add_argument("--clean-first", action="store_true")
    args = ap.parse_args()
    generate(args.mode, args.per_class, Path(args.out), args.seed, args.clean_first)
    return 0


if __name__ == "__main__":
    sys.exit(main())
