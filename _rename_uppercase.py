#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-shot: normalize data file names to UPPERCASE basename + lowercase extension.

Folders are NOT touched (they're class labels used by ImageFolder).
predictions.csv is rebuilt with uppercase values (not header).

Targets:
  D:/project/data/wm-811k/unknown/<class>/*.png|.PNG
  D:/project/data/positions/unknown/<class>/*.json|.JSON
  D:/project/data/wm-811k/classification_chips/<obj>/*.png|.PNG
  D:/project/data/wm-811k/obj_id_maps/<subfolder>/*.npy|.NPY *.png|.PNG
  D:/project/data/wm-811k/obj_id_maps/predictions.csv (column values uppercase)

Idempotent: if file is already in target form, skip.
Convention: <STEM_UPPERCASE>.<ext_lowercase>
"""
from __future__ import annotations
import csv, os, sys
from pathlib import Path

ROOTS_EXTS = [
    (Path("D:/project/data/wm-811k/unknown"),              {".png"}),
    (Path("D:/project/data/positions/unknown"),            {".json"}),
    (Path("D:/project/data/wm-811k/classification_chips"), {".png"}),
    (Path("D:/project/data/wm-811k/obj_id_maps"),          {".npy", ".png"}),
]
PRED_CSV = Path("D:/project/data/wm-811k/obj_id_maps/predictions.csv")


def case_insensitive_rename(src: Path, dst_name: str) -> bool:
    if src.name == dst_name:
        return True
    dst = src.with_name(dst_name)
    try:
        src.rename(dst); return True
    except FileExistsError:
        tmp = src.with_name(dst_name + ".__renamerename__")
        try:
            src.rename(tmp); tmp.rename(dst); return True
        except Exception as e:
            print(f"  [collision] {src} -> {dst}: {e}"); return False
    except Exception as e:
        print(f"  [err] {src} -> {dst_name}: {e}"); return False


def normalize_name(name: str) -> str:
    """STEM uppercase + ext lowercase. Multiple suffixes (.tar.gz) treated single ext."""
    p = Path(name)
    return p.stem.upper() + p.suffix.lower()


def rename_root(root: Path, exts: set) -> None:
    if not root.exists():
        print(f"[skip] {root} not exist")
        return
    n_total = n_renamed = n_already = 0
    for p in root.rglob("*"):
        if not p.is_file(): continue
        if exts and p.suffix.lower() not in exts: continue
        n_total += 1
        new_name = normalize_name(p.name)
        if new_name != p.name:
            if case_insensitive_rename(p, new_name):
                n_renamed += 1
        else:
            n_already += 1
    print(f"[{root}] total={n_total} renamed={n_renamed} already={n_already}")


def upper_predictions_csv(csv_path: Path) -> None:
    if not csv_path.exists():
        print(f"[skip-csv] {csv_path} not exist"); return
    tmp = csv_path.with_suffix(".csv.tmp")
    n_rows = 0
    with csv_path.open("r", encoding="utf-8", newline="") as fin, \
         tmp.open("w", encoding="utf-8", newline="") as fout:
        r = csv.reader(fin); w = csv.writer(fout)
        try:
            header = next(r); w.writerow(header)
        except StopIteration:
            tmp.unlink(missing_ok=True); print(f"[csv] empty: {csv_path}"); return
        for row in r:
            new_row = [c.upper() if isinstance(c, str) else c for c in row]
            w.writerow(new_row); n_rows += 1
    csv_path.unlink(); tmp.rename(csv_path)
    print(f"[csv] {csv_path}: {n_rows} rows uppercased")


def main():
    print("=== normalize data file names (STEM uppercase + ext lowercase) ===")
    for root, exts in ROOTS_EXTS:
        rename_root(root, exts)
    print()
    print("=== uppercase predictions.csv values ===")
    upper_predictions_csv(PRED_CSV)
    print("\n[done]")


if __name__ == "__main__":
    sys.exit(main())
