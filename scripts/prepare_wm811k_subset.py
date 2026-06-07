#!/usr/bin/env python3
"""Prepare a small WM-811K subset as ImageFolder PNGs for this pipeline.

Input:
  - Kaggle WM-811K `LSWMD.pkl` / `LSWMD.pkl.zip`, or
  - `--download` with configured Kaggle CLI credentials.

Output:
  data/images/wm811k_small/
    all/<class>/*.png
    train/<class>/*.png
    eval/<class>/*.png

The rendered PNG keeps only failure dies as grade color by default:
  waferMap value 2 -> palette index 6 (red)
  everything else -> palette index 0 (white)

This matches the current "grade 0-7 only, no border/background shortcut" loader policy.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from _common import resolve_path


DEFAULT_CLASSES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-pkl", type=str, default=None,
                   help="LSWMD.pkl path. If omitted, search --raw-dir.")
    p.add_argument("--raw-dir", type=str, default="data/raw/wm811k",
                   help="download/search directory for LSWMD.pkl")
    p.add_argument("--download", action="store_true",
                   help="download Kaggle qingyi/wm811k-wafer-map into --raw-dir")
    p.add_argument("--out-root", type=str, default="data/images/wm811k_small",
                   help="output root")
    p.add_argument("--classes", type=str, default=",".join(DEFAULT_CLASSES),
                   help="comma-separated failure classes")
    p.add_argument("--per-class", type=int, default=50,
                   help="samples per defect class")
    p.add_argument("--normal", type=int, default=0,
                   help="include this many 'none' wafers as Normal")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clean", action="store_true",
                   help="remove output root before writing")
    p.add_argument("--show-normal-dies", action="store_true",
                   help="render waferMap value 1 as light gray instead of white")
    p.add_argument("--copy-mode", choices=["copy", "link"], default="copy",
                   help="copy/link all->train/eval files")
    p.add_argument("--exclude-manifest", type=str, default=None,
                   help="manifest.csv whose row_index values are excluded")
    return p.parse_args()


def _run_download(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle", "datasets", "download",
        "-d", "qingyi/wm811k-wafer-map",
        "-p", str(raw_dir),
        "--unzip",
    ]
    try:
        rc = subprocess.run(cmd).returncode
    except FileNotFoundError:
        raise SystemExit(
            "kaggle CLI not found. Install/configure Kaggle first, or pass --source-pkl.\n"
            "  pip install kaggle\n"
            "  put kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME/KAGGLE_KEY")
    if rc != 0:
        raise SystemExit(f"Kaggle download failed rc={rc}")


def _find_pkl(raw_dir: Path, source_pkl: str | None) -> Path:
    if source_pkl:
        p = resolve_path(source_pkl)
        if p.exists():
            return p
        raise SystemExit(f"--source-pkl not found: {p}")

    raw_dir = resolve_path(raw_dir)
    direct = list(raw_dir.rglob("LSWMD.pkl")) + list(raw_dir.rglob("*.pkl"))
    if direct:
        return sorted(direct, key=lambda p: p.stat().st_size, reverse=True)[0]

    zips = list(raw_dir.rglob("*.zip"))
    for z in zips:
        try:
            with zipfile.ZipFile(z) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".pkl")]
                if not names:
                    continue
                raw_dir.mkdir(parents=True, exist_ok=True)
                zf.extract(names[0], raw_dir)
                return raw_dir / names[0]
        except zipfile.BadZipFile:
            continue
    raise SystemExit(
        f"LSWMD.pkl not found under {raw_dir}\n"
        f"  use --download, or pass --source-pkl /path/to/LSWMD.pkl")


def _label_value(x: Any) -> str:
    arr = np.asarray(x, dtype=object).reshape(-1)
    if arr.size == 0:
        return ""
    val = arr[0]
    if val is None:
        return ""
    s = str(val).strip()
    return "" if s.lower() in {"", "nan"} else s


def _load_dataframe(pkl: Path):
    try:
        import pandas as pd
    except Exception as e:
        raise SystemExit(f"pandas is required to read WM-811K pickle: {e}")
    return pd.read_pickle(pkl)


def _load_excluded_rows(manifest: str | None) -> set[int]:
    if not manifest:
        return set()
    p = resolve_path(manifest)
    if not p.exists():
        raise SystemExit(f"--exclude-manifest not found: {p}")
    out: set[int] = set()
    with p.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                out.add(int(row["row_index"]))
            except Exception:
                continue
    return out


def _palette() -> list[int]:
    pal = [[255, 255, 255] for _ in range(256)]
    pal[1] = [210, 210, 210]
    pal[2] = [0, 150, 25]
    pal[3] = [0, 0, 255]
    pal[4] = [217, 29, 255]
    pal[5] = [255, 255, 0]
    pal[6] = [255, 0, 0]
    pal[7] = [0, 0, 0]
    return [v for rgb in pal for v in rgb]


def _render_wafer_map(wmap: Any, image_size: int, show_normal_dies: bool) -> Image.Image:
    arr = np.asarray(wmap)
    if arr.ndim != 2:
        raise ValueError(f"waferMap must be 2-D, got {arr.shape}")
    idx = np.zeros(arr.shape, dtype=np.uint8)
    if show_normal_dies:
        idx[arr == 1] = 1
    idx[arr == 2] = 6

    h, w = idx.shape
    scale = max(1, image_size // max(h, w))
    new_size = (max(1, w * scale), max(1, h * scale))
    img = Image.frombytes("P", (idx.shape[1], idx.shape[0]), idx.tobytes())
    img.putpalette(_palette())
    img = img.resize(new_size, Image.Resampling.NEAREST)

    canvas = Image.new("P", (image_size, image_size), color=0)
    canvas.putpalette(_palette())
    x = (image_size - new_size[0]) // 2
    y = (image_size - new_size[1]) // 2
    canvas.paste(img, (x, y))
    return canvas


def _place(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "link":
        try:
            if dst.exists():
                dst.unlink()
            dst.hardlink_to(src)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def main():
    args = parse_args()
    raw_dir = resolve_path(args.raw_dir)
    if args.download:
        _run_download(raw_dir)
    pkl = _find_pkl(raw_dir, args.source_pkl)
    out_root = resolve_path(args.out_root)
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    for d in (out_root / "all", out_root / "train", out_root / "eval"):
        d.mkdir(parents=True, exist_ok=True)

    wanted = [x.strip() for x in args.classes.split(",") if x.strip()]
    wanted_set = set(wanted)
    rng = random.Random(args.seed)
    excluded_rows = _load_excluded_rows(args.exclude_manifest)
    if excluded_rows:
        print(f"[exclude] {len(excluded_rows)} row_index values from {resolve_path(args.exclude_manifest)}",
              flush=True)

    print(f"[load] {pkl}", flush=True)
    df = _load_dataframe(pkl)
    if "waferMap" not in df.columns or "failureType" not in df.columns:
        raise SystemExit(f"Unexpected columns: {list(df.columns)}")

    buckets: dict[str, list[int]] = {c: [] for c in wanted}
    if args.normal > 0:
        buckets["Normal"] = []
    for i, row in df.iterrows():
        if int(i) in excluded_rows:
            continue
        raw_label = _label_value(row["failureType"])
        label = "Normal" if raw_label.lower() == "none" else raw_label
        if label in wanted_set or (label == "Normal" and args.normal > 0):
            buckets.setdefault(label, []).append(i)

    manifest_rows = []
    all_counts = {}
    for label in list(buckets):
        idxs = list(buckets[label])
        rng.shuffle(idxs)
        limit = args.normal if label == "Normal" else args.per_class
        idxs = idxs[:max(0, limit)]
        all_counts[label] = len(idxs)
        n_train = int(len(idxs) * args.train_ratio)
        for rank, row_idx in enumerate(idxs):
            split = "train" if rank < n_train else "eval"
            row = df.loc[row_idx]
            img = _render_wafer_map(row["waferMap"], args.image_size, args.show_normal_dies)
            name = f"{label}_{rank:04d}_row-{row_idx}.png"
            all_path = out_root / "all" / label / name
            all_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(all_path)
            dst = out_root / split / label / name
            _place(all_path, dst, args.copy_mode)
            manifest_rows.append({
                "row_index": int(row_idx),
                "class": label,
                "split": split,
                "all_path": str(all_path),
                "split_path": str(dst),
                "wafer_shape": "x".join(map(str, np.asarray(row["waferMap"]).shape)),
            })

    with (out_root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "row_index", "class", "split", "all_path", "split_path", "wafer_shape",
        ])
        w.writeheader()
        w.writerows(manifest_rows)

    info = {
        "source_pkl": str(pkl),
        "out_root": str(out_root),
        "classes": wanted,
        "per_class": args.per_class,
        "normal": args.normal,
        "train_ratio": args.train_ratio,
        "image_size": args.image_size,
        "show_normal_dies": args.show_normal_dies,
        "counts": all_counts,
    }
    (out_root / "summary.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OUT] {out_root.resolve()}", flush=True)
    print(f"[counts] {all_counts}", flush=True)
    print("[next]", flush=True)
    print(f"  train: {out_root / 'train'}", flush=True)
    print(f"  eval:  {out_root / 'eval'}", flush=True)


if __name__ == "__main__":
    main()
