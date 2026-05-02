#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wait for obj_id_maps build to finish, verify outputs, then dispatch compound CNN training.

Steps:
  1. Poll: D:/project/data/wm-811k/obj_id_maps/_meta.json + count(*.npy)
  2. On build done: copy `_meta.json` + write `counts.txt` (per-class wafer + obj_id pixel
     distribution) into the run dir so the build run is self-contained.
  3. Dispatch: python cnn_train_compound.py — stdout goes to <run-dir>/compound_dispatch.log
     (cnn_train_compound itself writes its own logs_compound/<run>/ folder per its convention).
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, time
from collections import Counter
from pathlib import Path

import numpy as np

OBJ_ID_DIR = Path("D:/project/data/wm-811k/obj_id_maps")
META = OBJ_ID_DIR / "_meta.json"
PNG_ROOT = Path("D:/project/data/wm-811k/unknown")                                   # for EXPECTED_NPY auto-count
JSON_ROOT = Path("D:/project/data/positions/unknown")
PREDICTIONS_CSV = OBJ_ID_DIR / "predictions.csv"
POLL_SEC = 60


def expected_npy_count() -> int:
    """Auto-derive: count wafer PNGs that have matching JSON. Hardcode 금지 (CLAUDE.md 절대 규칙)."""
    if not PNG_ROOT.exists() or not JSON_ROOT.exists():
        return 0
    n = 0
    for cls_dir in PNG_ROOT.iterdir():
        if not cls_dir.is_dir():
            continue
        for png in cls_dir.glob("*.png"):
            if (JSON_ROOT / cls_dir.name / (png.stem + ".json")).exists():
                n += 1
    return n


def npy_count() -> int:
    if not OBJ_ID_DIR.exists():
        return 0
    return sum(1 for _ in OBJ_ID_DIR.rglob("*.npy"))


def build_done(expected: int) -> bool:
    if not META.exists():
        return False
    return npy_count() >= expected * 0.99


def write_counts_summary(run_dir: Path) -> None:
    """Per-subfolder wafer count + per-obj_id pixel count (across all maps).
    Subfolder is whatever scheme `_build_obj_id_maps.py` used (e.g. <device>_<date>).
    """
    out = run_dir / "counts.txt"
    per_subfolder: Counter = Counter()
    per_obj_pixels: Counter = Counter()
    total_npy = 0
    if OBJ_ID_DIR.exists():
        for cls_dir in sorted(p for p in OBJ_ID_DIR.iterdir() if p.is_dir()):
            for npy in cls_dir.glob("*.npy"):
                per_subfolder[cls_dir.name] += 1
                total_npy += 1
                try:
                    arr = np.load(npy)
                    vals, cnts = np.unique(arr, return_counts=True)
                    for v, c in zip(vals, cnts):
                        per_obj_pixels[int(v)] += int(c)
                except Exception:
                    continue
    obj_id_to_label = []
    try:
        meta = json.loads(META.read_text(encoding="utf-8"))
        obj_id_to_label = meta.get("obj_id_to_label", [])
    except Exception:
        pass
    lines = []
    lines.append(f"total_wafers_with_npy = {total_npy}")
    lines.append("")
    lines.append("=== per-subfolder wafer count ===")
    for sf, n in sorted(per_subfolder.items()):
        lines.append(f"  {sf:40s} {n:5d}")
    lines.append("")
    lines.append("=== per-obj_id pixel count (across all maps; map shape derived from .npy) ===")
    total_pixels = sum(per_obj_pixels.values()) or 1
    for oid in sorted(per_obj_pixels.keys()):
        label = obj_id_to_label[oid] if oid < len(obj_id_to_label) else f"id{oid}"
        c = per_obj_pixels[oid]
        lines.append(f"  obj_id={oid} {label:20s} pixels={c:>10d}  ({100.0*c/total_pixels:5.2f}%)")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _finalize_run_dir(run_dir: Path, status: str) -> Path:
    """Rename run_dir to <name>_<status>. status in {OK, ABORTED}."""
    new_name = run_dir.name + f"_{status}"
    target = run_dir.with_name(new_name)
    try:
        run_dir.rename(target)
        print(f"[chain] run_dir finalized: {run_dir.name} -> {new_name}", flush=True)
        return target
    except Exception as e:
        print(f"[chain] rename to {new_name} failed: {e}", flush=True)
        return run_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="logs_obj/build_<TS>/ — orchestration log + meta archive root")
    ap.add_argument("--no-compound", action="store_true",
                    help="skip cnn_train_compound dispatch (just archive build outputs)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ----- sweep: 이전 run 들 중 status suffix 없는 폴더 = 이전 abort 잔재 → _ABORTED 처리 -----
    parent = run_dir.parent
    if parent.exists():
        for p in parent.iterdir():
            if not p.is_dir(): continue
            if p == run_dir: continue
            if p.name.endswith("_OK") or p.name.endswith("_ABORTED") or p.name.endswith("_RUNNING"):
                continue
            try:
                p.rename(p.with_name(p.name + "_ABORTED"))
                print(f"[chain] swept stale run: {p.name} -> {p.name}_ABORTED", flush=True)
            except Exception as e:
                print(f"[chain] sweep failed for {p}: {e}", flush=True)

    expected = expected_npy_count()
    print(f"[chain] run_dir={run_dir}", flush=True)
    print(f"[chain] expected_npy={expected} (auto-counted from PNG_ROOT × JSON_ROOT)", flush=True)
    print(f"[chain] waiting for obj_id_maps build to finish...", flush=True)
    try:
        while not build_done(expected):
            n = npy_count()
            meta_ok = META.exists()
            ts = time.strftime("%H:%M:%S")
            print(f"[chain {ts}] npy={n}/{expected} meta={meta_ok}", flush=True)
            time.sleep(POLL_SEC)
    except KeyboardInterrupt:
        print("[chain] interrupted while polling — marking ABORTED", flush=True)
        _finalize_run_dir(run_dir, "ABORTED")
        return 130

    n = npy_count()
    print(f"[chain] build done — npy={n}/{expected} meta={META}", flush=True)

    # 1. copy _meta.json into run_dir
    try:
        shutil.copy2(META, run_dir / "obj_id_maps_meta.json")
        print(f"[chain] copied {META.name} -> {run_dir / 'obj_id_maps_meta.json'}", flush=True)
    except Exception as e:
        print(f"[chain] meta copy failed: {e}", flush=True)

    # 2. copy predictions.csv into run_dir
    try:
        if PREDICTIONS_CSV.exists():
            shutil.copy2(PREDICTIONS_CSV, run_dir / "predictions.csv")
            print(f"[chain] copied {PREDICTIONS_CSV.name} -> {run_dir / 'predictions.csv'}  "
                  f"({PREDICTIONS_CSV.stat().st_size / (1024*1024):.1f} MB)", flush=True)
        else:
            print(f"[chain] predictions.csv missing at {PREDICTIONS_CSV} — skip", flush=True)
    except Exception as e:
        print(f"[chain] predictions.csv copy failed: {e}", flush=True)

    # 3. write counts summary
    try:
        write_counts_summary(run_dir)
        print(f"[chain] wrote counts.txt", flush=True)
    except Exception as e:
        print(f"[chain] counts.txt write failed: {e}", flush=True)

    # 3. show meta highlights
    try:
        meta = json.loads(META.read_text(encoding="utf-8"))
        print(f"[chain] meta: n_chip_objects={meta.get('n_chip_objects')} "
              f"chip_classes={meta.get('chip_classes')}", flush=True)
    except Exception as e:
        print(f"[chain] meta read failed: {e}", flush=True)

    if args.no_compound:
        print("[chain] --no-compound: skipping compound train dispatch", flush=True)
        _finalize_run_dir(run_dir, "OK")
        return 0

    # 4. dispatch compound training — cnn_train_compound.py writes its own logs_compound/<run>/
    compound_log = run_dir / "compound_dispatch.log"
    with compound_log.open("w", encoding="utf-8") as f:
        f.write(f"=== compound train dispatch {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    cmd = [sys.executable, "-u", "cnn_train_compound.py"]
    print(f"[chain] dispatch: {' '.join(cmd)}  log={compound_log}", flush=True)
    with compound_log.open("a", encoding="utf-8") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"[chain] training PID={proc.pid}", flush=True)
    rc = proc.wait()
    print(f"[chain] training exit_code={rc}", flush=True)
    _finalize_run_dir(run_dir, "OK" if rc == 0 else "ABORTED")
    return rc


if __name__ == "__main__":
    sys.exit(main())
