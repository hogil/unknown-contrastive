#!/usr/bin/env python3
"""Synthetic wafer generator — known-cnn 의 _sample_gen / _sample_canvas_gen 그대로 사용.

사용자 정책 (260527):
  "wafer 생성 코드에서 chip 내부 불량 모양만 known-cnn 참조"
  → wafer-level 알고리즘 자체 작성 X. known-cnn 의 정본 (_sample_gen / _sample_canvas_gen)
    그대로 호출. heatmap 의존도 mirror.

산출: data/images/unknown/<class>/wafer_<idx>.png
"""
from __future__ import annotations

# ===================================================================
# === CONFIG ===
# ===================================================================
OUTPUT_DIR          = "E:/data/images/unknown"            # ★ 절대규: 모든 이미지 E:/data/images/
OUTPUT_PX           = 0                    # 0 = 6400 원본 유지, >0 = resize
SEED                = 42
N_PER_CLASS         = 100
N_NORMAL            = 2000

CLASSES = [
    "Center_bank_boundary", "Center_fork", "Center_invalid_main",
    "Center_scratch", "Center_scratch_rot",
    "Donut_bank_boundary", "Donut_fork", "Donut_invalid_main",
    "Donut_scratch", "Donut_scratch_rot",
    "Edge-Bottom_bank_boundary", "Edge-Bottom_fork", "Edge-Bottom_invalid_main",
    "Edge-Bottom_scratch", "Edge-Bottom_scratch_rot",
    "Edge-Ring_bank_boundary", "Edge-Ring_fork", "Edge-Ring_invalid_main",
    "Edge-Ring_scratch", "Edge-Ring_scratch_rot",
    "Edge-Top_bank_boundary", "Edge-Top_fork", "Edge-Top_invalid_main",
    "Edge-Top_scratch", "Edge-Top_scratch_rot",
    "Full_bank_boundary", "Full_fork", "Full_invalid_main",
    "Full_scratch", "Full_scratch_rot",
    "Thick-Edge_fork", "Thick-Edge_invalid_main",
    "Normal",
    "BrokenRing", "CenterCircle", "CenterDonut", "CrescentArc",
    "CrossScratch", "DiagonalSmear", "ParallelScratches",
    "RingDots", "Row", "Starburst",
]
# ===================================================================

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

# import path + env (heatmap / out_dir) — _sample_gen import 전 설정
_SCRIPTS = Path(__file__).resolve().parent
_REPO = _SCRIPTS.parent
sys.path.insert(0, str(_SCRIPTS))
from _common import resolve_path

os.environ["HEATMAP_DIR"] = str(_REPO / "dist_learn" / "_dist_heatmaps")
_TMP = _REPO / "_tmp_wafer_gen"
(_TMP / "png").mkdir(parents=True, exist_ok=True)
(_TMP / "json").mkdir(parents=True, exist_ok=True)
(_TMP / "chips").mkdir(parents=True, exist_ok=True)
os.environ["WAFER_PNG_OUT_DIR"] = str(_TMP / "png")
os.environ["WAFER_JSON_OUT_DIR"] = str(_TMP / "json")
os.environ["CLASSIFICATION_CHIPS_ROOT"] = str(_TMP / "chips")
os.environ["SKIP_CHIP_CROPS"] = "1"

from PIL import Image  # noqa: E402

import _sample_gen as sg                  # noqa: E402
import _sample_canvas_gen as cg           # noqa: E402


def split_main_sub(cls: str) -> tuple[str, str]:
    if cls == "Normal":
        return "Normal", "normal"
    parts = cls.split("_", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (cls, cls)


def render_wafer(cls: str, seed: int, dst_path: Path) -> bool:
    """class 별 wafer 합성 → dst_path 저장."""
    try:
        Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
        if cls in cg.CANVAS_CLASSES:
            canvas, chip_meta, palette = cg.render_canvas_in_memory(cls, seed)
            img = Image.frombytes("P", (sg.SIZE, sg.SIZE), canvas.tobytes())
            img.putpalette(palette)                          # palette 'P' 유지 (RGB 변환 X)
        else:
            main, sub = split_main_sub(cls)
            png_path, _, _, _, _, _ = sg.render(main, sub, seed)
            img = Image.open(png_path).copy()                # palette 'P' 유지 (load + 복사)
            try:
                Path(png_path).unlink()
            except Exception:
                pass

        # ★ 정책: 이미지는 항상 palette PNG (mode 'P'). categorical → resize 는 NEAREST.
        if OUTPUT_PX > 0 and img.size != (OUTPUT_PX, OUTPUT_PX):
            img = img.resize((OUTPUT_PX, OUTPUT_PX), Image.NEAREST)
        if img.mode != "P":
            raise ValueError(f"{cls}: palette PNG 여야 하는데 mode={img.mode}")
        img.save(dst_path, optimize=False)
        return True
    except Exception as e:
        import traceback
        print(f"  [ERR] {cls} seed={seed}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return False


def _render_one(task):
    """ProcessPoolExecutor worker — (cls, seed, dst) 한 장 합성."""
    cls, seed, dst = task
    return cls, render_wafer(cls, seed, Path(dst))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, default=None,
                   help="생성 출력 폴더. 예: E:/data/images/unknown_260601")
    p.add_argument("--seed", type=int, default=None, help="SEED override.")
    p.add_argument("--n-per-class", type=int, default=None, help="Normal 외 class별 생성 수.")
    p.add_argument("--n-normal", type=int, default=None, help="Normal 생성 수.")
    p.add_argument("--output-px", type=int, default=None,
                   help="생성 PNG 한 변 크기. 0이면 원본 6400 유지, 예: 1024.")
    p.add_argument("--exclude-invalid", action="store_true",
                   help="class 이름에 invalid가 들어간 class는 생성하지 않음.")
    p.add_argument("--workers", type=int, default=None,
                   help="ProcessPool worker 수. 기본은 os.cpu_count().")
    return p.parse_args()


def main():
    global OUTPUT_PX
    args = parse_args()
    output_dir = args.output_dir or OUTPUT_DIR
    seed_base = SEED if args.seed is None else args.seed
    n_per_class = N_PER_CLASS if args.n_per_class is None else args.n_per_class
    n_normal = N_NORMAL if args.n_normal is None else args.n_normal
    if args.output_px is not None:
        OUTPUT_PX = args.output_px
    n_workers_cfg = args.workers
    classes = [c for c in CLASSES if (not args.exclude_invalid or "invalid" not in c.lower())]

    out = resolve_path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) 합성 task 수집 (skip 로직 — 가벼워서 순차)
    tasks = []
    skipped = 0
    for cls in classes:
        sub_dir = out / cls
        sub_dir.mkdir(exist_ok=True)
        n_target = n_normal if cls == "Normal" else n_per_class
        existing = len(list(sub_dir.glob("*.png")))
        if existing >= n_target:
            print(f"[skip] {cls}: {existing} >= {n_target}")
            skipped += existing
            continue
        print(f"[gen ] {cls}: {existing} -> {n_target}", flush=True)
        for i in range(existing, n_target):
            seed = seed_base + abs(hash(cls)) % 10000 + i
            dst = sub_dir / f"wafer_{i:04d}.png"
            tasks.append((cls, seed, str(dst)))

    # 2) ProcessPoolExecutor — CPU 코어 전부 사용 (환경 사양 다 씀)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    n_workers = max(1, n_workers_cfg if n_workers_cfg is not None else (os.cpu_count() or 8))
    total = 0
    if tasks:
        print(f"[gen] {len(tasks)} wafers → {n_workers} workers (CPU 코어 전부 활용)", flush=True)
        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            futs = [exe.submit(_render_one, t) for t in tasks]
            done = 0
            for fut in as_completed(futs):
                _cls, ok = fut.result()
                done += 1
                if ok:
                    total += 1
                if done % 50 == 0 or done == len(tasks):
                    print(f"  [{done}/{len(tasks)}] ok={total}", flush=True)

    if _TMP.exists():
        try:
            shutil.rmtree(_TMP)
        except Exception:
            pass

    print(f"\n[OUT] {out.resolve()}")
    print(f"  requested: {output_dir}")
    print(f"  generated: {total}, skipped(existing): {skipped}, classes: {len(classes)}")


if __name__ == "__main__":
    main()
