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
OUTPUT_DIR          = "data/images/unknown"
OUTPUT_PX           = 0                    # 0 = 6400 원본 유지, >0 = resize
SEED                = 42
N_PER_CLASS         = 100
N_NORMAL            = 200

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
        if cls in cg.CANVAS_CLASSES:
            canvas, chip_meta, palette = cg.render_canvas_in_memory(cls, seed)
            img_p = Image.frombytes("P", (sg.SIZE, sg.SIZE), canvas.tobytes())
            img_p.putpalette(palette)
            img = img_p.convert("RGB")
            try:
                Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        else:
            main, sub = split_main_sub(cls)
            png_path, _, _, _, _, _ = sg.render(main, sub, seed)
            img = Image.open(png_path).convert("RGB")
            try:
                Path(png_path).unlink()
            except Exception:
                pass

        if OUTPUT_PX > 0 and img.size != (OUTPUT_PX, OUTPUT_PX):
            img = img.resize((OUTPUT_PX, OUTPUT_PX), Image.LANCZOS)
        img.save(dst_path, optimize=False)
        return True
    except Exception as e:
        import traceback
        print(f"  [ERR] {cls} seed={seed}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return False


def main():
    out = resolve_path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for cls in CLASSES:
        sub_dir = out / cls
        sub_dir.mkdir(exist_ok=True)
        n_target = N_NORMAL if cls == "Normal" else N_PER_CLASS
        existing = len(list(sub_dir.glob("*.png")))
        if existing >= n_target:
            print(f"[skip] {cls}: {existing} >= {n_target}")
            total += existing
            continue
        print(f"[gen ] {cls}: {existing} -> {n_target}", flush=True)
        for i in range(existing, n_target):
            seed = SEED + abs(hash(cls)) % 10000 + i
            dst = sub_dir / f"wafer_{i:04d}.png"
            if render_wafer(cls, seed, dst):
                total += 1

    if _TMP.exists():
        try:
            shutil.rmtree(_TMP)
        except Exception:
            pass

    print(f"\n[OUT] {out.resolve()}")
    print(f"  total images: {total}, classes: {len(CLASSES)}")


if __name__ == "__main__":
    main()
