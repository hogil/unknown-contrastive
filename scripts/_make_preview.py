#!/usr/bin/env python3
"""43 class × 3 sample wafer → flat folder, class 별 따로 파일.

class 폴더 X, montage X — 그저 file 마다 1 wafer.
파일명: <class>_<idx>.png  (예: Center_scratch_0.png, Center_scratch_1.png, ..., Starburst_2.png)

사용: python scripts/_make_preview.py
산출: data/_preview/<class>_<idx>.png (43 × 3 = 129 파일)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import resolve_path
import generate_data as gen

# ===================================================================
OUT_DIR    = "E:/data/images/_preview"  # ★ 절대규: 모든 이미지 E:/data/images/  (flat, class subdir X)
N_SAMPLES  = 3                    # class 당 sample 수
TILE_PX    = 0                    # 0 = 6400 원본 유지, 또는 resize 크기
SEED_BASE  = 42
# ===================================================================


def main():
    out = resolve_path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    # 기존 flat preview 정리
    for p in out.glob("*.png"):
        p.unlink()

    total = 0
    for cls in gen.CLASSES:
        for i in range(N_SAMPLES):
            seed = SEED_BASE + abs(hash(cls)) % 10000 + i
            dst = out / f"{cls}_{i}.png"
            print(f"[{total + 1:>3}/{len(gen.CLASSES) * N_SAMPLES}] {cls}_{i}", flush=True)
            ok = gen.render_wafer(cls, seed, dst)
            if ok:
                # optional resize — palette 'P' 유지 (categorical → NEAREST, RGB 변환 X)
                if TILE_PX > 0:
                    from PIL import Image
                    Image.open(dst).resize(
                        (TILE_PX, TILE_PX), Image.NEAREST).save(dst, optimize=False)
                total += 1

    print(f"\n[OUT] {out.resolve()}")
    print(f"  total files: {total}  ({len(gen.CLASSES)} classes × {N_SAMPLES} samples)")
    print(f"  file format: <class>_<idx>.png (flat, no class subdir)")


if __name__ == "__main__":
    main()
