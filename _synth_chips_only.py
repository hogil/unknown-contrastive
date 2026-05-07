#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chip-only synthesizer — 200x200 chip 만 합성 (wafer 안 만듦).

v19 적용된 alpha_fork / alpha_scratch / alpha_scratch_rot / alpha_bank_boundary +
intensity tier (strong/mid/weak) + grade shift + 2px border + 8-color palette PNG 출력.

Usage:
    python _synth_chips_only.py --per-class 200 --out D:/project/data/wm-811k/classification_chips
    python _synth_chips_only.py --per-class 50 --classes fork scratch scratch_rot

wafer 합성 안 함. 빠르고 가벼움 (chip 200x200 만).
"""
from __future__ import annotations

import argparse
import sys
import os
import time
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Import sister repo dist_apply
sys.path.insert(0, str(Path("D:/project/known-cnn/dist_apply")))
import _sample_gen as sg


def _make_palette() -> bytes:
    """Reuse 8-color palette from _sample_gen palette."""
    return sg.PALETTE


def _try_font(size):
    for path in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(path):
            try: return ImageFont.truetype(path, size)
            except Exception: pass
    return ImageFont.load_default()


def render_chip(obj: str, rng: np.random.Generator,
                intensity_tier: str = None,
                bin_id: int = None,
                add_border: bool = True,
                add_bin_text: bool = True) -> np.ndarray:
    """Render single 200x200 chip canvas with v19 alpha modulation.

    obj: 'fork'/'scratch'/'scratch_rot'/'bank_boundary'/'invalid_main'
    intensity_tier: 'strong'/'mid'/'weak' (None → random per pick_intensity_tier)
    bin_id: defect bin (200-299). None → random from kind=00C bin pool.
    """
    CHIP = sg.CHIP
    if intensity_tier is None:
        intensity_tier = sg.pick_intensity_tier(rng)
    if bin_id is None:
        bin_id = int(rng.choice(sg.DEFECT_BIN_POOL['00C'], p=sg.DEFECT_BIN_WEIGHTS))

    is_invalid = (obj == 'invalid_main')
    if is_invalid:
        canvas = np.full((CHIP, CHIP), 31, dtype=np.uint8)  # palette 31 = white
    else:
        # baseline grade canvas
        baseline_tier = sg.pick_baseline_tier(rng)
        cum_base = sg.CUM_BASELINE_TIERS[baseline_tier]
        u = rng.random((CHIP, CHIP))
        canvas = np.searchsorted(cum_base, u).astype(np.uint8)

        # alpha modulation
        alpha_scale = sg.INTENSITY_ALPHA_SCALE[intensity_tier]
        alpha = sg.ALPHA_FNS[obj](rng) * alpha_scale
        cum_obj = np.cumsum(sg.shifted_object_dist(obj, intensity_tier))

        if obj in ('fork', 'scratch', 'scratch_rot'):
            # 2-stage sampling (matches _sample_gen render @line 954)
            u_base = rng.random((CHIP, CHIP))
            grades_base = np.searchsorted(sg.CUM_BASE, u_base).astype(np.uint8)
            u1 = rng.random((CHIP, CHIP))
            is_defect = u1 < alpha
            # 260507 v5: per-obj smoothstep — peak grade 2 미세 ↑ (사용자 "전체 미세하게 좀만 높게").
            # fork:              0.55/0.90 → 0.50/0.88   (peak 라인 grade 2 약간 ↑)
            # scratch / scr_rot: 0.65/0.93 → 0.60/0.91   (peak window 살짝 넓힘)
            if obj == 'fork':
                lo_t2, hi_t2 = 0.50, 0.88
            else:  # scratch / scratch_rot
                lo_t2, hi_t2 = 0.60, 0.91
            t2 = np.clip((alpha - lo_t2) / (hi_t2 - lo_t2), 0.0, 1.0).astype(np.float32)
            p_2 = (t2 * t2 * (3.0 - 2.0 * t2)).astype(np.float32)
            u2 = rng.random((CHIP, CHIP))
            is_2 = u2 < p_2
            u3 = rng.random((CHIP, CHIP))
            defect_other = np.where(u3 < 0.95, np.uint8(1),
                            np.where(u3 < 0.99, np.uint8(3), np.uint8(4)))
            defect_grade = np.where(is_2, np.uint8(2), defect_other)
            grades = np.where(is_defect, defect_grade, grades_base).astype(np.uint8)
        else:
            # 3-way zone mix (bank_boundary etc)
            # 260507 v5: split 0.5/0.5 → 0.45/0.55 (center weight 더 일찍 켜짐 → peak grade 2 미세 ↑)
            t_low = np.clip(alpha / 0.45, 0.0, 1.0).astype(np.float32)
            t_high = np.clip((alpha - 0.45) / 0.55, 0.0, 1.0).astype(np.float32)
            s_low = (t_low * t_low * (3.0 - 2.0 * t_low)).astype(np.float32)
            s_high = (t_high * t_high * (3.0 - 2.0 * t_high)).astype(np.float32)
            mask_low = (alpha < 0.45).astype(np.float32)
            mask_high = 1.0 - mask_low
            w_bg = (mask_low * (1.0 - s_low)).astype(np.float32)
            w_edge = (mask_low * s_low + mask_high * (1.0 - s_high)).astype(np.float32)
            w_center = (mask_high * s_high).astype(np.float32)
            cum_mixed = (w_bg[..., None] * sg.CUM_DEFECT_BG[None, None, :] +
                         w_edge[..., None] * sg.CUM_EDGE[None, None, :] +
                         w_center[..., None] * cum_obj[None, None, :])
            uu = rng.random((CHIP, CHIP))
            grades = (uu[..., None] < cum_mixed).argmax(axis=-1).astype(np.uint8)
        canvas = grades

    # 2px border
    if add_border:
        if obj == 'invalid_main':
            border_color = sg.KEY_TO_INDEX.get('border_inv', 30)
        else:
            border_color = sg.BIN_TO_BORDER_IDX.get(bin_id, sg.KEY_TO_INDEX.get('border_etc', 25))
        canvas[:2, :] = border_color
        canvas[-2:, :] = border_color
        canvas[:, :2] = border_color
        canvas[:, -2:] = border_color

    # Bin number text — INVALID chip 만 (다른 class 는 chip 안 text 없음)
    if add_bin_text and is_invalid:
        img = Image.frombytes('P', (CHIP, CHIP), canvas.tobytes())
        img.putpalette(_make_palette())
        draw = ImageDraw.Draw(img)
        font = _try_font(64)
        text = str(bin_id)
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            ty = CHIP // 2 - th // 2 - bbox[1]
        except Exception:
            tw, th, ty = 60, 40, CHIP // 2 - 20
        idx_text = sg.KEY_TO_INDEX.get('text', sg.KEY_TO_INDEX.get('border_inv', 30))
        draw.text((CHIP // 2 - tw // 2, ty), text, fill=idx_text, font=font)
        return img

    img = Image.frombytes('P', (CHIP, CHIP), canvas.tobytes())
    img.putpalette(_make_palette())
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=200)
    ap.add_argument('--out', type=str, default='D:/project/data/wm-811k/classification_chips')
    ap.add_argument('--classes', nargs='*',
                    default=['bank_boundary', 'fork', 'scratch', 'scratch_rot', 'invalid_main'])
    ap.add_argument('--seed', type=int, default=20260506)
    ap.add_argument('--clean-first', action='store_true',
                    help='delete existing classification_chips/<class>/*.png before generation')
    ap.add_argument('--prefix-pool', type=str, default=None,
                    help='ASCII prefix file (one prefix per line). Default: random 3-letter')
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    rng_master = np.random.default_rng(args.seed)
    t0 = time.time()
    total_made = 0

    for cls in args.classes:
        cls_dir = out_root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        if args.clean_first:
            cnt = 0
            for p in cls_dir.glob('*.png'):
                p.unlink()
                cnt += 1
            print(f"[clean] {cls}: removed {cnt} old PNGs", flush=True)

        for i in range(args.per_class):
            seed_i = int(rng_master.integers(0, 2**31 - 1))
            rng_i = np.random.default_rng(seed_i)
            prefix = ''.join(rng_i.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), size=3))
            kind = '00P' if rng_i.random() < 0.5 else '00C'
            w_idx = int(rng_i.integers(1, 25))
            yld = float(rng_i.uniform(85, 99))
            sys_n = int(rng_i.integers(1, 30))
            TD = str(rng_i.choice(['EE', 'PT', 'PE']))
            LT = str(rng_i.choice(['NORMAL', 'PWQ', 'ENGINEER']))
            x_abs = int(rng_i.integers(0, 32))
            y_abs = int(rng_i.integers(0, 32))
            bin_id = int(rng_i.choice(sg.DEFECT_BIN_POOL[kind], p=sg.DEFECT_BIN_WEIGHTS))

            img = render_chip(cls, rng_i, bin_id=bin_id)

            base = (f"{prefix}{rng_i.integers(100,999):03d}_{kind}_{w_idx:02d}"
                    f"_20260501_010000_{TD}_{LT}_X{x_abs}_Y{y_abs}_B{bin_id}")
            out_path = cls_dir / f"{base}.png"
            img.save(out_path, optimize=False, compress_level=1)
            total_made += 1

        print(f"[gen] {cls}: +{args.per_class}  (total in dir: {len(list(cls_dir.glob('*.png')))})", flush=True)

    elapsed = time.time() - t0
    print(f"\n[OK] {total_made} chips ({len(args.classes)} classes × {args.per_class}) "
          f"in {elapsed:.1f}s ({total_made / max(0.1, elapsed):.1f} img/s)")
    print(f"Output: {out_root}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
