#!/usr/bin/env python3
"""Synthetic wafer image generator — class 별 distinct pattern.

CONFIG 의 N_PER_CLASS 만큼 각 class 마다 PNG 생성.
산출: data/images/unknown/<class>/wafer_<seed>.png

real WM-811K 데이터 없을 때 demo / pipeline 검증용. paper 결과 측면은
실제 데이터로 진행해야 함.
"""
# ===================================================================
# === CONFIG ===
# ===================================================================
OUTPUT_DIR          = "data/images/unknown"     # 프로젝트 상대
IMG_SIZE            = 384
N_PER_CLASS         = 100                        # 각 class 당 PNG 수 (43 class × 100 = 4300)
N_NORMAL            = 200                        # Normal class 만 더 많이
SEED                = 42

# Split A 21 class (CNN supervised) + Split B 22 class (Contrastive) — 동일 def
CLASSES = [
    # main pattern × sub-class (Center/Donut/Edge-Bottom/Edge-Ring/Edge-Top/Full)
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
    # canvas 10
    "BrokenRing", "CenterCircle", "CenterDonut", "CrescentArc",
    "CrossScratch", "DiagonalSmear", "ParallelScratches",
    "RingDots", "Row", "Starburst",
]
# ===================================================================

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from _common import resolve_path


def main_pattern_of(cls: str) -> str:
    if cls == "Normal":
        return "normal"
    if cls in {"BrokenRing","CenterCircle","CenterDonut","CrescentArc","CrossScratch",
               "DiagonalSmear","ParallelScratches","RingDots","Row","Starburst"}:
        return cls   # canvas — 자기 자신
    return cls.split("_")[0]


def sub_pattern_of(cls: str) -> str:
    if "_" in cls:
        return "_".join(cls.split("_")[1:])
    return ""


def make_base():
    """wafer base — circular gray plate."""
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(40, 40, 40))
    draw = ImageDraw.Draw(img)
    cx = cy = IMG_SIZE // 2
    r = IMG_SIZE // 2 - 8
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(95, 70, 80))
    return img


def add_main_region(draw, main: str, color):
    """main pattern 의 영역에 base color 적용."""
    cx = cy = IMG_SIZE // 2
    r = IMG_SIZE // 2 - 8
    if main == "Center":
        cr = r // 2
        draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=color)
    elif main == "Donut":
        for rad in range(r // 3, int(r * 0.75)):
            if rad % 6 < 4:
                draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=color, width=1)
    elif main == "Edge-Top":
        draw.pieslice([cx - r, cy - r, cx + r, cy + r], -120, -60, fill=color)
    elif main == "Edge-Bottom":
        draw.pieslice([cx - r, cy - r, cx + r, cy + r], 60, 120, fill=color)
    elif main == "Edge-Ring":
        # ring band 전체 둘레
        for ang in range(0, 360, 5):
            x1 = cx + int(r * 0.85 * math.cos(math.radians(ang)))
            y1 = cy + int(r * 0.85 * math.sin(math.radians(ang)))
            draw.ellipse([x1 - 4, y1 - 4, x1 + 4, y1 + 4], fill=color)
    elif main == "Full":
        # 전체 dot density
        for _ in range(800):
            x = random.randint(0, IMG_SIZE - 1); y = random.randint(0, IMG_SIZE - 1)
            if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
                draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color)
    elif main == "Thick-Edge":
        # 두꺼운 ring band
        for rad in range(int(r * 0.7), r):
            draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=color, width=1)


def add_sub_pattern(draw, main: str, sub: str, color):
    """sub-pattern: bank_boundary, fork, invalid_main, scratch, scratch_rot."""
    cx = cy = IMG_SIZE // 2
    if sub == "bank_boundary":
        # 수직 bank 선
        draw.line([cx, 20, cx, IMG_SIZE - 20], fill=color, width=3)
        draw.line([cx + 60, 20, cx + 60, IMG_SIZE - 20], fill=(180, 100, 60), width=2)
    elif sub == "fork":
        # Y 모양
        draw.line([cx, IMG_SIZE // 4, cx, IMG_SIZE * 3 // 4], fill=color, width=4)
        draw.line([cx, IMG_SIZE // 2, cx - 60, IMG_SIZE * 3 // 4], fill=color, width=3)
        draw.line([cx, IMG_SIZE // 2, cx + 60, IMG_SIZE * 3 // 4], fill=color, width=3)
    elif sub == "invalid_main":
        # 빈 영역 + 텍스트 같은 점
        for _ in range(15):
            x = random.randint(50, IMG_SIZE - 50)
            y = random.randint(50, IMG_SIZE - 50)
            draw.rectangle([x, y, x + 8, y + 8], fill=(220, 200, 60))
    elif sub == "scratch":
        # 대각선
        draw.line([60, 60, IMG_SIZE - 60, IMG_SIZE - 60], fill=color, width=3)
    elif sub == "scratch_rot":
        # 다른 각도 대각선 (-21°)
        rad = math.radians(-21)
        x1 = cx - 100 * math.cos(rad); y1 = cy - 100 * math.sin(rad)
        x2 = cx + 100 * math.cos(rad); y2 = cy + 100 * math.sin(rad)
        draw.line([(x1, y1), (x2, y2)], fill=color, width=4)


def render_canvas(cls: str, color):
    """Canvas patterns — 각자 고유 패턴."""
    img = make_base()
    draw = ImageDraw.Draw(img)
    cx = cy = IMG_SIZE // 2
    r = IMG_SIZE // 2 - 20

    if cls == "BrokenRing":
        for ang in range(0, 360, 12):
            if random.random() > 0.4: continue
            x = cx + int(r * math.cos(math.radians(ang)))
            y = cy + int(r * math.sin(math.radians(ang)))
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=color)
    elif cls == "CenterCircle":
        draw.ellipse([cx - 50, cy - 50, cx + 50, cy + 50], outline=color, width=5)
    elif cls == "CenterDonut":
        draw.ellipse([cx - 70, cy - 70, cx + 70, cy + 70], outline=color, width=10)
        draw.ellipse([cx - 30, cy - 30, cx + 30, cy + 30], fill=(40, 40, 40))
    elif cls == "CrescentArc":
        draw.arc([cx - r, cy - r, cx + r, cy + r], -45, 135, fill=color, width=8)
    elif cls == "CrossScratch":
        draw.line([60, 60, IMG_SIZE - 60, IMG_SIZE - 60], fill=color, width=4)
        draw.line([IMG_SIZE - 60, 60, 60, IMG_SIZE - 60], fill=color, width=4)
    elif cls == "DiagonalSmear":
        for off in range(-100, 100, 8):
            draw.line([60 + off, 60, IMG_SIZE - 60 + off, IMG_SIZE - 60],
                      fill=color, width=2)
    elif cls == "ParallelScratches":
        for off in range(-80, 81, 30):
            draw.line([60, cy + off, IMG_SIZE - 60, cy + off], fill=color, width=3)
    elif cls == "RingDots":
        for ang in range(0, 360, 18):
            x = cx + int(r * 0.7 * math.cos(math.radians(ang)))
            y = cy + int(r * 0.7 * math.sin(math.radians(ang)))
            draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=color)
    elif cls == "Row":
        for y in range(80, IMG_SIZE - 80, 30):
            draw.line([60, y, IMG_SIZE - 60, y], fill=color, width=2)
    elif cls == "Starburst":
        for ang in range(0, 360, 15):
            x = cx + int(r * math.cos(math.radians(ang)))
            y = cy + int(r * math.sin(math.radians(ang)))
            draw.line([(cx, cy), (x, y)], fill=color, width=2)
    return img


def render_class(cls: str, seed: int):
    """class 별 wafer image."""
    random.seed(seed)
    np.random.seed(seed)
    main = main_pattern_of(cls)
    sub = sub_pattern_of(cls)

    # 색상 — main pattern 별 distinct
    color_map = {
        "Center": (220, 80, 80), "Donut": (80, 220, 80),
        "Edge-Bottom": (80, 80, 220), "Edge-Ring": (220, 220, 80),
        "Edge-Top": (220, 80, 220), "Full": (80, 220, 220),
        "Thick-Edge": (200, 140, 60),
        "normal": (95, 70, 80),
    }

    if main == "normal":
        img = make_base()
        # 약간의 random noise 만
        arr = np.array(img)
        noise = np.random.normal(0, 8, arr.shape).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    if main in {"BrokenRing","CenterCircle","CenterDonut","CrescentArc","CrossScratch",
                "DiagonalSmear","ParallelScratches","RingDots","Row","Starburst"}:
        # canvas
        color = (180, 180, 100)
        return render_canvas(cls, color)

    # main + sub
    img = make_base()
    draw = ImageDraw.Draw(img)
    main_color = color_map.get(main, (200, 100, 100))
    add_main_region(draw, main, main_color)
    if sub:
        sub_color = (240, 240, 240)
        add_sub_pattern(draw, main, sub, sub_color)
    # noise
    arr = np.array(img)
    noise = np.random.normal(0, 4, arr.shape).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def main():
    out = resolve_path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for cls in CLASSES:
        sub = out / cls
        sub.mkdir(exist_ok=True)
        n = N_NORMAL if cls == "Normal" else N_PER_CLASS
        existing = len(list(sub.glob("*.png")))
        if existing >= n:
            print(f"[skip] {cls}: already has {existing} ≥ {n}")
            total += existing
            continue
        print(f"[gen ] {cls}: {existing}/{n}", flush=True)
        for i in range(existing, n):
            img = render_class(cls, seed=SEED + hash(cls) % 10000 + i)
            img.save(sub / f"wafer_{i:04d}.png", optimize=False)
            total += 1
        if (CLASSES.index(cls) + 1) % 5 == 0:
            print(f"  ({CLASSES.index(cls) + 1}/{len(CLASSES)} classes done)")
    print(f"\n[OUT] {out.resolve()}")
    print(f"  total images: {total}")
    print(f"  classes: {len(CLASSES)}")
    print(f"\n다음 단계:")
    print(f"  python scripts/_split_data.py    # CNN/Contrastive train/eval 분리")
    print(f"  python scripts/train_pipeline.py # 학습")


if __name__ == "__main__":
    main()
