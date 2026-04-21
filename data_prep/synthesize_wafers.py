"""WM-811K 9 class 결함 패턴을 프로토타입으로 fail-map 포맷 4000×4000 합성.

WM-811K 원본 wafer는 매우 noisy하다: 특정 "Center" 라벨 wafer 도 중앙 cluster
뿐 아니라 normal 영역 전반에 scatter 된 개별 결함과 grade 변동이 깔려있다.
이 스크립트는 그 texture 를 모사한다.

Composition (모든 class 공통)
----------------------------
1. **Global scatter layer**: chip 영역 전체에 low-amplitude uniform noise 로
   naturally 발생하는 개별 결함/경계 die 재현.
2. **Random hot micro-spots**: 수백~수천 개의 작은 국소 bump (반경 3-12 px,
   intensity 0.3-0.9) 를 chip 안에 흩뿌림. 이게 WM-811K 의 "여기저기 결함"
   texture 를 만든다.
3. **Class pattern layer**: 해당 class 의 특징적 defect 분포 (center 집중,
   donut 링 등) 를 위에 더함.
4. **Density → grade 연속 매핑**: threshold 없이 density ∈ [0,1] 을 8-bin
   으로 직접 매핑해 grade 0..7 을 자연스럽게 섞음. 0.05 같은 약한 신호도
   grade 1 로 살아남는다.

WM-811K 원본에 들어있는 noise rate 를 class 별로 다르게 세팅:
  - Random / Near-full : 강함
  - Center / Donut / Edge-Ring / Edge-Loc / Loc / Scratch : 중간
  - none : 약함
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.palette_io import get_failmap_palette  # noqa: E402

CLASSES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none",
]


def _class_token(cls: str) -> str:
    return cls.replace("-", "").replace("_", "").upper()


# ---------------------------------------------------------------------------
# Class-specific pattern densities (layer 3).
# 모두 (H, W) float32 in [0, 1] 반환.
# ---------------------------------------------------------------------------
def pat_center(rng, H, W, cx, cy, R) -> np.ndarray:
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
    cluster_r = rng.uniform(R * 0.15, R * 0.35)
    return np.clip(1.0 - r / cluster_r, 0.0, 1.0).astype(np.float32)


def pat_donut(rng, H, W, cx, cy, R) -> np.ndarray:
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
    ring_r = rng.uniform(R * 0.4, R * 0.65)
    ring_w = rng.uniform(R * 0.08, R * 0.15)
    return np.exp(-(((r - ring_r) / ring_w) ** 2)).astype(np.float32)


def pat_edge_ring(rng, H, W, cx, cy, R) -> np.ndarray:
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)
    ring_r = rng.uniform(R * 0.85, R * 0.97)
    ring_w = rng.uniform(R * 0.04, R * 0.08)
    return np.exp(-(((r - ring_r) / ring_w) ** 2)).astype(np.float32)


def pat_edge_loc(rng, H, W, cx, cy, R) -> np.ndarray:
    angle = rng.uniform(0, 2 * np.pi)
    dist = rng.uniform(R * 0.7, R * 0.92)
    px = cx + dist * np.cos(angle)
    py = cy + dist * np.sin(angle)
    patch_r = rng.uniform(R * 0.1, R * 0.25)
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((xx - px) ** 2 + (yy - py) ** 2).astype(np.float32)
    return np.exp(-((r / patch_r) ** 2)).astype(np.float32)


def pat_loc(rng, H, W, cx, cy, R) -> np.ndarray:
    angle = rng.uniform(0, 2 * np.pi)
    dist = rng.uniform(R * 0.15, R * 0.55)
    px = cx + dist * np.cos(angle)
    py = cy + dist * np.sin(angle)
    patch_r = rng.uniform(R * 0.08, R * 0.2)
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((xx - px) ** 2 + (yy - py) ** 2).astype(np.float32)
    return np.exp(-((r / patch_r) ** 2)).astype(np.float32)


def pat_scratch(rng, H, W, cx, cy, R) -> np.ndarray:
    angle = rng.uniform(0, np.pi)
    offset = rng.uniform(-R * 0.4, R * 0.4)
    line_w = rng.uniform(R * 0.008, R * 0.02)
    length = rng.uniform(R * 0.5, R * 1.6)
    perp_x = -np.sin(angle)
    perp_y = np.cos(angle)
    line_cx = cx + offset * perp_x
    line_cy = cy + offset * perp_y
    yy, xx = np.ogrid[:H, :W]
    d_perp = np.abs((xx - line_cx) * perp_x + (yy - line_cy) * perp_y).astype(np.float32)
    d_par = np.abs((xx - line_cx) * np.cos(angle) + (yy - line_cy) * np.sin(angle)).astype(np.float32)
    return (np.exp(-((d_perp / line_w) ** 2)) * (d_par < length / 2)).astype(np.float32)


def pat_random(rng, H, W, cx, cy, R) -> np.ndarray:
    # Random 은 패턴 자체가 "산발". 약한 균일 bias + 중간-강도 hot spots.
    d = np.full((H, W), 0.15, dtype=np.float32)
    return d


def pat_near_full(rng, H, W, cx, cy, R) -> np.ndarray:
    base = rng.uniform(0.65, 0.85)
    return np.full((H, W), base, dtype=np.float32)


def pat_none(rng, H, W, cx, cy, R) -> np.ndarray:
    return np.zeros((H, W), dtype=np.float32)


PATTERN_FNS = {
    "Center": pat_center,
    "Donut": pat_donut,
    "Edge-Loc": pat_edge_loc,
    "Edge-Ring": pat_edge_ring,
    "Loc": pat_loc,
    "Near-full": pat_near_full,
    "Random": pat_random,
    "Scratch": pat_scratch,
    "none": pat_none,
}

# Scatter / noise profile per class (base amplitude of global noise layer +
# number and intensity of micro hot-spots).
NOISE_PROFILE = {
    "Center":    dict(noise=0.08, n_spots=(400, 900),   spot_r=(3, 10), spot_amp=(0.25, 0.6)),
    "Donut":     dict(noise=0.08, n_spots=(400, 900),   spot_r=(3, 10), spot_amp=(0.25, 0.6)),
    "Edge-Loc":  dict(noise=0.07, n_spots=(350, 800),   spot_r=(3, 10), spot_amp=(0.25, 0.6)),
    "Edge-Ring": dict(noise=0.08, n_spots=(500, 1100),  spot_r=(3, 10), spot_amp=(0.25, 0.6)),
    "Loc":       dict(noise=0.07, n_spots=(350, 800),   spot_r=(3, 10), spot_amp=(0.25, 0.6)),
    "Near-full": dict(noise=0.15, n_spots=(1500, 3500), spot_r=(3, 14), spot_amp=(0.40, 0.8)),
    "Random":    dict(noise=0.12, n_spots=(600, 1400),  spot_r=(2, 9),  spot_amp=(0.25, 0.55)),
    "Scratch":   dict(noise=0.07, n_spots=(300, 700),   spot_r=(3, 10), spot_amp=(0.25, 0.6)),
    "none":      dict(noise=0.04, n_spots=(50, 200),    spot_r=(2, 6),  spot_amp=(0.20, 0.45)),
}


def _scatter_spots(rng, H, W, n, r_range, amp_range) -> np.ndarray:
    """Fast vectorized micro hot-spot layer. 각 spot 은 box (정사각형) 로 근사."""
    canvas = np.zeros((H, W), dtype=np.float32)
    ys = rng.integers(0, H, size=n)
    xs = rng.integers(0, W, size=n)
    rs = rng.integers(r_range[0], r_range[1] + 1, size=n)
    amps = rng.uniform(amp_range[0], amp_range[1], size=n).astype(np.float32)
    for y, x, r, a in zip(ys, xs, rs, amps):
        y0 = max(0, int(y) - int(r))
        y1 = min(H, int(y) + int(r) + 1)
        x0 = max(0, int(x) - int(r))
        x1 = min(W, int(x) + int(r) + 1)
        canvas[y0:y1, x0:x1] += a
    return canvas


def density_to_grade(density: np.ndarray) -> np.ndarray:
    """Continuous density [0,1] -> palette grade 0..7 with overlapping bins:
        density * 8.0 을 floor + clip. 값 0.125 마다 grade 1 씩 상승.
    """
    g = np.clip((density * 8.0).astype(np.int32), 0, 7)
    return g.astype(np.uint8)


def synthesize(class_name: str, rng: np.random.Generator, H: int, W: int) -> np.ndarray:
    # Wafer geometry — small jitter per sample
    cx = W // 2 + int(rng.integers(-W // 80, W // 80))
    cy = H // 2 + int(rng.integers(-H // 80, H // 80))
    R = int(min(H, W) * rng.uniform(0.46, 0.49))

    yy, xx = np.ogrid[:H, :W]
    inside = ((xx - cx) ** 2 + (yy - cy) ** 2) <= R ** 2

    # Layer 1: global uniform noise + layer 2: micro hot-spots
    prof = NOISE_PROFILE[class_name]
    noise = rng.uniform(0.0, prof["noise"], size=(H, W)).astype(np.float32)
    n_spots = int(rng.integers(prof["n_spots"][0], prof["n_spots"][1] + 1))
    spots = _scatter_spots(rng, H, W, n_spots, prof["spot_r"], prof["spot_amp"])

    # Layer 3: class-specific pattern
    pattern = PATTERN_FNS[class_name](rng, H, W, cx, cy, R)

    # Blend layers (pattern weight varies slightly per wafer for variety)
    pattern_w = rng.uniform(0.75, 1.0)
    density = noise + spots + pattern_w * pattern

    # Cell-level Gaussian perturbation so no region is perfectly uniform
    density += rng.normal(0.0, 0.04, size=(H, W)).astype(np.float32)
    density = np.clip(density, 0.0, 1.0)

    # Map density to grade 0..7
    grade = density_to_grade(density)

    # Outside wafer -> 31 (transparent)
    out = np.full((H, W), 31, dtype=np.uint8)
    out[inside] = grade[inside]
    return out


def _filename_for(cls: str, idx: int, base_dt: datetime) -> str:
    tok = _class_token(cls)
    t = base_dt + timedelta(seconds=idx + 1)
    return f"{tok}_00P_W{idx:03d}_{t.strftime('%Y%m%d')}_{t.strftime('%H%M%S')}.png"


def save_palette_png(idx: np.ndarray, path: Path, palette: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(idx, mode="P")
    img.putpalette(palette)
    img.save(path, transparency=31, optimize=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/wm811k")
    ap.add_argument("--size", type=int, default=4000)
    ap.add_argument("--n-per-class", type=int, default=222)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base-date", default="20260420")
    ap.add_argument("--classes", nargs="*", default=None,
                    help="class 부분집합만 생성 (default: 전체 9개)")
    args = ap.parse_args()

    H = W = args.size
    palette = get_failmap_palette()
    out_root = Path(args.out)
    base_dt = datetime.strptime(args.base_date + "_000000", "%Y%m%d_%H%M%S")
    classes = args.classes if args.classes else CLASSES

    summary: dict[str, int] = {}
    t_start = time.time()
    total_done = 0

    for cls_i, cls in enumerate(classes):
        rng = np.random.default_rng(args.seed + cls_i * 10_000)
        cls_t0 = time.time()
        for i in range(args.n_per_class):
            try:
                arr = synthesize(cls, rng, H, W)
                fname = _filename_for(cls, i, base_dt)
                save_palette_png(arr, out_root / cls / fname, palette)
                total_done += 1
            except Exception as e:
                print(f"[skip] {cls}/{i}: {e}")
        summary[cls] = args.n_per_class
        dt = time.time() - cls_t0
        print(f"[{cls:10s}] synth {args.n_per_class} @ {H}x{W} in {dt:.1f}s")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    elapsed = time.time() - t_start
    print(f"\n[done] total={total_done} @ {H}x{W} in {elapsed:.0f}s")
    print(f"[out]  {out_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
