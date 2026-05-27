"""7 obj-less wafer-canvas — 단순 확률 분포 방식.

핵심 spec (사용자 명시):
- alpha(x, y) = P(defect pixel | position) 확률 함수
- 가운데 (line center) 매우 sharp peak, 좌우 양끝 자연 0 → wafer normal area 와 동일
- 단일 sharp Gaussian (제곱 + heavy core) — shoulder/plateau 없음, 자연 fade
- pixel 별 random sample: rng.uniform() < alpha → defect grade, 그 외 baseline

이전 round 폐기 사항:
- _perp_5stage (strong+med+weak weighted sum) → shoulder 만들음
- segment_strengths / linear interp / perp_offsets → 분절/wave 인위적
- 3-way grade mix → wide elevated rectangle 영역 만듦
- chip border BIN-color (default) → 네모 격자 강조

새 spec:
- _perp = exp(-d²/σ²) 단일 Gaussian sharp (사용자 "Gaussian 제곱")
- alpha 자체가 P(defect pixel) — direct sampling
- chip border = defect pixel 비율 높은 chip 만 BIN

7 class:
  DiagonalSmear, CrossScratch, CrescentArc, SpiralTrail,
  ParallelScratches, EdgeSmudge, BrokenRing
"""
import os, json
import numpy as np
from PIL import Image

try:
    from _sample_gen import (
        PALETTE, KEY_TO_INDEX, IDX_BG,
        BIN_TO_BORDER_IDX, DEFECT_BIN_POOL, DEFECT_BIN_WEIGHTS,
        SIZE, GRID, CHIP, PNG_OUT_DIR, JSON_OUT_DIR,
        CUM_BASE, CUM_BASELINE_TIERS, INTENSITY_ALPHA_SCALE,
        pick_baseline_tier, pick_intensity_tier,
        rand_prefix, LT_OPTIONS, TM_OPTIONS,
    )
    from _fq_metadata import add_synthetic_fq_to_json
except ImportError:
    from _sample_gen import (
        PALETTE, KEY_TO_INDEX, IDX_BG,
        BIN_TO_BORDER_IDX, DEFECT_BIN_POOL, DEFECT_BIN_WEIGHTS,
        SIZE, GRID, CHIP, PNG_OUT_DIR, JSON_OUT_DIR,
        CUM_BASE, CUM_BASELINE_TIERS, INTENSITY_ALPHA_SCALE,
        pick_baseline_tier, pick_intensity_tier,
        rand_prefix, LT_OPTIONS, TM_OPTIONS,
    )
    from _fq_metadata import add_synthetic_fq_to_json

CANVAS_CLASSES = [
    "DiagonalSmear", "CrossScratch", "CrescentArc",
    "ParallelScratches", "BrokenRing",                                  # EdgeSmudge 제거 (CrescentArc 겹침)
    "RingDots", "CenterDonut", "Row", "Starburst",
    "CenterCircle",                                                     # 추가: 가운데 solid disk (CenterDonut 의 채워진 버전)
]

# class 별 grade scale (alpha → mean grade 매핑)
# alpha 0 → grade 0, alpha 1 → grade max_g. Gaussian sampling 으로 자연 연속 분포 — boundary X.
# class 마다 max_g 다르게 (어떤 class 약함, 어떤 class 강함) → class 별 grade 색 다양성.
# 사용자: 대부분 grade 1, 2 만. 일부 class 만 더 강 (3+). soft mapping.
# peak alpha 0.30-0.50 → center grade ≈ 0.4 * max_g.
# 사용자 spec: alpha = baseline ↔ peak distribution mix weight.
#   alpha 0 → pure baseline (normal): P(0)=0.83, P(1)=0.15, P(2+)=0.02 → wafer normal 과 동일
#   alpha 0.3 → 70% baseline + 30% peak: P(0) 감소, P(1) 증가, P(2) 등장
#   alpha 1.0 → pure peak: class 별 다양 grade
#
# class 별 PEAK 분포 — peak alpha 일 때 grade dist
CLASS_PEAK_DIST = {
    # idx:                [    0,    1,    2,    3,    4,    5,    6,    7  ]
    "DiagonalSmear":     [0.00, 0.60, 0.28, 0.10, 0.02, 0.00, 0.00, 0.00],   # peak 1 위주, 일부 2-3
    "CrossScratch":      [0.00, 0.40, 0.30, 0.18, 0.08, 0.03, 0.01, 0.00],   # 1-3 mix, 일부 4-5
    "CrescentArc":       [0.00, 0.50, 0.30, 0.15, 0.05, 0.00, 0.00, 0.00],   # 1-3
    "ParallelScratches": [0.00, 0.30, 0.28, 0.20, 0.12, 0.07, 0.03, 0.00],   # 1-5 wide
    "BrokenRing":        [0.00, 0.20, 0.22, 0.20, 0.18, 0.10, 0.07, 0.03],   # 강, 1-6 wide
    "RingDots":          [0.00, 0.45, 0.30, 0.15, 0.07, 0.03, 0.00, 0.00],   # ring 위 dot, 1-3
    "CenterDonut":       [0.00, 0.50, 0.30, 0.13, 0.05, 0.02, 0.00, 0.00],   # 얇은 center donut, 1-3
    "Row":               [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],   # grade 7 만 (선명)
    "Starburst":         [0.00, 0.40, 0.28, 0.18, 0.10, 0.03, 0.01, 0.00],   # center spot 강 + radial line
    "CenterCircle":      [0.00, 0.42, 0.30, 0.16, 0.08, 0.03, 0.01, 0.00],   # solid center disk, 1-4 mix
}
def _build_peak_cum(class_name):
    d = np.array(CLASS_PEAK_DIST[class_name], dtype=np.float64)
    d /= d.sum()
    return np.cumsum(d).astype(np.float32)
CUM_BASE_F = CUM_BASE.astype(np.float32)


def _perp_sharp(d, sigma):
    """Sharp narrow Lorentzian peak + wide heavy Lorentzian tail.
    sharp σ_s=0.5σ (매우 좁고 날카) + wide σ_w=5σ weight 0.6 (크고 넓은 tail).
    d=0: 1.0(clip) | σ:0.78 | 2σ:0.58 | 4σ:0.40 | 8σ:0.18 | 15σ:0.06 | 25σ:0.024."""
    sharp = 1.0 / (1.0 + (d/(0.5*sigma))**2)            # 좁고 날카로운 peak
    wide  = 0.60 / (1.0 + (d/(5*sigma))**2)             # 크고 넓은 tail
    return np.minimum(sharp + wide, 1.0).astype(np.float32)


def _yy_xx_full():
    yy = np.arange(SIZE, dtype=np.float32)[:, None]
    xx = np.arange(SIZE, dtype=np.float32)[None, :]
    return yy, xx


# ===================== 7 alpha 함수 (= P(defect pixel | position) 분포) =====================

def _along_taper(d_along, half_len, sigma_end):
    """along-line direction taper — line center 1.0, 양 끝 부드럽게 0 fade.
    half_len 안 = 1.0, 외쪽 fade Gaussian."""
    a = np.ones_like(d_along, dtype=np.float32)
    abs_d = np.abs(d_along)
    fade = np.where(abs_d > half_len,
                    np.exp(-((abs_d - half_len)**2) / (sigma_end*sigma_end)),
                    1.0).astype(np.float32)
    return fade


def alpha_diagonal_smear(rng):
    """Diagonal line — 길이 partial (full diameter X), 양 끝 자연 fade."""
    yy, xx = _yy_xx_full()
    angle = (45.0 + rng.uniform(-5, 5)) * np.pi / 180
    cy = SIZE/2 + rng.uniform(-CHIP*0.3, CHIP*0.3)
    cx = SIZE/2 + rng.uniform(-CHIP*0.3, CHIP*0.3)
    sigma = CHIP * rng.uniform(0.10, 0.20)
    peak = rng.uniform(0.18, 0.50)                        # round 29 v15: lower bound 낮춤 (weak peak 가능)
    half_len = SIZE * rng.uniform(0.20, 0.35)             # line 길이 짧게 — partial
    sigma_end = CHIP * rng.uniform(2.0, 4.0)              # 양 끝 fade 영역
    cos_a = float(np.cos(angle)); sin_a = float(np.sin(angle))
    d_perp = cos_a*(yy - cy) - sin_a*(xx - cx)
    d_along = sin_a*(yy - cy) + cos_a*(xx - cx)
    return peak * _perp_sharp(d_perp, sigma) * _along_taper(d_along, half_len, sigma_end)


def alpha_cross_scratch(rng):
    """Cross — line 길이 partial, 양 끝 fade."""
    yy, xx = _yy_xx_full()
    cy = SIZE/2 + rng.uniform(-CHIP*0.3, CHIP*0.3)
    cx = SIZE/2 + rng.uniform(-CHIP*0.3, CHIP*0.3)
    base_angle = rng.uniform(-0.04, 0.04)
    sigma = CHIP * rng.uniform(0.10, 0.20)
    peak = rng.uniform(0.18, 0.50)                        # round 29 v15: lower bound
    half_len = SIZE * rng.uniform(0.20, 0.32)
    sigma_end = CHIP * rng.uniform(2.0, 3.5)
    a_full = np.zeros((SIZE, SIZE), dtype=np.float32)
    for ang in [base_angle, base_angle + np.pi/2]:
        cos_a = float(np.cos(ang)); sin_a = float(np.sin(ang))
        d_perp = cos_a*(yy - cy) - sin_a*(xx - cx)
        d_along = sin_a*(yy - cy) + cos_a*(xx - cx)
        a = peak * _perp_sharp(d_perp, sigma) * _along_taper(d_along, half_len, sigma_end)
        np.maximum(a_full, a, out=a_full)
    return a_full


def alpha_crescent_arc(rng):
    """1/4-1/3 arc — radial Gaussian × angular smooth fade."""
    yy, xx = _yy_xx_full()
    cy, cx = SIZE/2, SIZE/2
    R = SIZE/2
    r_center = R * rng.uniform(0.78, 0.88)
    sigma_r = R * rng.uniform(0.008, 0.016)
    # arc 위치 고정-ish: bottom 만 (visual identity)
    side = -np.pi / 2                                      # bottom
    arc_len = rng.uniform(np.pi/3, np.pi/2.5)
    th_center = side + rng.uniform(-0.15, 0.15)
    sigma_th = arc_len / 2.5                            # smooth angular taper
    peak = rng.uniform(0.18, 0.50)                      # round 29 v15: lower bound
    dy = yy - cy; dx = xx - cx
    r = np.sqrt(dy*dy + dx*dx)
    theta = np.arctan2(dy, dx)
    delta = (theta - th_center + np.pi) % (2*np.pi) - np.pi
    a_r = _perp_sharp(r - r_center, sigma_r)
    a_th = np.exp(-delta*delta / (sigma_th*sigma_th)).astype(np.float32)
    return peak * a_r * a_th


def alpha_spiral_trail(rng):
    """Archimedean spiral — perp distance Gaussian sharp."""
    yy, xx = _yy_xx_full()
    cy, cx = SIZE/2, SIZE/2
    direction = float(rng.choice([1, -1]))
    r0 = CHIP * rng.uniform(1.5, 3.0)
    pitch = CHIP * rng.uniform(1.5, 2.5) / (2*np.pi)
    n_turns = rng.uniform(1.5, 3.0)
    sigma = CHIP * rng.uniform(0.10, 0.20)
    peak = rng.uniform(0.30, 0.50)
    n_pts = int(150 * n_turns)
    t = np.linspace(0, 2*np.pi*n_turns, n_pts).astype(np.float32)
    sr = r0 + pitch * t
    sx = (cx + sr * np.cos(direction * t)).astype(np.float32)
    sy = (cy + sr * np.sin(direction * t)).astype(np.float32)
    a_full = np.zeros((SIZE, SIZE), dtype=np.float32)
    chunk_h = 128
    for y0 in range(0, SIZE, chunk_h):
        y1 = min(y0 + chunk_h, SIZE)
        ys = yy[y0:y1, :]
        d2 = (sy[:, None, None] - ys[None, :, :])**2 + (sx[:, None, None] - xx[None, :, :])**2
        d_min = np.sqrt(d2.min(axis=0))
        del d2
        a_full[y0:y1, :] = peak * _perp_sharp(d_min, sigma)
    return a_full


def alpha_parallel_scratches(rng):
    """3-5 parallel — line 길이 partial."""
    yy, xx = _yy_xx_full()
    angle = (45.0 + rng.uniform(-5, 5)) * np.pi / 180
    n_lines = int(rng.integers(3, 6))
    spacing = CHIP * rng.uniform(3.5, 5.5)
    sigma = CHIP * rng.uniform(0.10, 0.20)
    peak = rng.uniform(0.18, 0.50)                        # round 29 v15: lower bound
    half_len = SIZE * rng.uniform(0.20, 0.32)
    sigma_end = CHIP * rng.uniform(2.0, 3.5)
    perp_y = -np.sin(angle); perp_x = np.cos(angle)
    cos_a = float(np.cos(angle)); sin_a = float(np.sin(angle))
    a_full = np.zeros((SIZE, SIZE), dtype=np.float32)
    for i in range(n_lines):
        offset = (i - (n_lines-1)/2) * spacing
        cy_i = SIZE/2 + offset * perp_y
        cx_i = SIZE/2 + offset * perp_x
        d_perp = cos_a*(yy - cy_i) - sin_a*(xx - cx_i)
        d_along = sin_a*(yy - cy_i) + cos_a*(xx - cx_i)
        a = peak * _perp_sharp(d_perp, sigma) * _along_taper(d_along, half_len, sigma_end)
        np.maximum(a_full, a, out=a_full)
    return a_full


def alpha_edge_smudge(rng):
    """Wide annular near edge — radial sharp Gaussian × angular sector or full."""
    yy, xx = _yy_xx_full()
    cy, cx = SIZE/2, SIZE/2
    R = SIZE/2
    r_center = R * rng.uniform(0.75, 0.85)              # ring near edge
    sigma_r = R * rng.uniform(0.025, 0.045)             # narrower ring
    half_ring = True                                       # 모두 half-ring (visual identity 유지)
    peak = rng.uniform(0.22, 0.60)                         # round 29 v15: lower bound
    dy = yy - cy; dx = xx - cx
    r = np.sqrt(dy*dy + dx*dx)
    a_r = _perp_sharp(r - r_center, sigma_r)
    if half_ring:
        theta = np.arctan2(dy, dx)
        # bottom-half 고정 (visual identity)
        theta_center = -np.pi/2 + rng.uniform(-0.15, 0.15)
        delta = (theta - theta_center + np.pi) % (2*np.pi) - np.pi
        sigma_th = rng.uniform(0.7, 1.2)
        a_th = np.exp(-delta*delta / (sigma_th*sigma_th)).astype(np.float32)
        return peak * a_r * a_th
    return peak * a_r


def alpha_broken_ring(rng):
    """Annular ring × Gaussian gap dips — sharp peak."""
    yy, xx = _yy_xx_full()
    cy, cx = SIZE/2, SIZE/2
    R = SIZE/2
    r_center = R * rng.uniform(0.85, 0.93)
    sigma_r = R * rng.uniform(0.007, 0.014)
    n_gaps = int(rng.integers(1, 4))
    gap_centers = rng.uniform(-np.pi, np.pi, size=n_gaps).astype(np.float32)
    gap_widths = rng.uniform(np.pi/6, np.pi/3, size=n_gaps).astype(np.float32)
    peak = rng.uniform(0.18, 0.50)                          # round 29 v15: lower bound
    dy = yy - cy; dx = xx - cx
    r = np.sqrt(dy*dy + dx*dx)
    theta = np.arctan2(dy, dx)
    a = _perp_sharp(r - r_center, sigma_r)
    gap_mask = np.ones_like(a)
    for gc, gw in zip(gap_centers, gap_widths):
        delta = (theta - float(gc) + np.pi) % (2*np.pi) - np.pi
        sigma_g = float(gw) / 3.0
        dip = np.exp(-delta*delta / (sigma_g*sigma_g)).astype(np.float32)
        # 사용자: "완전 결핍 X, noise 로 듬성듬성" — gap 영역 도 alpha 의 ~20% 유지
        gap_mask *= (1.0 - 0.80 * dip)
    return peak * a * gap_mask


def alpha_ring_dots(rng):
    """ring 위 N angular dot — 260507 v5.2: dot 위치 고정 (사용자 directive
    "ring dots 는 dot들 위치가 바뀌면안된다 동일 반지름에 동일 위치").
    r_center / n_dots / th_off 모두 fixed. peak (밝기) 만 wafer 별 random 유지 + pink noise baseline 자연 variation.
    """
    yy, xx = _yy_xx_full()
    cy, cx = SIZE/2, SIZE/2
    R = SIZE/2
    # ★ FIXED — 모든 wafer 동일 위치, 동일 반지름
    r_center = R * 0.55
    n_dots = 18
    th_off = 0.0
    sigma_blob = CHIP * 0.30                             # FIXED dot 크기 (chip 의 30%)
    peak = rng.uniform(0.40, 0.60)                       # 밝기만 random — floor 0.30 너무 약했음 (seed2 invisible)
    a_full = np.zeros((SIZE, SIZE), dtype=np.float32)
    for i in range(n_dots):
        th = th_off + 2*np.pi * i / n_dots
        cy_d = cy + r_center * np.sin(th)
        cx_d = cx + r_center * np.cos(th)
        d = np.sqrt((yy - cy_d)**2 + (xx - cx_d)**2).astype(np.float32)
        blob = peak * _perp_sharp(d, sigma_blob)
        np.maximum(a_full, blob, out=a_full)
    return a_full


def alpha_starburst(rng):
    """center small filled spot + N 개 radial ray (외부로 뻗는 sharp line)."""
    yy, xx = _yy_xx_full()
    cy, cx = SIZE/2, SIZE/2
    R = SIZE/2
    dy = yy - cy; dx = xx - cx
    r = np.sqrt(dy*dy + dx*dx)
    theta_pix = np.arctan2(dy, dx)
    # center 작은 빈 원 (ring/donut) — filled X
    r_ring = R * rng.uniform(0.025, 0.055)              # 작은 ring radius
    sigma_ring = R * rng.uniform(0.005, 0.010)          # 매우 얇은 ring
    a_center = _perp_sharp(r - r_ring, sigma_ring)
    # radial rays N — 모두 같은 길이 + 매우 가는 line
    n_rays = int(rng.integers(8, 14))
    th_off = rng.uniform(0, 2*np.pi)
    sigma_perp = CHIP * rng.uniform(0.05, 0.10)                          # 훨 작게
    ray_len = R * rng.uniform(0.55, 0.75)                                # 모든 ray 같은 길이
    a_lines = np.zeros((SIZE, SIZE), dtype=np.float32)
    for i in range(n_rays):
        ray_angle = th_off + 2*np.pi * i / n_rays + rng.uniform(-0.05, 0.05)
        dth = (theta_pix - ray_angle + np.pi) % (2*np.pi) - np.pi
        d_perp = r * np.sin(dth)
        forward = np.cos(dth) > 0
        in_len = (r < ray_len) & forward
        a_ray = _perp_sharp(d_perp, sigma_perp) * in_len.astype(np.float32)
        np.maximum(a_lines, a_ray, out=a_lines)
    peak_c = rng.uniform(0.40, 0.85)                      # round 29 v15: lower bound
    peak_l = rng.uniform(0.18, 0.50)                      # round 29 v15: lower bound
    return np.maximum(peak_c * a_center, peak_l * a_lines)


def alpha_row(rng):
    """direct line draw — 확률 분포 X, 1-2 px thin line, 매우 작은 크기 (사용자 spec).

    핵심 spec (사용자 명시):
    - PIL ImageDraw.line 직접 (alpha-based stochastic X)
    - **y 값 strict 동일, x 값만 변함** (round 27: angle = 0 fix per user)
    - line_len = chip * uniform(1/6, 1/4) — chip 1/6 ~ 1/4 매우 짧음
    - width 1-2 px (매우 얇음)
    - n_lines 40-69 random scatter inside wafer
    - alpha = 1.0 binary (line pixel) → grade 7 dominant (CLASS_PEAK_DIST)
    - chip border decision: alpha mean 작아도 alpha max 1.0 → max bonus 적용
    """
    from PIL import Image as _Image, ImageDraw as _ImageDraw
    cy, cx = SIZE/2, SIZE/2
    R = SIZE/2
    img = _Image.new('L', (SIZE, SIZE), 0)
    draw = _ImageDraw.Draw(img)
    n_lines = int(rng.integers(40, 70))                       # 갯수 늘림 (사용자 spec)
    for _ in range(n_lines):
        while True:
            ly = rng.uniform(-R*0.85, R*0.85) + cy
            lx = rng.uniform(-R*0.85, R*0.85) + cx
            if (ly-cy)**2 + (lx-cx)**2 < (R*0.85)**2:
                break
        line_len = CHIP * rng.uniform(1/6, 1/4)              # chip 1/6 ~ 1/4 (사용자 spec)
        # angle = 0 strict horizontal (round 27 user spec: y 동일, x 만 변함)
        x0 = lx - line_len/2
        x1 = lx + line_len/2
        y0 = ly
        y1 = ly
        width = int(rng.integers(1, 3))                       # 1-2 px thin
        draw.line([(x0, y0), (x1, y1)], fill=255, width=width)
    arr = (np.array(img) > 0).astype(np.float32)              # binary line mask
    return arr                                                 # alpha = 1.0 on line, 0 elsewhere


def alpha_center_donut(rng):
    """가운데 얇은 donut — small ring near center."""
    yy, xx = _yy_xx_full()
    cy, cx = SIZE/2, SIZE/2
    R = SIZE/2
    r_center = R * rng.uniform(0.08, 0.16)               # center 매우 작은 ring
    sigma_r = R * rng.uniform(0.006, 0.014)              # 매우 얇은 ring
    peak = rng.uniform(0.18, 0.50)                       # round 29 v15: lower bound
    dy = yy - cy; dx = xx - cx
    r = np.sqrt(dy*dy + dx*dx)
    return peak * _perp_sharp(r - r_center, sigma_r)


def alpha_center_circle(rng):
    """center round patch — soft gradient core, irregular boundary, no sharp edge.

    Round 28 (사용자 feedback):
    - ❌ 컴퍼스 같은 완벽한 원 (옛: r = √(dy²+dx²) only)
      → ✅ angular harmonics 3-5개 (1/k decreasing amp) 로 boundary irregular
    - ❌ 영역 딱 끝남 (옛: inside boolean + narrow Gaussian σ=0.008-0.020 R)
      → ✅ Gaussian sharp + Lorentzian heavy-tail 합 — boundary 없는 smooth gradient
    - ❌ 안쪽 단순 채워짐 (옛: inside = 1.0 binary)
      → ✅ profile(r_eff) center peak → 외곽 falloff continuous gradient
    """
    yy, xx = _yy_xx_full()
    cy, cx = SIZE/2, SIZE/2
    R = SIZE/2
    dy = yy - cy; dx = xx - cx
    r = np.sqrt(dy*dy + dx*dx)
    theta = np.arctan2(dy, dx)

    # angular perturbation — 3-5 harmonics, decreasing amp (1/k)
    n_harm = int(rng.integers(3, 6))
    angular = np.zeros_like(r)
    for k in range(1, n_harm + 1):
        amp = rng.uniform(0.06, 0.14) / k
        ph = rng.uniform(0, 2 * np.pi)
        angular += amp * np.sin(k * theta + ph)

    # radial scales — sharp + wide heavy-tail
    base = R * rng.uniform(0.15, 0.22)
    sigma_sharp = base * rng.uniform(0.7, 1.0)
    sigma_wide  = base * rng.uniform(2.0, 3.0)

    # angular-perturbed effective radius (1 ± ~0.15)
    r_eff = r * (1.0 + angular)

    profile = (
        rng.uniform(0.55, 0.70) * np.exp(- (r_eff / sigma_sharp) ** 2) +     # sharp center
        rng.uniform(0.25, 0.40) / (1.0 + (r_eff / sigma_wide) ** 2)            # Lorentzian halo
    )
    peak = rng.uniform(0.25, 0.65)                                             # round 29 v15: lower bound
    return (peak * profile).astype(np.float32)


ALPHA_FN = {
    "DiagonalSmear":     alpha_diagonal_smear,
    "CrossScratch":      alpha_cross_scratch,
    "CrescentArc":       alpha_crescent_arc,
    "ParallelScratches": alpha_parallel_scratches,
    "BrokenRing":        alpha_broken_ring,
    "RingDots":          alpha_ring_dots,
    "CenterDonut":       alpha_center_donut,
    "Row":               alpha_row,
    "Starburst":         alpha_starburst,
    "CenterCircle":      alpha_center_circle,
}


# ===================== Main render =====================

def _canvas_wafer_inside_mask():
    centers = (np.arange(GRID, dtype=np.float32) + 0.5) * CHIP
    yy, xx = np.meshgrid(centers, centers, indexing='ij')
    cy = cx = SIZE / 2.0
    r = SIZE / 2.0
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= (r ** 2)


def render_canvas_in_memory(class_name, seed):
    """In-memory variant of render_canvas: no PNG/JSON file writes.
    Returns (canvas np.uint8 array SIZE x SIZE, chip_meta dict, PALETTE bytes).
    Used by mega_matrix.ood_chips for OOD chip synthesis without temp wafer files."""
    rng = np.random.default_rng(seed)
    inside = _canvas_wafer_inside_mask()
    inside_pix = np.repeat(np.repeat(inside, CHIP, axis=0), CHIP, axis=1)
    baseline_tier = pick_baseline_tier(rng)
    intensity_tier = pick_intensity_tier(rng)
    canvas_tau = float(rng.choice([0.20, 0.30, 0.40], p=[0.25, 0.50, 0.25]))
    cum_base_use = CUM_BASELINE_TIERS[baseline_tier].astype(np.float32)
    alpha = ALPHA_FN[class_name](rng) * INTENSITY_ALPHA_SCALE[intensity_tier]

    def _bilinear_field(rng, h, w, lo, hi):
        coarse = rng.uniform(lo, hi, size=(h, w)).astype(np.float32)
        yi = np.linspace(0, h - 1, SIZE, dtype=np.float32)
        xi = np.linspace(0, w - 1, SIZE, dtype=np.float32)
        yi0 = np.clip(np.floor(yi).astype(np.int32), 0, h - 2)
        xi0 = np.clip(np.floor(xi).astype(np.int32), 0, w - 2)
        fy = yi - yi0; fx = xi - xi0
        return (
            (1 - fy)[:, None] * ((1 - fx)[None, :] * coarse[yi0][:, xi0] +
                                 fx[None, :]       * coarse[yi0][:, xi0 + 1]) +
            fy[:, None]       * ((1 - fx)[None, :] * coarse[yi0 + 1][:, xi0] +
                                 fx[None, :]       * coarse[yi0 + 1][:, xi0 + 1])
        ).astype(np.float32)

    field_medium = _bilinear_field(rng, 32, 32, 0.50, 1.50)
    field_high = _bilinear_field(rng, 128, 128, 0.65, 1.40)
    alpha *= field_medium * field_high
    fine_noise = rng.normal(0, 0.04, (SIZE, SIZE)).astype(np.float32)
    alpha = np.clip(alpha + fine_noise, 0, 1).astype(np.float32)
    del field_medium, field_high, fine_noise
    alpha *= inside_pix.astype(np.float32)

    u = rng.random((SIZE, SIZE))
    canvas = np.searchsorted(cum_base_use, u).astype(np.uint8); del u
    canvas[~inside_pix] = IDX_BG

    cum_peak = _build_peak_cum(class_name)
    for y0 in range(0, SIZE, 800):
        y1 = min(y0 + 800, SIZE)
        a_c = alpha[y0:y1, :]
        if a_c.max() < 0.01: continue
        cum_mixed = ((1 - a_c)[..., None] * cum_base_use[None, None, :] +
                     a_c[..., None] * cum_peak[None, None, :])
        uu = rng.random((y1-y0, SIZE)).astype(np.float32)
        grades = (uu[..., None] < cum_mixed).argmax(axis=-1).astype(np.uint8)
        mask = a_c > 0.01
        canvas[y0:y1][mask] = grades[mask]

    chip_meta = {}
    kind = '00P' if rng.random() < 0.5 else '00C'
    bin_pool = DEFECT_BIN_POOL[kind]
    for gy in range(GRID):
        for gx in range(GRID):
            if not inside[gy, gx]: continue
            y0, x0 = gy*CHIP, gx*CHIP
            chip_alpha = alpha[y0:y0+CHIP, x0:x0+CHIP]
            chip_alpha_mean = float(chip_alpha.mean())
            chip_alpha_max = float(chip_alpha.max())
            if chip_alpha_max < canvas_tau: continue
            score = max(chip_alpha_mean * 3.0, (chip_alpha_max - 0.5) * 1.5)
            p_def = min(score, 1.0)
            if rng.random() > p_def: continue
            b = int(rng.choice(bin_pool, p=DEFECT_BIN_WEIGHTS))
            chip_meta[(gy, gx)] = {'kind': 'defect', 'obj': None, 'bin': b, 'inside': True}
    del alpha

    IDX_BORDER_NORMAL = KEY_TO_INDEX["border"]
    for gy in range(GRID):
        for gx in range(GRID):
            if not inside[gy, gx]: continue
            y0, x0 = gy*CHIP, gx*CHIP; y1, x1 = y0+CHIP, x0+CHIP
            meta = chip_meta.get((gy, gx))
            if meta is None:
                bw, c = 1, IDX_BORDER_NORMAL
            else:
                bw, c = 2, BIN_TO_BORDER_IDX.get(meta['bin'], KEY_TO_INDEX["border_etc"])
            canvas[y0:y0+bw, x0:x1] = c; canvas[y1-bw:y1, x0:x1] = c
            canvas[y0:y1, x0:x0+bw] = c; canvas[y0:y1, x1-bw:x1] = c

    return canvas, chip_meta, PALETTE


def render_canvas(class_name, seed):
    rng = np.random.default_rng(seed)
    inside = _canvas_wafer_inside_mask()                # 32x32
    inside_pix = np.repeat(np.repeat(inside, CHIP, axis=0), CHIP, axis=1)

    # 0. Round 29 v15: per-wafer tier (baseline noise + defect intensity + chip-inclusion τ)
    baseline_tier  = pick_baseline_tier(rng)            # clean/normal/hazy
    intensity_tier = pick_intensity_tier(rng)           # strong/mid/weak
    canvas_tau     = float(rng.choice([0.20, 0.30, 0.40], p=[0.25, 0.50, 0.25]))
    cum_base_use   = CUM_BASELINE_TIERS[baseline_tier].astype(np.float32)

    # 1. wafer alpha = P(defect pixel | position) * intensity scale
    alpha = ALPHA_FN[class_name](rng) * INTENSITY_ALPHA_SCALE[intensity_tier]

    # 1b. multi-scale noise + along-line wave (line 따라 alpha 강/약 자연 변동)
    def _bilinear_field(rng, h, w, lo, hi):
        coarse = rng.uniform(lo, hi, size=(h, w)).astype(np.float32)
        yi = np.linspace(0, h - 1, SIZE, dtype=np.float32)
        xi = np.linspace(0, w - 1, SIZE, dtype=np.float32)
        yi0 = np.clip(np.floor(yi).astype(np.int32), 0, h - 2)
        xi0 = np.clip(np.floor(xi).astype(np.int32), 0, w - 2)
        fy = yi - yi0; fx = xi - xi0
        return (
            (1 - fy)[:, None] * ((1 - fx)[None, :] * coarse[yi0][:, xi0] +
                                 fx[None, :]       * coarse[yi0][:, xi0 + 1]) +
            fy[:, None]       * ((1 - fx)[None, :] * coarse[yi0 + 1][:, xi0] +
                                 fx[None, :]       * coarse[yi0 + 1][:, xi0 + 1])
        ).astype(np.float32)

    # multi-scale spatial noise — 더 강한 변동 (line 균일 X)
    field_medium = _bilinear_field(rng, 32, 32, 0.50, 1.50)      # ~chip 1 patch, 강한 변동
    field_high   = _bilinear_field(rng, 128, 128, 0.65, 1.40)    # ~chip 0.25 patch
    alpha *= field_medium * field_high
    fine_noise = rng.normal(0, 0.04, (SIZE, SIZE)).astype(np.float32)
    alpha = np.clip(alpha + fine_noise, 0, 1).astype(np.float32)
    del field_medium, field_high, fine_noise

    alpha *= inside_pix.astype(np.float32)              # outside-wafer = 0

    # 2. baseline canvas (round 29 v15: per-wafer baseline_tier)
    u = rng.random((SIZE, SIZE))
    canvas = np.searchsorted(cum_base_use, u).astype(np.uint8); del u
    canvas[~inside_pix] = IDX_BG

    # 3. baseline ↔ peak distribution mix (사용자 spec: alpha=0 → baseline, alpha=1 → peak)
    #    cum_mixed = (1-alpha)*cum_base_use + alpha*CUM_PEAK
    cum_peak = _build_peak_cum(class_name)
    chunk_h = 800
    for y0 in range(0, SIZE, chunk_h):
        y1 = min(y0 + chunk_h, SIZE)
        a_c = alpha[y0:y1, :]                            # (chunk_h, SIZE)
        if a_c.max() < 0.01: continue                    # 거의 0 → baseline 그대로
        cum_mixed = ((1 - a_c)[..., None] * cum_base_use[None, None, :] +
                     a_c[..., None] * cum_peak[None, None, :])
        uu = rng.random((y1-y0, SIZE)).astype(np.float32)
        grades = (uu[..., None] < cum_mixed).argmax(axis=-1).astype(np.uint8)
        # alpha 0 영역 baseline 그대로, alpha > 0.01 만 modulation
        mask = a_c > 0.01
        canvas[y0:y1][mask] = grades[mask]

    # 4. defect chip 결정 — alpha mean + max 둘 다 strict (line 직접 지나가는 chip 만 BIN)
    #    Round 29 v15: per-wafer canvas_tau (0.20/0.30/0.40) — weak wafer 일수록 high τ → 적은 chip
    chip_meta = {}
    kind = '00P' if rng.random() < 0.5 else '00C'
    bin_pool = DEFECT_BIN_POOL[kind]
    for gy in range(GRID):
        for gx in range(GRID):
            if not inside[gy, gx]: continue
            y0, x0 = gy*CHIP, gx*CHIP
            chip_alpha = alpha[y0:y0+CHIP, x0:x0+CHIP]
            chip_alpha_mean = float(chip_alpha.mean())
            chip_alpha_max = float(chip_alpha.max())
            # primary filter: line 직접 통과 chip 만 (sparse Row line 도 통과)
            if chip_alpha_max < canvas_tau: continue
            # P(defect) — mean primary + sparse line (Row) 의 max bonus
            score = max(chip_alpha_mean * 3.0, (chip_alpha_max - 0.5) * 1.5)
            p_def = min(score, 1.0)
            if rng.random() > p_def: continue
            b = int(rng.choice(bin_pool, p=DEFECT_BIN_WEIGHTS))
            chip_meta[(gy, gx)] = {'kind': 'defect', 'obj': None, 'bin': b, 'inside': True}

    del alpha

    # 5. border (defect = 2px BIN, normal = 1px gray) — wafer fail map spec
    IDX_BORDER_NORMAL = KEY_TO_INDEX["border"]
    for gy in range(GRID):
        for gx in range(GRID):
            if not inside[gy, gx]: continue
            y0, x0 = gy*CHIP, gx*CHIP; y1, x1 = y0+CHIP, x0+CHIP
            meta = chip_meta.get((gy, gx))
            if meta is None:
                bw, c = 1, IDX_BORDER_NORMAL
            else:
                bw, c = 2, BIN_TO_BORDER_IDX.get(meta['bin'], KEY_TO_INDEX["border_etc"])
            canvas[y0:y0+bw, x0:x1] = c; canvas[y1-bw:y1, x0:x1] = c
            canvas[y0:y1, x0:x0+bw] = c; canvas[y0:y1, x1-bw:x1] = c

    img = Image.frombytes('P', (SIZE, SIZE), canvas.tobytes())
    img.putpalette(PALETTE)

    # 6. yield/sys/TD/LT
    sys_bins = {285,286,287,288,290,291} if kind == '00P' else {300,385,386,388,389,390}
    netd = int(inside.sum())
    sys_count = sum(1 for m in chip_meta.values() if m['bin'] in sys_bins)
    n_def_count = len(chip_meta)
    gd_count = netd - n_def_count
    yld = 100.0 * gd_count / max(1, netd)
    syp = 100.0 * sys_count / max(1, netd)
    TD = LT_OPTIONS[int(rng.integers(0, len(LT_OPTIONS)))]
    LT = TM_OPTIONS[int(rng.integers(0, len(TM_OPTIONS)))]

    cls_label = class_name
    png_dir  = os.path.join(PNG_OUT_DIR, cls_label)
    json_dir = os.path.join(JSON_OUT_DIR, cls_label)
    os.makedirs(png_dir, exist_ok=True); os.makedirs(json_dir, exist_ok=True)
    prefix = rand_prefix(rng); w_idx = int(rng.integers(1, 25))
    base = f"{prefix}_{kind}_{w_idx:02d}_20260501_010000_{yld:.1f}_{syp:.0f}_{TD}_{LT}"
    png_path  = os.path.join(png_dir,  base + ".png")
    json_path = os.path.join(json_dir, base + ".json")
    img.save(png_path, optimize=False, compress_level=1)

    # 7. JSON
    chips_list = []
    norm_rng = np.random.default_rng(seed + 99999)
    for gy in range(GRID):
        for gx in range(GRID):
            if not inside[gy, gx]: continue
            if (gy, gx) in chip_meta:
                bin_val = chip_meta[(gy, gx)]['bin']
            else:
                bin_val = int(norm_rng.integers(1, 200))
            x0, y0 = gx*CHIP, gy*CHIP; x1, y1 = x0+CHIP, y0+CHIP
            x_cal = gx - (GRID//2 - 1); y_cal = gy - (GRID//2)
            chips_list.append({
                "x_abs": gx, "y_abs": gy, "b": str(bin_val),
                "x_cal": x_cal, "y_cal": y_cal,
                "rect": {"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                         "quad": [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]},
            })
    xs_edges = [k*CHIP for k in range(GRID+1)]
    ys_edges = [k*CHIP for k in range(GRID+1)]
    json_obj = {
        "bucket_b_key": "", "root": prefix, "step": kind, "wafer": f"W{w_idx:02d}",
        "stime": "20260501_010000", "partid": "", "tester": TD, "device": LT, "pgm": "",
        "netd": netd, "gd": int(gd_count),
        "yield": f"{yld:.2f}", "sys": f"{syp:.2f}",
        "tm": LT, "lt": TD,
        "wafer_meta": {                                                               # round 29 v15: per-wafer tier
            "synth_round": "29_v15",
            "baseline_tier": baseline_tier,
            "intensity_tier": intensity_tier,
            "canvas_tau": round(canvas_tau, 4),
        },
        "coord": {
            "rot_code": 5,
            "x_min_abs": 0, "y_min_abs": 0, "x_max_abs": GRID-1, "y_max_abs": GRID-1,
            "tiles_w_rot": GRID, "tiles_h_rot": GRID,
            "grid_edges": {"xs": xs_edges, "ys": ys_edges},
            "canvas": {"width": SIZE, "height": SIZE},
            "scale": {"sx": 1.0, "sy": 1.0},
            "border": 1, "defect_border": 2,
            "center_rule": {"even_x_zero": "left", "even_y_zero": "down"}
        },
        "chips": chips_list,
    }
    add_synthetic_fq_to_json(json_obj, cls_label, base)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, ensure_ascii=False, separators=(',', ':'))
    return png_path, n_def_count


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=2)
    p.add_argument('--classes', nargs='+', default=None)
    args = p.parse_args()
    target = args.classes if args.classes else CANVAS_CLASSES
    print(f"Generating {args.n} sample(s) per class for {len(target)} class(es)...")
    for ci, cls in enumerate(target):
        for s in range(args.n):
            seed = ci*100000 + s + 1
            try:
                path, n_def = render_canvas(cls, seed)
                print(f"  {cls}[{s+1}/{args.n}]  defect_chips={n_def:3d}  {os.path.basename(path)}")
            except Exception as e:
                print(f"  {cls}[{s+1}/{args.n}]  ERROR: {e}")
                import traceback; traceback.print_exc()
    print("Done.")
