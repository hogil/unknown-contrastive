"""GPU pipeline sample generator — single process + GPU compute + ThreadPool save.

Reuses 모든 설정 from _sample_gen.py.
주요 변경:
- 단일 프로세스 (CUDA context 1번만 init)
- baseline + alpha + mixed sampling 모두 GPU
- PIL build + PNG save + JSON write는 ThreadPoolExecutor (overlap with GPU)
- 결과: GPU 활용률 ↑, wall time ~3-4분 expected (vs CPU multi 12분)

Usage:
    python _sample_gen_gpu.py --n 200 --save-workers 8
"""
import os, time, json, sys, argparse
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from concurrent.futures import ThreadPoolExecutor
from _fq_metadata import add_synthetic_fq_to_json

# Reuse setup from _sample_gen
from _sample_gen import (
    PALETTE, KEY_TO_INDEX,
    SIZE, GRID, CHIP, HEATMAP_DIR, PNG_OUT_DIR, JSON_OUT_DIR,
    BASELINE, CUM_BASE,
    DEFECT_BG_DIST, CUM_DEFECT_BG,
    EDGE_DIST, CUM_EDGE,
    OBJECT_DISTS,
    DEFECT_BUDGET,
    LT_OPTIONS, TM_OPTIONS,
    CLASSES, OBJECTS,
    CHIP_OBJECTS, NORMAL_OBJECTS,
    CHIP_BASE_ALPHA,
    DEFECT_BIN_POOL, DEFECT_BIN_WEIGHTS,
    BIN_TO_BORDER_IDX,
    rand_prefix,
    _wafer_inside_mask,
    select_distribution_chips,
    select_random_invalid,
    select_normal_chips,
    outside_chips_as_invalid,
    assign_defect_bin, assign_invalid_bin, pick_mixed_object,
    FONT_BIG,
    build_tasks as base_build_tasks,
    save_chip_crops,
    _pink_noise_field_2d,
)

assert torch.cuda.is_available(), "GPU pipeline requires CUDA"
DEVICE = torch.device('cuda')
DTYPE  = torch.float32

# ===== GPU constants (precomputed once) =====
XC_2D_T = torch.arange(CHIP, device=DEVICE, dtype=DTYPE)[None, :].expand(CHIP, CHIP).contiguous()
YC_2D_T = torch.arange(CHIP, device=DEVICE, dtype=DTYPE)[:, None].expand(CHIP, CHIP).contiguous()

CUM_BASE_T      = torch.tensor(CUM_BASE,      device=DEVICE, dtype=DTYPE)
CUM_DEFECT_BG_T = torch.tensor(CUM_DEFECT_BG, device=DEVICE, dtype=DTYPE)
CUM_EDGE_T      = torch.tensor(CUM_EDGE,      device=DEVICE, dtype=DTYPE)
CUM_OBJ_T = {k: torch.tensor(np.cumsum(v), device=DEVICE, dtype=DTYPE)
             for k, v in OBJECT_DISTS.items()}

IDX_BG         = KEY_TO_INDEX["bg"]
IDX_TEXT       = KEY_TO_INDEX["text"]
IDX_BORDER     = KEY_TO_INDEX["border"]
IDX_BORDER_INV = KEY_TO_INDEX["border_inv"]
ONE_T  = torch.tensor(1.0, device=DEVICE, dtype=DTYPE)

# ===== torch alpha functions =====
def _perp_profile_t(d, sigma_s=1.5, sigma_m=3.0, sigma_w=6.0):
    strong = 1.00 * torch.exp(-d**2 / (2*sigma_s**2))
    med    = 0.55 * torch.exp(-d**2 / (2*sigma_m**2))
    weak   = 0.25 * torch.exp(-d**2 / (2*sigma_w**2))
    return torch.minimum(strong + med + weak, ONE_T)

def alpha_bank_boundary_t(rng):
    """v19y (260506) — sync with CPU _sample_gen.py: tail 길게 (sigma_w 12→30-45),
    per-chip weak 15%, per-line sigma random, peak sharp + halo wide."""
    a = torch.full((CHIP, CHIP), CHIP_BASE_ALPHA, device=DEVICE, dtype=DTYPE)
    n_seg = 10; seg_len = CHIP // n_seg
    chip_weak = rng.random() < 0.15
    s_lo, s_hi = (0.78, 0.92) if chip_weak else (0.85, 1.00)
    seg_lo, seg_hi = (0.70, 0.90) if chip_weak else (0.70, 1.00)
    for cx in [50, 100, 150]:
        sigma_s_i = float(rng.uniform(0.5, 1.0))
        sigma_m_i = float(rng.uniform(5.0, 9.0))
        sigma_w_i = float(rng.uniform(30.0, 45.0))                          # tail 길게 (halo wide)
        s = float(rng.uniform(s_lo, s_hi))
        seg = torch.from_numpy(rng.uniform(seg_lo, seg_hi, size=n_seg).astype(np.float32)).to(DEVICE)
        y_noise = seg.repeat_interleave(seg_len)[:CHIP].unsqueeze(1)
        line = _perp_profile_t(XC_2D_T - cx, sigma_s_i, sigma_m_i, sigma_w_i) * y_noise * s
        a = torch.maximum(a, line)
    for cy in [100]:
        sigma_s_i = float(rng.uniform(0.5, 1.0))
        sigma_m_i = float(rng.uniform(5.0, 9.0))
        sigma_w_i = float(rng.uniform(30.0, 45.0))
        s = float(rng.uniform(s_lo, s_hi))
        seg = torch.from_numpy(rng.uniform(seg_lo, seg_hi, size=n_seg).astype(np.float32)).to(DEVICE)
        x_noise = seg.repeat_interleave(seg_len)[:CHIP].unsqueeze(0)
        line = _perp_profile_t(YC_2D_T - cy, sigma_s_i, sigma_m_i, sigma_w_i) * x_noise * s
        a = torch.maximum(a, line)
    return a

def alpha_scratch_t(rng):
    """v19z++ (260506) — sync with CPU. n_lines 5-10 + per-line y center/length 산포."""
    a = torch.full((CHIP, CHIP), CHIP_BASE_ALPHA, device=DEVICE, dtype=DTYPE)
    n_lines = int(rng.integers(5, 11))                                  # v19z++: 5-10
    weak_only = rng.random() < 0.02
    s_lo, s_hi = (0.96, 1.00) if weak_only else (0.96, 1.00)
    cx_arr = sorted(rng.uniform(15, 185, size=n_lines).tolist())
    y_centers = rng.uniform(50, 150, size=n_lines)                      # 중심 산포
    y_halfs = rng.uniform(35, 95, size=n_lines)                         # 반길이 산포
    for i in range(n_lines):
        cx = float(cx_arr[i])
        y_start = float(max(0.0, y_centers[i] - y_halfs[i]))
        y_end = float(min(float(CHIP), y_centers[i] + y_halfs[i]))
        in_range = ((YC_2D_T >= y_start) & (YC_2D_T <= y_end)).to(DTYPE)
        s = float(rng.uniform(s_lo, s_hi))
        line = _perp_profile_t(XC_2D_T - cx, 1.0, 2.5, 5.0) * in_range * s
        a = torch.maximum(a, line)
    return a


def alpha_fork_t(rng):
    """v19z++ (260506) — sync with CPU. peak sharper (sigma 1.0-1.5), min 7 legs uniform spacing."""
    a = torch.full((CHIP, CHIP), CHIP_BASE_ALPHA, device=DEVICE, dtype=DTYPE)
    weak_only = rng.random() < 0.02
    s_lo, s_hi = (0.98, 1.0) if weak_only else (0.98, 1.0)
    cy_h = float(rng.uniform(30, 130))
    fork_width = float(rng.uniform(80, 130)) if not weak_only else float(rng.uniform(70, 100))
    fork_x_center = float(rng.uniform(fork_width / 2 + 15, CHIP - fork_width / 2 - 15))
    x_lo = fork_x_center - fork_width / 2
    x_hi = fork_x_center + fork_width / 2
    # v20 (260507): fork peak 두께 ↑ 2px → 4px (사용자 directive). 1.0-1.5 → 1.8-2.5
    s_h = float(rng.uniform(s_lo, s_hi))
    sigma_h = float(rng.uniform(1.8, 2.5))
    in_x = ((XC_2D_T >= x_lo) & (XC_2D_T <= x_hi)).to(DTYPE)                          # 260507: drop .unsqueeze(0) — XC_2D_T already (CHIP,CHIP)
    line_h = _perp_profile_t(YC_2D_T - cy_h, sigma_h, sigma_h * 1.5, sigma_h * 3.0) * in_x * s_h
    a = torch.maximum(a, line_h)
    # v19z++: 7-9 legs + linspace + ±3 jitter (uniform spacing)
    n_legs = int(rng.integers(7, 10))
    leg_height = float(rng.uniform(70, 130)) if not weak_only else float(rng.uniform(60, 100))
    leg_y_end = min(cy_h + leg_height, float(CHIP - 5))
    in_y = ((YC_2D_T >= cy_h) & (YC_2D_T <= leg_y_end)).to(DTYPE)                     # 260507: drop .unsqueeze(1)
    leg_xs_base = np.linspace(x_lo + 8, x_hi - 8, n_legs)
    leg_xs_jit = rng.uniform(-3, 3, size=n_legs)
    for k in range(n_legs):
        cx = float(leg_xs_base[k] + leg_xs_jit[k])
        s = float(rng.uniform(s_lo, s_hi))
        sigma_v = float(rng.uniform(1.7, 2.4))                          # v20: 두께 ↑ 0.9-1.4 → 1.7-2.4
        line = _perp_profile_t(XC_2D_T - cx, sigma_v, sigma_v * 1.5, sigma_v * 3.0) * in_y * s
        a = torch.maximum(a, line)
    return a


def alpha_scratch_rot_t(rng):
    """v19z++ (260506) — sync with CPU. n_lines 7-12 + per-line along center/length 산포."""
    a = torch.full((CHIP, CHIP), CHIP_BASE_ALPHA, device=DEVICE, dtype=DTYPE)
    n_lines = int(rng.integers(7, 13))                                  # v19z++: 7-12
    weak_only = rng.random() < 0.02
    s_lo, s_hi = (0.95, 1.00) if weak_only else (0.95, 1.00)
    theta = -21.0 * float(np.pi) / 180.0
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
    # v19z++: per-line along-axis center/length (taper 적용)
    along_centers = rng.uniform(0.30, 0.70, size=n_lines)
    along_halfs = rng.uniform(0.22, 0.45, size=n_lines)
    # along-axis normalized coord (rotated frame)
    d_along_t = sin_t * (XC_2D_T - 100.0) + cos_t * (YC_2D_T - 100.0)
    a_min = float(d_along_t.min().item()); a_max = float(d_along_t.max().item())
    along_norm_t = torch.clamp((d_along_t - a_min) / (a_max - a_min + 1e-6), 0, 1)
    for i in range(n_lines):
        cx = float(rng.uniform(15, 185))
        cy = float(rng.uniform(15, 185))
        s = float(rng.uniform(s_lo, s_hi))
        # per-line along taper (line 길이/위치 산포)
        along_start = float(along_centers[i] - along_halfs[i])
        along_end = float(along_centers[i] + along_halfs[i])
        fade_w = float(rng.uniform(0.04, 0.10))
        taper_along = torch.clamp((along_norm_t - along_start) / fade_w, 0, 1) * \
                      torch.clamp((along_end - along_norm_t) / fade_w, 0, 1)
        d_perp = cos_t * (XC_2D_T - cx) - sin_t * (YC_2D_T - cy)
        line = _perp_profile_t(d_perp, 1.0, 2.5, 5.0) * taper_along * s
        a = torch.maximum(a, line)
    return a

def alpha_geometric_random_t(rng):
    a = torch.full((CHIP, CHIP), CHIP_BASE_ALPHA, device=DEVICE, dtype=DTYPE)
    n_shapes = int(rng.integers(1, 5))
    for _ in range(n_shapes):
        c = int(rng.integers(0, 3))
        if c == 0:
            cx, cy = float(rng.uniform(30, 170)), float(rng.uniform(30, 170))
            sigma = float(rng.uniform(8, 18))
            blob = torch.exp(-((XC_2D_T - cx)**2 + (YC_2D_T - cy)**2) / (2*sigma**2))
            a = torch.maximum(a, blob)
        elif c == 1:
            angle = float(rng.uniform(0, np.pi)); cx = float(rng.uniform(50, 150)); cy = float(rng.uniform(50, 150))
            cos_t, sin_t = float(np.cos(angle)), float(np.sin(angle))
            d_perp = (XC_2D_T - cx) * cos_t - (YC_2D_T - cy) * sin_t
            d_along = (XC_2D_T - cx) * sin_t + (YC_2D_T - cy) * cos_t
            length = float(rng.uniform(30, 90)); sigma_l = float(rng.uniform(1.5, 3.0))
            in_range = (torch.abs(d_along) < length / 2).to(DTYPE)
            line = torch.exp(-d_perp**2 / (2*sigma_l**2)) * in_range
            a = torch.maximum(a, line)
        else:
            cx, cy = float(rng.uniform(50, 150)), float(rng.uniform(50, 150))
            r = float(rng.uniform(20, 60)); thick = float(rng.uniform(2.0, 4.0))
            d_r = torch.sqrt((XC_2D_T - cx)**2 + (YC_2D_T - cy)**2)
            arc = torch.exp(-(d_r - r)**2 / (2*thick**2))
            a = torch.maximum(a, arc)
    return a

def alpha_small_dot_t(rng):
    a = torch.full((CHIP, CHIP), CHIP_BASE_ALPHA, device=DEVICE, dtype=DTYPE)
    n = int(rng.integers(1, 5))
    for _ in range(n):
        cx = float(rng.uniform(20, 180)); cy = float(rng.uniform(20, 180))
        sigma = float(rng.uniform(1.5, 4.0))
        dot = torch.exp(-((XC_2D_T - cx)**2 + (YC_2D_T - cy)**2) / (2*sigma**2))
        a = torch.maximum(a, dot)
    return a

def alpha_fragment_t(rng):
    a = torch.full((CHIP, CHIP), CHIP_BASE_ALPHA, device=DEVICE, dtype=DTYPE)
    n = int(rng.integers(1, 3))
    for _ in range(n):
        angle = float(rng.uniform(0, 2*np.pi))
        cx = float(rng.uniform(30, 170)); cy = float(rng.uniform(30, 170))
        cos_t, sin_t = float(np.cos(angle)), float(np.sin(angle))
        d_perp = (XC_2D_T - cx) * cos_t - (YC_2D_T - cy) * sin_t
        d_along = (XC_2D_T - cx) * sin_t + (YC_2D_T - cy) * cos_t
        length = float(rng.uniform(15, 60)); sigma = float(rng.uniform(1.5, 3.0))
        in_range = (torch.abs(d_along) < length / 2).to(DTYPE)
        line = torch.exp(-d_perp**2 / (2*sigma**2)) * in_range
        a = torch.maximum(a, line)
    return a

def alpha_irregular_t(rng):
    a = torch.full((CHIP, CHIP), CHIP_BASE_ALPHA, device=DEVICE, dtype=DTYPE)
    n = int(rng.integers(2, 5))
    for _ in range(n):
        cx = float(rng.uniform(20, 180)); cy = float(rng.uniform(20, 180))
        sx = float(rng.uniform(3, 12)); sy = float(rng.uniform(3, 12))
        blob = torch.exp(-((XC_2D_T - cx)**2 / (2*sx**2) + (YC_2D_T - cy)**2 / (2*sy**2)))
        a = torch.maximum(a, blob)
    return a

def alpha_ring_small_t(rng):
    a = torch.full((CHIP, CHIP), CHIP_BASE_ALPHA, device=DEVICE, dtype=DTYPE)
    cx = float(rng.uniform(40, 160)); cy = float(rng.uniform(40, 160))
    r = float(rng.uniform(8, 25)); thick = float(rng.uniform(1.5, 3.0))
    d_r = torch.sqrt((XC_2D_T - cx)**2 + (YC_2D_T - cy)**2)
    ring = torch.exp(-(d_r - r)**2 / (2*thick**2))
    return torch.maximum(a, ring)

ALPHA_FNS_T = {                                                                          # v19 (260506) — 3 obj 별 별도 함수 (placeholder 폐기)
    'bank_boundary':    alpha_bank_boundary_t,
    'fork':             alpha_fork_t,                                                   # v19: cross pattern, weakest tier
    'scratch':          alpha_scratch_t,                                                # v19: vertical lines, strongest tier
    'scratch_rot':      alpha_scratch_rot_t,                                            # v19: ~71° rotated lines, mid tier
    'geometric_random': alpha_geometric_random_t,
    'small_dot':        alpha_small_dot_t,
    'fragment':         alpha_fragment_t,
    'irregular':        alpha_irregular_t,
    'ring_small':       alpha_ring_small_t,
}

# ===== Wafer-canvas freehand patterns (chuck mark 등 물리력 자국) =====
# 라인 중심 대부분 grade 1, 강한 spot grade 2 (사용자 spec)
LINE_DIST = np.array([0.05, 0.65, 0.25, 0.04, 0.005, 0.003, 0.001, 0.001], dtype=np.float64)
LINE_DIST = LINE_DIST / LINE_DIST.sum()
CUM_LINE_T = torch.tensor(np.cumsum(LINE_DIST), device=DEVICE, dtype=DTYPE)

def make_starburst_alpha(rng):
    """척 핀이 wafer 중심 부근에 박혀 방사형으로 긁힌 자국."""
    img = Image.new('L', (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(img)
    cx = int(SIZE/2 + rng.uniform(-300, 300))
    cy = int(SIZE/2 + rng.uniform(-300, 300))
    # 중심 핀 자국 (작은 원/blob)
    blob_r = int(rng.uniform(80, 180))
    draw.ellipse([cx-blob_r, cy-blob_r, cx+blob_r, cy+blob_r], fill=255)
    # 방사형 scratch 라인 6-14개 (random angles, length)
    n_lines = int(rng.integers(6, 15))
    for _ in range(n_lines):
        ang = float(rng.uniform(0, 2*np.pi))
        length = int(rng.uniform(700, 2200))
        x1 = cx + int(length * np.cos(ang)); y1 = cy + int(length * np.sin(ang))
        thickness = int(rng.integers(2, 6))
        draw.line([(cx, cy), (x1, y1)], fill=255, width=thickness)
    # smooth falloff
    img = img.filter(ImageFilter.GaussianBlur(radius=4))
    return np.array(img).astype(np.float32) / 255.0

def make_commacluster_alpha(rng):
    """척의 여러 핀/엣지가 ring 모양으로 wafer에 닿아 생긴 작은 호 자국 다수."""
    img = Image.new('L', (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(img)
    # ring center + radius (random per sample)
    cx = SIZE/2 + rng.uniform(-300, 300)
    cy = SIZE/2 + rng.uniform(-300, 300)
    ring_r = float(rng.uniform(700, 1800))
    # 작은 arc 10-30개 ring 따라
    n_arcs = int(rng.integers(10, 31))
    for _ in range(n_arcs):
        ang = float(rng.uniform(0, 2*np.pi))
        r_jit = ring_r + float(rng.uniform(-250, 250))
        ax = int(cx + r_jit * np.cos(ang)); ay = int(cy + r_jit * np.sin(ang))
        arc_r = int(rng.uniform(20, 60))
        a_start = float(rng.uniform(0, 360))
        a_end = a_start + float(rng.uniform(70, 200))
        thick = int(rng.integers(2, 5))
        bbox = [ax-arc_r, ay-arc_r, ax+arc_r, ay+arc_r]
        draw.arc(bbox, a_start, a_end, fill=255, width=thick)
    img = img.filter(ImageFilter.GaussianBlur(radius=2))
    return np.array(img).astype(np.float32) / 255.0

def make_diagonalsmear_alpha(rng):
    img = Image.new('L', (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(img)
    slope = float(rng.uniform(0.65, 1.35)) * (1 if rng.random() < 0.5 else -1)
    offset = float(rng.uniform(-900, 900))
    for _ in range(int(rng.integers(2, 5))):
        jitter = float(rng.uniform(-180, 180))
        width = int(rng.integers(18, 55))
        x0, x1 = -400, SIZE + 400
        y0 = int(SIZE / 2 + offset + jitter - slope * SIZE / 2)
        y1 = int(y0 + slope * (x1 - x0))
        draw.line([(x0, y0), (x1, y1)], fill=255, width=width)
    img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(8, 18))))
    return np.array(img).astype(np.float32) / 255.0

def make_crossscratch_alpha(rng):
    img = Image.new('L', (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(img)
    cx = int(SIZE / 2 + rng.uniform(-450, 450))
    cy = int(SIZE / 2 + rng.uniform(-450, 450))
    base_ang = float(rng.uniform(0, np.pi))
    for ang in (base_ang, base_ang + float(rng.uniform(0.65, 1.2))):
        length = int(rng.uniform(2200, 4200))
        x0 = cx - int(length * np.cos(ang))
        y0 = cy - int(length * np.sin(ang))
        x1 = cx + int(length * np.cos(ang))
        y1 = cy + int(length * np.sin(ang))
        draw.line([(x0, y0), (x1, y1)], fill=255, width=int(rng.integers(4, 12)))
    img = img.filter(ImageFilter.GaussianBlur(radius=3))
    return np.array(img).astype(np.float32) / 255.0

def make_crescentarc_alpha(rng):
    img = Image.new('L', (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(img)
    cx = int(SIZE / 2 + rng.uniform(-650, 650))
    cy = int(SIZE / 2 + rng.uniform(-650, 650))
    radius = int(rng.uniform(1350, 2550))
    start = float(rng.uniform(0, 360))
    extent = float(rng.uniform(70, 150))
    for k in range(int(rng.integers(2, 5))):
        rr = radius + int(rng.uniform(-90, 90))
        box = [cx - rr, cy - rr, cx + rr, cy + rr]
        draw.arc(box, start + k * 8, start + extent + k * 8, fill=255,
                 width=int(rng.integers(24, 70)))
    img = img.filter(ImageFilter.GaussianBlur(radius=10))
    return np.array(img).astype(np.float32) / 255.0

def make_spiraltrail_alpha(rng):
    img = Image.new('L', (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(img)
    cx = SIZE / 2 + rng.uniform(-280, 280)
    cy = SIZE / 2 + rng.uniform(-280, 280)
    turn = float(rng.uniform(1.1, 2.4))
    theta0 = float(rng.uniform(0, 2 * np.pi))
    pts = []
    for t in np.linspace(0, turn * 2 * np.pi, 180):
        r = 180 + 0.55 * t / (2 * np.pi) * SIZE
        ang = theta0 + t
        pts.append((int(cx + r * np.cos(ang)), int(cy + r * np.sin(ang))))
    draw.line(pts, fill=255, width=int(rng.integers(8, 24)))
    img = img.filter(ImageFilter.GaussianBlur(radius=5))
    return np.array(img).astype(np.float32) / 255.0

def make_parallelscratches_alpha(rng):
    img = Image.new('L', (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(img)
    ang = float(rng.uniform(0, np.pi))
    normal = ang + np.pi / 2
    cx = SIZE / 2 + rng.uniform(-300, 300)
    cy = SIZE / 2 + rng.uniform(-300, 300)
    for i in range(int(rng.integers(4, 10))):
        off = (i - 4) * rng.uniform(80, 180) + rng.uniform(-60, 60)
        mx = cx + off * np.cos(normal)
        my = cy + off * np.sin(normal)
        length = int(rng.uniform(1300, 3600))
        x0 = int(mx - length * np.cos(ang))
        y0 = int(my - length * np.sin(ang))
        x1 = int(mx + length * np.cos(ang))
        y1 = int(my + length * np.sin(ang))
        draw.line([(x0, y0), (x1, y1)], fill=255, width=int(rng.integers(3, 9)))
    img = img.filter(ImageFilter.GaussianBlur(radius=3))
    return np.array(img).astype(np.float32) / 255.0

def make_edgesmudge_alpha(rng):
    img = Image.new('L', (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(img)
    ang = float(rng.uniform(0, 2 * np.pi))
    cx = int(SIZE / 2 + 2450 * np.cos(ang))
    cy = int(SIZE / 2 + 2450 * np.sin(ang))
    rx = int(rng.uniform(260, 620))
    ry = int(rng.uniform(700, 1350))
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)
    for _ in range(int(rng.integers(3, 8))):
        jx = int(cx + rng.uniform(-300, 300))
        jy = int(cy + rng.uniform(-500, 500))
        rr = int(rng.uniform(80, 220))
        draw.ellipse([jx - rr, jy - rr, jx + rr, jy + rr], fill=180)
    img = img.filter(ImageFilter.GaussianBlur(radius=35))
    return np.array(img).astype(np.float32) / 255.0

def make_blobchain_alpha(rng):
    img = Image.new('L', (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(img)
    cx = SIZE / 2 + rng.uniform(-350, 350)
    cy = SIZE / 2 + rng.uniform(-350, 350)
    ang = float(rng.uniform(0, 2 * np.pi))
    curve = float(rng.uniform(-0.9, 0.9))
    for i in range(int(rng.integers(9, 20))):
        t = (i - 8) * rng.uniform(120, 230)
        bend = curve * (t * t) / 1800
        x = int(cx + t * np.cos(ang) + bend * np.cos(ang + np.pi / 2) + rng.uniform(-80, 80))
        y = int(cy + t * np.sin(ang) + bend * np.sin(ang + np.pi / 2) + rng.uniform(-80, 80))
        rx = int(rng.uniform(35, 120))
        ry = int(rng.uniform(35, 150))
        draw.ellipse([x - rx, y - ry, x + rx, y + ry], fill=255)
    img = img.filter(ImageFilter.GaussianBlur(radius=8))
    return np.array(img).astype(np.float32) / 255.0

def make_brokenring_alpha(rng):
    img = Image.new('L', (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(img)
    cx = int(SIZE / 2 + rng.uniform(-260, 260))
    cy = int(SIZE / 2 + rng.uniform(-260, 260))
    radius = int(rng.uniform(900, 1850))
    for _ in range(int(rng.integers(4, 9))):
        start = float(rng.uniform(0, 360))
        extent = float(rng.uniform(18, 65))
        rr = radius + int(rng.uniform(-180, 180))
        box = [cx - rr, cy - rr, cx + rr, cy + rr]
        draw.arc(box, start, start + extent, fill=255, width=int(rng.integers(18, 50)))
    img = img.filter(ImageFilter.GaussianBlur(radius=6))
    return np.array(img).astype(np.float32) / 255.0

# Override the prototype PIL-shape generators above with continuous wafer-scale
# alpha fields. These follow the chip-object alpha-field idea, but treat the
# whole wafer as one large chip: no pasted primitives, no blur augmentation, no
# flip, and only small angle/location jitter around a fixed process location.
WX = np.arange(SIZE, dtype=np.float32)[None, :]
WY = np.arange(SIZE, dtype=np.float32)[:, None]

def _empty_wafer_alpha():
    return np.zeros((SIZE, SIZE), dtype=np.float32)

def _clip_alpha(a):
    np.clip(a, 0.0, 1.0, out=a)
    return a

def _angle(deg):
    return np.deg2rad(float(deg))

def _add_tube(a, cx, cy, deg, half_len, sigma, strength=1.0,
              taper=220.0, waviness=0.0, wave_len=1100.0, phase=0.0):
    theta = _angle(deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx = WX - np.float32(cx)
    dy = WY - np.float32(cy)
    along = dx * cos_t + dy * sin_t
    perp = -dx * sin_t + dy * cos_t
    if waviness:
        perp = perp - np.float32(waviness) * np.sin(along / np.float32(wave_len) + np.float32(phase))
    core = np.exp(-(perp * perp) / np.float32(2.0 * sigma * sigma)).astype(np.float32)
    over = np.maximum(np.abs(along) - np.float32(half_len), 0.0)
    gate = np.exp(-(over * over) / np.float32(2.0 * taper * taper)).astype(np.float32)
    texture = (0.82 + 0.18 * np.sin(along / np.float32(380.0) + np.float32(phase))).astype(np.float32)
    np.maximum(a, np.float32(strength) * core * gate * texture, out=a)
    return a

def _add_blob(a, cx, cy, sx, sy, deg=0.0, strength=1.0):
    theta = _angle(deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx = WX - np.float32(cx)
    dy = WY - np.float32(cy)
    xr = dx * cos_t + dy * sin_t
    yr = -dx * sin_t + dy * cos_t
    field = np.exp(-((xr * xr) / np.float32(2.0 * sx * sx) +
                     (yr * yr) / np.float32(2.0 * sy * sy))).astype(np.float32)
    np.maximum(a, np.float32(strength) * field, out=a)
    return a

def _add_arc(a, cx, cy, radius, center_deg, span_deg, radial_sigma,
             strength=1.0, edge_soft_deg=8.0):
    dx = WX - np.float32(cx)
    dy = WY - np.float32(cy)
    rr = np.sqrt(dx * dx + dy * dy).astype(np.float32)
    ang = np.arctan2(dy, dx).astype(np.float32)
    center = np.float32(_angle(center_deg))
    delta = np.arctan2(np.sin(ang - center), np.cos(ang - center)).astype(np.float32)
    radial = np.exp(-((rr - np.float32(radius)) ** 2) /
                    np.float32(2.0 * radial_sigma * radial_sigma)).astype(np.float32)
    half_span = np.float32(_angle(span_deg) / 2.0)
    edge_soft = np.float32(_angle(edge_soft_deg))
    over = np.maximum(np.abs(delta) - half_span, 0.0)
    gate = np.exp(-(over * over) / np.float32(2.0 * edge_soft * edge_soft)).astype(np.float32)
    np.maximum(a, np.float32(strength) * radial * gate, out=a)
    return a

def make_starburst_alpha(rng):
    a = _empty_wafer_alpha()
    cx = SIZE / 2 + rng.uniform(-120, 120)
    cy = SIZE / 2 + rng.uniform(-120, 120)
    _add_blob(a, cx, cy, rng.uniform(70, 120), rng.uniform(70, 120), strength=0.95)
    for deg in [8, 42, 86, 132, 188, 236, 282, 326]:
        _add_tube(a, cx, cy, deg + rng.uniform(-5, 5), rng.uniform(850, 1900),
                  rng.uniform(6, 16), strength=rng.uniform(0.55, 0.95),
                  taper=260, waviness=rng.uniform(8, 28), phase=rng.uniform(0, 6.28))
    return _clip_alpha(a)

def make_commacluster_alpha(rng):
    a = _empty_wafer_alpha()
    cx = SIZE / 2 + rng.uniform(-120, 120)
    cy = SIZE / 2 + rng.uniform(-120, 120)
    radius = rng.uniform(980, 1550)
    for deg in [28, 44, 62, 84, 112, 134]:
        _add_arc(a, cx, cy, radius + rng.uniform(-90, 90),
                 deg + rng.uniform(-5, 5), rng.uniform(10, 22),
                 rng.uniform(14, 28), strength=rng.uniform(0.55, 1.0),
                 edge_soft_deg=4)
    return _clip_alpha(a)

def make_diagonalsmear_alpha(rng):
    a = _empty_wafer_alpha()
    cx = SIZE / 2 + rng.uniform(-160, 160)
    cy = SIZE / 2 + rng.uniform(-220, 220)
    base_deg = 23.0 + rng.uniform(-4, 4)
    nx = -np.sin(_angle(base_deg))
    ny = np.cos(_angle(base_deg))
    for offset in [-120, 0, 115]:
        _add_tube(a, cx + offset * nx, cy + offset * ny, base_deg,
                  rng.uniform(1800, 2700), rng.uniform(18, 42),
                  strength=rng.uniform(0.45, 0.88), taper=360,
                  waviness=rng.uniform(10, 35), phase=rng.uniform(0, 6.28))
    return _clip_alpha(a)

def make_crossscratch_alpha(rng):
    a = _empty_wafer_alpha()
    cx = SIZE * 0.56 + rng.uniform(-120, 120)
    cy = SIZE * 0.40 + rng.uniform(-120, 120)
    _add_tube(a, cx, cy, 8 + rng.uniform(-4, 4), rng.uniform(1600, 2300),
              rng.uniform(7, 17), strength=0.9, taper=260,
              waviness=rng.uniform(5, 18), phase=rng.uniform(0, 6.28))
    _add_tube(a, cx, cy, 78 + rng.uniform(-4, 4), rng.uniform(950, 1500),
              rng.uniform(6, 15), strength=0.75, taper=230,
              waviness=rng.uniform(4, 14), phase=rng.uniform(0, 6.28))
    return _clip_alpha(a)

def make_crescentarc_alpha(rng):
    a = _empty_wafer_alpha()
    cx = SIZE / 2 + rng.uniform(-100, 100)
    cy = SIZE / 2 + rng.uniform(-80, 140)
    _add_arc(a, cx, cy, rng.uniform(1900, 2350), 138 + rng.uniform(-6, 6),
             rng.uniform(42, 70), rng.uniform(30, 65), strength=0.95, edge_soft_deg=7)
    _add_arc(a, cx, cy, rng.uniform(1760, 2180), 145 + rng.uniform(-5, 5),
             rng.uniform(24, 48), rng.uniform(18, 42), strength=0.58, edge_soft_deg=6)
    return _clip_alpha(a)

def make_spiraltrail_alpha(rng):
    a = _empty_wafer_alpha()
    cx = SIZE / 2 + rng.uniform(-100, 100)
    cy = SIZE / 2 + rng.uniform(-100, 100)
    theta0 = _angle(205 + rng.uniform(-7, 7))
    turn = rng.uniform(1.15, 1.65)
    prev_x = prev_y = None
    for t in np.linspace(0.0, turn * 2.0 * np.pi, 10):
        r = 240 + 430 * t / (2.0 * np.pi)
        x = cx + r * np.cos(theta0 + t)
        y = cy + r * np.sin(theta0 + t)
        if prev_x is not None:
            dx = x - prev_x
            dy = y - prev_y
            deg = np.rad2deg(np.arctan2(dy, dx))
            _add_tube(a, (x + prev_x) / 2, (y + prev_y) / 2, deg,
                      np.hypot(dx, dy) / 2, rng.uniform(10, 24),
                      strength=rng.uniform(0.55, 0.92), taper=90,
                      waviness=rng.uniform(2, 8), phase=rng.uniform(0, 6.28))
        prev_x, prev_y = x, y
    return _clip_alpha(a)

def make_parallelscratches_alpha(rng):
    a = _empty_wafer_alpha()
    base_deg = 96 + rng.uniform(-4, 4)
    cx = SIZE * 0.48 + rng.uniform(-130, 130)
    cy = SIZE * 0.42 + rng.uniform(-130, 130)
    nx = -np.sin(_angle(base_deg))
    ny = np.cos(_angle(base_deg))
    for offset in np.linspace(-260, 260, int(rng.integers(5, 8))):
        _add_tube(a, cx + offset * nx + rng.uniform(-35, 35),
                  cy + offset * ny + rng.uniform(-35, 35),
                  base_deg + rng.uniform(-2, 2), rng.uniform(900, 1650),
                  rng.uniform(5, 13), strength=rng.uniform(0.45, 0.88),
                  taper=180, waviness=rng.uniform(3, 12), phase=rng.uniform(0, 6.28))
    return _clip_alpha(a)

def make_edgesmudge_alpha(rng):
    a = _empty_wafer_alpha()
    cx = SIZE * 0.62 + rng.uniform(-130, 130)
    cy = SIZE * 0.13 + rng.uniform(-80, 120)
    _add_blob(a, cx, cy, rng.uniform(360, 650), rng.uniform(170, 330),
              deg=rng.uniform(-8, 8), strength=0.85)
    _add_blob(a, cx + rng.uniform(-170, 170), cy + rng.uniform(40, 180),
              rng.uniform(180, 320), rng.uniform(80, 180),
              deg=rng.uniform(-8, 8), strength=0.55)
    return _clip_alpha(a)

def make_blobchain_alpha(rng):
    a = _empty_wafer_alpha()
    base_deg = 32 + rng.uniform(-5, 5)
    cx = SIZE * 0.40 + rng.uniform(-120, 120)
    cy = SIZE * 0.58 + rng.uniform(-120, 120)
    for t in np.linspace(-900, 1050, int(rng.integers(8, 13))):
        bend = 0.00009 * t * t
        x = cx + t * np.cos(_angle(base_deg)) - bend * np.sin(_angle(base_deg)) + rng.uniform(-50, 50)
        y = cy + t * np.sin(_angle(base_deg)) + bend * np.cos(_angle(base_deg)) + rng.uniform(-50, 50)
        _add_blob(a, x, y, rng.uniform(45, 115), rng.uniform(35, 100),
                  deg=base_deg + rng.uniform(-8, 8), strength=rng.uniform(0.45, 0.95))
    return _clip_alpha(a)

def make_brokenring_alpha(rng):
    a = _empty_wafer_alpha()
    cx = SIZE / 2 + rng.uniform(-110, 110)
    cy = SIZE / 2 + rng.uniform(-110, 110)
    radius = rng.uniform(1200, 1700)
    for deg in [18, 78, 146, 218, 292]:
        _add_arc(a, cx, cy, radius + rng.uniform(-85, 85),
                 deg + rng.uniform(-7, 7), rng.uniform(18, 42),
                 rng.uniform(20, 48), strength=rng.uniform(0.48, 0.9),
                 edge_soft_deg=5)
    return _clip_alpha(a)

WAFER_CANVAS_PATTERNS = {
    'Starburst':          make_starburst_alpha,
    'CommaCluster':       make_commacluster_alpha,
    'DiagonalSmear':      make_diagonalsmear_alpha,
    'CrossScratch':       make_crossscratch_alpha,
    'CrescentArc':        make_crescentarc_alpha,
    'SpiralTrail':        make_spiraltrail_alpha,
    'ParallelScratches':  make_parallelscratches_alpha,
    'EdgeSmudge':         make_edgesmudge_alpha,
    'BlobChain':          make_blobchain_alpha,
    'BrokenRing':         make_brokenring_alpha,
}

NEW_WAFER_CANVAS_CLASSES = [
    'DiagonalSmear',
    'CrossScratch',
    'CrescentArc',
    'SpiralTrail',
    'ParallelScratches',
    'EdgeSmudge',
    'BlobChain',
    'BrokenRing',
]

def build_tasks(n_per_class):
    tasks = list(base_build_tasks(n_per_class))
    seed_base = len(CLASSES) * 100000
    for ci, cls in enumerate(NEW_WAFER_CANVAS_CLASSES):
        for s in range(n_per_class):
            tasks.append((cls, '_wafer_canvas', seed_base + ci * 100000 + s))
    return tasks

# ===== GPU render =====
def render_gpu(class_name, object_name, seed):
    rng = np.random.default_rng(seed)
    inside = _wafer_inside_mask()                                                     # numpy (32,32) bool

    # Wafer-canvas 패턴 (물리력 자국) — 별도 경로
    if class_name in WAFER_CANVAS_PATTERNS:
        return render_wafer_canvas(class_name, seed, rng, inside)

    # Kind
    if class_name == 'Normal':
        kind = '00P' if rng.random() < 0.5 else '00C'
    elif object_name == 'invalid_main':
        kind = '00C'
    else:
        kind = '00P' if rng.random() < 0.5 else '00C'

    # Mask planning
    normal_obj_map = None
    if class_name == 'Normal':
        defect_mask, normal_obj_map = select_normal_chips(rng, inside)
        invalid_inside_mask = select_random_invalid(rng, defect_mask, inside, n=10)
    elif object_name == 'invalid_main':
        invalid_dist = select_distribution_chips(class_name, rng, inside)
        invalid_random = select_random_invalid(rng, invalid_dist, inside, n=10)
        invalid_inside_mask = invalid_dist | invalid_random
        defect_mask = np.zeros((GRID, GRID), dtype=bool)
    else:
        defect_mask = select_distribution_chips(class_name, rng, inside)
        invalid_inside_mask = select_random_invalid(rng, defect_mask, inside, n=15)
    invalid_mask = invalid_inside_mask & ~defect_mask

    # chip_meta
    chip_meta = {}
    if object_name != 'invalid_main':
        for gy in range(GRID):
            for gx in range(GRID):
                if defect_mask[gy, gx]:
                    if normal_obj_map is not None:
                        obj_actual = normal_obj_map[(gy, gx)]
                    else:
                        obj_actual = pick_mixed_object(object_name, rng, mix_ratio=0.05)  # 260506: 0.25 → 0.05 (CPU sample_gen 정의 통일, 25% mixed → 5%)
                    chip_meta[(gy,gx)] = {'kind':'defect', 'obj': obj_actual,
                                          'bin': assign_defect_bin(kind, rng), 'inside': True}
    for gy in range(GRID):
        for gx in range(GRID):
            if invalid_mask[gy, gx]:
                chip_meta[(gy,gx)] = {'kind':'invalid', 'obj': None,
                                      'bin': assign_invalid_bin(kind, rng), 'inside': True}

    # GPU canvas baseline — 260507 v4: wafer-level pink noise (1/f^1.5) field 으로 통일
    # CUM_BASE_T searchsorted REVERT — chip 경계 seam 제거 (사용자 directive 260507).
    # floor 0.08 + cap 0.35 (max 강도 ↓), per-wafer overall_scale = beta(2,8) random.
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    pink_field_np = _pink_noise_field_2d(rng, SIZE, exponent=1.5)                     # GPU torch.fft 사용 (auto)
    p_bg_field_t = torch.from_numpy(pink_field_np).to(DEVICE)
    overall_scale = float(rng.beta(2, 8))
    p_bg_field_t = torch.clamp(overall_scale * (0.5 + 1.0 * p_bg_field_t), 0.08, 0.35)
    u_bg = torch.rand((SIZE, SIZE), generator=g, device=DEVICE)
    is_bg_t = u_bg < p_bg_field_t
    u_bg2 = torch.rand((SIZE, SIZE), generator=g, device=DEVICE)
    _U8_0 = torch.tensor(0, device=DEVICE, dtype=torch.uint8)
    _U8_1 = torch.tensor(1, device=DEVICE, dtype=torch.uint8)
    _U8_2 = torch.tensor(2, device=DEVICE, dtype=torch.uint8)
    bg_grade_t = torch.where(u_bg2 < 0.92, _U8_1, _U8_2)
    canvas_t = torch.where(is_bg_t, bg_grade_t, _U8_0)
    del u_bg, u_bg2, is_bg_t, bg_grade_t, pink_field_np
    inside_t = torch.from_numpy(inside.astype(np.uint8)).to(DEVICE)
    inside_pix_t = inside_t.repeat_interleave(CHIP, dim=0).repeat_interleave(CHIP, dim=1)
    canvas_t.masked_fill_(inside_pix_t == 0, IDX_BG)

    # Per-chip GPU — 260507 v5 unified (canonical = `_synth_chips_only.py`)
    #   fork:           2-stage smoothstep 0.50/0.88
    #   scratch / scr_rot: 2-stage smoothstep 0.60/0.91
    #   bank_boundary 등: 3-way zone mix with split 0.45/0.55
    # chip 의 non-defect baseline = wafer pink slice (chip 경계 seam 제거).
    _U8_3 = torch.tensor(3, device=DEVICE, dtype=torch.uint8)
    _U8_4 = torch.tensor(4, device=DEVICE, dtype=torch.uint8)
    for (gy, gx), meta in chip_meta.items():
        if meta['kind'] != 'defect': continue
        obj = meta['obj']
        alpha = ALPHA_FNS_T[obj](rng)
        y0, x0 = gy*CHIP, gx*CHIP

        # chip pink slice baseline (모든 obj 공통)
        chip_p_bg_t = p_bg_field_t[y0:y0+CHIP, x0:x0+CHIP]
        u_bgc = torch.rand((CHIP, CHIP), generator=g, device=DEVICE)
        is_chip_bg = u_bgc < chip_p_bg_t
        u_bgc2 = torch.rand((CHIP, CHIP), generator=g, device=DEVICE)
        chip_bg_grade = torch.where(u_bgc2 < 0.92, _U8_1, _U8_2)
        pink_baseline = torch.where(is_chip_bg, chip_bg_grade, _U8_0)

        if obj in ('fork', 'scratch', 'scratch_rot'):
            # 2-stage
            if obj == 'fork':
                lo_t2, hi_t2 = 0.50, 0.88
            else:
                lo_t2, hi_t2 = 0.60, 0.91
            u1 = torch.rand((CHIP, CHIP), generator=g, device=DEVICE)
            is_defect = u1 < alpha
            t2 = torch.clamp((alpha - lo_t2) / (hi_t2 - lo_t2), 0.0, 1.0)
            p_2 = t2 * t2 * (3.0 - 2.0 * t2)
            u2 = torch.rand((CHIP, CHIP), generator=g, device=DEVICE)
            is_2 = u2 < p_2
            u3 = torch.rand((CHIP, CHIP), generator=g, device=DEVICE)
            # defect_other: 95% grade 1, 4% grade 3, 1% grade 4
            defect_other = torch.where(u3 < 0.95, _U8_1,
                            torch.where(u3 < 0.99, _U8_3, _U8_4))
            defect_grade = torch.where(is_2, _U8_2, defect_other)
            grades = torch.where(is_defect, defect_grade, pink_baseline)
        else:
            # 3-way zone mix (bank_boundary 등) — split 0.45/0.55
            cum_obj_t = CUM_OBJ_T[obj]
            t_low = torch.clamp(alpha / 0.45, 0.0, 1.0)
            t_high = torch.clamp((alpha - 0.45) / 0.55, 0.0, 1.0)
            s_low = t_low * t_low * (3.0 - 2.0 * t_low)
            s_high = t_high * t_high * (3.0 - 2.0 * t_high)
            mask_low = (alpha < 0.45).to(DTYPE)
            mask_high = 1.0 - mask_low
            w_bg = mask_low * (1.0 - s_low)
            w_edge = mask_low * s_low + mask_high * (1.0 - s_high)
            w_center = mask_high * s_high
            cum_mixed = (w_bg.unsqueeze(-1)     * CUM_DEFECT_BG_T[None, None, :] +
                         w_edge.unsqueeze(-1)   * CUM_EDGE_T[None, None, :] +
                         w_center.unsqueeze(-1) * cum_obj_t[None, None, :])
            u_chip = torch.rand((CHIP, CHIP), generator=g, device=DEVICE)
            grades = (u_chip.unsqueeze(-1) < cum_mixed).int().argmax(dim=-1).to(torch.uint8)
            grades = torch.where(alpha < 0.05, pink_baseline, grades)
        canvas_t[y0:y0+CHIP, x0:x0+CHIP] = grades

    # Invalid fill (GPU)
    for (gy, gx), meta in chip_meta.items():
        if meta['kind'].startswith('invalid'):
            y0, x0 = gy*CHIP, gx*CHIP
            canvas_t[y0:y0+CHIP, x0:x0+CHIP] = 31

    # Borders (CPU since per-chip, fast either way)
    canvas = canvas_t.cpu().numpy()
    for gy in range(GRID):
        for gx in range(GRID):
            if not inside[gy, gx]: continue
            y0, x0 = gy*CHIP, gx*CHIP; y1, x1 = y0+CHIP, x0+CHIP
            meta = chip_meta.get((gy, gx))
            if meta is None:
                b, c = 1, IDX_BORDER
            elif meta['kind'] == 'defect':
                b, c = 2, BIN_TO_BORDER_IDX.get(meta['bin'], KEY_TO_INDEX["border_etc"])
            else:
                b, c = 2, IDX_BORDER_INV
            canvas[y0:y0+b, x0:x1] = c; canvas[y1-b:y1, x0:x1] = c
            canvas[y0:y1, x0:x0+b] = c; canvas[y0:y1, x1-b:x1] = c

    # Yield/Sys
    sys_bins = {285,286,287,288,290,291} if kind == '00P' else {300,385,386,388,389,390}
    netd = int(inside.sum())
    inside_chip_meta_count = sum(1 for m in chip_meta.values() if m['inside'])
    normal_inside = netd - inside_chip_meta_count
    gd_count = normal_inside + sum(1 for m in chip_meta.values() if m['inside'] and m['bin'] is not None and m['bin'] < 200)
    sys_count = sum(1 for m in chip_meta.values() if m['inside'] and m['bin'] in sys_bins)
    yld = 100.0 * gd_count / max(1, netd)
    syp = 100.0 * sys_count / max(1, netd)
    TD = LT_OPTIONS[int(rng.integers(0, len(LT_OPTIONS)))]
    LT = TM_OPTIONS[int(rng.integers(0, len(TM_OPTIONS)))]
    prefix = rand_prefix(rng); w_idx = int(rng.integers(1, 25))

    return {
        'canvas': canvas, 'chip_meta': chip_meta, 'kind': kind,
        'class_name': class_name, 'object_name': object_name, 'seed': seed,
        'prefix': prefix, 'w_idx': w_idx, 'yld': yld, 'syp': syp,
        'TD': TD, 'LT': LT, 'gd_count': gd_count, 'inside': inside,
    }

def render_wafer_canvas(class_name, seed, rng, inside):
    """물리력 자국: wafer 전체 도화지에 직접 라인/스머지/arc 를 그림.
    chip-grid alpha 패턴 안 씀. 그린 라인 위에서 grade 1 dominant + grade 2 spot.
    chip 경계는 normal 1px gray + 자국 통과한 chip은 random BIN 2px border.
    """
    kind = '00P' if rng.random() < 0.5 else '00C'

    # 1. wafer-canvas alpha field (CPU PIL)
    alpha_np = WAFER_CANVAS_PATTERNS[class_name](rng)
    inside_pix_np = np.repeat(np.repeat(inside.astype(np.float32), CHIP, axis=0), CHIP, axis=1)
    alpha_np = alpha_np * inside_pix_np                                               # 외곽 0

    # 2. baseline canvas (GPU) — 260507 v4: wafer-level pink noise field
    g_t = torch.Generator(device=DEVICE).manual_seed(seed)
    pink_field_np = _pink_noise_field_2d(rng, SIZE, exponent=1.5)
    p_bg_field_t = torch.from_numpy(pink_field_np).to(DEVICE)
    overall_scale = float(rng.beta(2, 8))
    p_bg_field_t = torch.clamp(overall_scale * (0.5 + 1.0 * p_bg_field_t), 0.08, 0.35)
    u_bg = torch.rand((SIZE, SIZE), generator=g_t, device=DEVICE)
    is_bg_t = u_bg < p_bg_field_t
    u_bg2 = torch.rand((SIZE, SIZE), generator=g_t, device=DEVICE)
    _U8_0 = torch.tensor(0, device=DEVICE, dtype=torch.uint8)
    _U8_1 = torch.tensor(1, device=DEVICE, dtype=torch.uint8)
    _U8_2 = torch.tensor(2, device=DEVICE, dtype=torch.uint8)
    bg_grade_t = torch.where(u_bg2 < 0.92, _U8_1, _U8_2)
    canvas_t = torch.where(is_bg_t, bg_grade_t, _U8_0)
    del u_bg, u_bg2, is_bg_t, bg_grade_t, pink_field_np
    inside_t = torch.from_numpy(inside.astype(np.uint8)).to(DEVICE)
    inside_pix_t = inside_t.repeat_interleave(CHIP, dim=0).repeat_interleave(CHIP, dim=1)
    canvas_t.masked_fill_(inside_pix_t == 0, IDX_BG)

    # 3. line modulation: alpha > 0.05 위치만 처리 (sparse)
    alpha_t = torch.from_numpy(alpha_np).to(DEVICE)
    mask_t = alpha_t > 0.05
    if mask_t.any():
        a_act = alpha_t[mask_t]
        w_bg = torch.clamp((0.40 - a_act) / 0.40, 0, 1)
        t_raw = torch.clamp((a_act - 0.40) / 0.60, 0, 1)
        w_center = t_raw ** 4
        w_edge = torch.clamp(1.0 - w_bg - w_center, 0, 1)
        cum_mixed = (w_bg.unsqueeze(-1)     * CUM_DEFECT_BG_T[None, :] +
                     w_edge.unsqueeze(-1)   * CUM_EDGE_T[None, :] +
                     w_center.unsqueeze(-1) * CUM_LINE_T[None, :])
        u_act = torch.rand(len(a_act), generator=g_t, device=DEVICE)
        grades = (u_act.unsqueeze(-1) < cum_mixed).int().argmax(dim=-1).to(torch.uint8)
        canvas_t[mask_t] = grades

    # CPU로 가져와 chip-level analysis
    canvas = canvas_t.cpu().numpy()

    # 4. chip status: alpha field 기반 — 라인이 통과한 chip만 defect
    chip_meta = {}
    for gy in range(GRID):
        for gx in range(GRID):
            if not inside[gy, gx]: continue
            y0, x0 = gy*CHIP, gx*CHIP
            sub_alpha = alpha_np[y0:y0+CHIP, x0:x0+CHIP]
            if sub_alpha.max() > 0.30:                                                 # 라인이 chip 내부 통과
                chip_meta[(gy, gx)] = {'kind': 'defect', 'obj': None,
                                        'bin': assign_defect_bin(kind, rng), 'inside': True}
    # invalid 산재 (15)
    defect_mask = np.zeros((GRID, GRID), bool)
    for (gy, gx) in chip_meta: defect_mask[gy, gx] = True
    invalid_inside_mask = select_random_invalid(rng, defect_mask, inside, n=15)
    for gy in range(GRID):
        for gx in range(GRID):
            if invalid_inside_mask[gy, gx]:
                chip_meta[(gy, gx)] = {'kind': 'invalid', 'obj': None,
                                        'bin': assign_invalid_bin(kind, rng), 'inside': True}
                y0, x0 = gy*CHIP, gx*CHIP
                canvas[y0:y0+CHIP, x0:x0+CHIP] = 31

    # 5. borders
    for gy in range(GRID):
        for gx in range(GRID):
            if not inside[gy, gx]: continue
            y0, x0 = gy*CHIP, gx*CHIP; y1, x1 = y0+CHIP, x0+CHIP
            meta = chip_meta.get((gy, gx))
            if meta is None:
                b, c = 1, IDX_BORDER
            elif meta['kind'] == 'defect':
                b, c = 2, BIN_TO_BORDER_IDX.get(meta['bin'], KEY_TO_INDEX["border_etc"])
            else:
                b, c = 2, IDX_BORDER_INV
            canvas[y0:y0+b, x0:x1] = c; canvas[y1-b:y1, x0:x1] = c
            canvas[y0:y1, x0:x0+b] = c; canvas[y0:y1, x1-b:x1] = c

    # 6. yield/sys
    sys_bins = {285,286,287,288,290,291} if kind == '00P' else {300,385,386,388,389,390}
    netd = int(inside.sum())
    inside_meta_count = sum(1 for m in chip_meta.values() if m['inside'])
    normal_inside = netd - inside_meta_count
    gd_count = normal_inside + sum(1 for m in chip_meta.values()
                                    if m['inside'] and m['bin'] is not None and m['bin'] < 200)
    sys_count = sum(1 for m in chip_meta.values() if m['inside'] and m['bin'] in sys_bins)
    yld = 100.0 * gd_count / max(1, netd)
    syp = 100.0 * sys_count / max(1, netd)
    TD = LT_OPTIONS[int(rng.integers(0, len(LT_OPTIONS)))]
    LT = TM_OPTIONS[int(rng.integers(0, len(TM_OPTIONS)))]
    prefix = rand_prefix(rng); w_idx = int(rng.integers(1, 25))

    return {
        'canvas': canvas, 'chip_meta': chip_meta, 'kind': kind,
        'class_name': class_name, 'object_name': '_wafer_canvas', 'seed': seed,
        'prefix': prefix, 'w_idx': w_idx, 'yld': yld, 'syp': syp,
        'TD': TD, 'LT': LT, 'gd_count': gd_count, 'inside': inside,
    }

def save_to_disk(r):
    """Thread-safe save: PIL build + text + PNG + JSON."""
    if r['class_name'] == 'Normal' or r['class_name'] in WAFER_CANVAS_PATTERNS:
        cls_label = r['class_name']
    else:
        cls_label = f"{r['class_name']}_{r['object_name']}"
    png_dir = os.path.join(PNG_OUT_DIR, cls_label); os.makedirs(png_dir, exist_ok=True)
    json_dir = os.path.join(JSON_OUT_DIR, cls_label); os.makedirs(json_dir, exist_ok=True)
    base = f"{r['prefix']}_{r['kind']}_{r['w_idx']:02d}_20260501_010000_{r['yld']:.1f}_{r['syp']:.0f}_{r['TD']}_{r['LT']}"
    png_path  = os.path.join(png_dir,  base + ".png")
    json_path = os.path.join(json_dir, base + ".json")

    img = Image.frombytes('P', (SIZE, SIZE), r['canvas'].tobytes())
    img.putpalette(PALETTE)
    draw = ImageDraw.Draw(img)
    for (gy, gx), meta in r['chip_meta'].items():
        if meta['kind'] != 'invalid': continue
        text = str(meta['bin']); cx_px = gx*CHIP + CHIP/2; cy_px = gy*CHIP + CHIP/2
        try:
            bbox = draw.textbbox((0,0), text, font=FONT_BIG)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
            ty = cy_px - th/2 - bbox[1]
        except Exception:
            tw = th = 40; ty = cy_px - th/2
        draw.text((cx_px - tw/2, ty), text, fill=IDX_TEXT, font=FONT_BIG)
    img.save(png_path, optimize=False, compress_level=1)

    # Per-chip object-true crops (chip-object classifier dataset)
    save_chip_crops(img, r['chip_meta'], base)

    # JSON
    chips_list = []
    norm_rng = np.random.default_rng(r['seed'] + 99999)
    chip_meta = r['chip_meta']; inside = r['inside']
    for gy in range(GRID):
        for gx in range(GRID):
            if not inside[gy, gx]: continue
            if (gy, gx) in chip_meta and chip_meta[(gy,gx)]['bin'] is not None:
                bin_val = chip_meta[(gy,gx)]['bin']
            else:
                bin_val = int(norm_rng.integers(1, 200))
            x0, y0 = gx*CHIP, gy*CHIP; x1, y1 = x0+CHIP, y0+CHIP
            chips_list.append({"x_abs": gx, "y_abs": gy, "b": str(bin_val),
                               "x_cal": gx - (GRID//2-1), "y_cal": gy - (GRID//2),
                               "rect": {"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                                        "quad": [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]}})
    xs_edges = [k*CHIP for k in range(GRID+1)]; ys_edges = [k*CHIP for k in range(GRID+1)]
    json_obj = {
        "bucket_b_key": "", "root": r['prefix'], "step": r['kind'],
        "wafer": f"W{r['w_idx']:02d}", "stime": "20260501_010000",
        "partid": "", "tester": r['TD'], "device": r['LT'], "pgm": "",
        "netd": int(inside.sum()), "gd": int(r['gd_count']),
        "yield": f"{r['yld']:.2f}", "sys": f"{r['syp']:.2f}",
        "tm": r['LT'], "lt": r['TD'],
        "coord": {"rot_code": 5, "x_min_abs": 0, "y_min_abs": 0,
                  "x_max_abs": GRID-1, "y_max_abs": GRID-1,
                  "tiles_w_rot": GRID, "tiles_h_rot": GRID,
                  "grid_edges": {"xs": xs_edges, "ys": ys_edges},
                  "canvas": {"width": SIZE, "height": SIZE},
                  "scale": {"sx": 1.0, "sy": 1.0},
                  "border": 1, "defect_border": 2,
                  "center_rule": {"even_x_zero": "left", "even_y_zero": "down"}},
        "chips": chips_list,
    }
    add_synthetic_fq_to_json(json_obj, cls_label, base)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, ensure_ascii=False, separators=(',', ':'))
    return png_path

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=200)
    p.add_argument('--save-workers', type=int, default=8, help='thread pool for PNG save')
    p.add_argument('--max-pending', type=int, default=24, help='backpressure: pending saves')
    p.add_argument('--only-class', type=str, default=None, help='generate only this distribution (e.g. Normal)')
    p.add_argument('--seed-offset', type=int, default=0, help='offset seeds (avoid filename collision with previous run)')
    args = p.parse_args()

    tasks = build_tasks(args.n)
    if args.only_class:
        tasks = [t for t in tasks if t[0] == args.only_class]
        print(f"Filter --only-class={args.only_class}: {len(tasks)} tasks")
    if args.seed_offset:
        tasks = [(c, o, s + args.seed_offset) for (c, o, s) in tasks]
    n_total = len(tasks)
    print(f"GPU pipeline: {n_total} tasks, save_workers={args.save_workers}")
    t0 = time.time(); done = 0; ok = 0; fail = 0
    last_log = t0

    pending = []
    with ThreadPoolExecutor(max_workers=args.save_workers) as save_pool:
        for cls, obj, seed in tasks:
            try:
                result = render_gpu(cls, obj, seed)                                   # GPU compute (sequential)
                fut = save_pool.submit(save_to_disk, result)                          # async save
                pending.append((cls, obj, seed, fut))
            except Exception as e:
                fail += 1; done += 1
                print(f"  FAIL {cls}_{obj} seed={seed}: {e}", flush=True)
            # Backpressure: wait if too many pending
            while len(pending) >= args.max_pending:
                cls_p, obj_p, seed_p, fut_p = pending.pop(0)
                try:
                    fut_p.result(); ok += 1
                except Exception as e:
                    fail += 1; print(f"  SAVE FAIL {cls_p}_{obj_p}: {e}", flush=True)
                done += 1
                now = time.time()
                if now - last_log > 30 or done == n_total:
                    rate = done / (now - t0); eta = (n_total - done) / rate if rate > 0 else 0
                    print(f"  [{done:5d}/{n_total}] ok={ok} fail={fail} | rate={rate:.2f}/s | elapsed={(now-t0)/60:.1f}m | eta={eta/60:.1f}m", flush=True)
                    last_log = now
        # drain
        for cls_p, obj_p, seed_p, fut_p in pending:
            try: fut_p.result(); ok += 1
            except Exception as e: fail += 1; print(f"  SAVE FAIL {cls_p}_{obj_p}: {e}", flush=True)
            done += 1
    print(f"\nDone in {(time.time()-t0)/60:.1f}m. ok={ok} fail={fail}")
    print(f"  PNG  -> {PNG_OUT_DIR}")
    print(f"  JSON -> {JSON_OUT_DIR}")
