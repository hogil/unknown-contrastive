#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-contained chip synthesis for the H100 single-SOTA reproduction.

FULLY INDEPENDENT — imports nothing from dist_apply / chip_multilabel / mega_matrix.
Algorithms (palette, grade tiers, Lorentzian/Gaussian alpha profiles, the 4 single
defects, min-blend combos, Normal/Invalid, OOD) are ported faithfully from the
project's validated generators so the produced folders are equivalent to the
reference `classification_chips/` (train) and `chip_multilabel_v15direct_n2000/`
(eval) sets.

Chip = 200x200 mode='P' palette PNG. 4 train defects: bank_boundary, fork,
scratch, scratch_rot. Eval adds 6 two-combos + 4 OOD + Normal + Invalid.
"""
from __future__ import annotations

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CHIP = 200

# ============================================================================
# Palette  (25 named colors; index 31 forced white for invalid_fill)
# ============================================================================
def _hex(s): return [int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)]

_PALETTE_HEX = {
    "chip0": "#FFFFFF", "chip1": "#9B9B9B", "chip2": "#009619", "chip3": "#0000FF",
    "chip4": "#D91DFF", "chip5": "#FFFF00", "chip6": "#FF0000", "chip7": "#000000",
    "bg": "#DCEEFF", "text": "#000001", "border": "#BEBEBE", "border_inv": "#FF9900",
    "border_b285": "#0099FF", "border_b286": "#FF714F", "border_b287": "#66FFCC",
    "border_b288": "#DA26CD", "border_b290": "#FFD700", "border_b291": "#32CD32",
    "border_b300": "#AAAAAA", "border_b385": "#00C8FF", "border_b386": "#FF00C8",
    "border_b388": "#00FF66", "border_b389": "#FF6666", "border_b390": "#6666FF",
    "border_etc": "#999999",
}
_PALETTE_ORDER = [
    "chip0", "chip1", "chip2", "chip3", "chip4", "chip5", "chip6", "chip7",
    "bg", "text", "border", "border_inv",
    "border_b285", "border_b286", "border_b287", "border_b288", "border_b290", "border_b291",
    "border_b300", "border_b385", "border_b386", "border_b388", "border_b389", "border_b390",
    "border_etc",
]
KEY_TO_INDEX = {k: i for i, k in enumerate(_PALETTE_ORDER)}
PALETTE = []
for _k in _PALETTE_ORDER:
    PALETTE.extend(_hex(_PALETTE_HEX[_k]))
while len(PALETTE) < 96:
    PALETTE.append(0)
PALETTE[31 * 3:31 * 3 + 3] = [255, 255, 255]   # idx 31 = white (invalid fill)

_BIN_TO_BORDER_KEY = {
    285: "border_b285", 286: "border_b286", 287: "border_b287",
    288: "border_b288", 290: "border_b290", 291: "border_b291",
    300: "border_b300", 385: "border_b385", 386: "border_b386",
    388: "border_b388", 389: "border_b389", 390: "border_b390",
}
BIN_TO_BORDER_IDX = {b: KEY_TO_INDEX[k] for b, k in _BIN_TO_BORDER_KEY.items()}

DEFECT_BIN_POOL = {
    '00P': [285, 286, 287, 288, 290, 291],
    '00C': [300, 385, 386, 388, 389, 390],
}
DEFECT_BIN_WEIGHTS = np.array([6, 5, 4, 3, 2, 1], dtype=np.float64)
DEFECT_BIN_WEIGHTS /= DEFECT_BIN_WEIGHTS.sum()

# ============================================================================
# Grade tier distributions
# ============================================================================
BASELINE = np.array([0.83, 0.15, 0.012, 0.005, 0.002, 0.0008, 0.0001, 0.0001], dtype=np.float64)
BASELINE /= BASELINE.sum()
CUM_BASE = np.cumsum(BASELINE)

DEFECT_BG_DIST = np.array([0.73, 0.25, 0.012, 0.005, 0.002, 0.0008, 0.0001, 0.0001], dtype=np.float64)
DEFECT_BG_DIST /= DEFECT_BG_DIST.sum()
CUM_DEFECT_BG = np.cumsum(DEFECT_BG_DIST)

EDGE_DIST = np.array([0.50, 0.40, 0.07, 0.02, 0.005, 0.003, 0.001, 0.001], dtype=np.float64)
EDGE_DIST /= EDGE_DIST.sum()
CUM_EDGE = np.cumsum(EDGE_DIST)

BASELINE_TIERS = {
    'clean':  np.array([0.95, 0.040, 0.008, 0.001, 0.0005, 0.0003, 0.0001, 0.0001], dtype=np.float64),
    'normal': BASELINE.copy(),
    'hazy':   np.array([0.65, 0.30, 0.040, 0.006, 0.002, 0.001, 0.0005, 0.0005], dtype=np.float64),
}
for _k in BASELINE_TIERS:
    BASELINE_TIERS[_k] /= BASELINE_TIERS[_k].sum()
CUM_BASELINE_TIERS = {k: np.cumsum(v) for k, v in BASELINE_TIERS.items()}

INTENSITY_ALPHA_SCALE = {'strong': 1.0, 'mid': 0.96, 'weak': 0.93}

OBJECT_DISTS = {
    'bank_boundary': np.array([0.003, 0.10, 0.85, 0.11, 0.004, 0.002, 0.001, 0.0], dtype=np.float64),
    'fork':          np.array([0.025, 0.13, 0.72, 0.15, 0.010, 0.003, 0.0, 0.0], dtype=np.float64),
    'scratch':       np.array([0.025, 0.13, 0.72, 0.15, 0.010, 0.003, 0.0, 0.0], dtype=np.float64),
    'scratch_rot':   np.array([0.025, 0.13, 0.72, 0.15, 0.010, 0.003, 0.0, 0.0], dtype=np.float64),
}
for _k in OBJECT_DISTS:
    OBJECT_DISTS[_k] /= OBJECT_DISTS[_k].sum()


def pick_baseline_tier(rng):
    return str(rng.choice(['clean', 'normal', 'hazy'], p=[0.30, 0.50, 0.20]))


def pick_intensity_tier(rng):
    return str(rng.choice(['strong', 'mid', 'weak'], p=[0.50, 0.30, 0.20]))


def shifted_object_dist(obj, tier):
    # grade shift is 0 for all tiers in the validated spec -> identity.
    return OBJECT_DISTS[obj]


# ============================================================================
# Alpha-field grid + profile helpers
# ============================================================================
XC, YC = np.meshgrid(np.arange(CHIP), np.arange(CHIP), indexing='xy')
CHIP_BASE_ALPHA = 0.0


def _perp_profile(d, sigma_s=1.5, sigma_m=3.0, sigma_w=6.0):
    strong = 1.00 * np.exp(-d**2 / (2 * sigma_s**2)).astype(np.float32)
    med = 0.55 * np.exp(-d**2 / (2 * sigma_m**2)).astype(np.float32)
    weak = 0.25 * np.exp(-d**2 / (2 * sigma_w**2)).astype(np.float32)
    return np.minimum(strong + med + weak, 1.0)


def _perp_asym_chip(d, sigma_left, sigma_right):
    sigma = np.where(d < 0, sigma_left, sigma_right).astype(np.float32)
    return (1.0 / (1.0 + (d / sigma) ** 2)).astype(np.float32)


def _along_smooth_1d(rng, n_along, length):
    var = rng.uniform(0.40, 1.0, size=n_along).astype(np.float32)
    pos = np.linspace(0, n_along - 1, length, dtype=np.float32)
    floor = np.clip(np.floor(pos).astype(np.int32), 0, n_along - 2)
    frac = (pos - floor).astype(np.float32)
    return (1 - frac) * var[floor] + frac * var[floor + 1]


def _along_taper_mask(length, line_start, line_end, fade_len=25.0):
    y = np.arange(length, dtype=np.float32)
    rise = np.clip((y - (line_start - fade_len)) / fade_len, 0.0, 1.0)
    fall = np.clip(((line_end + fade_len) - y) / fade_len, 0.0, 1.0)
    return (rise * fall).astype(np.float32)


def _along_smooth_range_1d(rng, length, lo, hi, n_segments=8):
    var = rng.uniform(lo, hi, size=n_segments).astype(np.float32)
    pos = np.linspace(0, n_segments - 1, length, dtype=np.float32)
    floor = np.clip(np.floor(pos).astype(np.int32), 0, n_segments - 2)
    frac = (pos - floor).astype(np.float32)
    return ((1 - frac) * var[floor] + frac * var[floor + 1]).astype(np.float32)


def _along_wobble_1d(rng, length, amp=2.0, n_segments=8):
    var = rng.uniform(-amp, amp, size=n_segments).astype(np.float32)
    pos = np.linspace(0, n_segments - 1, length, dtype=np.float32)
    floor = np.clip(np.floor(pos).astype(np.int32), 0, n_segments - 2)
    frac = (pos - floor).astype(np.float32)
    return ((1 - frac) * var[floor] + frac * var[floor + 1]).astype(np.float32)


def _along_center_peak(rng, length, line_start, line_end, sigma_factor=None, floor=0.0):
    if sigma_factor is None:
        sigma_factor = float(rng.uniform(1.6, 3.2))
    half = (line_end - line_start) / 2
    cy = (line_start + line_end) / 2 + float(rng.uniform(-half * 0.55, half * 0.55))
    sigma = max(half / sigma_factor, 8.0)
    yc = np.arange(length, dtype=np.float32)
    center_peak = np.exp(-((yc - cy) ** 2) / (2 * sigma * sigma)).astype(np.float32)
    smooth = _along_smooth_1d(rng, 8, length)
    env = (center_peak * (0.5 + 0.5 * smooth)).astype(np.float32)
    if floor > 0:
        env = np.maximum(env, floor)
    return env.astype(np.float32)


# ============================================================================
# 4 single-defect alpha fields
# ============================================================================
def _bank_line(rng, cx, weak_only):
    """bb line: LEFT-RIGHT SYMMETRIC shading + along-line (up-down) intensity
    distribution that varies but keeps a high floor so the line NEVER breaks:
      sharp core (-> grade-2 green, slightly widened peak) + faint symmetric wide
      smear (-> grade-1 grey shading both sides). Vertical (column cx)."""
    sigma_core = float(rng.uniform(1.7, 2.6)) if not weak_only else float(rng.uniform(1.3, 2.0))  # thicker green peak
    sigma_smear = float(rng.uniform(12.0, 26.0))     # wider symmetric shading half-width (음영 폭)
    smear_w = float(rng.uniform(0.30, 0.50))         # faint -> grey shading, not green
    core_j = _along_smooth_range_1d(rng, CHIP, 0.8, 1.3)[:, None]
    smear_j = _along_smooth_range_1d(rng, CHIP, 0.7, 1.3)[:, None]
    wobble_amp = float(rng.uniform(0.3, 0.8)) if not weak_only else float(rng.uniform(0.2, 0.6))
    cx_per_y = (cx + _along_wobble_1d(rng, CHIP, amp=wobble_amp, n_segments=8))[:, None]
    ad = np.abs((XC - cx_per_y).astype(np.float32))
    core = 1.0 / (1.0 + (ad / (sigma_core * core_j)) ** 2)
    smear = smear_w / (1.0 + (ad / (sigma_smear * smear_j)) ** 2)
    line = np.minimum(core + smear, 1.0).astype(np.float32)
    # along-line up-down distribution: center-peak with HIGH floor -> textured but never breaks
    strength = _along_center_peak(rng, CHIP, 0.0, float(CHIP),
                                  floor=float(rng.uniform(0.66, 0.80)),   # weak part raised (less fade)
                                  sigma_factor=float(rng.uniform(1.4, 2.6)))[:, None]
    return (line * strength).astype(np.float32)


def alpha_bank_boundary(rng):
    """bb = 3 vertical lines ALIGNED in a row (50/100/150) + ONE horizontal line
    (= a vertical line rotated 90 deg). Each line: symmetric smear, full-length
    continuous (does not break mid-line)."""
    weak_only = rng.random() < 0.18
    a = np.full((CHIP, CHIP), CHIP_BASE_ALPHA, dtype=np.float32)
    for cx in (50, 100, 150):                                  # ㅣ aligned, continuous
        a = np.maximum(a, _bank_line(rng, cx, weak_only))
    a = np.maximum(a, np.rot90(_bank_line(rng, 100, weak_only)).astype(np.float32))   # ㅡ rot 90
    return a


def alpha_fork(rng):
    a = np.full((CHIP, CHIP), CHIP_BASE_ALPHA, dtype=np.float32)
    weak_only = rng.random() < 0.02
    smear_factor = float(rng.uniform(4.0, 7.0))
    severity = float(rng.uniform(0.98, 1.0))
    cy_h = float(rng.uniform(30, 130))
    sigma_h_base = float(rng.uniform(1.8, 2.5)) if not weak_only else float(rng.uniform(1.5, 2.0))
    sigma_h_jitter = _along_smooth_range_1d(rng, CHIP, 0.7, 1.4)[None, :]
    sigma_h_top = (sigma_h_base * sigma_h_jitter).astype(np.float32)
    smear_factor_jitter_h = _along_smooth_range_1d(rng, CHIP, 0.5, 1.5)[None, :]
    smear_factor_h = (smear_factor * smear_factor_jitter_h).astype(np.float32)
    fork_width = float(rng.uniform(80, 130)) if not weak_only else float(rng.uniform(70, 100))
    fork_x_center = float(rng.uniform(fork_width / 2 + 15, CHIP - fork_width / 2 - 15))
    x_lo = fork_x_center - fork_width / 2
    x_hi = fork_x_center + fork_width / 2
    fade_x = float(rng.uniform(15, 25))
    taper_x = _along_taper_mask(CHIP, x_lo, x_hi, fade_len=fade_x)
    in_x = taper_x[None, :]
    wobble_amp_h = float(rng.uniform(0.3, 0.8)) if not weak_only else float(rng.uniform(0.2, 0.6))
    cy_h_wobble = cy_h + _along_wobble_1d(rng, CHIP, amp=wobble_amp_h, n_segments=8)
    cy_per_x = cy_h_wobble[None, :]
    along_x_smear_1d = _along_center_peak(rng, CHIP, x_lo, x_hi,
                                          floor=float(rng.uniform(0.18, 0.25)),
                                          sigma_factor=float(rng.uniform(2.5, 4.0)))
    along_x_strength_1d = _along_center_peak(rng, CHIP, x_lo, x_hi,
                                             floor=float(rng.uniform(0.78, 0.85)),
                                             sigma_factor=float(rng.uniform(1.4, 2.5)))
    along_x_smear = along_x_smear_1d[None, :]
    along_x_strength = along_x_strength_1d[None, :]
    sigma_h_bot_arr = (sigma_h_top * (1.0 + (smear_factor_h - 1.0) * along_x_smear)).astype(np.float32)
    s_h = (float(rng.uniform(0.55, 0.65)) if weak_only else 1.0) * severity
    d_perp_h = (YC - cy_per_x).astype(np.float32)
    perp_h = _perp_asym_chip(d_perp_h, sigma_h_top, sigma_h_bot_arr)
    line_h = perp_h * in_x * s_h * along_x_strength
    a = np.maximum(a, line_h)
    n_legs = int(rng.integers(7, 10))
    leg_xs_base = np.linspace(x_lo + 8, x_hi - 8, n_legs)
    leg_xs_jit = rng.uniform(-3, 3, size=n_legs).astype(np.float32)
    leg_xs = sorted((leg_xs_base + leg_xs_jit).tolist())
    leg_strengths = rng.uniform(0.88, 1.0, size=n_legs).astype(np.float32)
    leg_smear_bases = rng.uniform(0.5, 1.5, size=n_legs).astype(np.float32)
    if weak_only:
        n_keep = int(rng.integers(1, min(4, n_legs + 1)))
        keep_idx = set(rng.choice(n_legs, size=n_keep, replace=False).tolist())
        for k in range(n_legs):
            if k not in keep_idx:
                leg_strengths[k] = 0.0
    leg_height = float(rng.uniform(70, 130)) if not weak_only else float(rng.uniform(60, 100))
    leg_y_end = min(cy_h + leg_height, float(CHIP - 5))
    fade_y = float(rng.uniform(15, 25))
    taper_y = _along_taper_mask(CHIP, cy_h, leg_y_end, fade_len=fade_y)
    in_y = taper_y[:, None]
    for leg_idx, (cx, s_leg) in enumerate(zip(leg_xs, leg_strengths)):
        if s_leg < 0.10:
            continue
        sigma_v_base = float(rng.uniform(1.7, 2.4)) if not weak_only else float(rng.uniform(1.4, 2.0))
        smear_factor_leg = smear_factor * float(leg_smear_bases[leg_idx])
        sigma_v_jitter = _along_smooth_range_1d(rng, CHIP, 0.7, 1.4)[:, None]
        sigma_v_left = (sigma_v_base * sigma_v_jitter).astype(np.float32)
        smear_factor_jitter_v = _along_smooth_range_1d(rng, CHIP, 0.5, 1.5)[:, None]
        smear_factor_v = (smear_factor_leg * smear_factor_jitter_v).astype(np.float32)
        wobble_amp_v = float(rng.uniform(0.3, 0.8)) if not weak_only else float(rng.uniform(0.2, 0.6))
        cx_wobble = cx + _along_wobble_1d(rng, CHIP, amp=wobble_amp_v, n_segments=8)
        cx_per_y = cx_wobble[:, None]
        s_v_base = float(rng.uniform(0.55, 0.65)) if weak_only else 1.0
        s_v = severity * float(s_leg) * s_v_base
        along_y_smear_1d = _along_center_peak(rng, CHIP, cy_h, leg_y_end,
                                              floor=float(rng.uniform(0.18, 0.25)),
                                              sigma_factor=float(rng.uniform(2.5, 4.0)))
        along_y_strength_1d = _along_center_peak(rng, CHIP, cy_h, leg_y_end,
                                                 floor=float(rng.uniform(0.78, 0.85)),
                                                 sigma_factor=float(rng.uniform(1.4, 2.5)))
        along_y_smear = along_y_smear_1d[:, None]
        along_y_strength = along_y_strength_1d[:, None]
        sigma_v_right_arr = (sigma_v_left * (1.0 + (smear_factor_v - 1.0) * along_y_smear)).astype(np.float32)
        d_perp_v = (XC - cx_per_y).astype(np.float32)
        perp_v = _perp_asym_chip(d_perp_v, sigma_v_left, sigma_v_right_arr)
        line_v = perp_v * in_y * s_v * along_y_strength
        a = np.maximum(a, line_v)
    return a


def _scratch_line(rng, cx, weak_only, smear_factor):
    """ONE vertical scratch line at column cx — scratch's exact per-line rendering
    (asymmetric smear, along-line taper/center-peak/wobble). Returns (CHIP,CHIP) alpha."""
    y_center = float(rng.uniform(50, 150))
    y_half = float(rng.uniform(35, 95))
    line_y_start = max(0.0, y_center - y_half)
    line_y_end = min(float(CHIP), y_center + y_half)
    line_strength = float(rng.uniform(0.96, 1.00))
    smear_base = float(rng.uniform(0.5, 1.5))
    fade_len = float(rng.uniform(15, 28))
    in_y_range = _along_taper_mask(CHIP, line_y_start, line_y_end, fade_len=fade_len)[:, None]
    sigma_left_base = float(rng.uniform(0.5, 0.8)) if not weak_only else float(rng.uniform(0.5, 0.7))
    smear_factor_line = smear_factor * smear_base
    sigma_left_per_y = (sigma_left_base * _along_smooth_range_1d(rng, CHIP, 0.7, 1.4)[:, None]).astype(np.float32)
    smear_factor_per_y = (smear_factor_line * _along_smooth_range_1d(rng, CHIP, 0.5, 1.5)[:, None]).astype(np.float32)
    wobble_amp = float(rng.uniform(0.3, 0.8)) if not weak_only else float(rng.uniform(0.2, 0.6))
    cx_per_y = (cx + _along_wobble_1d(rng, CHIP, amp=wobble_amp, n_segments=8))[:, None]
    along_y_smear = _along_center_peak(rng, CHIP, line_y_start, line_y_end,
                                       floor=float(rng.uniform(0.18, 0.25)),
                                       sigma_factor=float(rng.uniform(2.5, 4.0)))[:, None]
    along_y_strength = _along_center_peak(rng, CHIP, line_y_start, line_y_end,
                                          floor=float(rng.uniform(0.78, 0.85)),
                                          sigma_factor=float(rng.uniform(1.4, 2.5)))[:, None]
    sigma_right_arr = (sigma_left_per_y * (1.0 + (smear_factor_per_y - 1.0) * along_y_smear)).astype(np.float32)
    perp = _perp_asym_chip((XC - cx_per_y).astype(np.float32), sigma_left_per_y, sigma_right_arr)
    return (perp * in_y_range * line_strength * along_y_strength).astype(np.float32)


def alpha_scratch(rng):
    a = np.full((CHIP, CHIP), CHIP_BASE_ALPHA, dtype=np.float32)
    weak_only = rng.random() < 0.18
    n_lines = int(rng.integers(5, 11))
    cx_list = sorted(rng.uniform(15, 185, size=n_lines).tolist())
    smear_factor = float(rng.uniform(35.0, 50.0)) if weak_only else float(rng.uniform(35.0, 60.0))
    for cx in cx_list:
        a = np.maximum(a, _scratch_line(rng, cx, weak_only, smear_factor))
    return a


SCRATCH_ROT_ANGLE = -21.0   # degrees; scratch_rot identity (distinguishes from vertical scratch)


def alpha_scratch_rot(rng):
    """scratch_rot = the vertical `scratch` alpha field ROTATED by -21 deg, so it
    inherits scratch's strong sideways (asymmetric) smear/shading. The angle is the
    only thing that separates it from `scratch` (rotation/flip aug stays forbidden).
    Larger working canvas before rotate so the diagonal lines fill the chip (no
    empty rotated corners)."""
    big = int(CHIP * 1.5)                       # 300 — pad so rotation doesn't clip line ends
    pad = (big - CHIP) // 2
    # build a vertical-scratch alpha on the big canvas by tiling the standard field
    a = alpha_scratch(rng)                      # (CHIP, CHIP) vertical scratch, strong smear
    canvas = np.zeros((big, big), dtype=np.float32)
    canvas[pad:pad + CHIP, pad:pad + CHIP] = a
    im = Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8))
    im = im.rotate(SCRATCH_ROT_ANGLE, resample=Image.BILINEAR, center=(big / 2, big / 2), fillcolor=0)
    rot = np.asarray(im, dtype=np.float32)[pad:pad + CHIP, pad:pad + CHIP] / 255.0
    return np.clip(rot, 0.0, 1.0).astype(np.float32)


ALPHA_FNS = {
    'bank_boundary': alpha_bank_boundary,
    'fork': alpha_fork,
    'scratch': alpha_scratch,
    'scratch_rot': alpha_scratch_rot,
}
SINGLE_DEFECTS = ['bank_boundary', 'fork', 'scratch', 'scratch_rot']
COMBO_2 = ['bank_boundary+fork', 'bank_boundary+scratch', 'bank_boundary+scratch_rot',
           'fork+scratch', 'fork+scratch_rot', 'scratch+scratch_rot']
OOD_CLASSES = ['CenterDonut', 'CrossScratch', 'DiagonalSmear', 'Starburst']
SPECIAL = ['Normal', 'Invalid']


# ============================================================================
# alpha -> per-pixel grade -> palette chip
# ============================================================================
def _try_font(size):
    for path in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def render_single_chip(obj, rng, intensity_tier=None, bin_id=None, add_border=True, bg_range=None):
    """Render one 200x200 mode='P' palette chip for a single defect class.
    bg_range overrides the background noise range (combos pass a brighter range)."""
    if intensity_tier is None:
        intensity_tier = pick_intensity_tier(rng)
    if bin_id is None:
        bin_id = int(rng.choice(DEFECT_BIN_POOL['00C'], p=DEFECT_BIN_WEIGHTS))

    bg = _normal_bg(rng, (CHIP, CHIP),
                    bg_range or SINGLE_BG_RANGE_BY_KEY.get(obj, SINGLE_BG_RANGE))   # per-chip noise bg

    alpha_scale = INTENSITY_ALPHA_SCALE[intensity_tier]
    alpha = ALPHA_FNS[obj](rng) * alpha_scale * DEFECT_ALPHA_BOOST.get(obj, 1.0)
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    cum_obj = np.cumsum(shifted_object_dist(obj, intensity_tier))

    # all 4 single defects use the same continuous 2-stage rule:
    #   is_defect = u < alpha (stochastic), defect grade 2 near line core else grade 1 (+rare 3/4),
    #   everything else = clean per-chip noise background. No hard region cut.
    grades_base = bg
    u1 = rng.random((CHIP, CHIP))
    is_defect = u1 < alpha
    if obj == 'fork':
        lo_t2, hi_t2 = 0.53, 0.90
    elif obj == 'bank_boundary':
        lo_t2, hi_t2 = 0.42, 0.84          # even more/denser grade-2 green at the peak
    else:                                  # scratch, scratch_rot
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

    canvas = grades
    if add_border:
        border_color = BIN_TO_BORDER_IDX.get(bin_id, KEY_TO_INDEX['border_etc'])
        canvas[:2, :] = border_color
        canvas[-2:, :] = border_color
        canvas[:, :2] = border_color
        canvas[:, -2:] = border_color

    img = Image.frombytes('P', (CHIP, CHIP), canvas.tobytes())
    img.putpalette(PALETTE)
    return img


def render_combo_chip(key, rng):
    """2-combo = pixel-wise RGB min of two freshly rendered single-defect chips
    (min keeps the darker/stronger defect per pixel -> 'still a defect, not white')."""
    a, b = key.split('+')
    rng_bg = COMBO_BG_RANGE_BY_KEY.get(key, COMBO_BG_RANGE)
    img_a = render_single_chip(a, rng, bg_range=rng_bg).convert('RGB')
    img_b = render_single_chip(b, rng, bg_range=rng_bg).convert('RGB')
    arr = np.minimum(np.asarray(img_a), np.asarray(img_b)).astype(np.uint8)
    return Image.fromarray(arr, mode='RGB')


# Per-chip background noise density ranges. The too-bright / too-clean extreme is
# excluded (user 260527): every chip background has visible noise.
#   Normal / OOD : moderate-dense.
#   single       : wider, extended toward dark.
#   combo        : per-single BRIGHTER — min-blend of two combines noise (~p1+p2),
#                  so each piece is rendered light to keep the blended combo from
#                  going too dark.
BG_NOISE_RANGE = (0.30, 0.62)          # Normal, OOD (noise IS the signal here)
SINGLE_BG_RANGE = (0.12, 0.28)         # fork / scratch / scratch_rot: a bit more background noise
SINGLE_BG_RANGE_BY_KEY = {             # bank_boundary keeps the lighter background (unchanged)
    'bank_boundary': (0.08, 0.22),
}
COMBO_BG_RANGE = (0.06, 0.14)          # per-single for combos: light (min-blend was too noisy)
COMBO_BG_RANGE_BY_KEY = {              # bb+scratch_rot lighter still
    'bank_boundary+scratch_rot': (0.05, 0.11),
}

# per-class defect-signal boost — defects must stay STRONG (real defect was getting
# buried under background noise). scratch / scratch_rot restored to full; fork kept up.
DEFECT_ALPHA_BOOST = {'fork': 1.15}


def _normal_bg(rng, shape=(CHIP, CHIP), bg_range=BG_NOISE_RANGE):
    """Per-chip noise background (Normal mechanism), density sampled uniformly from
    bg_range so it is never too bright/clean: grade 0 (white) with prob 1-p_noise,
    else grade 1 (95%) / grade 2 (5%). Used as the non-defect background for ALL
    classes (defect / combo / OOD / Normal)."""
    p_noise = float(rng.uniform(*bg_range))
    u = rng.random(shape)
    is_noise = u < p_noise
    u2 = rng.random(shape)
    noise_grade = np.where(u2 < 0.95, 1, 2).astype(np.uint8)
    return np.where(is_noise, noise_grade, 0).astype(np.uint8)


def render_normal_chip(rng):
    """Normal = diverse per-chip Beta(2,10) noise background (no defect)."""
    grades = _normal_bg(rng, (CHIP, CHIP))
    img = Image.frombytes('P', (CHIP, CHIP), grades.tobytes())
    img.putpalette(PALETTE)
    return img


def render_invalid_chip(rng):
    """Invalid = white interior + 2px orange border (idx 11) + centered black bin text."""
    grades = np.zeros((CHIP, CHIP), dtype=np.uint8)
    border_idx = KEY_TO_INDEX['border_inv']
    text_idx = KEY_TO_INDEX['text']
    grades[:2, :] = border_idx
    grades[-2:, :] = border_idx
    grades[:, :2] = border_idx
    grades[:, -2:] = border_idx
    img = Image.frombytes('P', (CHIP, CHIP), grades.tobytes())
    img.putpalette(PALETTE)
    draw = ImageDraw.Draw(img)
    font = _try_font(64)
    text = f"B{int(rng.integers(200, 300))}"
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = CHIP // 2 - tw // 2 - bbox[0]
        ty = CHIP // 2 - th // 2 - bbox[1]
    except Exception:
        tw, th = 120, 50
        tx, ty = CHIP // 2 - tw // 2, CHIP // 2 - th // 2
    draw.text((tx, ty), text, fill=int(text_idx), font=font)
    return img


# ============================================================================
# OOD chips — wafer-scale pattern, chip-tile extraction (NO full 6400^2 array).
#
# A wafer-level pattern (donut ring / cross / diagonal line / starburst rays) is
# defined in 6400x6400 wafer coordinates centred at SIZE/2. We never allocate the
# full wafer: for each 200x200 chip tile we evaluate the pattern alpha over only
# that tile's wafer-coordinate window. ONE pattern realization yields MANY chips
# (every grid cell the pattern passes through), matching the reference v15direct
# OOD distribution (mostly faint pattern crops, a few strong) — but memory-cheap.
# ============================================================================
SIZE = 6400
GRID = 32

CLASS_PEAK_DIST = {
    "DiagonalSmear": [0.00, 0.60, 0.28, 0.10, 0.02, 0.00, 0.00, 0.00],
    "CrossScratch":  [0.00, 0.40, 0.30, 0.18, 0.08, 0.03, 0.01, 0.00],
    "CenterDonut":   [0.00, 0.50, 0.30, 0.13, 0.05, 0.02, 0.00, 0.00],
    "Starburst":     [0.00, 0.40, 0.28, 0.18, 0.10, 0.03, 0.01, 0.00],
}


def _build_peak_cum(cls):
    d = np.array(CLASS_PEAK_DIST[cls], dtype=np.float64)
    d /= d.sum()
    return np.cumsum(d).astype(np.float32)


def _perp_sharp_canvas(d, sigma):
    """Wafer-scale sharp Lorentzian peak (0.5 sigma) + wide heavy tail (5 sigma, w 0.6)."""
    sharp = 1.0 / (1.0 + (d / (0.5 * sigma)) ** 2)
    wide = 0.60 / (1.0 + (d / (5.0 * sigma)) ** 2)
    return np.minimum(sharp + wide, 1.0).astype(np.float32)


def _along_taper_canvas(d_along, half_len, sigma_end):
    abs_d = np.abs(d_along)
    return np.where(abs_d > half_len,
                    np.exp(-((abs_d - half_len) ** 2) / (sigma_end * sigma_end)),
                    1.0).astype(np.float32)


def _sample_ood_params(cls, rng):
    """Sample FULL-WAFER-scale geometry once per realization (centred at SIZE/2).

    Patterns span most of the 6400px wafer so each 200px chip is a small, nearly
    flat crop (a faint line/arc segment) — the real 'slice the wafer into puzzle
    pieces' look. Line width (sigma) stays chip-scale (~20-40px) so a crop shows a
    thin defect streak, not a whole shrunk pattern."""
    R = SIZE / 2.0
    cyw = cxw = SIZE / 2.0
    if cls == "DiagonalSmear":
        return dict(angle=(45.0 + rng.uniform(-8, 8)) * np.pi / 180,
                    cy=cyw + rng.uniform(-R * 0.15, R * 0.15),
                    cx=cxw + rng.uniform(-R * 0.15, R * 0.15),
                    sigma=CHIP * rng.uniform(0.10, 0.22), peak=rng.uniform(0.20, 0.55),
                    half_len=SIZE * rng.uniform(0.32, 0.48), sigma_end=CHIP * rng.uniform(2.0, 4.0))
    if cls == "CrossScratch":
        return dict(base_angle=rng.uniform(-0.05, 0.05),
                    cy=cyw + rng.uniform(-R * 0.12, R * 0.12),
                    cx=cxw + rng.uniform(-R * 0.12, R * 0.12),
                    sigma=CHIP * rng.uniform(0.10, 0.22), peak=rng.uniform(0.20, 0.55),
                    half_len=SIZE * rng.uniform(0.30, 0.45), sigma_end=CHIP * rng.uniform(2.0, 3.5))
    if cls == "CenterDonut":
        # big wafer-spanning ring: radius 40-85% of wafer -> ring crosses many chips,
        # each chip sees a nearly-straight arc segment (low curvature over 200px).
        return dict(cy=cyw, cx=cxw, r_center=R * rng.uniform(0.40, 0.85),
                    sigma_r=CHIP * rng.uniform(0.10, 0.22), peak=rng.uniform(0.20, 0.55))
    if cls == "Starburst":
        n_rays = int(rng.integers(8, 14))
        th_off = rng.uniform(0, 2 * np.pi)
        return dict(cy=cyw, cx=cxw, R=R, r_ring=R * rng.uniform(0.05, 0.12),
                    sigma_ring=CHIP * rng.uniform(0.10, 0.20), n_rays=n_rays, th_off=th_off,
                    ray_angles=[th_off + 2 * np.pi * i / n_rays + rng.uniform(-0.05, 0.05)
                                for i in range(n_rays)],
                    sigma_perp=CHIP * rng.uniform(0.08, 0.16), ray_len=R * rng.uniform(0.70, 0.95),
                    peak_c=rng.uniform(0.40, 0.85), peak_l=rng.uniform(0.20, 0.55))
    raise ValueError(f"unknown OOD class: {cls}")


def _ood_alpha_on(cls, p, yy, xx):
    """Pattern alpha over a tile's wafer-coordinate window (yy, xx 2D arrays)."""
    if cls == "DiagonalSmear":
        cos_a, sin_a = np.cos(p["angle"]), np.sin(p["angle"])
        d_perp = cos_a * (yy - p["cy"]) - sin_a * (xx - p["cx"])
        d_along = sin_a * (yy - p["cy"]) + cos_a * (xx - p["cx"])
        return p["peak"] * _perp_sharp_canvas(d_perp, p["sigma"]) * \
            _along_taper_canvas(d_along, p["half_len"], p["sigma_end"])
    if cls == "CrossScratch":
        out = np.zeros_like(yy, dtype=np.float32)
        for ang in (p["base_angle"], p["base_angle"] + np.pi / 2):
            cos_a, sin_a = np.cos(ang), np.sin(ang)
            d_perp = cos_a * (yy - p["cy"]) - sin_a * (xx - p["cx"])
            d_along = sin_a * (yy - p["cy"]) + cos_a * (xx - p["cx"])
            a = p["peak"] * _perp_sharp_canvas(d_perp, p["sigma"]) * \
                _along_taper_canvas(d_along, p["half_len"], p["sigma_end"])
            out = np.maximum(out, a)
        return out
    if cls == "CenterDonut":
        r = np.sqrt((yy - p["cy"]) ** 2 + (xx - p["cx"]) ** 2)
        return p["peak"] * _perp_sharp_canvas(r - p["r_center"], p["sigma_r"])
    if cls == "Starburst":
        dy = yy - p["cy"]; dx = xx - p["cx"]
        r = np.sqrt(dy * dy + dx * dx)
        theta = np.arctan2(dy, dx)
        a_center = _perp_sharp_canvas(r - p["r_ring"], p["sigma_ring"])
        a_lines = np.zeros_like(yy, dtype=np.float32)
        for ray_angle in p["ray_angles"]:
            dth = (theta - ray_angle + np.pi) % (2 * np.pi) - np.pi
            d_perp = r * np.sin(dth)
            in_len = (r < p["ray_len"]) & (np.cos(dth) > 0)
            a_ray = _perp_sharp_canvas(d_perp, p["sigma_perp"]) * in_len.astype(np.float32)
            a_lines = np.maximum(a_lines, a_ray)
        return np.maximum(p["peak_c"] * a_center, p["peak_l"] * a_lines)
    raise ValueError(f"unknown OOD class: {cls}")


def _bilinear_tile(rng, hc, wc, lo, hi):
    """Coarse (hc x wc) uniform field bilinearly upsampled to CHIP x CHIP."""
    coarse = rng.uniform(lo, hi, size=(hc, wc)).astype(np.float32)
    yi = np.linspace(0, hc - 1, CHIP, dtype=np.float32)
    xi = np.linspace(0, wc - 1, CHIP, dtype=np.float32)
    y0 = np.clip(np.floor(yi).astype(np.int32), 0, hc - 2)
    x0 = np.clip(np.floor(xi).astype(np.int32), 0, wc - 2)
    fy = (yi - y0)[:, None]; fx = (xi - x0)[None, :]
    return ((1 - fy) * ((1 - fx) * coarse[y0][:, x0] + fx * coarse[y0][:, x0 + 1]) +
            fy * ((1 - fx) * coarse[y0 + 1][:, x0] + fx * coarse[y0 + 1][:, x0 + 1])).astype(np.float32)


def iter_ood_chips(cls, seed):
    """Yield (gy, gx, PIL 'P' chip) for ONE wafer-pattern realization — every grid
    cell the pattern passes through (faint + strong), without allocating the full
    wafer. The yielded chips are the puzzle pieces of the sliced wafer pattern:
    placing each at its (gy, gx) reconstructs the big donut/cross/line/starburst."""
    rng = np.random.default_rng(seed)
    baseline_tier = pick_baseline_tier(rng)
    intensity_tier = pick_intensity_tier(rng)
    a_scale = INTENSITY_ALPHA_SCALE[intensity_tier]
    canvas_tau = float(rng.choice([0.20, 0.30, 0.40], p=[0.25, 0.50, 0.25]))
    cum_base = CUM_BASELINE_TIERS[baseline_tier].astype(np.float32)
    cum_peak = _build_peak_cum(cls)
    p = _sample_ood_params(cls, rng)
    kind = '00P' if rng.random() < 0.5 else '00C'
    bin_pool = DEFECT_BIN_POOL[kind]
    R = SIZE / 2.0
    base_y = np.arange(CHIP, dtype=np.float32)[:, None]
    base_x = np.arange(CHIP, dtype=np.float32)[None, :]
    for gy in range(GRID):
        for gx in range(GRID):
            ccy = (gy + 0.5) * CHIP; ccx = (gx + 0.5) * CHIP
            if (ccy - SIZE / 2.0) ** 2 + (ccx - SIZE / 2.0) ** 2 > R * R:
                continue
            yy = base_y + gy * CHIP
            xx = base_x + gx * CHIP
            raw = _ood_alpha_on(cls, p, yy, xx) * a_scale
            if raw.max() < 0.01:
                continue                                          # pattern absent here -> skip cheaply
            alpha = raw * _bilinear_tile(rng, 2, 2, 0.50, 1.50) * _bilinear_tile(rng, 4, 4, 0.65, 1.40)
            alpha = np.clip(alpha + rng.normal(0, 0.04, (CHIP, CHIP)).astype(np.float32), 0.0, 1.0)
            amax = float(alpha.max()); amean = float(alpha.mean())
            if amax < canvas_tau:
                continue
            p_def = min(max(amean * 3.0, (amax - 0.5) * 1.5), 1.0)
            if rng.random() > p_def:
                continue
            chip = _normal_bg(rng, (CHIP, CHIP))   # diverse per-chip noise background (Normal mechanism)
            cum_mixed = ((1 - alpha)[..., None] * cum_base[None, None, :] +
                         alpha[..., None] * cum_peak[None, None, :])
            uu = rng.random((CHIP, CHIP)).astype(np.float32)
            grades = (uu[..., None] < cum_mixed).argmax(axis=-1).astype(np.uint8)
            mask = alpha > 0.01
            chip[mask] = grades[mask]
            b = int(rng.choice(bin_pool, p=DEFECT_BIN_WEIGHTS))
            border = BIN_TO_BORDER_IDX.get(b, KEY_TO_INDEX['border_etc'])
            chip[:2, :] = border; chip[-2:, :] = border
            chip[:, :2] = border; chip[:, -2:] = border
            img = Image.frombytes('P', (CHIP, CHIP), chip.tobytes())
            img.putpalette(PALETTE)
            yield gy, gx, img


def render_ood_chip(cls, rng):
    """Single OOD chip (one piece from a fresh pattern realization). For bulk gen,
    prefer iter_ood_chips (many pieces per pattern). Retries until a piece exists."""
    for _ in range(200):
        seed = int(rng.integers(0, 2**31 - 1))
        for _gy, _gx, img in iter_ood_chips(cls, seed):
            return img
    return render_normal_chip(rng)   # extreme fallback


# ============================================================================
# Unified dispatch
# ============================================================================
def render_chip(class_key, rng):
    """Render any eval class_key -> PIL Image (mode P for single/OOD/special, RGB for combo)."""
    if class_key in SINGLE_DEFECTS:
        return render_single_chip(class_key, rng)
    if class_key in COMBO_2:
        return render_combo_chip(class_key, rng)
    if class_key in OOD_CLASSES:
        return render_ood_chip(class_key, rng)
    if class_key == 'Normal':
        return render_normal_chip(rng)
    if class_key == 'Invalid':
        return render_invalid_chip(rng)
    raise ValueError(f"unknown class_key: {class_key}")
