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
    build_tasks,
    save_chip_crops,
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
    a = torch.full((CHIP, CHIP), CHIP_BASE_ALPHA, device=DEVICE, dtype=DTYPE)
    n_seg = 10; seg_len = CHIP // n_seg
    for cx in [50, 100, 150]:
        s = float(rng.uniform(0.90, 1.0))
        seg = torch.from_numpy(rng.uniform(0.55, 1.00, size=n_seg).astype(np.float32)).to(DEVICE)
        y_noise = seg.repeat_interleave(seg_len)[:CHIP].unsqueeze(1)
        line = _perp_profile_t(XC_2D_T - cx, 0.7, 3.0, 12.0) * y_noise * s
        a = torch.maximum(a, line)
    for cy in [100]:
        s = float(rng.uniform(0.90, 1.0))
        seg = torch.from_numpy(rng.uniform(0.55, 1.00, size=n_seg).astype(np.float32)).to(DEVICE)
        x_noise = seg.repeat_interleave(seg_len)[:CHIP].unsqueeze(0)
        line = _perp_profile_t(YC_2D_T - cy, 0.7, 3.0, 12.0) * x_noise * s
        a = torch.maximum(a, line)
    return a

def alpha_particle_blast_t(rng):
    a = torch.full((CHIP, CHIP), CHIP_BASE_ALPHA, device=DEVICE, dtype=DTYPE)
    cx = float(rng.uniform(50, 150)); cy = float(rng.uniform(50, 150))
    sigma = float(rng.uniform(22, 35))
    blob = torch.exp(-((XC_2D_T - cx)**2 + (YC_2D_T - cy)**2) / (2*sigma**2))
    return torch.maximum(a, blob)

def alpha_scratch_t(rng):
    a = torch.full((CHIP, CHIP), CHIP_BASE_ALPHA, device=DEVICE, dtype=DTYPE)
    n_lines = int(rng.integers(5, 16))
    for _ in range(n_lines):
        cx = float(rng.uniform(15, 185))
        y_start = float(rng.uniform(0, 80)); y_end = float(rng.uniform(120, 200))
        in_range = ((YC_2D_T >= y_start) & (YC_2D_T <= y_end)).to(DTYPE)
        s = float(rng.uniform(0.80, 1.0))
        line = _perp_profile_t(XC_2D_T - cx, 1.0, 2.0, 4.0) * in_range * s
        a = torch.maximum(a, line)
    return a

def alpha_scratch_21deg_t(rng):
    a = torch.full((CHIP, CHIP), CHIP_BASE_ALPHA, device=DEVICE, dtype=DTYPE)
    n_lines = int(rng.integers(12, 19))
    theta = np.deg2rad(21.0)
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
    cy = 100; span_lo, span_hi = 15, 185
    spacing = (span_hi - span_lo) / (n_lines + 1)
    for i in range(n_lines):
        cx = span_lo + (i + 1) * spacing
        d_perp = (XC_2D_T - cx) * cos_t - (YC_2D_T - cy) * sin_t
        s = float(rng.uniform(0.92, 1.0))
        line = _perp_profile_t(d_perp, 0.7, 1.5, 3.0) * s
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

ALPHA_FNS_T = {
    'bank_boundary':    alpha_bank_boundary_t,
    'particle_blast':   alpha_particle_blast_t,
    'scratch':          alpha_scratch_t,
    'scratch_21deg':    alpha_scratch_21deg_t,
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

WAFER_CANVAS_PATTERNS = {
    'Starburst':    make_starburst_alpha,
    'CommaCluster': make_commacluster_alpha,
}

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
                        obj_actual = pick_mixed_object(object_name, rng, mix_ratio=0.25)
                    chip_meta[(gy,gx)] = {'kind':'defect', 'obj': obj_actual,
                                          'bin': assign_defect_bin(kind, rng), 'inside': True}
    for gy in range(GRID):
        for gx in range(GRID):
            if invalid_mask[gy, gx]:
                chip_meta[(gy,gx)] = {'kind':'invalid', 'obj': None,
                                      'bin': assign_invalid_bin(kind, rng), 'inside': True}

    # GPU canvas baseline
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    u = torch.rand((SIZE, SIZE), generator=g, device=DEVICE)
    canvas_t = torch.searchsorted(CUM_BASE_T, u).to(torch.uint8)
    del u
    inside_t = torch.from_numpy(inside.astype(np.uint8)).to(DEVICE)
    inside_pix_t = inside_t.repeat_interleave(CHIP, dim=0).repeat_interleave(CHIP, dim=1)
    canvas_t.masked_fill_(inside_pix_t == 0, IDX_BG)

    # Per-chip GPU
    for (gy, gx), meta in chip_meta.items():
        if meta['kind'] != 'defect': continue
        obj = meta['obj']
        alpha = ALPHA_FNS_T[obj](rng)
        cum_obj_t = CUM_OBJ_T[obj]
        center_power = {'bank_boundary': 6, 'particle_blast': 4, 'scratch': 5, 'scratch_21deg': 8,
                        'geometric_random': 5, 'small_dot': 4, 'fragment': 5, 'irregular': 4, 'ring_small': 5}.get(obj, 4)
        w_bg = torch.clamp((0.40 - alpha) / 0.40, 0.0, 1.0)
        t_raw = torch.clamp((alpha - 0.40) / 0.60, 0.0, 1.0)
        w_center = t_raw ** center_power
        w_edge = torch.clamp(1.0 - w_bg - w_center, 0.0, 1.0)
        cum_mixed = (w_bg.unsqueeze(-1)     * CUM_DEFECT_BG_T[None, None, :] +
                     w_edge.unsqueeze(-1)   * CUM_EDGE_T[None, None, :] +
                     w_center.unsqueeze(-1) * cum_obj_t[None, None, :])
        u_chip = torch.rand((CHIP, CHIP), generator=g, device=DEVICE)
        grades = (u_chip.unsqueeze(-1) < cum_mixed).int().argmax(dim=-1).to(torch.uint8)
        y0, x0 = gy*CHIP, gx*CHIP
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
    """물리력 자국 (Starburst, CommaCluster): wafer 전체 도화지에 직접 라인 그림.
    chip-grid alpha 패턴 안 씀. 그린 라인 위에서 grade 1 dominant + grade 2 spot.
    chip 경계는 normal 1px gray + 자국 통과한 chip은 random BIN 2px border.
    """
    kind = '00P' if rng.random() < 0.5 else '00C'

    # 1. wafer-canvas alpha field (CPU PIL)
    alpha_np = WAFER_CANVAS_PATTERNS[class_name](rng)
    inside_pix_np = np.repeat(np.repeat(inside.astype(np.float32), CHIP, axis=0), CHIP, axis=1)
    alpha_np = alpha_np * inside_pix_np                                               # 외곽 0

    # 2. baseline canvas (GPU)
    g_t = torch.Generator(device=DEVICE).manual_seed(seed)
    u = torch.rand((SIZE, SIZE), generator=g_t, device=DEVICE)
    canvas_t = torch.searchsorted(CUM_BASE_T, u).to(torch.uint8); del u
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
    if r['class_name'] in ('Normal', 'Starburst', 'CommaCluster'):
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
