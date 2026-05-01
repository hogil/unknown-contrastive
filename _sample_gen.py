"""Unified sample generator v5 — fail-map convention.

Class set:
  Non-invalid_main objects (00P 또는 00C kind 가능):
    bank_boundary, particle_blast, scratch
  Invalid-main object (00C kind 전용):
    invalid_main  ← 분포에 따라 invalid chip 클러스터로 wafer 패턴 형성
  Wafer distributions (WM-811K cca/* heatmaps):
    Center, Donut, Edge-Ring, Edge-Loc, Loc, Random, Near-full
  → 클래스 = 7 dist × 4 obj = 28

Per-image elements:
  - 정상 chip: baseline 노이즈 only, NO border, NO text  (배경 grid line 없음)
  - 불량 chip: chip-내부 object 패턴(α modulation) + 2px BIN-color border + bin 숫자 text
  - invalid chip: palette index 31 (white) 가득 + 2px orange border + bin 숫자 text
  - 외곽(원 밖) chip: 자동 invalid 처리 (text 없음)

Mix: 클래스의 주 object(예: bank_boundary) defect chip 중 ~25%는 다른 object로 섞임.

Filename: <rand6>_<00P|00C>_<wafer:02d>_<ymd>_<hms>_<yield.1f>_<sys.0f>_<TD>_<LT>.png
  ex) abc123_00P_07_20260501_010000_92.4_18_PE_NORMAL.png

Bin rules:
  00P: invalid 200-279, sys defect 285-299 (subset {285,286,287,288,290,291})
  00C: invalid 200-299, sys defect 300+    (subset {300,385,386,388,389,390})
  yield = (bin<200 chip count) / netd × 100  (.1f)
  sys   = (sys defect bin count) / netd × 100 (.0f)
"""
import os, time, json, numpy as np
from PIL import Image, ImageDraw, ImageFont
from _fq_metadata import add_synthetic_fq_to_json
try:
    import torch
    _GPU = torch.cuda.is_available()
    _DEVICE = torch.device('cuda') if _GPU else torch.device('cpu')
except ImportError:
    _GPU = False; _DEVICE = None

# ===== Palette (fail-map style) =====
def hex_to_rgb(s): return [int(s[1:3],16), int(s[3:5],16), int(s[5:7],16)]
PALETTE_HEX_MAP = {
    "chip0":"#FFFFFF","chip1":"#9B9B9B","chip2":"#009619","chip3":"#0000FF",
    "chip4":"#D91DFF","chip5":"#FFFF00","chip6":"#FF0000","chip7":"#000000",
    "bg":"#DCEEFF","text":"#000001","border":"#BEBEBE","border_inv":"#FF9900",
    "border_b285":"#0099FF","border_b286":"#FF714F","border_b287":"#66FFCC",
    "border_b288":"#DA26CD","border_b290":"#FFD700","border_b291":"#32CD32",
    "border_b300":"#AAAAAA","border_b385":"#00C8FF","border_b386":"#FF00C8",
    "border_b388":"#00FF66","border_b389":"#FF6666","border_b390":"#6666FF",
    "border_etc":"#999999",
}
PALETTE_INDEX_TO_KEY = [
    "chip0","chip1","chip2","chip3","chip4","chip5","chip6","chip7",
    "bg","text","border","border_inv",
    "border_b285","border_b286","border_b287","border_b288","border_b290","border_b291",
    "border_b300","border_b385","border_b386","border_b388","border_b389","border_b390",
    "border_etc",
]
KEY_TO_INDEX = {k:i for i,k in enumerate(PALETTE_INDEX_TO_KEY)}
PALETTE = []
for k in PALETTE_INDEX_TO_KEY: PALETTE.extend(hex_to_rgb(PALETTE_HEX_MAP[k]))
while len(PALETTE) < 96: PALETTE.append(0)
PALETTE[31*3:31*3+3] = [255, 255, 255]                                                # invalid_fill (idx 31) = white per user spec

IDX_BG         = KEY_TO_INDEX["bg"]
IDX_TEXT       = KEY_TO_INDEX["text"]
IDX_BORDER_INV = KEY_TO_INDEX["border_inv"]

# Per-bin border palette index lookup
BIN_TO_BORDER_KEY = {
    285:"border_b285", 286:"border_b286", 287:"border_b287",
    288:"border_b288", 290:"border_b290", 291:"border_b291",
    300:"border_b300", 385:"border_b385", 386:"border_b386",
    388:"border_b388", 389:"border_b389", 390:"border_b390",
}
BIN_TO_BORDER_IDX = {b: KEY_TO_INDEX[k] for b, k in BIN_TO_BORDER_KEY.items()}

# Defect chip bin pool — kind별 6개 bin. weight: 낮은 번호일수록 더 많이 (linear decay)
DEFECT_BIN_POOL = {
    '00P': [285, 286, 287, 288, 290, 291],
    '00C': [300, 385, 386, 388, 389, 390],
}
DEFECT_BIN_WEIGHTS = np.array([6, 5, 4, 3, 2, 1], dtype=np.float64)
DEFECT_BIN_WEIGHTS /= DEFECT_BIN_WEIGHTS.sum()                                        # [0.286, 0.238, 0.190, 0.143, 0.095, 0.048]

# ===== Spec =====
SIZE = 6400; GRID = 32; CHIP = 200
HEATMAP_DIR  = "D:/project/unknown-contrastive/_dist_heatmaps"
PNG_OUT_DIR  = "D:/project/data/wm-811k/unknown"
JSON_OUT_DIR = "D:/project/data/positions/unknown"
os.makedirs(PNG_OUT_DIR, exist_ok=True)
os.makedirs(JSON_OUT_DIR, exist_ok=True)

# normal chip baseline: P(0)+P(1) ≈ 98% — defect chip과 압도적 차이
BASELINE = np.array([0.83, 0.15, 0.012, 0.005, 0.002, 0.0008, 0.0001, 0.0001], dtype=np.float64)
BASELINE /= BASELINE.sum()
CUM_BASE = np.cumsum(BASELINE)

# defect chip 양호 영역 (라인 외 / 불량영역 밖): P(1) 0.25로 살짝 더 elevated
DEFECT_BG_DIST = np.array([0.73, 0.25, 0.012, 0.005, 0.002, 0.0008, 0.0001, 0.0001], dtype=np.float64)
DEFECT_BG_DIST /= DEFECT_BG_DIST.sum()
CUM_DEFECT_BG = np.cumsum(DEFECT_BG_DIST)

# 불량영역 가장자리 (zone edge): grade 1 40%로 낮춤 (zone 끝쪽 1 비율 너무 높지 않게)
# P(0)=50%, P(1)=40% → BG(25%)와 CENTER 사이 smoother transition
EDGE_DIST = np.array([0.50, 0.40, 0.07, 0.02, 0.005, 0.003, 0.001, 0.001], dtype=np.float64)
EDGE_DIST /= EDGE_DIST.sum()
CUM_EDGE = np.cumsum(EDGE_DIST)

# Object별 (main, sub) defect grade — main이 zone center에서 dominant, sub는 추가로 elevated
PRIMARY_GRADE = {
    'bank_boundary':  (1, 2),  # main=1, sub=2 (사용자: "main pixel을 1로하고 sub는 2로하자")
    'particle_blast': (4, 1),  # main=4 (severe), grade 1 elevated for edge transition
    'scratch':        (3, 1),  # main=3
    'scratch_21deg':  (3, 1),  # main=3 (scratch와 동일 grade, 라인 각도로 구분)
}

# 불량영역 중앙 (zone center, alpha=1): main grade 거의 대부분 (~80%)
OBJECT_DISTS = {
    # idx:           [  0,    1,    2,    3,    4,    5,     6,     7   ]
    'bank_boundary':  np.array([0.02, 0.10, 0.80, 0.05, 0.01, 0.005, 0.003, 0.002], dtype=np.float64),  # main=2(80%)
    'particle_blast': np.array([0.02, 0.10, 0.03, 0.03, 0.80, 0.01, 0.005, 0.005], dtype=np.float64),  # main=4(80%)
    'scratch':        np.array([0.02, 0.10, 0.03, 0.80, 0.03, 0.01, 0.005, 0.005], dtype=np.float64),  # main=3(80%)
    'scratch_21deg':  np.array([0.02, 0.10, 0.03, 0.80, 0.03, 0.01, 0.005, 0.005], dtype=np.float64),  # main=3(80%)
    'geometric_random': np.array([0.05, 0.40, 0.30, 0.15, 0.05, 0.03,  0.015, 0.005], dtype=np.float64),
    # Normal 전용 novel objects — main grade 다양 (registered 31 class와 구분되도록)
    'small_dot':        np.array([0.05, 0.20, 0.55, 0.15, 0.03, 0.015, 0.003, 0.002], dtype=np.float64),  # main=2
    'fragment':         np.array([0.05, 0.25, 0.30, 0.30, 0.05, 0.03,  0.015, 0.005], dtype=np.float64),  # mix 2-3
    'irregular':        np.array([0.05, 0.25, 0.30, 0.20, 0.10, 0.05,  0.03,  0.02 ], dtype=np.float64),  # 폭넓게
    'ring_small':       np.array([0.05, 0.20, 0.40, 0.25, 0.05, 0.03,  0.015, 0.005], dtype=np.float64),  # main=2-3
}
for k in OBJECT_DISTS: OBJECT_DISTS[k] /= OBJECT_DISTS[k].sum()

DEFECT_BUDGET = {
    'Center':       18,                                                                # 더 모음 (was 25)
    'Donut':        30,                                                                # 더 모음 (was 40)
    'Edge-Ring':    70,
    'Edge-Bottom':  6,                                                                 # split from Edge-Loc, 하단
    'Edge-Top':     6,                                                                 # split from Edge-Loc, 상단
    'Full':         250,                                                               # Random(50)~Near-full(500) 중간
    'Thick-Edge':   400,                                                               # randomized
    'Normal':       0,                                                                 # special: render_normal에서 처리
}

# ===== Random helpers =====
def rand_prefix(rng):
    letters = ''.join(chr(ord('a') + int(rng.integers(0, 26))) for _ in range(3))
    digits  = ''.join(str(int(rng.integers(0, 10)))             for _ in range(3))
    return letters + digits

LT_OPTIONS = ['PE', 'EE', 'PT']                                                       # Lot Type (filename 7번째 토큰)
TM_OPTIONS = ['NORMAL', 'PWQ', 'ENGINEER']                                            # Test Mode (filename 8번째 토큰)

# ===== Chip object alpha-fields =====
XC, YC = np.meshgrid(np.arange(CHIP), np.arange(CHIP), indexing='xy')

CHIP_BASE_ALPHA = 0.0                                                                  # 라인 외 영역은 DEFECT_BG_DIST 직접 사용 (object_dist mix 안 함 → 불량 pixel↑ 안 함)

def _perp_profile(d, sigma_s=1.5, sigma_m=3.0, sigma_w=6.0):
    """5단계 perpendicular density: 약-중-강-중-약 (좌-우 대칭).
    인접 라인 overlap 방지 위해 sigma 축소. 강 zone 매우 얇음.
    d: 라인으로부터 수직 거리. 좌우 대칭이므로 d**2 사용.
    """
    strong = 1.00 * np.exp(-d**2 / (2*sigma_s**2)).astype(np.float32)                  # 강 (매우 얇은 중앙)
    med    = 0.55 * np.exp(-d**2 / (2*sigma_m**2)).astype(np.float32)                  # 중
    weak   = 0.25 * np.exp(-d**2 / (2*sigma_w**2)).astype(np.float32)                  # 약
    return np.minimum(strong + med + weak, 1.0)

def alpha_bank_boundary(rng):
    """3 vertical + 1 horizontal bank cut.
    Center 매우 좁게 (sigma_s=0.7), line 외곽은 부드러운 transition (sigma_w=10).
    각 라인에 Y(또는 X) 방향 random 산포 (10 segments) → 라인이 균일하지 않음.
    center_power=6 (mixing단계 별도 적용)으로 grade-2 dominant zone 매우 얇음.
    """
    a = np.full((CHIP, CHIP), CHIP_BASE_ALPHA, dtype=np.float32)
    n_seg = 10
    seg_len = CHIP // n_seg                                                            # 20 px each
    for cx in [50, 100, 150]:
        s = rng.uniform(0.90, 1.0)
        seg_strengths = rng.uniform(0.55, 1.00, size=n_seg).astype(np.float32)
        y_noise = np.repeat(seg_strengths, seg_len)[:CHIP, None]                       # (CHIP, 1) — Y 방향 산포
        line = _perp_profile(XC - cx, sigma_s=0.7, sigma_m=3.0, sigma_w=12.0) * y_noise * s
        a = np.maximum(a, line)
    for cy in [100]:
        s = rng.uniform(0.90, 1.0)
        seg_strengths = rng.uniform(0.55, 1.00, size=n_seg).astype(np.float32)
        x_noise = np.repeat(seg_strengths, seg_len)[None, :CHIP]                       # (1, CHIP) — X 방향 산포
        line = _perp_profile(YC - cy, sigma_s=0.7, sigma_m=3.0, sigma_w=12.0) * x_noise * s
        a = np.maximum(a, line)
    return a

def alpha_particle_blast(rng):
    """Gaussian blob — center 진하고 외곽으로 부드럽게 감쇠. + chip base elevation."""
    a = np.full((CHIP, CHIP), CHIP_BASE_ALPHA, dtype=np.float32)
    cx = rng.uniform(50, 150); cy = rng.uniform(50, 150)
    sigma = rng.uniform(22, 35)
    blob = np.exp(-((XC - cx)**2 + (YC - cy)**2) / (2*sigma**2)).astype(np.float32)
    return np.maximum(a, blob)

def alpha_scratch(rng):
    """3~5 vertical 얇은 라인 (적게). 위치/길이 모두 random → 불균일.
    bank_boundary 대비 sigma 절반 정도로 얇음.
    """
    a = np.full((CHIP, CHIP), CHIP_BASE_ALPHA, dtype=np.float32)
    n_lines = int(rng.integers(5, 16))                                                 # 5-15 lines
    for _ in range(n_lines):
        cx = rng.uniform(15, 185)                                                      # 불균일 random
        y_start = rng.uniform(0, 80); y_end = rng.uniform(120, 200)                    # 불균일 길이
        in_range = ((YC >= y_start) & (YC <= y_end)).astype(np.float32)
        s = rng.uniform(0.80, 1.0)                                                     # 라인별 강도 변동 큼
        line = _perp_profile(XC - cx, sigma_s=1.0, sigma_m=2.0, sigma_w=4.0) * in_range * s
        a = np.maximum(a, line)
    return a

def alpha_scratch_21deg(rng):
    """12~18 매우 얇은 라인, 시계방향 21도 회전, 균일 간격.
    가운데 불량 pixel 영역 매우 좁음 (sigma_s=0.7, _m=1.5, _w=3.0).
    """
    a = np.full((CHIP, CHIP), CHIP_BASE_ALPHA, dtype=np.float32)
    n_lines = int(rng.integers(12, 19))
    theta = np.deg2rad(21.0)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cy = 100
    span_lo, span_hi = 15, 185
    spacing = (span_hi - span_lo) / (n_lines + 1)
    for i in range(n_lines):
        cx = span_lo + (i + 1) * spacing
        d_perp = (XC - cx) * cos_t - (YC - cy) * sin_t
        s = rng.uniform(0.92, 1.0)
        line = _perp_profile(d_perp, sigma_s=0.7, sigma_m=1.5, sigma_w=3.0) * s        # 매우 좁은 center
        a = np.maximum(a, line)
    return a

def alpha_small_dot(rng):
    """매우 작은 점 1-4개 (Normal 전용)."""
    a = np.full((CHIP, CHIP), CHIP_BASE_ALPHA, dtype=np.float32)
    n = int(rng.integers(1, 5))
    for _ in range(n):
        cx, cy = float(rng.uniform(20, 180)), float(rng.uniform(20, 180))
        sigma = float(rng.uniform(1.5, 4.0))
        dot = np.exp(-((XC - cx)**2 + (YC - cy)**2) / (2*sigma**2)).astype(np.float32)
        a = np.maximum(a, dot)
    return a

def alpha_fragment(rng):
    """짧은 line fragment 1-2개, random angle (Normal 전용)."""
    a = np.full((CHIP, CHIP), CHIP_BASE_ALPHA, dtype=np.float32)
    n = int(rng.integers(1, 3))
    for _ in range(n):
        angle = float(rng.uniform(0, 2*np.pi))
        cx, cy = float(rng.uniform(30, 170)), float(rng.uniform(30, 170))
        cos_t, sin_t = float(np.cos(angle)), float(np.sin(angle))
        d_perp  = (XC - cx) * cos_t - (YC - cy) * sin_t
        d_along = (XC - cx) * sin_t + (YC - cy) * cos_t
        length = float(rng.uniform(15, 60))
        sigma = float(rng.uniform(1.5, 3.0))
        in_range = (np.abs(d_along) < length / 2).astype(np.float32)
        line = np.exp(-d_perp**2 / (2*sigma**2)).astype(np.float32) * in_range
        a = np.maximum(a, line)
    return a

def alpha_irregular(rng):
    """비대칭 작은 blob 2-4개 (Normal 전용)."""
    a = np.full((CHIP, CHIP), CHIP_BASE_ALPHA, dtype=np.float32)
    n = int(rng.integers(2, 5))
    for _ in range(n):
        cx, cy = float(rng.uniform(20, 180)), float(rng.uniform(20, 180))
        sx = float(rng.uniform(3, 12)); sy = float(rng.uniform(3, 12))
        blob = np.exp(-((XC - cx)**2 / (2*sx**2) + (YC - cy)**2 / (2*sy**2))).astype(np.float32)
        a = np.maximum(a, blob)
    return a

def alpha_ring_small(rng):
    """작은 ring (반지름 8-25, Normal 전용)."""
    a = np.full((CHIP, CHIP), CHIP_BASE_ALPHA, dtype=np.float32)
    cx = float(rng.uniform(40, 160)); cy = float(rng.uniform(40, 160))
    r = float(rng.uniform(8, 25)); thick = float(rng.uniform(1.5, 3.0))
    d_r = np.sqrt((XC - cx)**2 + (YC - cy)**2)
    ring = np.exp(-(d_r - r)**2 / (2*thick**2)).astype(np.float32)
    return np.maximum(a, ring)

def alpha_geometric_random(rng):
    """Normal 클래스용 — 기하학적으로 다양한 작은 결함 (object로 카테고리화 안 된 형태).
    1-4 random shapes 합성: 작은 blob, 짧은 라인, 호 곡선 등.
    """
    a = np.full((CHIP, CHIP), CHIP_BASE_ALPHA, dtype=np.float32)
    n_shapes = int(rng.integers(1, 5))
    for _ in range(n_shapes):
        choice = int(rng.integers(0, 3))
        if choice == 0:                                                                # 작은 blob
            cx = rng.uniform(30, 170); cy = rng.uniform(30, 170)
            sigma = rng.uniform(8, 18)
            blob = np.exp(-((XC - cx)**2 + (YC - cy)**2) / (2*sigma**2)).astype(np.float32)
            a = np.maximum(a, blob)
        elif choice == 1:                                                              # 짧은 라인 (random angle)
            angle = rng.uniform(0, np.pi); cx = rng.uniform(50, 150); cy = rng.uniform(50, 150)
            cos_t, sin_t = np.cos(angle), np.sin(angle)
            d_perp = (XC - cx) * cos_t - (YC - cy) * sin_t
            d_along = (XC - cx) * sin_t + (YC - cy) * cos_t
            length = rng.uniform(30, 90); sigma_line = rng.uniform(1.5, 3.0)
            in_range = (np.abs(d_along) < length / 2).astype(np.float32)
            line = np.exp(-d_perp**2 / (2*sigma_line**2)).astype(np.float32) * in_range
            a = np.maximum(a, line)
        else:                                                                          # arc / 곡선
            cx = rng.uniform(50, 150); cy = rng.uniform(50, 150)
            r = rng.uniform(20, 60); thickness = rng.uniform(2.0, 4.0)
            d_r = np.sqrt((XC - cx)**2 + (YC - cy)**2)
            arc = np.exp(-(d_r - r)**2 / (2*thickness**2)).astype(np.float32)
            a = np.maximum(a, arc)
    return a

ALPHA_FNS = {
    'bank_boundary':    alpha_bank_boundary,
    'particle_blast':   alpha_particle_blast,
    'scratch':          alpha_scratch,
    'scratch_21deg':    alpha_scratch_21deg,
    'geometric_random': alpha_geometric_random,
    'small_dot':        alpha_small_dot,
    'fragment':         alpha_fragment,
    'irregular':        alpha_irregular,
    'ring_small':       alpha_ring_small,
}

CHIP_OBJECTS = ['bank_boundary', 'particle_blast', 'scratch', 'scratch_21deg']
# Normal 전용 = 등록된 4개 + 새로운 5개 novel (defect class와 구분)
NORMAL_OBJECTS = CHIP_OBJECTS + ['geometric_random', 'small_dot', 'fragment', 'irregular', 'ring_small']

# ===== Defect chip selection =====
def _wafer_inside_mask():
    pw = np.load(os.path.join(HEATMAP_DIR, "Center_p_wafer_32.npy"))
    return pw >= 0.10

def _cluster_around(cy0, cx0, n, spread, inside, rng):
    mask = np.zeros((GRID, GRID), dtype=bool)
    placed, tries = 0, 0
    while placed < n and tries < 200:
        tries += 1
        dy = int(round(rng.normal(0, spread))); dx = int(round(rng.normal(0, spread)))
        cy, cx = cy0 + dy, cx0 + dx
        if 0 <= cy < GRID and 0 <= cx < GRID and inside[cy, cx] and not mask[cy, cx]:
            mask[cy, cx] = True; placed += 1
    return mask

def select_distribution_chips(class_name, rng, inside, n_override=None):
    """Heatmap-based wafer-distribution chip selection (used for both defect & invalid_main)."""
    n = n_override if n_override is not None else DEFECT_BUDGET[class_name]
    if class_name in ('Center', 'Donut'):
        # 더 모이도록 heatmap에 power 적용
        hm = np.load(os.path.join(HEATMAP_DIR, f"{class_name}_p_defect_32.npy"))
        hm_pow = hm ** 3                                                              # 3제곱으로 sharper peak
        flat = (hm_pow * inside.astype(np.float32)).flatten()
        s = flat.sum()
        if s <= 0:
            ys, xs = np.where(inside); idx = rng.choice(len(ys), size=n, replace=False)
            mask = np.zeros((GRID, GRID), dtype=bool); mask[ys[idx], xs[idx]] = True
            return mask
        flat = flat / s
        chosen = rng.choice(GRID*GRID, size=min(n, int((flat>0).sum())), replace=False, p=flat)
        mask = np.zeros((GRID, GRID), dtype=bool); mask.flat[chosen] = True
        return mask
    elif class_name == 'Edge-Ring':
        hm = np.load(os.path.join(HEATMAP_DIR, "Edge-Ring_p_defect_32.npy"))
        flat = (hm * inside.astype(np.float32)).flatten()
        s = flat.sum()
        if s <= 0:
            ys, xs = np.where(inside); idx = rng.choice(len(ys), size=n, replace=False)
            mask = np.zeros((GRID, GRID), dtype=bool); mask[ys[idx], xs[idx]] = True
            return mask
        flat = flat / s
        chosen = rng.choice(GRID*GRID, size=min(n, int((flat>0).sum())), replace=False, p=flat)
        mask = np.zeros((GRID, GRID), dtype=bool); mask.flat[chosen] = True
        return mask
    elif class_name == 'Full':
        # Random + Near-full 평균 → 중간 밀도 (~250 chips)
        hm_r = np.load(os.path.join(HEATMAP_DIR, "Random_p_defect_32.npy"))
        hm_n = np.load(os.path.join(HEATMAP_DIR, "Near-full_p_defect_32.npy"))
        hm = (hm_r + hm_n) / 2.0
        flat = (hm * inside.astype(np.float32)).flatten()
        s = flat.sum()
        if s <= 0: s = 1.0
        flat = flat / max(s, 1e-12)
        chosen = rng.choice(GRID*GRID, size=min(n, int((flat>0).sum())), replace=False, p=flat)
        mask = np.zeros((GRID, GRID), dtype=bool); mask.flat[chosen] = True
        return mask
    elif class_name == 'Edge-Bottom':
        # wafer 하단 외곽: cy 큰 쪽에서 anchor 선택
        ys, xs = np.where(inside)
        d_bottom = ys.astype(float)                                                    # 하단일수록 ys 큼
        weights = np.exp((d_bottom - d_bottom.min()) / 3.0)                            # bottom 가중
        weights /= weights.sum()
        # 추가로 외곽 가까운 chip 선호
        d_center = np.sqrt((ys - GRID/2)**2 + (xs - GRID/2)**2)
        edge_w = (d_center / (GRID/2)) ** 2                                            # edge 가중
        combined = weights * edge_w
        combined /= combined.sum() if combined.sum() > 0 else 1.0
        k = rng.choice(len(ys), p=combined)
        cy0, cx0 = ys[k], xs[k]
        return _cluster_around(cy0, cx0, n, spread=1.3, inside=inside, rng=rng)
    elif class_name == 'Edge-Top':
        ys, xs = np.where(inside)
        d_top = (GRID - ys).astype(float)                                              # 상단일수록 큼
        weights = np.exp((d_top - d_top.min()) / 3.0)
        weights /= weights.sum()
        d_center = np.sqrt((ys - GRID/2)**2 + (xs - GRID/2)**2)
        edge_w = (d_center / (GRID/2)) ** 2
        combined = weights * edge_w
        combined /= combined.sum() if combined.sum() > 0 else 1.0
        k = rng.choice(len(ys), p=combined)
        cy0, cx0 = ys[k], xs[k]
        return _cluster_around(cy0, cx0, n, spread=1.3, inside=inside, rng=rng)
    elif class_name == 'Thick-Edge':
        # 두꺼운 외곽 ring: inner radius와 chip 개수 모두 random → 매번 다른 모양
        ys, xs = np.where(inside)
        cy0, cx0 = GRID/2, GRID/2
        d = np.sqrt((ys - cy0)**2 + (xs - cx0)**2)
        # inner radius 0.30~0.55 random per sample
        inner_ratio = rng.uniform(0.30, 0.55)
        in_ring = d > (GRID/2) * inner_ratio
        candidates = np.where(in_ring)[0]
        # n_to_place도 random within range
        n_random = int(rng.integers(int(n*0.7), int(n*1.1) + 1))
        n_to_place = min(n_random, len(candidates))
        chosen = rng.choice(candidates, size=n_to_place, replace=False)
        mask = np.zeros((GRID, GRID), dtype=bool); mask[ys[chosen], xs[chosen]] = True
        return mask
    raise ValueError(class_name)

def select_random_invalid(rng, exclude_mask, inside, n=15):
    eligible = (~exclude_mask) & inside
    ys, xs = np.where(eligible)
    if len(ys) < n: n = len(ys)
    if n == 0:
        return np.zeros((GRID, GRID), dtype=bool)
    idx = rng.choice(len(ys), size=n, replace=False)
    mask = np.zeros((GRID, GRID), dtype=bool); mask[ys[idx], xs[idx]] = True
    return mask

def outside_chips_as_invalid(inside): return ~inside

def select_normal_chips(rng, inside):
    """Normal 클래스 = 등록된 31 defect class와 분포 모양이 다름 (open-set).
    절대 단일 등록 분포(Center/Donut/Edge-*/Full/Thick-Edge) heatmap 직접 사용 X.
    - defect chip 수: 10-200 random
    - 배치 strategy: 매번 random 선택
        (a) pure uniform random scatter (50%)
        (b) multiple small clusters at random anchors (50%) — 2-6개 cluster, 각 cluster 작음
    - object: 9종 mix (novel 비중 ↑)
    """
    target_total = int(rng.integers(10, 151))                                          # 10-150 chips
    ys, xs = np.where(inside); n_inside = len(ys)
    mask = np.zeros((GRID, GRID), dtype=bool)
    obj_map = {}
    if n_inside == 0:
        return mask, obj_map

    # 순수 uniform random scatter — 절대 뭉치지 않음 (등록 분포 어떤 것과도 안 닮음)
    n_pick = min(target_total, n_inside)
    idx = rng.choice(n_inside, size=n_pick, replace=False)
    mask[ys[idx], xs[idx]] = True

    # 9 object types, novel 비중 ↑
    obj_p = np.array([0.05, 0.05, 0.05, 0.05, 0.10, 0.20, 0.20, 0.20, 0.10])
    obj_p = obj_p / obj_p.sum()
    for (gy, gx) in np.argwhere(mask):
        obj_map[(int(gy), int(gx))] = NORMAL_OBJECTS[int(rng.choice(len(NORMAL_OBJECTS), p=obj_p))]
    return mask, obj_map

# ===== Bin assignment =====
def assign_defect_bin(kind, rng):
    """Object 무관, 6개 defect bin 중 weighted 샘플링 (낮은 번호 더 많이)."""
    return int(rng.choice(DEFECT_BIN_POOL[kind], p=DEFECT_BIN_WEIGHTS))

def assign_invalid_bin(kind, rng):
    if kind == '00P':
        return int(rng.integers(200, 280))
    return int(rng.integers(200, 300))

def pick_mixed_object(primary, rng, mix_ratio=0.25):
    """75% primary, 25% other random object."""
    if rng.random() < mix_ratio:
        others = [o for o in CHIP_OBJECTS if o != primary]
        return others[int(rng.integers(0, len(others)))]
    return primary

# ===== Font for bin number text =====
def _try_font(size):
    for path in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(path):
            try: return ImageFont.truetype(path, size)
            except Exception: pass
    return ImageFont.load_default()

FONT_BIG = _try_font(64)

# ===== Main render =====
def render(class_name, object_name, seed):
    rng = np.random.default_rng(seed)
    inside = _wafer_inside_mask()

    # 1) Choose kind
    if object_name == 'invalid_main':
        kind = '00C'
    else:
        kind = '00P' if rng.random() < 0.5 else '00C'

    # 2) Plan defect / invalid masks
    normal_obj_map = None                                                             # Normal 클래스에서 chip별 object 지정용
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

    # 3) Per-chip bin & object assignment (inside-wafer chips only)
    chip_meta = {}
    if object_name != 'invalid_main':
        for gy in range(GRID):
            for gx in range(GRID):
                if defect_mask[gy, gx]:
                    if normal_obj_map is not None:
                        obj_actual = normal_obj_map[(gy, gx)]                          # Normal: 미리 지정된 object
                    else:
                        obj_actual = pick_mixed_object(object_name, rng, mix_ratio=0.25)
                    chip_meta[(gy,gx)] = {'kind':'defect', 'obj': obj_actual,
                                          'bin': assign_defect_bin(kind, rng), 'inside': True}
    for gy in range(GRID):
        for gx in range(GRID):
            if invalid_mask[gy, gx]:
                chip_meta[(gy,gx)] = {'kind':'invalid', 'obj': None,
                                      'bin': assign_invalid_bin(kind, rng), 'inside': True}

    # 4) Canvas: baseline grades, then bg color overwrites OUTSIDE-wafer cells (no chip there)
    if _GPU:
        # GPU 가속: 40M float random + searchsorted 가장 무거운 op
        g_t = torch.Generator(device=_DEVICE).manual_seed(seed)
        u_t = torch.rand((SIZE, SIZE), generator=g_t, device=_DEVICE)
        cum_base_t = torch.tensor(CUM_BASE, device=_DEVICE, dtype=torch.float32)
        canvas_t = torch.searchsorted(cum_base_t, u_t).to(torch.uint8)
        canvas = canvas_t.cpu().numpy()
        del u_t, canvas_t
    else:
        u = rng.random((SIZE, SIZE))
        canvas = np.searchsorted(CUM_BASE, u).astype(np.uint8); del u
    inside_pix = np.repeat(np.repeat(inside, CHIP, axis=0), CHIP, axis=1)             # 6400x6400 bool
    canvas[~inside_pix] = IDX_BG                                                       # outside-wafer = bg color (no chip)

    # 5) Defect chips: alpha modulation per chip with assigned object
    for (gy, gx), meta in chip_meta.items():
        if meta['kind'] != 'defect': continue
        obj = meta['obj']
        alpha = ALPHA_FNS[obj](rng)
        cum_obj = np.cumsum(OBJECT_DISTS[obj])
        # 11단계 세분화 + 익스포넨셜 ramp toward CENTER
        # BG↔EDGE 0~0.40 (wider, smoother transition with normal area)
        # EDGE→CENTER 0.40~1.0 (power exp)
        center_power = {'bank_boundary': 6, 'particle_blast': 4, 'scratch': 5, 'scratch_21deg': 8,
                        'geometric_random': 5, 'small_dot': 4, 'fragment': 5, 'irregular': 4, 'ring_small': 5}.get(obj, 4)
        w_bg     = np.clip((0.40 - alpha) / 0.40, 0.0, 1.0).astype(np.float32)         # 1@α=0, 0@α=0.40 (wider)
        t_raw    = np.clip((alpha - 0.40) / 0.60, 0.0, 1.0).astype(np.float32)
        w_center = (t_raw ** center_power).astype(np.float32)
        w_edge   = np.clip(1.0 - w_bg - w_center, 0.0, 1.0).astype(np.float32)
        cum_mixed = (w_bg[..., None]     * CUM_DEFECT_BG[None,None,:] +
                     w_edge[..., None]   * CUM_EDGE[None,None,:] +
                     w_center[..., None] * cum_obj[None,None,:])
        uu = rng.random((CHIP, CHIP))
        grades = (uu[..., None] < cum_mixed).argmax(axis=-1).astype(np.uint8)
        y0, x0 = gy*CHIP, gx*CHIP
        canvas[y0:y0+CHIP, x0:x0+CHIP] = grades

    # 6) Invalid chip pixels = palette 31 white
    for (gy, gx), meta in chip_meta.items():
        if meta['kind'].startswith('invalid'):
            y0, x0 = gy*CHIP, gx*CHIP
            canvas[y0:y0+CHIP, x0:x0+CHIP] = 31

    # 7) Borders for ALL inside-wafer chips
    #    normal=1px gray, defect=2px BIN-color, invalid=2px orange
    #    outside-wafer cells (no chip) = no border
    IDX_BORDER_NORMAL = KEY_TO_INDEX["border"]
    for gy in range(GRID):
        for gx in range(GRID):
            if not inside[gy, gx]: continue                                           # outside chip 없음 → border 없음
            y0, x0 = gy*CHIP, gx*CHIP; y1, x1 = y0+CHIP, x0+CHIP
            meta = chip_meta.get((gy, gx))
            if meta is None:                                                          # normal chip
                b, c = 1, IDX_BORDER_NORMAL
            elif meta['kind'] == 'defect':
                b, c = 2, BIN_TO_BORDER_IDX.get(meta['bin'], KEY_TO_INDEX["border_etc"])
            else:                                                                     # invalid
                b, c = 2, IDX_BORDER_INV
            canvas[y0:y0+b, x0:x1] = c; canvas[y1-b:y1, x0:x1] = c
            canvas[y0:y1, x0:x0+b] = c; canvas[y0:y1, x1-b:x1] = c

    # 8) Build palette image + draw bin number text on INVALID chips only
    img = Image.frombytes('P', (SIZE, SIZE), canvas.tobytes())
    img.putpalette(PALETTE)
    draw = ImageDraw.Draw(img)
    for (gy, gx), meta in chip_meta.items():
        if meta['kind'] != 'invalid': continue                                        # text only on invalid
        text = str(meta['bin'])
        y0, x0 = gy*CHIP, gx*CHIP
        cx_px, cy_px = x0 + CHIP/2, y0 + CHIP/2
        try:
            bbox = draw.textbbox((0,0), text, font=FONT_BIG)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
            ty = cy_px - th/2 - bbox[1] if isinstance(FONT_BIG, ImageFont.FreeTypeFont) else cy_px - th/2
        except Exception:
            tw = th = 40; ty = cy_px - th/2
        draw.text((cx_px - tw/2, ty), text, fill=IDX_TEXT, font=FONT_BIG)

    # 9) Yield / Sys / TD / LT
    sys_bins = {285,286,287,288,290,291} if kind == '00P' else {300,385,386,388,389,390}
    netd = int(inside.sum())                                                          # total chips inside wafer
    gd_count  = sum(1 for m in chip_meta.values() if m['inside'] and m['bin'] is not None and m['bin'] < 200)
    # Normal chips (inside wafer, no entry in chip_meta) all count as good (bin<200)
    inside_chip_meta_count = sum(1 for m in chip_meta.values() if m['inside'])
    normal_inside = netd - inside_chip_meta_count
    gd_count += normal_inside
    sys_count = sum(1 for m in chip_meta.values() if m['inside'] and m['bin'] in sys_bins)
    yld = 100.0 * gd_count / max(1, netd)
    syp = 100.0 * sys_count / max(1, netd)
    TD = LT_OPTIONS[int(rng.integers(0, len(LT_OPTIONS)))]                            # filename slot "TD" = LT value (PE/EE/PT)
    LT = TM_OPTIONS[int(rng.integers(0, len(TM_OPTIONS)))]                            # filename slot "LT" = TM value (NORMAL/...)

    # 10) Save PNG → wm-811k/unknown/<class>/<filename>.png
    cls_label = f"{class_name}_{object_name}"
    png_dir   = os.path.join(PNG_OUT_DIR, cls_label)
    json_dir  = os.path.join(JSON_OUT_DIR, cls_label)
    os.makedirs(png_dir, exist_ok=True); os.makedirs(json_dir, exist_ok=True)
    prefix = rand_prefix(rng); w_idx = int(rng.integers(1, 25))
    base   = f"{prefix}_{kind}_{w_idx:02d}_20260501_010000_{yld:.1f}_{syp:.0f}_{TD}_{LT}"
    png_path  = os.path.join(png_dir,  base + ".png")
    json_path = os.path.join(json_dir, base + ".json")
    img.save(png_path, optimize=False, compress_level=1)                              # 빠른 저장 (압축 약함, 파일 2배 큼)

    # 11) Generate matching positions JSON (fail-map docs schema + synthetic FTN/QTN)
    chips_list = []
    norm_rng = np.random.default_rng(seed + 99999)
    for gy in range(GRID):
        for gx in range(GRID):
            if not inside[gy, gx]: continue
            if (gy, gx) in chip_meta and chip_meta[(gy,gx)]['bin'] is not None:
                bin_val = chip_meta[(gy,gx)]['bin']
            else:
                bin_val = int(norm_rng.integers(1, 200))                              # normal chip random good bin
            x0, y0 = gx*CHIP, gy*CHIP; x1, y1 = x0+CHIP, y0+CHIP
            x_cal = gx - (GRID//2 - 1)                                                # GRID even → centerize_col
            y_cal = gy - (GRID//2)                                                     # centerize_row
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
        "netd": int(inside.sum()), "gd": int(gd_count),
        "yield": f"{yld:.2f}", "sys": f"{syp:.2f}",
        "tm": LT, "lt": TD,
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
    return png_path, int(defect_mask.sum()), int(invalid_inside_mask.sum()), kind, yld, syp

CLASSES = ['Center', 'Donut', 'Edge-Ring', 'Edge-Bottom', 'Edge-Top', 'Full', 'Thick-Edge',
           'Normal', 'Starburst', 'CommaCluster']
OBJECTS = ['bank_boundary', 'particle_blast', 'scratch', 'scratch_21deg', 'invalid_main']

# Build task list:
#   6 정규 dist (Center/Donut/Edge-Ring/Edge-Bottom/Edge-Top/Full) × 5 obj = 30
#   + Thick-Edge_invalid_main = 1
#   + Normal (object 없음, 단일 클래스) = 1
#   = 32 classes
def build_tasks(n_per_class):
    tasks = []
    for ci, cls in enumerate(CLASSES):
        for oi, obj in enumerate(OBJECTS):
            if cls == 'Thick-Edge' and obj != 'invalid_main': continue
            if cls == 'Normal' and oi > 0: continue                                   # Normal은 1 클래스
            if cls in ('Starburst', 'CommaCluster') and oi > 0: continue              # wafer-canvas 1 클래스
            for s in range(n_per_class):
                seed = ci*100000 + oi*10000 + s
                tasks.append((cls, obj, seed))
    return tasks

def _worker(args):
    cls, obj, seed = args
    try:
        path, n_def, n_inv, kind, yld, syp = render(cls, obj, seed)
        return (cls, obj, seed, True, None)
    except Exception as e:
        return (cls, obj, seed, False, str(e))

if __name__ == '__main__':
    import argparse
    from concurrent.futures import ProcessPoolExecutor, as_completed
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=200, help='samples per class')
    p.add_argument('--workers', type=int, default=4, help='parallel workers')
    args = p.parse_args()

    tasks = build_tasks(args.n)
    n_total = len(tasks)
    print(f"Total tasks: {n_total} ({len(CLASSES)*len(OBJECTS)-7+1} classes × {args.n} samples)")
    print(f"Workers: {args.workers}")
    total_t0 = time.time()
    done = 0; ok = 0; fail = 0
    last_log = total_t0
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        for cls, obj, seed, success, err in exe.map(_worker, tasks, chunksize=1):
            done += 1
            if success: ok += 1
            else: fail += 1; print(f"  FAIL {cls}_{obj} seed={seed}: {err}")
            now = time.time()
            if now - last_log > 30 or done == n_total:                                # log every 30s
                elapsed = now - total_t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n_total - done) / rate if rate > 0 else 0
                print(f"  [{done:5d}/{n_total}] ok={ok} fail={fail} | rate={rate:.2f}/s | elapsed={elapsed/60:.1f}m | eta={eta/60:.1f}m")
                last_log = now
    print(f"\nDone in {(time.time()-total_t0)/60:.1f}m. ok={ok} fail={fail}")
    print(f"  PNG  -> {PNG_OUT_DIR}")
    print(f"  JSON -> {JSON_OUT_DIR}")
