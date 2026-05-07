"""Unified sample generator v5 — fail-map convention.

Class set:
  Non-invalid_main objects (00P 또는 00C kind 가능):
    bank_boundary, fork, scratch, scratch_rot                                          (round 26: particle_blast→fork, scratch_21deg→scratch_rot)
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


# ===== v4 pink noise (260507): wafer-level 1/f^β FFT for sensor-like background =====
# 사용자 directive 260507: defect 영역 변경 X, non-defect 영역만 pink noise (chip seam 제거 위해
# wafer 전체 1개 field → 각 chip 200×200 slice).
def _pink_noise_field_2d(rng, size, exponent=1.5):
    """Generate (size, size) 1/f^exponent pink noise field, normalized to [0, 1] float32.
    GPU torch.fft 가능하면 사용 (6400² 약 0.05s), 아니면 CPU numpy (3-5s).
    """
    if _GPU:
        seed = int(rng.integers(0, 2**31 - 1))
        g_t = torch.Generator(device=_DEVICE).manual_seed(seed)
        white_t = torch.randn((size, size), generator=g_t, device=_DEVICE)
        f1 = torch.fft.fftfreq(size, device=_DEVICE)
        fx, fy = torch.meshgrid(f1, f1, indexing='xy')
        freq_mag = torch.sqrt(fx * fx + fy * fy); freq_mag[0, 0] = 1.0
        spectrum = 1.0 / (freq_mag ** exponent); spectrum[0, 0] = 0.0
        filt_t = torch.real(torch.fft.ifft2(torch.fft.fft2(white_t) * spectrum))
        f_min = float(filt_t.min()); f_max = float(filt_t.max())
        if (f_max - f_min) > 1e-9:
            out_t = (filt_t - f_min) / (f_max - f_min)
            arr = out_t.to(torch.float32).cpu().numpy()
        else:
            arr = np.full((size, size), 0.5, dtype=np.float32)
        del white_t, filt_t
        return arr
    white = rng.standard_normal((size, size))
    freqs = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(freqs, freqs)
    freq_mag = np.sqrt(fx * fx + fy * fy); freq_mag[0, 0] = 1.0
    spectrum = 1.0 / (freq_mag ** exponent); spectrum[0, 0] = 0.0
    filt = np.real(np.fft.ifft2(np.fft.fft2(white) * spectrum))
    f_min, f_max = float(filt.min()), float(filt.max())
    if (f_max - f_min) > 1e-9:
        return ((filt - f_min) / (f_max - f_min)).astype(np.float32)
    return np.full_like(filt, 0.5, dtype=np.float32)

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
HEATMAP_DIR  = "D:/project/known-cnn/dist_learn/_dist_heatmaps"
PNG_OUT_DIR  = os.environ.get("WAFER_PNG_OUT_DIR",  "D:/project/data/wm-811k/unknown")
JSON_OUT_DIR = os.environ.get("WAFER_JSON_OUT_DIR", "D:/project/data/positions/unknown")
CHIP_OBJ_OUT_DIR = "D:/project/data/wm-811k/classification_chips"               # chip-object crop dataset (per-chip true label)
CHIP_OBJECT_LABELS = ('bank_boundary', 'fork', 'scratch',
                     'scratch_rot', 'invalid_main')
CHIP_OBJ_PER_CLASS_CAP = 200                                                    # 260506 v19: 100 → 200 (master per-class 200 source 확보)
os.makedirs(PNG_OUT_DIR, exist_ok=True)
os.makedirs(JSON_OUT_DIR, exist_ok=True)


def _wafer_basename_chip_key(base: str) -> str:
    """Drop yield/sys (idx 5, 6) tokens from 9-token wafer basename for chip filename."""
    parts = base.split("_")
    if len(parts) >= 9:
        return "_".join(parts[:5] + parts[7:])
    return base


MIN_CHIP_DEFECT_RATIO = 0.10                                                            # 260506: stricter (was 0.03 — too lenient, grade 2 sprinkle 통과)
MIN_CHIP_STRONG_GRADE_RATIO = 0.02                                                      # 260506: NEW — grade ≥3 (real line/pattern) 픽셀이 chip 의 ≥2% 여야 진짜 defect chip

def save_chip_crops(img, chip_meta: dict, base: str, out_root: str = CHIP_OBJ_OUT_DIR) -> dict:
    """Save per-chip 200x200 crops to classification_chips/<obj>/<wafer_key>_X<x>_Y<y>_B<bin>.png.

    chip_meta entry: {'kind': 'defect'|'invalid', 'obj': <obj_name|None>, 'bin': int, 'inside': bool}
    - kind=='defect': crop labeled with chip_meta['obj']
    - kind=='invalid': crop labeled 'invalid_main' (no quality filter — full white = valid signal)
    - obj NOT in CHIP_OBJECT_LABELS skipped (5 obj only)
    - per-class cap CHIP_OBJ_PER_CLASS_CAP (default 100)
    - **defect quality filter (round 29 v15)**: skip chip if (grade ≥ 2 & < 31) pixel ratio
      < MIN_CHIP_DEFECT_RATIO (0.03). v15 weak intensity tier 가 grade 0/1 dominant chip
      을 만들면 시각적으로 양호 chip 처럼 보임 → 이런 chip 이 fork/scratch 폴더에 들어가면
      chip CNN 학습 mislabeled 됨. 양호 같은 chip 은 학습에서 제외.
    Returns counts dict: {label: count}.
    """
    counts = {}
    if not chip_meta:
        return counts
    wafer_key = _wafer_basename_chip_key(base)
    for (gy, gx), meta in chip_meta.items():
        if not meta.get('inside', False):
            continue
        b = meta.get('bin')
        if b is None or b < 200:
            continue
        if meta['kind'] == 'invalid':
            obj = 'invalid_main'
        else:
            obj = meta.get('obj')
        if obj not in CHIP_OBJECT_LABELS:
            continue
        out_dir = os.path.join(out_root, obj)
        os.makedirs(out_dir, exist_ok=True)
        # round 26 hard cap: stop saving if class already at cap
        if CHIP_OBJ_PER_CLASS_CAP is not None:
            try:
                existing = sum(1 for _ in os.scandir(out_dir))
            except OSError:
                existing = 0
            if existing >= CHIP_OBJ_PER_CLASS_CAP:
                continue
        x0, y0 = gx * CHIP, gy * CHIP
        x1, y1 = x0 + CHIP, y0 + CHIP
        crop = img.crop((x0, y0, x1, y1))
        # 260506 — strict quality filter (was round 29 v15 lenient).
        # 양호 chip 이 fork/scratch 폴더에 들어가는 문제 fix.
        # 이중 검사:
        #   (1) total defect ratio (grade 2-30) ≥ 10% — chip 안의 defect 면적이 충분
        #   (2) strong grade ratio (grade 3-30) ≥ 2% — sprinkle (grade 2) 만 있는 chip 차단
        # invalid_main 은 white-dominant 가 정상 신호 → filter X
        if obj != 'invalid_main':
            arr = np.asarray(crop, dtype=np.uint8)                                        # palette indices 0-31
            n_defect_px = int(((arr >= 2) & (arr < 31)).sum())
            n_strong_px = int(((arr >= 3) & (arr < 31)).sum())                           # 260506: line/pattern grade 만
            if n_defect_px < int(MIN_CHIP_DEFECT_RATIO * arr.size):
                continue                                                                  # 양호 처럼 보임 → skip
            if n_strong_px < int(MIN_CHIP_STRONG_GRADE_RATIO * arr.size):
                continue                                                                  # sprinkle 만 있고 line pattern 없음 → skip
        crop_name = f"{wafer_key}_X{gx}_Y{gy}_B{b}.png"                                    # uppercase X/Y/B
        crop.save(os.path.join(out_dir, crop_name))
        counts[obj] = counts.get(obj, 0) + 1
    return counts

# v19o (260506): v19n BG 변경 REVERT — 사용자 명시 "chip 영역 전체 비율 늘리면 안됨, 불량영역만"
# normal chip baseline: P(0)+P(1) ≈ 98% — defect chip과 압도적 차이
BASELINE = np.array([0.83, 0.15, 0.012, 0.005, 0.002, 0.0008, 0.0001, 0.0001], dtype=np.float64)
BASELINE /= BASELINE.sum()
CUM_BASE = np.cumsum(BASELINE)

# defect chip 양호 영역 (라인 외 / 불량영역 밖): P(1) 0.25로 살짝 더 elevated
DEFECT_BG_DIST = np.array([0.73, 0.25, 0.012, 0.005, 0.002, 0.0008, 0.0001, 0.0001], dtype=np.float64)
DEFECT_BG_DIST /= DEFECT_BG_DIST.sum()
CUM_DEFECT_BG = np.cumsum(DEFECT_BG_DIST)

# v19s (260506): EDGE 원복 — bank_boundary 가 이제 2-stage 사용하므로 EDGE 영향 없음
EDGE_DIST = np.array([0.50, 0.40, 0.07, 0.02, 0.005, 0.003, 0.001, 0.001], dtype=np.float64)
EDGE_DIST /= EDGE_DIST.sum()
CUM_EDGE = np.cumsum(EDGE_DIST)

# v15: pixel 2 0.22→0.26, pixel 1 0.45→0.48, pixel 0 0.28→0.23 (v13/v14 중간)
EDGE_LINE_DIST = np.array([0.23, 0.48, 0.26, 0.020, 0.005, 0.003, 0.002, 0.001], dtype=np.float64)
EDGE_LINE_DIST /= EDGE_LINE_DIST.sum()
CUM_EDGE_LINE = np.cumsum(EDGE_LINE_DIST)

# ===== Round 29 v15: per-wafer intensity / baseline tier system =====
# v14 (round28_subset_test_big) wafer-only ep3 test_f1 99.59 saturated — too easy.
# 합성 데이터가 너무 uniform: 양호 (BASELINE) 와 불량 (OBJECT_DISTS) 모두 fixed deterministic.
# 실제 fab data 는 wafer 마다 강도 변동 + 양호 영역 noise 분포 변동.
# 이 round 에서 per-wafer 3-tier mix 도입:
#   baseline_tier: clean / normal / hazy → 양호 영역 grade 분포 변동
#   intensity_tier: strong / mid / weak → 불량 강도 (alpha scale + grade shift)
#   invalid_pct: 0.05 / 0.10 / 0.15 / 0.20 → invalid chip 비율 변동
# JSON wafer_meta 에 picked tiers 기록 → analysis / debug.
BASELINE_TIERS = {
    # v19o: BG 원복 (v19n REVERT)
    'clean':  np.array([0.95, 0.040, 0.008, 0.001, 0.0005, 0.0003, 0.0001, 0.0001], dtype=np.float64),
    'normal': BASELINE.copy(),
    'hazy':   np.array([0.65, 0.30,  0.040, 0.006, 0.002,  0.001,  0.0005, 0.0005], dtype=np.float64),
}
for _k in BASELINE_TIERS: BASELINE_TIERS[_k] /= BASELINE_TIERS[_k].sum()
CUM_BASELINE_TIERS = {k: np.cumsum(v) for k, v in BASELINE_TIERS.items()}

# intensity tier → alpha scale + grade shift
# v19q (260506): 약 chip floor 더 ↑ (image #24 너무 약 → #25 처럼 패턴 명확)
INTENSITY_ALPHA_SCALE = {'strong': 1.0, 'mid': 0.96, 'weak': 0.93}
INTENSITY_GRADE_SHIFT = {'strong': 0,   'mid': 0,    'weak': 0}

def pick_baseline_tier(rng):
    return str(rng.choice(['clean', 'normal', 'hazy'], p=[0.30, 0.50, 0.20]))

def pick_intensity_tier(rng):
    return str(rng.choice(['strong', 'mid', 'weak'], p=[0.50, 0.30, 0.20]))

def pick_invalid_pct(rng):
    return float(rng.choice([0.05, 0.10, 0.15, 0.20], p=[0.15, 0.25, 0.40, 0.20]))

def shifted_object_dist(obj, tier):
    """OBJECT_DISTS[obj] grade weights 를 strong/mid/weak 에 따라 0/1/2 step shift down.
    grade i 의 mass 는 grade max(0, i-shift) 로 이동. 합 1.0 보존."""
    base = OBJECT_DISTS[obj]
    shift = INTENSITY_GRADE_SHIFT[tier]
    if shift == 0: return base
    out = np.zeros_like(base)
    for i, p in enumerate(base):
        out[max(0, i - shift)] += p
    return out / out.sum()

# Object별 (main, sub) defect grade — main이 zone center에서 dominant, sub는 추가로 elevated
PRIMARY_GRADE = {
    'bank_boundary':  (1, 2),  # main=1, sub=2 (사용자: "main pixel을 1로하고 sub는 2로하자")
    'fork':           (3, 1),  # round 26 — main=3 (scratch 와 동일)
    'scratch':        (3, 1),  # main=3
    'scratch_rot':  (3, 1),  # main=3 (scratch와 동일 grade, 라인 각도로 구분)
}

# 불량영역 중앙 (zone center, alpha=1): main grade 거의 대부분 (~80%)
# v19j (260506): bank_boundary 살짝만 ↑ (v19i 너무 강했음).
OBJECT_DISTS = {
    # idx:           [  0,    1,    2,    3,    4,    5,     6,     7   ]
    # bank_boundary: grade 2 0.84→0.85, grade 3 0.10→0.11
    'bank_boundary':  np.array([0.003, 0.10,  0.85, 0.11, 0.004, 0.002, 0.001, 0.0], dtype=np.float64),
    # fork/scratch/scratch_rot OBJECT_DISTS — 2-stage 가 peak 처리. else branch (안 사용) 만 정의용.
    'fork':           np.array([0.025, 0.13, 0.72, 0.15, 0.010, 0.003, 0.0,    0.0], dtype=np.float64),
    'scratch':        np.array([0.025, 0.13, 0.72, 0.15, 0.010, 0.003, 0.0,    0.0], dtype=np.float64),
    'scratch_rot':    np.array([0.025, 0.13, 0.72, 0.15, 0.010, 0.003, 0.0,    0.0], dtype=np.float64),
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
    """3-letter + 3-digit prefix, ALL UPPERCASE for filename consistency."""
    letters = ''.join(chr(ord('A') + int(rng.integers(0, 26))) for _ in range(3))
    digits  = ''.join(str(int(rng.integers(0, 10)))             for _ in range(3))
    return letters + digits

LT_OPTIONS = ['PE', 'EE', 'PT']                                                       # Lot Type (filename 7번째 토큰)
TM_OPTIONS = ['NORMAL', 'PWQ', 'ENGINEER']                                            # Test Mode (filename 8번째 토큰)

# ===== Chip object alpha-fields =====
XC, YC = np.meshgrid(np.arange(CHIP), np.arange(CHIP), indexing='xy')

CHIP_BASE_ALPHA = 0.0                                                                  # 라인 외 영역은 DEFECT_BG_DIST 직접 사용 (object_dist mix 안 함 → 불량 pixel↑ 안 함)

def _perp_sharp_chip(d, sigma):
    """Lorentzian sharp + heavy tail (chip-internal scale, round 26).
    sharp σ_s=0.5σ + wide σ_w=3σ weight 0.4 — 가운데 좁고 매우 sharp + 부드러운 꼬리.
    d=0:1.0(clip) | σ:0.86 | 2σ:0.51 | 4σ:0.24 | 8σ:0.075 → 부드럽게 0.
    """
    sharp = 1.0 / (1.0 + (d/(0.5*sigma))**2)
    wide  = 0.40 / (1.0 + (d/(3.0*sigma))**2)
    return np.minimum(sharp + wide, 1.0).astype(np.float32)


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
    # v19t (260506): 약 chip floor ↑ (image #49 너무 약) + edge fade 완만 (image #50 끝부분 abrupt)
    # per-chip weak 비율 30% → 15%, weak floor 0.45→0.70 (식별 가능 수준 유지)
    chip_weak = rng.random() < 0.15
    if chip_weak:
        s_range = (0.78, 0.92)
        seg_range = (0.70, 0.90)
    else:
        s_range = (0.85, 1.00)
        seg_range = (0.70, 1.00)
    for cx in [50, 100, 150]:
        # v19u (260506): tail 더 길게 (사용자 "양호영역으로 급격히 가는걸 더 완만"). sigma_w 18-28 → 30-45
        sigma_s_i = float(rng.uniform(0.5, 1.0))                                       # peak sharp 유지
        sigma_m_i = float(rng.uniform(5.0, 9.0))                                       # middle 더 wide
        sigma_w_i = float(rng.uniform(30.0, 45.0))                                     # tail 매우 길게 — grade 1 halo 넓음
        s = rng.uniform(*s_range)
        seg_strengths = rng.uniform(*seg_range, size=n_seg).astype(np.float32)
        y_noise = np.repeat(seg_strengths, seg_len)[:CHIP, None]                       # (CHIP, 1) — Y 방향 산포
        line = _perp_profile(XC - cx, sigma_s=sigma_s_i, sigma_m=sigma_m_i, sigma_w=sigma_w_i) * y_noise * s
        a = np.maximum(a, line)
    for cy in [100]:
        sigma_s_i = float(rng.uniform(0.5, 1.0))
        sigma_m_i = float(rng.uniform(5.0, 9.0))
        sigma_w_i = float(rng.uniform(30.0, 45.0))
        s = rng.uniform(*s_range)
        seg_strengths = rng.uniform(*seg_range, size=n_seg).astype(np.float32)
        x_noise = np.repeat(seg_strengths, seg_len)[None, :CHIP]                       # (1, CHIP) — X 방향 산포
        line = _perp_profile(YC - cy, sigma_s=sigma_s_i, sigma_m=sigma_m_i, sigma_w=sigma_w_i) * x_noise * s
        a = np.maximum(a, line)
    return a

def _perp_asym_chip(d, sigma_left, sigma_right):
    """Asymmetric Lorentzian — 왼쪽 sharp (sigma_left) + 오른쪽 wide tail (paint smear).
    사용자 round 27v2: 면도날로 째진 후 물이 왼쪽→오른쪽 흘러가는 효과.
    sigma_right 는 scalar 또는 broadcast 가능 array (along-line 위치별 smear 강도)."""
    sigma = np.where(d < 0, sigma_left, sigma_right).astype(np.float32)
    return (1.0 / (1.0 + (d / sigma) ** 2)).astype(np.float32)


def _along_smooth_1d(rng, n_along, length):
    """1D smooth random variation (along-line direction 산포). Returns (length,) array.
    v15: var range 0.55-1.0 → 0.40-1.0 (사용자 image #72/74 — 자연현상 = y값 일정 X)."""
    var = rng.uniform(0.40, 1.0, size=n_along).astype(np.float32)
    pos = np.linspace(0, n_along - 1, length, dtype=np.float32)
    floor = np.clip(np.floor(pos).astype(np.int32), 0, n_along - 2)
    frac = (pos - floor).astype(np.float32)
    return (1 - frac) * var[floor] + frac * var[floor + 1]


def _along_taper_mask(length, line_start, line_end, fade_len=25.0):
    """v10 A: Soft taper — hard `(y >= start) & (y <= end)` 의 자연 대체.
    line_start/line_end 양 끝 fade_len px 부드러운 0→1→0 fade. line endpoint 가
    sparse defect points 로 점진적으로 사라짐 (자연 불량 분포)."""
    y = np.arange(length, dtype=np.float32)
    rise = np.clip((y - (line_start - fade_len)) / fade_len, 0.0, 1.0)
    fall = np.clip(((line_end + fade_len) - y) / fade_len, 0.0, 1.0)
    return (rise * fall).astype(np.float32)


def _along_smooth_range_1d(rng, length, lo, hi, n_segments=8):
    """v16: 1D smooth random in [lo, hi] — y 따라 변동 (sigma_left, smear_factor 등 jitter 용).
    n_segments control point uniform(lo, hi) → linear interp."""
    var = rng.uniform(lo, hi, size=n_segments).astype(np.float32)
    pos = np.linspace(0, n_segments - 1, length, dtype=np.float32)
    floor = np.clip(np.floor(pos).astype(np.int32), 0, n_segments - 2)
    frac = (pos - floor).astype(np.float32)
    return ((1 - frac) * var[floor] + frac * var[floor + 1]).astype(np.float32)


def _along_wobble_1d(rng, length, amp=2.0, n_segments=8):
    """v10 C: 1D smooth random wobble — line cx 가 along axis 따라 ±amp px 변동.
    n_segments uniform(-amp, amp) → linear interp. 라인이 perfect 직선 X (자연 노이즈)."""
    var = rng.uniform(-amp, amp, size=n_segments).astype(np.float32)
    pos = np.linspace(0, n_segments - 1, length, dtype=np.float32)
    floor = np.clip(np.floor(pos).astype(np.int32), 0, n_segments - 2)
    frac = (pos - floor).astype(np.float32)
    return ((1 - frac) * var[floor] + frac * var[floor + 1]).astype(np.float32)


def _along_center_peak(rng, length, line_start, line_end, sigma_factor=None, floor=0.0):
    """along-line center peak — envelope 다양 (cy shift + sigma 폭 random, asymmetric).
    round 27v8: floor 인자 추가 (default 0.0 backward compat).
    smear envelope (sigma_right control) 용도는 floor=0 유지 — 양 끝 narrow 효과 살아야.
    line 강도 (alpha 곱) 용도는 floor=0.75 정도 — 양 끝에서도 sharp peak 보이게."""
    if sigma_factor is None:
        sigma_factor = float(rng.uniform(1.6, 3.2))             # envelope 폭 다양
    half = (line_end - line_start) / 2
    # v17: cy shift 0.25 → 0.55 (사용자 image #75 — bell curve center 가운데 몰림 X, 더 random 위치)
    cy = (line_start + line_end) / 2 + float(rng.uniform(-half * 0.55, half * 0.55))
    sigma = max(half / sigma_factor, 8.0)
    yc = np.arange(length, dtype=np.float32)
    center_peak = np.exp(-((yc - cy) ** 2) / (2 * sigma * sigma)).astype(np.float32)
    # v15: smooth control point 5→8 (변동 frequency ↑) + 곱 (0.5+0.5*smooth) (변동 폭 ↑)
    smooth = _along_smooth_1d(rng, 8, length)
    env = (center_peak * (0.5 + 0.5 * smooth)).astype(np.float32)
    if floor > 0:
        env = np.maximum(env, floor)
    return env.astype(np.float32)


def alpha_fork(rng):
    """v10: endpoint fade(A) + bell smear(B) + line wobble(C) + extended tail(D) + center wider(E,F).
    1 horizontal top line + 4-6 vertical legs. weak_only 20% (severity ↓ + leg drop)."""
    a = np.full((CHIP, CHIP), CHIP_BASE_ALPHA, dtype=np.float32)
    weak_only = rng.random() < 0.02
    # v19z (260506): fork 더 sharp — smear 8-14 → 4-7 (halo 절반), peak sigma 1.8-2.3 → 1.0-1.5
    smear_factor = float(rng.uniform(4.0, 7.0)) if weak_only else float(rng.uniform(4.0, 7.0))
    severity = float(rng.uniform(0.98, 1.0)) if weak_only else float(rng.uniform(0.98, 1.0))
    # v19w (260506): fork 크기 ↓ + 위치 random. cy_h chip 안 어디든 (50-75 → 30-130)
    cy_h = float(rng.uniform(30, 130))
    # v20 (260507): fork 두께 ↑ 2px → 4px (사용자 directive). sigma 1.0-1.5 → 1.8-2.5 (~2× thick)
    sigma_h_base = float(rng.uniform(1.8, 2.5)) if not weak_only else float(rng.uniform(1.5, 2.0))
    # v16: sigma_h X 따라 변동 (horizontal line 굵기 좌/우 변동)
    sigma_h_jitter = _along_smooth_range_1d(rng, CHIP, 0.7, 1.4)[None, :]
    sigma_h_top = (sigma_h_base * sigma_h_jitter).astype(np.float32)
    # v16: smear_factor X 따라 변동 (퍼지는 폭 강약)
    smear_factor_jitter_h = _along_smooth_range_1d(rng, CHIP, 0.5, 1.5)[None, :]
    smear_factor_h = (smear_factor * smear_factor_jitter_h).astype(np.float32)
    # v19w (260506): fork 가로 크기 ↓ — width 145→90 평균 (chip 의 절반 정도)
    fork_width = float(rng.uniform(80, 130)) if not weak_only else float(rng.uniform(70, 100))
    fork_x_center = float(rng.uniform(fork_width / 2 + 15, CHIP - fork_width / 2 - 15))
    x_lo = fork_x_center - fork_width / 2
    x_hi = fork_x_center + fork_width / 2
    # v10 A: in_x hard mask → soft taper (X axis 양 끝 fade)
    fade_x = float(rng.uniform(15, 25))
    taper_x = _along_taper_mask(CHIP, x_lo, x_hi, fade_len=fade_x)
    in_x = taper_x[None, :]                                                 # (1, CHIP) broadcast over Y
    # v10 C: cy_h 가 X 별 ±wobble (horizontal line 도 perfect 직선 X)
    # v19v (260506): fork horizontal top wobble 줄임 — 라인 흐믈거림 완화 (사용자 "bank 처럼 직선으로")
    wobble_amp_h = float(rng.uniform(0.3, 0.8)) if not weak_only else float(rng.uniform(0.2, 0.6))
    cy_h_wobble = cy_h + _along_wobble_1d(rng, CHIP, amp=wobble_amp_h, n_segments=8)
    cy_per_x = cy_h_wobble[None, :]                                         # (1, CHIP) — X 별 cy 다름
    # v10 B: smear floor 0.18-0.25, strength floor 0.55-0.65
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
    # v11: alpha=1 stamp 폐기 — stochastic mix 만으로 자연 분포
    # v19z++ (260506): legs 더 많이 + 균일 spacing (사용자 directive). 5-6 → 7-9 + linspace + ±3 jitter
    n_legs = int(rng.integers(7, 10))
    leg_xs_base = np.linspace(x_lo + 8, x_hi - 8, n_legs)
    leg_xs_jit = rng.uniform(-3, 3, size=n_legs).astype(np.float32)
    leg_xs = sorted((leg_xs_base + leg_xs_jit).tolist())
    leg_strengths = rng.uniform(0.88, 1.0, size=n_legs).astype(np.float32)
    # v17: 라인별 smear_base random (각 leg 별로 퍼지는 강도 다름)
    leg_smear_bases = rng.uniform(0.5, 1.5, size=n_legs).astype(np.float32)
    if weak_only:
        n_keep = int(rng.integers(1, min(4, n_legs + 1)))
        keep_idx = set(rng.choice(n_legs, size=n_keep, replace=False).tolist())
        for k in range(n_legs):
            if k not in keep_idx:
                leg_strengths[k] = 0.0
    # v19w (260506): legs 길이 ↓ — chip 끝까지가 아니라 cy_h + uniform(70, 130) 까지만
    leg_height = float(rng.uniform(70, 130)) if not weak_only else float(rng.uniform(60, 100))
    leg_y_end = min(cy_h + leg_height, float(CHIP - 5))
    fade_y = float(rng.uniform(15, 25))
    taper_y = _along_taper_mask(CHIP, cy_h, leg_y_end, fade_len=fade_y)
    in_y = taper_y[:, None]
    for leg_idx, (cx, s_leg) in enumerate(zip(leg_xs, leg_strengths)):
        if s_leg < 0.10: continue
        # v20 (260507): fork leg 두께 ↑ 2px → 4px (사용자 directive). 0.9-1.4 → 1.7-2.4
        sigma_v_base = float(rng.uniform(1.7, 2.4)) if not weak_only else float(rng.uniform(1.4, 2.0))
        smear_factor_leg = smear_factor * float(leg_smear_bases[leg_idx])   # v17: leg 별 smear 강도
        sigma_v_jitter = _along_smooth_range_1d(rng, CHIP, 0.7, 1.4)[:, None]
        sigma_v_left = (sigma_v_base * sigma_v_jitter).astype(np.float32)
        smear_factor_jitter_v = _along_smooth_range_1d(rng, CHIP, 0.5, 1.5)[:, None]
        smear_factor_v = (smear_factor_leg * smear_factor_jitter_v).astype(np.float32)
        # v19v: fork legs wobble 줄임
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


def alpha_scratch(rng):
    """v10: endpoint fade(A) + bell smear(B) + along wobble(C) + extended tail(D) + center wider(E,F).
    v19z++ (260506): n_lines explicit (3-8, 작은 case 도) + per-line y center/length 산포
    (사용자 directive: '최소 3개, 작은것도, 길이/중심 산포')."""
    a = np.full((CHIP, CHIP), CHIP_BASE_ALPHA, dtype=np.float32)
    weak_only = rng.random() < 0.18
    n_lines = int(rng.integers(5, 11))                                      # v19z++: 3-8 → 5-10 (사용자)
    cx_list = sorted(rng.uniform(15, 185, size=n_lines).tolist())
    # per-line y center + half-height (length 산포 + center 산포)
    y_centers = rng.uniform(50, 150, size=n_lines).astype(np.float32)       # 중심 50-150
    y_halfs = rng.uniform(35, 95, size=n_lines).astype(np.float32)          # 반길이 35-95 → 길이 70-190
    line_y_starts = np.maximum(0.0, y_centers - y_halfs).astype(np.float32)
    line_y_ends = np.minimum(float(CHIP), y_centers + y_halfs).astype(np.float32)
    if weak_only:
        line_strengths = rng.uniform(0.96, 1.00, size=n_lines).astype(np.float32)
        smear_factor = float(rng.uniform(35.0, 50.0))
    else:
        line_strengths = rng.uniform(0.96, 1.00, size=n_lines).astype(np.float32)
        smear_factor = float(rng.uniform(35.0, 60.0))
    smear_base_per_line = rng.uniform(0.5, 1.5, size=n_lines).astype(np.float32)
    for i, cx in enumerate(cx_list):
        line_y_start_i = float(line_y_starts[i])
        line_y_end_i = float(line_y_ends[i])
        fade_len_i = float(rng.uniform(15, 28))                              # per-line fade
        taper_1d = _along_taper_mask(CHIP, line_y_start_i, line_y_end_i, fade_len=fade_len_i)
        in_y_range = taper_1d[:, None]
        sigma_left_base = float(rng.uniform(0.5, 0.8)) if not weak_only else float(rng.uniform(0.5, 0.7))
        smear_factor_line = smear_factor * float(smear_base_per_line[i])
        sigma_left_jitter = _along_smooth_range_1d(rng, CHIP, 0.7, 1.4)[:, None]
        sigma_left_per_y = (sigma_left_base * sigma_left_jitter).astype(np.float32)
        smear_factor_jitter = _along_smooth_range_1d(rng, CHIP, 0.5, 1.5)[:, None]
        smear_factor_per_y = (smear_factor_line * smear_factor_jitter).astype(np.float32)
        wobble_amp = float(rng.uniform(0.3, 0.8)) if not weak_only else float(rng.uniform(0.2, 0.6))
        cx_wobble = cx + _along_wobble_1d(rng, CHIP, amp=wobble_amp, n_segments=8)
        cx_per_y = cx_wobble[:, None]
        along_y_smear_1d = _along_center_peak(rng, CHIP, line_y_start_i, line_y_end_i,
                                              floor=float(rng.uniform(0.18, 0.25)),
                                              sigma_factor=float(rng.uniform(2.5, 4.0)))
        along_y_strength_1d = _along_center_peak(rng, CHIP, line_y_start_i, line_y_end_i,
                                                 floor=float(rng.uniform(0.78, 0.85)),
                                                 sigma_factor=float(rng.uniform(1.4, 2.5)))
        along_y_smear = along_y_smear_1d[:, None]
        along_y_strength = along_y_strength_1d[:, None]
        sigma_right_arr = (sigma_left_per_y * (1.0 + (smear_factor_per_y - 1.0) * along_y_smear)).astype(np.float32)
        d_perp = (XC - cx_per_y).astype(np.float32)
        perp = _perp_asym_chip(d_perp, sigma_left_per_y, sigma_right_arr)
        line = perp * in_y_range * line_strengths[i] * along_y_strength
        a = np.maximum(a, line)
    return a


def alpha_scratch_rot(rng):
    """v10: -21° rotation lines + endpoint fade(A) + bell smear(B) + cx wobble(C) + tail↑(D) + center wider(E,F)."""
    a = np.full((CHIP, CHIP), CHIP_BASE_ALPHA, dtype=np.float32)
    theta = np.deg2rad(-21.0)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cy = 100
    weak_only = rng.random() < 0.18
    # v19z++ (260506): scratch_rot 도 explicit n_lines + per-line along center/length 산포
    n_lines = int(rng.integers(7, 13))                                      # 7-12 lines (사용자 directive)
    cx_list = sorted(rng.uniform(15, 185, size=n_lines).tolist())
    if weak_only:
        line_strengths = rng.uniform(0.95, 1.00, size=n_lines).astype(np.float32)
        smear_factor = float(rng.uniform(18.0, 24.0))
    else:
        line_strengths = rng.uniform(0.95, 1.00, size=n_lines).astype(np.float32)
        smear_factor = float(rng.uniform(18.0, 26.0))
    smear_base_per_line = rng.uniform(0.5, 1.5, size=n_lines).astype(np.float32)
    peak_along_per_line = rng.uniform(0.20, 0.80, size=n_lines).astype(np.float32)
    # v19z++ : per-line along-axis center + half-length (line 길이/위치 산포)
    along_centers_per_line = rng.uniform(0.30, 0.70, size=n_lines).astype(np.float32)
    along_halfs_per_line = rng.uniform(0.22, 0.45, size=n_lines).astype(np.float32)
    d_along = sin_t * (XC - 100) + cos_t * (YC - 100)
    a_min = float(d_along.min()); a_max = float(d_along.max())
    along_norm = np.clip((d_along - a_min) / (a_max - a_min + 1e-6), 0, 1)
    n_bins = 100
    along_idx = np.clip(((d_along - a_min) / (a_max - a_min + 1e-6) * (n_bins - 1)),
                        0, n_bins - 1).astype(np.int32)
    for i, cx in enumerate(cx_list):
        peak_along_i = float(peak_along_per_line[i])
        # v19z++ : per-line along-axis taper (line 별 시작/끝 위치 다름 → 길이/중심 산포)
        along_start_i = float(along_centers_per_line[i] - along_halfs_per_line[i])
        along_end_i = float(along_centers_per_line[i] + along_halfs_per_line[i])
        fade_w_i = float(rng.uniform(0.04, 0.10))
        taper_along_i = (np.clip((along_norm - along_start_i) / fade_w_i, 0, 1) *
                         np.clip((along_end_i - along_norm) / fade_w_i, 0, 1)).astype(np.float32)
        sigma_n_i = float(rng.uniform(0.18, 0.28))
        center_peak_smear_i = np.exp(-((along_norm - peak_along_i) ** 2) / (2 * sigma_n_i * sigma_n_i)).astype(np.float32)
        n_along = 8
        along_var_i = rng.uniform(0.40, 1.0, size=n_along).astype(np.float32)
        a_pos_i = along_norm * (n_along - 1)
        a_floor_i = np.clip(np.floor(a_pos_i).astype(np.int32), 0, n_along - 2)
        frac_a_i = (a_pos_i - a_floor_i).astype(np.float32)
        smooth_i = (1 - frac_a_i) * along_var_i[a_floor_i] + frac_a_i * along_var_i[a_floor_i + 1]
        along_smear_i = np.maximum(center_peak_smear_i * (0.5 + 0.5 * smooth_i),
                                   float(rng.uniform(0.18, 0.25))).astype(np.float32)
        sigma_n_str_i = float(rng.uniform(0.30, 0.45))
        center_peak_str_i = np.exp(-((along_norm - peak_along_i) ** 2) / (2 * sigma_n_str_i * sigma_n_str_i)).astype(np.float32)
        along_strength_i = (np.maximum(center_peak_str_i * (0.5 + 0.5 * smooth_i),
                                       float(rng.uniform(0.78, 0.85))) * taper_along_i).astype(np.float32)

        sigma_left_base = float(rng.uniform(0.5, 0.8)) if not weak_only else float(rng.uniform(0.5, 0.7))
        smear_factor_line = smear_factor * float(smear_base_per_line[i])    # v17: 라인별 smear 강도
        sigma_left_jitter_1d = _along_smooth_range_1d(rng, n_bins, 0.7, 1.4)
        smear_factor_jitter_1d = _along_smooth_range_1d(rng, n_bins, 0.5, 1.5)
        sigma_left_per_pix = (sigma_left_base * sigma_left_jitter_1d[along_idx]).astype(np.float32)
        smear_factor_per_pix = (smear_factor_line * smear_factor_jitter_1d[along_idx]).astype(np.float32)
        sigma_right_arr = (sigma_left_per_pix * (1.0 + (smear_factor_per_pix - 1.0) * along_smear_i)).astype(np.float32)
        # v19v: scratch_rot wobble 줄임
        wobble_amp = float(rng.uniform(0.3, 0.8)) if not weak_only else float(rng.uniform(0.2, 0.6))
        wobble_1d = _along_wobble_1d(rng, n_bins, amp=wobble_amp, n_segments=8)
        wobble_per_pix = wobble_1d[along_idx]
        d_perp = ((XC - cx) * cos_t - (YC - cy) * sin_t) - wobble_per_pix
        perp = _perp_asym_chip(d_perp, sigma_left_per_pix, sigma_right_arr)
        line = perp * line_strengths[i] * along_strength_i
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
    'fork':             alpha_fork,                                                    # round 26
    'scratch':          alpha_scratch,
    'scratch_rot':    alpha_scratch_rot,
    'geometric_random': alpha_geometric_random,
    'small_dot':        alpha_small_dot,
    'fragment':         alpha_fragment,
    'irregular':        alpha_irregular,
    'ring_small':       alpha_ring_small,
}

CHIP_OBJECTS = ['bank_boundary', 'fork', 'scratch', 'scratch_rot']                   # round 26 spec
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
        # round 4 사용자 spec: 하단 row 좁게 fix + cx 중앙 jitter (회전·affine 제거, 거의 같은 위치)
        cy0 = int(rng.integers(GRID - 3, GRID))                                        # GRID-3..GRID-1
        cx0 = int(GRID // 2 + rng.integers(-3, 4))                                     # -3..+3
        if not inside[cy0, cx0]:                                                       # fallback: nearest inside
            ys, xs = np.where(inside)
            k = int(np.argmin((ys - cy0)**2 + (xs - cx0)**2))
            cy0, cx0 = int(ys[k]), int(xs[k])
        return _cluster_around(cy0, cx0, n, spread=0.8, inside=inside, rng=rng)
    elif class_name == 'Edge-Top':
        # round 4 사용자 spec: 상단 row 좁게 fix + cx 중앙 jitter (회전·affine 제거)
        cy0 = int(rng.integers(0, 3))                                                  # 0..2
        cx0 = int(GRID // 2 + rng.integers(-3, 4))
        if not inside[cy0, cx0]:
            ys, xs = np.where(inside)
            k = int(np.argmin((ys - cy0)**2 + (xs - cx0)**2))
            cy0, cx0 = int(ys[k]), int(xs[k])
        return _cluster_around(cy0, cx0, n, spread=0.8, inside=inside, rng=rng)
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

def pick_mixed_object(primary, rng, mix_ratio=0.05):
    """round 4: 95% primary, 5% other random object (사용자 "극소수만"). round 3 default 0.25 → 0.05."""
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

    # 0) Round 29 v15: pick per-wafer tiers (baseline noise + defect intensity + invalid pct)
    baseline_tier  = pick_baseline_tier(rng)                                          # clean/normal/hazy
    intensity_tier = pick_intensity_tier(rng)                                         # strong/mid/weak
    invalid_pct    = pick_invalid_pct(rng)                                            # 0.05/0.10/0.15/0.20
    cum_base_use   = CUM_BASELINE_TIERS[baseline_tier]                                # baseline canvas 분포

    # 1) Choose kind
    if object_name == 'invalid_main':
        kind = '00C'
    else:
        kind = '00P' if rng.random() < 0.5 else '00C'

    # 2) Plan defect / invalid masks (round 29 v15: invalid_pct per-wafer)
    normal_obj_map = None                                                             # Normal 클래스에서 chip별 object 지정용
    if class_name == 'Normal':
        defect_mask, normal_obj_map = select_normal_chips(rng, inside)
        invalid_inside_mask = select_random_invalid(rng, defect_mask, inside,
                                                    n=max(2, int(defect_mask.sum() * invalid_pct)))
    elif object_name == 'invalid_main':
        invalid_dist = select_distribution_chips(class_name, rng, inside)
        invalid_random = select_random_invalid(rng, invalid_dist, inside,
                                               n=max(2, int(invalid_dist.sum() * invalid_pct * 0.67)))  # 0.10/0.15 ≈ 0.67
        invalid_inside_mask = invalid_dist | invalid_random
        defect_mask = np.zeros((GRID, GRID), dtype=bool)
    else:
        defect_mask = select_distribution_chips(class_name, rng, inside)
        invalid_inside_mask = select_random_invalid(rng, defect_mask, inside,
                                                    n=max(2, int(defect_mask.sum() * invalid_pct)))
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
                        obj_actual = pick_mixed_object(object_name, rng, mix_ratio=0.05)  # round 4
                    chip_meta[(gy,gx)] = {'kind':'defect', 'obj': obj_actual,
                                          'bin': assign_defect_bin(kind, rng), 'inside': True}
    for gy in range(GRID):
        for gx in range(GRID):
            if invalid_mask[gy, gx]:
                chip_meta[(gy,gx)] = {'kind':'invalid', 'obj': None,
                                      'bin': assign_invalid_bin(kind, rng), 'inside': True}

    # 4) Canvas: wafer-level pink noise field (1/f^1.5) → grade 0/1/2 sampling.
    #    260507 v4: tier searchsorted REVERT — chip 경계 seam 없는 자연 sensor drift.
    #    floor 0.08 + cap 0.35 (max 강도 ↓), per-wafer overall_scale = beta(2,8) random.
    #    baseline_tier 은 JSON wafer_meta 기록용으로만 유지 (rendering 무관).
    wafer_pink = _pink_noise_field_2d(rng, SIZE, exponent=1.5)                       # (SIZE,SIZE) float32 ∈ [0,1]
    overall_scale = float(rng.beta(2, 8))                                             # per-wafer noise level
    p_bg_field = np.clip(overall_scale * (0.5 + 1.0 * wafer_pink), 0.08, 0.35).astype(np.float32)
    u_bg = rng.random((SIZE, SIZE))
    is_bg = u_bg < p_bg_field
    u_bg2 = rng.random((SIZE, SIZE))
    bg_grade = np.where(u_bg2 < 0.92, np.uint8(1), np.uint8(2))                      # noise: 92% grade1 + 8% grade2
    canvas = np.where(is_bg, bg_grade, np.uint8(0)).astype(np.uint8)                 # non-noise = grade 0 (white)
    del u_bg, u_bg2, bg_grade, wafer_pink
    inside_pix = np.repeat(np.repeat(inside, CHIP, axis=0), CHIP, axis=1)             # 6400x6400 bool
    canvas[~inside_pix] = IDX_BG                                                       # outside-wafer = bg color (no chip)

    # 5) Defect chips: alpha modulation per chip with assigned object
    #    Round 29 v15: intensity_tier per-wafer scales alpha + shifts grade dist
    alpha_scale = INTENSITY_ALPHA_SCALE[intensity_tier]
    for (gy, gx), meta in chip_meta.items():
        if meta['kind'] != 'defect': continue
        obj = meta['obj']
        alpha = ALPHA_FNS[obj](rng) * alpha_scale                                     # weak/mid/strong scale
        cum_obj = np.cumsum(shifted_object_dist(obj, intensity_tier))                 # grade shift down
        # 11단계 세분화 + 익스포넨셜 ramp toward CENTER
        # BG↔EDGE 0~0.40 (wider, smoother transition with normal area)
        # EDGE→CENTER 0.40~1.0 (power exp)
        # round 27v8: 모든 obj 3-way mix (BG/EDGE/CENTER) 통일.
        # 사용자: "pixel 2 sharp peak (CENTER), pixel 1 wider spread (EDGE)" 정확.
        # v20: fork/scratch/scratch_rot — 2-stage 분포 sampling (사용자 ratio.png idea).
        #   Stage 1: P(defect) = alpha            → is_defect ← rand1 < alpha
        #   Stage 2: P(2|defect) = smoothstep(alpha, 0.65, 0.92)  → is_2 ← rand2 < p_2
        #   defect & not 2: 95% pixel 1 + 4% pixel 3 + 1% pixel 4 (자연 noise)
        #   not defect:    CUM_BASE baseline (정상 chip 같은 BG 자연 fade)
        # 기타 obj (bank_boundary 등) — 기존 v19 smoothstep 3-way zone mix.
        # 260507 v5 unified — `_synth_chips_only.py` 와 동일 logic:
        #   fork:           2-stage smoothstep 0.50/0.88
        #   scratch / scr_rot: 2-stage smoothstep 0.60/0.91
        #   bank_boundary 등: 3-way zone mix with split 0.45/0.55
        # chip 의 non-defect baseline = wafer pink slice (chip 경계 seam 제거).
        y0p, x0p = gy * CHIP, gx * CHIP
        chip_p_bg = p_bg_field[y0p:y0p + CHIP, x0p:x0p + CHIP]                       # 200x200 slice
        u_chip_bg = rng.random((CHIP, CHIP))
        is_chip_bg = u_chip_bg < chip_p_bg
        u_chip_bg2 = rng.random((CHIP, CHIP))
        chip_bg_grade = np.where(u_chip_bg2 < 0.92, np.uint8(1), np.uint8(2))
        pink_baseline = np.where(is_chip_bg, chip_bg_grade, np.uint8(0)).astype(np.uint8)

        if obj in ('fork', 'scratch', 'scratch_rot'):
            if obj == 'fork':
                lo_t2, hi_t2 = 0.50, 0.88
            else:
                lo_t2, hi_t2 = 0.60, 0.91
            u1 = rng.random((CHIP, CHIP))
            is_defect = u1 < alpha
            t2 = np.clip((alpha - lo_t2) / (hi_t2 - lo_t2), 0.0, 1.0).astype(np.float32)
            p_2 = (t2 * t2 * (3.0 - 2.0 * t2)).astype(np.float32)
            u2 = rng.random((CHIP, CHIP))
            is_2 = u2 < p_2
            u3 = rng.random((CHIP, CHIP))
            defect_other = np.where(u3 < 0.95, np.uint8(1),
                            np.where(u3 < 0.99, np.uint8(3), np.uint8(4)))
            defect_grade = np.where(is_2, np.uint8(2), defect_other)
            grades = np.where(is_defect, defect_grade, pink_baseline).astype(np.uint8)
        else:
            # 3-way zone mix (bank_boundary etc) — v5 split 0.45/0.55
            t_low = np.clip(alpha / 0.45, 0.0, 1.0).astype(np.float32)
            t_high = np.clip((alpha - 0.45) / 0.55, 0.0, 1.0).astype(np.float32)
            s_low = (t_low * t_low * (3.0 - 2.0 * t_low)).astype(np.float32)
            s_high = (t_high * t_high * (3.0 - 2.0 * t_high)).astype(np.float32)
            mask_low = (alpha < 0.45).astype(np.float32)
            mask_high = 1.0 - mask_low
            w_bg     = (mask_low * (1.0 - s_low)).astype(np.float32)
            w_edge   = (mask_low * s_low + mask_high * (1.0 - s_high)).astype(np.float32)
            w_center = (mask_high * s_high).astype(np.float32)
            cum_mixed = (w_bg[..., None]     * CUM_DEFECT_BG[None,None,:] +
                         w_edge[..., None]   * CUM_EDGE[None,None,:] +
                         w_center[..., None] * cum_obj[None,None,:])
            uu = rng.random((CHIP, CHIP))
            grades = (uu[..., None] < cum_mixed).argmax(axis=-1).astype(np.uint8)
            # 3-way mix 의 low-alpha (background) 픽셀도 pink baseline 으로 override → chip 경계 seam 제거
            grades = np.where(alpha < 0.05, pink_baseline, grades).astype(np.uint8)
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

    # 10b) Save per-chip object-true crops for chip-object classifier dataset.
    # chip_meta[(gy,gx)]['obj'] 는 75% primary + 25% mixed 의 실제 object 라벨이라
    # 같은 wafer 안에 다른 object 가 섞여 있어도 정확히 분류됨 (folder-suffix weak label 대체).
    # Round 29 v15: weak intensity wafer 의 chip 은 시각적으로 양호 chip 같음 →
    # chip CNN 학습 데이터 noise 됨 → strong/mid tier wafer 만 chip crop 사용.
    if intensity_tier in ('strong', 'mid'):
        save_chip_crops(img, chip_meta, base)

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
        "wafer_meta": {                                                               # round 29 v15: per-wafer tier
            "synth_round": "29_v15",
            "baseline_tier": baseline_tier,
            "intensity_tier": intensity_tier,
            "invalid_pct": round(invalid_pct, 4),
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
    return png_path, int(defect_mask.sum()), int(invalid_inside_mask.sum()), kind, yld, syp

CLASSES = ['Center', 'Donut', 'Edge-Ring', 'Edge-Bottom', 'Edge-Top', 'Full', 'Thick-Edge',
           'Normal', 'Starburst', 'CommaCluster']
OBJECTS = ['bank_boundary', 'fork', 'scratch', 'scratch_rot', 'invalid_main']         # round 26 spec

# Build task list:
#   6 정규 dist (Center/Donut/Edge-Ring/Edge-Bottom/Edge-Top/Full) × 5 obj = 30
#   + Thick-Edge_invalid_main = 1
#   + Normal (object 없음, 단일 클래스) = 1
#   = 32 classes
def build_tasks(n_per_class, thick_fork_n=50):
    """round 4: Thick-Edge_fork 만 thick_fork_n (default 50, 다른 class 의 1/4) — 사용자 "극소수만"."""
    tasks = []
    for ci, cls in enumerate(CLASSES):
        for oi, obj in enumerate(OBJECTS):
            if cls == 'Thick-Edge' and obj not in ('invalid_main', 'fork'): continue   # round 26: fork 추가
            if cls == 'Normal' and oi > 0: continue                                   # Normal은 1 클래스
            if cls in ('Starburst', 'CommaCluster') and oi > 0: continue              # wafer-canvas 1 클래스
            n_for_this = thick_fork_n if (cls == 'Thick-Edge' and obj == 'fork') else n_per_class
            for s in range(n_for_this):
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
    p.add_argument('--only-class', type=str, default=None,
                   help='generate only one distribution/class label, e.g. Normal or Normal_bank_boundary')
    p.add_argument('--seed-offset', type=int, default=0,
                   help='offset generated seeds to avoid overwriting an existing range')
    p.add_argument('--thick-fork-n', type=int, default=50,
                   help='round 4: Thick-Edge_fork sample count (사용자 "극소수만"). default 50.')
    args = p.parse_args()

    tasks = build_tasks(args.n, thick_fork_n=args.thick_fork_n)
    if args.only_class:
        tasks = [t for t in tasks if t[0] == args.only_class or f"{t[0]}_{t[1]}" == args.only_class]
        print(f"Filter --only-class={args.only_class}: {len(tasks)} tasks")
    if args.seed_offset:
        tasks = [(cls, obj, seed + args.seed_offset) for (cls, obj, seed) in tasks]
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
