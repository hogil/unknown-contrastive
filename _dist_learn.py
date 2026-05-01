"""Learn defect chip distribution from WM-811K cca/* per class.

각 클래스별로:
- defect mask = (pixel == 255)
- wafer mask = (pixel != 0)  i.e., 128 or 255 = inside wafer
- 모든 샘플 mask를 wafer 중심·반경으로 정규화한 좌표계에 누적
- 32x32 chip grid에 reproject한 P(defect|chip) heatmap 생성
- (256x256 고해상 heatmap도 함께 저장해서 시각화)
"""
import os, glob, numpy as np
from PIL import Image

CCA_ROOT = "D:/project/data/wm-811k/cca"
CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random"]
OUT_DIR = "D:/project/unknown-contrastive/_dist_heatmaps"
os.makedirs(OUT_DIR, exist_ok=True)

GRID = 32                                                                             # target chip grid
HIRES = 256                                                                           # high-res viz heatmap
MAX_PER_CLASS = 400                                                                   # cap for speed

def load_mask(path):
    """Return (defect_bool, wafer_bool) at native size."""
    arr = np.array(Image.open(path))
    if arr.ndim == 3: arr = arr[..., 0]                                               # take R channel from RGBA
    defect = (arr == 255)
    wafer  = (arr != 0)
    return defect, wafer

def fit_circle(wafer_mask):
    """Estimate wafer center (cy, cx) and radius from binary mask."""
    ys, xs = np.where(wafer_mask)
    if len(ys) == 0: return None
    cy, cx = ys.mean(), xs.mean()
    r = np.sqrt(((ys - cy)**2 + (xs - cx)**2).mean()) * np.sqrt(2)                    # rough
    return cy, cx, r

def reproject_to_unit(defect_mask, wafer_mask, target_size):
    """Reproject defect mask to [0,1]x[0,1] unit-disk centered grid of target_size."""
    fc = fit_circle(wafer_mask)
    if fc is None: return None
    cy, cx, r = fc
    yy, xx = np.mgrid[0:target_size, 0:target_size]
    # target center at (target_size/2, target_size/2) with radius target_size/2
    ty = (yy + 0.5) - target_size/2
    tx = (xx + 0.5) - target_size/2
    # map target coord -> source coord
    sy = cy + ty * (r / (target_size/2))
    sx = cx + tx * (r / (target_size/2))
    sy_i = np.clip(sy.round().astype(int), 0, defect_mask.shape[0]-1)
    sx_i = np.clip(sx.round().astype(int), 0, defect_mask.shape[1]-1)
    defect_resampled = defect_mask[sy_i, sx_i]
    wafer_resampled  = wafer_mask[sy_i, sx_i]
    # outside-wafer cells set to NaN-like (we'll mark separately)
    return defect_resampled.astype(np.float32), wafer_resampled

def build_class_heatmap(cls, target_size, max_n=MAX_PER_CLASS):
    files = sorted(glob.glob(os.path.join(CCA_ROOT, cls, "*.png")))[:max_n]
    if not files: return None, None, 0
    acc_defect = np.zeros((target_size, target_size), dtype=np.float64)
    acc_wafer  = np.zeros((target_size, target_size), dtype=np.float64)
    used = 0
    for f in files:
        defect, wafer = load_mask(f)
        rep = reproject_to_unit(defect, wafer, target_size)
        if rep is None: continue
        d, w = rep
        acc_defect += d
        acc_wafer  += w.astype(np.float64)
        used += 1
    # P(defect | wafer) per cell
    eps = 1e-6
    p_defect = acc_defect / (acc_wafer + eps)
    p_wafer  = acc_wafer / max(1, used)                                               # how often this cell is inside any wafer
    # mask cells that are rarely inside any wafer
    p_defect[p_wafer < 0.1] = 0.0
    return p_defect.astype(np.float32), p_wafer.astype(np.float32), used

def viz(arr, path):
    """Save normalized heatmap as 8-bit grayscale png."""
    a = arr.copy()
    if a.max() > 0: a = a / a.max()
    img = (a * 255).astype(np.uint8)
    Image.fromarray(img).save(path)

print("Building class heatmaps...")
for cls in CLASSES:
    pd32, pw32, n32 = build_class_heatmap(cls, GRID)
    pdh,  pwh, nh   = build_class_heatmap(cls, HIRES)
    if pd32 is None:
        print(f"  {cls}: SKIP (no files)")
        continue
    np.save(os.path.join(OUT_DIR, f"{cls}_p_defect_32.npy"), pd32)
    np.save(os.path.join(OUT_DIR, f"{cls}_p_wafer_32.npy"), pw32)
    viz(pd32, os.path.join(OUT_DIR, f"{cls}_p_defect_32.png"))
    viz(pdh,  os.path.join(OUT_DIR, f"{cls}_p_defect_256.png"))
    print(f"  {cls}: n={n32}  max_p={pd32.max():.3f}  mean_p={pd32.mean():.3f}  inside_cells={int((pw32>=0.1).sum())}/{GRID*GRID}")

print(f"\nSaved heatmaps to {OUT_DIR}")
