"""MixedWM38 zero-shot evaluation with NEW SOTA contrastive checkpoint.

Pipeline:
  1. Load Wafer_Map_Datasets.npz (38015 × 52×52 int + 38015 × 8 one-hot)
  2. Convert 52×52 int (0/1/2/3) to 384×384 RGB
  3. Extract embedding via NEW ckpt (iter 70, ts=260512_001719)
  4. HDBSCAN cluster (Tier1 protocol: eom mcs=12 ms=3, defect-only)
  5. Compute Tier 1+2 metrics vs composite multi-hot labels (38 unique)
  6. Compare with DECOR baseline (ARI 0.296)

Output: D:/project/unknown-contrastive/docs/paper/manager_report/_mixedwm38_zero_shot_result.json
"""
import os, sys, time, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import timm
from PIL import Image
import torchvision.transforms as T
from pathlib import Path

NPZ = 'D:/dataset/MixedWM38/Wafer_Map_Datasets.npz'
CKPT = 'D:/project/unknown-contrastive/outputs_contrastive_260512_001719/checkpoints/final_infer.pt'
OUT_DIR = Path('D:/project/unknown-contrastive/docs/paper/manager_report')
EMB_NPY = OUT_DIR / '_mixedwm38_embedding.npy'
LABEL_NPY = OUT_DIR / '_mixedwm38_labels.npy'
RESULT_JSON = OUT_DIR / '_mixedwm38_zero_shot_result.json'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}', flush=True)

# --- Step 1: Load data ---
print('Loading npz ...', flush=True)
d = np.load(NPZ)
maps = d['arr_0']
labels = d['arr_1']
print(f'  maps: {maps.shape} {maps.dtype}', flush=True)
print(f'  labels: {labels.shape}', flush=True)

# --- Step 2: Convert to 384×384 RGB ---
# pixel 0 (blank) -> black (0)
# pixel 1 (pass)  -> mid-grey (128)
# pixel 2 (fail)  -> red-ish, but for embedding we just need consistent palette
# pixel 3         -> another defect tier
# Use 3-channel: R = fail intensity, G = passing, B = blank
def to_rgb(wm):
    """52×52 int -> 52×52 RGB uint8."""
    r = (wm == 2).astype(np.uint8) * 255 + (wm == 3).astype(np.uint8) * 200
    g = (wm == 1).astype(np.uint8) * 200
    b = (wm == 0).astype(np.uint8) * 50
    return np.stack([r, g, b], axis=-1)

tf = T.Compose([
    T.Resize((384, 384), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# --- Step 3: Backbone + projection head from ckpt ---
print('Loading NEW SOTA backbone + head ...', flush=True)
backbone = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384', pretrained=False, num_classes=0).to(device).eval()
# Build a simple head: feature_dim -> 128 (matches CFG.PROJ_DIM)
feat_dim = backbone.num_features
import torch.nn as nn
head = nn.Sequential(
    nn.Linear(feat_dim, feat_dim),
    nn.GELU(),
    nn.Linear(feat_dim, 128),
).to(device).eval()

# Load ckpt state into backbone + head
ckpt = torch.load(CKPT, map_location=device, weights_only=False)
sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
# Split into backbone.* and head.* etc.
bb_sd = {k.replace('backbone.', '', 1): v for k, v in sd.items() if k.startswith('backbone.')}
head_sd = {k.replace('head.', '', 1): v for k, v in sd.items() if k.startswith('head.')}
miss_bb, unexp_bb = backbone.load_state_dict(bb_sd, strict=False)
miss_hd, unexp_hd = head.load_state_dict(head_sd, strict=False)
print(f'  backbone: missing={len(miss_bb)} unexpected={len(unexp_bb)}', flush=True)
print(f'  head    : missing={len(miss_hd)} unexpected={len(unexp_hd)}', flush=True)

# --- Step 4: Extract embeddings ---
if EMB_NPY.exists():
    print(f'Found cached {EMB_NPY} — loading', flush=True)
    embs = np.load(EMB_NPY)
else:
    print(f'Extracting {len(maps)} embeddings (BATCH=32) ...', flush=True)
    embs_list = []
    BATCH = 32
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(maps), BATCH):
            chunk = maps[start:start+BATCH]
            xs = []
            for wm in chunk:
                rgb = to_rgb(wm)
                img = Image.fromarray(rgb)
                xs.append(tf(img))
            x = torch.stack(xs).to(device)
            feat = backbone(x)
            emb = head(feat)
            emb = torch.nn.functional.normalize(emb, dim=1)
            embs_list.append(emb.cpu().numpy())
            if start % (BATCH * 50) == 0:
                done = start + BATCH
                el = time.time() - t0
                rate = done / max(el, 1e-6)
                eta = (len(maps) - done) / max(rate, 1e-6)
                print(f'  {done}/{len(maps)}  {el:.0f}s  rate={rate:.1f}/s  ETA={eta:.0f}s', flush=True)
    embs = np.vstack(embs_list)
    np.save(EMB_NPY, embs)
    print(f'  Saved {EMB_NPY}  shape={embs.shape}', flush=True)

# --- Step 5: Composite label per sample ---
composite = np.array([''.join(str(int(b)) for b in row) for row in labels])
uniq = sorted(set(composite))
print(f'\nComposite labels: {len(uniq)} unique', flush=True)
np.save(LABEL_NPY, composite)

# --- Step 6: HDBSCAN cluster + Tier 1+2 metrics ---
print('\nClustering with HDBSCAN (Tier1: eom mcs=12 ms=3) ...', flush=True)
import hdbscan
from sklearn.metrics import (
    adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, silhouette_score,
)

# defect-only: label sum > 0
defect_mask = labels.sum(axis=1) > 0
print(f'  defect samples: {defect_mask.sum()} / {len(labels)}', flush=True)

emb_d = embs[defect_mask]
y_d = composite[defect_mask]

t0 = time.time()
cl = hdbscan.HDBSCAN(min_cluster_size=12, min_samples=3,
                      cluster_selection_method='eom', metric='euclidean')
pred = cl.fit_predict(emb_d.astype(np.float32))
fit_t = time.time() - t0
print(f'  fit time: {fit_t:.1f}s', flush=True)

nm = pred != -1
n_clusters = len(set(pred[nm]))
noise_pct = (~nm).mean() * 100
classes = sorted(set(y_d))
captured = sum(1 for c in classes if any((y_d == c) & nm))
capture_rate = captured / len(classes)

if nm.sum() < 2:
    print('  too few clustered points — abort metric')
    sys.exit(1)

ari = adjusted_rand_score(y_d[nm], pred[nm])
ami = adjusted_mutual_info_score(y_d[nm], pred[nm])
nmi = normalized_mutual_info_score(y_d[nm], pred[nm])
hom = homogeneity_score(y_d[nm], pred[nm])
com = completeness_score(y_d[nm], pred[nm])

# silhouette cosine on clustered only
try:
    e_nm = emb_d[nm]
    e_nm_norm = e_nm / (np.linalg.norm(e_nm, axis=1, keepdims=True) + 1e-12)
    sil = float(silhouette_score(e_nm_norm, pred[nm], metric='cosine'))
except Exception as exc:
    sil = float('nan')
    print(f'  silhouette failed: {exc}', flush=True)

result = dict(
    dataset='MixedWM38',
    n_total=int(len(maps)),
    n_defect=int(defect_mask.sum()),
    n_normal=int((~defect_mask).sum()),
    n_unique_composite=len(uniq),
    n_clusters_found=n_clusters,
    noise_pct=float(noise_pct),
    p1_capture=float(capture_rate),
    p2_noise_pct=float(noise_pct),
    p3_completeness=float(com),
    p4_homogeneity=float(hom),
    ari=float(ari),
    ami=float(ami),
    nmi=float(nmi),
    silhouette=sil,
    hdbscan_cfg=dict(method='eom', mcs=12, ms=3, epsilon=0.0),
    fit_time_sec=fit_t,
    ckpt=CKPT,
    decor_arxiv_baseline_ari=0.296,
    delta_vs_decor=float(ari - 0.296),
)
print()
print('=== MixedWM38 zero-shot result ===')
for k, v in result.items():
    print(f'  {k}: {v}')

RESULT_JSON.write_text(json.dumps(result, indent=2))
print(f'\n[OUT] {RESULT_JSON}', flush=True)
print(f'[OUT] {EMB_NPY}', flush=True)
