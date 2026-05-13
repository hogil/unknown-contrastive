"""WM-811K cca/ (8-class, 1779 wafer) zero-shot evaluation with NEW SOTA ckpt.

Compare with:
  - ICH (arXiv 2404.15436) WM1K Hom 0.90
  - Mean Teacher+SupCon (arXiv 2411.18533) WM-811K F1 0.834 (supervised, different task)
"""
import os, sys, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import timm
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
import hdbscan
from sklearn.metrics import (
    adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, silhouette_score,
)

CKPT = 'D:/project/unknown-contrastive/outputs_contrastive_260512_001719/checkpoints/final_infer.pt'
DATA_ROOT = Path('D:/project/data/wm-811k/cca')
OUT = Path('D:/project/unknown-contrastive/docs/paper/manager_report/_wm811k_cca_eval_result.json')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}', flush=True)

backbone = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384', pretrained=False, num_classes=0).to(device).eval()
feat_dim = backbone.num_features
head = nn.Sequential(
    nn.Linear(feat_dim, feat_dim),
    nn.GELU(),
    nn.Linear(feat_dim, 128),
).to(device).eval()

ckpt = torch.load(CKPT, map_location=device, weights_only=False)
sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
bb_sd = {k.replace('backbone.', '', 1): v for k, v in sd.items() if k.startswith('backbone.')}
head_sd = {k.replace('head.', '', 1): v for k, v in sd.items() if k.startswith('head.')}
miss_bb, _ = backbone.load_state_dict(bb_sd, strict=False)
miss_hd, _ = head.load_state_dict(head_sd, strict=False)
print(f'  backbone missing={len(miss_bb)}  head missing={len(miss_hd)}', flush=True)

tf = T.Compose([
    T.Resize((384, 384), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Collect files
files = []
labels = []
for cls_dir in sorted(DATA_ROOT.iterdir()):
    if not cls_dir.is_dir() or cls_dir.name == 'labels': continue
    for f in sorted(cls_dir.glob('*.png')):
        files.append(str(f))
        labels.append(cls_dir.name)

print(f'\nTotal: {len(files)} files, {len(set(labels))} classes', flush=True)
print(f'Per-class:', flush=True)
from collections import Counter
for cls, n in sorted(Counter(labels).items()):
    print(f'  {cls}: {n}', flush=True)

# Extract embeddings
print(f'\nExtracting embeddings ...', flush=True)
embs = []
BATCH = 32
t0 = time.time()
with torch.no_grad():
    for start in range(0, len(files), BATCH):
        xs = []
        for fp in files[start:start+BATCH]:
            img = Image.open(fp).convert('RGB')  # RGBA → RGB
            xs.append(tf(img))
        x = torch.stack(xs).to(device)
        f_ = backbone(x)
        e = head(f_)
        e = torch.nn.functional.normalize(e, dim=1)
        embs.append(e.cpu().numpy())
embs = np.vstack(embs)
print(f'  embedding shape: {embs.shape}  elapsed={time.time()-t0:.0f}s', flush=True)

y = np.array(labels)

# Tier1 HDBSCAN
print(f'\n=== HDBSCAN (Tier1: eom mcs=12 ms=3) ===', flush=True)
cl = hdbscan.HDBSCAN(min_cluster_size=12, min_samples=3,
                      cluster_selection_method='eom', metric='euclidean')
pred = cl.fit_predict(embs.astype(np.float32))
nm = pred != -1
classes_uniq = sorted(set(y))
captured = sum(1 for c in classes_uniq if any((y == c) & nm))
cap = captured / len(classes_uniq)

result = dict(
    dataset='WM-811K_cca',
    n_total=int(len(files)),
    n_classes=len(classes_uniq),
    classes=classes_uniq,
    per_class_count={c: int((y==c).sum()) for c in classes_uniq},
)

if nm.sum() < 2:
    print('  ★ catastrophic — fewer than 2 clustered points')
    result.update(dict(cap=cap, noise=100.0, ari=0, ami=0, hom=0, com=0, sil=float('nan'), n_clusters=0))
else:
    ari = adjusted_rand_score(y[nm], pred[nm])
    ami = adjusted_mutual_info_score(y[nm], pred[nm])
    hom = homogeneity_score(y[nm], pred[nm])
    com = completeness_score(y[nm], pred[nm])
    noise = (~nm).mean()*100
    n_cl = len(set(pred[nm]))
    try:
        e_norm = embs[nm] / (np.linalg.norm(embs[nm], axis=1, keepdims=True) + 1e-12)
        sil = float(silhouette_score(e_norm, pred[nm], metric='cosine'))
    except Exception:
        sil = float('nan')
    print(f'  ARI={ari:.4f}  AMI={ami:.4f}  Hom={hom:.4f}  Comp={com:.4f}  Sil={sil:.4f}')
    print(f'  noise%={noise:.2f}  n_clusters={n_cl}  capture={cap:.4f}')

    # per-cluster majority class breakdown
    print(f'\n  Per-cluster majority class:')
    for c in sorted(set(pred[nm])):
        members = y[nm][pred[nm] == c]
        from collections import Counter
        mc = Counter(members).most_common(3)
        purity = mc[0][1] / sum(c2 for _, c2 in mc)
        print(f'    cluster {c}: n={len(members)}  top={mc[:3]}  purity={purity:.3f}')

    result.update(dict(
        p1_capture=float(cap),
        p2_noise_pct=float(noise),
        p3_completeness=float(com),
        p4_homogeneity=float(hom),
        ari=float(ari),
        ami=float(ami),
        sil=sil,
        n_clusters=n_cl,
    ))

result['ich_wm1k_hom_baseline'] = 0.90
result['ckpt'] = CKPT
OUT.write_text(json.dumps(result, indent=2))
print(f'\n[OUT] {OUT}', flush=True)
