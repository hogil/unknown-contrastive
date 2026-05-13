"""MixedWM38 val/test split eval using newly trained NEW recipe ckpt."""
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

CKPT = 'D:/project/unknown-contrastive/outputs_contrastive_260513_184308/checkpoints/final_infer.pt'
DATA_ROOT = Path('D:/dataset/MixedWM38_folder')
OUT = Path('D:/project/unknown-contrastive/docs/paper/manager_report/_mixedwm38_train_eval_result.json')

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

def extract_split(split_name):
    files = []
    labels = []
    for cls_dir in sorted((DATA_ROOT / split_name).iterdir()):
        if not cls_dir.is_dir(): continue
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() == '.png':
                files.append(str(f))
                labels.append(cls_dir.name)
    print(f'  {split_name}: {len(files)} files, {len(set(labels))} classes', flush=True)
    embs = []
    BATCH = 32
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(files), BATCH):
            xs = []
            for fp in files[start:start+BATCH]:
                img = Image.open(fp).convert('RGB')
                xs.append(tf(img))
            x = torch.stack(xs).to(device)
            f_ = backbone(x)
            e = head(f_)
            e = torch.nn.functional.normalize(e, dim=1)
            embs.append(e.cpu().numpy())
            if start % (BATCH * 50) == 0:
                el = time.time() - t0
                rate = (start + BATCH) / max(el, 1e-6)
                eta = (len(files) - start - BATCH) / max(rate, 1e-6)
                print(f'    {start+BATCH}/{len(files)}  {el:.0f}s  rate={rate:.1f}/s  ETA={eta:.0f}s', flush=True)
    return np.vstack(embs), np.array(labels)

def evaluate(emb, y, name):
    cl = hdbscan.HDBSCAN(min_cluster_size=12, min_samples=3,
                          cluster_selection_method='eom', metric='euclidean')
    pred = cl.fit_predict(emb.astype(np.float32))
    nm = pred != -1
    classes = sorted(set(y))
    captured = sum(1 for c in classes if any((y == c) & nm))
    cap = captured / len(classes)
    if nm.sum() < 2:
        return dict(cap=cap, noise=100.0, ari=0, ami=0, hom=0, com=0, sil=float('nan'), n_clusters=0)
    res = dict(
        cap=cap,
        noise=(~nm).mean()*100,
        ari=adjusted_rand_score(y[nm], pred[nm]),
        ami=adjusted_mutual_info_score(y[nm], pred[nm]),
        hom=homogeneity_score(y[nm], pred[nm]),
        com=completeness_score(y[nm], pred[nm]),
        n_clusters=len(set(pred[nm])),
    )
    try:
        e_norm = emb[nm] / (np.linalg.norm(emb[nm], axis=1, keepdims=True) + 1e-12)
        res['sil'] = float(silhouette_score(e_norm, pred[nm], metric='cosine'))
    except Exception:
        res['sil'] = float('nan')
    return res

result = {}
for split in ['val', 'test']:
    print(f'\n== {split} ==', flush=True)
    emb, y = extract_split(split)
    r = evaluate(emb, y, split)
    result[split] = r
    print(f'  {split}: cap={r["cap"]:.4f}  noise={r["noise"]:.2f}%  ARI={r["ari"]:.4f}  AMI={r["ami"]:.4f}  Hom={r["hom"]:.4f}  Comp={r["com"]:.4f}  Sil={r["sil"]:.4f}  n_clu={r["n_clusters"]}', flush=True)

result['decor_baseline_ari'] = 0.296
result['delta_vs_decor'] = result['test']['ari'] - 0.296
result['ckpt'] = CKPT

print('\n=== Summary ===')
for k, v in result.items():
    if isinstance(v, dict):
        print(f'  {k}: {v}')
    else:
        print(f'  {k}: {v}')

OUT.write_text(json.dumps(result, indent=2))
print(f'\n[OUT] {OUT}', flush=True)
