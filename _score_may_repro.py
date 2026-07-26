#!/usr/bin/env python3
# ★ 계획 Step 3 채점: _may_repro_src.py 재현 run 의 proj_ep{ep}.pt 를 epoch 별로 canonical HDBSCAN 채점.
# backbone = 학습에 쓴 동일 frozen backbone(backbone.pth), 경로 = training-faithful(normalize + manual GAP).
# reference (배포 b4 학습 proj): z_norm = 30/37, noise 3.1, Comp 0.995, Hom 0.939, Sil 0.807.
import os, sys, glob, re
import torch, timm, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torchvision import transforms as T
from PIL import Image

import os as _os
BACKBONE = sys.argv[2] if len(sys.argv) > 2 else r"D:/project/unknown-contrastive/_b4_backbone.pth"   # argv[2]=backbone override (FCMAE 등)
_rd = _os.environ.get("REPRO_DATA")   # env REPRO_DATA=<dir[,dir2]> 로 eval pool 교체 (기본 anchor)
ROOTS = [Path(x) for x in _rd.split(",")] if _rd else [Path("data/images/anchor_avg30_repro")]
RUN_GLOB = sys.argv[1] if len(sys.argv) > 1 else "runs/may_repro/run_*/checkpoints"
IMG = 384
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def build_proj():
    return nn.Sequential(nn.Linear(1024, 1024, bias=False), nn.BatchNorm1d(1024),
                         nn.ReLU(inplace=True), nn.Linear(1024, 128))

print(f"[load] backbone {BACKBONE}", flush=True)
bb = timm.create_model("convnextv2_base.fcmae_ft_in22k_in1k_384", pretrained=False, num_classes=0, global_pool="")
sd = torch.load(BACKBONE, map_location="cpu")
if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
for pref in ("model.", "backbone.", "module."):
    if any(k.startswith(pref) for k in sd.keys()):
        sd = {(k[len(pref):] if k.startswith(pref) else k): v for k, v in sd.items()}
bb.load_state_dict(sd, strict=False)
bb.eval().to(DEV)

tf = T.Compose([T.Resize((IMG, IMG), interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])   # training-faithful
paths, labels = [], []
for R in ROOTS:
    for cd in sorted(R.iterdir()):
        if cd.is_dir():
            for p in sorted(cd.rglob("*")):
                if p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                    paths.append(p); labels.append(cd.name)
print(f"[data] {len(paths)} imgs, {len(set(labels))} classes from {len(ROOTS)} roots", flush=True)

# backbone pooled feature 한 번만 (backbone frozen → epoch 무관)
feats = []
with torch.no_grad():
    for i in range(0, len(paths), 32):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i+32]]).to(DEV)
        feats.append(bb.forward_features(x).mean(dim=(2, 3)).cpu())
feats = torch.cat(feats, 0)
print(f"[feat] {tuple(feats.shape)} (backbone frozen, cached)", flush=True)

ckdirs = sorted(glob.glob(RUN_GLOB))
if not ckdirs:
    print(f"[err] no ckpt dir match {RUN_GLOB}"); sys.exit(1)
ckdir = ckdirs[-1]
print(f"[run] {ckdir}", flush=True)
eps = sorted(glob.glob(os.path.join(ckdir, "proj_ep*.pt")),
             key=lambda s: int(re.search(r"ep(\d+)", s).group(1)))

sys.path.insert(0, "scripts")
from eval_may37_checkpoints import calculate_metrics
IGN = {"Normal", "Random", "R"}
hdr = f"{'ep':>3s}  {'G':>6s} {'Q':>6s} {'L':>6s} | P1     noise% Comp   Hom    k/tgt/noise      frag   Sil    ARI"
print("\n=== 재현 epoch별 — canonical HDBSCAN (ref: 배포 b4 z=30/37 noise3.1 Comp.995 Hom.939 Sil.807) ===", flush=True)
print(hdr, flush=True); print("-"*len(hdr), flush=True)
for ck in eps:
    d = torch.load(ck, map_location="cpu")
    pj = {k[len("net."):] if k.startswith("net.") else k: v for k, v in d["proj"].items()}
    proj = build_proj(); proj.load_state_dict(pj); proj.eval().to(DEV)
    with torch.no_grad():
        z = F.normalize(proj(feats.to(DEV)), dim=1).cpu().numpy().astype("float32")
    r = calculate_metrics(z, labels, IGN)
    kstr = f"{r['k']}/{r['target_class_count']}/{r['noise_count']}"
    g, q, l = d.get("G", 0), d.get("Q", 0), d.get("L", 0)
    print(f"{d['epoch']:>3d}  {g:6.3f} {q:6.3f} {l:6.3f} | {str(r['P1_capture']):6s} "
          f"{r['P2_noise_pct']:5.1f} {r['P3_completeness']:.3f} {r['P4_homogeneity']:.3f} "
          f"{kstr:15s} {r['fragment_ratio']:.3f}  {r['Sil_cos']:.3f}  {r['ARI']:.3f}", flush=True)
print("\n[판정] P1 차이 ≤3, noise 차이 ≤3%p, Comp/Hom 차이 ≤0.03 이면 재현 성공.", flush=True)
print("[DONE_SCORE]", flush=True)
