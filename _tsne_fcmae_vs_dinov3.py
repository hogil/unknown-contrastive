#!/usr/bin/env python3
"""Frozen embedding t-SNE 비교: convnextv2 FCMAE(우리것) vs DINOv3 ConvNeXt-Base.

held-out novel wafers (wm811k_novel_disjoint_v1/novel_eval) 1500장을 두 backbone으로
frozen 임베딩 → 같은 transform → t-SNE 2D 나란히 + kNN top1/k5. CPU. 추가 학습 0.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "scripts"))
from _common import mask_palette_non_grade_to_white  # noqa: E402

EVAL_DIR = Path("E:/data/images/wm811k_novel_disjoint_v1/novel_eval")
FCMAE_NAME = "convnextv2_base.fcmae_ft_in22k_in1k_384"
FCMAE_WEIGHTS = REPO / "weights" / "convnextv2_base.fcmae_ft_in22k_in1k_384.pth"
DINOV3_NAME = "convnext_base.dinov3_lvd1689m"
IMG = 384
BATCH = 16
OUT = REPO / "result_grouping" / (time.strftime("%y%m%d_%H%M%S") + "_tsne_fcmae_vs_dinov3_novel")


def list_images(d: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    labels = [p.parent.name for p in paths]
    classes = sorted(set(labels))
    c2i = {c: i for i, c in enumerate(classes)}
    y = np.array([c2i[l] for l in labels], dtype=np.int64)
    return paths, labels, classes, y


def build_tfm():
    return T.Compose([
        T.Resize((IMG, IMG)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def embed(model, paths, tfm, dev):
    out = []
    for i in range(0, len(paths), BATCH):
        xs = []
        for p in paths[i:i + BATCH]:
            with Image.open(p) as im:
                img = mask_palette_non_grade_to_white(im).convert("RGB")
            xs.append(tfm(img))
        xb = torch.stack(xs).to(dev)
        z = F.normalize(model(xb), dim=1)
        out.append(z.float().cpu().numpy())
        print(f"    {min(i + BATCH, len(paths))}/{len(paths)}", flush=True)
    return np.concatenate(out, axis=0).astype(np.float32)


def knn_rates(emb, y, ks=(1, 5)):
    d = pairwise_distances(emb, metric="cosine")
    np.fill_diagonal(d, np.inf)
    r = {}
    for k in ks:
        idx = np.argpartition(d, k, axis=1)[:, :k]
        r[k] = float(np.mean([(y[idx[i]] == y[i]).mean() for i in range(len(y))]))
    return r


def load_fcmae(dev):
    import timm
    m = timm.create_model(FCMAE_NAME, pretrained=False, num_classes=0, global_pool="avg")
    sd = torch.load(FCMAE_WEIGHTS, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    msd = m.state_dict()
    compat = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
    m.load_state_dict(compat, strict=False)
    print(f"    FCMAE loaded {len(compat)} keys", flush=True)
    return m.to(dev).eval()


def load_dinov3(dev):
    import timm
    m = timm.create_model(DINOV3_NAME, pretrained=True, num_classes=0, global_pool="avg")
    return m.to(dev).eval()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu")
    paths, labels, classes, y = list_images(EVAL_DIR)
    print(f"[data] {len(paths)} images  classes={classes}", flush=True)
    tfm = build_tfm()

    results = {}
    for name, loader in [("FCMAE (ours)", load_fcmae), ("DINOv3", load_dinov3)]:
        print(f"[embed] {name}", flush=True)
        t0 = time.time()
        model = loader(dev)
        emb = embed(model, paths, tfm, dev)
        r = knn_rates(emb, y)
        z2 = TSNE(n_components=2, metric="cosine", init="pca",
                  perplexity=30, random_state=42).fit_transform(emb)
        results[name] = {"z2": z2, "knn": r}
        np.save(OUT / f"{name.split()[0]}_emb.npy", emb)
        print(f"    -> {name}: top1={r[1]:.4f}  k5={r[5]:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    cmap = plt.get_cmap("tab10").colors
    colors = {c: cmap[i % 10] for i, c in enumerate(classes)}
    labs = np.array(labels)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=150)
    for ax, (name, r) in zip(axes, results.items()):
        z2 = r["z2"]
        for c in classes:
            m = labs == c
            ax.scatter(z2[m, 0], z2[m, 1], s=14, c=[colors[c]], alpha=0.72,
                       edgecolors="black", linewidths=0.2)
        kn = r["knn"]
        ax.set_title(f"{name}\ntop1={kn[1] * 100:.1f}%   k5={kn[5] * 100:.1f}%", fontsize=13)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(alpha=0.2)
    legend = [Line2D([0], [0], marker="o", color="w", label=c,
                     markerfacecolor=colors[c], markersize=10) for c in classes]
    fig.legend(handles=legend, loc="upper center", ncol=len(classes), frameon=False, fontsize=11)
    fig.suptitle("Held-out NOVEL wafers (Donut / Edge-Loc / Random) - frozen embedding",
                 fontsize=14, y=1.03)
    fig.tight_layout()
    png = OUT / "tsne_fcmae_vs_dinov3.png"
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)

    print("\n=== SUMMARY ===", flush=True)
    for name, r in results.items():
        print(f"  {name:16s} top1={r['knn'][1] * 100:.1f}%  k5={r['knn'][5] * 100:.1f}%", flush=True)
    print(f"\n[OUT] {OUT}", flush=True)
    print(f"[PNG] {png}", flush=True)


if __name__ == "__main__":
    main()
