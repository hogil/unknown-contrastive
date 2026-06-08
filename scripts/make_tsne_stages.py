#!/usr/bin/env python3
"""Create stage-by-stage t-SNE panels for CNN/contrastive embeddings."""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from matplotlib.lines import Line2D
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances

sys.path.insert(0, str(Path(__file__).parent))
from _common import mask_palette_non_grade_to_white, resolve_path


BACKBONE = "convnextv2_base.fcmae_ft_in22k_in1k_384"
PROJ_DIM = 128


class ContrastiveInferModel(nn.Module):
    def __init__(self, backbone_name: str, proj_dim: int):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone_name, pretrained=False,
                                          num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.GELU(),
            nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x):
        f = self.backbone(x)
        z = self.proj(f)
        return F.normalize(z, dim=1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", required=True,
                   help="ImageFolder eval dir, e.g. data/images/wm811k_50/eval")
    p.add_argument("--cnn", required=True,
                   help="CNN best_model.pth")
    p.add_argument("--stage", action="append", default=[],
                   help="label=run_dir_or_best_model.pt. Repeatable.")
    p.add_argument("--out-root", default="result_grouping")
    p.add_argument("--tag", default="tsne_stages")
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--perplexity", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_")[:80] or "stage"


def find_model_path(path_str: str) -> Path:
    p = resolve_path(path_str)
    if p.is_dir():
        c = p / "contrastive" / "best_model.pt"
        if c.exists():
            return c
    if p.exists():
        return p
    raise SystemExit(f"stage model not found: {p}")


def parse_stage(s: str) -> tuple[str, Path]:
    if "=" not in s:
        p = find_model_path(s)
        return p.parent.parent.name if p.name == "best_model.pt" else p.stem, p
    label, path = s.split("=", 1)
    return label.strip(), find_model_path(path.strip())


def list_images(eval_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted(p for p in eval_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    if not paths:
        raise SystemExit(f"no images found under {eval_dir}")
    labels = [p.parent.name for p in paths]
    classes = sorted(set(labels))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[c] for c in labels], dtype=np.int64)
    return paths, labels, classes, y


def build_tfm(img_size: int):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_batch(paths, tfm):
    xs = []
    for p in paths:
        with Image.open(p) as im:
            im = mask_palette_non_grade_to_white(im).convert("RGB")
            xs.append(tfm(im))
    return torch.stack(xs, 0)


def load_cnn_model(ckpt: Path, device):
    import timm
    model = timm.create_model(BACKBONE, pretrained=False, num_classes=0, global_pool="avg")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    m_sd = model.state_dict()
    compat = {k: v for k, v in sd.items() if k in m_sd and m_sd[k].shape == v.shape}
    model.load_state_dict(compat, strict=False)
    return model.to(device).eval()


def load_contrastive_model(ckpt: Path, device):
    model = ContrastiveInferModel(BACKBONE, PROJ_DIM)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def extract(model, paths, tfm, device, batch: int):
    out = []
    for i in range(0, len(paths), batch):
        xb = load_batch(paths[i:i + batch], tfm).to(device)
        z = F.normalize(model(xb), dim=1)
        out.append(z.float().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def knn_same_rates(emb: np.ndarray, y: np.ndarray, ks=(1, 3, 5, 7, 9)):
    d = pairwise_distances(emb, metric="cosine")
    np.fill_diagonal(d, np.inf)
    rates = {}
    for k in ks:
        kk = min(k, len(y) - 1)
        idx = np.argpartition(d, kk, axis=1)[:, :kk]
        rates[str(k)] = float(np.mean([(y[idx[i]] == y[i]).mean() for i in range(len(y))]))
    return rates


def draw_single(z2, labels, classes, colors, title, out):
    fig, ax = plt.subplots(figsize=(8, 7), dpi=180)
    labels_np = np.array(labels)
    for c in classes:
        m = labels_np == c
        ax.scatter(z2[m, 0], z2[m, 1], s=46, c=[colors[c]],
                   alpha=0.88, edgecolors="black", linewidths=0.25)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def draw_sheet(stage_rows, labels, classes, colors, out):
    n = len(stage_rows)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.2, rows * 5.4), dpi=180)
    axes = np.atleast_1d(axes).reshape(-1)
    labels_np = np.array(labels)
    for ax, row in zip(axes, stage_rows):
        z2 = row["tsne"]
        for c in classes:
            m = labels_np == c
            ax.scatter(z2[m, 0], z2[m, 1], s=34, c=[colors[c]],
                       alpha=0.88, edgecolors="black", linewidths=0.22)
        ax.set_title(row["title"], fontsize=10)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, alpha=0.18)
    for ax in axes[n:]:
        ax.axis("off")
    legend = [Line2D([0], [0], marker="o", color="w", label=c,
                     markerfacecolor=colors[c], markersize=8) for c in classes]
    fig.legend(handles=legend, loc="center right", frameon=False)
    fig.suptitle("Stage-wise t-SNE by class (independent t-SNE per stage)", fontsize=15)
    fig.tight_layout(rect=[0, 0, 0.88, 0.95])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def draw_summary(diagnostics: dict, out: Path):
    stages = diagnostics["stages"]
    labels = [s["label"] for s in stages]
    top1 = [s["knn_same_rate"]["1"] * 100 for s in stages]
    k5 = [s["knn_same_rate"]["5"] * 100 for s in stages]
    x = np.arange(len(stages))

    fig = plt.figure(figsize=(max(10, len(stages) * 1.7), 7.2), dpi=180)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.15], hspace=0.38)
    ax = fig.add_subplot(gs[0, 0])
    width = 0.36
    b1 = ax.bar(x - width / 2, top1, width, label="top1 same-class", color="#2f78b7")
    b2 = ax.bar(x + width / 2, k5, width, label="k5 same-class", color="#e58b31")
    ax.set_ylim(0, 100)
    ax.set_ylabel("same-class neighbor rate (%)")
    ax.set_title("Embedding Separation Summary")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=22, ha="right")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, loc="upper right")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}",
                    ha="center", va="bottom", fontsize=8)

    table_ax = fig.add_subplot(gs[1, 0])
    table_ax.axis("off")
    cell_text = []
    best_top1 = max(top1) if top1 else 0
    best_k5 = max(k5) if k5 else 0
    for label, t1, kk in zip(labels, top1, k5):
        mark1 = "best" if abs(t1 - best_top1) < 1e-9 else ""
        mark5 = "best" if abs(kk - best_k5) < 1e-9 else ""
        cell_text.append([label, f"{t1:.1f}%", mark1, f"{kk:.1f}%", mark5])
    table = table_ax.table(
        cellText=cell_text,
        colLabels=["stage", "top1", "", "k5", ""],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")
        elif col == 0:
            cell.set_text_props(ha="left")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    eval_dir = resolve_path(args.eval_dir)
    cnn_ckpt = resolve_path(args.cnn)
    if not cnn_ckpt.exists():
        raise SystemExit(f"CNN checkpoint not found: {cnn_ckpt}")
    out_dir = resolve_path(args.out_root) / f"{datetime.now().strftime('%y%m%d_%H%M%S')}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths, labels, classes, y = list_images(eval_dir)
    tfm = build_tfm(args.img_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perplexity = args.perplexity or min(20, max(5, (len(paths) - 1) // 4))

    stage_specs: list[tuple[str, str, Path | None]] = [("CNN baseline", "cnn", cnn_ckpt)]
    for raw in args.stage:
        label, model_path = parse_stage(raw)
        stage_specs.append((label, "contrastive", model_path))

    cmap = plt.get_cmap("tab10").colors
    colors = {c: cmap[i % len(cmap)] for i, c in enumerate(classes)}
    stage_rows = []
    diagnostics = {
        "eval_dir": str(eval_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "n_images": len(paths),
        "classes": classes,
        "img_size": args.img_size,
        "perplexity": perplexity,
        "stages": [],
    }

    print(f"[data] {eval_dir.resolve()} n={len(paths)} classes={classes}", flush=True)
    print(f"[out] {out_dir.resolve()}", flush=True)
    for si, (label, kind, ckpt) in enumerate(stage_specs, 1):
        print(f"[stage {si}/{len(stage_specs)}] {label} <- {ckpt}", flush=True)
        model = load_cnn_model(ckpt, device) if kind == "cnn" else load_contrastive_model(ckpt, device)
        emb = extract(model, paths, tfm, device, args.batch)
        rates = knn_same_rates(emb, y)
        z2 = TSNE(n_components=2, perplexity=perplexity, init="pca",
                  learning_rate="auto", metric="cosine",
                  random_state=args.seed).fit_transform(emb)
        name = f"{si:02d}_{slug(label)}"
        np.save(out_dir / f"{name}_embedding.npy", emb)
        np.save(out_dir / f"{name}_tsne.npy", z2)
        title = f"{si}. {label}\ntop1={rates['1']*100:.1f}%  k5={rates['5']*100:.1f}%"
        draw_single(z2, labels, classes, colors, title, out_dir / f"{name}.png")
        stage_rows.append({"label": label, "title": title, "tsne": z2})
        diagnostics["stages"].append({
            "label": label,
            "kind": kind,
            "checkpoint": str(ckpt.resolve()),
            "embedding_dim": int(emb.shape[1]),
            "knn_same_rate": rates,
            "image": str((out_dir / f"{name}.png").resolve()),
        })

    draw_sheet(stage_rows, labels, classes, colors, out_dir / "tsne_stage_sheet.png")
    (out_dir / "paths.json").write_text(json.dumps([str(p.resolve()) for p in paths], indent=2),
                                        encoding="utf-8")
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False),
                                              encoding="utf-8")
    draw_summary(diagnostics, out_dir / "summary_metrics.png")
    print(f"[sheet] {(out_dir / 'tsne_stage_sheet.png').resolve()}", flush=True)
    print(f"[summary] {(out_dir / 'summary_metrics.png').resolve()}", flush=True)


if __name__ == "__main__":
    main()
