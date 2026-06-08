#!/usr/bin/env python3
"""Open-set embedding comparison with a shared vector dimension.

This is an eval-only script:
- labels are used only for metrics, not for training;
- Normal is excluded from metrics by default;
- raw FCMAE, CNN backbone, and contrastive checkpoints are compared after
  reducing all embeddings to the same PCA dimension.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    pairwise_distances,
    v_measure_score,
)

sys.path.insert(0, str(Path(__file__).parent))
from _common import ensure_backbone_weights, mask_palette_non_grade_to_white, resolve_path


BACKBONE = "convnextv2_base.fcmae_ft_in22k_in1k_384"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ContrastiveInferModel(nn.Module):
    def __init__(self, backbone_name: str, proj_dim: int, mode: str):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            backbone_name, pretrained=False, num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, proj_dim),
        )
        self.mode = str(mode).lower()

    def forward(self, x):
        f = F.normalize(self.backbone(x), dim=1)
        z = F.normalize(self.proj(f), dim=1)
        if self.mode == "projection":
            return z
        if self.mode == "backbone":
            return f
        raise ValueError(f"unsupported contrastive embed mode: {self.mode}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", required=True, help="ImageFolder eval dir.")
    p.add_argument("--cnn", default=None, help="CNN best_model.pth. Optional.")
    p.add_argument(
        "--contrastive",
        action="append",
        default=[],
        help="label=best_model.pt or label=run_dir. Repeatable.",
    )
    p.add_argument("--fcmae-weights", default=None, help="Raw FCMAE backbone .pth.")
    p.add_argument("--out-root", default="result_grouping")
    p.add_argument("--tag", default="open_set_embed_eval")
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--pca-dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ignore-class", action="append", default=["Normal"])
    p.add_argument(
        "--contrastive-embed-mode",
        default="projection",
        choices=["projection", "backbone"],
        help="Embedding exported from contrastive checkpoint.",
    )
    p.add_argument("--min-cluster-size", type=int, default=5)
    p.add_argument("--min-samples", type=int, default=2)
    p.add_argument(
        "--cluster-selection-method", default="eom", choices=["eom", "leaf"]
    )
    p.add_argument("--cluster-selection-epsilon", type=float, default=0.0)
    p.add_argument("--no-tsne", action="store_true")
    p.add_argument("--tsne-max", type=int, default=2000)
    return p.parse_args()


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")[:80] or "stage"


def strip_prefixes(sd: dict[str, Any], prefixes: tuple[str, ...]) -> dict[str, Any]:
    out = sd
    for pref in prefixes:
        if any(str(k).startswith(pref) for k in out):
            out = {
                (str(k)[len(pref):] if str(k).startswith(pref) else str(k)): v
                for k, v in out.items()
            }
    return out


def state_dict_from_checkpoint(
    path: Path,
    prefixes: tuple[str, ...] = ("module.", "model.", "backbone."),
) -> tuple[dict[str, Any], dict[str, Any]]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "state_dict" in ck:
        sd = ck["state_dict"]
    elif isinstance(ck, dict) and "model" in ck:
        sd = ck["model"]
    else:
        sd = ck
    if not isinstance(sd, dict):
        raise SystemExit(f"checkpoint has no state_dict: {path.resolve()}")
    return strip_prefixes(sd, prefixes), ck if isinstance(ck, dict) else {}


def list_images(eval_dir: Path, ignore_classes: set[str]):
    all_paths = sorted(
        p for p in eval_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not all_paths:
        raise SystemExit(f"no images found under {eval_dir.resolve()}")
    all_labels = [p.parent.name for p in all_paths]
    ignored_counts: dict[str, int] = {}
    kept = []
    for p, y in zip(all_paths, all_labels):
        if y in ignore_classes:
            ignored_counts[y] = ignored_counts.get(y, 0) + 1
        else:
            kept.append((p, y))
    if not kept:
        raise SystemExit(
            f"no metric images left after ignore_class={sorted(ignore_classes)}"
        )
    paths = [p for p, _ in kept]
    labels = [y for _, y in kept]
    classes = sorted(set(labels))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[c] for c in labels], dtype=np.int64)
    return all_paths, all_labels, paths, labels, classes, y, ignored_counts


def build_tfm(img_size: int):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_raw_fcmae(weights_path: Path, device):
    import timm

    model = timm.create_model(BACKBONE, pretrained=False, num_classes=0, global_pool="avg")
    sd, _ = state_dict_from_checkpoint(weights_path)
    m_sd = model.state_dict()
    compat = {k: v for k, v in sd.items() if k in m_sd and m_sd[k].shape == v.shape}
    load = model.load_state_dict(compat, strict=False)
    return model.to(device).eval(), {
        "source": str(weights_path.resolve()),
        "loaded_keys": len(compat),
        "missing_keys": len(load.missing_keys),
        "unexpected_keys": len(load.unexpected_keys),
        "classes": [],
    }


def load_cnn_backbone(ckpt: Path, device):
    import timm

    model = timm.create_model(BACKBONE, pretrained=False, num_classes=0, global_pool="avg")
    sd, meta = state_dict_from_checkpoint(ckpt)
    m_sd = model.state_dict()
    compat = {k: v for k, v in sd.items() if k in m_sd and m_sd[k].shape == v.shape}
    load = model.load_state_dict(compat, strict=False)
    classes = list(meta.get("classes") or [])
    return model.to(device).eval(), {
        "source": str(ckpt.resolve()),
        "loaded_keys": len(compat),
        "missing_keys": len(load.missing_keys),
        "unexpected_keys": len(load.unexpected_keys),
        "classes": classes,
    }


def infer_proj_dim(sd: dict[str, Any], default: int = 1024) -> int:
    for key in ("proj.2.weight", "module.proj.2.weight"):
        v = sd.get(key)
        if hasattr(v, "shape") and len(v.shape) >= 2:
            return int(v.shape[0])
    for k, v in sd.items():
        if str(k).endswith("proj.2.weight") and hasattr(v, "shape") and len(v.shape) >= 2:
            return int(v.shape[0])
    return int(default)


def find_contrastive_path(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        label = label.strip()
    else:
        label, path = "", raw
    p = resolve_path(path.strip())
    if p.is_dir():
        for cand in (p / "contrastive" / "best_model.pt", p / "best_model.pt"):
            if cand.exists():
                p = cand
                break
    if not p.exists():
        raise SystemExit(f"contrastive checkpoint not found: {p.resolve()}")
    if not label:
        label = p.parent.parent.name if p.name == "best_model.pt" else p.stem
    return label, p


def load_contrastive(ckpt: Path, device, mode: str):
    sd, meta = state_dict_from_checkpoint(ckpt, prefixes=("module.", "model."))
    proj_dim = infer_proj_dim(sd)
    model = ContrastiveInferModel(BACKBONE, proj_dim, mode=mode)
    load = model.load_state_dict(sd, strict=False)
    bad_missing = [k for k in load.missing_keys if not k.startswith("head.")]
    if bad_missing or load.unexpected_keys:
        raise SystemExit(
            f"contrastive checkpoint incompatible: {ckpt.resolve()}\n"
            f"missing={bad_missing[:10]} unexpected={load.unexpected_keys[:10]}"
        )
    return model.to(device).eval(), {
        "source": str(ckpt.resolve()),
        "proj_dim": proj_dim,
        "mode": mode,
        "loaded_epoch": meta.get("epoch"),
        "backbone_source": str(meta.get("backbone_source", "")),
        "classes": [],
    }


@torch.no_grad()
def extract_embeddings(model, paths, labels, tfm, device, batch: int, label: str):
    embs = []
    good_paths = []
    good_labels = []
    bad = []
    total = len(paths)
    for start in range(0, total, batch):
        chunk = paths[start:start + batch]
        xs = []
        chunk_paths = []
        chunk_labels = []
        for p, y in zip(chunk, labels[start:start + batch]):
            try:
                with Image.open(p) as im:
                    img = mask_palette_non_grade_to_white(im).convert("RGB")
                xs.append(tfm(img))
                chunk_paths.append(p)
                chunk_labels.append(y)
            except Exception as e:
                bad.append({"path": str(p.resolve()), "error": f"{type(e).__name__}: {e}"})
        if xs:
            xb = torch.stack(xs, 0).to(device, non_blocking=True)
            z = F.normalize(model(xb), dim=1)
            embs.append(z.float().cpu().numpy())
            good_paths.extend(chunk_paths)
            good_labels.extend(chunk_labels)
        done = min(start + batch, total)
        print(f"[embed:{label}] {done}/{total}", flush=True)
    if not embs:
        raise SystemExit(f"no valid images embedded for {label}")
    return np.concatenate(embs, axis=0).astype(np.float32), good_paths, good_labels, bad


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def to_same_dim(x: np.ndarray, target_dim: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    x0 = l2_normalize(x.astype(np.float32, copy=False))
    info: dict[str, Any] = {
        "input_dim": int(x0.shape[1]),
        "target_dim": int(target_dim),
        "method": "identity",
        "explained_variance_ratio_sum": None,
    }
    if x0.shape[1] > target_dim:
        pca = PCA(n_components=target_dim, svd_solver="randomized", random_state=seed)
        z = pca.fit_transform(x0).astype(np.float32, copy=False)
        info["method"] = "pca"
        info["explained_variance_ratio_sum"] = float(pca.explained_variance_ratio_.sum())
    elif x0.shape[1] == target_dim:
        z = x0
    else:
        z = x0
        info["method"] = "identity_dim_below_target"
    return l2_normalize(z), info


def knn_same_rates(emb: np.ndarray, y: np.ndarray, ks=(1, 3, 5, 7, 9)):
    d = pairwise_distances(emb, metric="cosine")
    np.fill_diagonal(d, np.inf)
    rates = {}
    for k in ks:
        kk = min(k, len(y) - 1)
        idx = np.argpartition(d, kk, axis=1)[:, :kk]
        rates[str(k)] = float(np.mean([(y[idx[i]] == y[i]).mean() for i in range(len(y))]))
    return rates


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total = len(y_true)
    if total == 0:
        return 0.0
    score = 0
    for c in sorted(set(int(x) for x in y_pred)):
        idx = np.where(y_pred == c)[0]
        vals, counts = np.unique(y_true[idx], return_counts=True)
        score += int(counts.max()) if len(vals) else 0
    return float(score / total)


def hdbscan_metrics(emb: np.ndarray, y: np.ndarray, args) -> dict[str, Any]:
    try:
        import hdbscan
    except Exception as e:
        return {"skipped": True, "reason": f"hdbscan import failed: {type(e).__name__}: {e}"}
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        cluster_selection_method=args.cluster_selection_method,
        cluster_selection_epsilon=args.cluster_selection_epsilon,
        allow_single_cluster=False,
    )
    pred = clusterer.fit_predict(emb)
    noise = int(np.sum(pred == -1))
    clusters = sorted(int(x) for x in np.unique(pred) if int(x) >= 0)
    counts = {str(c): int(np.sum(pred == c)) for c in clusters}
    return {
        "skipped": False,
        "params": {
            "min_cluster_size": args.min_cluster_size,
            "min_samples": args.min_samples,
            "metric": "euclidean",
            "cluster_selection_method": args.cluster_selection_method,
            "cluster_selection_epsilon": args.cluster_selection_epsilon,
            "allow_single_cluster": False,
        },
        "n_clusters": len(clusters),
        "noise": noise,
        "noise_pct": float(noise / max(1, len(pred))),
        "cluster_counts": counts,
        "ari": float(adjusted_rand_score(y, pred)),
        "ami": float(adjusted_mutual_info_score(y, pred)),
        "homogeneity": float(homogeneity_score(y, pred)),
        "completeness": float(completeness_score(y, pred)),
        "v_measure": float(v_measure_score(y, pred)),
        "purity": purity_score(y, pred),
    }


def draw_tsne_sheet(stage_rows, labels, classes, out: Path, seed: int):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from sklearn.manifold import TSNE

    n = len(stage_rows)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.2, rows * 5.4), dpi=180)
    axes = np.atleast_1d(axes).reshape(-1)
    labels_np = np.array(labels)
    cmap = plt.get_cmap("tab20").colors
    colors = {c: cmap[i % len(cmap)] for i, c in enumerate(classes)}
    perplexity = min(30, max(5, (len(labels) - 1) // 4))

    for ax, row in zip(axes, stage_rows):
        emb = row["embedding"]
        z2 = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            metric="cosine",
            random_state=seed,
        ).fit_transform(emb)
        for c in classes:
            m = labels_np == c
            ax.scatter(
                z2[m, 0], z2[m, 1], s=34, c=[colors[c]],
                alpha=0.88, edgecolors="black", linewidths=0.22
            )
        r = row["metrics"]["knn_same_rate"]
        ax.set_title(f"{row['label']}\ntop1={r['1']*100:.1f}% k5={r['5']*100:.1f}%", fontsize=10)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, alpha=0.18)
    for ax in axes[n:]:
        ax.axis("off")
    legend = [
        Line2D([0], [0], marker="o", color="w", label=c,
               markerfacecolor=colors[c], markersize=7)
        for c in classes
    ]
    fig.legend(handles=legend, loc="center right", frameon=False)
    fig.suptitle("Open-set held-out embedding comparison (same PCA dimension)", fontsize=15)
    fig.tight_layout(rect=[0, 0, 0.86, 0.95])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def write_summary(out_dir: Path, diagnostics: dict[str, Any]):
    lines = [
        "# Open-Set Embedding Evaluation",
        "",
        f"- output: `{out_dir.resolve()}`",
        f"- eval_dir: `{diagnostics['eval_dir']}`",
        f"- measured_images: {diagnostics['n_measured_images']}",
        f"- measured_classes: {diagnostics['n_measured_classes']}",
        f"- ignored_classes: {diagnostics['ignore_classes']}",
        f"- same_vector_dim: {diagnostics['same_vector_dim']}",
        "",
        "| stage | input dim | same dim | top1 | k3 | k5 | k7 | k9 | HDBSCAN clusters | noise | ARI | AMI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in diagnostics["stages"]:
        r = s["knn_same_rate"]
        h = s["hdbscan"]
        lines.append(
            f"| {s['label']} | {s['pca']['input_dim']} | {s['embedding_dim']} | "
            f"{r['1']:.4f} | {r['3']:.4f} | {r['5']:.4f} | {r['7']:.4f} | {r['9']:.4f} | "
            f"{h.get('n_clusters', '-')} | {h.get('noise_pct', 0.0):.4f} | "
            f"{h.get('ari', 0.0):.4f} | {h.get('ami', 0.0):.4f} |"
        )
    lines.extend([
        "",
        "## Held-Out Check",
        "",
        f"- cnn_checkpoint_classes: {diagnostics.get('cnn_checkpoint_classes', [])}",
        f"- measured_eval_classes: {diagnostics['classes']}",
        f"- overlap_with_cnn_classes: {diagnostics.get('overlap_with_cnn_classes', [])}",
        "",
    ])
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    eval_dir = resolve_path(args.eval_dir)
    out_dir = resolve_path(args.out_root) / f"{datetime.now().strftime('%y%m%d_%H%M%S')}_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_dir = out_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    ignore_classes = set(args.ignore_class or [])
    all_paths, all_labels, paths, labels, classes, y, ignored_counts = list_images(
        eval_dir, ignore_classes
    )
    same_dim = min(int(args.pca_dim), max(1, len(paths) - 1))
    tfm = build_tfm(args.img_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.fcmae_weights:
        fcmae_weights = resolve_path(args.fcmae_weights)
    else:
        fcmae_weights = ensure_backbone_weights("weights", BACKBONE)

    stages: list[tuple[str, str, Path | None]] = [("Raw FCMAE", "raw_fcmae", fcmae_weights)]
    if args.cnn:
        stages.append(("CNN backbone", "cnn", resolve_path(args.cnn)))
    for raw in args.contrastive:
        label, ckpt = find_contrastive_path(raw)
        stages.append((label, "contrastive", ckpt))

    diagnostics: dict[str, Any] = {
        "out_dir": str(out_dir.resolve()),
        "eval_dir": str(eval_dir.resolve()),
        "device": str(device),
        "img_size": args.img_size,
        "batch": args.batch,
        "pca_dim_requested": args.pca_dim,
        "same_vector_dim": same_dim,
        "ignore_classes": sorted(ignore_classes),
        "ignored_counts": ignored_counts,
        "n_total_images": len(all_paths),
        "n_measured_images": len(paths),
        "n_measured_classes": len(classes),
        "classes": classes,
        "class_counts": {c: int(sum(1 for yy in labels if yy == c)) for c in classes},
        "stages": [],
    }

    print(f"[eval_dir] {eval_dir.resolve()}", flush=True)
    print(f"[out_dir] {out_dir.resolve()}", flush=True)
    print(f"[classes] measured={len(classes)} images={len(paths)} ignored={ignored_counts}", flush=True)
    print(f"[same_dim] {same_dim}", flush=True)

    stage_rows = []
    reference_paths = None
    reference_labels = None
    cnn_classes: list[str] = []

    for idx, (label, kind, path) in enumerate(stages, 1):
        print(f"[stage {idx}/{len(stages)}] {label} <- {path.resolve() if path else ''}", flush=True)
        if kind == "raw_fcmae":
            model, meta = load_raw_fcmae(path, device)
        elif kind == "cnn":
            if path is None or not path.exists():
                raise SystemExit(f"CNN checkpoint not found: {path.resolve() if path else path}")
            model, meta = load_cnn_backbone(path, device)
            cnn_classes = list(meta.get("classes") or [])
        elif kind == "contrastive":
            model, meta = load_contrastive(path, device, args.contrastive_embed_mode)
        else:
            raise AssertionError(kind)

        raw_emb, good_paths, good_labels, bad = extract_embeddings(
            model, paths, labels, tfm, device, args.batch, slug(label)
        )
        if reference_paths is None:
            reference_paths = good_paths
            reference_labels = good_labels
            y_good = np.array([{c: i for i, c in enumerate(classes)}[c] for c in good_labels], dtype=np.int64)
        else:
            if [str(p) for p in good_paths] != [str(p) for p in reference_paths]:
                raise SystemExit("valid image set differs between stages; fix corrupt inputs first")
            y_good = np.array([{c: i for i, c in enumerate(classes)}[c] for c in good_labels], dtype=np.int64)

        emb, pca_info = to_same_dim(raw_emb, same_dim, args.seed)
        rates = knn_same_rates(emb, y_good)
        hdb = hdbscan_metrics(emb, y_good, args)
        name = f"{idx:02d}_{slug(label)}"
        np.save(emb_dir / f"{name}_raw.npy", raw_emb)
        np.save(emb_dir / f"{name}_same_dim.npy", emb)
        stage_diag = {
            "label": label,
            "kind": kind,
            "checkpoint": str(path.resolve()) if path else None,
            "meta": meta,
            "embedding_dim": int(emb.shape[1]),
            "pca": pca_info,
            "knn_same_rate": rates,
            "hdbscan": hdb,
            "corrupt_skipped": bad,
            "raw_embedding_file": str((emb_dir / f"{name}_raw.npy").resolve()),
            "same_dim_embedding_file": str((emb_dir / f"{name}_same_dim.npy").resolve()),
        }
        diagnostics["stages"].append(stage_diag)
        stage_rows.append({"label": label, "embedding": emb, "metrics": stage_diag})
        print(
            f"[metric:{label}] top1={rates['1']:.4f} k5={rates['5']:.4f} "
            f"clusters={hdb.get('n_clusters', '-')} noise={hdb.get('noise_pct', 0.0):.4f}",
            flush=True,
        )

    diagnostics["cnn_checkpoint_classes"] = cnn_classes
    diagnostics["overlap_with_cnn_classes"] = sorted(set(cnn_classes) & set(classes))
    if reference_paths is not None:
        (out_dir / "paths.json").write_text(
            json.dumps([str(p.resolve()) for p in reference_paths], indent=2),
            encoding="utf-8",
        )
    if reference_labels is not None:
        (out_dir / "labels.json").write_text(
            json.dumps(reference_labels, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    if not args.no_tsne and len(labels) <= args.tsne_max:
        tsne_path = out_dir / "tsne_same_dim_sheet.png"
        draw_tsne_sheet(stage_rows, reference_labels or labels, classes, tsne_path, args.seed)
        diagnostics["tsne_sheet"] = str(tsne_path.resolve())
        print(f"[tsne] {tsne_path.resolve()}", flush=True)

    (out_dir / "metrics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_summary(out_dir, diagnostics)
    print(f"[metrics] {(out_dir / 'metrics.json').resolve()}", flush=True)
    print(f"[summary] {(out_dir / 'summary.md').resolve()}", flush=True)


if __name__ == "__main__":
    main()
