#!/usr/bin/env python3
"""Build the SimCLR component-ablation report with fixed baselines.

This post-processes the current component-ablation CSV and prepends raw
embedding baselines measured on the same WM-811K v1 novel eval folder.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import hdbscan
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    pairwise_distances,
)


REPO = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO / "data" / "images" / "wm811k_novel_disjoint_v1" / "novel_eval"
TRAIN_DIR = REPO / "data" / "images" / "wm811k_novel_disjoint_v1" / "cnn_seen_train"
DOC_DIR = REPO / "docs" / "contrastive-eval"
IN_CSV = DOC_DIR / "SIMCLR_COMPONENT_ABLATION.csv"
OUT_CSV = DOC_DIR / "SIMCLR_COMPONENT_ABLATION_WITH_BASELINES.csv"
OUT_MD = DOC_DIR / "SIMCLR_COMPONENT_ABLATION_WITH_BASELINES.md"
K = 3


BASELINES = [
    {
        "stage": "B0",
        "method": "Raw FCMAE baseline",
        "embedding_path": REPO / "result_grouping" / "260609_142348_wm811k_novel_v1_raw_vs_Acnn" / "embeddings" / "01_Raw_FCMAE_raw.npy",
        "run_dir": REPO / "result_grouping" / "260609_142348_wm811k_novel_v1_raw_vs_Acnn",
        "model_path": REPO / "weights" / "convnextv2_base.fcmae_ft_in22k_in1k_384.pth",
    },
    {
        "stage": "B1",
        "method": "Raw CNN A-supervised baseline",
        "embedding_path": REPO / "result_grouping" / "260609_142348_wm811k_novel_v1_raw_vs_Acnn" / "embeddings" / "02_CNN_backbone_raw.npy",
        "run_dir": REPO / "result_grouping" / "260609_142348_wm811k_novel_v1_raw_vs_Acnn",
        "model_path": REPO / "runs" / "260609_141113_cnn_ddp" / "cnn" / "best_model.pth",
    },
    {
        "stage": "B2",
        "method": "Raw DINOv3 baseline",
        "embedding_path": REPO / "result_grouping" / "_dinov3_ncd_autoloop" / "embeddings" / "dinov3_convnext_base.npy",
        "run_dir": REPO / "result_grouping" / "_dinov3_ncd_autoloop",
        "model_path": "hf_hub:timm/convnext_base.dinov3_lvd1689m",
    },
]

P_COLS = [
    "p1_class_capture_rate",
    "p2_noise_pct",
    "p3_completeness",
    "p4_homogeneity",
]


def p_metrics_from_tier1(tier1: dict[str, Any]) -> dict[str, Any]:
    return {
        "p1_class_capture_rate": tier1.get("class_capture_rate", ""),
        "p2_noise_pct": tier1.get("noise_pct", ""),
        "p3_completeness": tier1.get("completeness", ""),
        "p4_homogeneity": tier1.get("homogeneity", ""),
    }


def backfill_p_metrics(row: dict[str, Any]) -> dict[str, Any]:
    if all(row.get(k, "") not in ("", None) for k in P_COLS):
        return row
    run_dir = row.get("run_dir")
    if run_dir:
        tier_path = Path(str(run_dir)) / "contrastive" / "tier1.json"
        if tier_path.exists():
            try:
                row.update(p_metrics_from_tier1(json.loads(tier_path.read_text(encoding="utf-8"))))
                return row
            except Exception:
                pass
    if row.get("p2_noise_pct", "") in ("", None) and row.get("hdbscan_noise_pct", "") not in ("", None):
        row["p2_noise_pct"] = row.get("hdbscan_noise_pct")
    return row


def list_labels(eval_dir: Path) -> tuple[np.ndarray, list[str]]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted(p for p in eval_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    labels = [p.parent.name for p in paths]
    classes = sorted(set(labels))
    c2i = {c: i for i, c in enumerate(classes)}
    return np.array([c2i[x] for x in labels], dtype=np.int64), classes


def l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def knn_same_rates(emb: np.ndarray, y: np.ndarray) -> dict[str, float]:
    d = pairwise_distances(emb, metric="cosine")
    np.fill_diagonal(d, np.inf)
    out = {}
    for k in (1, 3, 5, 7, 9):
        idx = np.argpartition(d, k, axis=1)[:, :k]
        out[str(k)] = float(np.mean([(y[idx[i]] == y[i]).mean() for i in range(len(y))]))
    return out


def distance_ratio(emb: np.ndarray, y: np.ndarray) -> float:
    d = pairwise_distances(emb, metric="cosine").astype(np.float32, copy=False)
    intra_vals, nearest_other_vals = [], []
    for c in sorted(int(v) for v in np.unique(y)):
        idx = np.where(y == c)[0]
        other = np.where(y != c)[0]
        if len(idx) <= 1 or len(other) == 0:
            continue
        intra = d[np.ix_(idx, idx)]
        tri = intra[np.triu_indices(len(idx), k=1)]
        intra_vals.append(float(np.mean(tri)) if len(tri) else 0.0)
        nearest_other_vals.append(float(np.mean(np.min(d[np.ix_(idx, other)], axis=1))))
    return float(np.mean(nearest_other_vals) / max(np.mean(intra_vals), 1e-12))


def kmeans_metrics(emb: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = KMeans(n_clusters=K, n_init=10, random_state=42).fit_predict(emb)
    return {
        "kmeans_ari": float(adjusted_rand_score(y, pred)),
        "kmeans_nmi": float(normalized_mutual_info_score(y, pred)),
        "kmeans_ami": float(adjusted_mutual_info_score(y, pred)),
    }


def hdbscan_metrics(emb: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    pred = hdbscan.HDBSCAN(
        min_cluster_size=12,
        min_samples=15,
        cluster_selection_method="leaf",
        cluster_selection_epsilon=0.06,
        metric="euclidean",
    ).fit_predict(emb)
    keep = pred != -1
    noise_pct = float((~keep).mean() * 100.0)
    if keep.sum() == 0:
        return {
            "hdbscan_ari": 0.0,
            "hdbscan_ami": 0.0,
            "hdbscan_noise_pct": noise_pct,
            "hdbscan_clusters": 0,
            "p1_class_capture_rate": 0.0,
            "p2_noise_pct": noise_pct,
            "p3_completeness": 0.0,
            "p4_homogeneity": 0.0,
        }
    true_k = y[keep]
    pred_k = pred[keep]
    can_score = len(set(true_k.tolist())) > 1 and len(set(pred_k.tolist())) > 1
    cluster_cls = defaultdict(Counter)
    for p, c in zip(pred_k, true_k):
        cluster_cls[int(p)][int(c)] += 1
    cls_total = Counter(int(c) for c in y.tolist())
    capture = {}
    for cls, total in cls_total.items():
        mx = max(
            (cnt for ccnt in cluster_cls.values() for c, cnt in ccnt.items() if c == cls),
            default=0,
        )
        capture[cls] = mx / max(1, total)
    return {
        "hdbscan_ari": float(adjusted_rand_score(true_k, pred_k)) if can_score else 0.0,
        "hdbscan_ami": float(adjusted_mutual_info_score(true_k, pred_k)) if can_score else 0.0,
        "hdbscan_noise_pct": noise_pct,
        "hdbscan_clusters": int(len(set(pred_k.tolist()))),
        "p1_class_capture_rate": float(np.mean(list(capture.values()))) if capture else 0.0,
        "p2_noise_pct": noise_pct,
        "p3_completeness": float(completeness_score(true_k, pred_k)) if can_score else 0.0,
        "p4_homogeneity": float(homogeneity_score(true_k, pred_k)) if can_score else 0.0,
    }


def baseline_row(spec: dict[str, Any], y: np.ndarray) -> dict[str, Any]:
    emb_path = Path(spec["embedding_path"])
    if not emb_path.exists():
        raise SystemExit(f"baseline embedding not found: {emb_path.resolve()}")
    emb = l2(np.load(emb_path).astype(np.float32, copy=False))
    km = kmeans_metrics(emb, y)
    kn = knn_same_rates(emb, y)
    hdb = hdbscan_metrics(emb, y)
    return {
        "stage": spec["stage"],
        "method": spec["method"],
        **km,
        "top1": kn["1"],
        "k3": kn["3"],
        "k5": kn["5"],
        "k7": kn["7"],
        "k9": kn["9"],
        "dist_ratio": distance_ratio(emb, y),
        **hdb,
        "loss": "",
        "nce": "",
        "local": "",
        "neco": "",
        "run_dir": str(Path(spec["run_dir"]).resolve()),
        "model_path": str(spec["model_path"] if isinstance(spec["model_path"], str) else Path(spec["model_path"]).resolve()),
        "embedding_path": str(emb_path.resolve()),
        "log_path": "",
    }


def parse_float(value: Any) -> Any:
    if value is None or value == "":
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def read_component_rows() -> list[dict[str, Any]]:
    if not IN_CSV.exists():
        return []
    rows = []
    with IN_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(backfill_p_metrics({k: parse_float(v) for k, v in row.items()}))
    return rows


def write_outputs(rows: list[dict[str, Any]], classes: list[str]) -> None:
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    cols = [
        "stage", "method", "kmeans_ari", "kmeans_nmi", "kmeans_ami",
        *P_COLS,
        "top1", "k3", "k5", "k7", "k9", "dist_ratio",
        "hdbscan_ari", "hdbscan_ami", "hdbscan_noise_pct", "hdbscan_clusters",
        "loss", "nce", "local", "neco", "run_dir", "model_path",
        "embedding_path", "log_path",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in cols})

    with OUT_MD.open("w", encoding="utf-8") as f:
        f.write("# SimCLR Component Ablation With Baselines\n\n")
        f.write(f"- train folder: `{TRAIN_DIR.resolve()}`\n")
        f.write(f"- eval folder: `{EVAL_DIR.resolve()}`\n")
        f.write(f"- eval classes: `{', '.join(classes)}`\n")
        f.write("- data: WM-811K class-disjoint v1, not generated synthetic wafer\n")
        f.write("- eval embedding: full 1024-dim backbone feature, L2-normalized\n")
        f.write("- primary metric: k-means(k=3) ARI on held-out novel classes\n")
        f.write("- fixed HDBSCAN: min_cluster_size=12, min_samples=15, leaf, epsilon=0.06\n\n")
        f.write("| Stage | Method | ARI | NMI | AMI | P1 capture | P2 noise | P3 comp | P4 homog | top1 | k5 | k9 | dist ratio | HDB ARI |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            row = backfill_p_metrics(row)
            f.write(
                f"| {row['stage']} | {row['method']} | {float(row.get('kmeans_ari') or 0):.4f} | "
                f"{float(row.get('kmeans_nmi') or 0):.4f} | {float(row.get('kmeans_ami') or 0):.4f} | "
                f"{float(row.get('p1_class_capture_rate') or 0):.4f} | "
                f"{float(row.get('p2_noise_pct') or 0):.2f}% | "
                f"{float(row.get('p3_completeness') or 0):.4f} | "
                f"{float(row.get('p4_homogeneity') or 0):.4f} | "
                f"{float(row.get('top1') or 0) * 100:.2f}% | {float(row.get('k5') or 0) * 100:.2f}% | "
                f"{float(row.get('k9') or 0) * 100:.2f}% | {float(row.get('dist_ratio') or 0):.4f} | "
                f"{float(row.get('hdbscan_ari') or 0):.4f} |\n"
            )
        f.write("\n## Artifacts\n\n")
        for row in rows:
            f.write(f"- {row['stage']} {row['method']}\n")
            f.write(f"  - run: `{row.get('run_dir', '')}`\n")
            f.write(f"  - model: `{row.get('model_path', '')}`\n")
            f.write(f"  - embedding: `{row.get('embedding_path', '')}`\n")
            if row.get("log_path"):
                f.write(f"  - log: `{row.get('log_path')}`\n")


def main() -> None:
    y, classes = list_labels(EVAL_DIR)
    baseline_rows = [baseline_row(spec, y) for spec in BASELINES]
    component_rows = read_component_rows()
    write_outputs(baseline_rows + component_rows, classes)
    print(f"[OUT] {OUT_MD.resolve()}")
    print(f"[OUT] {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
