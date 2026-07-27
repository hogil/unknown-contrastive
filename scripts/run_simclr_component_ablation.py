#!/usr/bin/env python3
"""Run SimCLR component ablations on the WM-811K novel split.

This is intentionally a thin driver around train_contrastive_ddp.py. It keeps
the data, backbone, epoch count, and eval fixed, then changes only one SSL
component at a time:

  base SimCLR -> queue -> neg-ignore -> local grid -> NeCo -> all

The script does not set CUDA_VISIBLE_DEVICES. If a scheduler has already
assigned devices, that environment is preserved.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
)

from build_simclr_component_report import (
    P_COLS,
    filter_eval_embedding,
    hdbscan_metrics as report_hdbscan_metrics,
    kmeans_metrics as report_kmeans_metrics,
)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


REPO = Path(__file__).resolve().parents[1]
TRAIN_DIR = Path("E:/data/images/wm811k_novel_disjoint_v1/cnn_seen_train")
EVAL_DIR = Path("E:/data/images/wm811k_novel_disjoint_v1/novel_eval")
DOC_DIR = REPO / "docs" / "contrastive-eval"
OUT_CSV = DOC_DIR / "SIMCLR_COMPONENT_ABLATION.csv"
OUT_MD = DOC_DIR / "SIMCLR_COMPONENT_ABLATION.md"
LOG_DIR = REPO / "result_grouping" / "_simclr_component_ablation_logs"
K = 3


def _append(out: list[str], *items: Any) -> list[str]:
    out.extend(str(x) for x in items)
    return out


CONDITIONS: list[dict[str, Any]] = [
    {
        "stage": "C0",
        "name": "base_simclr",
        "method": "Base SimCLR",
        "args": ["--no-queue", "--no-pseudo-neg-remove", "--ignore-neg-sim", "1.01"],
    },
    {
        "stage": "C1",
        "name": "queue",
        "method": "+ MoCo EMA queue",
        "args": ["--queue-size", "1024", "--no-pseudo-neg-remove", "--ignore-neg-sim", "1.01"],
    },
    {
        "stage": "C2",
        "name": "queue_neg_ignore",
        "method": "+ neg-ignore",
        "args": ["--queue-size", "1024", "--ignore-neg-sim", "0.70"],
    },
    {
        "stage": "C3",
        "name": "queue_neg_ignore_local",
        "method": "+ local grid",
        "args": [
            "--queue-size", "1024", "--ignore-neg-sim", "0.70",
            "--local-weight", "0.2", "--local-grid", "6",
            "--local-window", "1", "--local-tau", "0.1",
        ],
    },
    {
        "stage": "C4",
        "name": "queue_neg_ignore_neco",
        "method": "+ NeCo",
        "args": [
            "--queue-size", "1024", "--ignore-neg-sim", "0.70",
            "--neco-weight", "0.2", "--neco-grid", "6", "--neco-tau", "0.1",
        ],
    },
    {
        "stage": "C5",
        "name": "all_queue_neg_ignore_local_neco",
        "method": "+ local grid + NeCo",
        "args": [
            "--queue-size", "1024", "--ignore-neg-sim", "0.70",
            "--local-weight", "0.2", "--local-grid", "6",
            "--local-window", "1", "--local-tau", "0.1",
            "--neco-weight", "0.2", "--neco-grid", "6", "--neco-tau", "0.1",
        ],
    },
]


def list_labels(eval_dir: Path) -> tuple[np.ndarray, list[str]]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted(p for p in eval_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)
    labels = [p.parent.name for p in paths]
    classes = sorted(set(labels))
    c2i = {c: i for i, c in enumerate(classes)}
    return np.array([c2i[x] for x in labels], dtype=np.int64), classes


def l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def knn_same_rates(emb: np.ndarray, y: np.ndarray, ks=(1, 3, 5, 7, 9)) -> dict[str, float]:
    from sklearn.metrics import pairwise_distances

    d = pairwise_distances(emb, metric="cosine")
    np.fill_diagonal(d, np.inf)
    out = {}
    for k in ks:
        kk = min(k, len(y) - 1)
        idx = np.argpartition(d, kk, axis=1)[:, :kk]
        out[str(k)] = float(np.mean([(y[idx[i]] == y[i]).mean() for i in range(len(y))]))
    return out


def distance_ratio(emb: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import pairwise_distances

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
        "ari": float(adjusted_rand_score(y, pred)),
        "nmi": float(normalized_mutual_info_score(y, pred)),
        "ami": float(adjusted_mutual_info_score(y, pred)),
        "completeness": float(completeness_score(y, pred)),
        "homogeneity": float(homogeneity_score(y, pred)),
    }


def parse_out_dir(log_text: str) -> Path | None:
    matches = re.findall(r"\[OUT\]\s+(.+)", log_text)
    if not matches:
        return None
    return Path(matches[-1].strip().strip('"'))


def run_command(cmd: list[str], log_path: Path) -> tuple[int, str]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=os.environ.copy(),
        )
        assert proc.stdout is not None
        chunks = []
        for line in proc.stdout:
            print(line, end="", flush=True)
            f.write(line)
            chunks.append(line)
        rc = proc.wait()
    return rc, "".join(chunks)


def train_condition(cond: dict[str, Any], epochs: int, batch: int, img_size: int,
                    lr_backbone: float, lr_head: float) -> dict[str, Any]:
    tag = f"simclr_component_{cond['stage'].lower()}_{cond['name']}"
    run_epochs = cond.get("epochs", epochs)
    run_img_size = cond.get("img_size", img_size)
    run_lr_backbone = cond.get("lr_backbone", lr_backbone)
    run_lr_head = cond.get("lr_head", lr_head)
    cmd = [
        sys.executable, "-u", str(REPO / "scripts" / "train_contrastive_ddp.py"),
        "--tag", tag,
        "--backbone-name", "hf_hub:timm/convnext_base.dinov3_lvd1689m",
        "--train-dirs", str(TRAIN_DIR),
        "--eval-dirs", str(EVAL_DIR),
        "--epochs", str(run_epochs),
        "--batch", str(batch),
        "--img-size", str(run_img_size),
        "--proj-dim", "256",
        "--backbone-train-mode", "all",
        "--lr-backbone", str(run_lr_backbone),
        "--lr-head", str(run_lr_head),
        "--nce-temp", str(cond.get("nce_temp", "0.05")),
        "--loss-mode", "nce",
        "--infer-embed-mode", "backbone",
        "--neco-weight", "0",
        "--local-weight", "0",
        "--num-workers", "0",
    ]
    _append(cmd, *cond["args"])

    log_path = LOG_DIR / f"{cond['stage']}_{cond['name']}.log"
    t0 = time.time()
    rc, log_text = run_command(cmd, log_path)
    run_dir = parse_out_dir(log_text)
    if rc != 0:
        raise RuntimeError(f"{cond['stage']} {cond['name']} failed rc={rc}; log={log_path.resolve()}")
    if run_dir is None:
        raise RuntimeError(f"{cond['stage']} {cond['name']} produced no [OUT]; log={log_path.resolve()}")

    cl_dir = run_dir / "contrastive"
    emb_path = cl_dir / "embeddings.npy"
    paths_path = cl_dir / "paths.json"
    hist_path = cl_dir / "history.json"
    if not emb_path.exists():
        raise RuntimeError(f"missing embeddings: {emb_path.resolve()}")

    emb_all = l2(np.load(emb_path).astype(np.float32, copy=False))
    emb, y, classes = filter_eval_embedding(emb_all, paths_path if paths_path.exists() else None)
    km = report_kmeans_metrics(emb, y)
    kn = knn_same_rates(emb, y)
    hdb = report_hdbscan_metrics(emb, y)
    history = json.loads(hist_path.read_text(encoding="utf-8")) if hist_path.exists() else []
    final_loss = history[-1] if history else {}
    return {
        "stage": cond["stage"],
        "method": cond["method"],
        "run_dir": str(run_dir.resolve()),
        "model_path": str((cl_dir / "best_model.pt").resolve()),
        "embedding_path": str(emb_path.resolve()),
        "log_path": str(log_path.resolve()),
        "classes": ",".join(classes),
        "seconds": round(time.time() - t0, 1),
        "loss": final_loss.get("loss"),
        "nce": final_loss.get("nce"),
        "local": final_loss.get("local"),
        "neco": final_loss.get("neco"),
        "kmeans_ari": km["kmeans_ari"],
        "kmeans_nmi": km["kmeans_nmi"],
        "kmeans_ami": km["kmeans_ami"],
        "top1": kn["1"],
        "k3": kn["3"],
        "k5": kn["5"],
        "k7": kn["7"],
        "k9": kn["9"],
        "dist_ratio": distance_ratio(emb, y),
        **hdb,
    }


def write_outputs(rows: list[dict[str, Any]]) -> None:
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
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in cols})

    with OUT_MD.open("w", encoding="utf-8") as f:
        f.write("# SimCLR Component Ablation\n\n")
        f.write(f"- train folder: `{TRAIN_DIR.resolve()}`\n")
        f.write(f"- eval folder: `{EVAL_DIR.resolve()}`\n")
        f.write("- backbone: `hf_hub:timm/convnext_base.dinov3_lvd1689m`\n")
        f.write("- eval embedding: backbone feature\n")
        f.write("- primary metric: k-means(k=3) ARI on held-out novel classes\n\n")
        f.write("| Stage | Method | ARI | NMI | AMI | P1 capture | image cap | P2 noise | P3 comp | P4 homog | top1 | k5 | k9 | dist ratio | HDB ARI |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['stage']} | {r['method']} | {r['kmeans_ari']:.4f} | "
                f"{r['kmeans_nmi']:.4f} | {r['kmeans_ami']:.4f} | "
                f"{float(r.get('p1_capture_rate') or 0):.4f} | "
                f"{float(r.get('p1_image_capture_rate') or 0):.4f} | "
                f"{float(r.get('p2_noise_pct') or 0):.2f}% | "
                f"{float(r.get('p3_completeness') or 0):.4f} | "
                f"{float(r.get('p4_homogeneity') or 0):.4f} | "
                f"{r['top1']*100:.2f}% | {r['k5']*100:.2f}% | {r['k9']*100:.2f}% | "
                f"{r['dist_ratio']:.4f} | {float(r.get('hdbscan_ari') or 0):.4f} |\n"
            )
        f.write("\n## Artifacts\n\n")
        for r in rows:
            f.write(f"- {r['stage']} {r['method']}\n")
            f.write(f"  - run: `{r['run_dir']}`\n")
            f.write(f"  - model: `{r['model_path']}`\n")
            f.write(f"  - embedding: `{r['embedding_path']}`\n")
            f.write(f"  - log: `{r['log_path']}`\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--lr-backbone", type=float, default=2e-6)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--start-at", default=None, help="stage name, e.g. C3 or queue_neg_ignore")
    ap.add_argument("--only", default=None, help="run only one stage, e.g. C0 or base_simclr")
    args = ap.parse_args()

    if not TRAIN_DIR.exists():
        raise SystemExit(f"train folder not found: {TRAIN_DIR.resolve()}")
    if not EVAL_DIR.exists():
        raise SystemExit(f"eval folder not found: {EVAL_DIR.resolve()}")

    rows: list[dict[str, Any]] = []
    started = args.start_at is None
    for cond in CONDITIONS:
        if args.only and args.only not in {cond["stage"], cond["name"]}:
            continue
        if not started:
            started = args.start_at in {cond["stage"], cond["name"]}
        if not started:
            continue
        row = train_condition(cond, args.epochs, args.batch, args.img_size,
                              args.lr_backbone, args.lr_head)
        rows.append(row)
        write_outputs(rows)
        print(f"[ablation] updated {OUT_MD.resolve()}", flush=True)

    write_outputs(rows)
    print(f"[OUT] {OUT_MD.resolve()}")
    print(f"[OUT] {OUT_CSV.resolve()}")


if __name__ == "__main__":
    main()
