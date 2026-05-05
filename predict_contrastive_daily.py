#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Daily contrastive prediction for production-style wafer folders.

Input layout:
    <image_root>/<device>/<processid>/<yyyymmdd>/*.png

Output layout:
    <output_root>/<device>/<processid>/<yyyymmdd>/
      manifest.json
      run.log
      preds.parquet
      clusters.parquet
      cluster_members.parquet
      medoids/
      review/
      embeddings/
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import torch
from torchvision.datasets import ImageFolder

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", required=True,
                    help="root containing <device>/<processid>/<yyyymmdd>")
    ap.add_argument("--date", required=True,
                    help="date folder to predict, e.g. 20260505")
    ap.add_argument("--run-dir", default="outputs/logs_contrastive/overall",
                    help="trained contrastive run dir or overall mirror")
    ap.add_argument("--output-root", default="D:/project/result_contrastive")
    ap.add_argument("--device", default=None,
                    help="optional device folder filter")
    ap.add_argument("--processid", default=None,
                    help="optional processid folder filter")
    ap.add_argument("--normal-class", default="Normal_bank_boundary")
    ap.add_argument("--min-cluster-size", type=int, default=12)
    ap.add_argument("--min-samples", type=int, default=1)
    ap.add_argument("--hdbscan-metric", default="euclidean")
    ap.add_argument("--cluster-selection-method", choices=["leaf", "eom"],
                    default="eom")
    ap.add_argument("--cluster-selection-epsilon", type=float, default=0.0)
    ap.add_argument("--use-run-hdbscan", action="store_true",
                    help="use HDBSCAN settings stored in run_info.json")
    ap.add_argument("--knn-k", type=int, default=10)
    ap.add_argument("--normal-knn-ratio", type=float, default=0.60)
    ap.add_argument("--defect-knn-ratio", type=float, default=0.50)
    ap.add_argument("--normal-centroid-pct", type=float, default=95.0)
    ap.add_argument("--review-copy-limit", type=int, default=200)
    return ap.parse_args()


def apply_hdbscan_args(cfg: dict, args) -> dict:
    if args.use_run_hdbscan:
        return {}
    overrides = {
        "MIN_CLUSTER_SIZE": args.min_cluster_size,
        "MIN_SAMPLES": args.min_samples,
        "HDBSCAN_METRIC": args.hdbscan_metric,
        "CLUSTER_SELECTION_METHOD": args.cluster_selection_method,
        "CLUSTER_SELECTION_EPSILON": args.cluster_selection_epsilon,
    }
    cfg.update(overrides)
    return overrides


def setup_logger(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("predict_contrastive_daily")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); lg.addHandler(sh)
    fh = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt); lg.addHandler(fh)
    return lg


def find_date_dirs(image_root: Path, date: str, device: str | None,
                   processid: str | None) -> list[tuple[str, str, Path]]:
    out = []
    if device and processid:
        d = image_root / device / processid / date
        return [(device, processid, d)] if d.exists() else []
    for dev_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
        if device and dev_dir.name != device:
            continue
        for proc_dir in sorted(p for p in dev_dir.iterdir() if p.is_dir()):
            if processid and proc_dir.name != processid:
                continue
            date_dir = proc_dir / date
            if date_dir.exists():
                out.append((dev_dir.name, proc_dir.name, date_dir))
    return out


def list_images(date_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in date_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def load_run(run_dir: Path):
    info_path = run_dir / "run_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"run_info.json missing: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    cfg = info["cfg"] if "cfg" in info else info
    return info, cfg


def build_model(run_dir: Path, cfg: dict, device: torch.device, logger):
    import contrastive_unknown_n50 as contrastive

    init_backbone = run_dir / "_init_backbone.pth"
    if not init_backbone.exists():
        init_backbone = run_dir / "checkpoints" / "final_infer.pt"
    final_pt = run_dir / "checkpoints" / "final_infer.pt"
    if not final_pt.exists():
        raise FileNotFoundError(f"final_infer.pt missing: {final_pt}")

    contrastive.CFG.update({
        "BACKBONE_NAME": cfg["BACKBONE_NAME"],
        "IMAGE_SIZE": cfg["IMAGE_SIZE"],
        "PROJ_DIM": cfg["PROJ_DIM"],
        "FREEZE_BACKBONE": cfg.get("FREEZE_BACKBONE", True),
        "LOCAL_BACKBONE_WEIGHTS": str(init_backbone),
        "NUM_WORKERS": 0,
        "BATCH": 32,
        "PIN_MEMORY": True,
        "PREFETCH_FACTOR": 4,
        "PERSISTENT": False,
        "EMBED_LOG_TICKS": 10,
        "EMBED_LOG_EVERY_BATCH": False,
        "DROP_LAST": False,
    })
    logger.info(f"[model] load {final_pt}")
    model = contrastive.CL().to(device)
    ckpt = torch.load(final_pt, map_location=device, weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def extract_items(model, items, device, logger, label):
    import contrastive_unknown_n50 as contrastive
    t0 = time.time()
    emb, _labs, files, classes = contrastive.extract(
        model, items, device, logger, log_progress=True, log_name=label)
    logger.info(f"[{label}] embeddings {emb.shape} in {time.time()-t0:.1f}s")
    return emb.astype(np.float32), files, classes


def load_reference_embeddings(model, cfg: dict, device, logger):
    train_dir = Path(cfg["TRAIN_DIR"])
    if not train_dir.exists():
        raise FileNotFoundError(f"TRAIN_DIR missing for reference embeddings: {train_dir}")
    ds = ImageFolder(str(train_dir))
    items = [(Path(p), -1, ds.classes[int(y)]) for p, y in ds.samples]
    emb, _files, labels = extract_items(model, items, device, logger, "RefEmbed")
    emb /= np.maximum(1e-12, np.linalg.norm(emb, axis=1, keepdims=True))
    return emb, np.array(labels)


def run_hdbscan(emb: np.ndarray, cfg: dict, logger):
    params = dict(
        min_cluster_size=cfg.get("MIN_CLUSTER_SIZE", 12),
        min_samples=cfg.get("MIN_SAMPLES", 4),
        metric=cfg.get("HDBSCAN_METRIC", "euclidean"),
        cluster_selection_method=cfg.get("CLUSTER_SELECTION_METHOD", "leaf"),
        cluster_selection_epsilon=cfg.get("CLUSTER_SELECTION_EPSILON", 0.06),
        allow_single_cluster=cfg.get("ALLOW_SINGLE_CLUSTER", False),
    )
    logger.info(f"[hdbscan] params={params}")
    cl = hdbscan.HDBSCAN(**params)
    ids = cl.fit_predict(emb)
    probs = getattr(cl, "probabilities_", np.zeros(len(emb), dtype=np.float32))
    logger.info(f"[hdbscan] clusters={len(set(int(x) for x in ids if x >= 0))} "
                f"noise={int((ids == -1).sum())}")
    return ids.astype(np.int32), probs.astype(np.float32)


def compute_reference_scores(emb, ref_emb, ref_labels, normal_class, k, normal_centroid_pct):
    emb_n = emb / np.maximum(1e-12, np.linalg.norm(emb, axis=1, keepdims=True))
    sim = emb_n @ ref_emb.T
    kk = min(k, ref_emb.shape[0])
    top = np.argpartition(-sim, kth=np.arange(kk), axis=1)[:, :kk]
    row = np.arange(top.shape[0])[:, None]
    top = top[row, np.argsort(-sim[row, top], axis=1)]

    normal_ref = ref_emb[ref_labels == normal_class]
    if len(normal_ref) == 0:
        normal_centroid = ref_emb.mean(axis=0, keepdims=True)
        normal_threshold = 0.0
    else:
        normal_centroid = normal_ref.mean(axis=0, keepdims=True)
        normal_centroid /= np.maximum(1e-12, np.linalg.norm(normal_centroid, axis=1, keepdims=True))
        normal_ref_dist = 1.0 - (normal_ref @ normal_centroid.T).reshape(-1)
        normal_threshold = float(np.percentile(normal_ref_dist, normal_centroid_pct))

    out = []
    for i in range(emb.shape[0]):
        top_labels = ref_labels[top[i]]
        cnt = Counter(str(x) for x in top_labels)
        nearest_ref_class, nearest_ref_count = cnt.most_common(1)[0]
        normal_ratio = float(np.mean(top_labels == normal_class))
        normal_dist = float(1.0 - (emb_n[i:i+1] @ normal_centroid.T).item())
        out.append({
            "nearest_ref_class": nearest_ref_class,
            "nearest_ref_ratio": float(nearest_ref_count / kk),
            "knn_normal_ratio": normal_ratio,
            "nearest_normal_dist": normal_dist,
            "normal_dist_threshold": normal_threshold,
        })
    return out


def cluster_tables(ids, probs, image_paths, decisions):
    size_map = Counter(int(x) for x in ids)
    cluster_records = []
    member_records = []
    for cid in sorted(size_map):
        members = [i for i, x in enumerate(ids) if int(x) == cid]
        dec_counts = Counter(decisions[i] for i in members)
        cluster_records.append({
            "cluster_id": int(cid),
            "cluster_name": "noise" if cid == -1 else f"c{cid:03d}",
            "size": int(size_map[cid]),
            "mean_cluster_prob": float(np.mean(probs[members])) if members else 0.0,
            "dominant_decision": dec_counts.most_common(1)[0][0] if dec_counts else "",
        })
        for i in members:
            member_records.append({
                "cluster_id": int(cid),
                "cluster_name": "noise" if cid == -1 else f"c{cid:03d}",
                "input_path": str(image_paths[i]),
                "cluster_prob": float(probs[i]),
            })
    return cluster_records, member_records


def copy_medoids(out_dir, emb, ids, image_paths, review_indices, review_copy_limit):
    medoid_dir = out_dir / "medoids"; medoid_dir.mkdir(exist_ok=True)
    for cid in sorted(set(int(x) for x in ids if x >= 0)):
        idxs = np.where(ids == cid)[0]
        e = emb[idxs]
        center = e.mean(axis=0, keepdims=True)
        d = np.linalg.norm(e - center, axis=1)
        src = image_paths[int(idxs[int(np.argmin(d))])]
        shutil.copy2(src, medoid_dir / f"cluster_{cid:03d}__{src.name}")

    review_dir = out_dir / "review"; review_dir.mkdir(exist_ok=True)
    for i in review_indices[:review_copy_limit]:
        src = image_paths[i]
        shutil.copy2(src, review_dir / src.name)


def predict_one(date_dir_info, args, run_dir, cfg, model, ref_emb, ref_labels, device):
    dev, proc, date_dir = date_dir_info
    out_dir = Path(args.output_root) / dev / proc / args.date
    logger = setup_logger(out_dir)
    image_paths = list_images(date_dir)
    logger.info(f"[input] {date_dir} images={len(image_paths)}")
    if not image_paths:
        return

    items = [(p, -1, "unlabeled") for p in image_paths]
    emb, files, _classes = extract_items(model, items, device, logger, "ProdEmbed")
    emb_norm = emb / np.maximum(1e-12, np.linalg.norm(emb, axis=1, keepdims=True))
    ids, probs = run_hdbscan(emb_norm, cfg, logger)
    ref_scores = compute_reference_scores(
        emb_norm, ref_emb, ref_labels, args.normal_class, args.knn_k, args.normal_centroid_pct)

    size_map = Counter(int(x) for x in ids)
    decisions = []
    pred_rows = []
    review_indices = []
    for i, p in enumerate(image_paths):
        s = ref_scores[i]
        cid = int(ids[i])
        if (s["knn_normal_ratio"] >= args.normal_knn_ratio or
                s["nearest_normal_dist"] <= s["normal_dist_threshold"]):
            decision = "normal_like"
        elif cid == -1:
            decision = "review_noise"
        elif (s["nearest_ref_class"] != args.normal_class and
              s["nearest_ref_ratio"] >= args.defect_knn_ratio):
            decision = "known_defect_cluster"
        else:
            decision = "unknown_cluster"
        decisions.append(decision)
        if decision in {"review_noise", "unknown_cluster"}:
            review_indices.append(i)
        pred_rows.append({
            "input_path": str(p),
            "basename": p.name,
            "device": dev,
            "processid": proc,
            "yyyymmdd": args.date,
            "wafer_id": p.stem,
            "embedding_id": i,
            "cluster_id": cid,
            "cluster_prob": float(probs[i]),
            "cluster_size": int(size_map[cid]),
            "nearest_normal_dist": s["nearest_normal_dist"],
            "knn_normal_ratio": s["knn_normal_ratio"],
            "nearest_ref_class": s["nearest_ref_class"],
            "nearest_ref_ratio": s["nearest_ref_ratio"],
            "anomaly_score": s["nearest_normal_dist"],
            "decision": decision,
            "review_reason": "" if decision != "review_noise" else "hdbscan_noise",
        })

    cluster_rows, member_rows = cluster_tables(ids, probs, image_paths, decisions)
    pd.DataFrame(pred_rows).to_parquet(out_dir / "preds.parquet", index=False)
    pd.DataFrame(cluster_rows).to_parquet(out_dir / "clusters.parquet", index=False)
    pd.DataFrame(member_rows).to_parquet(out_dir / "cluster_members.parquet", index=False)
    emb_dir = out_dir / "embeddings"; emb_dir.mkdir(exist_ok=True)
    np.save(emb_dir / "embedding.npy", emb_norm)
    (emb_dir / "files.txt").write_text("\n".join(str(p) for p in image_paths), encoding="utf-8")
    copy_medoids(out_dir, emb_norm, ids, image_paths, review_indices, args.review_copy_limit)
    manifest = {
        "run_dir": str(run_dir),
        "input_dir": str(date_dir),
        "output_dir": str(out_dir),
        "n_images": len(image_paths),
        "decision_counts": dict(Counter(decisions)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"[done] {out_dir} decisions={manifest['decision_counts']}")


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    image_root = Path(args.image_root).resolve()
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    info, cfg = load_run(run_dir)
    hdbscan_overrides = apply_hdbscan_args(cfg, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bootstrap_logger = setup_logger(out_root / "_bootstrap" / args.date)
    bootstrap_logger.info(f"[main] run_dir={run_dir} device={device}")
    if hdbscan_overrides:
        bootstrap_logger.info(f"[main] hdbscan_overrides={hdbscan_overrides}")
    else:
        bootstrap_logger.info("[main] using HDBSCAN settings from run_info.json")
    model = build_model(run_dir, cfg, device, bootstrap_logger)
    ref_emb, ref_labels = load_reference_embeddings(model, cfg, device, bootstrap_logger)

    date_dirs = find_date_dirs(image_root, args.date, args.device, args.processid)
    bootstrap_logger.info(f"[main] date dirs={len(date_dirs)}")
    if not date_dirs:
        raise SystemExit(f"no date folders found for {args.date} under {image_root}")
    for info_tuple in date_dirs:
        predict_one(info_tuple, args, run_dir, cfg, model, ref_emb, ref_labels, device)


if __name__ == "__main__":
    main()
