#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone eval for contrastive learning runs.

Read: <run_dir>/run_info.json + checkpoints/final_infer.pt + UNKNOWN_DIR
Compute: ARI / NMI / cluster purity / silhouette + per-class noise %
         on with-Normal / without-Normal / optional stress-test.
         Normal is excluded from the primary class metric.
Write: <run_dir>/eval/{eval_summary.json, per_class_report.txt,
       cluster_report.parquet, class_fragmentation.parquet,
       retrieval_report.parquet, file_list.parquet, groups/, plots/, embeddings/}
After eval:
  - rename run_dir → <prev_name>_ari{:.2f}_nmi{:.2f} using without-Normal metrics
  - update <output_root>/overall/ if best without-Normal ARI

contrastive_unknown_n50.py 기반 — import only.
"""
import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import hdbscan
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    homogeneity_completeness_v_measure,
    silhouette_score,
    silhouette_samples,
)

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--unknown-dir", default=None,
                    help="override UNKNOWN_DIR from run_info.json")
    ap.add_argument("--eval-name", default="eval",
                    help="output subdirectory name under run_dir")
    ap.add_argument("--normal-class", default="Normal_bank_boundary")
    ap.add_argument("--min-cluster-size", type=int, default=None)
    ap.add_argument("--min-samples", type=int, default=None)
    ap.add_argument("--hdbscan-metric", default=None)
    ap.add_argument("--cluster-selection-method", choices=["leaf", "eom"],
                    default=None)
    ap.add_argument("--cluster-selection-epsilon", type=float, default=None)
    ap.add_argument("--stress-test", action="store_true")
    ap.add_argument("--stress-normal-mult", type=int, default=5)
    ap.add_argument("--stress-defect-per-class", type=int, default=100)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-embedding-export", dest="embedding_export",
                    action="store_false")
    ap.add_argument("--embedding-export", action="store_true", default=True)
    ap.add_argument("--no-rename", action="store_true",
                    help="skip <run>_ari{:.2f}_nmi{:.2f} rename")
    ap.add_argument("--no-overall", action="store_true",
                    help="skip overall/ best mirror update")
    ap.add_argument("--force-overall", action="store_true",
                    help="mirror this run to overall/ even if metric is lower")
    return ap.parse_args()


def hdbscan_overrides_from_args(args) -> dict:
    pairs = [
        ("min_cluster_size", "MIN_CLUSTER_SIZE"),
        ("min_samples", "MIN_SAMPLES"),
        ("hdbscan_metric", "HDBSCAN_METRIC"),
        ("cluster_selection_method", "CLUSTER_SELECTION_METHOD"),
        ("cluster_selection_epsilon", "CLUSTER_SELECTION_EPSILON"),
    ]
    return {
        key: getattr(args, attr)
        for attr, key in pairs
        if getattr(args, attr) is not None
    }


def apply_hdbscan_overrides(cfg: dict, args) -> dict:
    overrides = hdbscan_overrides_from_args(args)
    cfg.update(overrides)
    return overrides


def setup_logger(eval_dir: Path) -> logging.Logger:
    eval_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("eval_contrastive")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); lg.addHandler(sh)
    fh = logging.FileHandler(eval_dir / "eval.log", encoding="utf-8")
    fh.setFormatter(fmt); lg.addHandler(fh)
    return lg


def build_model(run_dir: Path, cfg: dict, device: torch.device, logger):
    import contrastive_unknown_n50 as contrastive
    init_backbone = run_dir / "_init_backbone.pth"
    if not init_backbone.exists():
        init_backbone = run_dir / "checkpoints" / "final_infer.pt"
        if not init_backbone.exists():
            sys.exit(f"[err] no backbone init found in {run_dir}")
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
    logger.info(f"[model] CL() init (backbone via {init_backbone.name})")
    model = contrastive.CL().to(device)
    final_pt = run_dir / "checkpoints" / "final_infer.pt"
    ckpt = torch.load(final_pt, map_location=device, weights_only=False)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info(f"[model] final_infer.pt loaded "
                f"(missing={len(missing)} unexpected={len(unexpected)})")
    model.eval()
    return model


def extract_embeddings(model, unknown_dir: Path, device, logger):
    import contrastive_unknown_n50 as contrastive
    ds = ImageFolder(str(unknown_dir))
    items = [(Path(p), -1, ds.classes[int(y)]) for p, y in ds.samples]
    logger.info(f"[extract] {len(items)} images, {len(ds.classes)} classes")
    t0 = time.time()
    emb, _labs, files, classes = contrastive.extract(
        model, items, device, logger,
        log_progress=True, log_name="EvalEmbed")
    logger.info(f"[extract] done {time.time()-t0:.1f}s, shape={emb.shape}")
    return emb, classes, files


def run_hdbscan(emb: np.ndarray, cfg: dict, logger, label="hdbscan"):
    params = dict(
        min_cluster_size=cfg.get("MIN_CLUSTER_SIZE", 12),
        min_samples=cfg.get("MIN_SAMPLES", 4),
        metric=cfg.get("HDBSCAN_METRIC", "euclidean"),
        cluster_selection_method=cfg.get("CLUSTER_SELECTION_METHOD", "leaf"),
        cluster_selection_epsilon=cfg.get("CLUSTER_SELECTION_EPSILON", 0.06),
        allow_single_cluster=cfg.get("ALLOW_SINGLE_CLUSTER", False),
    )
    logger.info(f"[{label}] params={params}")
    t0 = time.time()
    cl = hdbscan.HDBSCAN(**params)
    ids = cl.fit_predict(emb)
    logger.info(f"[{label}] {time.time()-t0:.1f}s, "
                f"clusters={len(set(int(x) for x in ids if x>=0))}, "
                f"noise={int((ids==-1).sum())}")
    return ids


def cluster_purity(gt_labels: np.ndarray, cluster_ids: np.ndarray) -> float:
    if len(gt_labels) == 0:
        return 0.0
    total = len(gt_labels); matched = 0
    for c in np.unique(cluster_ids):
        mask = cluster_ids == c
        if mask.sum() == 0: continue
        gts_in = gt_labels[mask]
        _, counts = np.unique(gts_in, return_counts=True)
        matched += counts.max()
    return float(matched / total)


def per_class_noise(gt_str: list, cluster_ids: np.ndarray) -> dict:
    out = {}
    gt_arr = np.array(gt_str)
    for cls in sorted(set(gt_str)):
        mask = gt_arr == cls
        n = int(mask.sum())
        n_noise = int((cluster_ids[mask] == -1).sum())
        out[cls] = {"n": n, "noise": n_noise,
                    "noise_pct": float(100.0*n_noise/max(1, n))}
    return out


def _write_table(records: list[dict], path: Path, logger):
    import pandas as pd
    df = pd.DataFrame.from_records(records)
    df.to_parquet(path, index=False)
    logger.info(f"[write] {path.name} rows={len(df)}")


def _group_map(cluster_ids: np.ndarray) -> dict[int, tuple[str, int]]:
    ids = sorted(int(x) for x in set(cluster_ids) if int(x) >= 0)
    out = {-1: ("group_noise", 0)}
    for i, cid in enumerate(ids, start=1):
        out[cid] = (f"group_{i:03d}", i)
    return out


def _split_wafer_basename(stem: str) -> dict:
    schema = ["prefix", "kind", "w_idx", "date", "time",
              "yld", "syp", "tester", "device"]
    parts = stem.split("_")
    return {name: (parts[i] if i < len(parts) else "")
            for i, name in enumerate(schema)}


def _link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            dst.unlink()
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def write_group_outputs(files: list[str], classes: list[str], cluster_ids: np.ndarray,
                        out_dir: Path, logger):
    """Write group folders and DB-friendly file_list.parquet."""
    groups_dir = out_dir / "groups"
    if groups_dir.exists():
        shutil.rmtree(groups_dir)
    groups_dir.mkdir(parents=True, exist_ok=True)

    group_map = _group_map(cluster_ids)
    size_map = {int(cid): int((cluster_ids == cid).sum())
                for cid in set(int(x) for x in cluster_ids)}
    rows = []
    for i, (file_s, cid_raw) in enumerate(zip(files, cluster_ids)):
        src = Path(file_s)
        cid = int(cid_raw)
        group_name, group_idx = group_map[cid]
        dst = groups_dir / group_name / f"{i:06d}__{src.name}"
        _link_or_copy(src, dst)

        row = {
            "path": str(src),
            "group_path": str(dst),
            "wafer_basename": src.stem,
            "wafer_group": group_name,
            "wafer_group_idx": group_idx,
            "cluster_id": cid,
            "group_size": int(size_map[cid]),
        }
        row.update(_split_wafer_basename(src.stem))
        rows.append(row)

    _write_table(rows, out_dir / "file_list.parquet", logger)
    logger.info(f"[write] groups images -> {groups_dir}")


def cluster_report_records(cluster_ids: np.ndarray, classes: list[str],
                           normal_class: str) -> tuple[list[dict], dict[int, str]]:
    cls_arr = np.array(classes)
    records = []
    dominant = {}
    for cid in sorted(set(int(x) for x in cluster_ids)):
        mask = cluster_ids == cid
        size = int(mask.sum())
        counts = {}
        for c in cls_arr[mask]:
            counts[str(c)] = counts.get(str(c), 0) + 1
        if counts:
            dom_cls, dom_n = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0]
        else:
            dom_cls, dom_n = "", 0
        dominant[cid] = dom_cls
        normal_n = int(counts.get(normal_class, 0))
        records.append({
            "cluster_id": cid,
            "cluster_name": "noise" if cid == -1 else f"c{cid:03d}",
            "size": size,
            "dominant_class": dom_cls,
            "dominant_count": int(dom_n),
            "purity": float(dom_n / max(1, size)),
            "normal_count": normal_n,
            "normal_pct": float(100.0 * normal_n / max(1, size)),
            "defect_count": int(size - normal_n),
            "n_classes": int(len(counts)),
        })
    return records, dominant


def class_fragmentation_records(cluster_ids: np.ndarray, classes: list[str],
                                normal_class: str) -> list[dict]:
    cls_arr = np.array(classes)
    records = []
    for cls in sorted(c for c in set(classes) if c != normal_class):
        mask = cls_arr == cls
        cids = cluster_ids[mask]
        n = int(mask.sum())
        non_noise = cids[cids != -1]
        uniq, cnts = np.unique(non_noise, return_counts=True)
        if len(cnts):
            best_i = int(np.argmax(cnts))
            dominant_cluster = int(uniq[best_i])
            dominant_count = int(cnts[best_i])
        else:
            dominant_cluster = -1
            dominant_count = 0
        records.append({
            "class": cls,
            "n": n,
            "n_noise": int((cids == -1).sum()),
            "noise_pct": float(100.0 * (cids == -1).sum() / max(1, n)),
            "n_clusters": int(len(set(int(x) for x in non_noise))),
            "dominant_cluster": dominant_cluster,
            "dominant_cluster_count": dominant_count,
            "dominant_cluster_recall": float(dominant_count / max(1, n)),
            "cluster_coverage": float(len(non_noise) / max(1, n)),
        })
    return records


def class_fragmentation_summary(class_records: list[dict]) -> dict:
    """Aggregate class_fragmentation_records → 사용자 criteria A/B/C 의 단일 dict.

    docs/contrastive-eval/METRICS.md 의 spec 그대로:
      A — class_capture_rate (모든 class 가 ≥1 cluster 에 잡힘)
      B — weighted_cluster_coverage (sample-weighted, 1 - defect noise rate)
      C — frac_single_cluster (n_clusters == 1 인 class 비율, 1 best)
    """
    if not class_records:
        return {"n_defect_classes": 0}
    n_total = len(class_records)
    n_captured = sum(1 for r in class_records if r["n_clusters"] >= 1)
    n_single = sum(1 for r in class_records if r["n_clusters"] == 1)
    n_split_2 = sum(1 for r in class_records if r["n_clusters"] == 2)
    n_split_3plus = sum(1 for r in class_records if r["n_clusters"] >= 3)
    total_n = sum(r["n"] for r in class_records)
    weighted_cov = (sum(r["cluster_coverage"] * r["n"] for r in class_records) /
                    max(1, total_n))
    mean_cov = sum(r["cluster_coverage"] for r in class_records) / n_total
    mean_n_clu = sum(r["n_clusters"] for r in class_records) / n_total
    return {
        "n_defect_classes": n_total,
        "n_classes_captured": n_captured,
        "n_classes_uncaptured": n_total - n_captured,
        "class_capture_rate": round(n_captured / n_total, 4),                            # P1
        "mean_cluster_coverage": round(mean_cov, 4),
        "weighted_cluster_coverage": round(weighted_cov, 4),                             # B
        "mean_n_clusters_per_class": round(mean_n_clu, 4),
        "n_classes_single_cluster": n_single,
        "n_classes_split_2": n_split_2,
        "n_classes_split_3plus": n_split_3plus,
        "frac_single_cluster": round(n_single / n_total, 4),                             # C
    }


def _ap_at_k(matches: np.ndarray, k: int) -> float:
    matches = matches[:k].astype(np.float32)
    if matches.sum() <= 0:
        return 0.0
    precision_at_i = np.cumsum(matches) / np.arange(1, len(matches) + 1)
    return float((precision_at_i * matches).sum() / min(matches.sum(), k))


def retrieval_report_records(emb: np.ndarray, classes: list[str],
                             normal_class: str, ks=(1, 5, 10)) -> list[dict]:
    cls_arr = np.array(classes)
    keep = cls_arr != normal_class
    idx = np.where(keep)[0]
    if len(idx) < 2:
        return []
    e = emb[idx].astype(np.float32)
    e /= np.maximum(1e-12, np.linalg.norm(e, axis=1, keepdims=True))
    sim = e @ e.T
    np.fill_diagonal(sim, -np.inf)
    max_k = min(max(ks), max(1, len(idx) - 1))
    order = np.argpartition(-sim, kth=np.arange(max_k), axis=1)[:, :max_k]
    row = np.arange(order.shape[0])[:, None]
    order = order[row, np.argsort(-sim[row, order], axis=1)]
    labels = cls_arr[idx]

    def summarize(mask, name):
        local = np.where(mask)[0]
        if len(local) == 0:
            return None
        rec = {"class": name, "n": int(len(local))}
        maps = []
        for q in local:
            neigh = order[q]
            matches = labels[neigh] == labels[q]
            maps.append(_ap_at_k(matches, min(10, max_k)))
            for k in ks:
                kk = min(k, max_k)
                rec[f"recall_at_{k}"] = rec.get(f"recall_at_{k}", 0.0) + float(matches[:kk].mean())
        for k in ks:
            rec[f"recall_at_{k}"] = float(rec[f"recall_at_{k}"] / len(local))
        rec["map_at_10"] = float(np.mean(maps)) if maps else 0.0
        return rec

    records = []
    overall = summarize(np.ones(len(idx), dtype=bool), "__overall__")
    if overall:
        records.append(overall)
    for cls in sorted(set(labels)):
        rec = summarize(labels == cls, str(cls))
        if rec:
            records.append(rec)
    return records


def normal_leakage_metrics(cluster_ids: np.ndarray, classes: list[str],
                           normal_class: str, dominant: dict[int, str]) -> dict:
    cls_arr = np.array(classes)
    normal_mask = cls_arr == normal_class
    n_normal = int(normal_mask.sum())
    if n_normal == 0:
        return {
            "n_normal": 0, "normal_noise_pct": None,
            "normal_leakage_to_defect_cluster": None,
            "normal_leakage_count": 0,
        }
    normal_cids = cluster_ids[normal_mask]
    noise_n = int((normal_cids == -1).sum())
    leakage = 0
    for cid in normal_cids:
        cid = int(cid)
        if cid != -1 and dominant.get(cid) != normal_class:
            leakage += 1
    return {
        "n_normal": n_normal,
        "normal_noise_pct": float(100.0 * noise_n / max(1, n_normal)),
        "normal_leakage_to_defect_cluster": float(100.0 * leakage / max(1, n_normal)),
        "normal_leakage_count": int(leakage),
    }


def metrics_on_set(emb_sub, gt_str_sub, cluster_ids_sub, label, logger):
    cls_uniq = sorted(set(gt_str_sub))
    cls_to_idx = {c: i for i, c in enumerate(cls_uniq)}
    gt_idx = np.array([cls_to_idx[c] for c in gt_str_sub])
    n = len(gt_str_sub)
    n_noise = int((cluster_ids_sub == -1).sum())
    n_clusters = len(set(int(x) for x in cluster_ids_sub if x >= 0))
    ari = float(adjusted_rand_score(gt_idx, cluster_ids_sub))
    nmi = float(normalized_mutual_info_score(gt_idx, cluster_ids_sub))
    ami = float(adjusted_mutual_info_score(gt_idx, cluster_ids_sub))
    hom, com, vmeas = homogeneity_completeness_v_measure(gt_idx, cluster_ids_sub)
    hom = float(hom); com = float(com); vmeas = float(vmeas)
    purity = cluster_purity(gt_idx, cluster_ids_sub)
    sil = None
    nn_mask = cluster_ids_sub != -1
    if nn_mask.sum() > 1 and len(set(cluster_ids_sub[nn_mask])) >= 2:
        try:
            sil = float(silhouette_score(emb_sub[nn_mask],
                                          cluster_ids_sub[nn_mask],
                                          metric="cosine"))
        except Exception as e:
            logger.warning(f"[{label}] silhouette failed: {e}")
    pcn = per_class_noise(list(gt_str_sub), cluster_ids_sub)
    return {
        "n": n, "n_clusters": n_clusters, "n_noise": n_noise,
        "noise_pct": float(100.0*n_noise/max(1, n)),
        "ari": ari, "nmi": nmi,
        "ami": ami,                                                                      # Vinh et al. 2010, chance-corrected (Tier 1)
        "homogeneity": hom,                                                              # Rosenberg & Hirschberg 2007 (Tier 2)
        "completeness": com,                                                             # Rosenberg & Hirschberg 2007 (Tier 1, P3)
        "v_measure": vmeas,                                                              # Rosenberg & Hirschberg 2007 (Tier 3, 디버그만)
        "cluster_purity": purity,
        "silhouette_cosine": sil, "per_class_noise": pcn,
    }


def parse_loss_curves(run_log: Path):
    if not run_log.exists():
        return []
    pat = re.compile(
        r"Epoch (\d+) done \| G=([0-9.\-naNeE]+) Q=([0-9.\-naNeE]+) "
        r"L=([0-9.\-naNeE]+) \| time=([0-9.]+)s")
    out = []
    for line in run_log.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            out.append({"epoch": int(m.group(1)),
                        "G": float(m.group(2)), "Q": float(m.group(3)),
                        "L": float(m.group(4)),
                        "time_s": float(m.group(5))})
    return out


def plot_cluster_class_heatmap(cluster_ids, gt_str, save_path, title):
    classes = sorted(set(gt_str))
    cls_idx = {c: i for i, c in enumerate(classes)}
    cl_uniq = sorted(set(int(x) for x in cluster_ids))
    cl_idx = {c: i for i, c in enumerate(cl_uniq)}
    M = np.zeros((len(cl_uniq), len(classes)), dtype=np.int32)
    for c, g in zip(cluster_ids, gt_str):
        M[cl_idx[int(c)], cls_idx[g]] += 1
    M_norm = M / np.maximum(1, M.sum(axis=1, keepdims=True))
    fig, ax = plt.subplots(figsize=(max(8, 0.25*len(classes)),
                                     max(6, 0.18*len(cl_uniq))))
    im = ax.imshow(M_norm, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cl_uniq)))
    ax.set_yticklabels([f"noise" if c==-1 else f"c{c:03d}" for c in cl_uniq],
                       fontsize=7)
    ax.set_xlabel("GT class"); ax.set_ylabel("Cluster"); ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                 label="row-normalized count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_size_histogram(cluster_ids, save_path):
    cl_uniq, cnts = np.unique(cluster_ids, return_counts=True)
    order = np.argsort(-cnts)
    cl_uniq = cl_uniq[order]; cnts = cnts[order]
    labels = [f"noise" if c==-1 else f"c{c:03d}" for c in cl_uniq]
    fig, ax = plt.subplots(figsize=(max(8, 0.2*len(cl_uniq)), 4))
    ax.bar(range(len(cnts)), cnts,
           color=["tomato" if c==-1 else "steelblue" for c in cl_uniq])
    ax.set_xticks(range(len(cl_uniq)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("size"); ax.set_title("Cluster size distribution (sorted)")
    for i, v in enumerate(cnts):
        if v > 0:
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_silhouette_per_sample(emb, cluster_ids, save_path):
    nn_mask = cluster_ids != -1
    if nn_mask.sum() < 2 or len(set(cluster_ids[nn_mask])) < 2:
        return
    try:
        sv = silhouette_samples(emb[nn_mask], cluster_ids[nn_mask],
                                 metric="cosine")
    except Exception:
        return
    cl = cluster_ids[nn_mask]
    order = np.lexsort((sv, cl))
    sv_sorted = sv[order]; cl_sorted = cl[order]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.04*len(sv))))
    cl_uniq = sorted(set(cl_sorted))
    cmap = plt.get_cmap("tab20"); y = 0
    for ci, c in enumerate(cl_uniq):
        m = cl_sorted == c; n = int(m.sum())
        ax.barh(range(y, y+n), sv_sorted[m], height=1.0,
                color=cmap(ci % 20), edgecolor="none")
        ax.text(-0.02, y+n/2, f"c{c:03d}", ha="right", va="center", fontsize=6)
        y += n
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlim(-0.2, 1.0); ax.set_xlabel("silhouette (cosine)")
    ax.set_yticks([])
    ax.set_title(f"silhouette per-sample (mean={float(sv.mean()):.3f})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(loss_records, save_path):
    if not loss_records:
        return
    eps = [r["epoch"] for r in loss_records]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(eps, [r["G"] for r in loss_records], "-o",
            label="G (Global InfoNCE)", lw=1.5, ms=3)
    ax.plot(eps, [r["Q"] for r in loss_records], "-s",
            label="Q (Queue InfoNCE)", lw=1.5, ms=3)
    ax.plot(eps, [r["L"] for r in loss_records], "-^",
            label="L (Local InfoNCE)", lw=1.5, ms=3)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title("Training loss curves (lower = better)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def write_per_class_report(metrics_dict: dict, save_path: Path):
    lines = ["# Per-class report", ""]
    for set_name, m in metrics_dict.items():
        lines.append(f"## {set_name}")
        lines.append(f"  n={m['n']} clusters={m['n_clusters']} "
                     f"noise={m['n_noise']} ({m['noise_pct']:.1f}%)")
        lines.append(f"  ARI={m['ari']:.4f}  NMI={m['nmi']:.4f}  "
                     f"purity={m['cluster_purity']:.4f}  "
                     f"silhouette={m['silhouette_cosine']}")
        lines.append("")
        lines.append(f"  per-class noise %:")
        rows = sorted(m["per_class_noise"].items(),
                      key=lambda x: -x[1]["noise_pct"])
        for cls, v in rows:
            lines.append(f"    {cls:<35s} n={v['n']:5d} "
                         f"noise={v['noise']:5d} ({v['noise_pct']:5.1f}%)")
        lines.append("")
    save_path.write_text("\n".join(lines), encoding="utf-8")


def update_overall(run_dir: Path, output_root: Path, m_with: dict,
                   m_without: dict, m_stress: dict, normal_metrics: dict,
                   args, logger):
    """If current without-Normal ARI > overall best, mirror run_dir to overall/."""
    overall_dir = output_root / "overall"
    overall_meta = output_root / "overall" / "_overall_meta.json"
    primary = m_without or m_with
    cur_ari = primary["ari"]
    prev_ari = -1.0
    if overall_meta.exists():
        try:
            prev_ari = float(json.loads(
                overall_meta.read_text(encoding="utf-8")).get("primary_metric_value", -1.0))
        except Exception:
            prev_ari = -1.0
    if args.force_overall or cur_ari > prev_ari:
        if args.force_overall and cur_ari <= prev_ari:
            logger.info(f"[overall] force mirror primary ARI: {cur_ari:.4f} <= {prev_ari:.4f} "
                        f"- mirror to {overall_dir}")
        else:
            logger.info(f"[overall] new best primary ARI: {cur_ari:.4f} > {prev_ari:.4f} "
                        f"- mirror to {overall_dir}")
        if overall_dir.exists():
            shutil.rmtree(overall_dir)
        # Mirror full run_dir (includes eval/, embeddings/, plots/)
        shutil.copytree(run_dir, overall_dir)
        meta = {
            "source_run": run_dir.name,
            "eval_name": args.eval_name,
            "unknown_dir_override": args.unknown_dir,
            "primary_metric": "ari_without_normal" if m_without else "ari_with_normal",
            "primary_metric_value": primary["ari"],
            "hdbscan_overrides": hdbscan_overrides_from_args(args),
            "ari": m_with["ari"], "nmi": m_with["nmi"],
            "cluster_purity": m_with["cluster_purity"],
            "silhouette_cosine": m_with["silhouette_cosine"],
            "noise_pct": m_with["noise_pct"],
            "ari_without_normal": m_without["ari"] if m_without else None,
            "nmi_without_normal": m_without["nmi"] if m_without else None,
            "cluster_purity_without_normal": m_without["cluster_purity"] if m_without else None,
            "defect_noise_pct": m_without["noise_pct"] if m_without else None,
            "normal_leakage_to_defect_cluster": normal_metrics.get("normal_leakage_to_defect_cluster"),
            "normal_noise_pct": normal_metrics.get("normal_noise_pct"),
            "ari_stress": m_stress["ari"] if m_stress else None,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        overall_meta.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        logger.info(f"[overall] not best (cur={cur_ari:.4f} <= "
                    f"prev={prev_ari:.4f}) - overall/ unchanged")


def rename_run_dir(run_dir: Path, primary: dict, logger) -> Path:
    """Append _ari{:.2f}_nmi{:.2f} to run_dir name."""
    suffix = f"_ari{primary['ari']:.2f}_nmi{primary['nmi']:.2f}"
    if run_dir.name.endswith(suffix):
        return run_dir
    new_name = run_dir.name + suffix
    new_dir = run_dir.parent / new_name
    if new_dir.exists():
        logger.warning(f"[rename] target exists, skipping: {new_dir}")
        return run_dir
    try:
        run_dir.rename(new_dir)
    except OSError as exc:
        logger.warning(f"[rename] failed, leaving run_dir unchanged: {exc}")
        return run_dir
    logger.info(f"[rename] {run_dir.name} -> {new_dir.name}")
    return new_dir


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        sys.exit(f"[err] run_dir not found: {run_dir}")

    eval_dir = run_dir / args.eval_name
    plots_dir = eval_dir / "plots"
    emb_dir = eval_dir / "embeddings"
    eval_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    if args.embedding_export:
        emb_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(eval_dir)
    logger.info(f"[main] run_dir={run_dir}")

    info_path = run_dir / "run_info.json"
    if not info_path.exists():
        sys.exit(f"[err] run_info.json missing: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    cfg = info["cfg"] if "cfg" in info else info
    hdbscan_overrides = apply_hdbscan_overrides(cfg, args)
    if hdbscan_overrides:
        logger.info(f"[main] hdbscan_overrides={hdbscan_overrides}")
    if args.unknown_dir is not None:
        cfg["UNKNOWN_DIR"] = args.unknown_dir
        logger.info(f"[main] UNKNOWN_DIR override={args.unknown_dir}")
    unknown_dir = Path(cfg["UNKNOWN_DIR"])
    if not unknown_dir.exists():
        sys.exit(f"[err] UNKNOWN_DIR missing: {unknown_dir}")
    logger.info(f"[main] UNKNOWN_DIR={unknown_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[main] device={device}")
    model = build_model(run_dir, cfg, device, logger)

    emb, classes, files = extract_embeddings(model, unknown_dir, device, logger)

    if args.embedding_export:
        np.save(emb_dir / "embedding.npy", emb)
        (emb_dir / "files.txt").write_text("\n".join(files), encoding="utf-8")
        (emb_dir / "classes.txt").write_text("\n".join(classes), encoding="utf-8")
        (emb_dir / "meta.json").write_text(json.dumps({
            "n_samples": len(files), "dim": int(emb.shape[1]),
            "model_ckpt": str(run_dir / "checkpoints" / "final_infer.pt"),
            "ts": datetime.now().isoformat(timespec="seconds"),
        }, indent=2), encoding="utf-8")
        logger.info(f"[export] embedding.npy {emb.shape} + meta")

    cluster_ids = run_hdbscan(emb, cfg, logger, label="hdbscan_full")
    cluster_records, dominant_by_cluster = cluster_report_records(
        cluster_ids, list(classes), args.normal_class)
    normal_metrics = normal_leakage_metrics(
        cluster_ids, list(classes), args.normal_class, dominant_by_cluster)
    class_records = class_fragmentation_records(
        cluster_ids, list(classes), args.normal_class)
    class_frag_summary = class_fragmentation_summary(class_records)                       # docs/contrastive-eval/METRICS.md
    retrieval_records = retrieval_report_records(
        emb, list(classes), args.normal_class)
    _write_table(cluster_records, eval_dir / "cluster_report.parquet", logger)
    _write_table(class_records, eval_dir / "class_fragmentation.parquet", logger)
    _write_table(retrieval_records, eval_dir / "retrieval_report.parquet", logger)
    write_group_outputs(files, list(classes), cluster_ids, eval_dir, logger)

    logger.info("[eval] === Set 1: with-Normal ===")
    m_with = metrics_on_set(emb, list(classes), cluster_ids,
                             label="with_normal", logger=logger)
    logger.info(f"  ARI={m_with['ari']:.4f} NMI={m_with['nmi']:.4f} "
                f"purity={m_with['cluster_purity']:.4f} "
                f"silhouette={m_with['silhouette_cosine']} "
                f"noise={m_with['noise_pct']:.1f}%")

    logger.info("[eval] === Set 2: without-Normal ===")
    keep = np.array([c != args.normal_class for c in classes])
    m_without = None
    if keep.sum() == 0:
        logger.warning("[eval] no non-Normal samples; skipping Set 2")
    else:
        m_without = metrics_on_set(
            emb[keep], [c for c, k in zip(classes, keep) if k],
            cluster_ids[keep], label="without_normal", logger=logger)
        logger.info(f"  ARI={m_without['ari']:.4f} NMI={m_without['nmi']:.4f} "
                    f"purity={m_without['cluster_purity']:.4f} "
                    f"silhouette={m_without['silhouette_cosine']} "
                    f"noise={m_without['noise_pct']:.1f}%")

    m_stress = None
    cl_stress_ids = None
    cls_stress = None
    if args.stress_test:
        logger.info("[eval] === Set 3: stress-test ===")
        normal_mask = np.array([c == args.normal_class for c in classes])
        defect_mask = ~normal_mask
        defect_idx_keep = []; per_cls_count = {}
        for i in np.where(defect_mask)[0]:
            cls = classes[i]
            if per_cls_count.get(cls, 0) < args.stress_defect_per_class:
                defect_idx_keep.append(i)
                per_cls_count[cls] = per_cls_count.get(cls, 0) + 1
        defect_idx_keep = np.array(defect_idx_keep, dtype=np.int64)
        normal_idx = np.where(normal_mask)[0]
        normal_idx_rep = np.tile(normal_idx, args.stress_normal_mult)
        stress_idx = np.concatenate([defect_idx_keep, normal_idx_rep])
        emb_stress = emb[stress_idx]
        cls_stress = [classes[i] for i in stress_idx]
        logger.info(f"[stress] defect={len(defect_idx_keep)} "
                    f"normal_rep={len(normal_idx_rep)} total={len(stress_idx)}")
        cl_stress_ids = run_hdbscan(emb_stress, cfg, logger,
                                     label="hdbscan_stress")
        m_stress = metrics_on_set(emb_stress, cls_stress, cl_stress_ids,
                                   label="stress_test", logger=logger)
        logger.info(f"  ARI={m_stress['ari']:.4f} NMI={m_stress['nmi']:.4f} "
                    f"purity={m_stress['cluster_purity']:.4f} "
                    f"silhouette={m_stress['silhouette_cosine']} "
                    f"noise={m_stress['noise_pct']:.1f}%")

    summary = {
        "run_dir": str(run_dir),
        "eval_dir": str(eval_dir),
        "ts": datetime.now().isoformat(timespec="seconds"),
        "n_total": len(files), "n_classes": len(set(classes)),
        "primary_metric": "ari_without_normal" if m_without else "ari_with_normal",
        "primary_metric_value": (m_without or m_with)["ari"],
        "with_normal": m_with, "without_normal": m_without,
        "normal_metrics": normal_metrics,
        "class_fragmentation_summary": class_frag_summary,                                # 사용자 criteria A/B/C aggregate
        "hdbscan_cfg": {
            "MIN_CLUSTER_SIZE": cfg.get("MIN_CLUSTER_SIZE"),
            "MIN_SAMPLES": cfg.get("MIN_SAMPLES"),
            "HDBSCAN_METRIC": cfg.get("HDBSCAN_METRIC"),
            "CLUSTER_SELECTION_METHOD": cfg.get("CLUSTER_SELECTION_METHOD"),
            "CLUSTER_SELECTION_EPSILON": cfg.get("CLUSTER_SELECTION_EPSILON"),
            "ALLOW_SINGLE_CLUSTER": cfg.get("ALLOW_SINGLE_CLUSTER"),
        },
        "hdbscan_overrides": hdbscan_overrides,
        "stress_test": m_stress,
        "stress_test_args": {
            "normal_mult": args.stress_normal_mult,
            "defect_per_class": args.stress_defect_per_class,
        } if args.stress_test else None,
    }
    (eval_dir / "eval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"[write] eval_summary.json")

    # Tier 1 1-2 줄 보고 (docs/contrastive-eval/METRICS.md spec)
    _m = m_without or m_with
    _def_noise = (normal_metrics.get("defect_noise_pct")
                  if isinstance(normal_metrics, dict) else None)
    _def_noise = _def_noise if _def_noise is not None else _m.get("noise_pct", 0.0)
    logger.info(
        f"[Tier 1] Completeness={_m.get('completeness', 0):.4f} "
        f"AMI={_m.get('ami', 0):.4f} "
        f"noise_def={_def_noise:.2f}% "
        f"capture={class_frag_summary.get('n_classes_captured', 0)}/"
        f"{class_frag_summary.get('n_defect_classes', 0)}"
        f"({class_frag_summary.get('class_capture_rate', 0):.4f})"
    )
    logger.info(
        f"[Class frag] coverage={class_frag_summary.get('weighted_cluster_coverage', 0):.4f} "
        f"single_cluster={class_frag_summary.get('n_classes_single_cluster', 0)}/"
        f"{class_frag_summary.get('n_defect_classes', 0)}"
        f"({class_frag_summary.get('frac_single_cluster', 0):.4f}) "
        f"mean_n_clusters={class_frag_summary.get('mean_n_clusters_per_class', 0):.2f}"
    )

    report_dict = {"with-Normal": m_with}
    if m_without is not None: report_dict["without-Normal"] = m_without
    if m_stress is not None: report_dict["stress-test"] = m_stress
    write_per_class_report(report_dict, eval_dir / "per_class_report.txt")
    logger.info(f"[write] per_class_report.txt")

    if not args.no_plots:
        logger.info("[plots] generating")
        plot_cluster_class_heatmap(
            cluster_ids, classes,
            plots_dir / "cluster_class_heatmap_with_normal.png",
            title=f"with-Normal | ARI={m_with['ari']:.3f} NMI={m_with['nmi']:.3f}")
        if m_without is not None:
            plot_cluster_class_heatmap(
                cluster_ids[keep],
                [c for c, k in zip(classes, keep) if k],
                plots_dir / "cluster_class_heatmap_without_normal.png",
                title=f"without-Normal | ARI={m_without['ari']:.3f} "
                      f"NMI={m_without['nmi']:.3f}")
        plot_size_histogram(cluster_ids,
                             plots_dir / "cluster_size_histogram.png")
        plot_silhouette_per_sample(emb, cluster_ids,
                                    plots_dir / "silhouette_per_sample.png")
        loss_records = parse_loss_curves(run_dir / "run.log")
        if loss_records:
            plot_loss_curves(loss_records, plots_dir / "loss_curves_GQL.png")
            logger.info(f"[plots] loss_curves: {len(loss_records)} epochs")
        if m_stress is not None and cl_stress_ids is not None:
            plot_cluster_class_heatmap(
                cl_stress_ids, cls_stress,
                plots_dir / "stress_test_heatmap.png",
                title=f"stress-test | ARI={m_stress['ari']:.3f} "
                      f"NMI={m_stress['nmi']:.3f}")
        logger.info("[plots] done")

    # Final summary log
    logger.info("\n=== EVAL SUMMARY ===")
    logger.info(f"with-Normal:    ARI={m_with['ari']:.4f}  "
                f"NMI={m_with['nmi']:.4f}  "
                f"purity={m_with['cluster_purity']:.4f}  "
                f"silhouette={m_with['silhouette_cosine']}  "
                f"noise={m_with['noise_pct']:.1f}%")
    if m_without is not None:
        logger.info(f"without-Normal: ARI={m_without['ari']:.4f}  "
                    f"NMI={m_without['nmi']:.4f}  "
                    f"purity={m_without['cluster_purity']:.4f}  "
                    f"silhouette={m_without['silhouette_cosine']}  "
                    f"noise={m_without['noise_pct']:.1f}%")
    logger.info(f"normal:         leakage_to_defect={normal_metrics.get('normal_leakage_to_defect_cluster')}%  "
                f"noise={normal_metrics.get('normal_noise_pct')}%")
    if m_stress is not None:
        logger.info(f"stress-test:    ARI={m_stress['ari']:.4f}  "
                    f"NMI={m_stress['nmi']:.4f}  "
                    f"purity={m_stress['cluster_purity']:.4f}  "
                    f"silhouette={m_stress['silhouette_cosine']}  "
                    f"noise={m_stress['noise_pct']:.1f}%")
    logger.info(f"=> {eval_dir}")

    # Rename run_dir + update overall
    output_root = run_dir.parent
    primary = m_without or m_with
    if not args.no_rename:
        new_run_dir = rename_run_dir(run_dir, primary, logger)
    else:
        new_run_dir = run_dir
    if not args.no_overall:
        update_overall(new_run_dir, output_root,
                       m_with, m_without, m_stress, normal_metrics, args, logger)


if __name__ == "__main__":
    main()
