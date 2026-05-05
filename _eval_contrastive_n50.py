#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone eval for contrastive learning runs.

Read: <run_dir>/run_info.json + checkpoints/final_infer.pt + UNKNOWN_DIR
Compute: ARI / NMI / cluster purity / silhouette + per-class noise %
         on 3 sets: with-Normal / without-Normal / stress-test
Write: <run_dir>/eval/{eval_summary.json, per_class_report.txt, plots/, embeddings/}
After eval:
  - rename run_dir → <prev_name>_ari{:.2f}_nmi{:.2f}
  - update <output_root>/overall/ if best ARI (mirror full run + _overall_meta.json)

contrastive.py 무수정 — import only.
"""
import argparse
import json
import logging
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
    silhouette_score,
    silhouette_samples,
)

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--normal-class", default="Normal_bank_boundary")
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
    return ap.parse_args()


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
    import contrastive
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
    import contrastive
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


def metrics_on_set(emb_sub, gt_str_sub, cluster_ids_sub, label, logger):
    cls_uniq = sorted(set(gt_str_sub))
    cls_to_idx = {c: i for i, c in enumerate(cls_uniq)}
    gt_idx = np.array([cls_to_idx[c] for c in gt_str_sub])
    n = len(gt_str_sub)
    n_noise = int((cluster_ids_sub == -1).sum())
    n_clusters = len(set(int(x) for x in cluster_ids_sub if x >= 0))
    ari = float(adjusted_rand_score(gt_idx, cluster_ids_sub))
    nmi = float(normalized_mutual_info_score(gt_idx, cluster_ids_sub))
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
        "ari": ari, "nmi": nmi, "cluster_purity": purity,
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
                   m_without: dict, m_stress: dict, args, logger):
    """If current ARI > overall best, mirror run_dir to overall/."""
    overall_dir = output_root / "overall"
    overall_meta = output_root / "overall" / "_overall_meta.json"
    cur_ari = m_with["ari"]
    prev_ari = -1.0
    if overall_meta.exists():
        try:
            prev_ari = float(json.loads(
                overall_meta.read_text(encoding="utf-8")).get("ari", -1.0))
        except Exception:
            prev_ari = -1.0
    if cur_ari > prev_ari:
        logger.info(f"[overall] new best ARI: {cur_ari:.4f} > {prev_ari:.4f} "
                    f"— mirror to {overall_dir}")
        if overall_dir.exists():
            shutil.rmtree(overall_dir)
        # Mirror full run_dir (includes eval/, embeddings/, plots/)
        shutil.copytree(run_dir, overall_dir)
        meta = {
            "source_run": run_dir.name,
            "ari": m_with["ari"], "nmi": m_with["nmi"],
            "cluster_purity": m_with["cluster_purity"],
            "silhouette_cosine": m_with["silhouette_cosine"],
            "noise_pct": m_with["noise_pct"],
            "ari_without_normal": m_without["ari"] if m_without else None,
            "nmi_without_normal": m_without["nmi"] if m_without else None,
            "ari_stress": m_stress["ari"] if m_stress else None,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        overall_meta.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        logger.info(f"[overall] not best (cur={cur_ari:.4f} <= "
                    f"prev={prev_ari:.4f}) — overall/ unchanged")


def rename_run_dir(run_dir: Path, m_with: dict, logger) -> Path:
    """Append _ari{:.2f}_nmi{:.2f} to run_dir name."""
    suffix = f"_ari{m_with['ari']:.2f}_nmi{m_with['nmi']:.2f}"
    if run_dir.name.endswith(suffix):
        return run_dir
    new_name = run_dir.name + suffix
    new_dir = run_dir.parent / new_name
    if new_dir.exists():
        logger.warning(f"[rename] target exists, skipping: {new_dir}")
        return run_dir
    run_dir.rename(new_dir)
    logger.info(f"[rename] {run_dir.name} → {new_dir.name}")
    return new_dir


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        sys.exit(f"[err] run_dir not found: {run_dir}")

    eval_dir = run_dir / "eval"
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
        "ts": datetime.now().isoformat(timespec="seconds"),
        "n_total": len(files), "n_classes": len(set(classes)),
        "with_normal": m_with, "without_normal": m_without,
        "stress_test": m_stress,
        "stress_test_args": {
            "normal_mult": args.stress_normal_mult,
            "defect_per_class": args.stress_defect_per_class,
        } if args.stress_test else None,
    }
    (eval_dir / "eval_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"[write] eval_summary.json")

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
    if m_stress is not None:
        logger.info(f"stress-test:    ARI={m_stress['ari']:.4f}  "
                    f"NMI={m_stress['nmi']:.4f}  "
                    f"purity={m_stress['cluster_purity']:.4f}  "
                    f"silhouette={m_stress['silhouette_cosine']}  "
                    f"noise={m_stress['noise_pct']:.1f}%")
    logger.info(f"=> {eval_dir}")

    # Rename run_dir + update overall
    output_root = run_dir.parent
    if not args.no_rename:
        new_run_dir = rename_run_dir(run_dir, m_with, logger)
    else:
        new_run_dir = run_dir
    if not args.no_overall:
        update_overall(new_run_dir, output_root,
                       m_with, m_without, m_stress, args, logger)


if __name__ == "__main__":
    main()
