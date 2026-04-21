#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Inference CLI for the unknown-contrastive project (contrastive_pred.py).

Loads a trained checkpoint (``final_infer.pt``) + centroids produced by
``contrastive.py`` / ``contrastive_com.py`` / ``contrastive_rev_com.py``,
re-uses the transform and model from ``contrastive.py``, embeds every
image under ``--images-dir`` and routes it to ``cluster_XXX/`` (or
``unknown/``) under ``--out-dir``.

Example::

    python contrastive_pred.py \\
        --checkpoint runs/foo/checkpoints/final_infer.pt \\
        --centroids-dir runs/foo/centroids \\
        --images-dir data/incoming \\
        --out-dir runs/foo/predictions \\
        --threshold-mode p95 \\
        --batch 128 --num-workers 8 --device auto

``--help`` works without any dataset present.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Tuple

# --- project root on sys.path so `common.*` and `contrastive` import cleanly
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import numpy as np


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("predict")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
    return logger


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------
def _pick_thresholds(
    meta: dict,
    cluster_ids: list,
    mode: str,
    logger: logging.Logger,
) -> np.ndarray:
    """Build a (K,) threshold vector from the metadata percentiles."""
    dist_pcts = meta.get("dist_percentiles", {})
    key = mode.lower()
    if key.startswith("p"):
        thresholds = []
        missing = 0
        for cid in cluster_ids:
            pcts = dist_pcts.get(str(cid), {})
            if key in pcts:
                thresholds.append(float(pcts[key]))
            else:
                missing += 1
                thresholds.append(float("inf"))
        if missing:
            logger.warning(
                f"[Threshold] {missing}/{len(cluster_ids)} cluster(s) missing '{key}', using inf for them"
            )
        return np.asarray(thresholds, dtype=np.float32)
    raise ValueError(f"unsupported threshold mode: {mode}")


# ---------------------------------------------------------------------------
# File IO helpers (mirrors contrastive.py::link_or_copy)
# ---------------------------------------------------------------------------
def _link_or_copy(src: Path, dst: Path, logger: logging.Logger) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "hardlink"
    except Exception as e:
        logger.debug(f"hardlink failed ({e!r}); falling back to copy")
    try:
        shutil.copy2(src, dst)
        return "copy"
    except shutil.SameFileError:
        return "same"


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _collect_images(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"images-dir does not exist: {root}")
    if root.is_file():
        return [root] if root.suffix.lower() in _IMAGE_EXTS else []
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
            files.append(p)
    files.sort()
    return files


def _unique_dst_name(dst_dir: Path, src: Path) -> Path:
    """Avoid overwriting on basename collisions."""
    candidate = dst_dir / src.name
    if not candidate.exists():
        return candidate
    stem, suf = src.stem, src.suffix
    i = 1
    while True:
        c = dst_dir / f"{stem}__{i}{suf}"
        if not c.exists():
            return c
        i += 1


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _resolve_device(flag: str) -> "torch.device":
    import torch

    if flag == "cpu":
        return torch.device("cpu")
    if flag == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model(checkpoint_path: Path, device, logger: logging.Logger):
    """Restore the trained CL model.

    Strategy:
      1. Try to read ``last_training.pt`` (beside ``final_infer.pt``) — it
         carries the original ``CFG`` which we use to override the defaults
         in ``contrastive.py`` before instantiating ``CL``.
      2. Fall back to ``contrastive.CFG`` as-is.
    """
    import torch

    # Import after sys.path is set up.
    import contrastive  # type: ignore

    sibling_cfg_path = checkpoint_path.parent / "last_training.pt"
    cfg_source = "default (contrastive.CFG)"
    if sibling_cfg_path.exists():
        try:
            side = torch.load(sibling_cfg_path, map_location="cpu", weights_only=False)
            side_cfg = side.get("cfg") if isinstance(side, dict) else None
            if isinstance(side_cfg, dict):
                contrastive.CFG.update(side_cfg)
                cfg_source = f"{sibling_cfg_path.name}"
        except Exception as e:
            logger.warning(f"could not read sibling cfg from {sibling_cfg_path}: {e!r}")

    # Never try to load training-only backbone weights during inference —
    # the checkpoint itself supplies them.
    contrastive.CFG["LOCAL_BACKBONE_WEIGHTS"] = ""
    contrastive.CFG["FREEZE_BACKBONE"] = True
    logger.info(f"[Model] CFG source: {cfg_source}")
    logger.info(
        f"[Model] backbone={contrastive.CFG['BACKBONE_NAME']}  "
        f"image_size={contrastive.CFG['IMAGE_SIZE']}  proj_dim={contrastive.CFG['PROJ_DIM']}"
    )

    model = contrastive.CL(logger=None)

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover — older torch without weights_only
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = None
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    if not isinstance(state_dict, dict):
        raise ValueError(f"cannot find state_dict inside {checkpoint_path}")
    state_dict = contrastive.strip_prefixes(state_dict)

    msg = model.load_state_dict(state_dict, strict=False)
    logger.info(
        f"[Model] loaded weights from {checkpoint_path}  "
        f"(missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})"
    )
    if msg.missing_keys:
        logger.info(f"[Model] sample missing keys: {msg.missing_keys[:5]}")
    if msg.unexpected_keys:
        logger.info(f"[Model] sample unexpected keys: {msg.unexpected_keys[:5]}")

    model.to(device)
    model.eval()
    return model, contrastive


# ---------------------------------------------------------------------------
# Embedding dataset
# ---------------------------------------------------------------------------
def _build_loader(paths: List[Path], tfm_fn, batch: int, num_workers: int, device):
    import torch
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    class _PredictDS(Dataset):
        def __init__(self, paths, t):
            self.paths = [str(p) for p in paths]
            self.t = t

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            p = self.paths[i]
            try:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    x = self.t(im)
                ok = True
                err = ""
            except Exception as e:
                x = torch.zeros(3, 224, 224)  # placeholder; corrected below
                # We cannot know final image_size here; defer: caller reshapes.
                ok = False
                err = f"{type(e).__name__}: {e}"
            return x, i, p, int(ok), err

    ds = _PredictDS(paths, tfm_fn)
    kw: dict = {}
    if num_workers > 0:
        kw = dict(num_workers=num_workers, persistent_workers=False)
    ld = DataLoader(
        ds,
        batch_size=batch,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        **kw,
    )
    return ld


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
def run_prediction(args, logger: logging.Logger) -> int:
    import torch
    import torch.nn.functional as F

    checkpoint = Path(args.checkpoint).resolve()
    centroids_dir = Path(args.centroids_dir).resolve()
    images_dir = Path(args.images_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint.exists():
        logger.error(f"checkpoint not found: {checkpoint}")
        return 2
    if not centroids_dir.exists():
        logger.error(f"centroids-dir not found: {centroids_dir}")
        return 2

    # --- import project modules lazily so `--help` never requires torch ---
    from common.centroids import load_cluster_centroids, assign_to_cluster

    pack = load_cluster_centroids(centroids_dir)
    centroids = pack["centroids"]
    cluster_ids = pack["cluster_ids"]
    meta = pack["meta"]
    clusterer = pack["clusterer"]
    logger.info(
        f"[Centroids] {centroids.shape[0]} cluster(s) loaded from {centroids_dir}"
    )

    # --- threshold mode ---
    threshold_mode = args.threshold_mode
    use_hdbscan = threshold_mode == "hdbscan"
    if use_hdbscan:
        try:
            import hdbscan  # noqa: F401
        except Exception as e:
            logger.error(
                f"--threshold-mode hdbscan requires the 'hdbscan' package "
                f"(import failed: {e!r}). Falling back to p95."
            )
            use_hdbscan = False
            threshold_mode = "p95"
        else:
            if clusterer is None:
                logger.error(
                    "clusterer.pkl not found in centroids-dir; falling back to p95"
                )
                use_hdbscan = False
                threshold_mode = "p95"

    thresholds = _pick_thresholds(meta, cluster_ids, threshold_mode, logger) \
        if not use_hdbscan else np.full((len(cluster_ids),), np.inf, dtype=np.float32)
    logger.info(
        f"[Threshold] mode={'hdbscan' if use_hdbscan else threshold_mode}  "
        f"values: {[round(float(t), 4) for t in thresholds]}"
    )

    # --- device + model ---
    device = _resolve_device(args.device)
    logger.info(f"[Device] {device}")
    model, contrastive_mod = _load_model(checkpoint, device, logger)
    tfm_fn = contrastive_mod.tfm(train=False)

    # --- collect images ---
    images = _collect_images(images_dir)
    logger.info(f"[Images] found {len(images)} image(s) under {images_dir}")
    if not images:
        logger.warning("no images to process — nothing to do")
        with (out_dir / "prediction_report.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "checkpoint": str(checkpoint),
                    "centroids_dir": str(centroids_dir),
                    "images_dir": str(images_dir),
                    "threshold_mode": threshold_mode,
                    "total": 0,
                    "assignments": [],
                    "_errors": [],
                },
                f, indent=2, ensure_ascii=False,
            )
        return 0

    # --- embed ---
    loader = _build_loader(images, tfm_fn, args.batch, args.num_workers, device)

    total = len(images)
    next_pct_tick = 10
    all_emb: List[np.ndarray] = []
    all_paths: List[str] = []
    errors: List[dict] = []

    start = time.time()
    processed = 0

    with torch.no_grad():
        for batch_x, idx_b, path_b, ok_b, err_b in loader:
            ok_mask = [bool(int(o)) for o in ok_b]
            good_idx_in_batch = [j for j, ok in enumerate(ok_mask) if ok]
            bad_idx_in_batch = [j for j, ok in enumerate(ok_mask) if not ok]

            for j in bad_idx_in_batch:
                errors.append({"path": str(path_b[j]), "error": str(err_b[j])})

            if good_idx_in_batch:
                xb = batch_x[good_idx_in_batch].to(device, non_blocking=True)
                try:
                    xb = xb.to(memory_format=torch.channels_last)
                except Exception:
                    pass
                z = model(xb)                           # already L2-normed by CL
                z = F.normalize(z, dim=1)               # defensive re-normalize
                emb_np = z.float().cpu().numpy()
                all_emb.append(emb_np)
                for j in good_idx_in_batch:
                    all_paths.append(str(path_b[j]))

            processed += len(ok_mask)
            pct = 100.0 * processed / max(1, total)
            if pct >= next_pct_tick or processed == total:
                elapsed = time.time() - start
                logger.info(
                    f"[Predict] {processed}/{total} ({pct:.1f}%) "
                    f"elapsed={elapsed:.1f}s"
                )
                while next_pct_tick <= pct:
                    next_pct_tick += 10

    if all_emb:
        emb_all = np.concatenate(all_emb, axis=0).astype(np.float32)
    else:
        D = int(meta.get("embedding_dim", centroids.shape[1] if centroids.size else 128))
        emb_all = np.zeros((0, D), dtype=np.float32)

    # --- assign ---
    if use_hdbscan:
        import hdbscan

        logger.info("[Assign] using hdbscan.approximate_predict")
        labels, probs = hdbscan.approximate_predict(clusterer, emb_all)
        assigned = np.asarray(labels, dtype=np.int64)
        # nearest + distance still useful for reporting
        if centroids.shape[0] > 0 and emb_all.shape[0] > 0:
            sims = emb_all @ centroids.T
            dists = 1.0 - sims
            nn_idx = np.argmin(dists, axis=1)
            nearest_dist = dists[np.arange(emb_all.shape[0]), nn_idx].astype(np.float32)
        else:
            nn_idx = np.full((emb_all.shape[0],), -1, dtype=np.int64)
            nearest_dist = np.full((emb_all.shape[0],), float("inf"), dtype=np.float32)
    else:
        assigned, nn_idx, nearest_dist = assign_to_cluster(
            emb_all, centroids, cluster_ids, thresholds
        )

    # --- place files ---
    unknown_dir = out_dir / "unknown"
    assignments_report: List[dict] = []
    cluster_sizes: dict = {}
    unknown_count = 0

    for i, src_str in enumerate(all_paths):
        src = Path(src_str)
        cid = int(assigned[i])
        dist = float(nearest_dist[i]) if np.isfinite(nearest_dist[i]) else None

        if cid == -1:
            unknown_count += 1
            dst_dir = unknown_dir
            cluster_folder = "unknown"
        else:
            cluster_folder = f"cluster_{cid:03d}"
            dst_dir = out_dir / cluster_folder
            cluster_sizes[cluster_folder] = cluster_sizes.get(cluster_folder, 0) + 1

        dst = _unique_dst_name(dst_dir, src)
        try:
            placement = _link_or_copy(src, dst, logger)
        except Exception as e:
            placement = "failed"
            errors.append({"path": src_str, "error": f"place failed: {e!r}"})
            dst = None

        assignments_report.append({
            "source": src_str,
            "cluster": cluster_folder,
            "cluster_id": cid,
            "distance": dist,
            "nearest_centroid_idx": int(nn_idx[i]) if int(nn_idx[i]) >= 0 else None,
            "dest": str(dst) if dst is not None else None,
            "placement": placement,
        })

    # --- reports ---
    report = {
        "checkpoint": str(checkpoint),
        "centroids_dir": str(centroids_dir),
        "images_dir": str(images_dir),
        "out_dir": str(out_dir),
        "threshold_mode": "hdbscan" if use_hdbscan else threshold_mode,
        "thresholds": [float(t) for t in thresholds],
        "cluster_ids": cluster_ids,
        "total": total,
        "processed_ok": len(all_paths),
        "unknown_count": unknown_count,
        "cluster_sizes": cluster_sizes,
        "assignments": assignments_report,
        "_errors": errors,
    }
    with (out_dir / "prediction_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with (out_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"total images           : {total}\n")
        f.write(f"processed successfully : {len(all_paths)}\n")
        f.write(f"load errors            : {len(errors)}\n")
        f.write(f"unknown                : {unknown_count}\n")
        f.write(f"threshold mode         : {'hdbscan' if use_hdbscan else threshold_mode}\n")
        f.write("\n-- cluster sizes --\n")
        for name in sorted(cluster_sizes):
            f.write(f"  {name}: {cluster_sizes[name]}\n")

    elapsed = time.time() - start
    logger.info(
        f"[Done] {len(all_paths)} assigned, {unknown_count} unknown, "
        f"{len(errors)} error(s) in {elapsed:.1f}s -> {out_dir}"
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="predict.py",
        description="Cluster-route new images using a trained contrastive model.",
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to final_infer.pt (the checkpoint saved by contrastive.py).")
    p.add_argument("--centroids-dir", required=True,
                   help="Directory containing centroids.npy + centroids_meta.json "
                        "(written by common.centroids.save_cluster_centroids).")
    p.add_argument("--images-dir", required=True,
                   help="Root directory to scan recursively for images.")
    p.add_argument("--out-dir", required=True,
                   help="Where cluster_XXX/ and unknown/ folders are written.")
    p.add_argument("--threshold-mode", default="p95",
                   choices=["p50", "p90", "p95", "p99", "hdbscan"],
                   help="Which per-cluster distance threshold to use (default: p95). "
                        "'hdbscan' uses clusterer.pkl and approximate_predict.")
    p.add_argument("--batch", type=int, default=128, help="Inference batch size.")
    p.add_argument("--num-workers", type=int, default=8, help="DataLoader workers.")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="Device selection (default: auto).")
    return p


def main(argv=None) -> int:
    logger = _setup_logger()
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return run_prediction(args, logger)
    except Exception as e:
        logger.exception(f"prediction failed: {e!r}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
