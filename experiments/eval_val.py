#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 4 CLI: val set 기반 cluster 품질 평가.

학습된 모델(`final_infer.pt`) + HDBSCAN clusterer(`clusterer.pkl`) +
val set(`data/wm811k_val/<class>/*.png`) 을 받아:

* val 이미지 → encoder → L2-normalized embedding
* ``hdbscan.approximate_predict`` 로 cluster 할당
* ARI / NMI / cluster_purity / silhouette(cosine) 계산
* ``outputs_<preset>_<ts>/eval_summary.json`` 저장

predict.py 와 동일한 모델 로딩 패턴(``last_training.pt`` 의 CFG 로 override).

실행 예::

    python experiments/eval_val.py \\
        --run-dir outputs_baseline_20260420_120000 \\
        --val-dir data/wm811k_val \\
        --batch 64 --num-workers 4 --device auto

    # Separate val/test outputs without overwriting:
    python experiments/eval_val.py --run-dir <rd> --val-dir data/wm811k_val  \\
        --out <rd>/eval_summary_val.json
    python experiments/eval_val.py --run-dir <rd> --val-dir data/wm811k_test \\
        --out <rd>/eval_summary_test.json
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.eval_metrics import (  # noqa: E402
    compute_cluster_metrics,
    compute_silhouette_cosine,
)


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("eval_val")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
    return logger


def _list_val_items(val_dir: Path) -> Tuple[list[Path], list[str]]:
    paths: list[Path] = []
    labels: list[str] = []
    for cdir in sorted(val_dir.iterdir()):
        if not cdir.is_dir():
            continue
        for p in sorted(cdir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                paths.append(p)
                labels.append(cdir.name)
    return paths, labels


def _resolve_device(dev: str):
    import torch

    if dev == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return dev


def _embed(paths: list[Path], run_dir: Path, batch: int, num_workers: int, device, logger):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    sys.path.insert(0, str(_PROJECT_ROOT))
    import contrastive  # type: ignore

    ckpt_path = run_dir / "checkpoints" / "final_infer.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")

    sibling = run_dir / "checkpoints" / "last_training.pt"
    if sibling.exists():
        side = torch.load(sibling, map_location="cpu", weights_only=False)
        side_cfg = side.get("cfg") if isinstance(side, dict) else None
        if isinstance(side_cfg, dict):
            contrastive.CFG.update(side_cfg)
    contrastive.CFG["LOCAL_BACKBONE_WEIGHTS"] = ""
    contrastive.CFG["FREEZE_BACKBONE"] = True

    model = contrastive.CL(logger=None)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    # NOTE: final_infer.pt saves CL.state_dict() directly; keys already look
    # like "backbone.conv1.weight" / "proj.0.weight". We only need to strip
    # optional DDP "module." prefix, NOT "backbone." (that would break loading).
    sd = contrastive.strip_prefixes(sd, prefixes=("module.", "model."))
    msg = model.load_state_dict(sd, strict=False)
    logger.info(f"loaded model (missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})")
    model.to(device).eval()

    tfm_val = contrastive.tfm(train=False)
    image_size = int(contrastive.CFG.get("IMAGE_SIZE", 224))

    class _DS(Dataset):
        def __init__(self, paths, t):
            self.paths = [str(p) for p in paths]
            self.t = t

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            with Image.open(self.paths[i]) as im:
                im = im.convert("RGB")
                x = self.t(im)
            return x, i

    loader = DataLoader(
        _DS(paths, tfm_val),
        batch_size=batch,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )

    embs = np.zeros((len(paths), int(contrastive.CFG.get("PROJ_DIM", 128))), dtype=np.float32)
    with torch.no_grad():
        for x, idx in loader:
            x = x.to(device, non_blocking=True)
            # CL.forward returns an already-L2-normalized projected embedding.
            feat = model(x)
            feat = torch.nn.functional.normalize(feat, dim=1)
            embs[idx.numpy()] = feat.cpu().numpy()

    return embs


def _assign_clusters(
    embeddings: np.ndarray, clusterer_path: Path, logger
) -> Tuple[np.ndarray, str]:
    """Return (labels, source_tag).

    ``source_tag`` ∈ {``"approximate_predict"``, ``"centroid_nearest"``,
    ``"all_noise_empty_centroids"``}. Callers record this in the eval summary
    so readers can tell which code path produced the metrics. This matters
    because the KEEP centroid filter only affects the centroid-nearest path
    — HDBSCAN's ``approximate_predict`` uses the pre-filter clusterer
    directly (see ``filter_cascaded`` meta field).
    """
    if not clusterer_path.exists():
        raise FileNotFoundError(f"missing clusterer: {clusterer_path}")
    with open(clusterer_path, "rb") as f:
        clusterer = pickle.load(f)
    try:
        import hdbscan

        # HDBSCAN doesn't pre-populate prediction_data_ unless
        # prediction_data=True at fit time; regenerate if missing.
        try:
            _ = clusterer.prediction_data_  # may raise AttributeError
        except AttributeError:
            try:
                clusterer.generate_prediction_data()
                logger.info("regenerated clusterer.prediction_data_ for approximate_predict")
            except Exception as gen_err:
                logger.warning(f"generate_prediction_data failed ({gen_err!r})")
        labels, _strengths = hdbscan.approximate_predict(clusterer, embeddings)
        logger.info(f"hdbscan.approximate_predict used  (n_clusters={len(np.unique(labels[labels!=-1]))})")
        return labels, "approximate_predict"
    except Exception as e:
        logger.warning(f"approximate_predict failed ({e!r}); using centroid-nearest fallback")
        centroids_npy = clusterer_path.parent / "centroids.npy"
        if not centroids_npy.exists():
            raise
        cents = np.load(centroids_npy)
        if cents.size == 0:
            logger.warning(
                "centroids.npy is empty (no kept clusters post-filter); "
                "assigning all val samples to noise (-1)."
            )
            labels = -1 * np.ones(embeddings.shape[0], dtype=np.int64)
            return labels, "all_noise_empty_centroids"
        cents = cents / (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-12)
        embn = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
        sim = embn @ cents.T
        labels = np.argmax(sim, axis=1).astype(np.int64)
        return labels, "centroid_nearest"


def read_kept_cluster_count(centroids_meta_path: Path) -> int:
    """Return ``len(cluster_ids)`` from ``centroids_meta.json``, or 0.

    Missing file / malformed JSON / absent key all collapse to 0 — the whole
    point of this meta field is to document that KEEP filter state in the
    eval summary even when artifacts are incomplete.
    """
    if not centroids_meta_path.exists():
        return 0
    try:
        meta = json.loads(centroids_meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    ids = meta.get("cluster_ids")
    if not isinstance(ids, list):
        return 0
    return len(ids)


def resolve_output_path(run_dir: Path, out: str | Path | None) -> Path:
    """Pick the ``eval_summary.json`` destination.

    * ``out=None`` → legacy default ``<run_dir>/eval_summary.json``.
    * ``out`` pointing at an existing directory (or ending in a separator) →
      write ``eval_summary.json`` inside it.
    * Otherwise ``out`` is treated as a full file path (parent must exist or
      will be created).

    A relative ``out`` is resolved against the current working directory, NOT
    ``run_dir`` — this matches argparse semantics for user-supplied paths.
    """
    run_dir = Path(run_dir)
    if out is None:
        return run_dir / "eval_summary.json"
    out_path = Path(out)
    # Treat as directory if it already exists as one, or if the user passed a
    # trailing separator (PurePath drops it, so check the raw string too).
    raw = str(out)
    looks_like_dir = (
        out_path.is_dir()
        or raw.endswith(("/", "\\"))
    )
    if looks_like_dir:
        return out_path / "eval_summary.json"
    return out_path


def _per_class_assignment(labels: np.ndarray, gt_names: list[str]) -> dict:
    out: dict[str, dict[str, int]] = {}
    for cls in sorted(set(gt_names)):
        out[cls] = {}
    for lbl, cls in zip(labels, gt_names):
        key = f"cluster_{int(lbl)}"
        out[cls][key] = out[cls].get(key, 0) + 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--val-dir", default="data/wm811k_val")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--silhouette-sample-size", type=int, default=5000)
    ap.add_argument("--silhouette-seed", type=int, default=42)
    ap.add_argument(
        "--out",
        default=None,
        help=(
            "Output path for eval_summary.json. Defaults to "
            "<run_dir>/eval_summary.json. Pass a directory to write "
            "<dir>/eval_summary.json, or a full file path (e.g. "
            "<run_dir>/eval_summary_test.json) to separate val/test runs."
        ),
    )
    args = ap.parse_args()

    logger = _setup_logger()
    run_dir = Path(args.run_dir)
    val_dir = Path(args.val_dir)
    if not run_dir.is_dir():
        logger.error(f"run_dir not found: {run_dir}")
        return 1
    if not val_dir.is_dir():
        logger.error(f"val_dir not found: {val_dir}")
        return 1

    paths, gt_names = _list_val_items(val_dir)
    if not paths:
        logger.error(f"no images found under {val_dir}")
        return 1
    logger.info(f"val set: {len(paths)} images across {len(set(gt_names))} classes")

    class_to_idx = {c: i for i, c in enumerate(sorted(set(gt_names)))}
    gt_labels = np.array([class_to_idx[c] for c in gt_names], dtype=np.int64)

    device = _resolve_device(args.device)
    logger.info(f"device: {device}")

    embs = _embed(paths, run_dir, args.batch, args.num_workers, device, logger)
    labels, assign_source = _assign_clusters(
        embs, run_dir / "centroids" / "clusterer.pkl", logger
    )
    approximate_predict_used = assign_source == "approximate_predict"
    kept_cluster_count = read_kept_cluster_count(
        run_dir / "centroids" / "centroids_meta.json"
    )
    # The KEEP filter gates centroids.npy only. When HDBSCAN's
    # approximate_predict succeeds (always preferred), it uses the
    # pre-filter clusterer.pkl directly — so filter state never cascades
    # into the metrics. Design intent: KEEP is for centroid/composite use
    # (predict.py, cluster_composite.py), NOT for eval. Always False today.
    filter_cascaded = False

    metrics = compute_cluster_metrics(labels, gt_labels)
    sil = compute_silhouette_cosine(
        embs, labels,
        sample_size=args.silhouette_sample_size,
        seed=args.silhouette_seed,
    )

    summary = {
        "n_val": int(len(paths)),
        "n_clusters_found": metrics["n_clusters_found"],
        "n_noise": metrics["n_noise"],
        "noise_ratio": metrics["noise_ratio"],
        "ari": metrics["ari"],
        "nmi": metrics["nmi"],
        "cluster_purity": metrics["cluster_purity"],
        "silhouette": sil["silhouette"],
        "silhouette_n_used": sil["n_used"],
        "per_cluster_purity": metrics["per_cluster_purity"],
        "per_class_assignment": _per_class_assignment(labels, gt_names),
        "class_to_idx": class_to_idx,
        "cluster_assignment_source": assign_source,
        "approximate_predict_used": approximate_predict_used,
        "filter_cascaded": filter_cascaded,
        "kept_cluster_count": kept_cluster_count,
    }
    out_path = resolve_output_path(run_dir, args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"wrote {out_path}")
    logger.info(
        f"ARI={summary['ari']:.4f}  NMI={summary['nmi']:.4f}  "
        f"purity={summary['cluster_purity']:.4f}  silhouette={summary['silhouette']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
