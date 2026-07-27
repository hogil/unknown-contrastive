#!/usr/bin/env python3
"""Evaluate May-2026 contrastive checkpoints with one fixed HDBSCAN protocol.

This evaluator is deliberately separate from the evolving trainer.  It makes
the historical-equivalent ladder comparable at every epoch:
projection embedding, Normal/R/Random excluded, and HDBSCAN eom/mcs=12/ms=3.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import hdbscan
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageFile
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    silhouette_score,
)
from torch.utils.data import DataLoader, Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from _common import mask_palette_non_grade_to_white  # noqa: E402
from cluster_metrics import capture_metrics  # noqa: E402


class EvalDataset(Dataset):
    def __init__(self, root: Path, image_size: int) -> None:
        self.root = root
        self.items: list[tuple[Path, str]] = []
        for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            for path in sorted(class_dir.rglob("*")):
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                    self.items.append((path, class_dir.name))
        norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), norm])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, label = self.items[index]
        try:
            with Image.open(path) as image:
                image = mask_palette_non_grade_to_white(image).convert("RGB")
        except Exception:
            image = Image.new("RGB", (self.transform.transforms[0].size[0], self.transform.transforms[0].size[1]))
        return self.transform(image), label, str(path)


def load_trainer_module():
    path = SCRIPT_DIR / "train_contrastive_ddp.py"
    spec = importlib.util.spec_from_file_location("may37_trainer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _clusterer() -> hdbscan.HDBSCAN:
    return hdbscan.HDBSCAN(
        min_cluster_size=12,
        min_samples=15,
        cluster_selection_method="leaf",
        cluster_selection_epsilon=0.06,
        metric="euclidean",
    )


def _summarize_predictions(
    emb: np.ndarray,
    labels: np.ndarray,
    pred: np.ndarray,
    capture_pred: np.ndarray,
    capture_labels: np.ndarray,
    ignored: set[str],
    scope: str,
) -> dict[str, float | int]:
    keep = pred != -1
    clustered_labels = labels[keep]
    clustered_pred = pred[keep]
    capture = capture_metrics(capture_pred, capture_labels, ignored)
    n_clusters = int(capture["cluster_count"])
    can_score = len(clustered_labels) > 1 and len(set(clustered_pred.tolist())) > 1
    if can_score:
        p3 = float(completeness_score(clustered_labels, clustered_pred))
        p4 = float(homogeneity_score(clustered_labels, clustered_pred))
        ami = float(adjusted_mutual_info_score(clustered_labels, clustered_pred))
        ari = float(adjusted_rand_score(clustered_labels, clustered_pred))
        sil = float(silhouette_score(emb[keep], clustered_pred, metric="cosine"))
    else:
        p3 = p4 = ami = ari = sil = 0.0
    n_measured = int(len(labels))
    noise = int((pred == -1).sum())
    capture_count = int(capture["capture_count"])
    target_class_count = int(capture["target_class_count"])
    dominant_cluster_count = int(capture["dominant_cluster_count"])
    if capture_count > dominant_cluster_count or dominant_cluster_count > n_clusters:
        raise RuntimeError("P1 capture count cannot exceed k")
    return {
        "P1_capture": f"{capture_count}/{target_class_count}",
        "P1_cap": round(float(capture["capture_rate"]), 4),
        "class_found_count": capture_count,
        "target_class_count": target_class_count,
        "k": n_clusters,
        "dominant_cluster_count": dominant_cluster_count,
        "legacy_presence_capture": f"{capture['legacy_presence_count']}/{target_class_count}",
        "legacy_presence_rate": round(float(capture["legacy_presence_rate"]), 4),
        "class_coverage": round(float(capture["class_coverage_rate"]), 4),
        "P2_noise_pct": round(100.0 * noise / max(1, n_measured), 2),
        "P3_completeness": round(p3, 4),
        "P4_homogeneity": round(p4, 4),
        "AMI": round(ami, 4),
        "ARI": round(ari, 4),
        "Sil_cos": round(sil, 4),
        "fragment_ratio": round(n_clusters / max(1, capture["target_class_count"]), 4),
        "n_measured": n_measured,
        "noise_count": noise,
        "scope": scope,
    }


def _expand_ignored(labels_array: np.ndarray, ignored: set[str]) -> set[str]:
    """★ 260721: Full_*/Normal* 도 배경 제외 (사용자: normal/full 은 채점에서 뺀다)."""
    extra = {c for c in set(labels_array.tolist()) if c.startswith(("Full", "Normal"))}
    return set(ignored) | extra


def _drop_megaclusters(pred: np.ndarray, frac: float = 0.20) -> np.ndarray:
    """★ 260721: 전체의 frac(20%) 이상인 단일 클러스터는 garbage(Normal/Full 뭉침)로 보고 noise(-1) 처리."""
    pred = pred.copy()
    n = len(pred)
    for c in set(pred.tolist()):
        if c != -1 and (pred == c).sum() >= frac * n:
            pred[pred == c] = -1
    return pred


def calculate_metrics(emb: np.ndarray, labels: list[str], ignored: set[str]) -> dict[str, float | int]:
    """Operational metric: cluster the full pool, then exclude background targets."""
    labels_array = np.asarray(labels)
    ignored = _expand_ignored(labels_array, ignored)          # ★ Full/Normal 제외
    measured = ~np.isin(labels_array, list(ignored))
    full_pred = _clusterer().fit_predict(emb)
    full_pred = _drop_megaclusters(full_pred, frac=0.20)      # ★ 20% 초대형 클러스터 제거
    return _summarize_predictions(
        emb[measured],
        labels_array[measured],
        full_pred[measured],
        full_pred,
        labels_array,
        ignored,
        "full_pool_canonical",
    )


def calculate_defect_only_metrics(
    emb: np.ndarray, labels: list[str], ignored: set[str]
) -> dict[str, float | int]:
    """May Tier1 geometry: remove background before fitting HDBSCAN."""
    labels_array = np.asarray(labels)
    ignored = _expand_ignored(labels_array, ignored)          # ★ Full/Normal 제외
    measured = ~np.isin(labels_array, list(ignored))
    defect_emb = emb[measured]
    defect_labels = labels_array[measured]
    pred = _clusterer().fit_predict(defect_emb)
    return _summarize_predictions(
        defect_emb,
        defect_labels,
        pred,
        pred,
        defect_labels,
        set(),
        "defect_only_tier1",
    )


def evaluate_checkpoint(trainer, checkpoint_path: Path, eval_root: Path, batch: int, workers: int) -> dict[str, float | int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    for key, value in cfg.items():
        if key.isupper() and hasattr(trainer, key):
            setattr(trainer, key, value)
    trainer.INFER_EMBED_MODE = str(cfg.get("INFER_EMBED_MODE", "projection"))
    trainer.FREEZE_BACKBONE = True
    backbone_ckpt = Path(cfg["BACKBONE_CKPT"]) if cfg.get("BACKBONE_CKPT") else None
    if backbone_ckpt is not None and not backbone_ckpt.exists():
        raise FileNotFoundError(f"checkpoint backbone is unavailable: {backbone_ckpt}")
    model = trainer.ContrastiveModel(
        cfg["BACKBONE"],
        int(cfg["PROJ_DIM"]),
        True,
        backbone_ckpt,
        use_predictor=False,
        pretrained_backbone=False,
        adapter_dim=int(cfg.get("ADAPTER_DIM", 0)),
        adapter_scale=float(cfg.get("ADAPTER_SCALE", 1.0)),
        spatial_adapter=str(cfg.get("SPATIAL_ADAPTER", "none")),
        spatial_reduction=int(cfg.get("SPATIAL_ADAPTER_REDUCTION", 4)),
        spatial_kernel=int(cfg.get("SPATIAL_ADAPTER_KERNEL", 7)),
        spatial_scale=float(cfg.get("SPATIAL_ADAPTER_SCALE", 1.0)),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    dataset = EvalDataset(eval_root, int(cfg["IMG_SIZE"]))
    loader_kwargs = {
        "batch_size": batch,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(dataset, **loader_kwargs)
    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    paths: list[str] = []
    with torch.no_grad():
        for batch_index, (images, batch_labels, batch_paths) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            embeddings.append(model.infer_embedding(images).float().cpu().numpy())
            labels.extend(batch_labels)
            paths.extend(batch_paths)
            if batch_index % max(1, len(loader) // 10) == 0 or batch_index == len(loader):
                print(f"[embed] {min(batch_index * batch, len(dataset))}/{len(dataset)}", flush=True)
    emb = np.concatenate(embeddings, axis=0)
    np.save(checkpoint_path.with_suffix(f".{trainer.INFER_EMBED_MODE}.npy"), emb)
    checkpoint_path.with_suffix(".paths.json").write_text(
        json.dumps({"paths": paths, "labels": labels}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics = calculate_metrics(emb, labels, set(cfg.get("EVAL_IGNORE_CLASSES", [])))
    metrics["checkpoint"] = checkpoint_path.name
    metrics["embedding"] = trainer.INFER_EMBED_MODE
    metrics["hdbscan"] = "eom,mcs=12,ms=3,eps=0.0"
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    checkpoint_dir = run_dir / "contrastive" / "epoch_checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if not checkpoints:
        raise SystemExit(f"no epoch checkpoints found: {checkpoint_dir}")
    trainer = load_trainer_module()
    rows = []
    for checkpoint_path in checkpoints:
        print(f"[eval] {checkpoint_path.name}", flush=True)
        metrics = evaluate_checkpoint(trainer, checkpoint_path, args.eval_root.resolve(), args.batch, max(0, args.workers))
        metrics["epoch"] = int(checkpoint_path.stem.rsplit("_", 1)[1])
        rows.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
    rows.sort(key=lambda row: int(row["epoch"]))
    out_json = run_dir / "contrastive" / "may37_epoch_metrics.json"
    out_csv = run_dir / "contrastive" / "may37_epoch_metrics.csv"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OUT] {out_json}")
    print(f"[OUT] {out_csv}")


if __name__ == "__main__":
    main()
