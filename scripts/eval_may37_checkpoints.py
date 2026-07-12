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


def calculate_metrics(emb: np.ndarray, labels: list[str], ignored: set[str]) -> dict[str, float | int]:
    labels_array = np.asarray(labels)
    measured = ~np.isin(labels_array, list(ignored))
    measured_labels = labels_array[measured]
    measured_emb = emb[measured]
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=12,
        min_samples=3,
        cluster_selection_method="eom",
        cluster_selection_epsilon=0.0,
        metric="euclidean",
    )
    pred = clusterer.fit_predict(measured_emb)
    keep = pred != -1
    clustered_labels = measured_labels[keep]
    clustered_pred = pred[keep]
    cluster_classes: dict[int, Counter[str]] = defaultdict(Counter)
    for cluster_id, label in zip(clustered_pred.tolist(), clustered_labels.tolist()):
        cluster_classes[int(cluster_id)][str(label)] += 1
    dominant = {
        cluster_id: counts.most_common(1)[0][0]
        for cluster_id, counts in cluster_classes.items()
        if counts
    }
    all_classes = sorted(set(measured_labels.tolist()))
    found = {
        label: any(cluster_label == label for cluster_label in dominant.values())
        for label in all_classes
    }
    class_totals = Counter(measured_labels.tolist())
    coverage = {}
    for label, total in class_totals.items():
        max_in_cluster = max(
            (counts.get(label, 0) for counts in cluster_classes.values()),
            default=0,
        )
        coverage[label] = max_in_cluster / total
    n_clusters = len(dominant)
    can_score = len(clustered_labels) > 1 and len(set(clustered_pred.tolist())) > 1
    if can_score:
        p3 = float(completeness_score(clustered_labels, clustered_pred))
        p4 = float(homogeneity_score(clustered_labels, clustered_pred))
        ami = float(adjusted_mutual_info_score(clustered_labels, clustered_pred))
        ari = float(adjusted_rand_score(clustered_labels, clustered_pred))
        sil = float(silhouette_score(measured_emb[keep], clustered_pred, metric="cosine"))
    else:
        p3 = p4 = ami = ari = sil = 0.0
    n_measured = int(measured.sum())
    noise = int((pred == -1).sum())
    return {
        # Historical P1/capture: each class's largest retained cluster share.
        # It is coverage, not class-discovery count; report both explicitly.
        "P1_cap": round(float(np.mean(list(coverage.values()))), 4) if coverage else 0.0,
        "class_found_rate": round(float(np.mean(list(found.values()))), 4) if found else 0.0,
        "class_found_count": int(sum(found.values())),
        "target_class_count": int(len(all_classes)),
        "P2_noise_pct": round(100.0 * noise / max(1, n_measured), 2),
        "P3_completeness": round(p3, 4),
        "P4_homogeneity": round(p4, 4),
        "AMI": round(ami, 4),
        "ARI": round(ari, 4),
        "Sil_cos": round(sil, 4),
        "k": int(n_clusters),
        "fragment_ratio": round(n_clusters / max(1, len(all_classes)), 4),
        "n_measured": n_measured,
        "noise_count": noise,
    }


def evaluate_checkpoint(trainer, checkpoint_path: Path, eval_root: Path, batch: int) -> dict[str, float | int]:
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
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    paths: list[str] = []
    with torch.no_grad():
        for images, batch_labels, batch_paths in loader:
            images = images.to(device, non_blocking=True)
            embeddings.append(model.infer_embedding(images).float().cpu().numpy())
            labels.extend(batch_labels)
            paths.extend(batch_paths)
    emb = np.concatenate(embeddings, axis=0)
    metrics = calculate_metrics(emb, labels, set(cfg.get("EVAL_IGNORE_CLASSES", [])))
    metrics["checkpoint"] = checkpoint_path.name
    metrics["embedding"] = trainer.INFER_EMBED_MODE
    metrics["hdbscan"] = "eom,mcs=12,ms=3,eps=0.0"
    np.save(checkpoint_path.with_suffix(".projection.npy"), emb)
    checkpoint_path.with_suffix(".paths.json").write_text(
        json.dumps({"paths": paths, "labels": labels}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
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
        metrics = evaluate_checkpoint(trainer, checkpoint_path, args.eval_root.resolve(), args.batch)
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
