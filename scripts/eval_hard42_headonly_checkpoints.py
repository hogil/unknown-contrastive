#!/usr/bin/env python3
"""Evaluate frozen-backbone hard-42 checkpoints in f/z/adapter spaces."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFile
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import _score_umapfree as scorelib  # noqa: E402
from _common import mask_palette_non_grade_to_white  # noqa: E402
from hard42_headonly_common import (  # noqa: E402
    STRICT_EXCLUDED,
    install_head_variant,
    load_trainer,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


class EvalDataset(Dataset):
    def __init__(self, root: Path, image_size: int) -> None:
        self.items = [
            (path, path.parent.name)
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        ]
        norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = T.Compose([T.Resize((image_size, image_size)), T.ToTensor(), norm])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, label = self.items[index]
        with Image.open(path) as image:
            image = mask_palette_non_grade_to_white(image).convert("RGB")
        return self.transform(image), label, str(path)


def parse_csv_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_epochs(value: str, checkpoints: list[Path]) -> list[Path]:
    if value.strip().lower() == "all":
        return checkpoints
    wanted = {int(item.strip()) for item in value.split(",") if item.strip()}
    selected = [path for path in checkpoints if int(path.stem.rsplit("_", 1)[1]) in wanted]
    missing = wanted - {int(path.stem.rsplit("_", 1)[1]) for path in selected}
    if missing:
        raise ValueError(f"requested epochs are unavailable: {sorted(missing)}")
    return selected


def embedding_sha256(embedding: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(embedding).tobytes()).hexdigest()


def retrieval(embedding: np.ndarray, labels: list[str], excluded: set[str]) -> tuple[float, float]:
    labels_array = np.asarray(labels)
    keep = ~np.isin(labels_array, list(excluded))
    vectors = embedding[keep]
    target = labels_array[keep]
    if len(vectors) < 6:
        return 0.0, 0.0
    neighbors = NearestNeighbors(n_neighbors=6, metric="cosine").fit(vectors)
    _, indices = neighbors.kneighbors(vectors)
    same = target[indices[:, 1:]] == target[:, None]
    return float(same[:, 0].mean()), float(same.mean())


def score_embedding(
    embedding: np.ndarray,
    labels: list[str],
    excluded: set[str],
    common: dict,
) -> list[dict]:
    embedding = scorelib.FP.l2(embedding.astype(np.float32, copy=False))
    predictions = scorelib.run_finch(embedding)
    predictions["louvain_res6"] = scorelib.run_louvain(embedding)
    predictions["hdbscan_raw_may"] = scorelib.run_hdbscan_raw(embedding)
    labels_array = np.asarray(labels)
    measured = ~np.isin(labels_array, list(excluded))
    top1, top5 = retrieval(embedding, labels, excluded)
    rows = []
    for method, pred in predictions.items():
        result = scorelib.score(pred, labels, excluded)
        measured_pred = np.asarray(pred)[measured]
        non_noise = measured_pred != -1
        try:
            sil = (
                float(silhouette_score(embedding[measured][non_noise], measured_pred[non_noise], metric="cosine"))
                if non_noise.sum() > 10 and len(set(measured_pred[non_noise].tolist())) > 1
                else None
            )
        except Exception:
            sil = None
        if int(result["capture_count"]) > int(result["k_tot"]):
            raise RuntimeError(
                f"canonical P1 invariant failed: capture={result['capture_count']} k={result['k_tot']}"
            )
        rows.append(
            {
                **common,
                "method": method,
                "P1_capture": float(result["capture"]),
                "P1_capture_count": int(result["capture_count"]),
                "P1_target_class_count": int(result["target_class_count"]),
                "P2_noise_pct": float(result["noise_pct"]),
                "P3_completeness": float(result["completeness"]),
                "P4_homogeneity": float(result["homogeneity"]),
                "ARI": float(result["ari"]),
                "AMI": float(result.get("ami", 0.0)),
                "Sil": sil,
                "k_total": int(result["k_tot"]),
                "k_classes": int(result["n_classes"]),
                "k_noise": int(result["k_noise"]),
                "fragment_ratio": float(result["k_tot"] / max(1, result["n_classes"])),
                "retrieval_top1": top1,
                "retrieval_top5": top5,
                "legacy_presence_count": int(result["legacy_presence_count"]),
                "legacy_presence_rate": float(result["legacy_presence_rate"]),
            }
        )
    return rows


def build_model(trainer, checkpoint: dict):
    cfg = checkpoint["config"]
    backbone_path = Path(cfg["BACKBONE_CKPT"])
    model = trainer.ContrastiveModel(
        cfg["BACKBONE"],
        int(cfg["PROJ_DIM"]),
        True,
        backbone_path,
        use_predictor=str(cfg.get("LOSS_MODE", "nce")).lower() == "simsiam",
        pretrained_backbone=False,
        adapter_dim=int(cfg.get("ADAPTER_DIM", 0)),
        adapter_scale=float(cfg.get("ADAPTER_SCALE", 1.0)),
        spatial_adapter=str(cfg.get("SPATIAL_ADAPTER", "none")),
        spatial_reduction=int(cfg.get("SPATIAL_ADAPTER_REDUCTION", 4)),
        spatial_kernel=int(cfg.get("SPATIAL_ADAPTER_KERNEL", 7)),
        spatial_scale=float(cfg.get("SPATIAL_ADAPTER_SCALE", 1.0)),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model, cfg


def extract(
    model,
    loader: DataLoader,
    device: torch.device,
    modes: set[str],
    concat_weights: list[float],
) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    outputs: dict[str, list[np.ndarray]] = {mode: [] for mode in modes}
    for weight in concat_weights:
        outputs[f"weighted_concat_z{int(round(weight * 100)):03d}"] = []
        if model.adapter is not None:
            outputs[f"adapter_concat_z{int(round(weight * 100)):03d}"] = []
    labels: list[str] = []
    paths: list[str] = []
    with torch.no_grad():
        for batch_index, (images, batch_labels, batch_paths) in enumerate(loader, 1):
            images = images.to(device, non_blocking=True)
            feature_map = model._forward_features(images)
            features = F.normalize(model._pool_features(feature_map), dim=1)
            adapted = F.normalize(model._adapt_features(features), dim=1)
            projected = F.normalize(model.proj(adapted), dim=1)
            current = {"backbone": features, "adapter": adapted, "projection": projected}
            for mode in modes:
                outputs[mode].append(current[mode].float().cpu().numpy())
            for weight in concat_weights:
                suffix = int(round(weight * 100))
                outputs[f"weighted_concat_z{suffix:03d}"].append(
                    F.normalize(torch.cat([features, projected * weight], dim=1), dim=1).float().cpu().numpy()
                )
                if model.adapter is not None:
                    outputs[f"adapter_concat_z{suffix:03d}"].append(
                        F.normalize(torch.cat([adapted, projected * weight], dim=1), dim=1).float().cpu().numpy()
                    )
            labels.extend(batch_labels)
            paths.extend(batch_paths)
            if batch_index % max(1, len(loader) // 10) == 0 or batch_index == len(loader):
                print(f"[embed] {min(batch_index * loader.batch_size, len(loader.dataset))}/{len(loader.dataset)}", flush=True)
    return {name: np.concatenate(chunks, axis=0) for name, chunks in outputs.items()}, labels, paths


def write_report(rows: list[dict], output_dir: Path, scope: str) -> None:
    csv_path = output_dir / f"hard42_{scope}_metrics.csv"
    json_path = output_dir / f"hard42_{scope}_metrics.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    selected = [
        row for row in rows
        if row["method"].startswith("finch_p2(") or row["method"] == "louvain_res6"
    ]
    lines = [
        f"# Hard-42 Head-Only Evaluation: {scope}",
        "",
        "P1 is unique dominant/main-class capture and is asserted not to exceed k.",
        "",
        "| Epoch | Embedding | Clusterer | P1 | P2 | P3 | P4 | ARI | Sil | k/class/noise | Fragment |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        p1 = f"{row['P1_capture_count']}/{row['P1_target_class_count']} ({row['P1_capture']:.3f})"
        sil = "" if row["Sil"] is None else f"{row['Sil']:.3f}"
        lines.append(
            f"| {row['epoch']} | {row['embedding_mode']} | {row['method']} | {p1} | "
            f"{row['P2_noise_pct']:.2f} | {row['P3_completeness']:.3f} | "
            f"{row['P4_homogeneity']:.3f} | {row['ARI']:.3f} | {sil} | "
            f"{row['k_total']}/{row['k_classes']}/{row['k_noise']} | {row['fragment_ratio']:.2f} |"
        )
    (output_dir / f"hard42_{scope}_metrics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--epochs", default="1,2,3,5,8,10")
    parser.add_argument("--modes", default="projection")
    parser.add_argument("--concat-proj-weights", default="")
    parser.add_argument("--exclude-classes", default=",".join(sorted(STRICT_EXCLUDED)))
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    eval_root = args.eval_root.resolve()
    checkpoint_dir = run_dir / "contrastive" / "epoch_checkpoints"
    checkpoints = parse_epochs(args.epochs, sorted(checkpoint_dir.glob("epoch_*.pt")))
    if not checkpoints:
        raise SystemExit(f"no checkpoints selected under {checkpoint_dir}")
    excluded = parse_csv_set(args.exclude_classes)
    modes = parse_csv_set(args.modes)
    if not modes <= {"backbone", "adapter", "projection"}:
        raise ValueError(f"unsupported modes: {sorted(modes)}")
    concat_weights = [float(item) for item in args.concat_proj_weights.split(",") if item.strip()]
    output_dir = run_dir / "contrastive" / "evaluation" / args.scope
    embedding_dir = output_dir / "embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)

    first = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    head_kind = str(first["config"].get("HEAD_KIND", "mlp"))
    adapter_dim = int(first["config"].get("ADAPTER_DIM", 128))
    del first
    trainer = load_trainer(f"hard42_eval_trainer_{os.getpid()}")
    install_head_variant(trainer, head_kind, adapter_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []
    expected_paths: list[str] | None = None
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model, cfg = build_model(trainer, checkpoint)
        model = model.to(device).eval()
        dataset = EvalDataset(eval_root, int(cfg["IMG_SIZE"]))
        loader_kwargs = {
            "batch_size": args.batch,
            "shuffle": False,
            "num_workers": max(0, args.workers),
            "pin_memory": device.type == "cuda",
        }
        if args.workers > 0:
            loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
        loader = DataLoader(dataset, **loader_kwargs)
        epoch = int(checkpoint_path.stem.rsplit("_", 1)[1])
        print(f"[eval] scope={args.scope} epoch={epoch} modes={sorted(modes)}", flush=True)
        embeddings, labels, paths = extract(model, loader, device, modes, concat_weights)
        if expected_paths is None:
            expected_paths = paths
            (output_dir / "eval_inventory.json").write_text(
                json.dumps({"paths": paths, "labels": labels}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        elif paths != expected_paths:
            raise RuntimeError("evaluation image order changed between checkpoints")
        for mode, embedding in embeddings.items():
            embedding_path = embedding_dir / f"epoch_{epoch:03d}_{mode}.npy"
            np.save(embedding_path, embedding.astype(np.float32, copy=False))
            common = {
                "scope": args.scope,
                "epoch": epoch,
                "embedding_mode": mode,
                "embedding_path": str(embedding_path.resolve()),
                "embedding_sha256": embedding_sha256(embedding),
                "checkpoint": str(checkpoint_path.resolve()),
                "head_kind": head_kind,
                "backbone": str(cfg.get("EXPERIMENT_BACKBONE", "")),
                "cell": str(cfg.get("EXPERIMENT_CELL", "")),
                "seed": int(cfg.get("SEED", 0)),
                "exclude_classes": ",".join(sorted(excluded)),
            }
            rows.extend(score_embedding(embedding, labels, excluded, common))
        del model, checkpoint, embeddings
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("evaluation produced no metric rows")
    write_report(rows, output_dir, args.scope)
    completion = {
        "scope": args.scope,
        "run_dir": str(run_dir),
        "eval_root": str(eval_root),
        "epochs": sorted({int(row["epoch"]) for row in rows}),
        "embedding_modes": sorted({str(row["embedding_mode"]) for row in rows}),
        "rows": len(rows),
    }
    (output_dir / "completion.json").write_text(
        json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[OUT] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
