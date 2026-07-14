#!/usr/bin/env python3
"""Run one manifest-locked May source-protocol B0-B6 control cell.

The documented May ablation used the trainer stored in git commit
``b796ecbe5f70c9b88944480292e12706b64db83b``.  That loss implementation is
not the current DDP trainer, so this runner materializes the archived source
and adds checkpoint-only epoch instrumentation without changing its losses.
The original May file list is no
longer available, so each control anchor is manifest-locked and reported as a
paired re-reproduction rather than an exact historical-score reproduction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from hard42_headonly_common import audit_backbone_checkpoint, sha256_file
from eval_may37_checkpoints import calculate_defect_only_metrics, calculate_metrics


ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "b796ecbe5f70c9b88944480292e12706b64db83b"
SOURCE_FILES = ("contrastive.py", "run_contrastive.py")
TAPT_CKPT = Path(r"D:\project\known-cnn\models\iter116J_frozen\best_model.pth")
FCMAE_CKPT = ROOT / "weights" / "convnextv2_base.fcmae_ft_in22k_in1k_384.pth"

CELLS = {
    "FROZEN": {"local": False, "local_weight": 0.0, "queue": False, "ignore": 1.0, "neco": 0.0},
    "PCA128": {"local": False, "local_weight": 0.0, "queue": False, "ignore": 1.0, "neco": 0.0},
    "RANDOM128": {"local": False, "local_weight": 0.0, "queue": False, "ignore": 1.0, "neco": 0.0},
    "B0": {"local": False, "local_weight": 0.0, "queue": False, "ignore": 1.0, "neco": 0.0},
    "B1": {"local": True, "local_weight": 0.5, "queue": False, "ignore": 1.0, "neco": 0.0},
    "B2": {"local": True, "local_weight": 1.0, "queue": False, "ignore": 1.0, "neco": 0.0},
    "B3": {"local": True, "local_weight": 1.0, "queue": True, "ignore": 1.0, "neco": 0.0},
    "B4": {"local": True, "local_weight": 1.0, "queue": True, "ignore": 0.72, "neco": 0.0},
    "B5": {"local": True, "local_weight": 1.0, "queue": True, "ignore": 0.72, "neco": 0.2},
    # May's actual best component recipe: NeCo replaces Local rather than
    # being added on top of it (historical iter 70 / B6).
    "B6": {"local": False, "local_weight": 0.0, "queue": True, "ignore": 0.72, "neco": 0.2},
}
DIAGNOSTIC_CELLS = {"FROZEN", "PCA128", "RANDOM128"}
IGNORED = {"Normal", "Random", "R"}


def git_source(name: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{SOURCE_COMMIT}:{name}"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
    )
    return result.stdout


def instrument_epoch_checkpoints(text: str) -> str:
    """Add state-only epoch snapshots after the archived optimizer step loop."""
    needle = (
        '        logger.info(f"Epoch {ep} done | G={run_g/max(1,n):.4f} '
        'Q={run_q/max(1,n):.4f} L={run_l/max(1,n):.4f} | time={time.time()-t0:.1f}s")\n'
    )
    if text.count(needle) != 1:
        raise RuntimeError("archived epoch-end marker changed; refusing ambiguous instrumentation")
    checkpoint_lines = (
        needle
        + '        _epoch_dir = Path(run_dir) / "checkpoints" / "epoch_checkpoints"\n'
        + '        _epoch_dir.mkdir(parents=True, exist_ok=True)\n'
        + '        torch.save({"state_dict": model.state_dict(), "epoch": ep}, '
        + '_epoch_dir / f"epoch_{ep:03d}.pt")\n'
    )
    return text.replace(needle, checkpoint_lines)


def materialize_source(output_root: Path) -> Path:
    source_dir = output_root / f"_source_{SOURCE_COMMIT[:12]}"
    source_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in SOURCE_FILES:
        original = git_source(name)
        text = instrument_epoch_checkpoints(original) if name == "contrastive.py" else original
        target = source_dir / name
        if not target.exists() or target.read_text(encoding="utf-8") != text:
            target.write_text(text, encoding="utf-8")
        hashes[name] = {
            "archived_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
            "materialized_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "instrumented": text != original,
        }
    (source_dir / "provenance.json").write_text(
        json.dumps({"git_commit": SOURCE_COMMIT, "sha256": hashes}, indent=2), encoding="utf-8"
    )
    return source_dir


def write_anchor_manifest(anchor: Path, output_root: Path) -> dict:
    """Lock the exact current control anchor without claiming it is May's lost list."""
    inventory: list[dict[str, int | str]] = []
    class_counts: Counter[str] = Counter()
    for class_dir in sorted(path for path in anchor.iterdir() if path.is_dir()):
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            inventory.append(
                {
                    "path": path.relative_to(anchor).as_posix(),
                    "bytes": int(path.stat().st_size),
                }
            )
            class_counts[class_dir.name] += 1
    if not inventory:
        raise ValueError(f"anchor has no images: {anchor}")
    encoded = json.dumps(inventory, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    manifest = {
        "anchor": str(anchor),
        "n_images": len(inventory),
        "class_counts": dict(sorted(class_counts.items())),
        "inventory_sha256": hashlib.sha256(encoded).hexdigest(),
        "inventory": inventory,
    }
    path = output_root / "anchor_manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("inventory_sha256") != manifest["inventory_sha256"]:
            raise RuntimeError(
                f"control anchor changed after runs began: {path}; choose a new output root"
            )
    else:
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def backbone_checkpoint(backbone: str) -> Path:
    checkpoint = TAPT_CKPT if backbone == "cnn_tapt" else FCMAE_CKPT
    if not checkpoint.exists():
        raise FileNotFoundError(f"backbone checkpoint is unavailable: {checkpoint}")
    return checkpoint


def archive_cfg(backbone_path: Path, cell: str, anchor: Path) -> dict:
    values = CELLS[cell]
    return {
        "TRAIN_DIR": str(anchor),
        "UNKNOWN_DIR": str(anchor),
        "OVERLAY_DIR": str(anchor),
        "IMAGE_SIZE": 384,
        "PROJ_DIM": 128,
        "FREEZE_BACKBONE": True,
        "EPOCHS": 0 if cell in DIAGNOSTIC_CELLS else 5,
        "WARMUP_EPOCHS": 1,
        # The archived Windows wrapper used this default for the May runs.
        "TRAIN_SAMPLING_RATIO": 0.25,
        "BATCH": 8,
        "NUM_WORKERS": 0,
        "PIN_MEMORY": False,
        "PERSISTENT": False,
        "PREFETCH_FACTOR": 4,
        "DROP_LAST": False,
        "LR_HEAD": float(values.get("lr_head", 1e-3)),
        "WD": 1e-6,
        "TEMP": float(values.get("temp", 0.07)),
        "SEED": 42,
        "USE_LOCAL": values["local"],
        "LOCAL_WEIGHT": values["local_weight"],
        "LOCAL_ANCHORS": "grid36_full",
        "LOCAL_SEARCH": "window",
        "LOCAL_WINDOW": 4,
        "LOCAL_POS_TOPK": 12,
        "LOCAL_POS_MIN_SIM": 0.70,
        "LOCAL_SUBBATCH": 32,
        "LOCAL_EVERY_N": 1,
        "USE_QUEUE": values["queue"],
        "QUEUE_SIZE": int(values.get("queue_size", 4096)),
        "QUEUE_WEIGHT": 1.0,
        "IGNORE_NEG_SIM": values["ignore"],
        "NECO_WEIGHT": values["neco"],
        # Fixed post-hoc scoring protocol used for the May component table.
        "MIN_CLUSTER_SIZE": 12,
        "MIN_SAMPLES": 3,
        "HDBSCAN_METRIC": "euclidean",
        "CLUSTER_SELECTION_METHOD": "eom",
        "CLUSTER_SELECTION_EPSILON": 0.0,
        "ALLOW_SINGLE_CLUSTER": False,
        "LOCAL_BACKBONE_WEIGHTS": str(backbone_path),
    }


def load_archived_module(source_dir: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, source_dir / "contrastive.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load archived trainer from {source_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def unwrap_state(checkpoint_path: Path) -> dict:
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(loaded, dict):
        for key in ("model", "model_state", "state_dict"):
            if isinstance(loaded.get(key), dict):
                return loaded[key]
    if not isinstance(loaded, dict):
        raise TypeError(f"checkpoint has no state dictionary: {checkpoint_path}")
    return loaded


def normalized_backbone_checkpoint(output_root: Path, backbone: str) -> Path:
    """Give the archived loader a plain state dict for both paired arms.

    The TAPT artifact stores weights under ``model`` while the archived May
    loader only unwraps ``state_dict``. Passing the original TAPT file would
    silently leave the trained cells on a random backbone.
    """
    original = backbone_checkpoint(backbone)
    source_sha256 = sha256_file(original)
    normalized_dir = output_root / "_normalized_backbones"
    normalized_dir.mkdir(parents=True, exist_ok=True)
    normalized = normalized_dir / f"{backbone}_{source_sha256[:12]}.pth"
    metadata = normalized.with_suffix(".json")
    if normalized.exists() and metadata.exists():
        record = json.loads(metadata.read_text(encoding="utf-8"))
        if record.get("source_sha256") == source_sha256:
            return normalized
    audit = audit_backbone_checkpoint(original)
    state = unwrap_state(original)
    torch.save(state, normalized)
    metadata.write_text(
        json.dumps(
            {
                "backbone": backbone,
                "source": str(original.resolve()),
                "source_sha256": source_sha256,
                "normalized": str(normalized.resolve()),
                "normalized_sha256": sha256_file(normalized),
                "state_keys": len(state),
                "load_audit": audit,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return normalized


def make_frozen_run(
    source_dir: Path,
    output_root: Path,
    backbone: str,
    anchor: Path,
    control_id: str,
    cell: str = "FROZEN",
) -> Path:
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_dir = output_root / f"{stamp}_{control_id}_{backbone}_{cell.lower()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    init_path = normalized_backbone_checkpoint(output_root, backbone)
    cfg = archive_cfg(init_path, cell, anchor)
    module = load_archived_module(source_dir, f"may_control_frozen_{backbone}_{stamp}")
    module.CFG.update(cfg)
    # Frozen rows are scored in backbone feature space.  Freeze the head seed
    # too so the saved diagnostic checkpoint remains repeatable.
    torch.manual_seed(int(cfg["SEED"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg["SEED"]))
    model = module.CL()
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, checkpoints / "final_infer.pt")
    (run_dir / "run_info.json").write_text(
        json.dumps(
            {"cfg": module.CFG, "source_commit": SOURCE_COMMIT, "frozen": True, "cell": cell},
            indent=2,
        ),
        encoding="utf-8",
    )
    del model
    return run_dir


def training_env(backbone: str, cell: str, anchor: Path, checkpoint: Path) -> dict[str, str]:
    values = CELLS[cell]
    env = os.environ.copy()
    env.update(
        {
            "BACKBONE_CKPT": str(checkpoint),
            "DATA_DIR": str(anchor),
            "EPOCHS": "5",
            "BATCH": "8",
            "IMAGE_SIZE": "384",
            "WARMUP_EPOCHS": "1",
            "TRAIN_SAMPLING_RATIO": "0.25",
            "USE_LOCAL": str(values["local"]).lower(),
            "LOCAL_WEIGHT": str(values["local_weight"]),
            "LOCAL_POS_TOPK": "12",
            "USE_QUEUE": str(values["queue"]).lower(),
            "QUEUE_SIZE": str(int(values.get("queue_size", 4096))),
            "IGNORE_NEG_SIM": str(values["ignore"]),
            "NCE_TEMP": str(float(values.get("temp", 0.07))),
            "LR_HEAD": str(float(values.get("lr_head", 1e-3))),
            "SEED": "42",
            "FREEZE_BACKBONE": "true",
            "BACKBONE_UNFREEZE_LAST_N": "0",
            "NECO_WEIGHT": str(values["neco"]),
            "MIN_CLUSTER_SIZE": "12",
            "MIN_SAMPLES": "3",
            "CLUSTER_SELECTION_METHOD": "eom",
            "CLUSTER_SELECTION_EPSILON": "0.0",
            "NUM_WORKERS": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def run_archived_training(
    source_dir: Path, output_root: Path, backbone: str, cell: str, anchor: Path, control_id: str
) -> Path:
    checkpoint = normalized_backbone_checkpoint(output_root, backbone)
    before = {path.resolve() for path in ROOT.glob("outputs_contrastive_*") if path.is_dir()}
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    log_path = logs / f"{stamp}_{control_id}_{backbone}_{cell.lower()}.log"
    command = [sys.executable, "-u", str(source_dir / "run_contrastive.py")]
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=source_dir,
            env=training_env(backbone, cell, anchor, checkpoint),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    after = {path.resolve() for path in ROOT.glob("outputs_contrastive_*") if path.is_dir()}
    created = sorted(after - before, key=lambda path: path.stat().st_mtime)
    if len(created) != 1:
        raise RuntimeError(f"expected one archived output directory, got {created}; log={log_path}")
    run_dir = output_root / f"{stamp}_{control_id}_{backbone}_{cell.lower()}"
    shutil.move(str(created[0]), str(run_dir))
    shutil.copy2(log_path, run_dir / "source_train.log")
    return run_dir


class EvalDataset(Dataset):
    def __init__(self, root: Path, transform) -> None:
        self.items: list[tuple[Path, str]] = []
        for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            for path in sorted(class_dir.rglob("*")):
                if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                    self.items.append((path, class_dir.name))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        path, label = self.items[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label, str(path)


def evaluate_run(
    source_dir: Path,
    run_dir: Path,
    mode: str,
    anchor: Path,
    anchor_manifest: dict,
    backbone: str,
    cell: str,
    control_id: str,
    primary: bool = True,
    checkpoint_path: Path | None = None,
    artifact_prefix: str | None = None,
    run_multicluster: bool = True,
) -> dict:
    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    cfg = info["cfg"]
    module = load_archived_module(source_dir, f"may_control_eval_{run_dir.name}")
    module.CFG.update(cfg)
    model = module.CL()
    checkpoint_path = checkpoint_path or (run_dir / "checkpoints" / "final_infer.pt")
    model.load_state_dict(unwrap_state(checkpoint_path), strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    dataset = EvalDataset(anchor, module.tfm(False))
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    paths: list[str] = []
    with torch.no_grad():
        for images, batch_labels, batch_paths in loader:
            images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            if mode == "projection":
                vectors = model(images)
            else:
                feature_map = model._forward_features(images)
                vectors = F.normalize(feature_map.mean(dim=(2, 3)), dim=1)
            embeddings.append(vectors.float().cpu().numpy())
            labels.extend(batch_labels)
            paths.extend(batch_paths)
    emb = np.concatenate(embeddings, axis=0)
    if mode == "pca128":
        emb = PCA(n_components=128, svd_solver="randomized", random_state=int(cfg["SEED"])).fit_transform(emb)
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    elif mode == "random128":
        rng = np.random.default_rng(int(cfg["SEED"]))
        projection = rng.standard_normal((emb.shape[1], 128), dtype=np.float32) / np.sqrt(128.0)
        emb = emb @ projection
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    emb = emb.astype(np.float32, copy=False)
    embedding_sha256 = hashlib.sha256(np.ascontiguousarray(emb).tobytes()).hexdigest()
    metrics = calculate_metrics(emb, labels, IGNORED)
    tier1 = calculate_defect_only_metrics(emb, labels, IGNORED)
    common = {
        "backbone": backbone,
        "cell": cell,
        "embedding": mode,
        "source_commit": SOURCE_COMMIT,
        "control_id": control_id,
        "anchor_manifest_sha256": anchor_manifest["inventory_sha256"],
        "anchor_n_images": anchor_manifest["n_images"],
        "embedding_sha256": embedding_sha256,
        "checkpoint": str(checkpoint_path),
    }
    metrics.update(
        {
            **common,
            "protocol": "archived_may_loss + full_pool_hdbscan + canonical_dominant_P1",
        }
    )
    tier1.update(
        {
            **common,
            "protocol": "archived_may_loss + defect_only_hdbscan + canonical_dominant_P1",
        }
    )
    artifact_name = f"{artifact_prefix}_{mode}" if artifact_prefix else mode
    embedding_dir = run_dir / "epoch_embeddings" if artifact_prefix else run_dir
    embedding_dir.mkdir(exist_ok=True)
    embedding_path = embedding_dir / f"embedding_{artifact_name}.npy"
    np.save(embedding_path, emb)
    (run_dir / "embedding_paths.json").write_text(
        json.dumps({"paths": paths, "labels": labels}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    canonical_dir = run_dir / "canonical_eval"
    canonical_dir.mkdir(exist_ok=True)
    metrics_dir = canonical_dir / "epoch_metrics" if artifact_prefix else canonical_dir
    metrics_dir.mkdir(exist_ok=True)
    (metrics_dir / f"metrics_{artifact_name}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tier1_path = metrics_dir / f"historical_tier1_defect_only_{artifact_name}.json"
    tier1_path.write_text(json.dumps(tier1, ensure_ascii=False, indent=2), encoding="utf-8")
    with (metrics_dir / f"metrics_{artifact_name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    if primary and not artifact_prefix:
        (canonical_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (canonical_dir / "historical_tier1_defect_only.json").write_text(
            json.dumps(tier1, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        shutil.copy2(metrics_dir / f"metrics_{artifact_name}.csv", canonical_dir / "metrics.csv")
    if run_multicluster:
        multicluster_csv = canonical_dir / f"multicluster_{artifact_name}.csv"
        score_env = os.environ.copy()
        score_env["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "_score_umapfree.py"),
                str(embedding_path),
                "--labels-from",
                "pool",
                "--pool",
                str(anchor),
                "--exclude-classes",
                ",".join(sorted(IGNORED)),
                "--skip-umap",
                "--out-csv",
                str(multicluster_csv),
            ],
            cwd=ROOT,
            env=score_env,
            check=True,
        )
    (run_dir / "reproduction_provenance.json").write_text(
        json.dumps(
            {
                "control_id": control_id,
                "source_commit": SOURCE_COMMIT,
                "anchor_manifest": str((run_dir.parent / "anchor_manifest.json")),
                "anchor_manifest_sha256": anchor_manifest["inventory_sha256"],
                "anchor_n_images": anchor_manifest["n_images"],
                "historical_reference": "May anchor file_list.parquet is unavailable; this is a paired control, not an exact historical dataset reproduction.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["cnn_tapt", "nocnn"], required=True)
    parser.add_argument("--cell", choices=sorted(CELLS), required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--control-id", default="may37_protocol_control")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs" / "may37_original_reproduction")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    anchor = args.anchor.resolve()
    if not anchor.exists():
        raise FileNotFoundError(f"control anchor is unavailable: {anchor}")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    anchor_manifest = write_anchor_manifest(anchor, output_root)
    source_dir = materialize_source(output_root)
    print(f"[SOURCE] git={SOURCE_COMMIT} dir={source_dir}", flush=True)
    print(
        f"[START] control={args.control_id} backbone={args.backbone} cell={args.cell} "
        f"anchor={anchor} n={anchor_manifest['n_images']} "
        f"sha256={anchor_manifest['inventory_sha256']}",
        flush=True,
    )
    if args.cell in DIAGNOSTIC_CELLS:
        run_dir = make_frozen_run(
            source_dir, output_root, args.backbone, anchor, args.control_id, args.cell
        )
        mode = {"FROZEN": "backbone", "PCA128": "pca128", "RANDOM128": "random128"}[args.cell]
    else:
        run_dir = run_archived_training(
            source_dir, output_root, args.backbone, args.cell, anchor, args.control_id
        )
        mode = "projection"
    print(f"[RUN_DIR] {run_dir}", flush=True)
    if not args.skip_eval:
        primary_metrics = evaluate_run(
            source_dir,
            run_dir,
            mode,
            anchor,
            anchor_manifest,
            args.backbone,
            args.cell,
            args.control_id,
        )
        if args.cell not in DIAGNOSTIC_CELLS:
            evaluate_run(
                source_dir,
                run_dir,
                "backbone",
                anchor,
                anchor_manifest,
                args.backbone,
                args.cell,
                args.control_id,
                primary=False,
            )
            epoch_dir = run_dir / "checkpoints" / "epoch_checkpoints"
            epoch_checkpoints = sorted(epoch_dir.glob("epoch_*.pt"))
            if [path.name for path in epoch_checkpoints] != [f"epoch_{epoch:03d}.pt" for epoch in range(1, 6)]:
                raise RuntimeError(f"incomplete epoch trajectory: {epoch_checkpoints}")
            trajectory = []
            for epoch, checkpoint_path in enumerate(epoch_checkpoints, 1):
                row = evaluate_run(
                    source_dir,
                    run_dir,
                    "projection",
                    anchor,
                    anchor_manifest,
                    args.backbone,
                    args.cell,
                    args.control_id,
                    primary=False,
                    checkpoint_path=checkpoint_path,
                    artifact_prefix=f"epoch_{epoch:03d}",
                    run_multicluster=False,
                )
                row["epoch"] = epoch
                trajectory.append(row)
            if trajectory[-1]["embedding_sha256"] != primary_metrics["embedding_sha256"]:
                raise RuntimeError("epoch-5 projection does not match final_infer projection")
            trajectory_path = run_dir / "canonical_eval" / "epoch_trajectory_projection.json"
            trajectory_path.write_text(
                json.dumps(trajectory, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        completion = {
            "control_id": args.control_id,
            "backbone": args.backbone,
            "cell": args.cell,
            "anchor_manifest_sha256": anchor_manifest["inventory_sha256"],
            "primary_embedding": mode,
            "primary_embedding_sha256": primary_metrics["embedding_sha256"],
            "epoch_trajectory_complete": args.cell in DIAGNOSTIC_CELLS or len(trajectory) == 5,
        }
        (run_dir / "completion.json").write_text(
            json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(primary_metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
