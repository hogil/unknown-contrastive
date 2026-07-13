#!/usr/bin/env python3
"""Run one source-faithful May B0-B5 contrastive cell.

The documented May ablation used the trainer stored in git commit
``b796ecbe5f70c9b88944480292e12706b64db83b``.  That loss implementation is
not the current DDP trainer, so this runner materializes the archived source
at run time and executes it unchanged.  Metrics are then recomputed with the
current canonical P1 definition.
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
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from eval_may37_checkpoints import calculate_metrics


ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "b796ecbe5f70c9b88944480292e12706b64db83b"
SOURCE_FILES = ("contrastive.py", "run_contrastive.py")
ANCHOR = ROOT / "data" / "images" / "anchor_avg30_repro"
TAPT_CKPT = Path(r"D:\project\known-cnn\models\iter116J_frozen\best_model.pth")
FCMAE_CKPT = ROOT / "weights" / "convnextv2_base.fcmae_ft_in22k_in1k_384.pth"

CELLS = {
    "FROZEN": {"local": False, "local_weight": 0.0, "queue": False, "ignore": 1.0, "neco": 0.0},
    "B0": {"local": False, "local_weight": 0.0, "queue": False, "ignore": 1.0, "neco": 0.0},
    "B1": {"local": True, "local_weight": 0.5, "queue": False, "ignore": 1.0, "neco": 0.0},
    "B2": {"local": True, "local_weight": 1.0, "queue": False, "ignore": 1.0, "neco": 0.0},
    "B3": {"local": True, "local_weight": 1.0, "queue": True, "ignore": 1.0, "neco": 0.0},
    "B4": {"local": True, "local_weight": 1.0, "queue": True, "ignore": 0.72, "neco": 0.0},
    "B5": {"local": True, "local_weight": 1.0, "queue": True, "ignore": 0.72, "neco": 0.2},
}
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


def materialize_source(output_root: Path) -> Path:
    source_dir = output_root / f"_source_{SOURCE_COMMIT[:12]}"
    source_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in SOURCE_FILES:
        text = git_source(name)
        target = source_dir / name
        if not target.exists() or target.read_text(encoding="utf-8") != text:
            target.write_text(text, encoding="utf-8")
        hashes[name] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (source_dir / "provenance.json").write_text(
        json.dumps({"git_commit": SOURCE_COMMIT, "sha256": hashes}, indent=2), encoding="utf-8"
    )
    return source_dir


def backbone_checkpoint(backbone: str) -> Path:
    checkpoint = TAPT_CKPT if backbone == "cnn_tapt" else FCMAE_CKPT
    if not checkpoint.exists():
        raise FileNotFoundError(f"backbone checkpoint is unavailable: {checkpoint}")
    return checkpoint


def archive_cfg(backbone_path: Path, cell: str) -> dict:
    values = CELLS[cell]
    return {
        "TRAIN_DIR": str(ANCHOR),
        "UNKNOWN_DIR": str(ANCHOR),
        "OVERLAY_DIR": str(ANCHOR),
        "IMAGE_SIZE": 384,
        "PROJ_DIM": 128,
        "FREEZE_BACKBONE": True,
        "EPOCHS": 0 if cell == "FROZEN" else 5,
        "WARMUP_EPOCHS": 1,
        # The archived Windows wrapper used this default for the May runs.
        "TRAIN_SAMPLING_RATIO": 0.25,
        "BATCH": 8,
        "NUM_WORKERS": 0,
        "PIN_MEMORY": False,
        "PERSISTENT": False,
        "PREFETCH_FACTOR": 4,
        "DROP_LAST": False,
        "LR_HEAD": 1e-3,
        "WD": 1e-6,
        "TEMP": 0.07,
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
        "QUEUE_SIZE": 4096,
        "QUEUE_WEIGHT": 1.0,
        "IGNORE_NEG_SIM": values["ignore"],
        "NECO_WEIGHT": values["neco"],
        # Fixed post-hoc scoring protocol used for the B0-B5 table.
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


def make_frozen_run(source_dir: Path, output_root: Path, backbone: str) -> Path:
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_dir = output_root / f"{stamp}_mayexact_{backbone}_frozen"
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = backbone_checkpoint(backbone)
    init_path = run_dir / "_init_backbone.pth"
    torch.save(unwrap_state(checkpoint), init_path)
    cfg = archive_cfg(init_path, "FROZEN")
    module = load_archived_module(source_dir, f"may_exact_frozen_{backbone}_{stamp}")
    module.CFG.update(cfg)
    model = module.CL()
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, checkpoints / "final_infer.pt")
    (run_dir / "run_info.json").write_text(
        json.dumps({"cfg": module.CFG, "source_commit": SOURCE_COMMIT, "frozen": True}, indent=2),
        encoding="utf-8",
    )
    del model
    return run_dir


def training_env(backbone: str, cell: str) -> dict[str, str]:
    checkpoint = backbone_checkpoint(backbone)
    values = CELLS[cell]
    env = os.environ.copy()
    env.update(
        {
            "BACKBONE_CKPT": str(checkpoint),
            "DATA_DIR": str(ANCHOR),
            "EPOCHS": "5",
            "BATCH": "8",
            "IMAGE_SIZE": "384",
            "WARMUP_EPOCHS": "1",
            "TRAIN_SAMPLING_RATIO": "0.25",
            "USE_LOCAL": str(values["local"]).lower(),
            "LOCAL_WEIGHT": str(values["local_weight"]),
            "LOCAL_POS_TOPK": "12",
            "USE_QUEUE": str(values["queue"]).lower(),
            "QUEUE_SIZE": "4096",
            "IGNORE_NEG_SIM": str(values["ignore"]),
            "NCE_TEMP": "0.07",
            "LR_HEAD": "0.001",
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


def run_archived_training(source_dir: Path, output_root: Path, backbone: str, cell: str) -> Path:
    before = {path.resolve() for path in ROOT.glob("outputs_contrastive_*") if path.is_dir()}
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    log_path = logs / f"{stamp}_mayexact_{backbone}_{cell.lower()}.log"
    command = [sys.executable, "-u", str(source_dir / "run_contrastive.py")]
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=source_dir, env=training_env(backbone, cell), stdout=handle, stderr=subprocess.STDOUT, check=True)
    after = {path.resolve() for path in ROOT.glob("outputs_contrastive_*") if path.is_dir()}
    created = sorted(after - before, key=lambda path: path.stat().st_mtime)
    if len(created) != 1:
        raise RuntimeError(f"expected one archived output directory, got {created}; log={log_path}")
    run_dir = output_root / f"{stamp}_mayexact_{backbone}_{cell.lower()}"
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


def evaluate_run(source_dir: Path, run_dir: Path, mode: str) -> dict:
    info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    cfg = info["cfg"]
    module = load_archived_module(source_dir, f"may_exact_eval_{run_dir.name}")
    module.CFG.update(cfg)
    model = module.CL()
    checkpoint = torch.load(run_dir / "checkpoints" / "final_infer.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(unwrap_state(run_dir / "checkpoints" / "final_infer.pt"), strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    dataset = EvalDataset(ANCHOR, module.tfm(False))
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
    metrics = calculate_metrics(emb, labels, IGNORED)
    metrics.update(
        {
            "backbone": "cnn_tapt" if "cnn_tapt" in run_dir.name else "nocnn",
            "cell": run_dir.name.rsplit("_", 1)[-1].upper(),
            "embedding": mode,
            "source_commit": SOURCE_COMMIT,
            "protocol": "archived_may_loss + full_pool_hdbscan + canonical_dominant_P1",
        }
    )
    np.save(run_dir / f"embedding_{mode}.npy", emb)
    (run_dir / "embedding_paths.json").write_text(
        json.dumps({"paths": paths, "labels": labels}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics_dir = run_dir / "canonical_eval"
    metrics_dir.mkdir(exist_ok=True)
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (metrics_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["cnn_tapt", "nocnn"], required=True)
    parser.add_argument("--cell", choices=sorted(CELLS), required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs" / "may37_original_reproduction")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    if not ANCHOR.exists():
        raise FileNotFoundError(f"reconstructed May anchor is unavailable: {ANCHOR}")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_dir = materialize_source(output_root)
    print(f"[SOURCE] git={SOURCE_COMMIT} dir={source_dir}", flush=True)
    print(f"[START] backbone={args.backbone} cell={args.cell} anchor={ANCHOR}", flush=True)
    if args.cell == "FROZEN":
        run_dir = make_frozen_run(source_dir, output_root, args.backbone)
        mode = "backbone"
    else:
        run_dir = run_archived_training(source_dir, output_root, args.backbone, args.cell)
        mode = "projection"
    print(f"[RUN_DIR] {run_dir}", flush=True)
    if not args.skip_eval:
        print(json.dumps(evaluate_run(source_dir, run_dir, mode), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
