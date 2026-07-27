#!/usr/bin/env python3
"""Train a label-free FCMAE projection head on an opaque image pool.

The FCMAE backbone is loaded from the repository's local checkpoint and kept
frozen.  Only the projection MLP from ``_ssl_methods.Net`` is optimized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _common import mask_palette_non_grade_to_white  # noqa: E402


FCMAE_TIMM = "convnextv2_base.fcmae_ft_in22k_in1k_384"
FCMAE_WEIGHTS = REPO / "weights" / "convnextv2_base.fcmae_ft_in22k_in1k_384.pth"
IMAGE_SIZE = 384
IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".jfif", ".bmp", ".tif", ".tiff",
    ".webp", ".ppm", ".pgm", ".pbm", ".pnm",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def atomic_torch_save(path: Path, value: Any, refuse_overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if refuse_overwrite and path.exists():
        raise FileExistsError(f"immutable checkpoint already exists: {path}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def list_images(pools: list[Path]) -> list[Path]:
    paths: set[Path] = set()
    for pool in pools:
        root = pool.expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"--pool is not a directory: {root}")
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.add(path.resolve())
    result = sorted(paths, key=lambda path: str(path))
    if len(result) < 2:
        raise ValueError("at least two supported images are required")
    return result


def manifest_bytes(paths: list[Path]) -> bytes:
    # The array is intentionally opaque.  No parent-folder component is read.
    return (json.dumps([str(path) for path in paths], ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def establish_manifest(out_dir: Path, paths: list[Path]) -> tuple[Path, str]:
    path = out_dir / "paths.json"
    expected = manifest_bytes(paths)
    if path.exists():
        actual = path.read_bytes()
        if json.loads(actual.decode("utf-8")) != [str(item) for item in paths]:
            raise ValueError(f"existing paths.json does not match --pool: {path}")
        return path, sha256_bytes(actual)
    atomic_bytes(path, expected)
    return path, sha256_bytes(expected)


def load_trusted(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def unwrap_state_dict(value: Any) -> dict[str, Any]:
    state = value
    if isinstance(state, dict):
        for key in ("state_dict", "model", "backbone"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict) or not state:
        raise ValueError("local FCMAE checkpoint does not contain a state dict")
    return state


def strip_prefix(key: str) -> str:
    for prefix in ("module.", "model.", "backbone."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def build_net(device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    if not FCMAE_WEIGHTS.is_file():
        raise FileNotFoundError(f"FCMAE weights not found: {FCMAE_WEIGHTS}")
    import timm

    from _ssl_methods import Net

    original_create_model = timm.create_model

    def local_create_model(model_name: str, *args: Any, **kwargs: Any):
        kwargs["pretrained"] = False
        return original_create_model(model_name, *args, **kwargs)

    timm.create_model = local_create_model
    try:
        net = Net("simclr", pdim=128, K=4096, timm_id=FCMAE_TIMM, head="mlp")
    finally:
        timm.create_model = original_create_model

    raw = unwrap_state_dict(load_trusted(FCMAE_WEIGHTS))
    target_state = net.bb.state_dict()
    compatible = {
        strip_prefix(key): value
        for key, value in raw.items()
        if strip_prefix(key) in target_state and target_state[strip_prefix(key)].shape == value.shape
    }
    if not compatible:
        raise ValueError("local FCMAE checkpoint has no compatible backbone weights")
    load_result = net.bb.load_state_dict(compatible, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise ValueError(
            "local FCMAE checkpoint is not an exact backbone load: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )

    first_projection = net.proj[0]
    if not isinstance(first_projection, torch.nn.Linear) or first_projection.bias is None:
        raise ValueError("_ssl_methods.Net mlp projection must expose a first Linear bias")
    with torch.no_grad():
        first_projection.bias.zero_()
    first_projection.bias.requires_grad = False

    for parameter in net.bb.parameters():
        parameter.requires_grad = False
    net = net.to(device)
    net.bb.eval()
    net.proj.train()
    return net, {
        "name": FCMAE_TIMM,
        "weights_path": str(FCMAE_WEIGHTS.resolve()),
        "weights_sha256": sha256_file(FCMAE_WEIGHTS),
        "loaded_keys": len(compatible),
        "missing_keys": len(load_result.missing_keys),
        "unexpected_keys": len(load_result.unexpected_keys),
        "method": "simclr",
        "head": "mlp",
        "pdim": 128,
        "backbone_frozen": True,
        "projection_first_linear_bias": False,
    }


class TwoViewDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, paths: list[Path], seed: int):
        self.paths = paths
        self.seed = seed
        self.resize = T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=T.InterpolationMode.BILINEAR)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.paths)

    def _view(self, image: Image.Image, generator: torch.Generator) -> torch.Tensor:
        image = self.resize(image)
        tensor = self.to_tensor(image)
        noise = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype) * 0.02
        return self.normalize((tensor + noise).clamp(0.0, 1.0))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[index]
        with Image.open(path) as image:
            os.environ["UC_PALETTE_MODE"] = "grade_only"
            image = mask_palette_non_grade_to_white(image).convert("RGB").copy()
        first = torch.Generator(device="cpu").manual_seed(self.seed + index * 2)
        second = torch.Generator(device="cpu").manual_seed(self.seed + index * 2 + 1)
        return self._view(image, first), self._view(image, second)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def restore_optimizer_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def capture_rng() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(value: dict[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if torch.cuda.is_available() and value.get("cuda") is not None:
        torch.cuda.set_rng_state_all(value["cuda"])


def make_recipe(args: argparse.Namespace, backbone_sha256: str) -> dict[str, Any]:
    return {
        "backbone": FCMAE_TIMM,
        "backbone_sha256": backbone_sha256,
        "method": "simclr",
        "head": "mlp",
        "pdim": 128,
        "backbone_frozen": True,
        "projection_first_linear_bias": False,
        "image_size": IMAGE_SIZE,
        "palette_mode": "grade_only",
        "geometry_augmentation": False,
        "gaussian_noise_sigma": 0.02,
        "global_infonce": True,
        "local_loss": False,
        "queue": True,
        "queue_size": args.queue_size,
        "ignore_neg_sim": args.ignore_neg_sim,
        "temperature": args.temperature,
        "lr_head": args.lr_head,
        "weight_decay": args.weight_decay,
        "sample_ratio": args.sample_ratio,
        "epochs": args.epochs,
        "batch": args.batch,
        "num_workers": args.num_workers,
        "seed": args.seed,
    }


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, queue: torch.Tensor, qptr: int,
                  temperature: float, ignore_neg_sim: float) -> tuple[torch.Tensor, int]:
    batch = z1.size(0)
    both = F.normalize(torch.cat((z1, z2), dim=0), dim=1)
    similarity = both @ both.t() / temperature
    diagonal = torch.eye(2 * batch, device=both.device, dtype=torch.bool)
    similarity = similarity.masked_fill(diagonal, -1e9)
    positive = torch.cat((torch.arange(batch, 2 * batch), torch.arange(batch)), dim=0).to(both.device)
    if ignore_neg_sim < 1.0:
        false_negative = (similarity * temperature) > ignore_neg_sim
        false_negative[torch.arange(2 * batch, device=both.device), positive] = False
        similarity = similarity.masked_fill(false_negative, -1e9)

    queue_snapshot = queue.detach().clone()
    queue_logits = (both @ queue_snapshot.t()) / temperature
    if ignore_neg_sim < 1.0:
        queue_logits = queue_logits.masked_fill((queue_logits * temperature) > ignore_neg_sim, -1e9)
    logits = torch.cat((similarity, queue_logits), dim=1)
    loss = F.cross_entropy(logits, positive)

    with torch.no_grad():
        keys = F.normalize(z2.detach(), dim=1)
        end = qptr + keys.size(0)
        if keys.size(0) >= queue.size(0):
            queue.copy_(keys[-queue.size(0):])
            qptr = 0
        elif end <= queue.size(0):
            queue[qptr:end] = keys
            qptr = end % queue.size(0)
        else:
            first = queue.size(0) - qptr
            queue[qptr:] = keys[:first]
            queue[:keys.size(0) - first] = keys[first:]
            qptr = keys.size(0) - first
    return loss, qptr


def checkpoint_state(net: torch.nn.Module, optimizer: torch.optim.Optimizer, queue: torch.Tensor,
                     qptr: int, gstep: int, config: dict[str, Any], manifest_hash: str,
                     recipe_hash: str, trainer_sha256: str, backbone_sha256: str,
                     seed: int, epoch: int) -> dict[str, Any]:
    return {
        "model": cpu_tree(net.state_dict()),
        "optimizer": cpu_tree(optimizer.state_dict()),
        "queue": queue.detach().cpu(),
        "qptr": int(qptr),
        "gstep": int(gstep),
        "config": config,
        "seed": int(seed),
        "epoch": int(epoch),
        "manifest_sha256": manifest_hash,
        "recipe_sha256": recipe_hash,
        "trainer_sha256": trainer_sha256,
        "backbone_sha256": backbone_sha256,
        "rng": capture_rng(),
    }


def lean_checkpoint_state(net: torch.nn.Module, config: dict[str, Any], manifest_hash: str,
                          recipe_hash: str, trainer_sha256: str, seed: int, epoch: int,
                          backbone_info: dict[str, Any]) -> dict[str, Any]:
    model = cpu_tree(net.state_dict())
    head = {key: value for key, value in model.items() if not key.startswith("bb.")}
    if not head or any(not key.startswith("proj.") for key in head):
        raise ValueError("head-only candidate must contain projection state only")
    return {
        "model": head,
        "model_scope": "head_only",
        "seed": int(seed),
        "epoch": int(epoch),
        "config": config,
        "manifest_sha256": manifest_hash,
        "recipe_sha256": recipe_hash,
        "trainer_sha256": trainer_sha256,
        "backbone_sha256": backbone_info["weights_sha256"],
        "backbone": {
            "path": backbone_info["weights_path"],
            "sha256": backbone_info["weights_sha256"],
        },
    }


def load_resume(path: Path, net: torch.nn.Module, optimizer: torch.optim.Optimizer,
                queue: torch.Tensor, config: dict[str, Any], manifest_hash: str,
                recipe_hash: str, trainer_sha256: str, backbone_sha256: str,
                device: torch.device) -> tuple[torch.Tensor, int, int, int, dict[str, Any]]:
    state = load_trusted(path)
    if not isinstance(state, dict):
        raise ValueError(f"resume checkpoint is not a mapping: {path}")
    if state.get("manifest_sha256") != manifest_hash:
        raise ValueError("resume manifest hash does not match current --pool")
    if state.get("recipe_sha256") != recipe_hash:
        raise ValueError("resume recipe hash does not match current arguments")
    if state.get("trainer_sha256") != trainer_sha256:
        raise ValueError("resume trainer implementation hash does not match current file")
    if state.get("backbone_sha256") != backbone_sha256:
        raise ValueError("resume backbone hash does not match current weights")
    if state.get("config", {}).get("recipe") != config["recipe"]:
        raise ValueError("resume recipe configuration does not match current arguments")
    if state.get("config", {}).get("trainer_sha256") != trainer_sha256:
        raise ValueError("resume config trainer implementation hash does not match current file")
    required = ("model", "optimizer", "queue", "qptr", "gstep", "seed", "epoch")
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(f"resume checkpoint missing keys: {', '.join(missing)}")
    if int(state["seed"]) != int(config["recipe"]["seed"]):
        raise ValueError("resume seed does not match current --seed")
    net.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    restore_optimizer_device(optimizer, device)
    restored_queue = state["queue"].to(device)
    if tuple(restored_queue.shape) != tuple(queue.shape):
        raise ValueError("resume queue shape does not match current recipe")
    if state.get("rng"):
        restore_rng(state["rng"])
    net.bb.eval()
    net.proj.train()
    return restored_queue, int(state["qptr"]), int(state["gstep"]), int(state["epoch"]), state


def validate_epoch_checkpoint(path: Path, epoch: int, manifest_hash: str,
                              recipe_hash: str, trainer_sha256: str,
                              backbone_info: dict[str, Any]) -> dict[str, Any]:
    state = load_trusted(path)
    if not isinstance(state, dict):
        raise ValueError(f"immutable checkpoint is not a mapping: {path}")
    if state.get("model_scope") != "head_only":
        raise ValueError(f"immutable checkpoint model_scope is not head_only: {path}")
    if int(state.get("epoch", -1)) != int(epoch):
        raise ValueError(f"immutable checkpoint epoch mismatch: {path}")
    if state.get("manifest_sha256") != manifest_hash:
        raise ValueError(f"immutable checkpoint manifest mismatch: {path}")
    if state.get("recipe_sha256") != recipe_hash:
        raise ValueError(f"immutable checkpoint recipe mismatch: {path}")
    if state.get("trainer_sha256") != trainer_sha256:
        raise ValueError(f"immutable checkpoint trainer implementation mismatch: {path}")
    if state.get("backbone_sha256") != backbone_info["weights_sha256"]:
        raise ValueError(f"immutable checkpoint backbone hash mismatch: {path}")
    expected_backbone = {
        "path": backbone_info["weights_path"],
        "sha256": backbone_info["weights_sha256"],
    }
    if state.get("backbone") != expected_backbone:
        raise ValueError(f"immutable checkpoint backbone provenance mismatch: {path}")
    model = state.get("model")
    if not isinstance(model, dict) or not model or any(key.startswith("bb.") for key in model):
        raise ValueError(f"immutable checkpoint contains backbone tensors: {path}")
    if any(not key.startswith("proj.") for key in model):
        raise ValueError(f"immutable checkpoint contains non-projection head tensors: {path}")
    if any(key in state for key in ("optimizer", "queue", "qptr", "gstep", "rng")):
        raise ValueError(f"immutable checkpoint contains resumable-training state: {path}")
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a label-free FCMAE projection head with Global SimCLR + queue")
    parser.add_argument("--pool", action="append", required=True, type=Path, help="image directory; repeatable")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--sample-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--queue-size", type=int, default=16384)
    parser.add_argument("--ignore-neg-sim", type=float, default=0.72)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--resume", type=Path, help="checkpoint to resume; defaults to checkpoint_latest.pt when present")
    parser.add_argument("--fresh", action="store_true", help="refuse automatic resume and start a new out-dir state")
    args = parser.parse_args()
    if args.fresh and args.resume:
        parser.error("--fresh and --resume cannot be used together")
    if args.batch < 2:
        parser.error("--batch must be at least 2 for BatchNorm projection")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if not 0 < args.sample_ratio <= 1:
        parser.error("--sample-ratio must be in (0, 1]")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.queue_size < 1:
        parser.error("--queue-size must be positive")
    if not 0 < args.temperature:
        parser.error("--temperature must be positive")
    if not 0 < args.lr_head:
        parser.error("--lr-head must be positive")
    if args.weight_decay < 0:
        parser.error("--weight-decay must be non-negative")
    if not 0 <= args.ignore_neg_sim:
        parser.error("--ignore-neg-sim must be non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")
    return args


def main() -> int:
    args = parse_args()
    os.environ["UC_PALETTE_MODE"] = "grade_only"
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    seed_everything(args.seed)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_path = out_dir / "checkpoint_latest.pt"
    immutable_paths = sorted(out_dir.glob("checkpoint_ep*.pt"))
    if args.fresh and (latest_path.exists() or immutable_paths):
        raise FileExistsError("--fresh requires an out-dir with no checkpoint_latest.pt or checkpoint_ep*.pt")
    trainer_sha256 = sha256_file(Path(__file__).resolve())
    paths = list_images(args.pool)
    manifest_path, manifest_hash = establish_manifest(out_dir, paths)
    backbone_sha256 = sha256_file(FCMAE_WEIGHTS)
    recipe = make_recipe(args, backbone_sha256)
    recipe_hash = sha256_bytes(canonical_json(recipe))
    config = {
        "schema_version": "label_free_projection_trainer.v1",
        "trainer_sha256": trainer_sha256,
        "backbone_sha256": backbone_sha256,
        "recipe": recipe,
        "input": {
            "pool_count": len(args.pool),
            "pool_paths": [str(path.expanduser().resolve()) for path in args.pool],
            "image_count": len(paths),
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_hash,
        },
    }
    config_hash = sha256_bytes(canonical_json(config))
    atomic_json(out_dir / "config.json", {**config, "config_sha256": config_hash, "recipe_sha256": recipe_hash})

    net, backbone_info = build_net(device)
    optimizer = torch.optim.AdamW(net.proj.parameters(), lr=args.lr_head, weight_decay=args.weight_decay)
    queue = F.normalize(torch.randn(args.queue_size, 128, device=device), dim=1)
    qptr = 0
    gstep = 0
    completed_epoch = 0
    resume_path = args.resume.expanduser().resolve() if args.resume else (latest_path if latest_path.exists() and not args.fresh else None)
    resumed_state: dict[str, Any] | None = None
    if resume_path is not None:
        queue, qptr, gstep, completed_epoch, resumed_state = load_resume(
            resume_path, net, optimizer, queue, config, manifest_hash, recipe_hash,
            trainer_sha256, backbone_sha256, device
        )
        immutable = out_dir / f"checkpoint_ep{completed_epoch:02d}.pt"
        if immutable.exists():
            validate_epoch_checkpoint(
                immutable, completed_epoch, manifest_hash, recipe_hash, trainer_sha256, backbone_info
            )
        else:
            recovered = lean_checkpoint_state(
                net, config, manifest_hash, recipe_hash, trainer_sha256,
                args.seed, completed_epoch, backbone_info
            )
            atomic_torch_save(immutable, recovered, refuse_overwrite=True)
        if completed_epoch >= args.epochs:
            print(f"[done] checkpoint already reached epoch {completed_epoch}", flush=True)
            return 0

    provenance = {
        "schema_version": "label_free_projection_trainer.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": trainer_sha256,
        "trainer_sha256": trainer_sha256,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "backbone": backbone_info,
        "config": str((out_dir / "config.json").resolve()),
        "config_sha256": config_hash,
        "recipe_sha256": recipe_hash,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "labels_read": False,
        "parent_folder_names_read": False,
        "evaluation_performed": False,
        "resume": str(resume_path) if resume_path else None,
    }
    atomic_json(out_dir / "provenance.json", provenance)

    for epoch in range(completed_epoch + 1, args.epochs + 1):
        rng = random.Random(args.seed + epoch)
        sample_count = min(len(paths), max(2, int(len(paths) * args.sample_ratio)))
        selected = rng.sample(paths, sample_count)
        dataset = TwoViewDataset(selected, seed=args.seed + epoch * 1_000_003)
        loader_generator = torch.Generator(device="cpu").manual_seed(args.seed + epoch * 10_000_019)
        loader = DataLoader(
            dataset,
            batch_size=args.batch,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            generator=loader_generator,
        )
        net.train()
        net.bb.eval()
        net.proj.train()
        loss_sum = 0.0
        updates = 0
        skipped = 0
        started = time.time()
        for first, second in loader:
            if first.size(0) < 2:
                skipped += 1
                continue
            first = first.to(device, non_blocking=device.type == "cuda")
            second = second.to(device, non_blocking=device.type == "cuda")
            _, z1 = net(first)
            _, z2 = net(second)
            optimizer.zero_grad(set_to_none=True)
            loss, qptr = info_nce_loss(z1, z2, queue, qptr, args.temperature, args.ignore_neg_sim)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu())
            updates += 1
            gstep += 1
        if updates == 0:
            raise RuntimeError("epoch produced no trainable batch; increase pool size or --batch")
        state = checkpoint_state(
            net, optimizer, queue, qptr, gstep, config, manifest_hash, recipe_hash,
            trainer_sha256, backbone_sha256, args.seed, epoch
        )
        immutable = out_dir / f"checkpoint_ep{epoch:02d}.pt"
        atomic_torch_save(latest_path, state)
        if immutable.exists():
            validate_epoch_checkpoint(immutable, epoch, manifest_hash, recipe_hash, trainer_sha256, backbone_info)
        else:
            candidate = lean_checkpoint_state(
                net, config, manifest_hash, recipe_hash, trainer_sha256,
                args.seed, epoch, backbone_info
            )
            atomic_torch_save(immutable, candidate, refuse_overwrite=True)
        print(
            f"[epoch {epoch}/{args.epochs}] loss={loss_sum / updates:.6f} "
            f"updates={updates} skipped={skipped} sample={sample_count} time={time.time() - started:.1f}s",
            flush=True,
        )
    print(f"[OUT] {out_dir}", flush=True)
    print(f"[OUT] {latest_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError, OSError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
