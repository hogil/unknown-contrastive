#!/usr/bin/env python3
"""Extract label-free, inference-faithful wafer embeddings.

The manifest is intentionally opaque: parent directory names are never read as
labels or metadata.  ``main.npy`` and ``weak_aug.npy`` use the exact order in
``paths.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts._common import mask_palette_non_grade_to_white  # noqa: E402


FCMAE_TIMM = "convnextv2_base.fcmae_ft_in22k_in1k_384"
FCMAE_WEIGHTS = REPO / "weights" / "convnextv2_base.fcmae_ft_in22k_in1k_384.pth"
IMAGE_SIZE = 384
IMAGE_EXTENSIONS = {
    suffix.lower()
    for suffix in Image.registered_extensions()
    if suffix.startswith(".")
}
IMAGE_EXTENSIONS.update(
    {".png", ".jpg", ".jpeg", ".jfif", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm", ".pbm", ".pnm"}
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_trusted(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def list_images(pools: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for pool in pools:
        root = pool.expanduser().resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"--pool is not a directory: {root}")
        paths.extend(
            p.resolve()
            for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    paths = sorted(set(paths), key=lambda p: str(p))
    if not paths:
        raise ValueError("no supported image files found under --pool")
    return paths


def write_manifest(path: Path, paths: list[Path]) -> tuple[bytes, str]:
    # A JSON array keeps this file opaque and contains no parent-folder labels.
    data = json.dumps([str(p) for p in paths], ensure_ascii=False, indent=2).encode("utf-8")
    path.write_bytes(data + b"\n")
    manifest_bytes = path.read_bytes()
    return manifest_bytes, sha256_bytes(manifest_bytes)


def load_manifest(path: Path) -> list[Path]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("paths.json must be a JSON array of absolute path strings")
    paths = [Path(item) for item in value]
    if not all(p.is_absolute() for p in paths):
        raise ValueError("paths.json contains a non-absolute path")
    return paths


def load_image(path: Path) -> Image.Image:
    os.environ["UC_PALETTE_MODE"] = "grade_only"
    with Image.open(path) as image:
        return mask_palette_non_grade_to_white(image).convert("RGB").copy()


def build_main_tensor(image: Image.Image) -> torch.Tensor:
    transform = T.Compose(
        [
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transform(image)


def make_batch(paths: list[Path], seed: int, add_noise: bool) -> torch.Tensor:
    tensors = []
    for index, path in enumerate(paths):
        image = load_image(path)
        tensor = build_main_tensor(image)
        if add_noise:
            # Noise is the only weak augmentation; geometry and palette are unchanged.
            generator = torch.Generator(device="cpu").manual_seed(seed + index)
            noise = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype)
            tensor = tensor + 0.01 * noise
        tensors.append(tensor)
    return torch.stack(tensors, dim=0)


def unwrap_state_dict(checkpoint: Any, key: str | None = None) -> dict[str, Any]:
    state = checkpoint.get(key) if key and isinstance(checkpoint, dict) else checkpoint
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict) or not state:
        raise ValueError("checkpoint does not contain a non-empty model state dict")
    return state


def load_frozen(device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    if not FCMAE_WEIGHTS.is_file():
        raise FileNotFoundError(f"FCMAE weights not found: {FCMAE_WEIGHTS}")
    import timm

    model = timm.create_model(FCMAE_TIMM, pretrained=False, num_classes=0, global_pool="avg")
    checkpoint = torch.load(FCMAE_WEIGHTS, map_location="cpu", weights_only=False)
    state = unwrap_state_dict(checkpoint)
    model_state = model.state_dict()
    compatible = {
        key.removeprefix("module."): value
        for key, value in state.items()
        if key.removeprefix("module.") in model_state
        and model_state[key.removeprefix("module.")].shape == value.shape
    }
    load = model.load_state_dict(compatible, strict=True)
    if not compatible:
        raise ValueError("FCMAE checkpoint had no compatible model weights")
    if load.missing_keys or load.unexpected_keys:
        raise ValueError(
            "FCMAE checkpoint is not an exact backbone load: "
            f"missing={load.missing_keys}, unexpected={load.unexpected_keys}"
        )
    return model.to(device).eval(), {
        "path": str(FCMAE_WEIGHTS.resolve()),
        "sha256": sha256_file(FCMAE_WEIGHTS),
        "loaded_keys": len(compatible),
        "missing_keys": len(load.missing_keys),
        "unexpected_keys": len(load.unexpected_keys),
    }


def infer_net_shape(state: dict[str, Any]) -> tuple[str, int, int, str]:
    keys = set(state)
    if any(key.startswith("proto.") for key in keys):
        method = "dino"
        proto_key = next(key for key in keys if key.endswith("proto.weight_v"))
        dino_k = int(state[proto_key].shape[0])
    elif any(key.startswith("pred.") for key in keys):
        method = "byol"
        dino_k = 4096
    else:
        method = "simclr"
        dino_k = 4096

    if "proj.weight" in state:
        pdim = int(state["proj.weight"].shape[0])
    elif "proj.3.weight" in state:
        pdim = int(state["proj.3.weight"].shape[0])
    elif "proj.2.weight" in state:
        pdim = int(state["proj.2.weight"].shape[0])
    else:
        raise ValueError("cannot infer projection dimension from checkpoint['model']")

    if "ad_gamma" in keys:
        head = "adapter"
    elif "ad_gammas.0" in keys:
        stages = 1 + max(
            int(key.split(".")[1])
            for key in keys
            if key.startswith("ad_gammas.") and key.split(".")[1].isdigit()
        )
        head = f"adapterN{stages}"
    elif "proj.1.weight" in state and "proj.0.weight" not in state:
        head = "ad"
    elif "proj.3.weight" in state:
        head = "mlp"
    elif "proj.weight" in state:
        head = "linear"
    else:
        raise ValueError("cannot infer projection head from checkpoint['model']")
    return method, pdim, dino_k, head


def build_safe_net(method: str, pdim: int, dino_k: int, head: str) -> torch.nn.Module:
    import timm

    from _ssl_methods import Net

    create_model = timm.create_model

    def local_create_model(model_name: str, *args: Any, **kwargs: Any):
        kwargs["pretrained"] = False
        return create_model(model_name, *args, **kwargs)

    timm.create_model = local_create_model
    try:
        return Net(method, pdim=pdim, K=dino_k, timm_id=FCMAE_TIMM, head=head)
    finally:
        timm.create_model = create_model


def load_head_only_ssl_net(checkpoint_path: Path, checkpoint: dict[str, Any], mode: str,
                           device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    if mode != "projection":
        raise ValueError("head_only checkpoints are supported only with --mode projection")
    required = (
        "model", "model_scope", "seed", "epoch", "config", "manifest_sha256",
        "recipe_sha256", "trainer_sha256", "backbone",
    )
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"head_only checkpoint missing keys: {', '.join(missing)}")
    if checkpoint.get("model_scope") != "head_only":
        raise ValueError("checkpoint model_scope is not head_only")
    if any(key in checkpoint for key in ("optimizer", "queue", "qptr", "gstep", "rng")):
        raise ValueError("head_only checkpoint contains resumable-training state")
    backbone = checkpoint["backbone"]
    expected_backbone = {
        "path": str(FCMAE_WEIGHTS.resolve()),
        "sha256": sha256_file(FCMAE_WEIGHTS),
    }
    if backbone != expected_backbone:
        raise ValueError("head_only checkpoint local backbone provenance does not match fixed weights")

    state = unwrap_state_dict(checkpoint, "model")
    if not state or any(not key.startswith("proj.") for key in state):
        raise ValueError("head_only checkpoint model must contain projection keys only")
    method, pdim, dino_k, head = infer_net_shape(state)
    net = build_safe_net(method, pdim, dino_k, head)

    raw = unwrap_state_dict(load_trusted(FCMAE_WEIGHTS))
    target_state = net.bb.state_dict()
    compatible = {
        key.removeprefix("module."): value
        for key, value in raw.items()
        if key.removeprefix("module.") in target_state
        and target_state[key.removeprefix("module.")].shape == value.shape
    }
    if not compatible:
        raise ValueError("fixed FCMAE checkpoint had no compatible backbone weights")
    backbone_load = net.bb.load_state_dict(compatible, strict=True)
    if backbone_load.missing_keys or backbone_load.unexpected_keys:
        raise ValueError(
            "head_only FCMAE backbone is not an exact load: "
            f"missing={backbone_load.missing_keys}, unexpected={backbone_load.unexpected_keys}"
        )

    result = net.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(not key.startswith("bb.") for key in result.missing_keys):
        raise ValueError(
            "head_only state is not strict-compatible: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    net = net.to(device).eval()
    return net, {
        "path": str(checkpoint_path.resolve()),
        "sha256": sha256_file(checkpoint_path),
        "seed": checkpoint.get("seed"),
        "epoch": checkpoint.get("epoch"),
        "gstep": None,
        "method": method,
        "pdim": pdim,
        "head": head,
        "model_keys": len(state),
        "backbone_loaded_keys": len(compatible),
        "model_scope": "head_only",
        "restore": "head_only_local_backbone+strict_head",
    }


def load_ssl_net(checkpoint_path: Path, mode: str, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_scope" in checkpoint:
        if checkpoint.get("model_scope") != "head_only":
            raise ValueError("unsupported checkpoint model_scope")
        return load_head_only_ssl_net(checkpoint_path, checkpoint, mode, device)
    state = unwrap_state_dict(checkpoint, "model")
    method, pdim, dino_k, head = infer_net_shape(state)
    import timm

    from _ssl_methods import Net

    # Net's training constructor defaults to pretrained=True.  The checkpoint
    # already contains the complete model, so avoid a network/cache-dependent
    # second initialization while still using the project Net implementation.
    create_model = timm.create_model

    def local_create_model(model_name: str, *args: Any, **kwargs: Any):
        kwargs["pretrained"] = False
        return create_model(model_name, *args, **kwargs)

    timm.create_model = local_create_model
    try:
        net = Net(method, pdim=pdim, K=dino_k, timm_id=FCMAE_TIMM, head=head)
    finally:
        timm.create_model = create_model
    net.load_state_dict(state, strict=True)
    net = net.to(device).eval()
    return net, {
        "path": str(checkpoint_path.resolve()),
        "sha256": sha256_file(checkpoint_path),
        "seed": checkpoint.get("seed") if isinstance(checkpoint, dict) else None,
        "gstep": checkpoint.get("gstep") if isinstance(checkpoint, dict) else None,
        "method": method,
        "pdim": pdim,
        "head": head,
        "model_keys": len(state),
        "restore": "strict",
    }


@torch.inference_mode()
def extract(
    model: torch.nn.Module,
    paths: list[Path],
    embedding_mode: str,
    device: torch.device,
    batch_size: int,
    seed: int,
    weak_aug: bool,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        x = make_batch(batch_paths, seed=seed + start, add_noise=weak_aug)
        result = model(x.to(device, non_blocking=device.type == "cuda"))
        if isinstance(result, tuple):
            f, z = result[:2]
            result = f if embedding_mode == "adapter" else z
        if result.ndim != 2:
            raise ValueError(f"expected rank-2 embedding, got shape {tuple(result.shape)}")
        result = F.normalize(result, dim=1)
        outputs.append(result.float().cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)


def save_npy(path: Path, array: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Label-free FCMAE/adapter/projection embedding extractor")
    parser.add_argument("--pool", action="append", required=True, type=Path, help="image pool; repeatable")
    parser.add_argument("--mode", choices=("frozen", "adapter", "projection"), required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, help="_ssl_methods checkpoint for adapter/projection")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0, help="deterministic weak-view noise seed")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.mode == "frozen" and args.checkpoint is not None:
        parser.error("--checkpoint is only valid with --mode adapter or projection")
    if args.mode != "frozen" and args.checkpoint is None:
        parser.error("--checkpoint is required with --mode adapter or projection")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")

    device = torch.device(args.device)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = list_images(args.pool)
    manifest_path = out_dir / "paths.json"
    _, manifest_hash = write_manifest(manifest_path, paths)
    manifest_paths = load_manifest(manifest_path)
    if manifest_paths != paths:
        raise RuntimeError("paths.json order verification failed before extraction")

    if args.mode == "frozen":
        model, weights_info = load_frozen(device)
        checkpoint_info: dict[str, Any] = {
            "path": None, "sha256": None, "seed": None, "gstep": None,
            "restore": "frozen_local_backbone",
        }
        output_mode = "frozen_f"
    else:
        checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
        model, checkpoint_info = load_ssl_net(checkpoint_path, args.mode, device)
        weights_info = {"path": str(FCMAE_WEIGHTS.resolve()), "sha256": sha256_file(FCMAE_WEIGHTS)}
        output_mode = "adapted_f" if args.mode == "adapter" else "projection_z"

    main_embeddings = extract(
        model, paths, embedding_mode=args.mode, device=device,
        batch_size=args.batch_size, seed=args.seed, weak_aug=False,
    )
    weak_embeddings = extract(
        model, paths, embedding_mode=args.mode, device=device,
        batch_size=args.batch_size, seed=args.seed, weak_aug=True,
    )
    if main_embeddings.shape != weak_embeddings.shape:
        raise RuntimeError(f"main/weak shape mismatch: {main_embeddings.shape} vs {weak_embeddings.shape}")
    if main_embeddings.shape[0] != len(manifest_paths):
        raise RuntimeError("embedding row count does not match paths.json")
    if not np.isfinite(main_embeddings).all() or not np.isfinite(weak_embeddings).all():
        raise RuntimeError("embedding output contains non-finite values")

    main_path = out_dir / "main.npy"
    weak_path = out_dir / "weak_aug.npy"
    save_npy(main_path, main_embeddings)
    save_npy(weak_path, weak_embeddings)
    main_check = np.load(main_path, mmap_mode="r")
    weak_check = np.load(weak_path, mmap_mode="r")
    order_verified = main_check.shape == weak_check.shape == main_embeddings.shape
    if not order_verified:
        raise RuntimeError("saved main.npy and weak_aug.npy shape/order verification failed")

    provenance = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "mode": args.mode,
        "seed": args.seed,
        "output_embedding": output_mode,
        "device": str(device),
        "backbone": FCMAE_TIMM,
        "weights": weights_info,
        "checkpoint": checkpoint_info,
        "restore_mode": checkpoint_info.get("restore"),
        "input": {
            "pool_count": len(args.pool),
            "pool_paths": [str(p.expanduser().resolve()) for p in args.pool],
            "image_count": len(paths),
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_hash,
        },
        "transform": {
            "main": "resize_384_bilinear+to_tensor+ImageNet_normalize+grade_only_palette_mask",
            "weak_aug": "main_view+gaussian_noise_only",
            "weak_noise_std": 0.01,
            "weak_seed": args.seed,
            "geometry_aug": False,
            "palette_mode": "grade_only",
        },
        "outputs": {
            "main": str(main_path),
            "weak_aug": str(weak_path),
            "main_sha256": sha256_file(main_path),
            "weak_aug_sha256": sha256_file(weak_path),
            "shape": list(main_embeddings.shape),
            "dtype": str(main_embeddings.dtype),
            "order_verified": bool(order_verified),
        },
        "label_free_contract": {
            "labels_read": False,
            "parent_folder_names_read": False,
            "manifest_is_opaque": True,
        },
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[OUT] {main_path}")
    print(f"[OUT] {weak_path}")
    print(f"[OUT] {manifest_path}")
    print(f"[OUT] {out_dir / 'provenance.json'}")


if __name__ == "__main__":
    main()
