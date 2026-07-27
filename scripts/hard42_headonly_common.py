#!/usr/bin/env python3
"""Shared model and provenance helpers for hard-42 head-only experiments."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
from _common import resolve_pool  # noqa: E402

TRAIN_ROOT = ROOT / "data" / "pools" / "unknown_train_defectaware_260710.json"
DEV_ROOT = ROOT / "data" / "pools" / "unknown_eval100.json"
HOLDOUT_ROOT = ROOT / "data" / "pools" / "unknown_holdout_100_260713.json"
TAPT_CKPT = Path(r"D:\project\known-cnn\models\iter116J_frozen\best_model.pth")
FCMAE_CKPT = ROOT / "weights" / "convnextv2_base.fcmae_ft_in22k_in1k_384.pth"
BACKBONE_NAME = "convnextv2_base.fcmae_ft_in22k_in1k_384"

KNOWN_DEFECTS = {
    "Center_bank_boundary",
    "Center_scratch",
    "Donut_bank_boundary",
    "Donut_fork",
    "Edge-Ring_bank_boundary",
    "Edge-Ring_scratch",
    "Edge-Top_fork",
    "Full_scratch",
    "ParallelScratches",
    "RingDots",
}
STRICT_EXCLUDED = {"Normal", "Random", "R", *KNOWN_DEFECTS}


def checkpoint_for(backbone: str) -> Path:
    if backbone == "cnn_tapt":
        return TAPT_CKPT
    if backbone == "nocnn":
        return FCMAE_CKPT
    raise ValueError(f"unknown backbone: {backbone}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> dict:
    rows = []
    counts: dict[str, int] = {}
    extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    if root.is_file() and root.suffix.lower() == ".json":
        paths, labels = resolve_pool(root, extensions=extensions)
        entries = [(Path(path), label) for path, label in zip(paths, labels)]
    else:
        entries = [
            (path, path.parent.name)
            for path in sorted(p for p in root.rglob("*") if p.is_file())
            if path.suffix.lower() in extensions
        ]
    for path, label in entries:
        counts[label] = counts.get(label, 0) + 1
        rows.append({"path": str(path.resolve()), "bytes": path.stat().st_size})
    encoded = json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return {
        "root": str(root.resolve()),
        "n_images": len(rows),
        "class_counts": dict(sorted(counts.items())),
        "inventory_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def load_trainer(module_name: str):
    path = SCRIPT_DIR / "train_contrastive_ddp.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MultiStageResidualAdapter(nn.Module):
    """Bounded zero-init residual adapter used by adapterN2/adapterN3."""

    def __init__(self, dim: int, hidden_dim: int, stages: int, scale_max: float = 0.1):
        super().__init__()
        self.stages = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )
            for _ in range(stages)
        )
        self.gammas = nn.ParameterList(nn.Parameter(torch.zeros(1)) for _ in range(stages))
        self.scale_max = float(scale_max)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        for stage, gamma in zip(self.stages, self.gammas):
            features = features + torch.tanh(gamma) * self.scale_max * stage(features)
        return features


def install_head_variant(module, head_kind: str, adapter_dim: int) -> None:
    """Install the requested head without editing the shared trainer."""
    original = module.ContrastiveModel
    normalized = head_kind.lower()
    if normalized not in {"linear", "mlp", "ad", "adapter", "adaptern2", "adaptern3"}:
        raise ValueError(f"unsupported head: {head_kind}")

    class HeadOnlyModel(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            feature_dim = int(self.backbone.num_features)
            projection_dim = int(args[1] if len(args) > 1 else kwargs["proj_dim"])
            if normalized == "linear":
                self.proj = nn.Linear(feature_dim, projection_dim)
                self.adapter = None
            elif normalized == "ad":
                self.proj = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(feature_dim, 512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, projection_dim),
                )
                self.adapter = None
            elif normalized in {"adaptern2", "adaptern3"}:
                stages = int(normalized[-1])
                self.adapter = MultiStageResidualAdapter(
                    feature_dim, int(adapter_dim), stages=stages, scale_max=0.1
                )

    module.ContrastiveModel = HeadOnlyModel


def unwrap_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint does not contain a state dict: {path}")
    return state


def audit_backbone_checkpoint(path: Path) -> dict:
    """Fail if a checkpoint would be silently skipped by the trainer loader."""
    import timm

    model = timm.create_model(BACKBONE_NAME, pretrained=False, num_classes=0, global_pool="avg")
    expected = model.state_dict()
    supplied = unwrap_checkpoint(path)
    compatible = {
        key: value
        for key, value in supplied.items()
        if key in expected and getattr(value, "shape", None) == expected[key].shape
    }
    matched_numel = sum(value.numel() for value in compatible.values())
    expected_numel = sum(value.numel() for value in expected.values())
    audit = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "matched_keys": len(compatible),
        "expected_keys": len(expected),
        "matched_numel": matched_numel,
        "expected_numel": expected_numel,
        "coverage": matched_numel / max(1, expected_numel),
    }
    del model, expected, supplied, compatible
    if matched_numel != expected_numel:
        raise RuntimeError(f"incomplete backbone checkpoint load: {audit}")
    return audit
