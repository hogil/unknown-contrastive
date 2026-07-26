#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checkpoint type classifier for the unknown-campaign runner.

Background (260726 real incident): the champion checkpoint
(runs/sweep/*/checkpoints/proj_ep*.pt, Linear(1024,1024,bias=False)
-> BatchNorm1d -> ReLU -> Linear(1024,128)) was fed into
scripts/predict_grouping_prod.py (Linear -> GELU -> Linear, no BN,
always expects a combined backbone+proj state_dict). load_state_dict
(strict=False) then reports missing/unexpected keys and the script
raises SystemExit. This module classifies checkpoints *before*
training/inference so that mismatch is caught statically.

Three checkpoint types:
  - "projection-only": top-level dict has a "proj" key only (siblings:
    epoch/G/Q/L), no backbone weights at all. Example:
    runs/sweep/*/checkpoints/proj_ep*.pt. Needs a separate --backbone.
    If proj_arch == "bn", only grouping_deploy.py can load it.
  - "full-contrastive": a single combined state_dict with both
    "backbone.*" and "proj.*" keys. If proj_arch == "gelu" it is
    compatible with predict_grouping_prod.py. If "bn", no loader in
    this repo currently accepts a combined backbone+BN-proj file.
  - "cnn-tapt": a known-cnn supervised classifier checkpoint (backbone
    stem/stages + head.fc, no "proj." keys at all). Valid only as a
    --backbone-init source for Tier-4 training, never as a predictor
    checkpoint.

This module does not modify any predictor script -- classification and
pre-flight blocking only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_proj_arch(proj_sd: dict) -> str:
    """Identify the projection-head architecture from its state_dict alone.

    - "bn":   Linear(d,d,bias=False) -> BatchNorm1d(d) -> ReLU -> Linear(d,m)
              (grouping_deploy.py::build_proj, _may_repro_src.py::Proj -- current champion)
    - "gelu": Linear(d,d) -> GELU -> Linear(d,m)
              (scripts/train_contrastive.py::ContrastiveModel,
               scripts/predict_grouping_prod.py::ContrastiveInferModel -- older/original)
    - "unknown": neither of the above
    """
    if not isinstance(proj_sd, dict) or not proj_sd:
        return "unknown"
    keys = list(proj_sd.keys())
    if any(k.startswith("net.") for k in keys):
        keys = [k[len("net."):] if k.startswith("net.") else k for k in keys]
    keyset = set(keys)
    if any(k.endswith("running_mean") or k.endswith("running_var") for k in keyset):
        return "bn"
    if "0.weight" in keyset and "2.weight" in keyset and keyset <= {"0.weight", "0.bias", "2.weight", "2.bias"}:
        return "gelu"
    return "unknown"


def _is_state_dict_like(d: Any) -> bool:
    return isinstance(d, dict) and len(d) > 0 and all(hasattr(v, "shape") for v in d.values())


def classify_checkpoint(path) -> dict:
    """Load one checkpoint and return its type + predictor-compatibility metadata.

    Always returns: type, wrapper, proj_arch, needs_backbone,
    compatible_predictors, blocked_as_predictor, detail.
    """
    path = Path(path)
    if not path.exists():
        return {"type": "missing", "wrapper": None, "proj_arch": None, "needs_backbone": None,
                 "compatible_predictors": [], "blocked_as_predictor": True,
                 "detail": f"checkpoint not found: {path}"}

    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        return {"type": "unknown", "wrapper": None, "proj_arch": None, "needs_backbone": None,
                 "compatible_predictors": [], "blocked_as_predictor": True,
                 "detail": f"checkpoint top-level object is {type(raw).__name__}, not a dict"}

    if isinstance(raw.get("state_dict"), dict) and _is_state_dict_like(raw["state_dict"]):
        sd, wrapper = raw["state_dict"], "state_dict"
    elif isinstance(raw.get("model"), dict) and _is_state_dict_like(raw["model"]):
        sd, wrapper = raw["model"], "model"  # known-cnn convention
    else:
        sd, wrapper = raw, "raw"

    keys = list(sd.keys()) if isinstance(sd, dict) else []
    has_backbone_prefix = any(k.startswith("backbone.") for k in keys)
    has_proj_prefix = any(k.startswith("proj.") for k in keys)

    has_head_fc = any(k.startswith("head.fc.") for k in keys)
    has_stem_or_stages = any(k.startswith("stem.") or k.startswith("stages.") for k in keys)
    if wrapper == "model" and has_head_fc and has_stem_or_stages and not has_proj_prefix:
        return {
            "type": "cnn-tapt", "wrapper": wrapper, "proj_arch": None, "needs_backbone": False,
            "compatible_predictors": [], "blocked_as_predictor": True,
            "classes": raw.get("classes"), "backbone_name": raw.get("backbone"),
            "detail": ("known-cnn supervised classifier checkpoint (stem/stages/head.fc, "
                       "no proj.* keys). Valid ONLY as a --backbone-init source for Tier-4 "
                       "TAPT contrastive training (strip head.*, keep backbone weights). "
                       "Never a valid grouping-predictor checkpoint -- blocked."),
        }

    if isinstance(raw.get("proj"), dict) and _is_state_dict_like(raw["proj"]) and not has_backbone_prefix:
        arch = detect_proj_arch(raw["proj"])
        compat = ["grouping_deploy.py"] if arch == "bn" else []
        return {
            "type": "projection-only", "wrapper": wrapper, "proj_arch": arch, "needs_backbone": True,
            "compatible_predictors": compat, "blocked_as_predictor": (arch != "bn"),
            "epoch": raw.get("epoch"),
            "detail": ("head-only checkpoint (e.g. runs/sweep/*/checkpoints/proj_ep*.pt). "
                       "Requires a separate --backbone frozen-weights file. "
                       f"proj_arch={arch!r}. Only grouping_deploy.py's build_proj()+load_proj() "
                       "(Linear->BN->ReLU->Linear) can load this. predict_grouping_prod.py "
                       "CANNOT: it expects a combined backbone+proj state_dict, not a bare proj "
                       "head -- routing this there is exactly today's SystemExit incident."),
        }

    if has_backbone_prefix and has_proj_prefix:
        proj_sub = {k[len("proj."):]: v for k, v in sd.items() if k.startswith("proj.")}
        arch = detect_proj_arch(proj_sub)
        compat = ["predict_grouping_prod.py"] if arch == "gelu" else []
        if arch == "gelu":
            arch_detail = "Compatible with predict_grouping_prod.py::ContrastiveInferModel."
        else:
            arch_detail = ("BN-style proj head (current champion recipe) inside a COMBINED "
                            "backbone+proj checkpoint has no existing loader in this repo "
                            "(grouping_deploy.py only loads a bare 'proj' head plus a separate "
                            "--backbone; predict_grouping_prod.py only loads the GELU-style "
                            "combined head). Blocked -- use the run's per-epoch "
                            "runs/.../checkpoints/proj_ep*.pt (projection-only type) with "
                            "grouping_deploy.py instead.")
        return {
            "type": "full-contrastive", "wrapper": wrapper, "proj_arch": arch, "needs_backbone": False,
            "compatible_predictors": compat, "blocked_as_predictor": (arch != "gelu"),
            "detail": f"combined backbone+proj state_dict, proj_arch={arch!r}. {arch_detail}",
        }

    return {
        "type": "unknown", "wrapper": wrapper, "proj_arch": None, "needs_backbone": None,
        "compatible_predictors": [], "blocked_as_predictor": True,
        "detail": f"unrecognized checkpoint layout (wrapper={wrapper}, n_keys={len(keys)}, "
                  f"sample={keys[:6]})",
    }


PREDICTOR_REQUIREMENTS = {
    "grouping_deploy.py": {
        "types": {"projection-only"},
        "proj_arch": {"bn"},
        "requires_backbone_arg": True,
    },
    "predict_grouping_prod.py": {
        "types": {"full-contrastive"},
        "proj_arch": {"gelu"},
        "requires_backbone_arg": False,
    },
}


def check_predictor_compatibility(predictor: str, ckpt_info: dict, has_backbone_arg: bool = False) -> tuple[bool, str]:
    """Verify a checkpoint/predictor pairing *before* dispatch (blocks today's incident).

    Returns (ok, message). ok=False means this pairing must never be executed.
    """
    req = PREDICTOR_REQUIREMENTS.get(predictor)
    if req is None:
        return False, f"unknown predictor {predictor!r} (no compatibility rule registered)"
    ck_type = ckpt_info.get("type")
    if ck_type in (None, "missing", "unknown"):
        return False, f"checkpoint type is {ck_type!r}: {ckpt_info.get('detail', '')}"
    if ck_type == "cnn-tapt":
        return False, ("checkpoint type is 'cnn-tapt' (known-cnn supervised classifier) -- "
                       "never valid as a predictor checkpoint, only as Tier-4 --backbone-init.")
    if ck_type not in req["types"]:
        return False, (f"{predictor} requires checkpoint type in {sorted(req['types'])}, "
                        f"got {ck_type!r}. {ckpt_info.get('detail', '')}")
    arch = ckpt_info.get("proj_arch")
    if arch not in req["proj_arch"]:
        return False, (
            f"{predictor} requires proj_arch in {sorted(req['proj_arch'])}, got {arch!r}. "
            "This is exactly the 260726 incident: a BN-style champion checkpoint "
            "(Linear->BN->ReLU->Linear) is not loadable by predict_grouping_prod.py's "
            "ContrastiveInferModel (Linear->GELU->Linear, no BN) and dies with SystemExit "
            "deep inside load_state_dict(). Route BN-arch checkpoints to grouping_deploy.py "
            "instead (paired with --backbone)."
        )
    if req["requires_backbone_arg"] and not has_backbone_arg:
        return False, (f"{predictor} needs checkpoint type={ck_type} which is head-only and "
                        f"requires a separate --backbone weights file; none was given")
    return True, "ok"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", required=True, help="checkpoint .pt/.pth path to classify")
    ap.add_argument("--predictor", choices=sorted(PREDICTOR_REQUIREMENTS), default=None,
                     help="if given, also verify compatibility with this predictor script")
    ap.add_argument("--has-backbone-arg", action="store_true",
                     help="pass if a separate --backbone weights file will also be supplied")
    a = ap.parse_args(argv)

    info = classify_checkpoint(a.check)
    print(json.dumps(info, indent=2, ensure_ascii=False, default=str))

    if a.predictor:
        ok, msg = check_predictor_compatibility(a.predictor, info, a.has_backbone_arg)
        print(f"\n[compat] predictor={a.predictor} ok={ok}\n  {msg}")
        return 0 if ok else 1
    return 0 if info["type"] not in ("missing", "unknown") else 1


if __name__ == "__main__":
    raise SystemExit(main())
