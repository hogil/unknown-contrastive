#!/usr/bin/env python3
"""Run the production frozen-first, label-free adaptation workflow.

This orchestrator deliberately delegates extraction, training, gating, and
grouping to the existing project scripts.  It never reads labels or interprets
parent directory names, and it never performs TAPT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


REPO = Path(__file__).resolve().parents[1]
EXTRACTOR = (REPO / "scripts/extract_label_free_embeddings.py").resolve()
TRAINER = (REPO / "scripts/train_label_free_projection.py").resolve()
GATE = (REPO / "scripts/label_free_gate.py").resolve()
GROUPER = (REPO / "scripts/group_label_free_embeddings.py").resolve()
IMAGE_SUFFIXES = {
    suffix.lower()
    for suffix in Image.registered_extensions()
    if suffix.startswith(".")
}
IMAGE_SUFFIXES.update(
    {".png", ".jpg", ".jpeg", ".jfif", ".bmp", ".tif", ".tiff", ".webp",
     ".ppm", ".pgm", ".pbm", ".pnm"}
)
FCMAE_WEIGHTS = (REPO / "weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth").resolve()
PRODUCTION_RECIPE = {
    "sample_ratio": 0.25,
    "num_workers": 0,
    "queue_size": 16384,
    "ignore_neg_sim": 0.72,
    "temperature": 0.07,
    "lr_head": 1e-3,
    "weight_decay": 1e-6,
}
OPERATIONAL_FALLBACK_EXIT = 31


class WorkflowError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def path_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    result: dict[str, Any] = {"path": str(resolved), "exists": resolved.exists()}
    if resolved.is_file():
        result["bytes"] = resolved.stat().st_size
        result["sha256"] = sha256_file(resolved)
    else:
        result["bytes"] = None
        result["sha256"] = None
    return result


def list_opaque_images(pools: Sequence[Path]) -> list[Path]:
    paths: set[Path] = set()
    for pool in pools:
        root = pool.resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"--pool is not a directory: {root}")
        paths.update(
            item.resolve()
            for item in root.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        )
    result = sorted(paths, key=lambda value: str(value))
    if not result:
        raise ValueError("--pool contains no supported image files")
    return result


def expected_manifest_bytes(paths: Sequence[Path]) -> bytes:
    # This mirrors the extractor's opaque ordered JSON manifest exactly.
    return (json.dumps([str(path) for path in paths], ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def manifest_fingerprint(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": sha256_bytes(data)}


def parse_rungs(value: str, epochs: int) -> tuple[list[int], list[int]]:
    raw = [part.strip() for part in value.split(",") if part.strip()]
    if not raw:
        raise ValueError("--rungs must contain at least one positive integer")
    try:
        values = [int(part) for part in raw]
    except ValueError as exc:
        raise ValueError("--rungs must be a comma-separated list of integers") from exc
    if any(item < 1 for item in values):
        raise ValueError("--rungs values must be positive")
    requested = sorted(set(values))
    return [item for item in requested if item <= epochs], [item for item in requested if item > epochs]


def command_text(command: Sequence[str]) -> str:
    return subprocess.list2cmdline([str(item) for item in command])


def run_logged(
    name: str,
    command: Sequence[str],
    log_path: Path,
    status: dict[str, Any],
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    record: dict[str, Any] = {
        "name": name,
        "command": [str(item) for item in command],
        "command_text": command_text(command),
        "log": str(log_path.resolve()),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    stall_minutes = float(status.get("args", {}).get("stall_minutes", 120.0))
    if stall_minutes <= 0:
        raise ValueError("--stall-minutes must be positive")
    last_progress = time.monotonic()
    last_signature: tuple[int, int] | None = None
    stalled = False
    with log_path.open("w", encoding="utf-8", errors="replace") as handle:
        process = subprocess.Popen(
            [str(item) for item in command],
            cwd=str(REPO),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        while process.poll() is None:
            try:
                stat = log_path.stat()
                signature = (stat.st_size, stat.st_mtime_ns)
            except OSError:
                signature = None
            if signature != last_signature:
                last_signature = signature
                last_progress = time.monotonic()
            if time.monotonic() - last_progress >= stall_minutes * 60:
                stalled = True
                handle.write(
                    f"\n[STALL] no log change for {stall_minutes:.1f} minutes; terminating child\n"
                )
                handle.flush()
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
            time.sleep(5)
        return_code = process.returncode
        if return_code is None:
            return_code = process.wait()
        if stalled:
            return_code = 124
    record.update({
        "return_code": return_code,
        "stalled": stalled,
        "stall_minutes": stall_minutes,
        "duration_seconds": round(time.time() - started, 3),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "log_file": path_record(log_path),
    })
    status.setdefault("commands", []).append(record)
    write_status(status)
    return int(return_code)


def write_status(status: dict[str, Any]) -> None:
    output = Path(status["paths"]["status_json"])
    atomic_json(output, status)


def artifact_list(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [path_record(path) for path in paths]


def provenance(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def extraction_cache_valid(
    directory: Path,
    *,
    mode: str,
    pools: Sequence[Path],
    expected_paths: Sequence[Path],
    extractor_sha256: str,
    weights_sha256: str,
    seed: int,
    checkpoint_sha256: str | None = None,
) -> bool:
    manifest = directory / "paths.json"
    main = directory / "main.npy"
    weak = directory / "weak_aug.npy"
    info = provenance(directory / "provenance.json")
    if not manifest.is_file() or not main.is_file() or not weak.is_file() or info is None:
        return False
    expected_bytes = expected_manifest_bytes(expected_paths)
    if manifest.read_bytes() != expected_bytes:
        return False
    if info.get("mode") != mode:
        return False
    if info.get("script") != str(EXTRACTOR) or info.get("script_sha256") != extractor_sha256:
        return False
    weights = info.get("weights", {})
    if weights.get("sha256") != weights_sha256:
        return False
    input_info = info.get("input", {})
    if input_info.get("manifest_sha256") != sha256_bytes(expected_bytes):
        return False
    if input_info.get("pool_paths") != [str(path.resolve()) for path in pools]:
        return False
    if checkpoint_sha256 is not None:
        checkpoint = info.get("checkpoint", {})
        if checkpoint.get("sha256") != checkpoint_sha256:
            return False
    outputs = info.get("outputs", {})
    if info.get("seed") != seed or info.get("transform", {}).get("weak_seed") != seed:
        return False
    try:
        main_array = np.load(main, mmap_mode="r")
        weak_array = np.load(weak, mmap_mode="r")
        expected_shape = tuple(outputs.get("shape", []))
        if main_array.shape != weak_array.shape or main_array.shape != expected_shape:
            return False
        if main_array.shape[0] != len(expected_paths):
            return False
        if not np.isfinite(main_array).all() or not np.isfinite(weak_array).all():
            return False
    except (OSError, ValueError, TypeError):
        return False
    return (
        outputs.get("order_verified") is True
        and outputs.get("main_sha256") == sha256_file(main)
        and outputs.get("weak_aug_sha256") == sha256_file(weak)
    )


def gate_cache_valid(
    path: Path,
    *,
    paths_manifest: Path,
    frozen_embedding: Path,
    adapted_embedding: Path | None,
    frozen_aug: Path,
    adapted_aug: Path | None,
    seed: int,
    augmentation_tolerance: float,
) -> bool:
    value = provenance(path)
    if value is None or value.get("schema_version") != "label_free_gate.v1":
        return False
    if value.get("provenance", {}).get("script_sha256") != sha256_file(GATE):
        return False
    if value.get("bootstrap", {}).get("seed") != seed:
        return False
    if value.get("thresholds", {}).get("augmentation_tolerance") != augmentation_tolerance:
        return False
    inputs = value.get("provenance", {}).get("input_files", {})
    expected = {
        "paths_manifest": paths_manifest,
        "frozen_embedding": frozen_embedding,
        "adapted_embedding": adapted_embedding,
        "frozen_aug_embedding": frozen_aug,
        "adapted_aug_embedding": adapted_aug,
    }
    for key, expected_path in expected.items():
        actual = inputs.get(key)
        if expected_path is None:
            if actual is not None:
                return False
            continue
        if not expected_path.is_file() or not isinstance(actual, dict):
            return False
        if actual.get("path") != str(expected_path.resolve()) or actual.get("sha256") != sha256_file(expected_path):
            return False
    return True


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise WorkflowError(f"expected JSON object: {path}")
    return value


def training_dir_for(out_dir: Path, manifest_hash: str, args: argparse.Namespace) -> Path:
    backbone_sha256 = sha256_file(FCMAE_WEIGHTS)
    expected_recipe = {
        "backbone": "convnextv2_base.fcmae_ft_in22k_in1k_384",
        "backbone_sha256": backbone_sha256,
        "method": "simclr",
        "head": "mlp",
        "pdim": 128,
        "backbone_frozen": True,
        "projection_first_linear_bias": False,
        "image_size": 384,
        "palette_mode": "grade_only",
        "geometry_augmentation": False,
        "gaussian_noise_sigma": 0.02,
        "global_infonce": True,
        "local_loss": False,
        "queue": True,
        "queue_size": PRODUCTION_RECIPE["queue_size"],
        "ignore_neg_sim": PRODUCTION_RECIPE["ignore_neg_sim"],
        "temperature": PRODUCTION_RECIPE["temperature"],
        "lr_head": PRODUCTION_RECIPE["lr_head"],
        "weight_decay": PRODUCTION_RECIPE["weight_decay"],
        "sample_ratio": PRODUCTION_RECIPE["sample_ratio"],
        "epochs": args.epochs,
        "batch": args.batch,
        "num_workers": PRODUCTION_RECIPE["num_workers"],
        "seed": args.seed,
    }
    base = (out_dir / "training").resolve()
    config = provenance(base / "config.json")
    trainer_hash = sha256_file(TRAINER)
    if config is None:
        return base
    if (
        config.get("recipe") == expected_recipe
        and config.get("input", {}).get("manifest_sha256") == manifest_hash
        and config.get("trainer_sha256") == trainer_hash
        and config.get("backbone_sha256") == backbone_sha256
    ):
        return base
    index = 1
    while True:
        candidate = (out_dir / f"training_rerun_{index:02d}").resolve()
        if not (candidate / "config.json").exists() and not list(candidate.glob("checkpoint_ep*.pt")):
            return candidate
        index += 1


def extraction_command(args: argparse.Namespace, pools: Sequence[Path], mode: str, out: Path, checkpoint: Path | None = None) -> list[str]:
    command = [
        str(Path(sys.executable).resolve()), "-u", str(EXTRACTOR),
    ]
    for pool in pools:
        command.extend(["--pool", str(pool.resolve())])
    command.extend([
        "--mode", mode,
        "--out-dir", str(out.resolve()),
        "--device", args.device,
        "--batch-size", str(args.batch),
        "--seed", str(args.seed),
    ])
    if checkpoint is not None:
        command.extend(["--checkpoint", str(checkpoint.resolve())])
    return command


def gate_command(
    args: argparse.Namespace,
    *,
    manifest: Path,
    frozen: Path,
    frozen_aug: Path,
    out: Path,
    adapted: Path | None = None,
    adapted_aug: Path | None = None,
) -> list[str]:
    command = [
        str(Path(sys.executable).resolve()), "-u", str(GATE),
        "--paths-manifest", str(manifest.resolve()),
        "--frozen-embedding", str(frozen.resolve()),
        "--frozen-aug-embedding", str(frozen_aug.resolve()),
        "--out", str(out.resolve()),
        "--seed", str(args.seed),
        "--augmentation-tolerance", str(args.augmentation_tolerance),
    ]
    if adapted is not None:
        command.extend(["--adapted-embedding", str(adapted.resolve())])
    if adapted_aug is not None:
        command.extend(["--adapted-aug-embedding", str(adapted_aug.resolve())])
    return command


def group_command(
    args: argparse.Namespace,
    *,
    manifest: Path,
    embedding: Path,
    gate: Path,
    out: Path,
    mode: str,
) -> list[str]:
    command = [
        str(Path(sys.executable).resolve()), "-u", str(GROUPER),
        "--paths-manifest", str(manifest.resolve()),
        "--embedding", str(embedding.resolve()),
        "--gate-json", str(gate.resolve()),
        "--out-dir", str(out.resolve()),
        "--model-mode", mode,
        "--reps", str(args.reps),
    ]
    if args.copy_groups:
        command.append("--copy-groups")
    return command


def candidate_eligible(gate: dict[str, Any]) -> bool:
    return bool(
        gate.get("workflow_action") == "use_adapted"
        and gate.get("selected_mode") == "adapted"
        and gate.get("adapted_absolute", {}).get("passed") is True
        and (
            gate.get("approval", {}).get("passed") is True
            or gate.get("rescue_approval", {}).get("passed") is True
        )
    )


def candidate_key(row: dict[str, Any]) -> tuple[float, float, float, float, int]:
    summary = row["gate"]["adapted"]
    return (
        -float(summary["non_noise_count"]),
        -float(summary["within_group_cosine_coherence"]),
        -float(summary["bootstrap_stability"]),
        float(summary["fragmentation_proxy"]),
        int(row["epoch"]),
    )


def finalize_operational_frozen_fallback(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    steps: Path,
    status: dict[str, Any],
    frozen_manifest: Path,
    frozen_main: Path,
    frozen_gate_value: dict[str, Any],
    adapted_error: dict[str, Any],
) -> int:
    fallback_gate = out_dir / "frozen_operational_fallback_gate.json"
    fallback = dict(frozen_gate_value)
    fallback["workflow_action"] = "adaptation_error_frozen_fallback"
    fallback["selected_mode"] = "frozen"
    fallback["selection_reason"] = "adapted_operational_error"
    fallback["adaptation_error"] = adapted_error
    atomic_json(fallback_gate, fallback)

    final_group_dir = out_dir / "grouping"
    status["adaptation_error"] = adapted_error
    status["frozen_fallback"] = {
        "attempted": True,
        "succeeded": False,
        "gate": path_record(fallback_gate),
    }
    write_status(status)
    try:
        code = run_logged(
            "group_frozen_operational_fallback",
            group_command(
                args,
                manifest=frozen_manifest,
                embedding=frozen_main,
                gate=fallback_gate,
                out=final_group_dir,
                mode="frozen",
            ),
            steps / "group_frozen_operational_fallback.log",
            status,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        code = None
        fallback_exception = str(exc)
    else:
        fallback_exception = None

    if code != 0:
        status["frozen_fallback"].update({
            "succeeded": False,
            "return_code": code,
            "error": fallback_exception,
        })
        status["final"] = {
            "workflow_action": "workflow_failed",
            "selected_mode": None,
            "adapted_succeeded": False,
            "frozen_fallback_succeeded": False,
            "exit_code": 2,
        }
        write_status(status)
        detail = fallback_exception or f"return code {code}"
        raise WorkflowError(
            f"adapted stage failed and frozen fallback grouping failed: {detail}"
        )

    selection = {
        "selected_mode": "frozen",
        "workflow_action": "adaptation_error_frozen_fallback",
        "reason": "adapted_operational_error",
        "adaptation_error": adapted_error,
        "fallback_gate": path_record(fallback_gate),
        "tapt_performed": False,
    }
    status["selection"] = selection
    status["frozen_fallback"].update({
        "succeeded": True,
        "return_code": 0,
        "grouping_dir": str(final_group_dir),
    })
    status["final"] = {
        "workflow_action": "adaptation_error_frozen_fallback",
        "selected_mode": "frozen",
        "adapted_succeeded": False,
        "frozen_fallback_succeeded": True,
        "grouping_dir": str(final_group_dir),
        "artifacts": artifact_list([
            final_group_dir / "clusters.csv",
            final_group_dir / "summary.json",
        ]),
        "exit_code": OPERATIONAL_FALLBACK_EXIT,
    }
    atomic_json(out_dir / "selection.json", selection)
    write_status(status)
    return OPERATIONAL_FALLBACK_EXIT


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen-first, label-free FCMAE projection adaptation and grouping; never TAPT."
    )
    parser.add_argument("--pool", action="append", required=True, type=Path, help="opaque image pool; repeatable")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rungs", default="1,2,3,4,6,8,10,15,20")
    parser.add_argument("--augmentation-tolerance", type=float, default=0.02)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--copy-groups", action="store_true")
    parser.add_argument("--stall-minutes", type=float, default=120.0,
                        help="terminate a child only after this many minutes without log-file change")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch < 2:
        raise ValueError("--batch must be at least 2")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.reps < 1:
        raise ValueError("--reps must be positive")
    if not 0.0 <= args.augmentation_tolerance <= 1.0:
        raise ValueError("--augmentation-tolerance must be in [0, 1]")
    if args.stall_minutes <= 0:
        raise ValueError("--stall-minutes must be positive")
    if args.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                raise ValueError("--device cuda requested but CUDA is unavailable")
        except ImportError as exc:
            raise ValueError("--device cuda requires torch") from exc

    pools = [pool.expanduser().resolve(strict=True) for pool in args.pool]
    expected_paths = list_opaque_images(pools)
    expected_manifest = expected_manifest_bytes(expected_paths)
    expected_manifest_hash = sha256_bytes(expected_manifest)
    if not FCMAE_WEIGHTS.is_file():
        raise WorkflowError(f"FCMAE weights not found: {FCMAE_WEIGHTS}")
    current_weights_sha256 = sha256_file(FCMAE_WEIGHTS)
    current_extractor_sha256 = sha256_file(EXTRACTOR)
    current_script_hashes = {
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "extractor_sha256": current_extractor_sha256,
        "trainer_sha256": sha256_file(TRAINER),
        "gate_sha256": sha256_file(GATE),
        "grouper_sha256": sha256_file(GROUPER),
        "backbone_sha256": current_weights_sha256,
    }
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = out_dir / "steps"
    frozen_dir = out_dir / "frozen"
    status: dict[str, Any] = {
        "schema_version": "label_free_adaptation.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow": "frozen-first label-free adaptation; no TAPT",
        "args": {
            "pool": [str(pool) for pool in pools],
            "out_dir": str(out_dir),
            "device": args.device,
            "batch": args.batch,
            "epochs": args.epochs,
            "seed": args.seed,
            "rungs": args.rungs,
            "augmentation_tolerance": args.augmentation_tolerance,
            "reps": args.reps,
            "copy_groups": args.copy_groups,
            "stall_minutes": args.stall_minutes,
        },
        "paths": {
            "repo": str(REPO),
            "extractor": str(EXTRACTOR),
            "trainer": str(TRAINER),
            "gate": str(GATE),
            "grouper": str(GROUPER),
            "frozen_dir": str(frozen_dir),
            "status_json": str((out_dir / "status.json").resolve()),
            "selection_json": str((out_dir / "selection.json").resolve()),
        },
        "input": {
            "opaque_image_count": len(expected_paths),
            "pool_paths": [str(pool) for pool in pools],
            "expected_manifest": {"bytes": len(expected_manifest), "sha256": expected_manifest_hash},
            "labels_read": False,
            "folder_names_read": False,
            "majority_label_used": False,
        },
        "fixed_recipe": PRODUCTION_RECIPE,
        "provenance": current_script_hashes,
        "commands": [],
        "steps": [],
        "candidates": [],
        "selection": None,
        "final": None,
    }
    write_status(status)

    frozen_manifest = frozen_dir / "paths.json"
    frozen_main = frozen_dir / "main.npy"
    frozen_weak = frozen_dir / "weak_aug.npy"
    if extraction_cache_valid(
        frozen_dir,
        mode="frozen",
        pools=pools,
        expected_paths=expected_paths,
        extractor_sha256=current_extractor_sha256,
        weights_sha256=current_weights_sha256,
        seed=args.seed,
    ):
        status["steps"].append({"name": "frozen_extract", "action": "cache_reuse", "artifacts": artifact_list([frozen_manifest, frozen_main, frozen_weak, frozen_dir / "provenance.json"])})
    else:
        code = run_logged("frozen_extract", extraction_command(args, pools, "frozen", frozen_dir), steps / "frozen_extract.log", status)
        if code != 0:
            raise WorkflowError(f"frozen extraction failed with return code {code}")
        status["steps"].append({"name": "frozen_extract", "action": "executed", "artifacts": artifact_list([frozen_manifest, frozen_main, frozen_weak, frozen_dir / "provenance.json"])})
    if not frozen_manifest.is_file() or frozen_manifest.read_bytes() != expected_manifest:
        raise WorkflowError("frozen paths.json does not match the opaque pool manifest")
    write_status(status)

    frozen_gate = frozen_dir / "gate.json"
    frozen_gate_cmd = gate_command(
        args,
        manifest=frozen_manifest,
        frozen=frozen_main,
        frozen_aug=frozen_weak,
        out=frozen_gate,
    )
    if gate_cache_valid(
        frozen_gate,
        paths_manifest=frozen_manifest,
        frozen_embedding=frozen_main,
        adapted_embedding=None,
        frozen_aug=frozen_weak,
        adapted_aug=None,
        seed=args.seed,
        augmentation_tolerance=args.augmentation_tolerance,
    ):
        status["steps"].append({"name": "frozen_gate", "action": "cache_reuse", "artifacts": artifact_list([frozen_gate])})
    else:
        code = run_logged("frozen_gate", frozen_gate_cmd, steps / "frozen_gate.log", status)
        if code != 0:
            raise WorkflowError(f"frozen gate failed with return code {code}")
        status["steps"].append({"name": "frozen_gate", "action": "executed", "artifacts": artifact_list([frozen_gate])})
    frozen_gate_value = load_json(frozen_gate)
    frozen_action = frozen_gate_value.get("workflow_action")
    status["frozen_gate"] = {"path": str(frozen_gate), "sha256": sha256_file(frozen_gate), "metrics": frozen_gate_value.get("frozen"), "approval": frozen_gate_value.get("frozen_approval"), "workflow_action": frozen_action}
    write_status(status)

    if frozen_action == "use_frozen":
        final_group_dir = out_dir / "grouping"
        code = run_logged("group_frozen", group_command(args, manifest=frozen_manifest, embedding=frozen_main, gate=frozen_gate, out=final_group_dir, mode="frozen"), steps / "group_frozen.log", status)
        if code != 0:
            raise WorkflowError(f"frozen grouping failed with return code {code}")
        status["selection"] = {"selected_mode": "frozen", "workflow_action": "use_frozen", "reason": "frozen_absolute_gate_passed", "gate": str(frozen_gate)}
        status["final"] = {"workflow_action": "use_frozen", "selected_mode": "frozen", "grouping_dir": str(final_group_dir), "artifacts": artifact_list([final_group_dir / "clusters.csv", final_group_dir / "summary.json"]), "exit_code": 0}
        atomic_json(out_dir / "selection.json", status["selection"])
        write_status(status)
        return 0

    if frozen_action != "adapt_required":
        raise WorkflowError(f"unexpected frozen gate workflow_action: {frozen_action!r}")

    training_dir = training_dir_for(out_dir, expected_manifest_hash, args)
    training_cmd = [str(Path(sys.executable).resolve()), "-u", str(TRAINER)]
    for pool in pools:
        training_cmd.extend(["--pool", str(pool)])
    training_cmd.extend([
        "--out-dir", str(training_dir), "--device", args.device,
        "--batch", str(args.batch), "--epochs", str(args.epochs), "--seed", str(args.seed),
        "--sample-ratio", str(PRODUCTION_RECIPE["sample_ratio"]),
        "--num-workers", str(PRODUCTION_RECIPE["num_workers"]),
        "--queue-size", str(PRODUCTION_RECIPE["queue_size"]),
        "--ignore-neg-sim", str(PRODUCTION_RECIPE["ignore_neg_sim"]),
        "--temperature", str(PRODUCTION_RECIPE["temperature"]),
        "--lr-head", str(PRODUCTION_RECIPE["lr_head"]),
        "--weight-decay", str(PRODUCTION_RECIPE["weight_decay"]),
    ])
    training_latest = training_dir / "checkpoint_latest.pt"
    if training_latest.is_file() and provenance(training_dir / "config.json") is not None:
        training_action = "resume_or_complete"
    else:
        training_action = "execute"
    try:
        code = run_logged("train_projection", training_cmd, steps / "train_projection.log", status)
    except (OSError, subprocess.SubprocessError) as exc:
        return finalize_operational_frozen_fallback(
            args=args,
            out_dir=out_dir,
            steps=steps,
            status=status,
            frozen_manifest=frozen_manifest,
            frozen_main=frozen_main,
            frozen_gate_value=frozen_gate_value,
            adapted_error={
                "stage": "train_projection",
                "kind": "launch_error",
                "message": str(exc),
                "return_code": None,
            },
        )
    if code != 0:
        return finalize_operational_frozen_fallback(
            args=args,
            out_dir=out_dir,
            steps=steps,
            status=status,
            frozen_manifest=frozen_manifest,
            frozen_main=frozen_main,
            frozen_gate_value=frozen_gate_value,
            adapted_error={
                "stage": "train_projection",
                "kind": "subprocess_failure",
                "message": f"projection training failed with return code {code}",
                "return_code": code,
            },
        )
    status["steps"].append({"name": "train_projection", "action": training_action, "training_dir": str(training_dir), "artifacts": artifact_list([training_dir / "config.json", training_dir / "provenance.json", training_latest])})
    write_status(status)

    requested_rungs, skipped_rungs = parse_rungs(args.rungs, args.epochs)
    status["rungs"] = {"requested": requested_rungs, "skipped_above_epochs": skipped_rungs}
    candidates: list[dict[str, Any]] = []
    adaptation_errors: list[dict[str, Any]] = []
    for epoch in requested_rungs:
        checkpoint = training_dir / f"checkpoint_ep{epoch:02d}.pt"
        candidate_dir = out_dir / "candidates" / f"ep{epoch:02d}"
        candidate_main = candidate_dir / "main.npy"
        candidate_weak = candidate_dir / "weak_aug.npy"
        candidate_gate = candidate_dir / "gate.json"
        if not checkpoint.is_file():
            row = {"epoch": epoch, "checkpoint": path_record(checkpoint), "status": "missing_checkpoint"}
            adaptation_errors.append({
                "stage": "adapted_extraction",
                "kind": "missing_checkpoint",
                "epoch": epoch,
                "message": f"adapted checkpoint is missing: {checkpoint}",
                "return_code": None,
            })
            status["candidates"].append(row)
            write_status(status)
            continue
        checkpoint_hash = sha256_file(checkpoint)
        if extraction_cache_valid(
            candidate_dir,
            mode="projection",
            pools=pools,
            expected_paths=expected_paths,
            extractor_sha256=current_extractor_sha256,
            weights_sha256=current_weights_sha256,
            seed=args.seed,
            checkpoint_sha256=checkpoint_hash,
        ):
            extraction_action = "cache_reuse"
        else:
            try:
                code = run_logged(f"extract_ep{epoch:02d}", extraction_command(args, pools, "projection", candidate_dir, checkpoint), steps / f"extract_ep{epoch:02d}.log", status)
            except (OSError, subprocess.SubprocessError) as exc:
                row = {"epoch": epoch, "checkpoint": path_record(checkpoint), "status": "extraction_failed", "return_code": None, "error": str(exc)}
                adaptation_errors.append({
                    "stage": "adapted_extraction",
                    "kind": "launch_error",
                    "epoch": epoch,
                    "message": str(exc),
                    "return_code": None,
                })
                status["candidates"].append(row)
                write_status(status)
                continue
            if code != 0:
                row = {"epoch": epoch, "checkpoint": path_record(checkpoint), "status": "extraction_failed", "return_code": code}
                adaptation_errors.append({
                    "stage": "adapted_extraction",
                    "kind": "subprocess_failure",
                    "epoch": epoch,
                    "message": f"adapted extraction failed with return code {code}",
                    "return_code": code,
                })
                status["candidates"].append(row)
                write_status(status)
                continue
            extraction_action = "executed"
        if not candidate_dir.joinpath("paths.json").is_file() or candidate_dir.joinpath("paths.json").read_bytes() != frozen_manifest.read_bytes():
            row = {"epoch": epoch, "checkpoint": path_record(checkpoint), "status": "manifest_mismatch", "candidate_manifest": path_record(candidate_dir / "paths.json"), "frozen_manifest": manifest_fingerprint(frozen_manifest)}
            adaptation_errors.append({
                "stage": "adapted_extraction",
                "kind": "manifest_mismatch",
                "epoch": epoch,
                "message": "adapted paths.json does not match frozen paths.json",
                "return_code": None,
            })
            status["candidates"].append(row)
            write_status(status)
            continue
        gate_action = "cache_reuse" if gate_cache_valid(
            candidate_gate,
            paths_manifest=frozen_manifest,
            frozen_embedding=frozen_main,
            adapted_embedding=candidate_main,
            frozen_aug=frozen_weak,
            adapted_aug=candidate_weak,
            seed=args.seed,
            augmentation_tolerance=args.augmentation_tolerance,
        ) else "execute"
        if gate_action == "execute":
            try:
                code = run_logged(f"gate_ep{epoch:02d}", gate_command(args, manifest=frozen_manifest, frozen=frozen_main, frozen_aug=frozen_weak, adapted=candidate_main, adapted_aug=candidate_weak, out=candidate_gate), steps / f"gate_ep{epoch:02d}.log", status)
            except (OSError, subprocess.SubprocessError) as exc:
                row = {"epoch": epoch, "checkpoint": path_record(checkpoint), "status": "gate_failed", "return_code": None, "error": str(exc)}
                adaptation_errors.append({
                    "stage": "adapted_gate",
                    "kind": "launch_error",
                    "epoch": epoch,
                    "message": str(exc),
                    "return_code": None,
                })
                status["candidates"].append(row)
                write_status(status)
                continue
            if code != 0:
                row = {"epoch": epoch, "checkpoint": path_record(checkpoint), "status": "gate_failed", "return_code": code}
                adaptation_errors.append({
                    "stage": "adapted_gate",
                    "kind": "subprocess_failure",
                    "epoch": epoch,
                    "message": f"adapted gate failed with return code {code}",
                    "return_code": code,
                })
                status["candidates"].append(row)
                write_status(status)
                continue
        try:
            gate_value = load_json(candidate_gate)
        except (OSError, ValueError, WorkflowError) as exc:
            row = {"epoch": epoch, "checkpoint": path_record(checkpoint), "status": "gate_output_invalid", "error": str(exc)}
            adaptation_errors.append({
                "stage": "adapted_gate",
                "kind": "invalid_output",
                "epoch": epoch,
                "message": str(exc),
                "return_code": None,
            })
            status["candidates"].append(row)
            write_status(status)
            continue
        row = {
            "epoch": epoch,
            "checkpoint": path_record(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "extraction_action": extraction_action,
            "gate_action": gate_action,
            "gate": gate_value,
            "eligible": candidate_eligible(gate_value),
            "artifacts": artifact_list([candidate_dir / "paths.json", candidate_main, candidate_weak, candidate_dir / "provenance.json", candidate_gate]),
            "status": "eligible" if candidate_eligible(gate_value) else "rejected",
        }
        candidates.append(row)
        status["candidates"].append(row)
        write_status(status)

    eligible = [row for row in candidates if row["eligible"]]
    if eligible:
        winner = sorted(eligible, key=candidate_key)[0]
        winner_epoch = int(winner["epoch"])
        winner_dir = out_dir / "candidates" / f"ep{winner_epoch:02d}"
        final_group_dir = out_dir / "grouping"
        try:
            code = run_logged("group_adapted", group_command(args, manifest=frozen_manifest, embedding=winner_dir / "main.npy", gate=winner_dir / "gate.json", out=final_group_dir, mode="adapted"), steps / "group_adapted.log", status)
        except (OSError, subprocess.SubprocessError) as exc:
            return finalize_operational_frozen_fallback(
                args=args,
                out_dir=out_dir,
                steps=steps,
                status=status,
                frozen_manifest=frozen_manifest,
                frozen_main=frozen_main,
                frozen_gate_value=frozen_gate_value,
                adapted_error={
                    "stage": "adapted_grouping",
                    "kind": "launch_error",
                    "epoch": winner_epoch,
                    "message": str(exc),
                    "return_code": None,
                },
            )
        if code != 0:
            return finalize_operational_frozen_fallback(
                args=args,
                out_dir=out_dir,
                steps=steps,
                status=status,
                frozen_manifest=frozen_manifest,
                frozen_main=frozen_main,
                frozen_gate_value=frozen_gate_value,
                adapted_error={
                    "stage": "adapted_grouping",
                    "kind": "subprocess_failure",
                    "epoch": winner_epoch,
                    "message": f"adapted grouping failed with return code {code}",
                    "return_code": code,
                },
            )
        selection = {
            "selected_mode": "adapted",
            "workflow_action": "use_adapted",
            "reason": "label_free_lexicographic_candidate_selection",
            "ranking": ["max adapted.non_noise_count", "max adapted.within_group_cosine_coherence", "max adapted.bootstrap_stability", "min adapted.fragmentation_proxy", "min epoch"],
            "winner": winner,
            "eligible_epochs": [int(row["epoch"]) for row in sorted(eligible, key=lambda item: int(item["epoch"]))],
        }
        status["selection"] = selection
        status["final"] = {"workflow_action": "use_adapted", "selected_mode": "adapted", "epoch": winner_epoch, "grouping_dir": str(final_group_dir), "artifacts": artifact_list([final_group_dir / "clusters.csv", final_group_dir / "summary.json"]), "exit_code": 0}
        atomic_json(out_dir / "selection.json", selection)
        write_status(status)
        return 0

    if adaptation_errors:
        return finalize_operational_frozen_fallback(
            args=args,
            out_dir=out_dir,
            steps=steps,
            status=status,
            frozen_manifest=frozen_manifest,
            frozen_main=frozen_main,
            frozen_gate_value=frozen_gate_value,
            adapted_error={
                "stage": "adapted_candidate_pipeline",
                "kind": "candidate_operational_failures",
                "message": "no adapted candidate was selectable after operational failures",
                "failures": adaptation_errors,
                "return_code": None,
            },
        )

    # No candidate passed the label-free contract.  Group the frozen output as
    # the safe operational result, but expose the requested human-review state.
    fallback_gate = out_dir / "frozen_fallback_gate.json"
    fallback = dict(frozen_gate_value)
    fallback["workflow_action"] = "pseudo_tapt_review_required"
    fallback["selected_mode"] = "frozen"
    fallback["selection_reason"] = "no_adapted_candidate_passed_label_free_gate"
    atomic_json(fallback_gate, fallback)
    final_group_dir = out_dir / "grouping"
    code = run_logged("group_frozen_fallback", group_command(args, manifest=frozen_manifest, embedding=frozen_main, gate=fallback_gate, out=final_group_dir, mode="frozen"), steps / "group_frozen_fallback.log", status)
    if code != 0:
        raise WorkflowError(f"frozen fallback grouping failed with return code {code}")
    selection = {
        "selected_mode": "frozen",
        "workflow_action": "pseudo_tapt_review_required",
        "reason": "no_adapted_candidate_passed_label_free_gate",
        "adapted_candidates_considered": len(candidates),
        "fallback_gate": path_record(fallback_gate),
        "tapt_performed": False,
    }
    status["selection"] = selection
    status["final"] = {"workflow_action": "pseudo_tapt_review_required", "selected_mode": "frozen", "grouping_dir": str(final_group_dir), "artifacts": artifact_list([final_group_dir / "clusters.csv", final_group_dir / "summary.json"]), "exit_code": 30}
    atomic_json(out_dir / "selection.json", selection)
    write_status(status)
    return 30


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, WorkflowError, subprocess.SubprocessError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
