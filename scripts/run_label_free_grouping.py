#!/usr/bin/env python3
"""Thin label-free frozen/adapted grouping orchestrator.

This module only coordinates existing scripts.  It never opens image files,
interprets parent folders, reads labels, or trains a model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
EXTRACTOR = (SCRIPTS / "extract_label_free_embeddings.py").resolve()
GATE = (SCRIPTS / "label_free_gate.py").resolve()
GROUPER = (SCRIPTS / "group_label_free_embeddings.py").resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    return value


def artifact_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    record: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.is_file():
        record["bytes"] = path.stat().st_size
        record["sha256"] = sha256_file(path)
    return record


def collect_artifacts(root: Path) -> list[dict[str, Any]]:
    """Record reproducibility artifacts without hashing copied review images."""
    if not root.exists():
        return []
    paths = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if "groups" in path.relative_to(root).parts:
            continue
        paths.append(artifact_record(path))
    return paths


def run_command(command: list[str], records: list[dict[str, Any]]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=str(REPO),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    records.append(
        {
            "command": [str(item) for item in command],
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    )
    return completed


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(payload), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run label-free frozen-first grouping with optional adapted rescue."
    )
    parser.add_argument("--pool", action="append", required=True, type=Path,
                        help="image pool; repeat for additional pools")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--adapted-checkpoint", type=Path)
    parser.add_argument("--adapted-mode", choices=("adapter", "projection"), default="adapter")
    parser.add_argument("--augmentation-tolerance", type=float, default=0.02)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--copy-groups", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.reps < 1:
        raise ValueError("--reps must be positive")
    if not 0.0 <= args.augmentation_tolerance <= 1.0:
        raise ValueError("--augmentation-tolerance must be in [0, 1]")

    out_dir = args.out_dir.expanduser().resolve()
    frozen_dir = out_dir / "frozen"
    adapted_dir = out_dir / "adapted"
    frozen_gate = out_dir / "frozen_gate.json"
    final_gate = out_dir / "final_gate.json"
    status_path = out_dir / "status.json"
    commands: list[dict[str, Any]] = []
    status: dict[str, Any] = {
        "schema_version": "label_free_grouping_run.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "repo": str(REPO),
        "arguments": vars(args),
        "label_free_contract": {
            "labels_read": False,
            "folder_names_read": False,
            "training_run": False,
        },
        "scripts": {
            "extract": artifact_record(EXTRACTOR),
            "gate": artifact_record(GATE),
            "grouper": artifact_record(GROUPER),
        },
        "subprocesses": commands,
        "decision": None,
        "outputs": [],
    }
    if args.adapted_checkpoint is not None:
        status["adapted_checkpoint"] = artifact_record(
            args.adapted_checkpoint.expanduser().resolve()
        )

    def finish(code: int) -> int:
        status["outputs"] = (
            collect_artifacts(frozen_dir)
            + collect_artifacts(adapted_dir)
            + collect_artifacts(out_dir / "groups")
            + [artifact_record(path) for path in (frozen_gate, final_gate) if path.exists()]
        )
        write_status(status_path, status)
        return code

    try:
        pools = [str(path.expanduser().resolve(strict=True)) for path in args.pool]
        if not GROUPER.is_file():
            status["decision"] = {"action": "error", "reason": "missing_group_script"}
            return finish(2)

        extract_frozen = [sys.executable, str(EXTRACTOR)]
        for pool in pools:
            extract_frozen.extend(("--pool", pool))
        extract_frozen.extend((
            "--mode", "frozen", "--out-dir", str(frozen_dir),
            "--device", args.device, "--batch-size", str(args.batch_size),
        ))
        result = run_command(extract_frozen, commands)
        if result.returncode:
            status["decision"] = {"action": "error", "reason": "frozen_extraction_failed"}
            return finish(result.returncode)

        frozen_manifest = frozen_dir / "paths.json"
        frozen_embedding = frozen_dir / "main.npy"
        frozen_weak = frozen_dir / "weak_aug.npy"
        frozen_gate_cmd = [
            sys.executable, str(GATE),
            "--paths-manifest", str(frozen_manifest),
            "--frozen-embedding", str(frozen_embedding),
            "--frozen-aug-embedding", str(frozen_weak),
            "--out", str(frozen_gate),
            "--augmentation-tolerance", str(args.augmentation_tolerance),
        ]
        result = run_command(frozen_gate_cmd, commands)
        if result.returncode:
            status["decision"] = {"action": "error", "reason": "frozen_gate_failed"}
            return finish(result.returncode)
        frozen_decision = json.loads(frozen_gate.read_text(encoding="utf-8"))
        frozen_action = frozen_decision.get("workflow_action")

        if frozen_action == "use_frozen":
            selected_mode = "frozen"
            selected_gate = frozen_gate
            selected_embedding = frozen_embedding
        elif frozen_action == "adapt_required":
            if args.adapted_checkpoint is None:
                status["decision"] = {
                    "action": "adapt_required",
                    "selected_mode": "none",
                    "gate": str(frozen_gate),
                    "reason": "frozen_gate_failed_and_no_adapted_checkpoint",
                }
                return finish(20)
            checkpoint = args.adapted_checkpoint.expanduser().resolve(strict=True)
            extract_adapted = [sys.executable, str(EXTRACTOR)]
            for pool in pools:
                extract_adapted.extend(("--pool", pool))
            extract_adapted.extend((
                "--mode", args.adapted_mode, "--checkpoint", str(checkpoint),
                "--out-dir", str(adapted_dir), "--device", args.device,
                "--batch-size", str(args.batch_size),
            ))
            result = run_command(extract_adapted, commands)
            if result.returncode:
                status["decision"] = {"action": "error", "reason": "adapted_extraction_failed"}
                return finish(result.returncode)
            final_gate_cmd = [
                sys.executable, str(GATE),
                "--paths-manifest", str(frozen_manifest),
                "--frozen-embedding", str(frozen_embedding),
                "--frozen-aug-embedding", str(frozen_weak),
                "--adapted-embedding", str(adapted_dir / "main.npy"),
                "--adapted-aug-embedding", str(adapted_dir / "weak_aug.npy"),
                "--out", str(final_gate),
                "--augmentation-tolerance", str(args.augmentation_tolerance),
            ]
            result = run_command(final_gate_cmd, commands)
            if result.returncode:
                status["decision"] = {"action": "error", "reason": "final_gate_failed"}
                return finish(result.returncode)
            final_decision = json.loads(final_gate.read_text(encoding="utf-8"))
            selected_mode = final_decision.get("selected_mode", "frozen")
            selected_gate = final_gate
            selected_embedding = (adapted_dir / "main.npy") if selected_mode == "adapted" else frozen_embedding
            final_action = final_decision.get("workflow_action")
            if final_action == "pseudo_tapt_review_required":
                selected_mode = "frozen"
                selected_embedding = frozen_embedding
        else:
            status["decision"] = {"action": "error", "reason": f"unexpected_frozen_action:{frozen_action}"}
            return finish(2)

        group_cmd = [
            sys.executable, str(GROUPER),
            "--paths-manifest", str(frozen_manifest),
            "--embedding", str(selected_embedding),
            "--gate-json", str(selected_gate),
            "--out-dir", str(out_dir / "groups"),
            "--model-mode", selected_mode,
            "--reps", str(args.reps),
        ]
        if args.copy_groups:
            group_cmd.append("--copy-groups")
        result = run_command(group_cmd, commands)
        if result.returncode:
            status["decision"] = {"action": "error", "reason": "grouping_failed"}
            return finish(result.returncode)

        gate_payload = json.loads(selected_gate.read_text(encoding="utf-8"))
        status["decision"] = {
            "action": gate_payload.get("workflow_action", "use_frozen"),
            "selected_mode": selected_mode,
            "selection_reason": gate_payload.get("selection_reason"),
            "gate": str(selected_gate.resolve()),
            "grouping_embedding": str(selected_embedding.resolve()),
        }
        return finish(0)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        status["decision"] = {"action": "error", "reason": type(exc).__name__, "message": str(exc)}
        return finish(2)


if __name__ == "__main__":
    raise SystemExit(main())
