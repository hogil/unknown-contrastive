#!/usr/bin/env python3
"""Screen one temperature axis for the hard-42 FCMAE residual adapter.

The existing seed-1, epoch-4 temperature 0.05 result is reused. New runs change
only temperature. Labels are used by the fixed development evaluator, never by
training. This script reports candidates but never launches a multi-seed run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import fcmae_fixed_protocol as protocol  # noqa: E402


TRAINER = (REPO / "_ssl_methods.py").resolve()
RUN_ROOT = (REPO / "runs/fcmae_adapter_temperature_screen_260725").resolve()
EMBEDDINGS = (RUN_ROOT / "embeddings").resolve()
RUN_LOG = (RUN_ROOT / "train.log").resolve()
CONFIG_JSON = (RUN_ROOT / "config.json").resolve()
PAPER_ROOT = (REPO / "docs/paper").resolve()
OUTPUT_JSON = (PAPER_ROOT / "FCMAE_ADAPTER_TEMP_SCREEN_260725.json").resolve()
OUTPUT_CSV = (PAPER_ROOT / "FCMAE_ADAPTER_TEMP_SCREEN_260725.csv").resolve()
OUTPUT_MD = (PAPER_ROOT / "FCMAE_ADAPTER_TEMP_SCREEN_260725.md").resolve()
SCREEN_PROTOCOL = (PAPER_ROOT / "FCMAE_ADAPTER_TEMP_SCREEN_260725_protocol.json").resolve()
SCREEN_MANIFEST = (
    PAPER_ROOT / "FCMAE_ADAPTER_TEMP_SCREEN_260725_eval_manifest.json"
).resolve()

SEED = 1
EPOCH = 4
TEMPERATURES = (0.05, 0.07, 0.10, 0.20)
SOURCE_CHECKPOINT = (
    REPO
    / "runs/fcmae_adapter_ep4_three_seed_260721/embeddings/fcmae_ad1_s1_ckpt.pt"
).resolve()
EXPECTED_GSTEP = 3296
PROVENANCE_SCHEMA = 2
PROGRESS_SCHEMA = 2


class StaleProgressError(RuntimeError):
    """A partial temperature cell cannot be resumed without explicit cleanup."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def temperature_tag(value: float) -> str:
    return f"t{int(round(value * 100)):03d}"


def embedding_path(value: float) -> Path:
    tag = f"fcmae_ad1_{temperature_tag(value)}_s{SEED}"
    return EMBEDDINGS / f"{tag}_ep{EPOCH}.npy"


def checkpoint_path(value: float) -> Path:
    tag = f"fcmae_ad1_{temperature_tag(value)}_s{SEED}"
    return EMBEDDINGS / f"{tag}_ckpt.pt"


def embedding_sidecar_path(value: float) -> Path:
    path = embedding_path(value)
    return path.with_name(f"{path.name}.provenance.json")


def progress_sidecar_path(value: float) -> Path:
    tag = f"fcmae_ad1_{temperature_tag(value)}_s{SEED}"
    return EMBEDDINGS / f"{tag}.progress.json"


def training_command(value: float) -> list[str]:
    tag = f"fcmae_ad1_{temperature_tag(value)}_s{SEED}"
    return [
        sys.executable,
        "-u",
        str(TRAINER),
        "--method",
        "simclr",
        "--timm",
        "convnextv2_base.fcmae_ft_in22k_in1k_384",
        "--freeze-backbone",
        "--head",
        "adapter",
        "--pdim",
        "128",
        "--seed",
        str(SEED),
        "--epochs",
        str(EPOCH),
        "--batch",
        "8",
        "--ckpt-every",
        "100",
        "--temp",
        str(value),
        "--train-dir",
        str(protocol.TRAIN_DIR.resolve()),
        "--eval-dir",
        str(protocol.EVAL_DIR.resolve()),
        "--wafer-rot-deg",
        "0",
        "--wafer-translate",
        "0",
        "--wafer-scale-min",
        "1.0",
        "--wafer-crop-min",
        "1.0",
        "--out-dir",
        str(EMBEDDINGS),
        "--tag",
        tag,
    ] + ([] if value == 0.05 else ["--fresh"])


def resume_command(command: list[str]) -> list[str]:
    if command.count("--fresh") != 1:
        raise ValueError("non-control fresh command must contain exactly one --fresh")
    return [item for item in command if item != "--fresh"]


def progress_contract(
    value: float,
    command: list[str],
    context: dict[str, Any],
) -> dict[str, Any]:
    source = Path(__file__).resolve()
    protocol_source = Path(protocol.__file__).resolve()
    train_manifest = Path(context["train_manifest"]).resolve()
    eval_manifest = Path(context["eval_manifest"]).resolve()
    return {
        "schema": PROGRESS_SCHEMA,
        "axis": "temperature",
        "temperature": f"{value:.2f}",
        "seed": SEED,
        "epoch": EPOCH,
        "tag": f"fcmae_ad1_{temperature_tag(value)}_s{SEED}",
        "command": command,
        "command_sha256": sha256_json(command),
        "source": str(source),
        "source_sha256": sha256_file(source),
        "baseline_source_checkpoint": str(SOURCE_CHECKPOINT),
        "baseline_source_checkpoint_sha256": sha256_file(SOURCE_CHECKPOINT),
        "trainer": str(TRAINER),
        "trainer_sha256": sha256_file(TRAINER),
        "protocol_source": str(protocol_source),
        "protocol_source_sha256": sha256_file(protocol_source),
        "protocol_id": context["protocol_id"],
        "scorer_bundle_sha256": context["scorer_bundle_sha256"],
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": context["train_manifest_sha256"],
        "train_manifest_file_sha256": sha256_file(train_manifest),
        "eval_dir": context["eval_dir"],
        "eval_manifest": str(eval_manifest),
        "eval_manifest_sha256": context["eval_manifest_sha256"],
        "eval_manifest_file_sha256": sha256_file(eval_manifest),
    }


def checkpoint_validation(
    value: float,
    *,
    require_complete: bool = False,
) -> tuple[bool, str, dict[str, Any]]:
    checkpoint = checkpoint_path(value)
    if not checkpoint.is_file():
        return False, "checkpoint is missing", {}
    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as error:
        return False, f"checkpoint load failed: {type(error).__name__}", {}
    if not isinstance(payload, Mapping):
        return False, "checkpoint payload is not a mapping", {}
    required = {"model", "opt", "queue", "qptr", "gstep", "center", "target"}
    missing = sorted(required - set(payload))
    if missing:
        return False, f"checkpoint keys are missing: {', '.join(missing)}", {}
    if not isinstance(payload["model"], Mapping) or not payload["model"]:
        return False, "checkpoint model state is empty", {}
    if not isinstance(payload["opt"], Mapping):
        return False, "checkpoint optimizer state is invalid", {}
    try:
        gstep = int(payload["gstep"])
        qptr = int(payload["qptr"])
        queue_shape = tuple(int(item) for item in payload["queue"].shape)
        center_shape = tuple(int(item) for item in payload["center"].shape)
    except (AttributeError, TypeError, ValueError):
        return False, "checkpoint tensor or counter metadata is invalid", {}
    if require_complete:
        if gstep != EXPECTED_GSTEP:
            return False, f"checkpoint gstep is not {EXPECTED_GSTEP}", {}
    elif gstep <= 0 or gstep > EXPECTED_GSTEP:
        return False, f"checkpoint gstep is outside 1..{EXPECTED_GSTEP}", {}
    if len(queue_shape) != 2 or queue_shape[1] != 128:
        return False, "checkpoint queue projection dimension is not 128", {}
    if qptr < 0 or qptr >= queue_shape[0]:
        return False, "checkpoint queue pointer is outside the queue", {}
    if len(center_shape) != 2:
        return False, "checkpoint center tensor is invalid", {}
    metadata = {
        "path": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
        "gstep": gstep,
        "qptr": qptr,
        "queue_shape": list(queue_shape),
        "center_shape": list(center_shape),
    }
    return True, "valid", metadata


def _read_progress_sidecar(value: float) -> dict[str, Any] | None:
    sidecar = progress_sidecar_path(value)
    if not sidecar.is_file():
        return None
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise StaleProgressError(
            f"temperature {value:.2f} progress sidecar is unreadable; "
            "use --clean-stale for an explicit fresh start"
        ) from error
    if not isinstance(payload, dict):
        raise StaleProgressError(
            f"temperature {value:.2f} progress sidecar is invalid; "
            "use --clean-stale for an explicit fresh start"
        )
    return payload


def _write_progress_state(
    value: float,
    contract: dict[str, Any],
    *,
    state: str,
    launch_mode: str,
    checkpoint: dict[str, Any] | None = None,
    exit_code: int | None = None,
) -> None:
    existing = _read_progress_sidecar(value)
    now = datetime.now().astimezone().isoformat()
    attempts = list(existing.get("attempts", [])) if existing else []
    attempt = {
        "prepared_at": now,
        "launch_mode": launch_mode,
        "checkpoint": checkpoint,
    }
    if exit_code is not None:
        attempt["exit_code"] = exit_code
    attempts.append(attempt)
    atomic_write_json(
        progress_sidecar_path(value),
        {
            "created_at": existing.get("created_at", now) if existing else now,
            "updated_at": now,
            "state": state,
            "contract": contract,
            "last_launch": attempt,
            "attempts": attempts,
        },
    )


def _update_progress_after_launch(
    value: float,
    contract: dict[str, Any],
    *,
    state: str,
    launch_mode: str,
    exit_code: int,
    checkpoint_after: dict[str, Any] | None,
) -> None:
    existing = _read_progress_sidecar(value)
    if existing is None or existing.get("contract") != contract:
        raise StaleProgressError(
            f"temperature {value:.2f} progress changed during launch"
        )
    attempts = list(existing.get("attempts", []))
    if not attempts:
        raise StaleProgressError(
            f"temperature {value:.2f} progress has no prepared launch"
        )
    if attempts[-1].get("launch_mode") != launch_mode:
        raise StaleProgressError(
            f"temperature {value:.2f} launch mode changed during execution"
        )
    attempts[-1] = {
        **attempts[-1],
        "finished_at": datetime.now().astimezone().isoformat(),
        "exit_code": exit_code,
        "checkpoint_after": checkpoint_after,
    }
    now = datetime.now().astimezone().isoformat()
    atomic_write_json(
        progress_sidecar_path(value),
        {
            **existing,
            "updated_at": now,
            "state": state,
            "last_launch": attempts[-1],
            "attempts": attempts,
        },
    )


def _temperature_artifacts(value: float) -> list[Path]:
    tag = f"fcmae_ad1_{temperature_tag(value)}_s{SEED}"
    paths = list(EMBEDDINGS.glob(f"{tag}_ep*.npy"))
    for path in (checkpoint_path(value), embedding_sidecar_path(value)):
        if path.exists():
            paths.append(path)
    return sorted(set(paths))


def prepare_non_control_launch(
    value: float,
    command: list[str],
    context: dict[str, Any],
    *,
    clean_stale: bool,
) -> tuple[list[str], dict[str, Any], str]:
    if value == 0.05:
        raise ValueError("temperature 0.05 does not use the progress resume path")
    contract = progress_contract(value, command, context)
    progress = progress_sidecar_path(value)
    if clean_stale:
        remove_stale_temperature_artifacts(
            value, remove_checkpoint=True, remove_progress=True
        )

    existing = _read_progress_sidecar(value)
    artifacts = _temperature_artifacts(value)
    if existing is None:
        if artifacts:
            raise StaleProgressError(
                f"temperature {value:.2f} has artifacts without an exact progress "
                "sidecar; files were preserved. Use --clean-stale for an explicit "
                "fresh start"
            )
        launch_mode = "fresh"
        checkpoint_metadata = None
        launch_command = command
    else:
        if existing.get("contract") != contract:
            raise StaleProgressError(
                f"temperature {value:.2f} progress provenance mismatch; files were "
                "preserved. Use --clean-stale for an explicit fresh start"
            )
        if embedding_sidecar_path(value).exists():
            raise StaleProgressError(
                f"temperature {value:.2f} has completed embedding provenance that "
                "is not reusable; files were preserved"
            )
        if checkpoint_path(value).is_file():
            valid, reason, checkpoint_metadata = checkpoint_validation(value)
            if not valid:
                raise StaleProgressError(
                    f"temperature {value:.2f} checkpoint is invalid ({reason}); "
                    "files were preserved. Use --clean-stale for an explicit fresh "
                    "start"
                )
            last_launch = existing.get("last_launch")
            expected_checkpoint = (
                last_launch.get("checkpoint_after")
                if isinstance(last_launch, dict)
                else None
            )
            if (
                existing.get("state") != "failed"
                or not isinstance(expected_checkpoint, dict)
                or expected_checkpoint.get("sha256")
                != checkpoint_metadata.get("sha256")
            ):
                raise StaleProgressError(
                    f"temperature {value:.2f} checkpoint is not bound to a "
                    "recorded failed launch; files were preserved. Use "
                    "--clean-stale for an explicit fresh start"
                )
            launch_mode = "resume"
            launch_command = resume_command(command)
        else:
            launch_mode = "fresh"
            checkpoint_metadata = None
            launch_command = command

    _write_progress_state(
        value,
        contract,
        state="launch_prepared",
        launch_mode=launch_mode,
        checkpoint=checkpoint_metadata,
    )
    if not progress.is_file():
        raise RuntimeError(f"progress sidecar was not written: {progress}")
    return launch_command, contract, launch_mode


def embedding_contract(
    value: float,
    context: dict[str, Any],
    launch_contract: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = checkpoint_path(value)
    protocol_source = Path(protocol.__file__).resolve()
    eval_manifest = Path(context["eval_manifest"]).resolve()
    baseline_origin = (
        {
            "path": str(SOURCE_CHECKPOINT),
            "sha256": sha256_file(SOURCE_CHECKPOINT),
        }
        if value == 0.05
        else None
    )
    return {
        "schema": PROVENANCE_SCHEMA,
        "axis": "temperature",
        "temperature": f"{value:.2f}",
        "seed": SEED,
        "epoch": EPOCH,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "baseline_origin_checkpoint": baseline_origin,
        "launch_contract": launch_contract,
        "launch_contract_sha256": sha256_json(launch_contract),
        "runner": launch_contract["source"],
        "runner_sha256": launch_contract["source_sha256"],
        "trainer": launch_contract["trainer"],
        "trainer_sha256": launch_contract["trainer_sha256"],
        "protocol_source": launch_contract["protocol_source"],
        "protocol_source_sha256": launch_contract[
            "protocol_source_sha256"
        ],
        "scorer_bundle_sha256": context["scorer_bundle_sha256"],
        "train_manifest": context["train_manifest"],
        "train_manifest_sha256": context["train_manifest_sha256"],
        "eval_dir": context["eval_dir"],
        "eval_manifest": str(eval_manifest),
        "eval_manifest_sha256": context["eval_manifest_sha256"],
        "eval_manifest_file_sha256": sha256_file(eval_manifest),
        "command_sha256": sha256_json(training_command(value)),
    }


def embedding_is_reusable(value: float, context: dict[str, Any]) -> bool:
    embedding = embedding_path(value)
    checkpoint = checkpoint_path(value)
    sidecar = embedding_sidecar_path(value)
    if not embedding.is_file() or not checkpoint.is_file() or not sidecar.is_file():
        return False
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        launch_contract = progress_contract(
            value,
            training_command(value),
            context,
        )
        valid_checkpoint, _, checkpoint_metadata = checkpoint_validation(
            value,
            require_complete=True,
        )
        if not valid_checkpoint:
            return False
        if value != 0.05:
            progress = _read_progress_sidecar(value)
            if (
                progress is None
                or progress.get("state") != "completed"
                or progress.get("contract") != launch_contract
                or not isinstance(progress.get("last_launch"), dict)
                or progress["last_launch"].get("checkpoint_after", {}).get(
                    "sha256"
                )
                != checkpoint_metadata["sha256"]
            ):
                return False
        return (
            payload.get("contract")
            == embedding_contract(value, context, launch_contract)
            and payload.get("embedding") == str(embedding)
            and payload.get("embedding_sha256") == sha256_file(embedding)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def write_embedding_sidecar(
    value: float,
    context: dict[str, Any],
    launch_contract: dict[str, Any],
) -> None:
    embedding = embedding_path(value)
    current_contract = progress_contract(
        value,
        training_command(value),
        context,
    )
    if launch_contract != current_contract:
        raise StaleProgressError(
            f"temperature {value:.2f} source provenance changed during launch"
        )
    atomic_write_json(
        embedding_sidecar_path(value),
        {
            "created_at": datetime.now().astimezone().isoformat(),
            "contract": embedding_contract(
                value,
                context,
                launch_contract,
            ),
            "embedding": str(embedding),
            "embedding_sha256": sha256_file(embedding),
        },
    )


def remove_stale_temperature_artifacts(
    value: float, *, remove_checkpoint: bool, remove_progress: bool = False
) -> None:
    tag = f"fcmae_ad1_{temperature_tag(value)}_s{SEED}"
    for path in EMBEDDINGS.glob(f"{tag}_ep*.npy"):
        path.unlink(missing_ok=True)
    embedding_sidecar_path(value).unlink(missing_ok=True)
    if remove_checkpoint:
        checkpoint_path(value).unlink(missing_ok=True)
    if remove_progress:
        progress_sidecar_path(value).unlink(missing_ok=True)


def stage_baseline_checkpoint() -> None:
    if not SOURCE_CHECKPOINT.is_file():
        raise FileNotFoundError(f"missing source checkpoint: {SOURCE_CHECKPOINT}")
    target = checkpoint_path(0.05)
    source_hash = sha256_file(SOURCE_CHECKPOINT)
    if target.is_file() and sha256_file(target) == source_hash:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    shutil.copyfile(SOURCE_CHECKPOINT, temporary)
    os.replace(temporary, target)
    if sha256_file(target) != source_hash:
        raise RuntimeError("staged temperature 0.05 checkpoint hash mismatch")


def spec(value: float) -> dict[str, Any]:
    return {
        "recipe": f"adapter_temp_{value:.2f}",
        "recipe_flags": (
            "no-TAPT FCMAE frozen backbone; residual adapter; pure SimCLR; "
            f"temperature={value:.2f}; fixed seed1 epoch4"
        ),
        "seed": SEED,
        "epoch": EPOCH,
        "embedding_space": "adapted_f",
        "path": embedding_path(value),
    }


def frozen_spec() -> dict[str, Any]:
    return {
        "recipe": "frozen",
        "recipe_flags": "FCMAE frozen; no training",
        "seed": "none",
        "epoch": 0,
        "embedding_space": "backbone_f",
        "path": protocol.FROZEN_EMBEDDING,
    }


def screen(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frozen = protocol.frozen_by_clusterer(rows)
    values: list[dict[str, Any]] = []
    for value in TEMPERATURES:
        recipe = f"adapter_temp_{value:.2f}"
        candidate_rows = [row for row in rows if row["recipe"] == recipe]
        if {row["clusterer"] for row in candidate_rows} != set(frozen):
            raise RuntimeError(f"missing clusterer row for temperature {value:.2f}")
        clusterers: dict[str, Any] = {}
        minimum_primary_delta = float("inf")
        accepted = True
        for row in candidate_rows:
            baseline = frozen[row["clusterer"]]
            deltas = {
                "P1_capture_count": int(row["P1_capture_count"])
                - int(baseline["P1_capture_count"]),
                "P2_noise_pct": float(row["P2_noise_pct"])
                - float(baseline["P2_noise_pct"]),
                "P3_completeness": float(row["P3_completeness"])
                - float(baseline["P3_completeness"]),
                "P4_homogeneity": float(row["P4_homogeneity"])
                - float(baseline["P4_homogeneity"]),
                "fragment_ratio": float(row["fragment_ratio"])
                - float(baseline["fragment_ratio"]),
            }
            checks = {
                "P1_preserved": deltas["P1_capture_count"] >= 0,
                "P2_not_worse": deltas["P2_noise_pct"] <= 1e-9,
                "P3_not_worse": deltas["P3_completeness"] >= -1e-9,
                "P4_not_worse": deltas["P4_homogeneity"] >= -1e-9,
            }
            cluster_accepted = all(checks.values())
            accepted = accepted and cluster_accepted
            minimum_primary_delta = min(
                minimum_primary_delta,
                deltas["P3_completeness"],
                deltas["P4_homogeneity"],
            )
            clusterers[row["clusterer"]] = {
                "accepted": cluster_accepted,
                "checks": checks,
                "delta": {key: round(value_, 6) for key, value_ in deltas.items()},
            }
        values.append(
            {
                "temperature": value,
                "accepted": accepted,
                "minimum_P3_P4_delta": round(minimum_primary_delta, 6),
                "clusterers": clusterers,
            }
        )
    accepted_values = sorted(
        (item for item in values if item["accepted"]),
        key=lambda item: (-item["minimum_P3_P4_delta"], item["temperature"]),
    )
    return {
        "contract": {
            "P1": "preserve frozen capture in FINCH-p2 and Louvain-res6",
            "P2": "do not increase frozen target noise in either clusterer",
            "P3_P4": "both must be non-worse in both clusterers",
            "ARI_AMI": "supporting only; excluded from screening",
            "selection": "proposal only; no automatic adoption or multi-seed launch",
        },
        "values": values,
        "proposed_temperature": (
            accepted_values[0]["temperature"] if accepted_values else None
        ),
    }


def write_outputs(
    rows: list[dict[str, Any]],
    gate: dict[str, Any],
    context: dict[str, Any],
    commands: dict[str, list[str]],
) -> None:
    PAPER_ROOT.mkdir(parents=True, exist_ok=True)
    protocol.write_csv(OUTPUT_CSV, rows)
    lines = [
        "# FCMAE Residual Adapter Temperature Screen (260725)",
        "",
        "- One-axis screen: temperature only",
        "- Baseline: temperature 0.05, seed 1, fixed epoch 4",
        "- New values: 0.07, 0.10, 0.20",
        "- P1/P2/P3/P4 decide screening; ARI/AMI are supporting only.",
        "- This report proposes a candidate but never launches multi-seed validation.",
        "",
        "## Rows",
        "",
        *protocol.row_table(rows),
        "",
        "## Gate",
        "",
    ]
    for item in gate["values"]:
        lines.append(
            f"- temp {item['temperature']:.2f}: accepted={item['accepted']}, "
            f"minimum P3/P4 delta={item['minimum_P3_P4_delta']:.6f}"
        )
    lines.extend(
        [
            "",
            f"- proposed temperature: {gate['proposed_temperature']}",
            "",
            "## Absolute Outputs",
            "",
            f"- JSON: `{OUTPUT_JSON}`",
            f"- CSV: `{OUTPUT_CSV}`",
            f"- run root: `{RUN_ROOT}`",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "protocol_id": context["protocol_id"],
        "screen": gate,
        "rows": rows,
        "provenance": {
            "protocol_id": context["protocol_id"],
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "trainer": str(TRAINER),
            "trainer_sha256": sha256_file(TRAINER),
            "protocol_source": str(Path(protocol.__file__).resolve()),
            "protocol_source_sha256": sha256_file(
                Path(protocol.__file__).resolve()
            ),
            "baseline_source_checkpoint": str(SOURCE_CHECKPOINT),
            "baseline_source_checkpoint_sha256": sha256_file(SOURCE_CHECKPOINT),
            "commands": commands,
            "scorer_bundle_sha256": context["scorer_bundle_sha256"],
            "train_manifest": context["train_manifest"],
            "train_manifest_sha256": context["train_manifest_sha256"],
            "train_manifest_file_sha256": sha256_file(
                Path(context["train_manifest"])
            ),
            "eval_manifest": context["eval_manifest"],
            "eval_manifest_sha256": context["eval_manifest_sha256"],
            "eval_manifest_file_sha256": sha256_file(
                Path(context["eval_manifest"])
            ),
            "embedding_sidecars": {
                f"{value:.2f}": {
                    "path": str(embedding_sidecar_path(value)),
                    "sha256": sha256_file(embedding_sidecar_path(value)),
                }
                for value in TEMPERATURES
            },
        },
        "outputs": {
            "json": str(OUTPUT_JSON),
            "csv": str(OUTPUT_CSV),
            "csv_sha256": sha256_file(OUTPUT_CSV),
            "markdown": str(OUTPUT_MD),
            "markdown_sha256": sha256_file(OUTPUT_MD),
        },
    }
    atomic_write_json(OUTPUT_JSON, payload)


def run_training(
    value: float,
    command: list[str],
    context: dict[str, Any],
    *,
    clean_stale: bool = False,
) -> None:
    expected = embedding_path(value)
    if embedding_is_reusable(value, context):
        print(f"[skip] {expected}", flush=True)
        return
    if value == 0.05:
        stage_baseline_checkpoint()
        remove_stale_temperature_artifacts(value, remove_checkpoint=False)
        checkpoint_before = sha256_file(checkpoint_path(value))
        launch_command = command
        progress_contract_value = progress_contract(value, command, context)
        launch_mode = "control"
    else:
        launch_command, progress_contract_value, launch_mode = (
            prepare_non_control_launch(
                value,
                command,
                context,
                clean_stale=clean_stale,
            )
        )
        checkpoint_before = None
    EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    captured: list[str] = []
    with RUN_LOG.open("a", encoding="utf-8") as log:
        log.write(
            f"\n[{datetime.now().astimezone().isoformat()}] "
            f"temperature={value:.2f}\n"
        )
        log.flush()
        process = subprocess.Popen(
            launch_command,
            cwd=REPO,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            if value == 0.05:
                captured.append(line)
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if value != 0.05:
        assert progress_contract_value is not None
        valid_checkpoint, checkpoint_reason, checkpoint_after = (
            checkpoint_validation(
                value,
                require_complete=code == 0,
            )
        )
        _update_progress_after_launch(
            value,
            progress_contract_value,
            state="child_completed" if code == 0 else "failed",
            launch_mode=launch_mode,
            exit_code=code,
            checkpoint_after=(
                checkpoint_after if valid_checkpoint else None
            ),
        )
        if code == 0 and not valid_checkpoint:
            raise RuntimeError(
                f"temperature {value:.2f} completed with invalid checkpoint "
                f"({checkpoint_reason})"
            )
    if code != 0:
        raise RuntimeError(
            f"temperature {value:.2f} training failed with exit code {code}"
        )
    if not expected.is_file():
        raise RuntimeError(f"missing expected embedding: {expected}")
    checkpoint = checkpoint_path(value)
    if not checkpoint.is_file():
        raise RuntimeError(f"missing source checkpoint for embedding: {checkpoint}")
    if value == 0.05:
        output = "".join(captured)
        if not re.search(
            rf"gstep\s+{EXPECTED_GSTEP}/{EXPECTED_GSTEP}", output
        ):
            raise RuntimeError("temperature 0.05 did not resume at fixed epoch 4")
        if re.search(r"(?m)^\[simclr ep\d+\]\s+loss=", output):
            raise RuntimeError("temperature 0.05 unexpectedly performed training")
        if sha256_file(checkpoint) != checkpoint_before:
            raise RuntimeError("temperature 0.05 extraction changed its checkpoint")
    assert progress_contract_value is not None
    if value == 0.05:
        valid_checkpoint, checkpoint_reason, _ = checkpoint_validation(
            value,
            require_complete=True,
        )
        if not valid_checkpoint:
            raise RuntimeError(
                f"temperature 0.05 control checkpoint is incomplete "
                f"({checkpoint_reason})"
            )
    write_embedding_sidecar(value, context, progress_contract_value)
    if value != 0.05:
        assert progress_contract_value is not None
        _update_progress_after_launch(
            value,
            progress_contract_value,
            state="completed",
            launch_mode=launch_mode,
            exit_code=0,
            checkpoint_after=checkpoint_after,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--score-only", action="store_true")
    parser.add_argument(
        "--clean-stale",
        action="store_true",
        help=(
            "explicitly delete stale non-control cell artifacts before a fresh "
            "start; never used by the guard"
        ),
    )
    args = parser.parse_args()
    if args.clean_stale and (args.dry_run or args.score_only):
        parser.error("--clean-stale requires a training run")

    commands = {f"{value:.2f}": training_command(value) for value in TEMPERATURES}
    config = {
        "created_at": datetime.now().astimezone().isoformat(),
        "axis": "temperature",
        "seed": SEED,
        "fixed_epoch": EPOCH,
        "temperatures": list(TEMPERATURES),
        "baseline_source_checkpoint": str(SOURCE_CHECKPOINT),
        "commands": commands,
        "progress_resume": {
            "sidecar": "atomic exact-provenance progress JSON per non-control cell",
            "mismatch": "fail closed unless --clean-stale is explicit",
        },
        "automatic_adoption": False,
    }
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(CONFIG_JSON, config)
    if args.dry_run:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return 0
    if not SOURCE_CHECKPOINT.is_file():
        raise FileNotFoundError(f"missing source checkpoint: {SOURCE_CHECKPOINT}")
    protocol.PROTOCOL_JSON = SCREEN_PROTOCOL
    protocol.EVAL_MANIFEST = SCREEN_MANIFEST
    context = protocol.protocol_context()
    if not args.score_only:
        for value in TEMPERATURES:
            run_training(
                value,
                commands[f"{value:.2f}"],
                context,
                clean_stale=args.clean_stale,
            )
    for value in TEMPERATURES:
        if not embedding_is_reusable(value, context):
            raise RuntimeError(
                f"missing or stale temperature {value:.2f} embedding provenance"
            )

    rows = protocol.score_specs(
        [frozen_spec(), *(spec(value) for value in TEMPERATURES)], context
    )
    gate = screen(rows)
    write_outputs(rows, gate, context, commands)
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    print(f"[out] {OUTPUT_JSON}")
    print(f"[out] {OUTPUT_CSV}")
    print(f"[out] {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
