#!/usr/bin/env python3
"""Screen queue size for the hard-42 FCMAE residual adapter.

This runner changes only queue size after a temperature has passed the
temperature screen. It is fail-closed when that accepted temperature or its
provenance is missing. Labels are used only by the fixed evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import fcmae_fixed_protocol as protocol  # noqa: E402


TRAINER = (REPO / "_ssl_methods.py").resolve()
TEMPERATURE_RUNNER = (
    REPO / "scripts/run_fcmae_adapter_temperature_screen.py"
).resolve()
TEMPERATURE_RESULT = (
    REPO / "docs/paper/FCMAE_ADAPTER_TEMP_SCREEN_260725.json"
).resolve()

RUN_ROOT = (REPO / "runs/fcmae_adapter_queue_size_screen_260725").resolve()
EMBEDDINGS = (RUN_ROOT / "embeddings").resolve()
RUN_LOG = (RUN_ROOT / "train.log").resolve()
CONFIG_JSON = (RUN_ROOT / "config.json").resolve()
PAPER_ROOT = (REPO / "docs/paper").resolve()
OUTPUT_JSON = (PAPER_ROOT / "FCMAE_ADAPTER_QUEUE_SCREEN_260725.json").resolve()
OUTPUT_CSV = (PAPER_ROOT / "FCMAE_ADAPTER_QUEUE_SCREEN_260725.csv").resolve()
OUTPUT_MD = (PAPER_ROOT / "FCMAE_ADAPTER_QUEUE_SCREEN_260725.md").resolve()
SCREEN_PROTOCOL = (
    PAPER_ROOT / "FCMAE_ADAPTER_QUEUE_SCREEN_260725_protocol.json"
).resolve()
SCREEN_MANIFEST = (
    PAPER_ROOT / "FCMAE_ADAPTER_QUEUE_SCREEN_260725_eval_manifest.json"
).resolve()

SEED = 3
EPOCH = 4
QUEUE_SIZES = (0, 1024, 2048, 4096)
PROVENANCE_SCHEMA = 1
HDBSCAN_CONFIG = {
    "min_cluster_size": 12,
    "min_samples": 15,
    "cluster_selection_method": "leaf",
    "cluster_selection_epsilon": 0.06,
}
PRIMARY_CLUSTERERS = {"FINCH-p2", "Louvain-res6"}
PRIMARY_FIELDS = {
    "P1_capture_count",
    "P1_target_class_count",
    "P2_noise_pct",
    "P3_completeness",
    "P4_homogeneity",
}


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


def queue_tag(value: int) -> str:
    return f"q{value:04d}"


def cell_tag(queue_size: int, temperature: float) -> str:
    return (
        f"fcmae_ad1_{queue_tag(queue_size)}_{temperature_tag(temperature)}"
        f"_s{SEED}"
    )


def embedding_path(queue_size: int, temperature: float) -> Path:
    return EMBEDDINGS / f"{cell_tag(queue_size, temperature)}_ep{EPOCH}.npy"


def checkpoint_path(queue_size: int, temperature: float) -> Path:
    return EMBEDDINGS / f"{cell_tag(queue_size, temperature)}_ckpt.pt"


def embedding_sidecar_path(queue_size: int, temperature: float) -> Path:
    path = embedding_path(queue_size, temperature)
    return path.with_name(f"{path.name}.provenance.json")


def training_command(queue_size: int, temperature: float) -> list[str]:
    command = [
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
        str(temperature),
        "--queue-size",
        str(queue_size),
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
        cell_tag(queue_size, temperature),
        "--fresh",
    ]
    if queue_size > 0:
        command.append("--use-queue")
    return command


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is invalid: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return payload


def load_accepted_temperature(context: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Return the accepted temperature only when its full evidence is current."""
    result = _read_json(TEMPERATURE_RESULT, "temperature screen result")
    try:
        screen = result["screen"]
        proposed_raw = screen["proposed_temperature"]
        values = screen["values"]
        rows = result["rows"]
        provenance = result["provenance"]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"temperature screen result is incomplete: {error}"
        ) from error
    if proposed_raw is None:
        raise RuntimeError("temperature screen has no accepted temperature")
    try:
        proposed = float(proposed_raw)
    except (TypeError, ValueError) as error:
        raise RuntimeError("accepted temperature is not numeric") from error
    if not math.isfinite(proposed) or proposed <= 0:
        raise RuntimeError("accepted temperature must be finite and positive")

    accepted = [
        item
        for item in values
        if isinstance(item, dict)
        and math.isclose(
            float(item.get("temperature", float("nan"))),
            proposed,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and item.get("accepted") is True
    ]
    if len(accepted) != 1:
        raise RuntimeError(
            "proposed temperature is not exactly one accepted screen value"
        )
    screen_contract = screen.get("contract", {})
    if "supporting only" not in str(screen_contract.get("ARI_AMI", "")).lower():
        raise RuntimeError("temperature screen does not mark ARI/AMI supporting-only")

    expected_provenance = {
        "script": str(TEMPERATURE_RUNNER),
        "script_sha256": sha256_file(TEMPERATURE_RUNNER),
        "trainer": str(TRAINER),
        "trainer_sha256": sha256_file(TRAINER),
        "protocol_source_sha256": sha256_file(Path(protocol.__file__).resolve()),
        "scorer_bundle_sha256": context["scorer_bundle_sha256"],
        "eval_manifest_sha256": context["eval_manifest_sha256"],
    }
    for key, expected in expected_provenance.items():
        if provenance.get(key) != expected:
            raise RuntimeError(f"temperature result provenance mismatch: {key}")

    source_eval_manifest = Path(str(provenance.get("eval_manifest", ""))).resolve()
    if (
        not source_eval_manifest.is_file()
        or provenance.get("eval_manifest_file_sha256")
        != sha256_file(source_eval_manifest)
    ):
        raise RuntimeError("temperature result eval manifest provenance is stale")

    key = f"{proposed:.2f}"
    sidecar_record = provenance.get("embedding_sidecars", {}).get(key)
    if not isinstance(sidecar_record, dict):
        raise RuntimeError("accepted temperature sidecar record is missing")
    sidecar = Path(str(sidecar_record.get("path", ""))).resolve()
    if (
        not sidecar.is_file()
        or sidecar_record.get("sha256") != sha256_file(sidecar)
    ):
        raise RuntimeError("accepted temperature sidecar record is stale")
    sidecar_payload = _read_json(sidecar, "accepted temperature sidecar")
    try:
        contract = sidecar_payload["contract"]
        source_checkpoint = Path(contract["source_checkpoint"]).resolve()
        embedding = Path(sidecar_payload["embedding"]).resolve()
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"accepted temperature sidecar is incomplete: {error}"
        ) from error
    expected_contract = {
        "axis": "temperature",
        "temperature": key,
        "trainer_sha256": sha256_file(TRAINER),
        "protocol_source_sha256": sha256_file(Path(protocol.__file__).resolve()),
        "scorer_bundle_sha256": context["scorer_bundle_sha256"],
        "train_manifest": context["train_manifest"],
        "train_manifest_sha256": context["train_manifest_sha256"],
        "eval_dir": context["eval_dir"],
        "eval_manifest_sha256": context["eval_manifest_sha256"],
    }
    for field, expected in expected_contract.items():
        if contract.get(field) != expected:
            raise RuntimeError(
                f"accepted temperature sidecar mismatch: {field}"
            )
    if (
        not source_checkpoint.is_file()
        or contract.get("source_checkpoint_sha256")
        != sha256_file(source_checkpoint)
    ):
        raise RuntimeError("accepted temperature checkpoint is stale")
    if (
        not embedding.is_file()
        or sidecar_payload.get("embedding_sha256") != sha256_file(embedding)
    ):
        raise RuntimeError("accepted temperature embedding is stale")

    recipe = f"adapter_temp_{proposed:.2f}"
    accepted_rows = [
        row for row in rows if isinstance(row, dict) and row.get("recipe") == recipe
    ]
    if (
        len(accepted_rows) != 2
        or {row.get("clusterer") for row in accepted_rows} != PRIMARY_CLUSTERERS
        or any(not PRIMARY_FIELDS.issubset(row) for row in accepted_rows)
    ):
        raise RuntimeError("accepted temperature FINCH/Louvain P1-P4 rows are stale")

    evidence = {
        "temperature": proposed,
        "temperature_result": str(TEMPERATURE_RESULT),
        "temperature_result_sha256": sha256_file(TEMPERATURE_RESULT),
        "acceptance_record_sha256": sha256_json(accepted[0]),
        "temperature_sidecar": str(sidecar),
        "temperature_sidecar_sha256": sha256_file(sidecar),
    }
    return proposed, evidence


def embedding_contract(
    queue_size: int,
    temperature: float,
    temperature_evidence: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = checkpoint_path(queue_size, temperature)
    protocol_source = Path(protocol.__file__).resolve()
    eval_manifest = Path(context["eval_manifest"]).resolve()
    command = training_command(queue_size, temperature)
    return {
        "schema": PROVENANCE_SCHEMA,
        "axis": "queue_size",
        "queue_size": queue_size,
        "queue_enabled": queue_size > 0,
        "temperature": f"{temperature:.2f}",
        "accepted_temperature_evidence": temperature_evidence,
        "seed": SEED,
        "epoch": EPOCH,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "trainer": str(TRAINER),
        "trainer_sha256": sha256_file(TRAINER),
        "protocol_source": str(protocol_source),
        "protocol_source_sha256": sha256_file(protocol_source),
        "scorer_bundle_sha256": context["scorer_bundle_sha256"],
        "train_manifest": context["train_manifest"],
        "train_manifest_sha256": context["train_manifest_sha256"],
        "eval_dir": context["eval_dir"],
        "eval_manifest": str(eval_manifest),
        "eval_manifest_sha256": context["eval_manifest_sha256"],
        "eval_manifest_file_sha256": sha256_file(eval_manifest),
        "hdbscan": HDBSCAN_CONFIG,
        "command": command,
        "command_sha256": sha256_json(command),
    }


def embedding_is_reusable(
    queue_size: int,
    temperature: float,
    temperature_evidence: dict[str, Any],
    context: dict[str, Any],
) -> bool:
    embedding = embedding_path(queue_size, temperature)
    checkpoint = checkpoint_path(queue_size, temperature)
    sidecar = embedding_sidecar_path(queue_size, temperature)
    if not embedding.is_file() or not checkpoint.is_file() or not sidecar.is_file():
        return False
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        return (
            payload.get("contract")
            == embedding_contract(
                queue_size, temperature, temperature_evidence, context
            )
            and payload.get("embedding") == str(embedding)
            and payload.get("embedding_sha256") == sha256_file(embedding)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def write_embedding_sidecar(
    queue_size: int,
    temperature: float,
    temperature_evidence: dict[str, Any],
    context: dict[str, Any],
) -> None:
    embedding = embedding_path(queue_size, temperature)
    atomic_write_json(
        embedding_sidecar_path(queue_size, temperature),
        {
            "created_at": datetime.now().astimezone().isoformat(),
            "contract": embedding_contract(
                queue_size, temperature, temperature_evidence, context
            ),
            "embedding": str(embedding),
            "embedding_sha256": sha256_file(embedding),
        },
    )


def remove_stale_queue_artifacts(queue_size: int, temperature: float) -> None:
    tag = cell_tag(queue_size, temperature)
    for path in EMBEDDINGS.glob(f"{tag}_ep*.npy"):
        path.unlink(missing_ok=True)
    embedding_sidecar_path(queue_size, temperature).unlink(missing_ok=True)
    checkpoint_path(queue_size, temperature).unlink(missing_ok=True)


def spec(queue_size: int, temperature: float) -> dict[str, Any]:
    return {
        "recipe": f"adapter_queue_{queue_size}",
        "recipe_flags": (
            "no-TAPT FCMAE frozen backbone; residual adapter; pure SimCLR; "
            f"accepted temperature={temperature:.2f}; queue_size={queue_size}; "
            f"fixed seed{SEED} epoch{EPOCH}"
        ),
        "seed": SEED,
        "epoch": EPOCH,
        "embedding_space": "adapted_f",
        "path": embedding_path(queue_size, temperature),
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
    for queue_size in QUEUE_SIZES:
        recipe = f"adapter_queue_{queue_size}"
        candidate_rows = [row for row in rows if row["recipe"] == recipe]
        if {row["clusterer"] for row in candidate_rows} != set(frozen):
            raise RuntimeError(f"missing clusterer row for queue size {queue_size}")
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
                "delta": {
                    key: round(delta, 6) for key, delta in deltas.items()
                },
            }
        values.append(
            {
                "queue_size": queue_size,
                "accepted": accepted,
                "minimum_P3_P4_delta": round(minimum_primary_delta, 6),
                "clusterers": clusterers,
            }
        )
    accepted_values = sorted(
        (item for item in values if item["accepted"]),
        key=lambda item: (-item["minimum_P3_P4_delta"], item["queue_size"]),
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
        "proposed_queue_size": (
            accepted_values[0]["queue_size"] if accepted_values else None
        ),
    }


def write_outputs(
    rows: list[dict[str, Any]],
    gate: dict[str, Any],
    context: dict[str, Any],
    temperature: float,
    temperature_evidence: dict[str, Any],
    commands: dict[str, list[str]],
) -> None:
    PAPER_ROOT.mkdir(parents=True, exist_ok=True)
    protocol.write_csv(OUTPUT_CSV, rows)
    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "protocol_id": context["protocol_id"],
        "accepted_temperature": temperature,
        "hdbscan": HDBSCAN_CONFIG,
        "screen": gate,
        "rows": rows,
        "provenance": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "trainer": str(TRAINER),
            "trainer_sha256": sha256_file(TRAINER),
            "protocol_source_sha256": sha256_file(
                Path(protocol.__file__).resolve()
            ),
            "commands": commands,
            "temperature_evidence": temperature_evidence,
            "scorer_bundle_sha256": context["scorer_bundle_sha256"],
            "train_manifest": context["train_manifest"],
            "train_manifest_sha256": context["train_manifest_sha256"],
            "eval_manifest": context["eval_manifest"],
            "eval_manifest_sha256": context["eval_manifest_sha256"],
            "eval_manifest_file_sha256": sha256_file(
                Path(context["eval_manifest"])
            ),
            "embedding_sidecars": {
                str(queue_size): {
                    "path": str(
                        embedding_sidecar_path(queue_size, temperature)
                    ),
                    "sha256": sha256_file(
                        embedding_sidecar_path(queue_size, temperature)
                    ),
                }
                for queue_size in QUEUE_SIZES
            },
        },
        "outputs": {
            "json": str(OUTPUT_JSON),
            "csv": str(OUTPUT_CSV),
            "markdown": str(OUTPUT_MD),
        },
    }
    atomic_write_json(OUTPUT_JSON, payload)
    lines = [
        "# FCMAE Residual Adapter Queue-Size Screen (260725)",
        "",
        "- One-axis screen: queue size only",
        f"- Accepted temperature: {temperature:.2f}",
        f"- Fixed seed/epoch: {SEED}/{EPOCH}",
        (
            "- Fixed HDBSCAN: mcs=12, min_samples=15, leaf, "
            "epsilon=0.06"
        ),
        "- P1/P2/P3/P4 decide screening; ARI/AMI are supporting only.",
        "- This report proposes a candidate but never launches multi-seed.",
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
            f"- queue {item['queue_size']}: accepted={item['accepted']}, "
            f"minimum P3/P4 delta={item['minimum_P3_P4_delta']:.6f}"
        )
    lines.extend(
        [
            "",
            f"- proposed queue size: {gate['proposed_queue_size']}",
            "",
            "## Absolute Outputs",
            "",
            f"- JSON: `{OUTPUT_JSON}`",
            f"- CSV: `{OUTPUT_CSV}`",
            f"- run root: `{RUN_ROOT}`",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_training(
    queue_size: int,
    temperature: float,
    temperature_evidence: dict[str, Any],
    command: list[str],
    context: dict[str, Any],
) -> None:
    expected = embedding_path(queue_size, temperature)
    if embedding_is_reusable(
        queue_size, temperature, temperature_evidence, context
    ):
        print(f"[skip] {expected}", flush=True)
        return
    remove_stale_queue_artifacts(queue_size, temperature)
    EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    with RUN_LOG.open("a", encoding="utf-8") as log:
        log.write(
            f"\n[{datetime.now().astimezone().isoformat()}] "
            f"queue_size={queue_size} temperature={temperature:.2f}\n"
        )
        log.flush()
        process = subprocess.Popen(
            command,
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
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code != 0:
        raise RuntimeError(
            f"queue size {queue_size} training failed with exit code {code}"
        )
    if not expected.is_file():
        raise RuntimeError(f"missing expected embedding: {expected}")
    if not checkpoint_path(queue_size, temperature).is_file():
        raise RuntimeError(
            "missing source checkpoint for queue-size embedding: "
            f"{checkpoint_path(queue_size, temperature)}"
        )
    write_embedding_sidecar(
        queue_size, temperature, temperature_evidence, context
    )


def validate_result(
    context: dict[str, Any],
    temperature: float,
    temperature_evidence: dict[str, Any],
) -> tuple[bool, str]:
    if not OUTPUT_JSON.is_file():
        return False, "result JSON is missing"
    try:
        payload = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
        values = payload["screen"]["values"]
        rows = payload["rows"]
        provenance = payload["provenance"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return False, f"result JSON is invalid: {error}"
    if (
        len(values) != len(QUEUE_SIZES)
        or len(rows) != 2 * (len(QUEUE_SIZES) + 1)
    ):
        return False, "result row/value count is stale"
    if sorted(item.get("queue_size") for item in values) != list(QUEUE_SIZES):
        return False, "result queue-size set is stale"
    expected = {
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "trainer_sha256": sha256_file(TRAINER),
        "protocol_source_sha256": sha256_file(Path(protocol.__file__).resolve()),
        "temperature_evidence": temperature_evidence,
        "scorer_bundle_sha256": context["scorer_bundle_sha256"],
        "train_manifest": context["train_manifest"],
        "train_manifest_sha256": context["train_manifest_sha256"],
        "eval_manifest": context["eval_manifest"],
        "eval_manifest_sha256": context["eval_manifest_sha256"],
        "eval_manifest_file_sha256": sha256_file(
            Path(context["eval_manifest"])
        ),
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            return False, f"result provenance mismatch: {key}"
    if payload.get("hdbscan") != HDBSCAN_CONFIG:
        return False, "result HDBSCAN contract is stale"
    recorded_sidecars = provenance.get("embedding_sidecars", {})
    for queue_size in QUEUE_SIZES:
        sidecar = embedding_sidecar_path(queue_size, temperature)
        if not embedding_is_reusable(
            queue_size, temperature, temperature_evidence, context
        ):
            return False, f"embedding provenance mismatch: queue {queue_size}"
        recorded = recorded_sidecars.get(str(queue_size), {})
        if (
            recorded.get("path") != str(sidecar)
            or recorded.get("sha256") != sha256_file(sidecar)
        ):
            return False, f"result sidecar mismatch: queue {queue_size}"
        expected_hash = sha256_file(embedding_path(queue_size, temperature))
        candidate_rows = [
            row
            for row in rows
            if row.get("recipe") == f"adapter_queue_{queue_size}"
        ]
        if (
            len(candidate_rows) != 2
            or {row.get("clusterer") for row in candidate_rows}
            != PRIMARY_CLUSTERERS
            or any(
                row.get("embedding_sha256") != expected_hash
                or not PRIMARY_FIELDS.issubset(row)
                for row in candidate_rows
            )
        ):
            return False, f"result embedding rows are stale: queue {queue_size}"
    return True, "current"


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--score-only", action="store_true")
    mode.add_argument("--validate-result", action="store_true")
    args = parser.parse_args()

    protocol.PROTOCOL_JSON = SCREEN_PROTOCOL
    protocol.EVAL_MANIFEST = SCREEN_MANIFEST
    context = protocol.protocol_context()
    temperature, temperature_evidence = load_accepted_temperature(context)
    commands = {
        str(queue_size): training_command(queue_size, temperature)
        for queue_size in QUEUE_SIZES
    }
    config = {
        "created_at": datetime.now().astimezone().isoformat(),
        "axis": "queue_size",
        "seed": SEED,
        "fixed_epoch": EPOCH,
        "queue_sizes": list(QUEUE_SIZES),
        "accepted_temperature": temperature,
        "accepted_temperature_evidence": temperature_evidence,
        "hdbscan": HDBSCAN_CONFIG,
        "commands": commands,
        "automatic_adoption": False,
    }
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(CONFIG_JSON, config)
    if args.dry_run:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return 0
    if args.validate_result:
        valid, reason = validate_result(
            context, temperature, temperature_evidence
        )
        print(reason)
        return 0 if valid else 1
    if not args.score_only:
        for queue_size in QUEUE_SIZES:
            run_training(
                queue_size,
                temperature,
                temperature_evidence,
                commands[str(queue_size)],
                context,
            )
    for queue_size in QUEUE_SIZES:
        if not embedding_is_reusable(
            queue_size, temperature, temperature_evidence, context
        ):
            raise RuntimeError(
                f"missing or stale queue {queue_size} embedding provenance"
            )

    rows = protocol.score_specs(
        [
            frozen_spec(),
            *(spec(queue_size, temperature) for queue_size in QUEUE_SIZES),
        ],
        context,
    )
    gate = screen(rows)
    write_outputs(
        rows,
        gate,
        context,
        temperature,
        temperature_evidence,
        commands,
    )
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    print(f"[out] {OUTPUT_JSON}")
    print(f"[out] {OUTPUT_CSV}")
    print(f"[out] {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
