#!/usr/bin/env python3
"""Screen one queue-size axis for the hard-42 FCMAE residual adapter.

This runner is a bounded follow-up for the case where the temperature screen
has no winner. It changes only queue enablement/size. The queue-off control
reuses the fixed seed-1, epoch-4, temperature-0.05 checkpoint; queue-enabled
cells train fresh from the same no-TAPT FCMAE initialization and fixed recipe.

Labels are used only by the fixed development evaluator. P1/P2/P3/P4 from
FINCH-p2 and Louvain-res6 decide the screen. ARI/AMI remain supporting records.
The runner proposes a value but never launches a multi-seed follow-up.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
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
RUN_ROOT = (REPO / "runs/fcmae_adapter_queue_screen_260725").resolve()
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

SEED = 1
EPOCH = 4
TEMPERATURE = 0.05
ADAPTER_ALPHA = 1.00
QUEUE_SIZES = (0, 1024, 2048, 4096)
SOURCE_CHECKPOINT = (
    REPO
    / "runs/fcmae_adapter_ep4_three_seed_260721/embeddings/fcmae_ad1_s1_ckpt.pt"
).resolve()
EXPECTED_GSTEP = 3296
PROVENANCE_SCHEMA = 1

# Kept fixed for report parity. The acceptance gate itself uses the two
# clusterers in fcmae_fixed_protocol, not HDBSCAN.
MAY_HDBSCAN = {
    "min_cluster_size": 12,
    "min_samples": 15,
    "cluster_selection_method": "leaf",
    "cluster_selection_epsilon": 0.06,
}

PRIMARY_FIELDS = (
    "P1_capture_count",
    "P1_target_class_count",
    "P1_capture",
    "P2_noise_pct",
    "P3_completeness",
    "P4_homogeneity",
)
SUPPORTING_FIELDS = ("ARI_supporting", "AMI_supporting")


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


def queue_key(queue_size: int) -> str:
    if queue_size not in QUEUE_SIZES:
        raise ValueError(f"unsupported queue size: {queue_size}")
    return "off" if queue_size == 0 else str(queue_size)


def queue_tag(queue_size: int) -> str:
    return f"q{queue_key(queue_size)}"


def recipe_name(queue_size: int) -> str:
    return f"adapter_queue_{queue_key(queue_size)}"


def embedding_path(queue_size: int) -> Path:
    tag = f"fcmae_ad1_{queue_tag(queue_size)}_s{SEED}"
    return EMBEDDINGS / f"{tag}_ep{EPOCH}.npy"


def checkpoint_path(queue_size: int) -> Path:
    tag = f"fcmae_ad1_{queue_tag(queue_size)}_s{SEED}"
    return EMBEDDINGS / f"{tag}_ckpt.pt"


def embedding_sidecar_path(queue_size: int) -> Path:
    path = embedding_path(queue_size)
    return path.with_name(f"{path.name}.provenance.json")


def fixed_recipe_contract() -> dict[str, Any]:
    return {
        "backbone": "convnextv2_base.fcmae_ft_in22k_in1k_384",
        "backbone_pretraining": "no-TAPT FCMAE",
        "freeze_backbone": True,
        "head": "adapter",
        "adapter_inference_scale": f"{ADAPTER_ALPHA:.2f}",
        "projection_dim": 128,
        "seed": SEED,
        "epoch": EPOCH,
        "batch": 8,
        "temperature": f"{TEMPERATURE:.2f}",
        "train_dir": str(protocol.TRAIN_DIR.resolve()),
        "eval_dir": str(protocol.EVAL_DIR.resolve()),
        "augmentation": {
            "wafer_rot_deg": 0,
            "wafer_translate": 0,
            "wafer_scale_min": 1.0,
            "wafer_crop_min": 1.0,
        },
        "loss": {
            "method": "simclr",
            "local": 0.0,
            "neco": 0.0,
            "negative_ignore": 1.01,
            "nv_filter": 0.0,
            "label_smoothing": 0.0,
        },
        "may_hdbscan": MAY_HDBSCAN,
        "screen_clusterers": ["FINCH-p2", "Louvain-res6"],
    }


def training_command(queue_size: int) -> list[str]:
    queue_key(queue_size)
    tag = f"fcmae_ad1_{queue_tag(queue_size)}_s{SEED}"
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
        str(TEMPERATURE),
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
        "--queue-size",
        str(1024 if queue_size == 0 else queue_size),
    ]
    if queue_size > 0:
        command.extend(["--use-queue", "--fresh"])
    return command


def manifest_file_hash(context: dict[str, Any], key: str) -> str:
    path = Path(context[key]).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing provenance manifest: {path}")
    return sha256_file(path)


def embedding_contract(
    queue_size: int, context: dict[str, Any]
) -> dict[str, Any]:
    checkpoint = checkpoint_path(queue_size)
    protocol_source = Path(protocol.__file__).resolve()
    baseline_origin = (
        {
            "path": str(SOURCE_CHECKPOINT),
            "sha256": sha256_file(SOURCE_CHECKPOINT),
        }
        if queue_size == 0
        else None
    )
    command = training_command(queue_size)
    fixed = fixed_recipe_contract()
    return {
        "schema": PROVENANCE_SCHEMA,
        "axis": "queue_size",
        "queue_enabled": queue_size > 0,
        "queue_size": queue_size,
        "seed": SEED,
        "epoch": EPOCH,
        "temperature": f"{TEMPERATURE:.2f}",
        "adapter_inference_scale": f"{ADAPTER_ALPHA:.2f}",
        "fixed_recipe": fixed,
        "fixed_recipe_sha256": sha256_json(fixed),
        "may_hdbscan": MAY_HDBSCAN,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "baseline_origin_checkpoint": baseline_origin,
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "trainer": str(TRAINER),
        "trainer_sha256": sha256_file(TRAINER),
        "protocol_source": str(protocol_source),
        "protocol_source_sha256": sha256_file(protocol_source),
        "scorer_bundle_sha256": context["scorer_bundle_sha256"],
        "train_manifest": context["train_manifest"],
        "train_manifest_sha256": context["train_manifest_sha256"],
        "train_manifest_file_sha256": manifest_file_hash(
            context, "train_manifest"
        ),
        "eval_dir": context["eval_dir"],
        "eval_manifest": context["eval_manifest"],
        "eval_manifest_sha256": context["eval_manifest_sha256"],
        "eval_manifest_file_sha256": manifest_file_hash(
            context, "eval_manifest"
        ),
        "command": command,
        "command_sha256": sha256_json(command),
    }


def embedding_is_reusable(
    queue_size: int, context: dict[str, Any]
) -> bool:
    embedding = embedding_path(queue_size)
    checkpoint = checkpoint_path(queue_size)
    sidecar = embedding_sidecar_path(queue_size)
    if not embedding.is_file() or not checkpoint.is_file() or not sidecar.is_file():
        return False
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        return (
            payload.get("contract") == embedding_contract(queue_size, context)
            and payload.get("embedding") == str(embedding)
            and payload.get("embedding_sha256") == sha256_file(embedding)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def write_embedding_sidecar(
    queue_size: int, context: dict[str, Any]
) -> None:
    embedding = embedding_path(queue_size)
    atomic_write_json(
        embedding_sidecar_path(queue_size),
        {
            "created_at": datetime.now().astimezone().isoformat(),
            "contract": embedding_contract(queue_size, context),
            "embedding": str(embedding),
            "embedding_sha256": sha256_file(embedding),
        },
    )


def remove_stale_queue_artifacts(
    queue_size: int, *, remove_checkpoint: bool
) -> None:
    tag = f"fcmae_ad1_{queue_tag(queue_size)}_s{SEED}"
    for path in EMBEDDINGS.glob(f"{tag}_ep*.npy"):
        path.unlink(missing_ok=True)
    embedding_sidecar_path(queue_size).unlink(missing_ok=True)
    if remove_checkpoint:
        checkpoint_path(queue_size).unlink(missing_ok=True)


def stage_queue_off_checkpoint() -> None:
    if not SOURCE_CHECKPOINT.is_file():
        raise FileNotFoundError(f"missing source checkpoint: {SOURCE_CHECKPOINT}")
    target = checkpoint_path(0)
    source_hash = sha256_file(SOURCE_CHECKPOINT)
    if target.is_file() and sha256_file(target) == source_hash:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    shutil.copyfile(SOURCE_CHECKPOINT, temporary)
    os.replace(temporary, target)
    if sha256_file(target) != source_hash:
        raise RuntimeError("staged queue-off checkpoint hash mismatch")


def spec(queue_size: int) -> dict[str, Any]:
    state = "off" if queue_size == 0 else f"on,size={queue_size}"
    return {
        "recipe": recipe_name(queue_size),
        "recipe_flags": (
            "no-TAPT FCMAE frozen backbone; residual adapter alpha=1.00; "
            f"temperature=0.05; queue={state}; fixed seed1 epoch4"
        ),
        "seed": SEED,
        "epoch": EPOCH,
        "embedding_space": "adapted_f",
        "path": embedding_path(queue_size),
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
        candidate_rows = [
            row for row in rows if row["recipe"] == recipe_name(queue_size)
        ]
        if {row["clusterer"] for row in candidate_rows} != set(frozen):
            raise RuntimeError(
                f"missing clusterer row for queue {queue_key(queue_size)}"
            )
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
                    key: round(value, 6) for key, value in deltas.items()
                },
                "supporting": {
                    "ARI": row["ARI_supporting"],
                    "AMI": row["AMI_supporting"],
                },
            }
        values.append(
            {
                "queue": queue_key(queue_size),
                "queue_size": queue_size,
                "accepted": accepted,
                "minimum_P3_P4_delta": round(minimum_primary_delta, 6),
                "clusterers": clusterers,
            }
        )
    accepted_values = sorted(
        (item for item in values if item["accepted"]),
        key=lambda item: (
            -item["minimum_P3_P4_delta"],
            item["queue_size"],
        ),
    )
    proposed = accepted_values[0] if accepted_values else None
    return {
        "contract": {
            "P1": "preserve frozen capture in FINCH-p2 and Louvain-res6",
            "P2": "do not increase frozen target noise in either clusterer",
            "P3_P4": "both must be non-worse in both clusterers",
            "ARI_AMI": "supporting only; excluded from screening",
            "selection": "proposal only; no automatic adoption or multi-seed launch",
        },
        "values": values,
        "proposed_queue": proposed["queue"] if proposed else None,
        "proposed_queue_size": proposed["queue_size"] if proposed else None,
        "automatic_followup_launched": False,
    }


def write_outputs(
    rows: list[dict[str, Any]],
    gate: dict[str, Any],
    context: dict[str, Any],
    commands: dict[str, list[str]],
) -> None:
    PAPER_ROOT.mkdir(parents=True, exist_ok=True)
    protocol.write_csv(OUTPUT_CSV, rows)
    fixed = fixed_recipe_contract()
    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "protocol_id": context["protocol_id"],
        "contract": (
            "One queue axis only; both clusterers preserve P1/P2 and "
            "do not worsen P3/P4"
        ),
        "screen": gate,
        "rows": rows,
        "provenance": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "trainer": str(TRAINER),
            "trainer_sha256": sha256_file(TRAINER),
            "protocol_source": str(Path(protocol.__file__).resolve()),
            "protocol_source_sha256": sha256_file(
                Path(protocol.__file__).resolve()
            ),
            "baseline_source_checkpoint": str(SOURCE_CHECKPOINT),
            "baseline_source_checkpoint_sha256": sha256_file(
                SOURCE_CHECKPOINT
            ),
            "fixed_recipe": fixed,
            "fixed_recipe_sha256": sha256_json(fixed),
            "may_hdbscan": MAY_HDBSCAN,
            "commands": commands,
            "commands_sha256": sha256_json(commands),
            "protocol_id": context["protocol_id"],
            "scorer_bundle_sha256": context["scorer_bundle_sha256"],
            "train_manifest": context["train_manifest"],
            "train_manifest_sha256": context["train_manifest_sha256"],
            "train_manifest_file_sha256": manifest_file_hash(
                context, "train_manifest"
            ),
            "eval_manifest": context["eval_manifest"],
            "eval_manifest_sha256": context["eval_manifest_sha256"],
            "eval_manifest_file_sha256": manifest_file_hash(
                context, "eval_manifest"
            ),
            "embedding_sidecars": {
                queue_key(queue_size): {
                    "path": str(embedding_sidecar_path(queue_size)),
                    "sha256": sha256_file(embedding_sidecar_path(queue_size)),
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
        "- One-axis screen: queue off / 1024 / 2048 / 4096.",
        "- Fixed: no-TAPT FCMAE, residual adapter alpha 1.00, seed 1, "
        "epoch 4, temperature 0.05, data and position-preserving augmentation.",
        "- P1/P2/P3/P4 decide screening in FINCH-p2 and Louvain-res6.",
        "- ARI/AMI are supporting only.",
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
            f"- queue {item['queue']}: accepted={item['accepted']}, "
            f"minimum P3/P4 delta={item['minimum_P3_P4_delta']:.6f}"
        )
    lines.extend(
        [
            "",
            f"- proposed queue: {gate['proposed_queue']}",
            "",
            "## Absolute Outputs",
            "",
            f"- JSON: `{OUTPUT_JSON}`",
            f"- CSV: `{OUTPUT_CSV}`",
            f"- run root: `{RUN_ROOT}`",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_result(context: dict[str, Any]) -> tuple[bool, str]:
    if not OUTPUT_JSON.is_file():
        return False, "result JSON is missing"
    try:
        payload = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
        values = payload["screen"]["values"]
        rows = payload["rows"]
        provenance = payload["provenance"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return False, f"result JSON is invalid: {error}"
    if len(values) != len(QUEUE_SIZES) or len(rows) != 2 * (
        len(QUEUE_SIZES) + 1
    ):
        return False, "result row/value count is stale"
    expected_sizes = list(QUEUE_SIZES)
    if sorted(item.get("queue_size") for item in values) != expected_sizes:
        return False, "result queue-size set is stale"
    commands = {
        queue_key(queue_size): training_command(queue_size)
        for queue_size in QUEUE_SIZES
    }
    fixed = fixed_recipe_contract()
    expected = {
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "trainer_sha256": sha256_file(TRAINER),
        "protocol_source_sha256": sha256_file(
            Path(protocol.__file__).resolve()
        ),
        "baseline_source_checkpoint_sha256": sha256_file(
            SOURCE_CHECKPOINT
        ),
        "fixed_recipe_sha256": sha256_json(fixed),
        "commands_sha256": sha256_json(commands),
        "protocol_id": context["protocol_id"],
        "scorer_bundle_sha256": context["scorer_bundle_sha256"],
        "train_manifest": context["train_manifest"],
        "train_manifest_sha256": context["train_manifest_sha256"],
        "train_manifest_file_sha256": manifest_file_hash(
            context, "train_manifest"
        ),
        "eval_manifest": context["eval_manifest"],
        "eval_manifest_sha256": context["eval_manifest_sha256"],
        "eval_manifest_file_sha256": manifest_file_hash(
            context, "eval_manifest"
        ),
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            return False, f"result provenance mismatch: {key}"
    if provenance.get("fixed_recipe") != fixed:
        return False, "result fixed recipe is stale"
    if provenance.get("may_hdbscan") != MAY_HDBSCAN:
        return False, "result May HDBSCAN contract is stale"
    if provenance.get("commands") != commands:
        return False, "result commands are stale"

    frozen_rows = [row for row in rows if row.get("recipe") == "frozen"]
    frozen_clusterers = {row.get("clusterer") for row in frozen_rows}
    if frozen_clusterers != {"FINCH-p2", "Louvain-res6"}:
        return False, "frozen clusterer rows are stale"
    recorded_sidecars = provenance.get("embedding_sidecars", {})
    for queue_size in QUEUE_SIZES:
        key = queue_key(queue_size)
        sidecar = embedding_sidecar_path(queue_size)
        if not embedding_is_reusable(queue_size, context):
            return False, f"embedding provenance mismatch: queue {key}"
        recorded = recorded_sidecars.get(key, {})
        if (
            recorded.get("path") != str(sidecar)
            or recorded.get("sha256") != sha256_file(sidecar)
        ):
            return False, f"result sidecar mismatch: queue {key}"
        expected_hash = sha256_file(embedding_path(queue_size))
        candidate_rows = [
            row
            for row in rows
            if row.get("recipe") == recipe_name(queue_size)
        ]
        if (
            {row.get("clusterer") for row in candidate_rows}
            != frozen_clusterers
        ):
            return False, f"result clusterer rows are stale: queue {key}"
        for row in candidate_rows:
            if row.get("embedding_sha256") != expected_hash:
                return False, f"result embedding rows are stale: queue {key}"
            if any(field not in row for field in PRIMARY_FIELDS):
                return False, f"result primary metrics are incomplete: queue {key}"
            if any(field not in row for field in SUPPORTING_FIELDS):
                return False, f"result supporting metrics are incomplete: queue {key}"
    return True, "current"


def run_training(
    queue_size: int,
    command: list[str],
    context: dict[str, Any],
) -> None:
    expected = embedding_path(queue_size)
    if embedding_is_reusable(queue_size, context):
        print(f"[skip] {expected}", flush=True)
        return
    if queue_size == 0:
        stage_queue_off_checkpoint()
        remove_stale_queue_artifacts(queue_size, remove_checkpoint=False)
        checkpoint_before = sha256_file(checkpoint_path(queue_size))
    else:
        remove_stale_queue_artifacts(queue_size, remove_checkpoint=True)
        checkpoint_before = None
    EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    captured: list[str] = []
    with RUN_LOG.open("a", encoding="utf-8") as log:
        log.write(
            f"\n[{datetime.now().astimezone().isoformat()}] "
            f"queue={queue_key(queue_size)}\n"
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
            captured.append(line)
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code != 0:
        raise RuntimeError(
            f"queue {queue_key(queue_size)} training failed with exit code {code}"
        )
    if not expected.is_file():
        raise RuntimeError(f"missing expected embedding: {expected}")
    checkpoint = checkpoint_path(queue_size)
    if not checkpoint.is_file():
        raise RuntimeError(f"missing source checkpoint for embedding: {checkpoint}")
    if queue_size == 0:
        output = "".join(captured)
        if not re.search(
            rf"gstep\s+{EXPECTED_GSTEP}/{EXPECTED_GSTEP}", output
        ):
            raise RuntimeError("queue-off control did not resume at fixed epoch 4")
        if re.search(r"(?m)^\[simclr ep\d+\]\s+loss=", output):
            raise RuntimeError("queue-off control unexpectedly performed training")
        if sha256_file(checkpoint) != checkpoint_before:
            raise RuntimeError("queue-off extraction changed its checkpoint")
    write_embedding_sidecar(queue_size, context)


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--score-only", action="store_true")
    mode.add_argument("--validate-result", action="store_true")
    args = parser.parse_args()

    commands = {
        queue_key(queue_size): training_command(queue_size)
        for queue_size in QUEUE_SIZES
    }
    config = {
        "created_at": datetime.now().astimezone().isoformat(),
        "axis": "queue_size",
        "queue_sizes": list(QUEUE_SIZES),
        "fixed_recipe": fixed_recipe_contract(),
        "baseline_source_checkpoint": str(SOURCE_CHECKPOINT),
        "commands": commands,
        "automatic_adoption": False,
        "automatic_followup": False,
    }
    if args.dry_run:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return 0
    if not SOURCE_CHECKPOINT.is_file():
        raise FileNotFoundError(f"missing source checkpoint: {SOURCE_CHECKPOINT}")
    protocol.PROTOCOL_JSON = SCREEN_PROTOCOL
    protocol.EVAL_MANIFEST = SCREEN_MANIFEST
    context = protocol.protocol_context()
    if args.validate_result:
        valid, reason = validate_result(context)
        print(reason)
        return 0 if valid else 1

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(CONFIG_JSON, config)
    if not args.score_only:
        for queue_size in QUEUE_SIZES:
            run_training(
                queue_size,
                commands[queue_key(queue_size)],
                context,
            )
    for queue_size in QUEUE_SIZES:
        if not embedding_is_reusable(queue_size, context):
            raise RuntimeError(
                "missing or stale queue "
                f"{queue_key(queue_size)} embedding provenance"
            )
    rows = protocol.score_specs(
        [frozen_spec(), *(spec(size) for size in QUEUE_SIZES)],
        context,
    )
    gate = screen(rows)
    write_outputs(rows, gate, context, commands)
    valid, reason = validate_result(context)
    if not valid:
        raise RuntimeError(f"new queue-screen result failed validation: {reason}")
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    print(f"[out] {OUTPUT_JSON}")
    print(f"[out] {OUTPUT_CSV}")
    print(f"[out] {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
