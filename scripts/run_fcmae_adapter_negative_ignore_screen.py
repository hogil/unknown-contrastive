#!/usr/bin/env python3
"""Screen the negative-ignore threshold for the fixed hard-42 adapter protocol.

This runner is intentionally standalone. It does not inspect, stop, or launch
the temperature or queue screens, and it never launches a follow-up run.

All trained cells use the same no-TAPT FCMAE frozen-backbone residual adapter,
seed, epoch, data, augmentation, temperature, and queue. The only experimental
variable is the explicit ``--ignore`` value:

* off -> 1.01 (the trainer applies the filter only below 1.0)
* 0.80
* 0.75
* 0.70

The historical B4 threshold 0.72 is trained and scored as a control reference,
not as a candidate. FINCH/Louvain P1/P2/P3/P4 decide the screen. ARI/AMI remain
supporting diagnostics.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import fcmae_fixed_protocol as protocol  # noqa: E402


RUNNER = Path(__file__).resolve()
TRAINER = (REPO / "_ssl_methods.py").resolve()
B4_REFERENCE_SOURCE = (
    REPO / "scripts/run_may37_original_ablation.py"
).resolve()
RUN_ROOT = (
    REPO / "runs/fcmae_adapter_negative_ignore_screen_260725"
).resolve()
EMBEDDINGS = (RUN_ROOT / "embeddings").resolve()
RUN_LOG = (RUN_ROOT / "train.log").resolve()
CONFIG_JSON = (RUN_ROOT / "config.json").resolve()
PAPER_ROOT = (REPO / "docs/paper").resolve()
OUTPUT_JSON = (
    PAPER_ROOT / "FCMAE_ADAPTER_NEGATIVE_IGNORE_SCREEN_260725.json"
).resolve()
OUTPUT_CSV = (
    PAPER_ROOT / "FCMAE_ADAPTER_NEGATIVE_IGNORE_SCREEN_260725.csv"
).resolve()
OUTPUT_MD = (
    PAPER_ROOT / "FCMAE_ADAPTER_NEGATIVE_IGNORE_SCREEN_260725.md"
).resolve()
SCREEN_PROTOCOL = (
    PAPER_ROOT / "FCMAE_ADAPTER_NEGATIVE_IGNORE_SCREEN_260725_protocol.json"
).resolve()
SCREEN_MANIFEST = (
    PAPER_ROOT
    / "FCMAE_ADAPTER_NEGATIVE_IGNORE_SCREEN_260725_eval_manifest.json"
).resolve()

SEED = 1
EPOCH = 4
TEMPERATURE = 0.05
QUEUE_SIZE = 4096
OFF_TRAINER_VALUE = 1.01
B4_CONTROL_VALUE = 0.72
PROVENANCE_SCHEMA = 1

SCREEN_CONDITIONS: tuple[dict[str, Any], ...] = (
    {"key": "off", "value": OFF_TRAINER_VALUE, "label": "off"},
    {"key": "neg080", "value": 0.80, "label": "0.80"},
    {"key": "neg075", "value": 0.75, "label": "0.75"},
    {"key": "neg070", "value": 0.70, "label": "0.70"},
)
CONTROL_CONDITION: dict[str, Any] = {
    "key": "b4_control_072",
    "value": B4_CONTROL_VALUE,
    "label": "0.72",
}
ALL_CONDITIONS: tuple[dict[str, Any], ...] = (
    *SCREEN_CONDITIONS,
    CONTROL_CONDITION,
)


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


def condition(key: str) -> dict[str, Any]:
    matches = [item for item in ALL_CONDITIONS if item["key"] == key]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate negative-ignore condition: {key}")
    return matches[0]


def tag(key: str) -> str:
    return f"fcmae_ad1_{key}_s{SEED}"


def embedding_path(key: str) -> Path:
    return EMBEDDINGS / f"{tag(key)}_ep{EPOCH}.npy"


def checkpoint_path(key: str) -> Path:
    return EMBEDDINGS / f"{tag(key)}_ckpt.pt"


def embedding_sidecar_path(key: str) -> Path:
    path = embedding_path(key)
    return path.with_name(f"{path.name}.provenance.json")


def trainer_negative_ignore_contract(path: Path = TRAINER) -> dict[str, Any]:
    """Fail closed if the trainer's CLI or NEG application semantics changed."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    defaults: list[float] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not (
            isinstance(function, ast.Attribute)
            and function.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--ignore"
        ):
            continue
        default_nodes = [
            keyword.value for keyword in node.keywords if keyword.arg == "default"
        ]
        if len(default_nodes) != 1:
            raise RuntimeError("trainer --ignore must have exactly one default")
        defaults.append(float(ast.literal_eval(default_nodes[0])))
    if defaults != [OFF_TRAINER_VALUE]:
        raise RuntimeError(
            "trainer --ignore default changed; refusing a potentially aliased sweep"
        )

    loss_nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "loss_fn"
    ]
    if len(loss_nodes) != 1:
        raise RuntimeError("trainer loss_fn contract is missing or ambiguous")
    loss_source = ast.unparse(loss_nodes[0])
    required_fragments = (
        "if args.ignore < 1.0:",
        "sim * t > args.ignore",
        "qneg * t > args.ignore",
    )
    missing = [
        fragment for fragment in required_fragments if fragment not in loss_source
    ]
    if missing:
        raise RuntimeError(
            "trainer NEG implementation changed; missing " + ", ".join(missing)
        )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "cli_flag": "--ignore",
        "explicit_off_value": OFF_TRAINER_VALUE,
        "active_when": "value < 1.0",
        "in_batch_mask": "sim * temp > args.ignore",
        "queue_mask": "qneg * temp > args.ignore",
    }


def b4_control_reference_contract(
    path: Path = B4_REFERENCE_SOURCE,
) -> dict[str, Any]:
    """Bind the 0.72 control to the checked-in historical B4 definition."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    cells_values: list[dict[str, Any]] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "CELLS"
            for target in node.targets
        ):
            continue
        value = ast.literal_eval(node.value)
        if isinstance(value, dict):
            cells_values.append(value)
    if len(cells_values) != 1 or "B4" not in cells_values[0]:
        raise RuntimeError("historical B4 reference is missing or ambiguous")
    b4 = cells_values[0]["B4"]
    if b4.get("queue") is not True or float(b4.get("ignore")) != B4_CONTROL_VALUE:
        raise RuntimeError("historical B4 is no longer queue + NEG 0.72")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "cell": "B4",
        "queue_enabled": True,
        "negative_ignore": B4_CONTROL_VALUE,
    }


def training_command(key: str) -> list[str]:
    item = condition(key)
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
        f"{TEMPERATURE:.2f}",
        "--use-queue",
        "--queue-size",
        str(QUEUE_SIZE),
        "--ignore",
        f"{float(item['value']):.2f}",
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
        tag(key),
        "--fresh",
    ]


def _single_option(command: list[str], option: str) -> str:
    positions = [index for index, value in enumerate(command) if value == option]
    if len(positions) != 1:
        raise RuntimeError(f"{option} must occur exactly once")
    index = positions[0]
    if index + 1 >= len(command) or command[index + 1].startswith("--"):
        raise RuntimeError(f"{option} must have one explicit value")
    return command[index + 1]


def validate_training_command(key: str, command: list[str]) -> None:
    """Validate the exact argv passed to Popen immediately before launch."""
    item = condition(key)
    if command[:3] != [sys.executable, "-u", str(TRAINER)]:
        raise RuntimeError("trainer executable/path changed")
    for flag in ("--freeze-backbone", "--use-queue", "--fresh"):
        if command.count(flag) != 1:
            raise RuntimeError(f"{flag} must occur exactly once")
    for forbidden in ("--local", "--neco", "--nv-filter", "--natural-aug"):
        if forbidden in command:
            raise RuntimeError(f"one-axis NEG screen forbids {forbidden}")

    expected_options = {
        "--method": "simclr",
        "--timm": "convnextv2_base.fcmae_ft_in22k_in1k_384",
        "--head": "adapter",
        "--pdim": "128",
        "--seed": str(SEED),
        "--epochs": str(EPOCH),
        "--batch": "8",
        "--temp": f"{TEMPERATURE:.2f}",
        "--queue-size": str(QUEUE_SIZE),
        "--train-dir": str(protocol.TRAIN_DIR.resolve()),
        "--eval-dir": str(protocol.EVAL_DIR.resolve()),
        "--wafer-rot-deg": "0",
        "--wafer-translate": "0",
        "--wafer-scale-min": "1.0",
        "--wafer-crop-min": "1.0",
        "--out-dir": str(EMBEDDINGS),
        "--tag": tag(key),
    }
    for option, expected in expected_options.items():
        actual = _single_option(command, option)
        if actual != expected:
            raise RuntimeError(f"{option} mismatch: expected {expected}, got {actual}")

    actual_ignore = float(_single_option(command, "--ignore"))
    expected_ignore = float(item["value"])
    if actual_ignore != expected_ignore:
        raise RuntimeError(
            f"{key} NEG override mismatch: expected {expected_ignore:.2f}, "
            f"got {actual_ignore:.2f}"
        )
    if key == "off" and actual_ignore < 1.0:
        raise RuntimeError("off must explicitly disable the trainer NEG branch")
    if key != "off" and not actual_ignore < 1.0:
        raise RuntimeError(f"{key} must explicitly activate the trainer NEG branch")


def _normalize_axis_command(command: list[str]) -> list[str]:
    normalized = list(command)
    for option, replacement in (
        ("--ignore", "<NEG_AXIS>"),
        ("--tag", "<TAG>"),
    ):
        index = normalized.index(option)
        normalized[index + 1] = replacement
    return normalized


def validate_axis_commands(
    commands: dict[str, list[str]],
) -> dict[str, Any]:
    expected_keys = {item["key"] for item in ALL_CONDITIONS}
    if set(commands) != expected_keys:
        raise RuntimeError("negative-ignore condition set is incomplete or stale")
    normalized: dict[str, list[str]] = {}
    for key, command in commands.items():
        validate_training_command(key, command)
        normalized[key] = _normalize_axis_command(command)
    common = next(iter(normalized.values()))
    if any(command != common for command in normalized.values()):
        raise RuntimeError("a non-NEG training variable changed across conditions")
    return {
        "axis": "negative_ignore",
        "only_variable": "--ignore",
        "common_command_sha256": sha256_json(common),
        "command_sha256": {
            key: sha256_json(command) for key, command in commands.items()
        },
    }


def all_commands() -> dict[str, list[str]]:
    return {item["key"]: training_command(item["key"]) for item in ALL_CONDITIONS}


def embedding_contract(
    key: str,
    context: dict[str, Any],
    commands: dict[str, list[str]],
) -> dict[str, Any]:
    item = condition(key)
    checkpoint = checkpoint_path(key)
    eval_manifest = Path(context["eval_manifest"]).resolve()
    axis_contract = validate_axis_commands(commands)
    return {
        "schema": PROVENANCE_SCHEMA,
        "axis": "negative_ignore",
        "condition": key,
        "condition_label": item["label"],
        "trainer_ignore_value": f"{float(item['value']):.2f}",
        "role": "control" if key == CONTROL_CONDITION["key"] else "candidate",
        "seed": SEED,
        "epoch": EPOCH,
        "fixed_temperature": TEMPERATURE,
        "fixed_queue_size": QUEUE_SIZE,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "runner": str(RUNNER),
        "runner_sha256": sha256_file(RUNNER),
        "trainer_contract": trainer_negative_ignore_contract(),
        "historical_b4_control": b4_control_reference_contract(),
        "protocol_source": str(Path(protocol.__file__).resolve()),
        "protocol_source_sha256": sha256_file(Path(protocol.__file__).resolve()),
        "scorer_bundle_sha256": context["scorer_bundle_sha256"],
        "train_manifest": context["train_manifest"],
        "train_manifest_sha256": context["train_manifest_sha256"],
        "eval_dir": context["eval_dir"],
        "eval_manifest": str(eval_manifest),
        "eval_manifest_sha256": context["eval_manifest_sha256"],
        "eval_manifest_file_sha256": sha256_file(eval_manifest),
        "command": commands[key],
        "command_sha256": sha256_json(commands[key]),
        "common_command_sha256": axis_contract["common_command_sha256"],
    }


def embedding_is_reusable(
    key: str,
    context: dict[str, Any],
    commands: dict[str, list[str]],
) -> bool:
    embedding = embedding_path(key)
    checkpoint = checkpoint_path(key)
    sidecar = embedding_sidecar_path(key)
    if not embedding.is_file() or not checkpoint.is_file() or not sidecar.is_file():
        return False
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        return (
            payload.get("contract") == embedding_contract(key, context, commands)
            and payload.get("embedding") == str(embedding)
            and payload.get("embedding_sha256") == sha256_file(embedding)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError, RuntimeError):
        return False


def write_embedding_sidecar(
    key: str,
    context: dict[str, Any],
    commands: dict[str, list[str]],
) -> None:
    embedding = embedding_path(key)
    atomic_write_json(
        embedding_sidecar_path(key),
        {
            "created_at": datetime.now().astimezone().isoformat(),
            "contract": embedding_contract(key, context, commands),
            "embedding": str(embedding),
            "embedding_sha256": sha256_file(embedding),
        },
    )


def remove_stale_artifacts(key: str) -> None:
    for path in EMBEDDINGS.glob(f"{tag(key)}_ep*.npy"):
        path.unlink(missing_ok=True)
    checkpoint_path(key).unlink(missing_ok=True)
    embedding_sidecar_path(key).unlink(missing_ok=True)
    (RUN_ROOT / f"{tag(key)}.log").unlink(missing_ok=True)


def run_training(
    key: str,
    command: list[str],
    context: dict[str, Any],
    commands: dict[str, list[str]],
) -> None:
    if embedding_is_reusable(key, context, commands):
        print(f"[skip] {embedding_path(key)}", flush=True)
        return
    validate_training_command(key, command)
    remove_stale_artifacts(key)
    EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    captured: list[str] = []
    with RUN_LOG.open("a", encoding="utf-8") as log:
        log.write(
            f"\n[{datetime.now().astimezone().isoformat()}] "
            f"condition={key} argv_sha256={sha256_json(command)}\n"
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
        raise RuntimeError(f"{key} training failed with exit code {code}")
    output = "".join(captured)
    if "[resume]" in output:
        raise RuntimeError(f"{key} unexpectedly resumed instead of fresh training")
    if not re.search(r"(?m)^\[simclr ep\d+\]\s+loss=", output):
        raise RuntimeError(f"{key} did not emit fresh SimCLR training epochs")
    if not embedding_path(key).is_file():
        raise RuntimeError(f"missing expected embedding: {embedding_path(key)}")
    if not checkpoint_path(key).is_file():
        raise RuntimeError(f"missing source checkpoint: {checkpoint_path(key)}")
    write_embedding_sidecar(key, context, commands)


def recipe_name(key: str) -> str:
    if key == CONTROL_CONDITION["key"]:
        return "neg_ignore_control_0.72"
    if key == "off":
        return "neg_ignore_off"
    return f"neg_ignore_{condition(key)['label']}"


def spec(key: str) -> dict[str, Any]:
    item = condition(key)
    role = "historical B4 threshold control" if key == CONTROL_CONDITION["key"] else "screen candidate"
    return {
        "recipe": recipe_name(key),
        "recipe_flags": (
            "no-TAPT FCMAE frozen backbone; residual adapter; "
            f"Global InfoNCE + queue{QUEUE_SIZE}; temp={TEMPERATURE:.2f}; "
            f"explicit --ignore={float(item['value']):.2f}; {role}"
        ),
        "seed": SEED,
        "epoch": EPOCH,
        "embedding_space": "adapted_f",
        "path": embedding_path(key),
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


def _comparison(
    row: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    delta = {
        "P1_capture_count": int(row["P1_capture_count"])
        - int(baseline["P1_capture_count"]),
        "P2_noise_pct": float(row["P2_noise_pct"])
        - float(baseline["P2_noise_pct"]),
        "P3_completeness": float(row["P3_completeness"])
        - float(baseline["P3_completeness"]),
        "P4_homogeneity": float(row["P4_homogeneity"])
        - float(baseline["P4_homogeneity"]),
    }
    checks = {
        "P1_preserved": delta["P1_capture_count"] >= 0,
        "P2_not_worse": delta["P2_noise_pct"] <= 1e-9,
        "P3_not_worse": delta["P3_completeness"] >= -1e-9,
        "P4_not_worse": delta["P4_homogeneity"] >= -1e-9,
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "delta": {key: round(value, 6) for key, value in delta.items()},
    }


def screen(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frozen = protocol.frozen_by_clusterer(rows)
    control_rows = {
        row["clusterer"]: row
        for row in rows
        if row["recipe"] == recipe_name(CONTROL_CONDITION["key"])
    }
    if set(control_rows) != set(frozen):
        raise RuntimeError("both B4 control clusterer rows are required")

    values: list[dict[str, Any]] = []
    for item in SCREEN_CONDITIONS:
        key = item["key"]
        candidate_rows = {
            row["clusterer"]: row
            for row in rows
            if row["recipe"] == recipe_name(key)
        }
        if set(candidate_rows) != set(frozen):
            raise RuntimeError(f"missing clusterer row for {key}")
        clusterers: dict[str, Any] = {}
        accepted = True
        minimum_control_delta = float("inf")
        for clusterer in ("FINCH-p2", "Louvain-res6"):
            versus_frozen = _comparison(candidate_rows[clusterer], frozen[clusterer])
            versus_control = _comparison(
                candidate_rows[clusterer], control_rows[clusterer]
            )
            eligible = versus_frozen["accepted"] and versus_control["accepted"]
            accepted = accepted and eligible
            minimum_control_delta = min(
                minimum_control_delta,
                versus_control["delta"]["P3_completeness"],
                versus_control["delta"]["P4_homogeneity"],
            )
            clusterers[clusterer] = {
                "accepted": eligible,
                "versus_frozen": versus_frozen,
                "versus_b4_control_0.72": versus_control,
            }
        values.append(
            {
                "condition": key,
                "label": item["label"],
                "trainer_ignore_value": item["value"],
                "accepted": accepted,
                "minimum_P3_P4_delta_vs_control": round(
                    minimum_control_delta, 6
                ),
                "clusterers": clusterers,
            }
        )

    order = {item["key"]: index for index, item in enumerate(SCREEN_CONDITIONS)}
    accepted_values = sorted(
        (item for item in values if item["accepted"]),
        key=lambda item: (
            -item["minimum_P3_P4_delta_vs_control"],
            order[item["condition"]],
        ),
    )
    return {
        "contract": {
            "one_axis": "only explicit trainer --ignore changes",
            "fixed_recipe": (
                f"no-TAPT FCMAE frozen-backbone residual adapter, "
                f"temp={TEMPERATURE:.2f}, queue={QUEUE_SIZE}, seed={SEED}, ep={EPOCH}"
            ),
            "P1": "preserve frozen and B4-control capture in both clusterers",
            "P2": "do not increase frozen or B4-control target noise",
            "P3_P4": "both non-worse vs frozen and B4 control in both clusterers",
            "ARI_AMI": "supporting only; excluded from screening",
            "selection": "proposal only; no automatic adoption or follow-up",
        },
        "control": {
            "condition": CONTROL_CONDITION["key"],
            "trainer_ignore_value": B4_CONTROL_VALUE,
            "historical_reference": b4_control_reference_contract(),
        },
        "values": values,
        "proposed_condition": (
            accepted_values[0]["condition"] if accepted_values else None
        ),
        "proposed_negative_ignore": (
            accepted_values[0]["label"] if accepted_values else None
        ),
    }


def result_provenance(
    context: dict[str, Any],
    commands: dict[str, list[str]],
) -> dict[str, Any]:
    axis_contract = validate_axis_commands(commands)
    return {
        "script": str(RUNNER),
        "script_sha256": sha256_file(RUNNER),
        "trainer_contract": trainer_negative_ignore_contract(),
        "protocol_source_sha256": sha256_file(Path(protocol.__file__).resolve()),
        "historical_b4_control": b4_control_reference_contract(),
        "protocol_id": context["protocol_id"],
        "scorer_bundle_sha256": context["scorer_bundle_sha256"],
        "train_manifest": context["train_manifest"],
        "train_manifest_sha256": context["train_manifest_sha256"],
        "eval_manifest": context["eval_manifest"],
        "eval_manifest_sha256": context["eval_manifest_sha256"],
        "eval_manifest_file_sha256": sha256_file(Path(context["eval_manifest"])),
        "commands": commands,
        "axis_contract": axis_contract,
        "embedding_sidecars": {
            item["key"]: {
                "path": str(embedding_sidecar_path(item["key"])),
                "sha256": sha256_file(embedding_sidecar_path(item["key"])),
                "source_checkpoint": str(checkpoint_path(item["key"])),
                "source_checkpoint_sha256": sha256_file(
                    checkpoint_path(item["key"])
                ),
            }
            for item in ALL_CONDITIONS
        },
    }


def write_outputs(
    rows: list[dict[str, Any]],
    gate: dict[str, Any],
    context: dict[str, Any],
    commands: dict[str, list[str]],
) -> None:
    PAPER_ROOT.mkdir(parents=True, exist_ok=True)
    protocol.write_csv(OUTPUT_CSV, rows)
    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "protocol_id": context["protocol_id"],
        "screen": gate,
        "rows": rows,
        "provenance": result_provenance(context, commands),
        "outputs": {
            "json": str(OUTPUT_JSON),
            "csv": str(OUTPUT_CSV),
            "markdown": str(OUTPUT_MD),
        },
    }
    atomic_write_json(OUTPUT_JSON, payload)
    lines = [
        "# FCMAE Residual Adapter Negative-Ignore Screen (260725)",
        "",
        "- One-axis screen: explicit trainer `--ignore` only.",
        f"- Fixed context: temp {TEMPERATURE:.2f}, queue {QUEUE_SIZE}, seed {SEED}, epoch {EPOCH}.",
        "- Candidates: off, 0.80, 0.75, 0.70.",
        "- Historical B4 threshold 0.72 is a control reference.",
        "- P1/P2/P3/P4 decide screening; ARI/AMI are supporting only.",
        "- This runner never launches a follow-up.",
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
            f"- {item['label']}: accepted={item['accepted']}, "
            "minimum P3/P4 delta vs 0.72="
            f"{item['minimum_P3_P4_delta_vs_control']:.6f}"
        )
    lines.extend(
        [
            "",
            f"- proposed condition: {gate['proposed_condition']}",
            f"- proposed negative-ignore: {gate['proposed_negative_ignore']}",
            "",
            "## Absolute Outputs",
            "",
            f"- JSON: `{OUTPUT_JSON}`",
            f"- CSV: `{OUTPUT_CSV}`",
            f"- run root: `{RUN_ROOT}`",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_result(
    context: dict[str, Any],
    commands: dict[str, list[str]],
) -> tuple[bool, str]:
    if not OUTPUT_JSON.is_file():
        return False, "result JSON is missing"
    try:
        payload = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
        rows = payload["rows"]
        values = payload["screen"]["values"]
        provenance = payload["provenance"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return False, f"result JSON is invalid: {error}"

    expected_recipes = {
        "frozen",
        *(recipe_name(item["key"]) for item in ALL_CONDITIONS),
    }
    if len(rows) != 2 * len(expected_recipes):
        return False, "result row count is stale"
    recipes = {row.get("recipe") for row in rows}
    if recipes != expected_recipes:
        return False, "result recipe set is stale"
    if {item.get("condition") for item in values} != {
        item["key"] for item in SCREEN_CONDITIONS
    }:
        return False, "result condition set is stale"
    try:
        expected_provenance = result_provenance(context, commands)
    except (OSError, RuntimeError, ValueError) as error:
        return False, f"current provenance is invalid: {error}"
    if provenance != expected_provenance:
        return False, "result provenance is stale"

    for item in ALL_CONDITIONS:
        key = item["key"]
        if not embedding_is_reusable(key, context, commands):
            return False, f"embedding provenance mismatch: {key}"
        embedding_hash = sha256_file(embedding_path(key))
        candidate_rows = [
            row for row in rows if row.get("recipe") == recipe_name(key)
        ]
        if len(candidate_rows) != 2 or any(
            row.get("embedding_sha256") != embedding_hash
            for row in candidate_rows
        ):
            return False, f"result embedding rows are stale: {key}"
    return True, "current"


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--score-only", action="store_true")
    mode.add_argument("--validate-result", action="store_true")
    args = parser.parse_args()

    commands = all_commands()
    axis_contract = validate_axis_commands(commands)
    trainer_contract = trainer_negative_ignore_contract()
    b4_reference = b4_control_reference_contract()
    config = {
        "created_at": datetime.now().astimezone().isoformat(),
        "axis": "negative_ignore",
        "screen_conditions": list(SCREEN_CONDITIONS),
        "control_condition": CONTROL_CONDITION,
        "seed": SEED,
        "fixed_epoch": EPOCH,
        "fixed_temperature": TEMPERATURE,
        "fixed_queue_size": QUEUE_SIZE,
        "commands": commands,
        "axis_contract": axis_contract,
        "trainer_contract": trainer_contract,
        "historical_b4_control": b4_reference,
        "automatic_adoption": False,
        "automatic_followup": False,
    }
    if args.dry_run:
        print(json.dumps(config, indent=2, ensure_ascii=False))
        return 0

    protocol.PROTOCOL_JSON = SCREEN_PROTOCOL
    protocol.EVAL_MANIFEST = SCREEN_MANIFEST
    context = protocol.protocol_context()
    if args.validate_result:
        valid, reason = validate_result(context, commands)
        print(reason)
        return 0 if valid else 1

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(CONFIG_JSON, config)
    if not args.score_only:
        for item in ALL_CONDITIONS:
            key = item["key"]
            run_training(key, commands[key], context, commands)
    for item in ALL_CONDITIONS:
        key = item["key"]
        if not embedding_is_reusable(key, context, commands):
            raise RuntimeError(f"missing or stale embedding provenance: {key}")

    rows = protocol.score_specs(
        [frozen_spec(), *(spec(item["key"]) for item in ALL_CONDITIONS)],
        context,
    )
    gate = screen(rows)
    write_outputs(rows, gate, context, commands)
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    print(f"[out] {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
