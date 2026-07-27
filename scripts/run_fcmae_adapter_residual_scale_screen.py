#!/usr/bin/env python3
"""Screen inference-time residual scale for hard-42 FCMAE adapter seed 1."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import fcmae_fixed_protocol as protocol  # noqa: E402


SOURCE_CHECKPOINT = (
    REPO
    / "runs/fcmae_adapter_ep4_three_seed_260721/embeddings/fcmae_ad1_s1_ckpt.pt"
).resolve()
TRAINER = (REPO / "_ssl_methods.py").resolve()
RUN_ROOT = (
    Path("E:/unknown-contrastive-runs/archives")
    / "fcmae_adapter_residual_scale_screen_260725"
).resolve()
EMBEDDINGS = (RUN_ROOT / "embeddings").resolve()
OUTPUT_JSON = (RUN_ROOT / "residual_scale_screen.json").resolve()
OUTPUT_CSV = (RUN_ROOT / "residual_scale_screen.csv").resolve()
OUTPUT_MD = (RUN_ROOT / "residual_scale_screen.md").resolve()
ALPHAS = (0.25, 0.50, 0.75, 1.00)
SEED = 1
EPOCH = 4
BATCH = 8
EXPECTED_GSTEP = 3296
PROVENANCE_SCHEMA = 1


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


def tag(alpha: float) -> str:
    return f"fcmae_ad1_s1_rs{int(round(alpha * 100)):03d}"


def checkpoint_path(alpha: float) -> Path:
    return EMBEDDINGS / f"{tag(alpha)}_ckpt.pt"


def embedding_path(alpha: float) -> Path:
    return EMBEDDINGS / f"{tag(alpha)}_ep4.npy"


def embedding_sidecar_path(alpha: float) -> Path:
    path = embedding_path(alpha)
    return path.with_name(f"{path.name}.provenance.json")


def load_source() -> tuple[dict[str, Any], str, float]:
    if not SOURCE_CHECKPOINT.is_file():
        raise FileNotFoundError("seed-1 epoch-4 source checkpoint is missing")
    source_hash = sha256_file(SOURCE_CHECKPOINT)
    state = torch.load(SOURCE_CHECKPOINT, map_location="cpu", weights_only=False)
    if int(state.get("gstep", -1)) != EXPECTED_GSTEP:
        raise ValueError(
            f"source gstep={state.get('gstep')} != expected {EXPECTED_GSTEP}"
        )
    gamma = state.get("model", {}).get("ad_gamma")
    if not isinstance(gamma, torch.Tensor) or gamma.numel() != 1:
        raise ValueError("source checkpoint must contain one model.ad_gamma tensor")
    gamma_value = float(gamma.item())
    if not np.isfinite(gamma_value):
        raise ValueError("source ad_gamma is not finite")
    return state, source_hash, gamma_value


def derive_checkpoint(
    source: dict[str, Any], source_hash: str, alpha: float
) -> Path:
    target = checkpoint_path(alpha)
    script_hash = sha256_file(Path(__file__).resolve())
    if target.is_file():
        try:
            existing = torch.load(target, map_location="cpu", weights_only=False)
            provenance = existing.get("residual_scale_provenance", {})
            if (
                provenance.get("source_sha256") == source_hash
                and provenance.get("alpha") == f"{alpha:.2f}"
                and provenance.get("script_sha256") == script_hash
                and int(existing.get("gstep", -1)) == EXPECTED_GSTEP
            ):
                return target
        except (OSError, RuntimeError, TypeError, ValueError):
            pass
    derived = copy.deepcopy(source)
    derived["model"]["ad_gamma"] = source["model"]["ad_gamma"].clone() * alpha
    derived["residual_scale_provenance"] = {
        "source_path": str(SOURCE_CHECKPOINT),
        "source_sha256": source_hash,
        "alpha": f"{alpha:.2f}",
        "script_sha256": script_hash,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".pt.tmp")
    torch.save(derived, temporary)
    temporary.replace(target)
    return target


def embedding_contract(
    alpha: float,
    source_hash: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = checkpoint_path(alpha)
    protocol_source = Path(protocol.__file__).resolve()
    eval_manifest = Path(context["eval_manifest"]).resolve()
    return {
        "schema": PROVENANCE_SCHEMA,
        "axis": "residual_scale",
        "alpha": f"{alpha:.2f}",
        "seed": SEED,
        "epoch": EPOCH,
        "expected_gstep": EXPECTED_GSTEP,
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_checkpoint_sha256": source_hash,
        "extraction_checkpoint": str(checkpoint),
        "extraction_checkpoint_sha256": sha256_file(checkpoint),
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
        "command_sha256": sha256_json(extraction_command(alpha)),
    }


def embedding_is_reusable(
    alpha: float,
    source_hash: str,
    context: dict[str, Any],
) -> bool:
    embedding = embedding_path(alpha)
    checkpoint = checkpoint_path(alpha)
    sidecar = embedding_sidecar_path(alpha)
    if not embedding.is_file() or not checkpoint.is_file() or not sidecar.is_file():
        return False
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        return (
            payload.get("contract") == embedding_contract(alpha, source_hash, context)
            and payload.get("embedding") == str(embedding)
            and payload.get("embedding_sha256") == sha256_file(embedding)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def write_embedding_sidecar(
    alpha: float,
    source_hash: str,
    context: dict[str, Any],
) -> None:
    embedding = embedding_path(alpha)
    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "contract": embedding_contract(alpha, source_hash, context),
        "embedding": str(embedding),
        "embedding_sha256": sha256_file(embedding),
    }
    atomic_write_json(embedding_sidecar_path(alpha), payload)


def remove_stale_embedding(alpha: float) -> None:
    embedding = embedding_path(alpha)
    projection = embedding.with_name(f"{embedding.stem}_proj.npy")
    for path in (embedding, projection, embedding_sidecar_path(alpha)):
        path.unlink(missing_ok=True)


def extraction_command(alpha: float) -> list[str]:
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
        str(BATCH),
        "--ckpt-every",
        "100",
        "--temp",
        "0.05",
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
        tag(alpha),
    ]


def extract(
    alpha: float,
    source_hash: str,
    context: dict[str, Any],
) -> None:
    expected = embedding_path(alpha)
    if embedding_is_reusable(alpha, source_hash, context):
        return
    remove_stale_embedding(alpha)
    checkpoint = checkpoint_path(alpha)
    before = sha256_file(checkpoint)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        extraction_command(alpha),
        cwd=REPO,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log = RUN_ROOT / f"{tag(alpha)}.log"
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"alpha {alpha:.2f} extraction failed: {log}")
    if not re.search(rf"gstep\s+{EXPECTED_GSTEP}/{EXPECTED_GSTEP}", result.stdout):
        raise RuntimeError(f"alpha {alpha:.2f} did not resume at fixed epoch 4")
    if re.search(r"(?m)^\[simclr ep\d+\]\s+loss=", result.stdout):
        raise RuntimeError(f"alpha {alpha:.2f} unexpectedly performed training")
    if sha256_file(checkpoint) != before:
        raise RuntimeError(f"alpha {alpha:.2f} extraction changed its checkpoint")
    if not expected.is_file():
        raise RuntimeError(f"alpha {alpha:.2f} embedding was not created")
    write_embedding_sidecar(alpha, source_hash, context)


def spec(alpha: float) -> dict[str, Any]:
    return {
        "recipe": f"residual_scale_{alpha:.2f}",
        "recipe_flags": (
            "no-TAPT FCMAE; seed1 epoch4 checkpoint; inference-only "
            f"ad_gamma scale={alpha:.2f}"
        ),
        "seed": SEED,
        "epoch": EPOCH,
        "embedding_space": "adapted_f",
        "path": embedding_path(alpha),
    }


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frozen = protocol.frozen_by_clusterer(rows)
    values = []
    for alpha in ALPHAS:
        candidate = [
            row for row in rows if row["recipe"] == f"residual_scale_{alpha:.2f}"
        ]
        checks = {}
        minimum_delta = float("inf")
        for row in candidate:
            base = frozen[row["clusterer"]]
            delta_p3 = float(row["P3_completeness"]) - float(base["P3_completeness"])
            delta_p4 = float(row["P4_homogeneity"]) - float(base["P4_homogeneity"])
            minimum_delta = min(minimum_delta, delta_p3, delta_p4)
            checks[row["clusterer"]] = {
                "P1_preserved": int(row["P1_capture_count"])
                >= int(base["P1_capture_count"]),
                "P2_not_worse": float(row["P2_noise_pct"])
                <= float(base["P2_noise_pct"]) + 1e-9,
                "P3_not_worse": delta_p3 >= -1e-9,
                "P4_not_worse": delta_p4 >= -1e-9,
                "P3_delta": round(delta_p3, 6),
                "P4_delta": round(delta_p4, 6),
            }
        accepted = len(checks) == 2 and all(
            all(value for key, value in item.items() if key.endswith(("preserved", "worse")))
            for item in checks.values()
        )
        values.append(
            {
                "alpha": alpha,
                "accepted": accepted,
                "minimum_P3_P4_delta": round(minimum_delta, 6),
                "clusterers": checks,
            }
        )
    accepted = sorted(
        (item for item in values if item["accepted"]),
        key=lambda item: (-item["minimum_P3_P4_delta"], item["alpha"]),
    )
    return {
        "values": values,
        "proposed_alpha": accepted[0]["alpha"] if accepted else None,
        "automatic_followup_launched": False,
    }


def write_outputs(
    rows: list[dict[str, Any]],
    gate: dict[str, Any],
    source_hash: str,
    source_gamma: float,
    original_hash_after: str,
    context: dict[str, Any],
) -> None:
    if source_hash != original_hash_after:
        raise RuntimeError("source checkpoint changed during the screen")
    protocol.write_csv(OUTPUT_CSV, rows)
    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "contract": "Both clusterers preserve P1/P2 and do not worsen P3/P4",
        "gate": gate,
        "rows": rows,
        "provenance": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "source_checkpoint": str(SOURCE_CHECKPOINT),
            "source_checkpoint_sha256": source_hash,
            "source_ad_gamma": source_gamma,
            "trainer_sha256": sha256_file(TRAINER),
            "protocol_source_sha256": sha256_file(Path(protocol.__file__).resolve()),
            "protocol_id": context["protocol_id"],
            "scorer_bundle_sha256": context["scorer_bundle_sha256"],
            "eval_manifest": context["eval_manifest"],
            "eval_manifest_sha256": context["eval_manifest_sha256"],
            "eval_manifest_file_sha256": sha256_file(Path(context["eval_manifest"])),
            "embedding_sidecars": {
                f"{alpha:.2f}": {
                    "path": str(embedding_sidecar_path(alpha)),
                    "sha256": sha256_file(embedding_sidecar_path(alpha)),
                }
                for alpha in ALPHAS
            },
        },
    }
    atomic_write_json(OUTPUT_JSON, payload)
    lines = [
        "# FCMAE Residual Scale Screen",
        "",
        "- Inference-only alpha screen; no training and no automatic follow-up.",
        "- P1/P2/P3/P4 are primary. ARI/AMI remain supporting columns.",
        "",
        *protocol.row_table(rows),
        "",
        "## Gate",
        "",
    ]
    lines.extend(
        f"- alpha {item['alpha']:.2f}: accepted={item['accepted']}, "
        f"minimum P3/P4 delta={item['minimum_P3_P4_delta']:.6f}"
        for item in gate["values"]
    )
    lines.append(f"- proposed alpha: {gate['proposed_alpha']}")
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_result(
    source_hash: str,
    context: dict[str, Any],
) -> tuple[bool, str]:
    if not OUTPUT_JSON.is_file():
        return False, "result JSON is missing"
    try:
        payload = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
        values = payload["gate"]["values"]
        rows = payload["rows"]
        provenance = payload["provenance"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return False, f"result JSON is invalid: {error}"
    if len(values) != len(ALPHAS) or len(rows) != 2 * (len(ALPHAS) + 1):
        return False, "result row/value count is stale"
    if sorted(item.get("alpha") for item in values) != list(ALPHAS):
        return False, "result alpha set is stale"
    expected = {
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "source_checkpoint_sha256": source_hash,
        "trainer_sha256": sha256_file(TRAINER),
        "protocol_source_sha256": sha256_file(Path(protocol.__file__).resolve()),
        "protocol_id": context["protocol_id"],
        "scorer_bundle_sha256": context["scorer_bundle_sha256"],
        "eval_manifest": context["eval_manifest"],
        "eval_manifest_sha256": context["eval_manifest_sha256"],
        "eval_manifest_file_sha256": sha256_file(Path(context["eval_manifest"])),
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            return False, f"result provenance mismatch: {key}"
    recorded_sidecars = provenance.get("embedding_sidecars", {})
    for alpha in ALPHAS:
        key = f"{alpha:.2f}"
        sidecar = embedding_sidecar_path(alpha)
        if not embedding_is_reusable(alpha, source_hash, context):
            return False, f"embedding provenance mismatch: alpha {key}"
        recorded = recorded_sidecars.get(key, {})
        if (
            recorded.get("path") != str(sidecar)
            or recorded.get("sha256") != sha256_file(sidecar)
        ):
            return False, f"result sidecar mismatch: alpha {key}"
        expected_hash = sha256_file(embedding_path(alpha))
        candidate_rows = [
            row
            for row in rows
            if row.get("recipe") == f"residual_scale_{alpha:.2f}"
        ]
        if len(candidate_rows) != 2 or any(
            row.get("embedding_sha256") != expected_hash for row in candidate_rows
        ):
            return False, f"result embedding rows are stale: alpha {key}"
    return True, "current"


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--score-only", action="store_true")
    mode.add_argument("--validate-result", action="store_true")
    args = parser.parse_args()
    source, source_hash, source_gamma = load_source()
    protocol.PROTOCOL_JSON = RUN_ROOT / "protocol.json"
    protocol.EVAL_MANIFEST = RUN_ROOT / "eval_manifest.json"
    context = protocol.protocol_context()
    if args.validate_result:
        valid, reason = validate_result(source_hash, context)
        print(reason)
        return 0 if valid else 1
    plan = {
        "source_checkpoint": str(SOURCE_CHECKPOINT),
        "source_checkpoint_sha256": source_hash,
        "source_ad_gamma": source_gamma,
        "expected_gstep": EXPECTED_GSTEP,
        "alphas": list(ALPHAS),
        "commands": {
            f"{alpha:.2f}": extraction_command(alpha) for alpha in ALPHAS
        },
        "automatic_followup": False,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    for alpha in ALPHAS:
        if not args.score_only:
            derive_checkpoint(source, source_hash, alpha)
            extract(alpha, source_hash, context)
        if not embedding_is_reusable(alpha, source_hash, context):
            raise RuntimeError(
                f"missing or stale alpha {alpha:.2f} embedding provenance"
            )
    frozen = {
        "recipe": "frozen",
        "recipe_flags": "FCMAE frozen; no training",
        "seed": "none",
        "epoch": 0,
        "embedding_space": "backbone_f",
        "path": protocol.FROZEN_EMBEDDING,
    }
    rows = protocol.score_specs([frozen, *(spec(alpha) for alpha in ALPHAS)], context)
    gate = evaluate(rows)
    write_outputs(
        rows,
        gate,
        source_hash,
        source_gamma,
        sha256_file(SOURCE_CHECKPOINT),
        context,
    )
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    print(f"[out] {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
