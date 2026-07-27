#!/usr/bin/env python3
"""Confirm a residual-scale proposal across FCMAE adapter seeds 1/3/5.

The runner is deliberately inference-only. It reads the accepted alpha from
the residual-scale screen, scales ``model.ad_gamma`` from each original epoch-4
checkpoint, extracts ``unknown_eval100`` embeddings, and applies the existing
fixed P1/P2/P3/P4 three-seed gate. ARI/AMI remain supporting diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
import run_fcmae_adapter_residual_scale_screen as residual  # noqa: E402


SCREEN_RESULT = (
    Path("E:/unknown-contrastive-runs/archives")
    / "fcmae_adapter_residual_scale_screen_260725/residual_scale_screen.json"
).resolve()
SCREEN_PROTOCOL = (
    Path("E:/unknown-contrastive-runs/archives")
    / "fcmae_adapter_residual_scale_screen_260725/protocol.json"
).resolve()
SOURCE_ROOT = (
    REPO / "runs/fcmae_adapter_ep4_three_seed_260721/embeddings"
).resolve()
RUN_ROOT = (REPO / "runs/fcmae_adapter_alpha_three_seed_260725").resolve()
EMBEDDINGS = (RUN_ROOT / "embeddings").resolve()
OUTPUT_JSON = (RUN_ROOT / "alpha_three_seed_gate.json").resolve()
TRAINER = (REPO / "_ssl_methods.py").resolve()

SEEDS = (1, 3, 5)
EPOCH = 4
BATCH = 8
EXPECTED_GSTEP = 3296
PROVENANCE_SCHEMA = 1

EXIT_SUCCESS = 0
EXIT_ERROR = 2
EXIT_NO_PROPOSAL = 20

EXPECTED_CLUSTERERS = ("FINCH-p2", "Louvain-res6")
SCREEN_CHECKS = (
    "P1_preserved",
    "P2_not_worse",
    "P3_not_worse",
    "P4_not_worse",
)


class NoProposal(RuntimeError):
    """The upstream screen has no alpha to confirm."""


class ContractError(RuntimeError):
    """An input or provenance contract is invalid."""


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


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def alpha_text(alpha: float) -> str:
    return f"{alpha:.2f}"


def run_tag(seed: int, alpha: float) -> str:
    return f"fcmae_ad1_s{seed}_a{int(round(alpha * 100)):03d}"


def source_checkpoint(seed: int) -> Path:
    return SOURCE_ROOT / f"fcmae_ad1_s{seed}_ckpt.pt"


def scaled_checkpoint(seed: int, alpha: float) -> Path:
    return EMBEDDINGS / f"{run_tag(seed, alpha)}_ckpt.pt"


def embedding_path(seed: int, alpha: float) -> Path:
    return EMBEDDINGS / f"{run_tag(seed, alpha)}_ep{EPOCH}.npy"


def embedding_sidecar_path(seed: int, alpha: float) -> Path:
    embedding = embedding_path(seed, alpha)
    return embedding.with_name(f"{embedding.name}.provenance.json")


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ContractError(f"{description} is missing: {path}") from error
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ContractError(f"{description} is invalid: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ContractError(f"{description} must be a JSON object: {path}")
    return payload


def read_screen_proposal() -> tuple[float, dict[str, Any]]:
    if not SCREEN_RESULT.is_file():
        raise NoProposal(f"residual screen result is not available: {SCREEN_RESULT}")
    payload = _read_json(SCREEN_RESULT, "residual screen result")
    gate = payload.get("gate")
    if not isinstance(gate, dict):
        raise ContractError("residual screen result has no gate object")
    proposed = gate.get("proposed_alpha")
    if proposed is None:
        raise NoProposal("residual screen proposed_alpha is null")
    try:
        alpha = float(proposed)
    except (TypeError, ValueError) as error:
        raise ContractError("residual screen proposed_alpha is not numeric") from error
    if not math.isfinite(alpha) or alpha <= 0:
        raise ContractError(f"invalid proposed_alpha: {proposed!r}")
    if not any(math.isclose(alpha, value, abs_tol=1e-12) for value in residual.ALPHAS):
        raise ContractError(
            f"proposed_alpha {alpha_text(alpha)} is outside the residual screen axis"
        )

    values = gate.get("values")
    if not isinstance(values, list):
        raise ContractError("residual screen gate.values is missing")
    matches = [
        item
        for item in values
        if isinstance(item, dict)
        and isinstance(item.get("alpha"), (int, float))
        and math.isclose(float(item["alpha"]), alpha, abs_tol=1e-12)
    ]
    if len(matches) != 1:
        raise ContractError("proposed_alpha must match exactly one gate value")
    selected = matches[0]
    if selected.get("accepted") is not True:
        raise ContractError("proposed_alpha is not accepted by the residual screen")
    clusterers = selected.get("clusterers")
    if not isinstance(clusterers, dict) or set(clusterers) != set(
        EXPECTED_CLUSTERERS
    ):
        raise ContractError("proposed_alpha must contain FINCH and Louvain checks")
    for clusterer in EXPECTED_CLUSTERERS:
        checks = clusterers[clusterer]
        if not isinstance(checks, dict) or any(
            checks.get(name) is not True for name in SCREEN_CHECKS
        ):
            raise ContractError(
                f"proposed_alpha did not pass every {clusterer} P1/P2/P3/P4 check"
            )
    return alpha, payload


def validate_screen_provenance(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    if SCREEN_RESULT.resolve() != residual.OUTPUT_JSON.resolve():
        raise ContractError("screen result path differs from residual runner output")
    context = _read_json(SCREEN_PROTOCOL, "residual screen protocol")
    if Path(context.get("eval_dir", "")).resolve() != protocol.EVAL_DIR.resolve():
        raise ContractError("residual screen eval_dir is not unknown_eval100")
    if Path(context.get("train_manifest", "")).resolve() != (
        protocol.TRAIN_MANIFEST.resolve()
    ):
        raise ContractError("residual screen train manifest path is stale")
    if context.get("train_manifest_sha256") != sha256_file(
        protocol.TRAIN_MANIFEST
    ):
        raise ContractError("residual screen train manifest hash is stale")

    _, current_eval_digest = protocol.build_eval_manifest()
    if context.get("eval_manifest_sha256") != current_eval_digest:
        raise ContractError("residual screen eval manifest is stale")
    _, current_scorer_bundle = protocol.scorer_provenance()
    if context.get("scorer_bundle_sha256") != current_scorer_bundle:
        raise ContractError("residual screen scorer bundle is stale")

    if not residual.SOURCE_CHECKPOINT.is_file():
        raise ContractError(
            f"residual screen source checkpoint is missing: "
            f"{residual.SOURCE_CHECKPOINT}"
        )
    source_hash = sha256_file(residual.SOURCE_CHECKPOINT)
    valid, reason = residual.validate_result(source_hash, context)
    if not valid:
        raise ContractError(f"residual screen provenance is invalid: {reason}")
    if payload.get("provenance", {}).get("protocol_id") != context.get(
        "protocol_id"
    ):
        raise ContractError("residual screen payload protocol_id is stale")
    return context, sha256_file(SCREEN_RESULT)


def load_source_state(seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    path = source_checkpoint(seed)
    if not path.is_file():
        raise ContractError(f"seed {seed} source checkpoint is missing: {path}")
    before = sha256_file(path)
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ContractError(f"seed {seed} source checkpoint is unreadable") from error
    if not isinstance(state, dict) or int(state.get("gstep", -1)) != EXPECTED_GSTEP:
        raise ContractError(
            f"seed {seed} source gstep={state.get('gstep')} "
            f"!= {EXPECTED_GSTEP}"
        )
    model = state.get("model")
    gamma = model.get("ad_gamma") if isinstance(model, dict) else None
    if not isinstance(gamma, torch.Tensor) or gamma.numel() != 1:
        raise ContractError(
            f"seed {seed} source must contain one model.ad_gamma tensor"
        )
    gamma_value = float(gamma.item())
    if not math.isfinite(gamma_value):
        raise ContractError(f"seed {seed} source ad_gamma is not finite")
    if sha256_file(path) != before:
        raise ContractError(f"seed {seed} source changed while it was read")
    return state, {
        "path": str(path.resolve()),
        "sha256": before,
        "ad_gamma": gamma_value,
        "gstep": EXPECTED_GSTEP,
    }


def checkpoint_contract(
    seed: int,
    alpha: float,
    source: dict[str, Any],
    screen_hash: str,
) -> dict[str, Any]:
    return {
        "schema": PROVENANCE_SCHEMA,
        "axis": "residual_scale_three_seed",
        "seed": seed,
        "epoch": EPOCH,
        "expected_gstep": EXPECTED_GSTEP,
        "alpha": alpha_text(alpha),
        "source_checkpoint": source["path"],
        "source_checkpoint_sha256": source["sha256"],
        "source_ad_gamma": source["ad_gamma"],
        "scaled_ad_gamma": source["ad_gamma"] * alpha,
        "screen_result": str(SCREEN_RESULT),
        "screen_result_sha256": screen_hash,
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }


def scaled_checkpoint_is_reusable(
    seed: int,
    alpha: float,
    source: dict[str, Any],
    screen_hash: str,
) -> bool:
    target = scaled_checkpoint(seed, alpha)
    if not target.is_file():
        return False
    try:
        state = torch.load(target, map_location="cpu", weights_only=False)
        gamma = state.get("model", {}).get("ad_gamma")
        return (
            int(state.get("gstep", -1)) == EXPECTED_GSTEP
            and state.get("alpha_three_seed_provenance")
            == checkpoint_contract(seed, alpha, source, screen_hash)
            and isinstance(gamma, torch.Tensor)
            and gamma.numel() == 1
            and math.isclose(
                float(gamma.item()),
                float(source["ad_gamma"]) * alpha,
                rel_tol=1e-6,
                abs_tol=1e-9,
            )
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def derive_scaled_checkpoint(
    seed: int,
    alpha: float,
    screen_hash: str,
) -> dict[str, Any]:
    state, source = load_source_state(seed)
    if scaled_checkpoint_is_reusable(seed, alpha, source, screen_hash):
        return source
    gamma = state["model"]["ad_gamma"]
    state["model"]["ad_gamma"] = gamma.clone() * alpha
    state["alpha_three_seed_provenance"] = checkpoint_contract(
        seed, alpha, source, screen_hash
    )
    target = scaled_checkpoint(seed, alpha)
    atomic_torch_save(target, state)
    if not scaled_checkpoint_is_reusable(seed, alpha, source, screen_hash):
        raise ContractError(f"seed {seed} derived checkpoint failed validation")
    if sha256_file(source_checkpoint(seed)) != source["sha256"]:
        raise ContractError(f"seed {seed} source checkpoint was modified")
    return source


def extraction_command(seed: int, alpha: float) -> list[str]:
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
        str(seed),
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
        run_tag(seed, alpha),
    ]


def embedding_contract(
    seed: int,
    alpha: float,
    source: dict[str, Any],
    screen_hash: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = scaled_checkpoint(seed, alpha)
    eval_manifest = Path(context["eval_manifest"]).resolve()
    return {
        **checkpoint_contract(seed, alpha, source, screen_hash),
        "extraction_checkpoint": str(checkpoint.resolve()),
        "extraction_checkpoint_sha256": sha256_file(checkpoint),
        "trainer": str(TRAINER),
        "trainer_sha256": sha256_file(TRAINER),
        "protocol_source": str(Path(protocol.__file__).resolve()),
        "protocol_source_sha256": sha256_file(Path(protocol.__file__).resolve()),
        "protocol_id": context["protocol_id"],
        "scorer_bundle_sha256": context["scorer_bundle_sha256"],
        "train_manifest": context["train_manifest"],
        "train_manifest_sha256": context["train_manifest_sha256"],
        "eval_dir": context["eval_dir"],
        "eval_manifest": str(eval_manifest),
        "eval_manifest_sha256": context["eval_manifest_sha256"],
        "eval_manifest_file_sha256": sha256_file(eval_manifest),
        "command_sha256": sha256_json(extraction_command(seed, alpha)),
    }


def embedding_is_reusable(
    seed: int,
    alpha: float,
    source: dict[str, Any],
    screen_hash: str,
    context: dict[str, Any],
) -> bool:
    embedding = embedding_path(seed, alpha)
    sidecar = embedding_sidecar_path(seed, alpha)
    if not embedding.is_file() or not sidecar.is_file():
        return False
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        return (
            payload.get("contract")
            == embedding_contract(seed, alpha, source, screen_hash, context)
            and payload.get("embedding") == str(embedding.resolve())
            and payload.get("embedding_sha256") == sha256_file(embedding)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def write_embedding_sidecar(
    seed: int,
    alpha: float,
    source: dict[str, Any],
    screen_hash: str,
    context: dict[str, Any],
) -> None:
    embedding = embedding_path(seed, alpha)
    atomic_write_json(
        embedding_sidecar_path(seed, alpha),
        {
            "created_at": datetime.now().astimezone().isoformat(),
            "contract": embedding_contract(
                seed, alpha, source, screen_hash, context
            ),
            "embedding": str(embedding.resolve()),
            "embedding_sha256": sha256_file(embedding),
        },
    )


def remove_stale_embedding(seed: int, alpha: float) -> None:
    embedding = embedding_path(seed, alpha)
    projection = embedding.with_name(f"{embedding.stem}_proj.npy")
    for path in (embedding, projection, embedding_sidecar_path(seed, alpha)):
        path.unlink(missing_ok=True)


def extract_embedding(
    seed: int,
    alpha: float,
    source: dict[str, Any],
    screen_hash: str,
    context: dict[str, Any],
) -> None:
    if embedding_is_reusable(seed, alpha, source, screen_hash, context):
        return
    remove_stale_embedding(seed, alpha)
    checkpoint = scaled_checkpoint(seed, alpha)
    checkpoint_hash = sha256_file(checkpoint)
    source_hash = sha256_file(source_checkpoint(seed))
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run(
        extraction_command(seed, alpha),
        cwd=REPO,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log = RUN_ROOT / f"{run_tag(seed, alpha)}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        raise ContractError(f"seed {seed} extraction failed: {log}")
    if not re.search(
        rf"gstep\s+{EXPECTED_GSTEP}/{EXPECTED_GSTEP}", result.stdout
    ):
        raise ContractError(f"seed {seed} did not resume at fixed epoch {EPOCH}")
    if re.search(r"(?m)^\[simclr ep\d+\]\s+loss=", result.stdout):
        raise ContractError(f"seed {seed} unexpectedly performed training")
    if sha256_file(checkpoint) != checkpoint_hash:
        raise ContractError(f"seed {seed} scaled checkpoint changed during extraction")
    if sha256_file(source_checkpoint(seed)) != source_hash:
        raise ContractError(f"seed {seed} source checkpoint changed during extraction")
    if not embedding_path(seed, alpha).is_file():
        raise ContractError(f"seed {seed} epoch-{EPOCH} embedding was not created")
    write_embedding_sidecar(seed, alpha, source, screen_hash, context)


def candidate_spec(seed: int, alpha: float) -> dict[str, Any]:
    return {
        "recipe": f"L0_alpha_{alpha_text(alpha)}_ep4",
        "recipe_flags": (
            "no-TAPT FCMAE; adapter pure SimCLR B0; fixed epoch4; "
            f"inference-only ad_gamma scale={alpha_text(alpha)}"
        ),
        "seed": seed,
        "epoch": EPOCH,
        "embedding_space": "adapted_f",
        "path": embedding_path(seed, alpha),
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


def evaluate_rows(rows: list[dict[str, Any]], alpha: float) -> dict[str, Any]:
    frozen_rows = [row for row in rows if row["recipe"] == "frozen"]
    candidate_rows = [
        row for row in rows if row["recipe"] == f"L0_alpha_{alpha_text(alpha)}_ep4"
    ]
    gate = protocol.evaluate_three_seed_gate(
        candidate_rows, frozen_rows, seeds=SEEDS
    )
    gate["alpha"] = alpha
    gate["ARI_AMI"] = "supporting only; excluded from gate"
    return gate


def write_result(
    alpha: float,
    screen_payload: dict[str, Any],
    screen_hash: str,
    context: dict[str, Any],
    sources: dict[int, dict[str, Any]],
    rows: list[dict[str, Any]],
    gate: dict[str, Any],
) -> None:
    seed_provenance: dict[str, Any] = {}
    for seed in SEEDS:
        sidecar = embedding_sidecar_path(seed, alpha)
        seed_provenance[str(seed)] = {
            "source_checkpoint": sources[seed],
            "scaled_checkpoint": str(scaled_checkpoint(seed, alpha).resolve()),
            "scaled_checkpoint_sha256": sha256_file(
                scaled_checkpoint(seed, alpha)
            ),
            "embedding": str(embedding_path(seed, alpha).resolve()),
            "embedding_sha256": sha256_file(embedding_path(seed, alpha)),
            "embedding_sidecar": str(sidecar.resolve()),
            "embedding_sidecar_sha256": sha256_file(sidecar),
        }
    payload = {
        "schema": PROVENANCE_SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(),
        "contract": (
            "No-TAPT FCMAE, fixed epoch4, inference-only residual alpha; "
            "FINCH-p2 and Louvain-res6 P1/P2/P3/P4 three-seed gate"
        ),
        "alpha": alpha,
        "accepted": bool(gate["accepted"]),
        "gate": gate,
        "rows": rows,
        "provenance": {
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "screen_result": str(SCREEN_RESULT),
            "screen_result_sha256": screen_hash,
            "screen_proposed_alpha": screen_payload["gate"]["proposed_alpha"],
            "screen_runner_sha256": screen_payload["provenance"][
                "script_sha256"
            ],
            "trainer": str(TRAINER),
            "trainer_sha256": sha256_file(TRAINER),
            "protocol_source": str(Path(protocol.__file__).resolve()),
            "protocol_source_sha256": sha256_file(
                Path(protocol.__file__).resolve()
            ),
            "protocol_id": context["protocol_id"],
            "scorer_bundle_sha256": context["scorer_bundle_sha256"],
            "train_manifest": context["train_manifest"],
            "train_manifest_sha256": context["train_manifest_sha256"],
            "eval_dir": context["eval_dir"],
            "eval_manifest": context["eval_manifest"],
            "eval_manifest_sha256": context["eval_manifest_sha256"],
            "eval_manifest_file_sha256": sha256_file(
                Path(context["eval_manifest"])
            ),
            "frozen_embedding": str(protocol.FROZEN_EMBEDDING.resolve()),
            "frozen_embedding_sha256": sha256_file(
                protocol.FROZEN_EMBEDDING
            ),
            "seeds": seed_provenance,
        },
    }
    atomic_write_json(OUTPUT_JSON, payload)


def validate_result(
    alpha: float,
    screen_hash: str,
    context: dict[str, Any],
) -> tuple[bool, str]:
    if not OUTPUT_JSON.is_file():
        return False, "result JSON is missing"
    try:
        payload = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
        rows = payload["rows"]
        provenance = payload["provenance"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return False, f"result JSON is invalid: {error}"
    if not math.isclose(float(payload.get("alpha")), alpha, abs_tol=1e-12):
        return False, "result alpha is stale"
    if len(rows) != 2 + 2 * len(SEEDS):
        return False, "result row count is stale"
    expected = {
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "screen_result_sha256": screen_hash,
        "trainer_sha256": sha256_file(TRAINER),
        "protocol_source_sha256": sha256_file(Path(protocol.__file__).resolve()),
        "protocol_id": context["protocol_id"],
        "scorer_bundle_sha256": context["scorer_bundle_sha256"],
        "train_manifest_sha256": context["train_manifest_sha256"],
        "eval_manifest_sha256": context["eval_manifest_sha256"],
        "eval_manifest_file_sha256": sha256_file(Path(context["eval_manifest"])),
        "frozen_embedding_sha256": sha256_file(protocol.FROZEN_EMBEDDING),
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            return False, f"result provenance mismatch: {key}"
    sources = provenance.get("seeds", {})
    for seed in SEEDS:
        try:
            _, source = load_source_state(seed)
        except ContractError as error:
            return False, str(error)
        if not scaled_checkpoint_is_reusable(seed, alpha, source, screen_hash):
            return False, f"seed {seed} scaled checkpoint provenance is stale"
        if not embedding_is_reusable(seed, alpha, source, screen_hash, context):
            return False, f"seed {seed} embedding provenance is stale"
        recorded = sources.get(str(seed), {})
        sidecar = embedding_sidecar_path(seed, alpha)
        if (
            recorded.get("source_checkpoint") != source
            or recorded.get("scaled_checkpoint_sha256")
            != sha256_file(scaled_checkpoint(seed, alpha))
            or recorded.get("embedding_sha256")
            != sha256_file(embedding_path(seed, alpha))
            or recorded.get("embedding_sidecar_sha256") != sha256_file(sidecar)
        ):
            return False, f"seed {seed} result artifact hashes are stale"
    frozen_rows = [row for row in rows if row.get("recipe") == "frozen"]
    candidate_rows = [
        row
        for row in rows
        if row.get("recipe") == f"L0_alpha_{alpha_text(alpha)}_ep4"
    ]
    try:
        recomputed = evaluate_rows(frozen_rows + candidate_rows, alpha)
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        return False, f"result gate cannot be recomputed: {error}"
    if payload.get("gate") != recomputed:
        return False, "result gate is stale"
    if payload.get("accepted") is not bool(recomputed["accepted"]):
        return False, "result accepted flag is stale"
    return True, "current"


def plan(alpha: float, screen_hash: str) -> dict[str, Any]:
    return {
        "mode": "inference_only",
        "no_tapt": True,
        "epoch": EPOCH,
        "alpha": alpha,
        "seeds": list(SEEDS),
        "screen_result": str(SCREEN_RESULT),
        "screen_result_sha256": screen_hash,
        "source_checkpoints": {
            str(seed): str(source_checkpoint(seed).resolve()) for seed in SEEDS
        },
        "commands": {
            str(seed): extraction_command(seed, alpha) for seed in SEEDS
        },
        "output": str(OUTPUT_JSON),
        "gate": "FINCH-p2 + Louvain-res6; P1/P2/P3/P4 primary",
        "ARI_AMI": "supporting only",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--score-only", action="store_true")
    mode.add_argument("--validate-result", action="store_true")
    args = parser.parse_args(argv)

    try:
        alpha, screen_payload = read_screen_proposal()
        context, screen_hash = validate_screen_provenance(screen_payload)
        if args.dry_run:
            print(json.dumps(plan(alpha, screen_hash), indent=2, ensure_ascii=False))
            return EXIT_SUCCESS
        if args.validate_result:
            valid, reason = validate_result(alpha, screen_hash, context)
            print(reason)
            return EXIT_SUCCESS if valid else EXIT_ERROR

        sources: dict[int, dict[str, Any]] = {}
        for seed in SEEDS:
            if args.score_only:
                _, source = load_source_state(seed)
            else:
                source = derive_scaled_checkpoint(seed, alpha, screen_hash)
                extract_embedding(
                    seed, alpha, source, screen_hash, context
                )
            if not embedding_is_reusable(
                seed, alpha, source, screen_hash, context
            ):
                raise ContractError(
                    f"seed {seed} embedding is missing or has stale provenance"
                )
            sources[seed] = source

        specs = [frozen_spec(), *(candidate_spec(seed, alpha) for seed in SEEDS)]
        rows = protocol.score_specs(specs, context)
        gate = evaluate_rows(rows, alpha)
        write_result(
            alpha,
            screen_payload,
            screen_hash,
            context,
            sources,
            rows,
            gate,
        )
        print(json.dumps(gate, indent=2, ensure_ascii=False))
        print(f"[out] {OUTPUT_JSON}")
        return EXIT_SUCCESS
    except NoProposal as error:
        print(
            json.dumps(
                {
                    "status": "no_op",
                    "reason": str(error),
                    "exit_code": EXIT_NO_PROPOSAL,
                },
                ensure_ascii=False,
            )
        )
        return EXIT_NO_PROPOSAL
    except ContractError as error:
        print(f"[error] {error}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
