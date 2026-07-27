#!/usr/bin/env python3
"""Prepare and run the fixed-epoch FCMAE adapter holdout validation.

This is an orchestration wrapper only.  It never trains a model.  Completed
``_ssl_methods.py`` checkpoints are copied to private staging directories so
the existing extractor can export epoch-4 embeddings for the image-disjoint
holdout without overwriting the development embeddings.

The final gate is delegated to ``fcmae_fixed_protocol.py``.  The existing
``rescore_unknown_strict_novel.py`` evaluator is also run as the canonical
holdout rescore/checkpointed CPU evaluator.  ARI and AMI are retained as
supporting columns and are not used by the gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from _common import resolve_pool  # noqa: E402

TRAIN_DIR = (REPO / "data/pools/unknown_train_defectaware_260710.json").resolve()
DEV_DIR = (REPO / "data/pools/unknown_eval100.json").resolve()
HOLDOUT_DIR = (REPO / "data/pools/unknown_holdout_100_260713.json").resolve()
HOLDOUT_MANIFEST = HOLDOUT_DIR

SEED_RUN_ROOT = (REPO / "runs/fcmae_adapter_ep4_three_seed_260721").resolve()
SEED_EMBEDDING_DIR = (SEED_RUN_ROOT / "embeddings").resolve()
SEED3_DEV_EMBEDDING = (SEED_EMBEDDING_DIR / "fcmae_ad1_s3_ep4.npy").resolve()
SEED3_CHECKPOINT_DEFAULT = (SEED_EMBEDDING_DIR / "fcmae_ad1_s3_ckpt.pt").resolve()

FROZEN_HOLDOUT = (
    REPO / "result_grouping/_hard42_holdout_260713/embeddings/fcmae_frozen_holdout.npy"
).resolve()
EXTRACTOR = (REPO / "_ssl_methods.py").resolve()
FIXED_PROTOCOL = (REPO / "scripts/fcmae_fixed_protocol.py").resolve()
HOLDOUT_EVALUATOR = (REPO / "scripts/rescore_unknown_strict_novel.py").resolve()

RUN_ROOT = (REPO / "runs/fcmae_adapter_holdout_validation_260725").resolve()
STAGING_ROOT = (RUN_ROOT / "staging").resolve()
HOLDOUT_EMBEDDINGS = (RUN_ROOT / "embeddings").resolve()
INTERIM_RESCORE = (RUN_ROOT / "canonical_rescore").resolve()
PAPER_ROOT = (REPO / "docs/paper/FCMAE_ADAPTER_HOLDOUT_260725").resolve()
FINAL_CSV = (PAPER_ROOT / "fcmae_adapter_holdout_260725.csv").resolve()
FINAL_MD = (PAPER_ROOT / "fcmae_adapter_holdout_260725.md").resolve()
FINAL_GATE = (PAPER_ROOT / "fcmae_adapter_holdout_260725_gate.json").resolve()

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FIXED_EPOCH = 4
SEEDS = (1, 3, 5)
HOLDOUT_EXPECTED_TARGET_CLASS_COUNT = 31
BATCH = 8
EPOCHS = 4
EXCLUDED_CLASSES = (
    "Normal",
    "Random",
    "R",
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
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_paths(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() == ".json":
        paths, _ = resolve_pool(root, extensions=tuple(IMAGE_EXTENSIONS))
        return [Path(path) for path in paths]
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def absolute(path: Path) -> str:
    return str(path.resolve())


def audit_manifest() -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": absolute(HOLDOUT_MANIFEST),
        "exists": HOLDOUT_MANIFEST.is_file(),
        "holdout_root": absolute(HOLDOUT_DIR),
    }
    if not HOLDOUT_MANIFEST.is_file():
        return result
    data = json.loads(HOLDOUT_MANIFEST.read_text(encoding="utf-8"))
    paths, labels = resolve_pool(HOLDOUT_DIR, extensions=tuple(IMAGE_EXTENSIONS))
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    result.update(
        {
            "sha256": sha256_file(HOLDOUT_MANIFEST),
            "protocol": "manifest-selected image-disjoint holdout",
            "selected_class_count": len(counts),
            "image_count": len(paths),
            "class_counts": dict(sorted(counts.items())),
            "image_count_matches_manifest": len(paths) == int(data.get("n_files", -1)),
        }
    )
    return result


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": absolute(path),
        "exists": path.is_file(),
    }
    if not path.is_file():
        return result
    result.update(
        {
            "bytes": path.stat().st_size,
            "mtime": datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(),
            "sha256": sha256_file(path),
        }
    )
    try:
        import torch

        loaded = torch.load(path, map_location="cpu", weights_only=False)
        result["format"] = "dict" if isinstance(loaded, dict) else type(loaded).__name__
        result["has_model"] = isinstance(loaded, dict) and "model" in loaded
        result["gstep"] = int(loaded.get("gstep", -1)) if isinstance(loaded, dict) else -1
        result["config_present"] = isinstance(loaded, dict) and "config" in loaded
        expected_steps = len(image_paths(TRAIN_DIR)) // BATCH * EPOCHS
        result["expected_gstep"] = expected_steps
        result["fixed_ep4_complete"] = (
            result["has_model"] and result["gstep"] >= expected_steps
        )
    except Exception as exc:  # A checkpoint can be read while another process is writing it.
        result["read_error"] = f"{type(exc).__name__}: {exc}"
        result["fixed_ep4_complete"] = False
    return result


def checkpoint_for_seed(seed: int, seed3_checkpoint: Path) -> Path:
    if seed == 3:
        return seed3_checkpoint
    return (SEED_EMBEDDING_DIR / f"fcmae_ad1_s{seed}_ckpt.pt").resolve()


def dev_embedding_for_seed(seed: int) -> Path:
    if seed == 3:
        return SEED3_DEV_EMBEDDING
    return (SEED_EMBEDDING_DIR / f"fcmae_ad1_s{seed}_ep4.npy").resolve()


def holdout_embedding_for_seed(seed: int) -> Path:
    return (HOLDOUT_EMBEDDINGS / f"unkda_fcmae_ad1_s{seed}_ep4.npy").resolve()


def audit(seed3_checkpoint: Path) -> dict[str, Any]:
    seed_rows = []
    for seed in SEEDS:
        checkpoint = checkpoint_for_seed(seed, seed3_checkpoint)
        dev_embedding = dev_embedding_for_seed(seed)
        holdout_embedding = holdout_embedding_for_seed(seed)
        seed_rows.append(
            {
                "seed": seed,
                "checkpoint": inspect_checkpoint(checkpoint),
                "dev_embedding": {
                    "path": absolute(dev_embedding),
                    "exists": dev_embedding.is_file(),
                    "bytes": dev_embedding.stat().st_size if dev_embedding.is_file() else None,
                },
                "holdout_embedding": {
                    "path": absolute(holdout_embedding),
                    "exists": holdout_embedding.is_file(),
                },
            }
        )
    return {
        "fixed_epoch": FIXED_EPOCH,
        "seeds": list(SEEDS),
        "train_dir": absolute(TRAIN_DIR),
        "dev_dir": absolute(DEV_DIR),
        "holdout": audit_manifest(),
        "frozen_holdout": {
            "path": absolute(FROZEN_HOLDOUT),
            "exists": FROZEN_HOLDOUT.is_file(),
        },
        "seed_rows": seed_rows,
        "existing_seed3_dev_state": absolute(SEED3_DEV_EMBEDDING),
        "extractor": absolute(EXTRACTOR),
        "fixed_protocol": absolute(FIXED_PROTOCOL),
        "holdout_evaluator": absolute(HOLDOUT_EVALUATOR),
    }


def extraction_command(seed: int, checkpoint: Path, staging: Path) -> list[str]:
    tag = f"fcmae_ad1_s{seed}"
    return [
        sys.executable,
        "-u",
        absolute(EXTRACTOR),
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
        str(EPOCHS),
        "--batch",
        str(BATCH),
        "--ckpt-every",
        "100",
        "--train-dir",
        absolute(TRAIN_DIR),
        "--eval-dir",
        absolute(HOLDOUT_DIR),
        "--wafer-rot-deg",
        "0",
        "--wafer-translate",
        "0",
        "--wafer-scale-min",
        "1.0",
        "--wafer-crop-min",
        "1.0",
        "--out-dir",
        absolute(staging),
        "--tag",
        tag,
    ]


def rescore_command() -> list[str]:
    return [
        sys.executable,
        "-u",
        absolute(HOLDOUT_EVALUATOR),
        "--embeddings",
        absolute(HOLDOUT_EMBEDDINGS),
        "--frozen",
        absolute(FROZEN_HOLDOUT),
        "--pool",
        absolute(HOLDOUT_DIR),
        "--output-dir",
        absolute(INTERIM_RESCORE),
        "--refresh",
    ]


def print_command(command: list[str]) -> None:
    print(f"[cmd] {subprocess.list2cmdline([str(item) for item in command])}")


def dry_run_report(state: dict[str, Any], seed3_checkpoint: Path) -> int:
    print(json.dumps(state, indent=2, ensure_ascii=False))
    print("[paths] all paths above are absolute")
    blocked = False
    for seed in SEEDS:
        row = next(item for item in state["seed_rows"] if item["seed"] == seed)
        checkpoint = Path(row["checkpoint"]["path"])
        complete = bool(row["checkpoint"].get("fixed_ep4_complete", False))
        staging = (STAGING_ROOT / f"seed{seed}").resolve()
        if complete:
            print(f"[ready] seed {seed} checkpoint: {absolute(checkpoint)}")
            print(f"[stage] seed {seed}: {absolute(staging)}")
            print_command(extraction_command(seed, checkpoint, staging))
        else:
            blocked = True
            print(
                f"[blocked] seed {seed} requires a readable completed epoch-4 checkpoint: "
                f"{absolute(checkpoint)}"
            )
            print_command(extraction_command(seed, checkpoint, staging))
    print("[after-extraction]")
    print_command(rescore_command())
    print(f"[final-csv] {absolute(FINAL_CSV)}")
    print(f"[final-md] {absolute(FINAL_MD)}")
    print(f"[final-gate-json] {absolute(FINAL_GATE)}")
    if blocked:
        print("[status] BLOCKED: do not run until seed1/3/5 fixed-epoch checkpoints are complete")
        return 2
    print("[status] READY: master may run extraction and scoring")
    return 0


def run_extraction(seed: int, checkpoint: Path) -> Path:
    staging = (STAGING_ROOT / f"seed{seed}").resolve()
    staging.mkdir(parents=True, exist_ok=True)
    staged_checkpoint = (staging / f"fcmae_ad1_s{seed}_ckpt.pt").resolve()
    shutil.copy2(checkpoint, staged_checkpoint)
    command = extraction_command(seed, staged_checkpoint, staging)
    print_command(command)
    subprocess.run(command, cwd=REPO, check=True)
    source = (staging / f"fcmae_ad1_s{seed}_ep4.npy").resolve()
    if not source.is_file():
        raise FileNotFoundError(f"extractor did not produce {absolute(source)}")
    target = holdout_embedding_for_seed(seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target


def validate_ready(state: dict[str, Any]) -> None:
    required = [
        ("holdout root", Path(state["holdout"]["holdout_root"])),
        ("holdout manifest", Path(state["holdout"]["path"])),
        ("frozen holdout embedding", Path(state["frozen_holdout"]["path"])),
    ]
    for name, path in required:
        if not path.is_file() and name != "holdout root":
            raise FileNotFoundError(f"missing {name}: {absolute(path)}")
        if name == "holdout root" and not path.is_dir():
            raise FileNotFoundError(f"missing {name}: {absolute(path)}")
    for row in state["seed_rows"]:
        checkpoint = row["checkpoint"]
        if not checkpoint.get("fixed_ep4_complete", False):
            raise RuntimeError(
                f"seed {row['seed']} checkpoint is not a completed fixed-epoch checkpoint: "
                f"{checkpoint['path']}"
            )


def load_fixed_protocol():
    import importlib.util

    spec = importlib.util.spec_from_file_location("fcmae_fixed_protocol_holdout", FIXED_PROTOCOL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import fixed protocol: {absolute(FIXED_PROTOCOL)}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_holdout_report(fp, rows: list[dict[str, Any]], frozen_rows: list[dict[str, Any]], gate: dict[str, Any], context: dict[str, Any], audit_state: dict[str, Any]) -> None:
    FINAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    fp.write_csv(FINAL_CSV, rows)
    lines = [
        "# FCMAE Adapter Fixed-Epoch Image-Disjoint Holdout Validation",
        "",
        f"- protocol_id: `{context['protocol_id']}`",
        f"- holdout: `{absolute(HOLDOUT_DIR)}`",
        f"- holdout manifest: `{absolute(HOLDOUT_MANIFEST)}`",
        f"- fixed point: residual adapter, pure SimCLR, epoch {FIXED_EPOCH}, seeds 1/3/5",
        "- P1/P2/P3/P4 and fragmentation decide the gate; ARI/AMI are supporting diagnostics only.",
        "- The holdout is image-disjoint from the development evaluation according to its manifest.",
        "",
        "## Gate",
        "",
        f"- accepted: **{str(gate['accepted']).upper()}**",
        f"- next axis: {gate['next_axis']}",
        "",
        "## Frozen Baseline",
        "",
    ]
    lines.extend(fp.row_table(frozen_rows))
    lines.extend(["", "## Fixed-epoch Candidate Rows", ""])
    lines.extend(fp.row_table(rows))
    lines.extend(["", "## Gate Details", ""])
    for clusterer, detail in gate["clusterers"].items():
        lines.append(f"### {clusterer}")
        lines.append("")
        lines.append(f"- accepted: {detail['accepted']}")
        for name, passed in detail["checks"].items():
            lines.append(f"- {name}: {passed}")
        means = detail["mean_delta"]
        lines.append(
            "- mean deltas: recov={recov:+.4f}, P3={p3:+.4f}, P4={p4:+.4f}, "
            "fragment improvement={frag:+.4f}, Sil*={sil:+.4f}, ARI*={ari:+.4f}, AMI*={ami:+.4f}".format(
                recov=means["recov"],
                p3=means["P3_completeness"],
                p4=means["P4_homogeneity"],
                frag=means["fragment_improvement"],
                sil=means["Sil_supporting"],
                ari=means["ARI_supporting"],
                ami=means["AMI_supporting"],
            )
        )
        lines.append("")
    lines.extend(
        [
            "## Provenance",
            "",
            f"- fixed protocol: `{absolute(FIXED_PROTOCOL)}`",
            f"- existing holdout evaluator: `{absolute(HOLDOUT_EVALUATOR)}`",
            f"- frozen holdout embedding: `{absolute(FROZEN_HOLDOUT)}`",
            f"- interim evaluator output: `{absolute(INTERIM_RESCORE)}`",
            "",
            "Raw CSV includes ARI/AMI as supporting columns; neither can rescue a failed P1/P2/P3/P4 gate.",
        ]
    )
    FINAL_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    gate_payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "protocol_id": context["protocol_id"],
        "gate": gate,
        "audit": audit_state,
        "outputs": {
            "csv": absolute(FINAL_CSV),
            "md": absolute(FINAL_MD),
            "gate_json": absolute(FINAL_GATE),
        },
    }
    gate_text = json.dumps(gate_payload, indent=2, ensure_ascii=False)
    gate_temp = FINAL_GATE.with_name(f".{FINAL_GATE.name}.{os.getpid()}.tmp")
    gate_temp.write_text(gate_text, encoding="utf-8")
    os.replace(gate_temp, FINAL_GATE)


def run_validation(seed3_checkpoint: Path) -> None:
    state = audit(seed3_checkpoint)
    validate_ready(state)
    HOLDOUT_EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        target = holdout_embedding_for_seed(seed)
        if target.is_file():
            print(f"[skip] existing holdout embedding: {absolute(target)}")
            continue
        run_extraction(seed, checkpoint_for_seed(seed, seed3_checkpoint))

    INTERIM_RESCORE.mkdir(parents=True, exist_ok=True)
    command = rescore_command()
    print_command(command)
    subprocess.run(command, cwd=REPO, check=True)

    fp = load_fixed_protocol()
    fp.EVAL_DIR = HOLDOUT_DIR
    fp.EVAL_MANIFEST = (PAPER_ROOT / "holdout_eval_manifest.json").resolve()
    fp.PROTOCOL_JSON = (PAPER_ROOT / "fcmae_adapter_holdout_protocol.json").resolve()
    context = fp.protocol_context(
        expected_target_class_count=HOLDOUT_EXPECTED_TARGET_CLASS_COUNT
    )
    specs = [
        {
            "recipe": "frozen",
            "recipe_flags": "FCMAE frozen backbone f; holdout baseline",
            "seed": 0,
            "epoch": 0,
            "embedding_space": "frozen_f",
            "path": FROZEN_HOLDOUT,
        }
    ]
    for seed in SEEDS:
        specs.append(
            {
                "recipe": "L0_base_ep4",
                "recipe_flags": "zero-init residual adapter; pure SimCLR B0; fixed epoch4",
                "seed": seed,
                "epoch": FIXED_EPOCH,
                "embedding_space": "adapted_f",
                "path": holdout_embedding_for_seed(seed),
            }
        )
    rows = fp.score_specs(specs, context)
    frozen_rows = [row for row in rows if row["recipe"] == "frozen"]
    candidate_rows = [row for row in rows if row["recipe"] != "frozen"]
    gate = fp.evaluate_three_seed_gate(candidate_rows, frozen_rows, seeds=SEEDS)
    write_holdout_report(fp, rows, frozen_rows, gate, context, state)
    print(f"[out] {absolute(FINAL_CSV)}")
    print(f"[out] {absolute(FINAL_MD)}")
    print(f"[out] {absolute(FINAL_GATE)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit and validate FCMAE adapter epoch-4 models on the image-disjoint hard-42 holdout. "
            "This command never trains; use --dry-run to print the master execution plan."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="audit absolute paths and print commands only")
    mode.add_argument("--run", action="store_true", help="extract, rescore, and write the final gate artifacts")
    parser.add_argument(
        "--seed3-checkpoint",
        type=Path,
        default=SEED3_CHECKPOINT_DEFAULT,
        help="completed seed-3 adapter checkpoint; the existing dev .npy alone cannot produce holdout embeddings",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed3_checkpoint = args.seed3_checkpoint.expanduser().resolve()
    state = audit(seed3_checkpoint)
    if args.dry_run:
        return dry_run_report(state, seed3_checkpoint)
    run_validation(seed3_checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
