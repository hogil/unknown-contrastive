#!/usr/bin/env python3
"""Run the May iter-70 NEW recipe with TAPT and raw-FCMAE paired backbones.

The committed May wrapper only calls its NeCo hook through the Local-loss
branch.  That makes ``USE_LOCAL=false, NECO_WEIGHT>0`` a silent no-op.  This
runner materializes the archived source and applies one recorded semantic
patch: the Local branch becomes an execution hook that returns NeCo only.
Everything else is held fixed, including the locked reconstructed anchor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_may37_original_ablation as may


CONTROL_ID = "may_new_tapt_removed_paired_2260"
CELL = "NEW_FIXED"
SEMANTIC_RECIPE = "Global InfoNCE + NeCo(0.2) + Queue(4096) + NEG(0.72); Local OFF"
PATCH_ID = "neco_replace_local_v1"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected one {label} marker, found {text.count(old)}")
    return text.replace(old, new, 1)


def patch_contrastive(text: str) -> str:
    text = may.instrument_epoch_checkpoints(text)
    marker = "    model = CL().to(device).to(memory_format=torch.channels_last)\n"
    replacement = marker + (
        "    _epoch_dir = Path(run_dir) / \"checkpoints\" / \"epoch_checkpoints\"\n"
        "    _epoch_dir.mkdir(parents=True, exist_ok=True)\n"
        "    torch.save({\"state_dict\": model.state_dict(), \"epoch\": 0}, "
        "_epoch_dir / \"epoch_000.pt\")\n"
    )
    return replace_once(text, marker, replacement, "epoch-0 checkpoint")


def patch_wrapper(text: str) -> str:
    text = replace_once(
        text,
        "def install_neco_loss(weight: float, tau: float = 0.1) -> None:\n",
        "def install_neco_loss(weight: float, tau: float = 0.1, replace_local: bool = False) -> None:\n",
        "NeCo function signature",
    )
    standard_start = text.index(
        "def install_neco_loss(weight: float, tau: float = 0.1, replace_local: bool = False) -> None:\n"
    )
    standard_end = text.index("def install_neco_loss_zone_aware", standard_start)
    standard_block = replace_once(
        text[standard_start:standard_end],
        "        local_loss = _orig_local(f1, f2, **kwargs)\n",
        "        local_loss = f1.new_zeros(()) if replace_local else _orig_local(f1, f2, **kwargs)\n",
        "standard NeCo local term",
    )
    text = text[:standard_start] + standard_block + text[standard_end:]
    old_call = (
        "        install_neco_loss(\n"
        "            weight=_neco_w,\n"
        "            tau=float(os.environ.get(\"NECO_TAU\", 0.1)),\n"
        "        )\n"
    )
    new_call = (
        "        install_neco_loss(\n"
        "            weight=_neco_w,\n"
        "            tau=float(os.environ.get(\"NECO_TAU\", 0.1)),\n"
        "            replace_local=os.environ.get(\"NECO_REPLACE_LOCAL\", \"false\").lower() == \"true\",\n"
        "        )\n"
    )
    return replace_once(text, old_call, new_call, "NeCo install call")


def materialize_patched_source(output_root: Path) -> tuple[Path, dict]:
    source_dir = output_root / f"_source_{may.SOURCE_COMMIT[:12]}_{PATCH_ID}"
    source_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict[str, str | bool]] = {}
    for name in may.SOURCE_FILES:
        original = may.git_source(name)
        if name == "contrastive.py":
            patched = patch_contrastive(original)
        elif name == "run_contrastive.py":
            patched = patch_wrapper(original)
        else:
            patched = original
        compile(patched, name, "exec")
        target = source_dir / name
        target.write_text(patched, encoding="utf-8")
        records[name] = {
            "archived_sha256": hashlib.sha256(original.encode("utf-8")).hexdigest(),
            "patched_sha256": hashlib.sha256(patched.encode("utf-8")).hexdigest(),
            "patched": patched != original,
        }
    provenance = {
        "source_commit": may.SOURCE_COMMIT,
        "patch_id": PATCH_ID,
        "patch_reason": (
            "Archived NeCo was reachable only when USE_LOCAL=true. The patch uses that branch "
            "as an execution hook but returns NeCo only, matching the documented iter-70 formula."
        ),
        "semantic_recipe": SEMANTIC_RECIPE,
        "files": records,
    }
    provenance_path = source_dir / "semantic_patch_provenance.json"
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    return source_dir, provenance


def training_env(backbone: str, seed: int, anchor: Path, checkpoint: Path) -> dict[str, str]:
    env = may.training_env(backbone, "B6", anchor, checkpoint)
    env.update(
        {
            "SEED": str(seed),
            # Execution hook only. The patched wrapper removes Local from the value returned.
            "USE_LOCAL": "true",
            "LOCAL_WEIGHT": "1.0",
            "NECO_WEIGHT": "0.2",
            "NECO_TAU": "0.1",
            "NECO_REPLACE_LOCAL": "true",
            "USE_QUEUE": "true",
            "QUEUE_SIZE": "4096",
            "IGNORE_NEG_SIM": "0.72",
            "NCE_TEMP": "0.07",
            "LR_HEAD": "0.001",
            "TRAIN_SAMPLING_RATIO": "0.25",
            "EPOCHS": "5",
            "BATCH": "8",
            "IMAGE_SIZE": "384",
            "FREEZE_BACKBONE": "true",
        }
    )
    return env


def run_training(
    source_dir: Path,
    output_root: Path,
    backbone: str,
    seed: int,
    anchor: Path,
) -> Path:
    checkpoint = may.normalized_backbone_checkpoint(output_root, backbone)
    before = {path.resolve() for path in ROOT.glob("outputs_contrastive_*") if path.is_dir()}
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    tag = f"{CONTROL_ID}_{backbone}_{CELL.lower()}_s{seed}"
    log_path = logs / f"{stamp}_{tag}.log"
    command = [sys.executable, "-u", str(source_dir / "run_contrastive.py")]
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            cwd=source_dir,
            env=training_env(backbone, seed, anchor, checkpoint),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    after = {path.resolve() for path in ROOT.glob("outputs_contrastive_*") if path.is_dir()}
    created = sorted(after - before, key=lambda path: path.stat().st_mtime)
    if len(created) != 1:
        raise RuntimeError(f"expected one archived output directory, got {created}; log={log_path}")
    run_dir = output_root / f"{stamp}_{tag}"
    shutil.move(str(created[0]), str(run_dir))
    shutil.copy2(log_path, run_dir / "source_train.log")

    run_info_path = run_dir / "run_info.json"
    run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
    run_info["cell"] = CELL
    run_info["backbone"] = backbone
    run_info["semantic_recipe"] = SEMANTIC_RECIPE
    run_info["semantic_patch_id"] = PATCH_ID
    run_info["cfg"].update(
        {
            "SEED": seed,
            "NECO_WEIGHT": 0.2,
            "NECO_TAU": 0.1,
            "NECO_REPLACE_LOCAL": True,
            "SEMANTIC_USE_LOCAL": False,
            "USE_LOCAL_EXECUTION_HOOK": True,
        }
    )
    run_info_path.write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir


def evaluate(
    source_dir: Path,
    run_dir: Path,
    output_root: Path,
    anchor: Path,
    anchor_manifest: dict,
    backbone: str,
    seed: int,
    patch_provenance: dict,
) -> Path:
    final_projection = may.evaluate_run(
        source_dir,
        run_dir,
        "projection",
        anchor,
        anchor_manifest,
        backbone,
        CELL,
        CONTROL_ID,
    )
    final_backbone = may.evaluate_run(
        source_dir,
        run_dir,
        "backbone",
        anchor,
        anchor_manifest,
        backbone,
        CELL,
        CONTROL_ID,
        primary=False,
    )

    epoch_dir = run_dir / "checkpoints" / "epoch_checkpoints"
    checkpoints = sorted(epoch_dir.glob("epoch_*.pt"))
    expected = [f"epoch_{epoch:03d}.pt" for epoch in range(6)]
    if [path.name for path in checkpoints] != expected:
        raise RuntimeError(f"incomplete epoch trajectory: {[path.name for path in checkpoints]}")
    trajectory = []
    for epoch, checkpoint_path in enumerate(checkpoints):
        row = may.evaluate_run(
            source_dir,
            run_dir,
            "projection",
            anchor,
            anchor_manifest,
            backbone,
            CELL,
            CONTROL_ID,
            primary=False,
            checkpoint_path=checkpoint_path,
            artifact_prefix=f"epoch_{epoch:03d}",
            run_multicluster=(epoch == 0),
        )
        row["epoch"] = epoch
        trajectory.append(row)
    if trajectory[-1]["embedding_sha256"] != final_projection["embedding_sha256"]:
        raise RuntimeError("epoch-5 projection does not match final projection")
    trajectory_path = run_dir / "canonical_eval" / "epoch_trajectory_projection.json"
    trajectory_path.write_text(json.dumps(trajectory, ensure_ascii=False, indent=2), encoding="utf-8")

    provenance = {
        "control_id": CONTROL_ID,
        "source_commit": may.SOURCE_COMMIT,
        "semantic_patch": patch_provenance,
        "semantic_recipe": SEMANTIC_RECIPE,
        "backbone": backbone,
        "backbone_checkpoint": str(may.backbone_checkpoint(backbone).resolve()),
        "backbone_checkpoint_sha256": may.sha256_file(may.backbone_checkpoint(backbone)),
        "seed": seed,
        "anchor_manifest": str((output_root / "anchor_manifest.json").resolve()),
        "anchor_manifest_sha256": anchor_manifest["inventory_sha256"],
        "anchor_n_images": anchor_manifest["n_images"],
        "historical_reference": (
            "Original May file_list.parquet is unavailable; this is a manifest-locked paired "
            "recipe reproduction, not an exact historical-score reproduction."
        ),
    }
    (run_dir / "reproduction_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    completion = {
        "control_id": CONTROL_ID,
        "backbone": backbone,
        "cell": CELL,
        "seed": seed,
        "semantic_recipe": SEMANTIC_RECIPE,
        "anchor_manifest_sha256": anchor_manifest["inventory_sha256"],
        "primary_embedding": "projection",
        "primary_embedding_sha256": final_projection["embedding_sha256"],
        "frozen_backbone_embedding_sha256": final_backbone["embedding_sha256"],
        "epoch_trajectory_complete": len(trajectory) == 6,
        "metrics_path": str((run_dir / "canonical_eval" / "metrics_projection.json").resolve()),
        "tier1_metrics_path": str(
            (run_dir / "canonical_eval" / "historical_tier1_defect_only_projection.json").resolve()
        ),
        "trajectory_path": str(trajectory_path.resolve()),
    }
    completion_path = run_dir / "completion.json"
    completion_path.write_text(json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf-8")
    return completion_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", choices=["cnn_tapt", "nocnn"], required=True)
    parser.add_argument("--seed", type=int, choices=[1, 2, 42], required=True)
    parser.add_argument(
        "--anchor",
        type=Path,
        default=ROOT / "data" / "images" / "anchor_avg30_repro",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "runs" / CONTROL_ID,
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    anchor = args.anchor.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    anchor_manifest = may.write_anchor_manifest(anchor, output_root)
    source_dir, patch_provenance = materialize_patched_source(output_root)
    print(
        f"[PREPARED] source={source_dir} patch={PATCH_ID} anchor_sha={anchor_manifest['inventory_sha256']}",
        flush=True,
    )
    if args.prepare_only:
        return
    run_dir = run_training(source_dir, output_root, args.backbone, args.seed, anchor)
    print(f"[RUN_DIR] {run_dir}", flush=True)
    completion = evaluate(
        source_dir,
        run_dir,
        output_root,
        anchor,
        anchor_manifest,
        args.backbone,
        args.seed,
        patch_provenance,
    )
    print(f"[COMPLETE] {completion}", flush=True)


if __name__ == "__main__":
    main()
