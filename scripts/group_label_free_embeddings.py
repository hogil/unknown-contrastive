#!/usr/bin/env python3
"""Production label-free grouping for precomputed embeddings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from scripts.label_free_gate import (
        HDBSCAN_CONFIG,
        hdbscan_fit,
        l2_normalize,
        load_embedding,
        read_path_manifest,
    )
    from scripts.predict_grouping_prod import save_grouping_representatives
except ImportError:
    from label_free_gate import HDBSCAN_CONFIG, hdbscan_fit, l2_normalize, load_embedding, read_path_manifest
    from predict_grouping_prod import save_grouping_representatives


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_gate(path: Path, model_mode: str, paths_manifest: Path, embedding: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload.get("selected_mode")
    if selected is not None and selected != model_mode:
        raise ValueError(
            f"gate selected_mode={selected!r} does not match --model-mode={model_mode!r}"
        )
    provenance = payload.get("provenance")
    input_files = provenance.get("input_files") if isinstance(provenance, dict) else None
    if not isinstance(input_files, dict):
        raise ValueError("gate provenance.input_files is missing")

    def expected_hash(name: str) -> str:
        record = input_files.get(name)
        if not isinstance(record, dict) or not isinstance(record.get("sha256"), str):
            raise ValueError(f"gate provenance input hash is missing: {name}")
        return record["sha256"]

    actual_manifest_hash = sha256(paths_manifest)
    if expected_hash("paths_manifest") != actual_manifest_hash:
        raise ValueError("gate paths manifest hash does not match current input")
    embedding_key = f"{model_mode}_embedding"
    if expected_hash(embedding_key) != sha256(embedding):
        raise ValueError(f"gate selected embedding hash does not match current input: {embedding_key}")
    return payload


def validate_paths(paths: list[str]) -> None:
    for index, value in enumerate(paths):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"manifest row {index} is not a non-empty string")
        if not Path(value).is_absolute():
            raise ValueError(f"manifest row {index} is not an absolute path")


def copy_groups(out_dir: Path, paths: list[str], prediction: np.ndarray) -> None:
    groups_dir = out_dir / "groups"
    groups_dir.mkdir(parents=True, exist_ok=True)
    for group_id in sorted(int(value) for value in set(prediction.tolist()) if value >= 0):
        target = groups_dir / f"group_{group_id:03d}"
        target.mkdir(parents=True, exist_ok=True)
        for rank, index in enumerate(np.flatnonzero(prediction == group_id), 1):
            source = Path(paths[int(index)])
            if not source.is_file():
                continue
            shutil.copy2(source, target / f"{rank:06d}_{source.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Group opaque, label-free embedding rows with fixed HDBSCAN.")
    parser.add_argument("--paths-manifest", type=Path, required=True)
    parser.add_argument("--embedding", type=Path, required=True)
    parser.add_argument("--gate-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-mode", choices=("frozen", "adapted"), required=True)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--copy-groups", action="store_true")
    args = parser.parse_args()
    if args.reps < 1:
        raise ValueError("--reps must be positive")

    args.paths_manifest = args.paths_manifest.resolve()
    args.embedding = args.embedding.resolve()
    args.gate_json = args.gate_json.resolve()
    args.out_dir = args.out_dir.resolve()

    paths = read_path_manifest(args.paths_manifest)
    validate_paths(paths)
    embedding = l2_normalize(load_embedding(args.embedding))
    if len(paths) != embedding.shape[0]:
        raise ValueError(
            f"manifest/embedding row mismatch: {len(paths)} paths vs {embedding.shape[0]} rows"
        )
    gate = read_gate(args.gate_json, args.model_mode, args.paths_manifest, args.embedding)
    prediction = hdbscan_fit(embedding)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    group_ids = sorted(int(value) for value in set(prediction.tolist()) if value >= 0)
    sizes = {group_id: int(np.count_nonzero(prediction == group_id)) for group_id in group_ids}
    selected_summary = gate.get(args.model_mode, {})
    stability = selected_summary.get("bootstrap_stability")
    rows = []
    for index, path in enumerate(paths):
        group_id = int(prediction[index])
        rows.append({
            "path": path,
            "group_id": group_id,
            "review_status": "noise" if group_id < 0 else "candidate",
            "group_size": 0 if group_id < 0 else sizes[group_id],
            "global_grouping_stability": "" if group_id < 0 or stability is None else f"{float(stability):.8f}",
            "model_mode": args.model_mode,
        })
    with (args.out_dir / "clusters.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    saved = save_grouping_representatives(args.out_dir, embedding, prediction, paths, args.reps)
    if args.copy_groups:
        copy_groups(args.out_dir, paths, prediction)

    summary = {
        "schema_version": "label_free_grouping.v1",
        "model_mode": args.model_mode,
        "n": len(paths),
        "embedding_dim": int(embedding.shape[1]),
        "cluster_count": len(group_ids),
        "noise_count": int(np.count_nonzero(prediction < 0)),
        "noise_pct": 100.0 * float(np.count_nonzero(prediction < 0)) / len(prediction),
        "groups": {str(group_id): sizes[group_id] for group_id in group_ids},
        "stability_scope": "global_gate_bootstrap_not_per_cluster",
        "representatives_saved": saved,
        "copy_groups": bool(args.copy_groups),
        "hdbscan": HDBSCAN_CONFIG,
        "gate_decision": {
            "selected_mode": gate.get("selected_mode"),
            "workflow_action": gate.get("workflow_action"),
            "selection_reason": gate.get("selection_reason"),
        },
        "provenance": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": [str(value) for value in sys.argv],
            "python": platform.python_version(),
            "paths_manifest": {"path": str(args.paths_manifest.resolve()), "sha256": sha256(args.paths_manifest)},
            "embedding": {"path": str(args.embedding.resolve()), "sha256": sha256(args.embedding)},
            "gate_json": {"path": str(args.gate_json.resolve()), "sha256": sha256(args.gate_json)},
            "manifest_order_used": True,
            "manifest_row_count": len(paths),
            "labels_read": False,
            "folder_names_read": False,
            "majority_label_used": False,
        },
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"cluster_count": len(group_ids), "noise_count": summary["noise_count"]}, indent=2))
    print(f"[OUT] {(args.out_dir / 'clusters.csv').resolve()}")
    print(f"[OUT] {(args.out_dir / 'summary.json').resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2)
