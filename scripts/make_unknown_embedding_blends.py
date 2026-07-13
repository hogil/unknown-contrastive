#!/usr/bin/env python3
"""Create label-free frozen/trained embedding blends for hard-unknown scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "result_grouping" / "_field_robust" / "embeddings" / "frozen_unknown_dinov3_grade_only_260709.npy"
TRAINED = ROOT / "result_grouping" / "_unknown_mixed260710" / "embeddings" / "unkda_nv050_ep6.npy"
OUTPUT = ROOT / "result_grouping" / "_unknown_mixed260710" / "embeddings"
DEFAULT_WEIGHTS = (0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", type=Path, default=FROZEN)
    parser.add_argument("--trained", type=Path, default=TRAINED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--weights", type=float, nargs="+", default=DEFAULT_WEIGHTS)
    parser.add_argument("--tag-prefix", default="unkda_nv050")
    parser.add_argument("--epoch", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def fingerprint(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def main() -> None:
    args = parse_args()
    frozen_path = args.frozen.resolve()
    trained_path = args.trained.resolve()
    output_dir = args.output_dir.resolve()
    if not frozen_path.exists() or not trained_path.exists():
        raise FileNotFoundError("Both frozen and trained embedding files are required")

    weights = tuple(sorted(set(args.weights)))
    if not weights or any(weight <= 0.0 or weight >= 1.0 for weight in weights):
        raise ValueError("weights must be strictly between 0 and 1")

    frozen = l2(np.load(frozen_path).astype(np.float32))
    trained = l2(np.load(trained_path).astype(np.float32))
    if frozen.shape != trained.shape:
        raise ValueError(f"Embedding shape mismatch: frozen={frozen.shape}, trained={trained.shape}")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, object]] = []
    for trained_weight in weights:
        tag = f"{args.tag_prefix}_blend{round(trained_weight * 100):03d}_ep{args.epoch}.npy"
        destination = output_dir / tag
        if destination.exists() and not args.force:
            print(f"[skip] {destination}")
        else:
            # Each source is unit-normalized. Concatenation preserves a fixed
            # row norm and makes cosine similarity a convex blend of both spaces.
            blend = np.concatenate(
                (
                    np.sqrt(1.0 - trained_weight, dtype=np.float32) * frozen,
                    np.sqrt(trained_weight, dtype=np.float32) * trained,
                ),
                axis=1,
            )
            np.save(destination, blend.astype(np.float32, copy=False))
            print(f"[out] {destination}")
        outputs.append({"trained_weight": trained_weight, "embedding": str(destination)})

    manifest = {
        "protocol": "unit-normalize each source; concatenate sqrt(1-w)*frozen with sqrt(w)*trained; no labels used",
        "frozen": fingerprint(frozen_path),
        "trained": fingerprint(trained_path),
        "tag_prefix": args.tag_prefix,
        "epoch": args.epoch,
        "outputs": outputs,
    }
    manifest_path = output_dir / "unknown_nv050_blend_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[manifest] {manifest_path}")


if __name__ == "__main__":
    main()
