#!/usr/bin/env python3
"""Build an image-disjoint hard-unknown holdout from the unused source pool.

The existing ``unknown_eval100`` images are excluded by filename.  This keeps
the evaluation classes unchanged while ensuring no image appears in both the
development pool and this holdout.  The source has no unused sample for one
class (``Thick-Edge_fork``), which is recorded rather than silently replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "images" / "unknown_train_all"
REFERENCE = ROOT / "data" / "images" / "unknown_eval100"
OUTPUT = ROOT / "data" / "images" / "unknown_holdout_100_260713"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--seed", default="260713")
    return parser.parse_args()


def stable_key(path: Path, seed: str) -> str:
    return hashlib.sha256(f"{seed}:{path.name}".encode("utf-8")).hexdigest()


def ensure_hardlink(source: Path, destination: Path) -> bool:
    if destination.exists():
        if os.path.samefile(source, destination):
            return False
        raise FileExistsError(f"Holdout destination already points elsewhere: {destination}")
    os.link(source, destination)
    return True


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    reference = args.reference.resolve()
    output = args.output.resolve()
    if not source.is_dir() or not reference.is_dir():
        raise FileNotFoundError("Both --source and --reference must be image-root directories")
    if args.per_class <= 0:
        raise ValueError("--per-class must be positive")

    output.mkdir(parents=True, exist_ok=True)
    selected: dict[str, list[str]] = {}
    omitted: dict[str, int] = {}
    linked = 0
    existing = 0

    for reference_dir in sorted(path for path in reference.iterdir() if path.is_dir()):
        class_name = reference_dir.name
        source_dir = source / class_name
        reference_names = {path.name for path in reference_dir.iterdir() if path.is_file()}
        candidates = [
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.name not in reference_names
        ] if source_dir.is_dir() else []
        candidates.sort(key=lambda path: stable_key(path, args.seed))
        if len(candidates) < args.per_class:
            omitted[class_name] = len(candidates)
            continue

        destination_dir = output / class_name
        destination_dir.mkdir(exist_ok=True)
        chosen = candidates[:args.per_class]
        for image in chosen:
            if ensure_hardlink(image, destination_dir / image.name):
                linked += 1
            else:
                existing += 1
        selected[class_name] = [image.name for image in chosen]

    manifest = {
        "protocol": "image-disjoint holdout: source filenames overlapping the development eval are excluded",
        "source": str(source),
        "reference_eval": str(reference),
        "output": str(output),
        "per_class": args.per_class,
        "seed": args.seed,
        "selected_classes": sorted(selected),
        "selected_count": sum(len(images) for images in selected.values()),
        "omitted_classes": omitted,
        "selected_files": selected,
    }
    manifest_path = output / "manifest_260713.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OUT] {output}")
    print(f"[MANIFEST] {manifest_path}")
    print(f"[SUMMARY] classes={len(selected)} images={manifest['selected_count']} linked={linked} existing={existing}")
    if omitted:
        print(f"[OMITTED] {omitted}")


if __name__ == "__main__":
    main()
