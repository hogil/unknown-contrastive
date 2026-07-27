#!/usr/bin/env python3
"""Deterministic cross-split exact and perceptual duplicate audit.

The auditor is read-only with respect to manifests and images.  It writes one
atomic JSON report and never copies, moves, links, or deletes image data.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps


SCHEMA_VERSION = "manifest_overlap_audit.v1"
DHASH_METHOD = "dhash64-v1:exif_transpose,L,9x8,LANCZOS,left>right"
BLOCK_FIELDS = ("block_id", "lot", "source", "temporal_batch", "group")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_obj(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _is_within(path: Path, roots: Iterable[Path]) -> bool:
    candidate = os.path.normcase(str(path))
    for root in roots:
        root_text = os.path.normcase(str(root))
        try:
            if os.path.commonpath((candidate, root_text)) == root_text:
                return True
        except ValueError:
            continue
    return False


def _resolve_roots(values: Iterable[str | Path]) -> list[Path]:
    roots = []
    for value in values:
        root = Path(value)
        if not root.is_absolute():
            raise ValueError(f"allowed root must be absolute: {value}")
        resolved = root.resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError(f"allowed root is not a directory: {resolved}")
        roots.append(resolved)
    if not roots:
        raise ValueError("at least one allowed root is required")
    return sorted(set(roots), key=lambda item: os.path.normcase(str(item)))


def _block_id(row: dict, relative_path: str) -> tuple[str | None, str]:
    for field in BLOCK_FIELDS:
        value = row.get(field)
        if value not in (None, ""):
            return str(value), f"explicit:{field}"
    stem = Path(relative_path).stem
    token = stem.replace("-", "_").split("_", 1)[0].strip()
    if token:
        return token, "fallback:filename_token_v1"
    return None, "unavailable"


def _manifest_records(manifest_path: Path, allowed_roots: list[Path]) -> tuple[dict, list[dict]]:
    manifest_path = manifest_path.resolve(strict=True)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must be a JSON object: {manifest_path}")

    root_value = payload.get("root")
    root = None
    if root_value is not None:
        root_candidate = Path(root_value)
        if not root_candidate.is_absolute():
            raise ValueError(f"manifest root must be absolute: {manifest_path}")
        root = root_candidate.resolve(strict=True)
        if not root.is_dir() or not _is_within(root, allowed_roots):
            raise ValueError(f"manifest root outside allowed roots: {root}")

    raw_rows = payload.get("files", payload.get("paths"))
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError(f"manifest files/paths must be a non-empty list: {manifest_path}")

    records = []
    for index, raw in enumerate(raw_rows):
        if isinstance(raw, str):
            row, path_value = {}, raw
        elif isinstance(raw, dict) and isinstance(raw.get("path"), str):
            row, path_value = raw, raw["path"]
        else:
            raise ValueError(f"malformed manifest row {index}: {manifest_path}")
        candidate = Path(path_value)
        if not candidate.is_absolute():
            if root is None:
                raise ValueError(f"relative path without manifest root at row {index}")
            candidate = root / candidate
        resolved = candidate.resolve(strict=True)
        if not resolved.is_file() or not _is_within(resolved, allowed_roots):
            raise ValueError(f"image outside allowed roots or not a file: {resolved}")
        relative = (
            resolved.relative_to(root).as_posix()
            if root is not None and _is_within(resolved, (root,))
            else resolved.name
        )
        block, block_method = _block_id(row, relative)
        records.append(
            {
                "index": index,
                "path": resolved,
                "relative_path": relative,
                "block_id": block,
                "block_method": block_method,
            }
        )
    return {
        "path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "root": str(root) if root is not None else None,
        "count": len(records),
    }, records


def _dhash64_bytes(payload: bytes) -> int:
    with Image.open(io.BytesIO(payload)) as image:
        pixels = list(
            ImageOps.exif_transpose(image)
            .convert("L")
            .resize((9, 8), Image.Resampling.LANCZOS)
            .getdata()
        )
    value = 0
    for row in range(8):
        base = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[base + column] > pixels[base + column + 1])
    return value


def _add_image_fingerprints(records: list[dict]) -> None:
    for record in records:
        path = record["path"]
        # The source images are large.  Compute the byte hash and perceptual
        # hash from one immutable in-memory read instead of streaming each file
        # once for SHA-256 and opening it from disk a second time for dHash.
        payload = path.read_bytes()
        record["bytes"] = len(payload)
        record["sha256"] = hashlib.sha256(payload).hexdigest()
        record["dhash64"] = _dhash64_bytes(payload)


def _limited_cross_examples(
    left: list[dict], right: list[dict], limit: int, *, distance: int | None = None
) -> list[dict]:
    examples = []
    for train in left:
        for validation in right:
            if train["sha256"] == validation["sha256"] and distance is not None:
                continue
            row = {
                "train": train["relative_path"],
                "validation": validation["relative_path"],
            }
            if distance is None:
                row["sha256"] = train["sha256"]
                row["bytes"] = train["bytes"]
            else:
                row["hamming_distance"] = distance
            examples.append(row)
            if len(examples) >= limit:
                return examples
    return examples


def _exact_report(train: list[dict], validation: list[dict], limit: int) -> dict:
    train_by_sha, validation_by_sha = defaultdict(list), defaultdict(list)
    for row in train:
        train_by_sha[row["sha256"]].append(row)
    for row in validation:
        validation_by_sha[row["sha256"]].append(row)
    shared = sorted(set(train_by_sha) & set(validation_by_sha))
    examples = []
    pair_count = 0
    for digest in shared:
        left = sorted(train_by_sha[digest], key=lambda row: row["relative_path"])
        right = sorted(validation_by_sha[digest], key=lambda row: row["relative_path"])
        pair_count += len(left) * len(right)
        if len(examples) < limit:
            examples.extend(_limited_cross_examples(left, right, limit - len(examples)))

    validation_paths = {os.path.normcase(str(row["path"])) for row in validation}
    same_paths = sorted(
        row["relative_path"]
        for row in train
        if os.path.normcase(str(row["path"])) in validation_paths
    )
    return {
        "shared_content_hashes": len(shared),
        "content_pair_count": pair_count,
        "same_resolved_path_count": len(same_paths),
        "same_resolved_path_examples": same_paths[:limit],
        "examples": examples,
    }


def _segments(threshold: int) -> list[tuple[int, int]]:
    count = threshold + 1
    base, remainder = divmod(64, count)
    widths = [base + (1 if index < remainder else 0) for index in range(count)]
    offset, result = 0, []
    for width in widths:
        result.append((offset, width))
        offset += width
    return result


def _near_report(train: list[dict], validation: list[dict], threshold: int, limit: int) -> dict:
    if not 0 <= threshold < 64:
        raise ValueError("near threshold must be between 0 and 63")
    train_groups, validation_groups = defaultdict(list), defaultdict(list)
    for row in train:
        train_groups[row["dhash64"]].append(row)
    for row in validation:
        validation_groups[row["dhash64"]].append(row)

    index: dict[tuple[int, int], set[int]] = defaultdict(set)
    segments = _segments(threshold)
    for fingerprint in validation_groups:
        for segment_index, (offset, width) in enumerate(segments):
            index[(segment_index, (fingerprint >> offset) & ((1 << width) - 1))].add(fingerprint)

    candidate_hash_pairs = set()
    for train_hash in train_groups:
        possible = set()
        for segment_index, (offset, width) in enumerate(segments):
            possible.update(index.get((segment_index, (train_hash >> offset) & ((1 << width) - 1)), ()))
        for validation_hash in possible:
            distance = (train_hash ^ validation_hash).bit_count()
            if distance <= threshold:
                candidate_hash_pairs.add((train_hash, validation_hash, distance))

    examples, histogram, pair_count = [], Counter(), 0
    for train_hash, validation_hash, distance in sorted(candidate_hash_pairs, key=lambda item: (item[2], item[0], item[1])):
        left = sorted(train_groups[train_hash], key=lambda row: row["relative_path"])
        right = sorted(validation_groups[validation_hash], key=lambda row: row["relative_path"])
        same_sha_pairs = sum(
            left_count * right_count
            for digest, left_count in Counter(row["sha256"] for row in left).items()
            if (right_count := Counter(row["sha256"] for row in right).get(digest, 0))
        )
        count = len(left) * len(right) - same_sha_pairs
        if count <= 0:
            continue
        pair_count += count
        histogram[str(distance)] += count
        if len(examples) < limit:
            examples.extend(
                _limited_cross_examples(left, right, limit - len(examples), distance=distance)
            )
    return {
        "method": DHASH_METHOD,
        "threshold": threshold,
        "candidate_pair_count": pair_count,
        "distance_histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
        "examples": examples,
        "interpretation": "review candidate only; perceptual similarity is not automatic duplicate proof",
    }


def _provenance_report(train: list[dict], validation: list[dict], limit: int) -> dict:
    method_counts = Counter(row["block_method"] for row in train + validation)
    train_blocks = {row["block_id"] for row in train if row["block_id"] is not None}
    validation_blocks = {row["block_id"] for row in validation if row["block_id"] is not None}
    overlap = sorted(train_blocks & validation_blocks)
    return {
        "method_counts": dict(sorted(method_counts.items())),
        "train_unique_blocks": len(train_blocks),
        "validation_unique_blocks": len(validation_blocks),
        "cross_split_block_overlap_count": len(overlap),
        "cross_split_block_overlap_examples": overlap[:limit],
        "fallback_limitation": "filename_token_v1 is a heuristic block key, not proven source/lot lineage",
    }


def audit(
    train_manifest: str | Path,
    validation_manifest: str | Path,
    allowed_roots: Iterable[str | Path],
    *,
    near_threshold: int = 5,
    max_examples: int = 50,
) -> dict:
    roots = _resolve_roots(allowed_roots)
    train_meta, train = _manifest_records(Path(train_manifest), roots)
    validation_meta, validation = _manifest_records(Path(validation_manifest), roots)
    _add_image_fingerprints(train)
    _add_image_fingerprints(validation)
    train_meta["image_content_sha256"] = sha256_obj(
        [(row["relative_path"], row["sha256"]) for row in train]
    )
    validation_meta["image_content_sha256"] = sha256_obj(
        [(row["relative_path"], row["sha256"]) for row in validation]
    )
    exact = _exact_report(train, validation, max_examples)
    near = _near_report(train, validation, near_threshold, max_examples)
    provenance = _provenance_report(train, validation, max_examples)
    overlap_found = exact["content_pair_count"] > 0 or exact["same_resolved_path_count"] > 0
    review_required = near["candidate_pair_count"] > 0
    status = "overlap_found" if overlap_found else ("review_required" if review_required else "clean")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": {"train": train_meta, "validation": validation_meta},
        "allowlist_roots": [str(root) for root in roots],
        "validation": {"all_paths_exist": True, "all_paths_within_allowlist": True},
        "exact": exact,
        "near": near,
        "provenance": provenance,
        "status": status,
        "review_required": review_required,
    }


def atomic_write_json(path: str | Path, value: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--allowed-root", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--near-threshold", type=int, default=5)
    parser.add_argument("--max-examples", type=int, default=50)
    args = parser.parse_args(argv)
    try:
        result = audit(
            args.train_manifest,
            args.validation_manifest,
            args.allowed_root,
            near_threshold=args.near_threshold,
            max_examples=args.max_examples,
        )
        atomic_write_json(args.out, result)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"manifest overlap audit failed: {exc}\n")
    return 1 if result["status"] == "overlap_found" else 0


if __name__ == "__main__":
    raise SystemExit(main())
