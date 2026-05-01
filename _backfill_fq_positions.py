#!/usr/bin/env python3
"""Backfill synthetic FTN/QTN vectors into existing unknown positions JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, Set

from _fq_metadata import DEFAULT_FQ_ITEM_COUNT, add_synthetic_fq_to_json


DEFAULT_POSITIONS_ROOT = Path("D:/project/data/positions/unknown")


def iter_json_files(root: Path, classes: Optional[Set[str]] = None) -> Iterable[Path]:
    if classes:
        for cls in sorted(classes):
            cls_dir = root / cls
            if cls_dir.exists():
                yield from sorted(cls_dir.glob("*.json"))
        return
    yield from sorted(root.rglob("*.json"))


def needs_top_update(obj: dict, item_count: int) -> bool:
    return (
        not obj.get("partid")
        or not obj.get("pgm")
        or len(obj.get("ftn_keys") or []) != item_count
        or len(obj.get("qtn_keys") or []) != item_count
    )


def chip_has_fq(obj: dict, item_count: int) -> bool:
    chips = obj.get("chips") or []
    if not chips:
        return True
    first = chips[0]
    return (
        isinstance(first.get("f"), list)
        and isinstance(first.get("q"), list)
        and len(first["f"]) == item_count
        and len(first["q"]) == item_count
    )


def process_file(path: Path, item_count: int, overwrite_fq: bool, dry_run: bool) -> str:
    class_label = path.parent.name
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    top_update = needs_top_update(obj, item_count)
    fq_update = overwrite_fq or not chip_has_fq(obj, item_count)
    if not top_update and not fq_update:
        return "skipped"

    add_synthetic_fq_to_json(
        obj,
        class_label=class_label,
        seed_text=path.stem,
        item_count=item_count,
        overwrite_fq=overwrite_fq or fq_update,
    )

    if not dry_run:
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    return "updated"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--positions-root", type=Path, default=DEFAULT_POSITIONS_ROOT)
    p.add_argument("--item-count", type=int, default=DEFAULT_FQ_ITEM_COUNT)
    p.add_argument("--class", dest="classes", action="append",
                   help="class folder name to backfill; may be repeated")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite-fq", action="store_true",
                   help="replace existing f/q arrays instead of only filling missing ones")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    files = list(iter_json_files(args.positions_root, set(args.classes) if args.classes else None))
    if args.limit:
        files = files[:args.limit]

    print(f"positions_root={args.positions_root}")
    print(f"files={len(files)} item_count={args.item_count} overwrite_fq={args.overwrite_fq} dry_run={args.dry_run}")

    updated = 0
    skipped = 0
    failed = 0
    for i, path in enumerate(files, 1):
        try:
            status = process_file(path, args.item_count, args.overwrite_fq, args.dry_run)
            if status == "updated":
                updated += 1
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            print(f"[FAIL] {path}: {exc}")
        if i == 1 or i % 100 == 0 or i == len(files):
            print(f"[{i}/{len(files)}] updated={updated} skipped={skipped} failed={failed}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
