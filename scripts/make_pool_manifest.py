#!/usr/bin/env python3
"""Generate a pool manifest (data/pools/<name>.json) from an existing derived
pool directory, resolved against a single master dataset directory.

사용자 지시(260726): "데이터셋은 1개로 운영하고 코드에서 선택하는 걸로 바꿔라."
파생 pool(예: unknown_eval100)의 파일 목록을 마스터(E:/data/images/unknown)
기준 상대경로로 기록 — 물리 복사본 없이 --pool <manifest.json> 으로 동일 결과 재현
(scripts/_common.py::resolve_pool 이 소비).

마이그레이션 시 사용:
  python scripts/make_pool_manifest.py --source <기존_파생_pool> \
      --master E:/data/images/unknown --out data/pools/<pool_name>.json

전체 클래스 선택:
  python scripts/make_pool_manifest.py --classes Normal,RingDots \
      --master E:/data/images/unknown --out data/pools/<pool_name>.json

기본은 파일 크기로 저비용 검증(모든 파일, stat()만 — CPU/IO 최소). --verify-md5 로
content-hash 전수검증(느림, 큰 pool 엔 비권장).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts._common import resolve_path  # noqa: E402

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", help="existing derived pool directory (class-subdir layout)")
    ap.add_argument("--master", required=True, help="single master dataset directory (same class-subdir layout)")
    ap.add_argument("--out", required=True, help="output manifest .json path")
    ap.add_argument(
        "--classes",
        help="comma-separated master class names; selects whole classes without a derived directory",
    )
    ap.add_argument("--verify-md5", action="store_true",
                     help="full content-hash verify every file against master (slow/CPU-heavy). "
                          "Default off: verifies file size only (cheap, stat()-only).")
    a = ap.parse_args()

    master = resolve_path(a.master).resolve(strict=True)
    out_path = resolve_path(a.out)

    selected_classes = [name.strip() for name in (a.classes or "").split(",") if name.strip()]
    if selected_classes:
        if a.source:
            raise ValueError("--source and --classes are mutually exclusive")
        entries = []
        missing_classes = []
        for label in selected_classes:
            class_dir = master / label
            if not class_dir.is_dir():
                missing_classes.append(label)
                continue
            for path in sorted(
                p for p in class_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ):
                entries.append({"path": path.relative_to(master).as_posix(), "label": label})
        if missing_classes:
            raise ValueError(f"class(es) missing under --master: {', '.join(missing_classes)}")
        if not entries:
            raise ValueError("no image files selected by --classes")
        manifest = {
            "root": master.relative_to(REPO).as_posix() if master.is_relative_to(REPO) else str(master),
            "source_pool": None,
            "selection": {"classes": selected_classes},
            "n_files": len(entries),
            "verify": "master-direct",
            "files": entries,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[make_pool_manifest] {len(entries)} files -> {out_path}")
        print(f"[OUT] {out_path}")
        return

    if not a.source:
        raise ValueError("one of --source or --classes is required")
    source = resolve_path(a.source).resolve(strict=True)
    files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise ValueError(f"no image files found under --source: {source}")

    entries = []
    mismatches = []
    for f in files:
        rel = f.relative_to(source)
        mpath = master / rel
        if not mpath.exists():
            mismatches.append(f"missing under master: {rel}")
            continue
        ok = md5_file(f) == md5_file(mpath) if a.verify_md5 else f.stat().st_size == mpath.stat().st_size
        if not ok:
            mismatches.append(f"content differs from master: {rel}")
            continue
        label = rel.parts[0] if len(rel.parts) > 1 else mpath.parent.name
        entries.append({"path": rel.as_posix(), "label": label})

    if mismatches:
        preview = "\n  ".join(mismatches[:20])
        raise ValueError(
            f"{len(mismatches)} file(s) in --source do not match --master by "
            f"relative path (source is not a pure master subset):\n  {preview}"
        )

    manifest = {
        "root": master.relative_to(REPO).as_posix() if master.is_relative_to(REPO) else str(master),
        "source_pool": str(source),
        "n_files": len(entries),
        "verify": "md5" if a.verify_md5 else "size",
        "files": entries,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[make_pool_manifest] {len(entries)} files -> {out_path}")
    print(f"[OUT] {out_path}")


if __name__ == "__main__":
    main()
