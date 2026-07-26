#!/usr/bin/env python3
"""Audit data/pools/*.json manifests for cross-pool data leakage.

team-lead directive (260726, codex 캠페인 계획 2절): several existing pools
turned out to share images (e.g. unknown_eval100 vs unknown_train_defectaware
1,000 files) — any performance number computed on top of those pools is
suspect until this is fixed. This tool is the reusable detector.

Checks, given N manifest paths:
  (a) SHA-256 content duplicates — catches the same physical file reachable
      under two different relative paths (e.g. duplicated/re-exported
      images). Path comparison alone is NOT sufficient for this case, hence
      SHA-256 is mandatory (team-lead directive, explicit).
  (b) resolved absolute-path duplicates — cheap, catches the common case
      where two manifests reference literally the same file (same master
      root + same relative path). Computed first since it needs no I/O.
  (c) allowlist violations — every resolved file must live under one of the
      allowed read-only data roots (see ALLOWLIST_ROOTS). Anything outside
      fails preflight.
  (d) label/class overlap across manifest pairs — informational only. A
      shared class name is not itself leakage (e.g. a class can legitimately
      appear in both a train pool and a class-disjoint different pool's
      "known" bucket) — combine with (a)/(b) for the real leakage signal.

Usage:
    python scripts/audit_pool_leakage.py --manifests data/pools/a.json data/pools/b.json ...
    python scripts/audit_pool_leakage.py --manifests data/pools/*.json --json-out out.json

SHA-256 hashing is I/O heavy; results are cached in
scripts/.cache/pool_sha_cache.json keyed by (resolved path, size, mtime) so
repeat audits over shared master files (most pools here derive from the same
E:/data/images/<dataset> root) are cheap. Hashing uses a small thread pool
(--workers, default 2 — CPU is already saturated by concurrent training,
per team-lead directive: "워커 2 이하").
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts._common import resolve_path  # noqa: E402

DEFAULT_CACHE = REPO / "scripts" / ".cache" / "pool_sha_cache.json"

# Read-only allowlist — every file referenced by any audited manifest must
# resolve under one of these roots. team-lead directive (260726): images may
# only be *read* from here, never copied/moved/linked/deleted.
ALLOWLIST_ROOTS = [
    Path("E:/data/images/unknown"),
    Path("E:/data/images/unknown_multi"),
    Path("E:/data/images/mixedwm38"),
    Path("E:/data/images/severstal"),
    Path("E:/data/images/hf_dtd"),
    Path("E:/data/images/hf_flowers102"),
    Path("E:/data/images/hf_resisc45"),
]


def _norm(p: Path) -> str:
    """Case-insensitive, forward-slash, resolved string key (Windows-safe path identity)."""
    return str(p).replace("\\", "/").lower()


def is_allowed(path: Path) -> bool:
    key = _norm(path)
    return any(key == _norm(root) or key.startswith(_norm(root) + "/") for root in ALLOWLIST_ROOTS)


def load_manifest_entries(manifest_path: Path) -> list[dict]:
    """Return [{"path": <resolved Path>, "label": str|None}] for one manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest.get("root")
    root_dir = resolve_path(root) if root else None
    entries = []
    for item in manifest["files"]:
        rel = Path(item["path"])
        resolved = (root_dir / rel) if (root_dir is not None and not rel.is_absolute()) else rel
        entries.append({"path": resolved, "label": item.get("label")})
    return entries


class ShaCache:
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self.data: dict[str, list] = {}
        if cache_path.exists():
            try:
                self.data = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.data = {}
        self._dirty = False

    def get(self, path: Path) -> str | None:
        key = _norm(path)
        st = path.stat()
        entry = self.data.get(key)
        if entry and entry[0] == st.st_size and entry[1] == st.st_mtime:
            return entry[2]
        return None

    def put(self, path: Path, sha: str) -> None:
        key = _norm(path)
        st = path.stat()
        self.data[key] = [st.st_size, st.st_mtime, sha]
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.data), encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_all(paths: list[Path], cache: ShaCache, workers: int) -> dict[str, str]:
    """path(_norm key) -> sha256, using the cache and a small thread pool for misses."""
    result: dict[str, str] = {}
    missing: list[Path] = []
    for p in paths:
        if not p.exists():
            continue
        cached = cache.get(p)
        if cached is not None:
            result[_norm(p)] = cached
        else:
            missing.append(p)

    def _work(p: Path):
        return p, sha256_file(p)

    if missing:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for p, sha in ex.map(_work, missing):
                result[_norm(p)] = sha
                cache.put(p, sha)
    return result


def audit(manifest_paths: list[Path], cache_path: Path = DEFAULT_CACHE, workers: int = 2,
          compute_sha: bool = True) -> dict:
    names = [p.stem for p in manifest_paths]
    if len(set(names)) != len(names):
        raise ValueError(f"manifest stems must be unique for reporting: {names}")

    per_manifest: dict[str, list[dict]] = {}
    allowlist_violations: dict[str, list[str]] = {}
    missing_files: dict[str, list[str]] = {}
    for name, mp in zip(names, manifest_paths):
        entries = load_manifest_entries(mp)
        per_manifest[name] = entries
        violations = [str(e["path"]) for e in entries if not is_allowed(e["path"])]
        if violations:
            allowlist_violations[name] = violations
        missing = [str(e["path"]) for e in entries if not e["path"].exists()]
        if missing:
            missing_files[name] = missing

    # (b) path-level overlap — no I/O required
    path_sets = {name: {_norm(e["path"]) for e in entries} for name, entries in per_manifest.items()}
    path_overlap_pairs = {}
    for a, b in combinations(names, 2):
        common = path_sets[a] & path_sets[b]
        if common:
            path_overlap_pairs[f"{a} ^ {b}"] = {"n": len(common), "sample": sorted(common)[:10]}

    # (d) label/class overlap — informational
    label_sets = {name: {e["label"] for e in entries if e["label"]} for name, entries in per_manifest.items()}
    class_overlap_pairs = {}
    for a, b in combinations(names, 2):
        common = label_sets[a] & label_sets[b]
        if common:
            class_overlap_pairs[f"{a} ^ {b}"] = sorted(common)

    result = {
        "manifests": {name: {"path": str(mp), "n_files": len(per_manifest[name])}
                      for name, mp in zip(names, manifest_paths)},
        "allowlist_violations": allowlist_violations,
        "missing_files": missing_files,
        "path_overlap_pairs": path_overlap_pairs,
        "class_overlap_pairs": class_overlap_pairs,
    }

    if not compute_sha:
        return result

    cache = ShaCache(cache_path)
    all_paths = [e["path"] for entries in per_manifest.values() for e in entries]
    # unique by normalized key to avoid re-hashing the same physical file twice
    unique_paths = {}
    for p in all_paths:
        unique_paths.setdefault(_norm(p), p)
    sha_by_path = hash_all(list(unique_paths.values()), cache, workers)
    cache.save()

    sha_sets = {}
    for name, entries in per_manifest.items():
        shas = set()
        for e in entries:
            sha = sha_by_path.get(_norm(e["path"]))
            if sha:
                shas.add(sha)
        sha_sets[name] = shas

    sha_overlap_pairs = {}
    for a, b in combinations(names, 2):
        common = sha_sets[a] & sha_sets[b]
        if common:
            sha_overlap_pairs[f"{a} ^ {b}"] = {"n": len(common), "sample": sorted(common)[:5]}

    # within-manifest content duplicates (same file content appearing twice
    # under different relative paths inside one manifest)
    within_duplicates = {}
    for name, entries in per_manifest.items():
        by_sha: dict[str, list[str]] = defaultdict(list)
        for e in entries:
            sha = sha_by_path.get(_norm(e["path"]))
            if sha:
                by_sha[sha].append(str(e["path"]))
        dupes = {sha: paths for sha, paths in by_sha.items() if len(paths) > 1}
        if dupes:
            within_duplicates[name] = {sha: paths for sha, paths in list(dupes.items())[:20]}

    result["sha_overlap_pairs"] = sha_overlap_pairs
    result["within_manifest_sha_duplicates"] = within_duplicates
    result["n_files_hashed"] = len(sha_by_path)
    return result


def print_report(result: dict) -> None:
    print("=== manifests ===")
    for name, info in result["manifests"].items():
        print(f"  {name}: n_files={info['n_files']}  ({info['path']})")

    if result["allowlist_violations"]:
        print("\n=== ALLOWLIST VIOLATIONS (FAIL) ===")
        for name, viol in result["allowlist_violations"].items():
            print(f"  {name}: {len(viol)} file(s) outside allowlist, e.g. {viol[:3]}")
    else:
        print("\n=== allowlist: OK (all files under allowed roots) ===")

    if result["missing_files"]:
        print("\n=== MISSING FILES (manifest references nonexistent path) ===")
        for name, miss in result["missing_files"].items():
            print(f"  {name}: {len(miss)} missing, e.g. {miss[:3]}")

    print("\n=== path-level overlap (same resolved path in >1 manifest) ===")
    if result["path_overlap_pairs"]:
        for pair, info in result["path_overlap_pairs"].items():
            print(f"  {pair}: {info['n']} shared file(s)")
    else:
        print("  none")

    if "sha_overlap_pairs" in result:
        print("\n=== SHA-256 content overlap (same bytes in >1 manifest, any path) ===")
        if result["sha_overlap_pairs"]:
            for pair, info in result["sha_overlap_pairs"].items():
                print(f"  {pair}: {info['n']} shared file(s) by content")
        else:
            print("  none")
        print(f"  ({result['n_files_hashed']} unique physical files hashed)")

    if result.get("within_manifest_sha_duplicates"):
        print("\n=== within-manifest content duplicates (same file listed twice under different paths) ===")
        for name, dupes in result["within_manifest_sha_duplicates"].items():
            print(f"  {name}: {len(dupes)} duplicate content group(s)")

    print("\n=== class/label overlap across manifest pairs (informational — not leakage by itself) ===")
    if result["class_overlap_pairs"]:
        for pair, classes in result["class_overlap_pairs"].items():
            print(f"  {pair}: {len(classes)} shared class(es)")
    else:
        print("  none")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifests", nargs="+", required=True, help="manifest .json paths (globs allowed)")
    ap.add_argument("--cache", default=str(DEFAULT_CACHE), help="SHA-256 cache json path")
    ap.add_argument("--workers", type=int, default=2, help="hashing thread-pool size (default 2, CPU-safe)")
    ap.add_argument("--no-sha", action="store_true", help="skip SHA-256 pass (path-overlap + allowlist only, fast)")
    ap.add_argument("--json-out", default=None, help="write full result JSON here")
    ap.add_argument("--fail-on-leak", action="store_true",
                     help="exit 1 if any path/SHA overlap or allowlist violation is found")
    args = ap.parse_args()

    manifest_paths: list[Path] = []
    for pattern in args.manifests:
        matched = sorted(glob.glob(pattern))
        if matched:
            manifest_paths.extend(Path(m) for m in matched)
        else:
            manifest_paths.append(resolve_path(pattern))
    manifest_paths = sorted(set(manifest_paths), key=lambda p: p.stem)
    for mp in manifest_paths:
        if not mp.exists():
            raise SystemExit(f"manifest not found: {mp}")

    result = audit(manifest_paths, cache_path=resolve_path(args.cache), workers=args.workers,
                    compute_sha=not args.no_sha)
    print_report(result)

    if args.json_out:
        out_path = resolve_path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[OUT] {out_path}")

    if args.fail_on_leak:
        leaked = bool(result["allowlist_violations"] or result["path_overlap_pairs"]
                      or result.get("sha_overlap_pairs") or result.get("within_manifest_sha_duplicates"))
        if leaked:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
