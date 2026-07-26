"""Unit tests for scripts/audit_pool_leakage.py (team-lead directive 260726:
allowlist blocking + SHA-256 dedup are the two mechanical guarantees the
leakage audit tool must provide)."""
from __future__ import annotations

import json
from pathlib import Path

from scripts import audit_pool_leakage as apl


def _write_manifest(path: Path, root, files: list[dict]) -> Path:
    path.write_text(json.dumps({"root": root, "n_files": len(files), "files": files}), encoding="utf-8")
    return path


def test_allowlist_blocks_paths_outside_roots(tmp_path):
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"not a real image, just bytes")
    manifest = _write_manifest(
        tmp_path / "bad_manifest.json", None,
        [{"path": str(outside), "label": "X"}],
    )

    result = apl.audit([manifest], compute_sha=False)

    assert "bad_manifest" in result["allowlist_violations"]
    assert str(outside) in result["allowlist_violations"]["bad_manifest"]


def test_allowlist_accepts_real_pool_roots():
    # every file referenced by the pre-existing production pools must resolve
    # under an allowlisted root — this is the negative-control counterpart to
    # the block test above.
    for root in apl.ALLOWLIST_ROOTS:
        assert apl.is_allowed(root / "some_class" / "some_file.png")
    assert not apl.is_allowed(Path("D:/project/unknown-contrastive/data/pools/whatever.json"))


def test_path_level_dedup_across_manifests(tmp_path, monkeypatch):
    monkeypatch.setattr(apl, "ALLOWLIST_ROOTS", [tmp_path])
    shared = tmp_path / "class_a" / "img1.png"
    shared.parent.mkdir(parents=True, exist_ok=True)
    shared.write_bytes(b"content")

    manifest_a = _write_manifest(tmp_path / "a.json", str(tmp_path), [{"path": "class_a/img1.png", "label": "A"}])
    manifest_b = _write_manifest(tmp_path / "b.json", str(tmp_path), [{"path": "class_a/img1.png", "label": "A"}])

    result = apl.audit([manifest_a, manifest_b], compute_sha=False)

    assert result["path_overlap_pairs"]["a ^ b"]["n"] == 1


def test_sha_dedup_catches_same_content_under_different_paths(tmp_path, monkeypatch):
    """The exact case path-comparison alone would miss: identical bytes
    reachable via two different relative paths (e.g. a re-exported/duplicated
    image) must still be caught — this is why SHA-256 is mandatory."""
    monkeypatch.setattr(apl, "ALLOWLIST_ROOTS", [tmp_path])
    (tmp_path / "left").mkdir()
    (tmp_path / "right").mkdir()
    payload = b"identical wafer image bytes"
    (tmp_path / "left" / "wafer_0001.png").write_bytes(payload)
    (tmp_path / "right" / "wafer_0001_copy.png").write_bytes(payload)

    manifest_left = _write_manifest(tmp_path / "left_pool.json", str(tmp_path),
                                     [{"path": "left/wafer_0001.png", "label": "A"}])
    manifest_right = _write_manifest(tmp_path / "right_pool.json", str(tmp_path),
                                      [{"path": "right/wafer_0001_copy.png", "label": "A"}])

    result = apl.audit([manifest_left, manifest_right], cache_path=tmp_path / "cache.json", workers=2)

    # different resolved paths -> no path-level overlap ...
    assert result["path_overlap_pairs"] == {}
    # ... but SHA-256 content is identical -> must be flagged
    assert result["sha_overlap_pairs"]["left_pool ^ right_pool"]["n"] == 1


def test_within_manifest_sha_duplicate_detection(tmp_path, monkeypatch):
    monkeypatch.setattr(apl, "ALLOWLIST_ROOTS", [tmp_path])
    payload = b"same bytes twice in one manifest"
    (tmp_path / "img_a.png").write_bytes(payload)
    (tmp_path / "img_b.png").write_bytes(payload)

    manifest = _write_manifest(
        tmp_path / "dupe_pool.json", str(tmp_path),
        [{"path": "img_a.png", "label": "A"}, {"path": "img_b.png", "label": "A"}],
    )

    result = apl.audit([manifest], cache_path=tmp_path / "cache2.json", workers=2)

    assert "dupe_pool" in result["within_manifest_sha_duplicates"]


def test_sha_cache_reuses_hash_for_unchanged_file(tmp_path, monkeypatch):
    monkeypatch.setattr(apl, "ALLOWLIST_ROOTS", [tmp_path])
    f = tmp_path / "cached.png"
    f.write_bytes(b"cache me")
    cache_path = tmp_path / "cache.json"
    cache = apl.ShaCache(cache_path)
    assert cache.get(f) is None
    sha = apl.sha256_file(f)
    cache.put(f, sha)
    cache.save()

    reloaded = apl.ShaCache(cache_path)
    assert reloaded.get(f) == sha
