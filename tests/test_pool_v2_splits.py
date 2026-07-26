"""Unit tests for data/pools/v2/ (built by scripts/make_v2_pool_splits.py):
class-disjoint verification + file-level leakage guard on the actual
generated strict_novel splits (team-lead directive 260726).

These read the JSON manifests already written under data/pools/v2/ — no
image bytes are touched (path-level dedup only, no SHA pass), so they don't
need the E: drive mounted. If a dataset hasn't been built yet the test for
it is skipped rather than failing the whole run.
"""
from __future__ import annotations

import json

import pytest

from scripts import audit_pool_leakage as apl

POOLS_V2 = apl.REPO / "data" / "pools" / "v2"

TRIPLET_DATASETS = ["unknown", "mixedwm38", "hf_dtd", "hf_flowers102", "hf_resisc45"]


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _labels(manifest: dict) -> set[str]:
    return {f["label"] for f in manifest["files"] if f.get("label")}


@pytest.mark.parametrize("dataset", TRIPLET_DATASETS)
def test_train_val_test_classes_are_disjoint(dataset):
    d = POOLS_V2 / dataset
    if not d.is_dir():
        pytest.skip(f"data/pools/v2/{dataset} not built yet (run scripts/make_v2_pool_splits.py)")
    train = _load(d / "strict_novel_train.json")
    val = _load(d / "strict_novel_val.json")
    test = _load(d / "strict_novel_test.json")

    val_labels = _labels(val)
    test_labels = _labels(test)
    train_labels = _labels(train)

    # unknown keeps Normal in the train bucket only — every other class must
    # be assigned to exactly one of {train, val, test}.
    assert train_labels.isdisjoint(val_labels), train_labels & val_labels
    assert train_labels.isdisjoint(test_labels), train_labels & test_labels
    assert val_labels.isdisjoint(test_labels), val_labels & test_labels


@pytest.mark.parametrize("dataset", TRIPLET_DATASETS)
def test_no_file_leakage_between_train_val_test(dataset):
    """label leakage 차단: even if class sets are disjoint, no individual
    file may be referenced by more than one split (path-level; SHA-level
    is covered separately by scripts/audit_pool_leakage.py's own tests and
    was already run for every dataset during split generation)."""
    d = POOLS_V2 / dataset
    if not d.is_dir():
        pytest.skip(f"data/pools/v2/{dataset} not built yet (run scripts/make_v2_pool_splits.py)")
    manifests = [d / "strict_novel_train.json", d / "strict_novel_val.json", d / "strict_novel_test.json"]
    result = apl.audit(manifests, compute_sha=False)

    assert result["path_overlap_pairs"] == {}, result["path_overlap_pairs"]
    assert result["allowlist_violations"] == {}, result["allowlist_violations"]


def test_severstal_each_fold_train_test_disjoint():
    d = POOLS_V2 / "severstal"
    if not d.is_dir():
        pytest.skip("data/pools/v2/severstal not built yet (run scripts/make_v2_pool_splits.py)")
    fold_manifests = sorted(d.glob("fold*_train.json"))
    if not fold_manifests:
        pytest.skip("no severstal fold manifests found")

    for train_path in fold_manifests:
        fold_tag = train_path.name[: -len("_train.json")]
        test_path = d / f"{fold_tag}_novel_test.json"
        assert test_path.exists(), f"missing paired novel_test manifest for {fold_tag}"

        train = _load(train_path)
        test = _load(test_path)
        # held-out classes must not appear anywhere in this fold's train split
        assert _labels(train).isdisjoint(_labels(test)), (fold_tag, _labels(train) & _labels(test))

        result = apl.audit([train_path, test_path], compute_sha=False)
        assert result["path_overlap_pairs"] == {}, (fold_tag, result["path_overlap_pairs"])


def test_severstal_fold_count_is_leave_two_out_over_four_classes():
    d = POOLS_V2 / "severstal"
    if not d.is_dir():
        pytest.skip("data/pools/v2/severstal not built yet (run scripts/make_v2_pool_splits.py)")
    fold_manifests = sorted(d.glob("fold*_train.json"))
    if not fold_manifests:
        pytest.skip("no severstal fold manifests found")
    # C(4, 2) == 6 folds
    assert len(fold_manifests) == 6


def test_unknown_multi_is_marked_stress_test_and_excluded_from_training_use():
    path = POOLS_V2 / "unknown_multi" / "stress_test_all.json"
    if not path.exists():
        pytest.skip("data/pools/v2/unknown_multi not built yet")
    manifest = _load(path)
    assert manifest["track"] == "stress_test"
    assert manifest["n_files"] > 0
