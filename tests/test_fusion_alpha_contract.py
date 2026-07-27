import numpy as np
import pytest

from scripts import fusion_alpha_common as common
from scripts import run_fusion_alpha_unlabeled as worker1
from scripts.fusion_alpha_common import (DIALS, FROZEN_PRE_NOISE_CAPS, INTERIOR_ALPHAS,
    atomic_json_new, fuse, gate_all, gate_ratio, q75, select)


def metric(k, noise, passed=True):
    return {"k": k, "pre_reassign_noise": noise, "stability": .8 if passed else .7,
            "coherence": .9, "over_merge": 0}


def candidates():
    return {alpha: {dial: metric(10, 1) for dial in DIALS} for alpha in INTERIOR_ALPHAS}


def test_weighted_cosine_geometry_and_endpoint_identity():
    frozen = np.array([[3., 4.], [0., 2.]], dtype=np.float32)
    projection = np.array([[5., 0.], [2., 2.]], dtype=np.float32)
    assert fuse(frozen, projection, 0.0) is frozen
    assert fuse(frozen, projection, 1.0) is projection
    actual = fuse(frozen, projection, .2)
    f = frozen / np.linalg.norm(frozen, axis=1, keepdims=True)
    p = projection / np.linalg.norm(projection, axis=1, keepdims=True)
    expected = np.concatenate((np.sqrt(.8) * f, np.sqrt(.2) * p), axis=1)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    assert np.allclose(actual, expected)


def test_q75_caps_and_lexicographic_interior_only_rule():
    values = [1, 4, 9, 16]
    assert q75(values) == 10.75
    rows = candidates()
    rows[.02][.005]["k"] = 20
    rows[.05][.005]["k"] = 30
    rows[.05][.01]["pre_reassign_noise"] = 2
    choice = select(rows)
    assert choice["q75"][.005] == q75([20, 30, 10, 10, 10, 10])
    derivation = choice["q75_derivations"][.005]
    assert derivation == {"sorted_population": [10, 10, 10, 10, 20, 30], "n": 6, "h": 3.75, "j": 3, "g": .75, "q75": 17.5}
    assert choice["base_pass_membership"][.005] == list(INTERIOR_ALPHAS)
    assert choice["selected_alpha"] == .02
    rows[.02][.005]["pre_reassign_noise"] = 836
    assert .02 not in select(rows)["eligible"]
    assert FROZEN_PRE_NOISE_CAPS == {.005: 835, .01: 7, .02: 18}


def test_all_scientific_predicates_are_required_per_ratio():
    frozen = {"P1_unique_dominant_capture": 1, "macro_image_cap": 1, "minimum_image_cap": 1,
              "purity_weighted": 1, "ARI": 1, "AMI": 1, "pre_reassign_noise": 2,
              "post_reassign_noise": 2, "fragmentation": 2, "lost_classes": ["x"],
              "captured_classes": ["a"]}
    selected = dict(frozen); selected["ARI"] = 2
    assert gate_ratio(selected, frozen)
    worse = dict(selected); worse["AMI"] = .5
    assert not gate_ratio(worse, frozen)
    assert not gate_all({.005: selected, .01: worse, .02: selected}, {dial: frozen for dial in DIALS})
    assert not gate_ratio(frozen, frozen)  # strict favorable difference is mandatory


def test_worker1_real_success_and_endpoint_failure_use_synthetic_fixture(tmp_path, monkeypatch):
    root = tmp_path / "root"; root.mkdir()
    fake_r3 = tmp_path / "r3.json"; fake_r3.write_text("{}")
    frozen = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    projection = frozen.copy()
    predictions = {}
    for dial in worker1.DIAL_SPECS:
        predictions[f"{dial['mcs']}_{dial['min_samples']}_frozen"] = np.array([0, 0])
        predictions[f"{dial['mcs']}_{dial['min_samples']}_ep12"] = np.array([0, 0])
    monkeypatch.setattr(worker1, "ROOT", root)
    monkeypatch.setattr(worker1, "R3", fake_r3)
    monkeypatch.setattr(worker1, "ENDPOINT_EXPECTED", {0.0: {d: (1, 0) for d in DIALS}, 1.0: {d: (1, 0) for d in DIALS}})
    raw_metric = {"k": 1, "pre_reassign_noise": 0, "stability": .8, "coherence": .9, "over_merge": 0}
    selection = {"per_dial": [{"dial": {"mcs": dial["mcs"], "ms": dial["min_samples"]}, "metric_schema": {"ep12": list(raw_metric)}} for dial in worker1.DIAL_SPECS]}
    monkeypatch.setattr(worker1, "validate_unlabeled_inputs", lambda opened: (["p0", "p1"], frozen, projection, predictions, selection, opened))
    monkeypatch.setattr(worker1, "label_free_row", lambda z, dial: (metric(1, 0), np.array([0, 0]), raw_metric))
    result = worker1.run_screen()
    assert result["status"] == "selected"
    assert (root / "selection_seal.json").is_file()
    assert (root / "unlabeled_process_receipt.json").is_file()
    predictions[f"21_5_frozen"] = np.array([-1, 0])
    failed = tmp_path / "failed"; failed.mkdir(); monkeypatch.setattr(worker1, "ROOT", failed)
    with pytest.raises(RuntimeError, match="endpoint"):
        worker1.run_screen()
    assert not (failed / "selection_seal.json").exists()


def test_fail_closed_no_candidate_and_create_new(tmp_path):
    rows = candidates()
    for alpha in INTERIOR_ALPHAS:
        for dial in DIALS:
            rows[alpha][dial]["stability"] = 0
    assert select(rows)["status"] == "no_candidate"
    target = tmp_path / "new.json"
    atomic_json_new(target, {"one": 1})
    with pytest.raises(RuntimeError):
        atomic_json_new(target, {"two": 2})


def test_recovery_r3_lineage_is_distinct_from_current_fusion_r3():
    selection = {"schema_version": "rule_c_label_free_selection.v3", "status": "selected", "selection_valid": True,
                 "selected_epoch": 12, "labels_used": False, "offline_labels_evaluated": False,
                 "source_bundle": {"bundle_sha256": common.HASHES["bundle"], "npz_sha256": common.HASHES["npz"],
                                   "pool_path": str(common.UNLABELED), "pool_sha256": common.HASHES["unlabeled"],
                                   "pool_count": 4178, "ordered_paths_sha256": common.ORDERED_PATHS_SHA256},
                 "r3_binding_sha256": common.RECOVERY_R3_FILE_SHA256,
                 "r3_evidence_packet_sha256": common.RECOVERY_R3_EVIDENCE_SHA256,
                 "r3_decision_binding_sha256": common.RECOVERY_R3_DECISION_BINDING_SHA256,
                 "source_v2_snapshot_sha256": common.SOURCE_V2_SNAPSHOT_SHA256,
                 "selector_source_sha256": common.SELECTOR_SOURCE_SHA256,
                 "pool": str(common.UNLABELED), "pool_sha256": common.HASHES["unlabeled"],
                 "selection_manifest_count": 4178, "selection_manifest_sha256": common.HASHES["unlabeled"],
                 "selected_checkpoint": {"selected_checkpoint_sha256": "6db73a0ae9aefdffe8213598a79959c4908bf5616a4785ee22ab59c82d679346"}}
    seal = {"schema_version": "rule_c_epoch_seal.v3", "selected_epoch": 12, "bundle_sha256": common.HASHES["bundle"],
            "npz_sha256": common.HASHES["npz"], "selected_checkpoint_sha256": selection["selected_checkpoint"]["selected_checkpoint_sha256"],
            "selection_snapshot_path": str(common.SELECTION_V3), "selection_snapshot_sha256": common.HASHES["selection_v3"],
            "pool_path": str(common.UNLABELED), "pool_sha256": common.HASHES["unlabeled"], "pool_count": 4178,
            "ordered_paths_sha256": common.ORDERED_PATHS_SHA256,
            "source_v2_snapshot_sha256": common.SOURCE_V2_SNAPSHOT_SHA256,
            "selector_source_sha256": common.SELECTOR_SOURCE_SHA256,
            "r3_binding_sha256": common.RECOVERY_R3_FILE_SHA256, "labels_used": False}
    bundle = {"npz_path": str(common.NPZ), "npz_sha256": common.HASHES["npz"],
              "pool_sha256": common.HASHES["unlabeled"]}
    common._lineage(selection, seal, bundle)
    selection["r3_binding_sha256"] = common.HASHES["r3_file"]
    with pytest.raises(RuntimeError, match="selection lineage"):
        common._lineage(selection, seal, bundle)


def test_current_panel_envelope_binds_nested_result_evidence():
    opened = []
    common.verify_runtime_bindings(opened)
    opened_paths = {item["path"] for item in opened}
    assert str(common.R3) in opened_paths
    assert str(common.EVIDENCE) in opened_paths
