"""Unit tests for scripts/campaign_checkpoint.py and scripts/run_unknown_campaign.py.

Run: python -m pytest tests/test_run_unknown_campaign.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.campaign_checkpoint import (
    classify_checkpoint,
    check_predictor_compatibility,
    detect_proj_arch,
    sha256_file,
)
from scripts.run_unknown_campaign import (
    Registry,
    Campaign,
    evaluate_gate,
    check_allowlist,
    check_disk_guard,
    build_command,
    preflight,
    sha256_obj,
    DEFAULT_GATES,
    validate_panel_dispatch,
    recompute_contract,
)

# ===================== fixtures: synthetic checkpoints =====================
def _bn_proj_sd():
    net = nn.Sequential(nn.Linear(8, 8, bias=False), nn.BatchNorm1d(8), nn.ReLU(True), nn.Linear(8, 4))
    return {f"net.{k}": v for k, v in net.state_dict().items()}


def _gelu_proj_sd():
    net = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 4))
    return dict(net.state_dict())


@pytest.fixture
def projection_only_bn_ckpt(tmp_path) -> Path:
    p = tmp_path / "proj_ep20.pt"
    torch.save({"proj": _bn_proj_sd(), "epoch": 20, "G": 0.1, "Q": 0.2, "L": 0.05}, p)
    return p


@pytest.fixture
def full_contrastive_gelu_ckpt(tmp_path) -> Path:
    p = tmp_path / "final_infer.pt"
    sd = {f"backbone.{k}": v for k, v in {"stem.weight": torch.randn(3, 3)}.items()}
    sd.update({f"proj.{k}": v for k, v in _gelu_proj_sd().items()})
    torch.save({"state_dict": sd}, p)
    return p


@pytest.fixture
def full_contrastive_bn_ckpt(tmp_path) -> Path:
    """Combined backbone+proj with the BN head -- no loader exists in this repo for this shape."""
    p = tmp_path / "combined_bn.pt"
    sd = {f"backbone.{k}": v for k, v in {"stem.weight": torch.randn(3, 3)}.items()}
    sd.update({f"proj.{k}": v for k, v in _bn_proj_sd().items()})
    torch.save({"state_dict": sd}, p)
    return p


@pytest.fixture
def cnn_tapt_ckpt(tmp_path) -> Path:
    p = tmp_path / "cnn_best_model.pth"
    sd = {"stem.0.weight": torch.randn(4, 3, 3, 3), "stages.0.blocks.0.conv.weight": torch.randn(4, 4, 3, 3),
          "head.norm.weight": torch.randn(4), "head.fc.weight": torch.randn(5, 4), "head.fc.bias": torch.randn(5)}
    torch.save({"model": sd, "classes": ["a", "b", "c", "d", "e"], "backbone": "convnextv2_base",
                "val_acc": 0.9, "epoch": 30}, p)
    return p


# ===================== scripts/campaign_checkpoint.py =====================
class TestClassifyCheckpoint:
    def test_projection_only_bn(self, projection_only_bn_ckpt):
        info = classify_checkpoint(projection_only_bn_ckpt)
        assert info["type"] == "projection-only"
        assert info["proj_arch"] == "bn"
        assert info["needs_backbone"] is True
        assert "grouping_deploy.py" in info["compatible_predictors"]
        assert info["blocked_as_predictor"] is False

    def test_full_contrastive_gelu(self, full_contrastive_gelu_ckpt):
        info = classify_checkpoint(full_contrastive_gelu_ckpt)
        assert info["type"] == "full-contrastive"
        assert info["proj_arch"] == "gelu"
        assert "predict_grouping_prod.py" in info["compatible_predictors"]
        assert info["blocked_as_predictor"] is False

    def test_full_contrastive_bn_has_no_loader(self, full_contrastive_bn_ckpt):
        """Combined backbone+proj with a BN head -- neither predictor script can load it today."""
        info = classify_checkpoint(full_contrastive_bn_ckpt)
        assert info["type"] == "full-contrastive"
        assert info["proj_arch"] == "bn"
        assert info["compatible_predictors"] == []
        assert info["blocked_as_predictor"] is True

    def test_cnn_tapt(self, cnn_tapt_ckpt):
        info = classify_checkpoint(cnn_tapt_ckpt)
        assert info["type"] == "cnn-tapt"
        assert info["compatible_predictors"] == []
        assert info["blocked_as_predictor"] is True
        assert info["classes"] == ["a", "b", "c", "d", "e"]

    def test_missing_file(self, tmp_path):
        info = classify_checkpoint(tmp_path / "does_not_exist.pt")
        assert info["type"] == "missing"
        assert info["blocked_as_predictor"] is True

    def test_unrecognized_layout(self, tmp_path):
        p = tmp_path / "garbage.pt"
        torch.save({"totally": "unrelated", "fields": 123}, p)
        info = classify_checkpoint(p)
        assert info["type"] == "unknown"
        assert info["blocked_as_predictor"] is True

    def test_detect_proj_arch_empty(self):
        assert detect_proj_arch({}) == "unknown"
        assert detect_proj_arch(None) == "unknown"


class TestPredictorCompatibility:
    """These four cases are the literal 260726 incident + its fix, pinned as tests."""

    def test_bn_head_blocked_from_predict_grouping_prod(self, projection_only_bn_ckpt):
        info = classify_checkpoint(projection_only_bn_ckpt)
        ok, msg = check_predictor_compatibility("predict_grouping_prod.py", info, has_backbone_arg=False)
        assert ok is False
        assert "260726" in msg or "SystemExit" in msg

    def test_bn_head_allowed_via_grouping_deploy_with_backbone(self, projection_only_bn_ckpt):
        info = classify_checkpoint(projection_only_bn_ckpt)
        ok, msg = check_predictor_compatibility("grouping_deploy.py", info, has_backbone_arg=True)
        assert ok is True

    def test_bn_head_blocked_via_grouping_deploy_without_backbone(self, projection_only_bn_ckpt):
        info = classify_checkpoint(projection_only_bn_ckpt)
        ok, msg = check_predictor_compatibility("grouping_deploy.py", info, has_backbone_arg=False)
        assert ok is False
        assert "backbone" in msg

    def test_gelu_head_allowed_via_predict_grouping_prod(self, full_contrastive_gelu_ckpt):
        info = classify_checkpoint(full_contrastive_gelu_ckpt)
        ok, msg = check_predictor_compatibility("predict_grouping_prod.py", info, has_backbone_arg=False)
        assert ok is True

    def test_gelu_head_blocked_via_grouping_deploy(self, full_contrastive_gelu_ckpt):
        info = classify_checkpoint(full_contrastive_gelu_ckpt)
        ok, msg = check_predictor_compatibility("grouping_deploy.py", info, has_backbone_arg=True)
        assert ok is False

    def test_cnn_tapt_blocked_everywhere(self, cnn_tapt_ckpt):
        info = classify_checkpoint(cnn_tapt_ckpt)
        for predictor in ("grouping_deploy.py", "predict_grouping_prod.py"):
            ok, msg = check_predictor_compatibility(predictor, info, has_backbone_arg=True)
            assert ok is False
            assert "cnn-tapt" in msg

    def test_unknown_predictor_name_rejected(self, projection_only_bn_ckpt):
        info = classify_checkpoint(projection_only_bn_ckpt)
        ok, msg = check_predictor_compatibility("some_other_script.py", info)
        assert ok is False


# ===================== evaluate_gate =====================
def _passing_metrics(n_seeds=3):
    return {
        "n_seeds": n_seeds,
        "primary_wafer": {"unknown": {"capture_drop_pp": 0.0}, "mixedwm38": {"capture_drop_pp": 0.5}},
        "background_far": {"far_per_batch_delta": 0.1},
        "far_or_noise_improvements": {"unknown": {"far_improve_pp": 4.0}, "mixedwm38": {"noise_improve_pp": 5.0}},
        "ari": {"drop": 0.0}, "ami": {"drop": 0.0},
        "temporal": {"far_events": 0, "detect_batches": 1, "novel_per_batch": 25},
        "clean546": {"capture": "7/7", "noise_pct": 2.6},
    }


class TestEvaluateGate:
    def test_all_pass(self):
        passed, reasons = evaluate_gate(DEFAULT_GATES, _passing_metrics())
        assert passed is True
        assert reasons == []

    def test_empty_metrics_fails_closed(self):
        passed, reasons = evaluate_gate(DEFAULT_GATES, {})
        assert passed is False
        assert len(reasons) > 0

    def test_capture_drop_too_much(self):
        m = _passing_metrics()
        m["primary_wafer"]["unknown"]["capture_drop_pp"] = 2.0
        passed, reasons = evaluate_gate(DEFAULT_GATES, m)
        assert passed is False
        assert any("capture dropped" in r for r in reasons)

    def test_ari_drop_too_much(self):
        m = _passing_metrics()
        m["ari"]["drop"] = 0.05
        passed, reasons = evaluate_gate(DEFAULT_GATES, m)
        assert passed is False
        assert any("ARI dropped" in r for r in reasons)

    def test_insufficient_seeds(self):
        m = _passing_metrics(n_seeds=1)
        passed, reasons = evaluate_gate(DEFAULT_GATES, m)
        assert passed is False
        assert any("n_seeds" in r for r in reasons)

    def test_n_seeds_override_relaxes_requirement(self):
        m = _passing_metrics(n_seeds=1)
        passed, reasons = evaluate_gate(DEFAULT_GATES, m, n_seeds_min_override=1)
        assert passed is True

    def test_clean546_noise_regression(self):
        m = _passing_metrics()
        m["clean546"]["noise_pct"] = 10.0  # far from the 2.57% transductive target
        passed, reasons = evaluate_gate(DEFAULT_GATES, m)
        assert passed is False
        assert any("clean546 noise_pct" in r for r in reasons)

    def test_clean546_capture_regression(self):
        m = _passing_metrics()
        m["clean546"]["capture"] = "6/7"
        passed, reasons = evaluate_gate(DEFAULT_GATES, m)
        assert passed is False

    def test_far_or_noise_improvement_insufficient_datasets(self):
        m = _passing_metrics()
        m["far_or_noise_improvements"] = {"unknown": {"far_improve_pp": 4.0}}  # only 1 dataset improved
        passed, reasons = evaluate_gate(DEFAULT_GATES, m)
        assert passed is False
        assert any("datasets improved" in r for r in reasons)

    def test_severstal_optional_when_absent(self):
        m = _passing_metrics()
        assert "severstal" not in m
        passed, _ = evaluate_gate(DEFAULT_GATES, m)
        assert passed is True  # absent severstal metric does not block -- only enforced once supplied

    def test_severstal_milestone_enforced_when_present(self):
        m = _passing_metrics()
        m["severstal"] = {"capture": "4/4", "noise_pct": 70.0}  # only 7.74pp improvement, needs >=20pp
        passed, reasons = evaluate_gate(DEFAULT_GATES, m)
        assert passed is False
        assert any("severstal noise_pct improved" in r for r in reasons)

    def test_severstal_milestone_passes_comfortably_above_threshold(self):
        m = _passing_metrics()
        m["severstal"] = {"capture": "4/4", "noise_pct": 50.0}  # 27.74pp improvement, clear of the 20pp bar
        passed, reasons = evaluate_gate(DEFAULT_GATES, m)
        assert passed is True


# ===================== allowlist / disk guard =====================
class TestSafetyGuards:
    @pytest.mark.parametrize("field,reason", [
        ("source_snapshot_sha256", "panel_source_binding_drift"),
        ("data_snapshot_sha256", "panel_data_binding_drift"),
        ("backbone_snapshot_sha256", "panel_backbone_binding_drift"),
        ("env_snapshot_sha256", "panel_env_binding_drift"),
    ])
    def test_recomputed_binding_mutations_fail_closed(self, monkeypatch, tmp_path, field, reason):
        item={"id":"base","tier":3,"step":"base","seed":42}
        binding={"config_snapshot_sha256":"config","ordered_queue_sha256":"queue","source_snapshot_sha256":"source","data_snapshot_sha256":"data","backbone_snapshot_sha256":"backbone","env_snapshot_sha256":"env"}
        monkeypatch.setattr("scripts.run_unknown_campaign.validate_panel_bundle",lambda *_:(True,"ok",{}))
        monkeypatch.setattr("scripts.run_unknown_campaign.read_json",lambda _:{"action":"experiment_design","proposed_queue":[item],"binding":binding})
        monkeypatch.setattr("scripts.run_unknown_campaign.sha256_obj",lambda _:"queue")
        current=dict(binding); current[field]="mutated"
        monkeypatch.setattr("scripts.run_unknown_campaign.recompute_contract",lambda *_:current)
        cfg={"outer_loop":{"initial_queue":{"cells":[item]}}}
        ok,got=validate_panel_dispatch(tmp_path/"r3.json",None,item,"experiment_design","config",cfg,tmp_path/"c")
        assert not ok and got==reason


@pytest.mark.parametrize("queue", [[], [{"id":"base","tier":3,"step":"base","seed":42},{"id":"base","tier":3,"step":"base","seed":42}], [{"id":"other","tier":3,"step":"base","seed":42}]])
def test_panel_dispatch_rejects_queue_subset_duplicate_or_wrong_item(monkeypatch, tmp_path, queue):
    expected=[{"id":"base","tier":3,"step":"base","seed":42},{"id":"lr","tier":3,"step":"lr","seed":42}]
    evidence={"action":"experiment_design","proposed_queue":queue,"binding":{"config_snapshot_sha256":"c","ordered_queue_sha256":sha256_obj(queue)}}
    monkeypatch.setattr("scripts.run_unknown_campaign.validate_panel_bundle",lambda *_:(True,"ok",{}))
    monkeypatch.setattr("scripts.run_unknown_campaign.read_json",lambda _:evidence)
    ok,reason=validate_panel_dispatch(tmp_path/"r3",None,expected[0],"experiment_design","c",{"outer_loop":{"initial_queue":{"cells":expected}}})
    assert not ok and reason in {"panel_queue_not_exact_config_order","panel_queue_ids_invalid"}

    def test_full_campaign_cli_is_explicitly_disabled(self):
        with pytest.raises(SystemExit, match="full campaign CLI is disabled"):
            Campaign().run(resume=True,max_tier=None,stop_after_current=False)
    def test_allowlist_accepts_subpath(self, tmp_path):
        root = tmp_path / "unknown"
        sub = root / "BrokenRing"
        sub.mkdir(parents=True)
        assert check_allowlist(sub, [str(root)]) is True

    def test_allowlist_rejects_outside_path(self, tmp_path):
        root = tmp_path / "unknown"
        other = tmp_path / "definitely_not_allowlisted"
        root.mkdir()
        other.mkdir()
        assert check_allowlist(other, [str(root)]) is False

    def test_allowlist_rejects_empty_list(self, tmp_path):
        assert check_allowlist(tmp_path, []) is False

    def test_disk_guard_passes_tiny_threshold(self):
        ok, detail = check_disk_guard({"D": 0.001})
        assert ok is True
        assert detail["D"]["ok"] is True

    def test_disk_guard_fails_impossible_threshold(self):
        ok, detail = check_disk_guard({"D": 1e9})  # 1 exabyte -- never available
        assert ok is False
        assert detail["D"]["ok"] is False


# ===================== registry =====================
class TestRegistry:
    def test_append_and_read_all(self, tmp_path):
        reg = Registry(tmp_path / "registry.jsonl")
        reg.append({"tier": 1, "step": "predict", "seed": None, "status": "completed"})
        reg.append({"tier": 2, "step": "train", "seed": 42, "status": "failed"})
        recs = reg.read_all()
        assert len(recs) == 2
        assert recs[0]["tier"] == 1 and recs[1]["seed"] == 42

    def test_append_only_never_rewrites(self, tmp_path):
        path = tmp_path / "registry.jsonl"
        reg = Registry(path)
        reg.append({"tier": 1, "step": "a", "seed": None, "status": "failed"})
        reg2 = Registry(path)
        reg2.append({"tier": 1, "step": "a", "seed": None, "status": "completed"})
        # both lines survive -- "latest wins" is a read-time concern, not a write-time rewrite
        assert len(path.read_text(encoding="utf-8").splitlines()) == 2

    def test_completed_steps_latest_attempt_wins(self, tmp_path):
        reg = Registry(tmp_path / "registry.jsonl")
        reg.append({"tier": 2, "step": "train", "seed": 1, "status": "failed"})
        reg.append({"tier": 2, "step": "train", "seed": 1, "status": "completed"})  # retry succeeded
        assert (2, "train", 1) in reg.completed_steps()

    def test_completed_steps_excludes_still_failed(self, tmp_path):
        reg = Registry(tmp_path / "registry.jsonl")
        reg.append({"tier": 1, "step": "predict", "seed": None, "status": "failed"})
        assert (1, "predict", None) not in reg.completed_steps()

    def test_current_champion_tracks_latest_promotion(self, tmp_path):
        reg = Registry(tmp_path / "registry.jsonl")
        reg.append({"tier": 1, "step": "predict", "seed": None, "status": "completed",
                     "run_id": "run_a", "champion_after": "run_a"})
        assert reg.current_champion()["run_id"] == "run_a"

    def test_current_champion_none_when_never_promoted(self, tmp_path):
        reg = Registry(tmp_path / "registry.jsonl")
        reg.append({"tier": 1, "step": "predict", "seed": None, "status": "failed", "champion_after": None})
        assert reg.current_champion() is None

    def test_tolerates_torn_last_line(self, tmp_path):
        path = tmp_path / "registry.jsonl"
        reg = Registry(path)
        reg.append({"tier": 1, "step": "a", "seed": None, "status": "completed"})
        with open(path, "a", encoding="utf-8") as f:
            f.write('{"tier": 2, "step": "b", incomplete')  # simulate a killed-mid-write process
        assert len(reg.read_all()) == 1  # torn line dropped, first record still readable


# ===================== command building =====================
class TestBuildCommand:
    def test_substitutes_placeholders(self):
        step = {"command": ["{python}", "grouping_deploy.py", "--out", "{run_dir}", "--seed", "{seed}"]}
        argv = build_command(step, {"python": "python", "run_dir": "runs/x", "seed": 42})
        assert argv == ["python", "grouping_deploy.py", "--out", "runs/x", "--seed", "42"]

    def test_missing_placeholder_raises(self):
        step = {"command": ["{python}", "--unresolved", "{nope}"]}
        with pytest.raises(ValueError):
            build_command(step, {"python": "python"})


# ===================== hashing determinism =====================
class TestHashing:
    def test_sha256_file_matches_hashlib(self, tmp_path):
        import hashlib
        p = tmp_path / "f.bin"
        p.write_bytes(b"hello unknown-campaign" * 1000)
        assert sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()

    def test_sha256_obj_order_independent(self):
        a = sha256_obj({"x": 1, "y": 2})
        b = sha256_obj({"y": 2, "x": 1})
        assert a == b

    def test_sha256_obj_sensitive_to_value_change(self):
        a = sha256_obj({"x": 1})
        b = sha256_obj({"x": 2})
        assert a != b


# ===================== new: env-var command building =====================
from scripts.run_unknown_campaign import build_env


class TestBuildEnv:
    def test_substitutes_placeholders(self):
        step = {"env": {"REPRO_SEED": "{seed}", "REPRO_DATA": "{manifest}"}}
        env = build_env(step, {"python": "python", "run_dir": "runs/x", "seed": 42, "manifest": "m.json"})
        assert env == {"REPRO_SEED": "42", "REPRO_DATA": "m.json"}

    def test_no_env_returns_empty(self):
        assert build_env({}, {"python": "python"}) == {}

    def test_missing_placeholder_raises(self):
        step = {"env": {"REPRO_SEED": "{nope}"}}
        with pytest.raises(ValueError):
            build_env(step, {"python": "python"})


class TestCampaignDependenciesAndSealing:
    def test_same_seed_dependency_uses_recorded_checkpoint(self, tmp_path):
        checkpoint = tmp_path / "seed42.pt"
        checkpoint.write_bytes(b"checkpoint")
        registry = tmp_path / "registry.jsonl"
        campaign = Campaign(registry_path=registry)
        _, producer = campaign._find_step(2, "train_frozen_recipe")
        Registry(registry).append({
            "campaign_id": campaign.campaign_id,
            "config_snapshot_sha256": campaign.config_snapshot_sha256,
            "step_fingerprint": campaign._step_fingerprint(2, producer, 42),
            "tier": 2, "step": "train_frozen_recipe", "seed": 42,
            "status": "completed", "produced_checkpoint_path": str(checkpoint),
            "artifact_path": str(checkpoint), "artifact_sha256": sha256_file(checkpoint)})
        context, error = campaign._dependency_context(
            {"depends_on": {"tier": 2, "step": "train_frozen_recipe", "same_seed": True}}, 42)
        assert error is None
        assert context["dependency_checkpoint"] == str(checkpoint.resolve())
        assert context["dependency_seed"] == 42

    def test_missing_dependency_checkpoint_fails_closed(self, tmp_path):
        registry = tmp_path / "registry.jsonl"
        campaign = Campaign(registry_path=registry)
        _, error = campaign._dependency_context(
            {"depends_on": {"tier": 2, "step": "train_frozen_recipe", "same_seed": True}}, 1)
        assert error == "dependency_not_completed"

    def test_aggregate_requires_three_distinct_completed_seed_records(self, tmp_path):
        registry = tmp_path / "registry.jsonl"
        ledger = Registry(registry)
        campaign = Campaign(registry_path=registry)
        _, producer = campaign._find_step(2, "select_epoch_label_free")
        for seed in (1, 2, 42):
            artifact = tmp_path / f"s{seed}.json"
            artifact.write_text("{}", encoding="utf-8")
            ledger.append({
                "campaign_id": campaign.campaign_id,
                "config_snapshot_sha256": campaign.config_snapshot_sha256,
                "step_fingerprint": campaign._step_fingerprint(2, producer, seed),
                "tier": 2, "step": "select_epoch_label_free", "seed": seed,
                "status": "completed", "run_id": f"s{seed}",
                "artifact_path": str(artifact), "artifact_sha256": sha256_file(artifact)})
        records = campaign._aggregate_seed_records(
            {"tier": 2, "step": "select_epoch_label_free", "seeds": [1, 2, 42]})
        assert [record["run_id"] for record in records] == ["s1", "s2", "s42"]

    def test_sealed_test_reference_fails_before_manifest_access(self, tmp_path):
        sealed = tmp_path / "sealed.json"  # deliberately not created/read
        cfg = {"sealed_test_pools": {"unknown": str(sealed)}, "safety": {"min_free_gb": {"D": 0.001}}}
        ok, blockers, detail = preflight({"manifest": str(sealed), "pool": str(sealed)}, cfg)
        assert ok is False and blockers == ["sealed_test_reference"]
        assert detail["sealed_test"] == "fail_closed"


class TestSupervisorQueue:
    def test_post_gate_cells_are_not_materialized_until_explicit_pass(self):
        from scripts.run_unknown_supervisor import initial_state, materialize_initial_queue, materialize_post_gate_queue
        cfg = {"outer_loop": {"initial_queue": {"cells": [{"id": "base"}]},
                              "post_gate_cells": [{"id": "a"}, {"id": "b"}]}}
        state = initial_state(cfg)
        materialize_initial_queue(state)
        assert state["queue"] == []
        state["panel"] = {"approved": True, "action": "experiment_design",
                          "r3_artifact_path": "r3.json", "evidence_packet_sha": "abc"}
        materialize_initial_queue(state)
        materialize_post_gate_queue(state)
        assert [item["id"] for item in state["queue"]] == ["base"]
        state["initial_gate_passed"] = True
        state["initial_design_approval_sha"] = "abc"
        state["completed"] = [{"id": "base"}]
        materialize_post_gate_queue(state)
        assert [item["id"] for item in state["queue"]] == ["base"]
        state["panel"] = {"approved": True, "action": "next_queue",
                          "r3_artifact_path": "next-r3.json", "evidence_packet_sha": "next"}
        materialize_post_gate_queue(state)
        assert [item["id"] for item in state["queue"]] == ["base", "a", "b"]


# ===================== new: baseline-diff metrics adapter =====================
from scripts.run_unknown_campaign import (
    _capture_frac, load_temporal_metrics, load_background_far_delta, build_gate_metrics,
)


class TestCaptureFrac:
    def test_parses_fraction(self):
        assert _capture_frac("31/31") == 1.0
        assert _capture_frac("4/7") == pytest.approx(4 / 7)

    def test_none_for_bad_input(self):
        assert _capture_frac(None) is None
        assert _capture_frac("not-a-fraction") is None
        assert _capture_frac("0/0") is None  # ZeroDivisionError -> None, not a crash


class TestTemporalAdapter:
    """Against a synthetic summary_tables.csv shaped exactly like perf-temporal's real output."""

    @pytest.fixture
    def report_dir(self, tmp_path):
        import csv
        rows = [
            {"metric": "FAR_alarms_over_4_bg_batches", "arm": "frozen", "P": "10", "K": "1",
             "size05": "5", "size10": "5", "size20": "5", "size30": "5", "unit": "count"},
            {"metric": "FAR_alarms_over_4_bg_batches", "arm": "champion", "P": "10", "K": "1",
             "size05": "0", "size10": "0", "size20": "0", "size30": "0", "unit": "count"},
            {"metric": "true_positive_detection_lag", "arm": "frozen", "P": "10", "K": "1",
             "size05": "3", "size10": "1", "size20": "0", "size30": "1", "unit": "batches_after_t0"},
            {"metric": "true_positive_detection_lag", "arm": "champion", "P": "10", "K": "1",
             "size05": "3", "size10": "1", "size20": "0", "size30": "1", "unit": "batches_after_t0"},
        ]
        p = tmp_path / "summary_tables.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        return tmp_path

    def test_reads_champion_operating_point(self, report_dir):
        op = {"P": "10", "K": "1", "size_col": "size20", "arm": "champion"}
        m = load_temporal_metrics(report_dir, op)
        assert m == {"far_events": 0, "detect_batches": 0, "novel_per_batch": 20, "held_out_batches": 4}

    def test_missing_file_returns_none(self, tmp_path):
        op = {"P": "10", "K": "1", "size_col": "size20", "arm": "champion"}
        assert load_temporal_metrics(tmp_path, op) is None

    def test_missing_arm_returns_none(self, report_dir):
        op = {"P": "10", "K": "1", "size_col": "size20", "arm": "does_not_exist"}
        assert load_temporal_metrics(report_dir, op) is None

    def test_background_far_delta_champion_vs_frozen(self, report_dir):
        op = {"P": "10", "K": "1", "size_col": "size20", "arm": "champion"}
        bg = load_background_far_delta(report_dir, op, baseline_arm="frozen")
        # champion 0/4=0 alarms/batch vs frozen 5/4=1.25 alarms/batch -> -1.25 (improvement)
        assert bg["far_per_batch_delta"] == pytest.approx(-1.25)

    def test_background_far_delta_missing_baseline_arm(self, report_dir):
        op = {"P": "10", "K": "1", "size_col": "size20", "arm": "champion"}
        assert load_background_far_delta(report_dir, op, baseline_arm="nonexistent") is None


class TestBuildGateMetrics:
    def test_fills_from_baseline_and_candidate(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "offline_summary.json").write_text(json.dumps(
            {"P1_capture": "30/31", "P2_noise_pct": 5.0, "ARI": 0.80, "AMI": 0.90}), encoding="utf-8")
        cfg = {"baselines": {"unknown_eval100": {"P1_capture": "31/31", "noise_pct": 10.0, "ARI": 0.814, "AMI": 0.92}}}
        step = {"baseline_dataset": "unknown_eval100", "seeds": [1, 2, 42]}
        m = build_gate_metrics(step, cfg, run_dir)
        assert m["n_seeds"] == 3
        assert m["primary_wafer"]["unknown_eval100"]["capture_drop_pp"] == pytest.approx(
            (1.0 - 30 / 31) * 100.0, abs=1e-3)  # implementation rounds to 4 decimals
        assert m["ari"]["drop"] == pytest.approx(0.014, abs=1e-9)
        assert m["ami"]["drop"] == pytest.approx(0.02, abs=1e-9)
        assert m["far_or_noise_improvements"]["unknown_eval100"]["noise_improve_pp"] == pytest.approx(5.0)

    def test_empty_when_no_candidate_file(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        cfg = {"baselines": {"unknown_eval100": {"P1_capture": "31/31", "noise_pct": 0.0}}}
        step = {"baseline_dataset": "unknown_eval100"}
        m = build_gate_metrics(step, cfg, run_dir)
        assert "primary_wafer" not in m and "ari" not in m

    def test_empty_when_no_baseline_configured(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "offline_summary.json").write_text(json.dumps({"P1_capture": "1/1"}), encoding="utf-8")
        m = build_gate_metrics({"baseline_dataset": "no_such_dataset"}, {"baselines": {}}, run_dir)
        assert "primary_wafer" not in m

    def test_clean546_and_severstal_subblocks(self, tmp_path):
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "offline_summary.json").write_text(json.dumps(
            {"P1_capture": "7/7", "P2_noise_pct": 2.5}), encoding="utf-8")
        cfg = {"baselines": {"clean546": {"P1_capture": "4/7", "noise_pct": 60.0}}}
        m = build_gate_metrics({"baseline_dataset": "clean546"}, cfg, run_dir)
        assert m["clean546"] == {"capture": "7/7", "noise_pct": 2.5}

    def test_temporal_and_background_far_wired(self, tmp_path):
        import csv
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        rows = [
            {"metric": "FAR_alarms_over_4_bg_batches", "arm": "frozen", "P": "10", "K": "1",
             "size05": "5", "size10": "5", "size20": "5", "size30": "5", "unit": "count"},
            {"metric": "FAR_alarms_over_4_bg_batches", "arm": "champion", "P": "10", "K": "1",
             "size05": "0", "size10": "0", "size20": "0", "size30": "0", "unit": "count"},
            {"metric": "true_positive_detection_lag", "arm": "frozen", "P": "10", "K": "1",
             "size05": "3", "size10": "1", "size20": "0", "size30": "1", "unit": "batches_after_t0"},
            {"metric": "true_positive_detection_lag", "arm": "champion", "P": "10", "K": "1",
             "size05": "3", "size10": "1", "size20": "0", "size30": "1", "unit": "batches_after_t0"},
        ]
        with open(tmp_path / "summary_tables.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        step = {"temporal_report_path": str(tmp_path), "operating_point": {"P": "10", "K": "1", "size_col": "size20", "arm": "champion"}}
        m = build_gate_metrics(step, {}, run_dir)
        assert m["temporal"] == {"far_events": 0, "detect_batches": 0, "novel_per_batch": 20, "held_out_batches": 4}
        assert m["background_far"]["far_per_batch_delta"] == pytest.approx(-1.25)


# ===================== resolve_trainer_output_dir (260726 team-lead fix) =====================
from scripts.run_unknown_campaign import resolve_trainer_output_dir
import time


class TestResolveTrainerOutputDir:
    def test_finds_single_candidate_after_start(self, tmp_path):
        before = time.time()
        target = tmp_path / "may_repro" / "abl_campaign_t2_s1_B4_260726_120000"
        target.mkdir(parents=True)
        found = resolve_trainer_output_dir(tmp_path, "_campaign_t2_s1", "B4", before)
        assert found == target

    def test_ignores_stale_dir_created_before_start(self, tmp_path):
        stale = tmp_path / "may_repro" / "abl_campaign_t2_s1_B4_260726_090000"
        stale.mkdir(parents=True)
        after = time.time() + 5  # well past the function's 2s OS-precision tolerance
        with pytest.raises(RuntimeError, match="no trainer output dir found"):
            resolve_trainer_output_dir(tmp_path, "_campaign_t2_s1", "B4", after)

    def test_raises_on_zero_candidates(self, tmp_path):
        with pytest.raises(RuntimeError, match="no trainer output dir found"):
            resolve_trainer_output_dir(tmp_path, "_campaign_t2_s1", "B4", time.time())

    def test_raises_on_ambiguous_multiple_candidates(self, tmp_path):
        before = time.time()
        (tmp_path / "may_repro" / "abl_campaign_t2_s1_B4_260726_120000").mkdir(parents=True)
        (tmp_path / "may_repro" / "abl_campaign_t2_s1_B4_260726_120500").mkdir(parents=True)
        with pytest.raises(RuntimeError, match="ambiguous"):
            resolve_trainer_output_dir(tmp_path, "_campaign_t2_s1", "B4", before)

    def test_different_seed_tag_not_matched(self, tmp_path):
        before = time.time()
        (tmp_path / "may_repro" / "abl_campaign_t2_s2_B4_260726_120000").mkdir(parents=True)
        with pytest.raises(RuntimeError, match="no trainer output dir found"):
            resolve_trainer_output_dir(tmp_path, "_campaign_t2_s1", "B4", before)
@pytest.mark.parametrize("tamper,reason", [
    ("manifest", "split overlap audit manifest binding mismatch"),
    ("content", "split overlap audit image-content binding mismatch"),
    ("tool", "split overlap audit tool binding mismatch"),
])
def test_recompute_contract_rejects_tampered_split_audit_binding(monkeypatch, tmp_path, tamper, reason):
    import hashlib
    from types import SimpleNamespace
    root = tmp_path / "root"; root.mkdir(); (root / "train.bin").write_bytes(b"train"); (root / "validation.bin").write_bytes(b"validation")
    train, validation = tmp_path / "train.json", tmp_path / "validation.json"
    train.write_text(json.dumps({"root": str(root), "files": ["train.bin"]})); validation.write_text(json.dumps({"root": str(root), "files": ["validation.bin"]}))
    tool = tmp_path / "scripts" / "audit_manifest_overlap.py"; tool.parent.mkdir(); tool.write_text("# audit")
    train_rows=[("train.bin", hashlib.sha256(b"train").hexdigest())]; validation_rows=[("validation.bin", hashlib.sha256(b"validation").hexdigest())]
    report={"schema_version":"manifest_overlap_audit.v1","status":"clear","review_required":False,
            "inputs":{"train":{"path":str(train),"manifest_sha256":sha256_file(train),"image_content_sha256":sha256_obj(train_rows)},"validation":{"path":str(validation),"manifest_sha256":sha256_file(validation),"image_content_sha256":sha256_obj(validation_rows)}},
            "exact":{"content_pair_count":0,"same_resolved_path_count":0},"near":{"threshold":8,"candidate_pair_count":0},"tool_sha256":sha256_file(tool)}
    if tamper == "manifest": report["inputs"]["train"]["manifest_sha256"]="bad"
    elif tamper == "content": report["inputs"]["validation"]["image_content_sha256"]="bad"
    else: report["tool_sha256"]="bad"
    audit=tmp_path / "audit.json"; audit.write_text(json.dumps(report))
    cfg={"tiers":[{"tier":1,"steps":[{"name":"cell","recipe":{},"env":{},"command":[],"rule_c":{"unlabeled_pool":"train.json","offline_pool":"validation.json"}}]}],
         "safety":{"allowlist_roots":[str(root)],"split_overlap_audits":[{"name":"split","train_manifest":"train.json","validation_manifest":"validation.json","artifact":"audit.json","near_threshold":8}]}}
    config=tmp_path / "config.json"; config.write_text(json.dumps(cfg))
    monkeypatch.setattr("scripts.run_unknown_campaign.REPO_ROOT", tmp_path)
    monkeypatch.setattr("scripts.run_unknown_campaign.REPRO_SOURCE_FILES", ())
    monkeypatch.setattr("scripts.run_unknown_campaign._git_binding", lambda: {})
    monkeypatch.setattr("scripts.run_unknown_campaign.subprocess.run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=""))
    with pytest.raises(ValueError, match=reason):
        recompute_contract(cfg, config, [{"id":"cell","tier":1,"step":"cell","seed":1}])


def test_recompute_contract_hashes_shared_physical_content_once(monkeypatch, tmp_path):
    from types import SimpleNamespace
    root = tmp_path / "root"; root.mkdir()
    shared, backbone = root / "shared.bin", root / "backbone.bin"
    shared.write_bytes(b"shared-content"); backbone.write_bytes(b"backbone")
    manifests = []
    for name in ("train", "unlabeled", "validation"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"root": str(root), "files": ["shared.bin"]}))
        manifests.append(path.name)
    cfg = {"tiers": [{"tier": 1, "steps": [
        {"name": name, "recipe": {}, "env": {"REPRO_DATA": manifests[0]}, "command": [],
         "rule_c": {"unlabeled_pool": manifests[1], "offline_pool": manifests[2], "backbone": "root/backbone.bin"}}
        for name in ("one", "two")]}], "safety": {"allowlist_roots": [str(root)]}}
    config = tmp_path / "config.json"; config.write_text(json.dumps(cfg))
    calls = []
    import scripts.run_unknown_campaign as campaign
    real_sha256_file = campaign.sha256_file
    def counted(path):
        calls.append(Path(path).resolve())
        return real_sha256_file(path)
    monkeypatch.setattr(campaign, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(campaign, "REPRO_SOURCE_FILES", ())
    monkeypatch.setattr(campaign, "_git_binding", lambda: {})
    monkeypatch.setattr(campaign, "sha256_file", counted)
    monkeypatch.setattr(campaign.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=""))
    result = recompute_contract(cfg, config, [
        {"id": "one", "tier": 1, "step": "one", "seed": 1},
        {"id": "two", "tier": 1, "step": "two", "seed": 2},
    ])
    assert calls.count(shared.resolve()) == 1
    assert calls.count(backbone.resolve()) == 1
    assert result["recompute_contract"]["backbones"] == [
        {"path": str(backbone.resolve()), "sha256": real_sha256_file(backbone)},
        {"path": str(backbone.resolve()), "sha256": real_sha256_file(backbone)},
    ]
