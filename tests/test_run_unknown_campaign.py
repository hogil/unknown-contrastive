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
    evaluate_gate,
    check_allowlist,
    check_disk_guard,
    build_command,
    sha256_obj,
    DEFAULT_GATES,
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
        "background_far": {"increase_pp": 0.1},
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
        assert m == {"far_events": 0, "detect_batches": 0, "novel_per_batch": 20}

    def test_missing_file_returns_none(self, tmp_path):
        op = {"P": "10", "K": "1", "size_col": "size20", "arm": "champion"}
        assert load_temporal_metrics(tmp_path, op) is None

    def test_missing_arm_returns_none(self, report_dir):
        op = {"P": "10", "K": "1", "size_col": "size20", "arm": "does_not_exist"}
        assert load_temporal_metrics(report_dir, op) is None

    def test_background_far_delta_champion_vs_frozen(self, report_dir):
        op = {"P": "10", "K": "1", "size_col": "size20", "arm": "champion"}
        bg = load_background_far_delta(report_dir, op, baseline_arm="frozen")
        assert bg["increase_pp"] == 0 - 5  # champion 0 alarms vs frozen 5 -> improvement, not increase

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
        assert m["temporal"] == {"far_events": 0, "detect_batches": 0, "novel_per_batch": 20}
        assert m["background_far"]["increase_pp"] == -5
