from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_label_free_adaptation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_label_free_adaptation", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
adaptation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adaptation)


def _argument(command: list[str], name: str) -> Path:
    return Path(command[command.index(name) + 1])


def _write_extraction(command: list[str], image: Path) -> None:
    output = _argument(command, "--out-dir")
    output.mkdir(parents=True, exist_ok=True)
    (output / "paths.json").write_bytes(
        adaptation.expected_manifest_bytes([image.resolve()])
    )
    np.save(output / "main.npy", np.ones((1, 2), dtype=np.float32))
    np.save(output / "weak_aug.npy", np.ones((1, 2), dtype=np.float32))
    (output / "provenance.json").write_text("{}\n", encoding="utf-8")


def _frozen_gate() -> dict:
    return {
        "schema_version": "label_free_gate.v1",
        "workflow_action": "adapt_required",
        "selected_mode": "frozen",
        "frozen": {"non_noise_count": 1},
        "frozen_approval": {"passed": False},
        "provenance": {"input_files": {}},
    }


def _adapted_gate(*, eligible: bool) -> dict:
    return {
        "schema_version": "label_free_gate.v1",
        "workflow_action": "use_adapted" if eligible else "use_frozen",
        "selected_mode": "adapted" if eligible else "frozen",
        "adapted_absolute": {"passed": eligible},
        "approval": {"passed": eligible},
        "rescue_approval": None,
        "adapted": {
            "non_noise_count": 1,
            "within_group_cosine_coherence": 0.9,
            "bootstrap_stability": 0.9,
            "fragmentation_proxy": 1.0,
        },
    }


def _prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    scenario: str,
) -> tuple[list[str], list[str], Path]:
    pool = tmp_path / "pool"
    pool.mkdir()
    image = pool / "opaque.png"
    image.write_bytes(b"opaque-image")
    weights = tmp_path / "fcmae.pth"
    weights.write_bytes(b"weights")
    out_dir = tmp_path / "out"
    calls: list[str] = []

    monkeypatch.setattr(adaptation, "FCMAE_WEIGHTS", weights)
    monkeypatch.setattr(adaptation, "extraction_cache_valid", lambda *args, **kwargs: False)
    monkeypatch.setattr(adaptation, "gate_cache_valid", lambda *args, **kwargs: False)

    def fake_run_logged(
        name: str,
        command: list[str],
        log_path: Path,
        status: dict,
    ) -> int:
        calls.append(name)
        if name == "frozen_extract":
            if scenario == "frozen_extract_failure":
                return 7
            _write_extraction(command, image)
            return 0
        if name == "frozen_gate":
            _argument(command, "--out").write_text(
                json.dumps(_frozen_gate()), encoding="utf-8"
            )
            return 0
        if name == "train_projection":
            if scenario in {
                "training_failure",
                "training_failure_fallback_failure",
            }:
                return 8
            training_dir = _argument(command, "--out-dir")
            training_dir.mkdir(parents=True, exist_ok=True)
            (training_dir / "config.json").write_text("{}\n", encoding="utf-8")
            (training_dir / "provenance.json").write_text("{}\n", encoding="utf-8")
            (training_dir / "checkpoint_latest.pt").write_bytes(b"checkpoint")
            (training_dir / "checkpoint_ep01.pt").write_bytes(b"checkpoint")
            return 0
        if name == "extract_ep01":
            if scenario == "extraction_failure":
                return 9
            _write_extraction(command, image)
            return 0
        if name == "gate_ep01":
            _argument(command, "--out").write_text(
                json.dumps(
                    _adapted_gate(eligible=scenario != "quality_rejection")
                ),
                encoding="utf-8",
            )
            return 0
        if name == "group_adapted" and scenario == "adapted_grouping_failure":
            return 10
        if (
            name == "group_frozen_operational_fallback"
            and scenario == "training_failure_fallback_failure"
        ):
            return 11
        if name.startswith("group_"):
            grouping_dir = _argument(command, "--out-dir")
            grouping_dir.mkdir(parents=True, exist_ok=True)
            (grouping_dir / "clusters.csv").write_text(
                "path,group_id\nopaque.png,0\n", encoding="utf-8"
            )
            (grouping_dir / "summary.json").write_text("{}\n", encoding="utf-8")
            return 0
        raise AssertionError(f"unexpected command: {name}")

    monkeypatch.setattr(adaptation, "run_logged", fake_run_logged)
    argv = [
        "--pool", str(pool),
        "--out-dir", str(out_dir),
        "--epochs", "1",
        "--rungs", "1",
        "--batch", "2",
    ]
    return argv, calls, out_dir


def test_training_failure_produces_distinct_frozen_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv, calls, out_dir = _prepare(
        tmp_path, monkeypatch, scenario="training_failure"
    )

    assert adaptation.main(argv) == adaptation.OPERATIONAL_FALLBACK_EXIT

    status = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    assert status["adaptation_error"]["stage"] == "train_projection"
    assert status["frozen_fallback"]["succeeded"] is True
    assert status["final"]["workflow_action"] == "adaptation_error_frozen_fallback"
    assert status["final"]["adapted_succeeded"] is False
    assert status["final"]["frozen_fallback_succeeded"] is True
    assert status["final"]["exit_code"] == 31
    assert "group_frozen_operational_fallback" in calls


def test_candidate_extraction_failure_uses_operational_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv, _, out_dir = _prepare(
        tmp_path, monkeypatch, scenario="extraction_failure"
    )

    assert adaptation.main(argv) == adaptation.OPERATIONAL_FALLBACK_EXIT

    status = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    assert status["adaptation_error"]["stage"] == "adapted_candidate_pipeline"
    assert status["adaptation_error"]["failures"][0]["stage"] == "adapted_extraction"
    assert status["frozen_fallback"]["succeeded"] is True


def test_adapted_grouping_failure_falls_back_but_remains_abnormal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv, _, out_dir = _prepare(
        tmp_path, monkeypatch, scenario="adapted_grouping_failure"
    )

    assert adaptation.main(argv) == adaptation.OPERATIONAL_FALLBACK_EXIT

    status = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    assert status["adaptation_error"]["stage"] == "adapted_grouping"
    assert status["final"]["selected_mode"] == "frozen"
    assert status["final"]["adapted_succeeded"] is False
    assert status["final"]["frozen_fallback_succeeded"] is True


def test_quality_rejection_preserves_exit30_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv, calls, out_dir = _prepare(
        tmp_path, monkeypatch, scenario="quality_rejection"
    )

    assert adaptation.main(argv) == 30

    status = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    assert status["selection"]["workflow_action"] == "pseudo_tapt_review_required"
    assert status["final"]["exit_code"] == 30
    assert "adaptation_error" not in status
    assert "group_frozen_fallback" in calls


def test_frozen_extraction_failure_does_not_claim_fallback_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv, calls, out_dir = _prepare(
        tmp_path, monkeypatch, scenario="frozen_extract_failure"
    )

    with pytest.raises(adaptation.WorkflowError):
        adaptation.main(argv)

    status = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    assert status["selection"] is None
    assert status["final"] is None
    assert not (out_dir / "selection.json").exists()
    assert not any(name.startswith("group_") for name in calls)


def test_failed_frozen_fallback_remains_a_real_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv, _, out_dir = _prepare(
        tmp_path,
        monkeypatch,
        scenario="training_failure_fallback_failure",
    )

    with pytest.raises(adaptation.WorkflowError):
        adaptation.main(argv)

    status = json.loads((out_dir / "status.json").read_text(encoding="utf-8"))
    assert status["frozen_fallback"]["succeeded"] is False
    assert status["final"]["workflow_action"] == "workflow_failed"
    assert status["final"]["frozen_fallback_succeeded"] is False
    assert status["final"]["exit_code"] == 2
    assert not (out_dir / "selection.json").exists()
