from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts import run_fcmae_adapter_temperature_screen as temperature


def _configure(monkeypatch, tmp_path: Path) -> dict[str, str]:
    source = tmp_path / "source.pt"
    train_manifest = tmp_path / "train_manifest.json"
    eval_manifest = tmp_path / "eval_manifest.json"
    source.write_bytes(b"baseline-source")
    train_manifest.write_text('{"train": true}\n', encoding="utf-8")
    eval_manifest.write_text('{"eval": true}\n', encoding="utf-8")
    monkeypatch.setattr(temperature, "SOURCE_CHECKPOINT", source)
    monkeypatch.setattr(temperature, "EMBEDDINGS", tmp_path / "embeddings")
    monkeypatch.setattr(temperature, "RUN_LOG", tmp_path / "train.log")
    return {
        "protocol_id": "temperature-resume-test",
        "scorer_bundle_sha256": "scorer-bundle",
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": temperature.sha256_file(train_manifest),
        "eval_dir": str(tmp_path / "eval"),
        "eval_manifest": str(eval_manifest),
        "eval_manifest_sha256": "eval-content-manifest",
    }


def _write_valid_checkpoint(path: Path, *, gstep: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": {"weight": torch.zeros(1)},
            "opt": {},
            "queue": torch.zeros(4, 128),
            "qptr": 0,
            "gstep": gstep,
            "center": torch.zeros(1, 1),
            "target": None,
        },
        path,
    )


def test_progress_sidecar_precedes_fresh_launch_and_binds_exact_provenance(
    monkeypatch, tmp_path: Path
) -> None:
    context = _configure(monkeypatch, tmp_path)
    value = 0.07
    command = temperature.training_command(value)

    launch, contract, mode = temperature.prepare_non_control_launch(
        value, command, context, clean_stale=False
    )

    assert mode == "fresh"
    assert launch == command
    assert "--fresh" in launch
    sidecar = temperature.progress_sidecar_path(value)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["state"] == "launch_prepared"
    assert payload["contract"] == contract
    assert contract["command"] == command
    assert contract["command_sha256"] == temperature.sha256_json(command)
    assert contract["source_sha256"] == temperature.sha256_file(
        Path(temperature.__file__).resolve()
    )
    assert contract["trainer_sha256"] == temperature.sha256_file(
        temperature.TRAINER
    )
    assert contract["protocol_source_sha256"] == temperature.sha256_file(
        Path(temperature.protocol.__file__).resolve()
    )
    assert contract["train_manifest_file_sha256"] == temperature.sha256_file(
        Path(context["train_manifest"])
    )
    assert contract["eval_manifest_file_sha256"] == temperature.sha256_file(
        Path(context["eval_manifest"])
    )
    assert not list(sidecar.parent.glob(f".{sidecar.name}.*.tmp"))


def test_matching_progress_and_valid_checkpoint_resume_without_fresh(
    monkeypatch, tmp_path: Path
) -> None:
    context = _configure(monkeypatch, tmp_path)
    value = 0.07
    command = temperature.training_command(value)
    _, contract, launch_mode = temperature.prepare_non_control_launch(
        value, command, context, clean_stale=False
    )
    _write_valid_checkpoint(temperature.checkpoint_path(value), gstep=1600)
    valid, reason, metadata = temperature.checkpoint_validation(value)
    assert valid, reason
    temperature._update_progress_after_launch(
        value,
        contract,
        state="failed",
        launch_mode=launch_mode,
        exit_code=1,
        checkpoint_after=metadata,
    )

    launch, _, mode = temperature.prepare_non_control_launch(
        value, command, context, clean_stale=False
    )

    assert mode == "resume"
    assert "--fresh" not in launch
    assert launch == temperature.resume_command(command)
    payload = json.loads(
        temperature.progress_sidecar_path(value).read_text(encoding="utf-8")
    )
    assert payload["last_launch"]["launch_mode"] == "resume"
    assert payload["last_launch"]["checkpoint"]["gstep"] == 1600


def test_unbound_checkpoint_cannot_resume(
    monkeypatch, tmp_path: Path
) -> None:
    context = _configure(monkeypatch, tmp_path)
    value = 0.07
    command = temperature.training_command(value)
    temperature.prepare_non_control_launch(
        value, command, context, clean_stale=False
    )
    checkpoint = temperature.checkpoint_path(value)
    _write_valid_checkpoint(checkpoint, gstep=1600)

    with pytest.raises(
        temperature.StaleProgressError,
        match="not bound to a recorded failed launch",
    ):
        temperature.prepare_non_control_launch(
            value, command, context, clean_stale=False
        )

    assert checkpoint.is_file()


def test_progress_mismatch_fails_closed_and_preserves_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    context = _configure(monkeypatch, tmp_path)
    value = 0.07
    command = temperature.training_command(value)
    temperature.prepare_non_control_launch(
        value, command, context, clean_stale=False
    )
    checkpoint = temperature.checkpoint_path(value)
    _write_valid_checkpoint(checkpoint)
    sidecar = temperature.progress_sidecar_path(value)
    checkpoint_before = checkpoint.read_bytes()
    sidecar_before = sidecar.read_bytes()
    changed_context = dict(context)
    changed_context["train_manifest_sha256"] = "changed"

    with pytest.raises(temperature.StaleProgressError, match="mismatch"):
        temperature.prepare_non_control_launch(
            value, command, changed_context, clean_stale=False
        )

    assert checkpoint.read_bytes() == checkpoint_before
    assert sidecar.read_bytes() == sidecar_before


def test_invalid_checkpoint_fails_closed_until_explicit_clean_fresh(
    monkeypatch, tmp_path: Path
) -> None:
    context = _configure(monkeypatch, tmp_path)
    value = 0.07
    command = temperature.training_command(value)
    temperature.prepare_non_control_launch(
        value, command, context, clean_stale=False
    )
    checkpoint = temperature.checkpoint_path(value)
    checkpoint.write_bytes(b"truncated")

    with pytest.raises(temperature.StaleProgressError, match="checkpoint is invalid"):
        temperature.prepare_non_control_launch(
            value, command, context, clean_stale=False
        )
    assert checkpoint.read_bytes() == b"truncated"

    launch, _, mode = temperature.prepare_non_control_launch(
        value, command, context, clean_stale=True
    )
    assert mode == "fresh"
    assert "--fresh" in launch
    assert not checkpoint.exists()


def test_reusable_completed_embedding_provenance_is_not_rewritten(
    monkeypatch, tmp_path: Path
) -> None:
    context = _configure(monkeypatch, tmp_path)
    value = 0.07
    command = temperature.training_command(value)
    _, contract, launch_mode = temperature.prepare_non_control_launch(
        value, command, context, clean_stale=False
    )
    checkpoint = temperature.checkpoint_path(value)
    embedding = temperature.embedding_path(value)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    _write_valid_checkpoint(checkpoint, gstep=temperature.EXPECTED_GSTEP)
    embedding.write_bytes(b"completed-embedding")
    valid, reason, metadata = temperature.checkpoint_validation(
        value, require_complete=True
    )
    assert valid, reason
    temperature._update_progress_after_launch(
        value,
        contract,
        state="child_completed",
        launch_mode=launch_mode,
        exit_code=0,
        checkpoint_after=metadata,
    )
    temperature.write_embedding_sidecar(value, context, contract)
    temperature._update_progress_after_launch(
        value,
        contract,
        state="completed",
        launch_mode=launch_mode,
        exit_code=0,
        checkpoint_after=metadata,
    )
    sidecar = temperature.embedding_sidecar_path(value)
    before = sidecar.read_bytes()

    def _unexpected_launch(*_args, **_kwargs):
        raise AssertionError("completed embedding must skip the trainer")

    monkeypatch.setattr(temperature.subprocess, "Popen", _unexpected_launch)
    temperature.run_training(
        value,
        temperature.training_command(value),
        context,
        clean_stale=False,
    )

    assert sidecar.read_bytes() == before
    progress = json.loads(
        temperature.progress_sidecar_path(value).read_text(encoding="utf-8")
    )
    assert progress["state"] == "completed"


def test_completed_checkpoint_requires_exact_fixed_gstep(
    monkeypatch, tmp_path: Path
) -> None:
    _configure(monkeypatch, tmp_path)
    value = 0.07
    _write_valid_checkpoint(
        temperature.checkpoint_path(value),
        gstep=temperature.EXPECTED_GSTEP - 1,
    )

    valid, reason, metadata = temperature.checkpoint_validation(
        value, require_complete=True
    )

    assert not valid
    assert reason == f"checkpoint gstep is not {temperature.EXPECTED_GSTEP}"
    assert metadata == {}


def test_source_change_during_launch_prevents_embedding_sidecar(
    monkeypatch, tmp_path: Path
) -> None:
    context = _configure(monkeypatch, tmp_path)
    value = 0.07
    command = temperature.training_command(value)
    launch_contract = temperature.progress_contract(value, command, context)
    embedding = temperature.embedding_path(value)
    checkpoint = temperature.checkpoint_path(value)
    embedding.parent.mkdir(parents=True, exist_ok=True)
    embedding.write_bytes(b"embedding")
    _write_valid_checkpoint(checkpoint, gstep=temperature.EXPECTED_GSTEP)
    changed_contract = dict(launch_contract)
    changed_contract["source_sha256"] = "changed"

    with pytest.raises(
        temperature.StaleProgressError,
        match="source provenance changed",
    ):
        temperature.write_embedding_sidecar(value, context, changed_contract)

    assert not temperature.embedding_sidecar_path(value).exists()
