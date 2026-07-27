from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts import run_fcmae_adapter_residual_scale_screen as residual
from scripts import run_fcmae_adapter_temperature_screen as temperature


REPO = Path(__file__).resolve().parents[1]


def _context(tmp_path: Path) -> dict[str, str]:
    train_manifest = tmp_path / "train_manifest.json"
    eval_manifest = tmp_path / "eval_manifest.json"
    train_manifest.write_text('{"train": true}\n', encoding="utf-8")
    eval_manifest.write_text('{"eval": true}\n', encoding="utf-8")
    return {
        "protocol_id": "test-protocol",
        "scorer_bundle_sha256": "scorer-bundle",
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": residual.sha256_file(train_manifest),
        "eval_dir": str(tmp_path / "eval"),
        "eval_manifest": str(eval_manifest),
        "eval_manifest_sha256": "eval-content-manifest",
    }


def _configure_residual(monkeypatch, tmp_path: Path) -> tuple[Path, dict[str, str]]:
    source = tmp_path / "source.pt"
    source.write_bytes(b"source-checkpoint")
    embeddings = tmp_path / "residual_embeddings"
    output = tmp_path / "residual_result.json"
    monkeypatch.setattr(residual, "SOURCE_CHECKPOINT", source)
    monkeypatch.setattr(residual, "EMBEDDINGS", embeddings)
    monkeypatch.setattr(residual, "OUTPUT_JSON", output)
    return source, _context(tmp_path)


def _configure_temperature(
    monkeypatch, tmp_path: Path
) -> tuple[Path, dict[str, str]]:
    source = tmp_path / "temperature_source.pt"
    source.write_bytes(b"temperature-source")
    monkeypatch.setattr(temperature, "SOURCE_CHECKPOINT", source)
    monkeypatch.setattr(temperature, "EMBEDDINGS", tmp_path / "temperature_embeddings")
    return source, _context(tmp_path)


def _write_temperature_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": {"weight": torch.zeros(1)},
            "opt": {},
            "queue": torch.zeros(4, 128),
            "qptr": 0,
            "gstep": temperature.EXPECTED_GSTEP,
            "center": torch.zeros(1, 1),
            "target": None,
        },
        path,
    )


def test_residual_sidecar_binds_alpha_checkpoint_and_eval(
    monkeypatch, tmp_path: Path
) -> None:
    source, context = _configure_residual(monkeypatch, tmp_path)
    alpha = 0.25
    checkpoint = residual.checkpoint_path(alpha)
    embedding = residual.embedding_path(alpha)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"derived-checkpoint")
    embedding.write_bytes(b"embedding")
    source_hash = residual.sha256_file(source)

    assert not residual.embedding_is_reusable(alpha, source_hash, context)
    residual.write_embedding_sidecar(alpha, source_hash, context)
    assert residual.embedding_is_reusable(alpha, source_hash, context)

    sidecar = residual.embedding_sidecar_path(alpha)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["contract"]["alpha"] = "0.50"
    residual.atomic_write_json(sidecar, payload)
    assert not residual.embedding_is_reusable(alpha, source_hash, context)

    residual.write_embedding_sidecar(alpha, source_hash, context)
    checkpoint.write_bytes(b"changed-checkpoint")
    assert not residual.embedding_is_reusable(alpha, source_hash, context)

    residual.write_embedding_sidecar(alpha, source_hash, context)
    changed_context = dict(context)
    changed_context["eval_manifest_sha256"] = "changed-eval-manifest"
    assert not residual.embedding_is_reusable(alpha, source_hash, changed_context)


def test_residual_alpha_one_uses_local_extraction_artifact(
    monkeypatch, tmp_path: Path
) -> None:
    source, _ = _configure_residual(monkeypatch, tmp_path)

    assert residual.embedding_path(1.0).parent == residual.EMBEDDINGS
    assert residual.embedding_path(1.0) != source.with_name("fcmae_ad1_s1_ep4.npy")


def test_residual_result_validator_rejects_stale_json(
    monkeypatch, tmp_path: Path
) -> None:
    source, context = _configure_residual(monkeypatch, tmp_path)
    monkeypatch.setattr(residual, "ALPHAS", (0.25,))
    alpha = 0.25
    source_hash = residual.sha256_file(source)
    checkpoint = residual.checkpoint_path(alpha)
    embedding = residual.embedding_path(alpha)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    embedding.write_bytes(b"embedding")
    residual.write_embedding_sidecar(alpha, source_hash, context)
    sidecar = residual.embedding_sidecar_path(alpha)
    embedding_hash = residual.sha256_file(embedding)
    payload = {
        "gate": {"values": [{"alpha": alpha}]},
        "rows": [
            {"recipe": "frozen"},
            {"recipe": "frozen"},
            {
                "recipe": "residual_scale_0.25",
                "embedding_sha256": embedding_hash,
            },
            {
                "recipe": "residual_scale_0.25",
                "embedding_sha256": embedding_hash,
            },
        ],
        "provenance": {
            "script_sha256": residual.sha256_file(Path(residual.__file__).resolve()),
            "source_checkpoint_sha256": source_hash,
            "trainer_sha256": residual.sha256_file(residual.TRAINER),
            "protocol_source_sha256": residual.sha256_file(
                Path(residual.protocol.__file__).resolve()
            ),
            "protocol_id": context["protocol_id"],
            "scorer_bundle_sha256": context["scorer_bundle_sha256"],
            "eval_manifest": context["eval_manifest"],
            "eval_manifest_sha256": context["eval_manifest_sha256"],
            "eval_manifest_file_sha256": residual.sha256_file(
                Path(context["eval_manifest"])
            ),
            "embedding_sidecars": {
                "0.25": {
                    "path": str(sidecar),
                    "sha256": residual.sha256_file(sidecar),
                }
            },
        },
    }
    residual.atomic_write_json(residual.OUTPUT_JSON, payload)

    valid, reason = residual.validate_result(source_hash, context)
    assert valid, reason

    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar_payload["contract"]["alpha"] = "0.50"
    residual.atomic_write_json(sidecar, sidecar_payload)
    valid, reason = residual.validate_result(source_hash, context)
    assert not valid
    assert "embedding provenance" in reason

    residual.write_embedding_sidecar(alpha, source_hash, context)
    payload["provenance"]["embedding_sidecars"]["0.25"]["sha256"] = (
        residual.sha256_file(sidecar)
    )
    payload["provenance"]["script_sha256"] = "stale-runner"
    residual.atomic_write_json(residual.OUTPUT_JSON, payload)
    valid, reason = residual.validate_result(source_hash, context)
    assert not valid
    assert "script_sha256" in reason


def test_temperature_sidecar_binds_exact_value_and_source_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    source, context = _configure_temperature(monkeypatch, tmp_path)
    value = 0.05
    checkpoint = temperature.checkpoint_path(value)
    embedding = temperature.embedding_path(value)
    checkpoint.parent.mkdir(parents=True)
    _write_temperature_checkpoint(checkpoint)
    embedding.write_bytes(b"temperature-embedding")
    launch_contract = temperature.progress_contract(
        value, temperature.training_command(value), context
    )

    assert not temperature.embedding_is_reusable(value, context)
    temperature.write_embedding_sidecar(value, context, launch_contract)
    assert temperature.embedding_is_reusable(value, context)

    sidecar = temperature.embedding_sidecar_path(value)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["contract"]["temperature"] = "0.07"
    temperature.atomic_write_json(sidecar, payload)
    assert not temperature.embedding_is_reusable(value, context)

    temperature.write_embedding_sidecar(value, context, launch_contract)
    source.write_bytes(b"changed-temperature-source")
    assert not temperature.embedding_is_reusable(value, context)


def test_temperature_new_value_requires_matching_checkpoint_sidecar(
    monkeypatch, tmp_path: Path
) -> None:
    _, context = _configure_temperature(monkeypatch, tmp_path)
    value = 0.07
    checkpoint = temperature.checkpoint_path(value)
    embedding = temperature.embedding_path(value)
    command = temperature.training_command(value)
    _, launch_contract, launch_mode = temperature.prepare_non_control_launch(
        value, command, context, clean_stale=False
    )
    _write_temperature_checkpoint(checkpoint)
    embedding.write_bytes(b"trained-embedding")
    valid, reason, metadata = temperature.checkpoint_validation(
        value, require_complete=True
    )
    assert valid, reason
    temperature._update_progress_after_launch(
        value,
        launch_contract,
        state="child_completed",
        launch_mode=launch_mode,
        exit_code=0,
        checkpoint_after=metadata,
    )
    temperature.write_embedding_sidecar(value, context, launch_contract)
    temperature._update_progress_after_launch(
        value,
        launch_contract,
        state="completed",
        launch_mode=launch_mode,
        exit_code=0,
        checkpoint_after=metadata,
    )
    assert temperature.embedding_is_reusable(value, context)

    _write_temperature_checkpoint(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["model"]["weight"] = torch.ones(1)
    torch.save(payload, checkpoint)
    assert not temperature.embedding_is_reusable(value, context)


def test_wrapper_requires_current_python_result_validation() -> None:
    wrapper = (
        REPO / "scripts/run_fcmae_adapter_residual_scale_after_holdout.ps1"
    ).read_text(encoding="utf-8")

    assert "& $Python -u $Runner --validate-result" in wrapper
