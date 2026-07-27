from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_fcmae_adapter_queue_size_screen as queue_screen


def _context(tmp_path: Path) -> dict[str, str]:
    train_manifest = tmp_path / "train_manifest.json"
    queue_eval_manifest = tmp_path / "queue_eval_manifest.json"
    train_manifest.write_text('{"train": true}\n', encoding="utf-8")
    queue_eval_manifest.write_text('{"eval": true}\n', encoding="utf-8")
    return {
        "protocol_id": "queue-test-protocol",
        "scorer_bundle_sha256": "scorer-bundle",
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": queue_screen.sha256_file(train_manifest),
        "eval_dir": str(tmp_path / "eval"),
        "eval_manifest": str(queue_eval_manifest),
        "eval_manifest_sha256": "eval-content-manifest",
    }


def _configure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[dict[str, str], Path]:
    runner = tmp_path / "temperature_runner.py"
    trainer = tmp_path / "trainer.py"
    result = tmp_path / "temperature_result.json"
    runner.write_text("# temperature runner\n", encoding="utf-8")
    trainer.write_text("# trainer\n", encoding="utf-8")
    monkeypatch.setattr(queue_screen, "TEMPERATURE_RUNNER", runner)
    monkeypatch.setattr(queue_screen, "TEMPERATURE_RESULT", result)
    monkeypatch.setattr(queue_screen, "TRAINER", trainer)
    monkeypatch.setattr(queue_screen, "EMBEDDINGS", tmp_path / "queue_embeddings")
    return _context(tmp_path), result


def _write_accepted_temperature(
    tmp_path: Path,
    result: Path,
    context: dict[str, str],
    *,
    temperature: float = 0.07,
    accepted: bool = True,
) -> dict:
    checkpoint = tmp_path / "temperature_checkpoint.pt"
    embedding = tmp_path / "temperature_embedding.npy"
    sidecar = tmp_path / "temperature_embedding.npy.provenance.json"
    source_eval_manifest = tmp_path / "temperature_eval_manifest.json"
    checkpoint.write_bytes(b"temperature-checkpoint")
    embedding.write_bytes(b"temperature-embedding")
    source_eval_manifest.write_text('{"eval": true}\n', encoding="utf-8")
    key = f"{temperature:.2f}"
    sidecar_payload = {
        "contract": {
            "axis": "temperature",
            "temperature": key,
            "trainer_sha256": queue_screen.sha256_file(queue_screen.TRAINER),
            "protocol_source_sha256": queue_screen.sha256_file(
                Path(queue_screen.protocol.__file__).resolve()
            ),
            "scorer_bundle_sha256": context["scorer_bundle_sha256"],
            "train_manifest": context["train_manifest"],
            "train_manifest_sha256": context["train_manifest_sha256"],
            "eval_dir": context["eval_dir"],
            "eval_manifest_sha256": context["eval_manifest_sha256"],
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": queue_screen.sha256_file(checkpoint),
        },
        "embedding": str(embedding),
        "embedding_sha256": queue_screen.sha256_file(embedding),
    }
    queue_screen.atomic_write_json(sidecar, sidecar_payload)
    primary = {
        "P1_capture_count": 32,
        "P1_target_class_count": 32,
        "P2_noise_pct": 0.0,
        "P3_completeness": 0.9,
        "P4_homogeneity": 0.95,
    }
    value = {
        "temperature": temperature,
        "accepted": accepted,
        "minimum_P3_P4_delta": 0.0,
    }
    payload = {
        "screen": {
            "contract": {
                "ARI_AMI": "supporting only; excluded from screening"
            },
            "values": [value],
            "proposed_temperature": temperature,
        },
        "rows": [
            {
                "recipe": f"adapter_temp_{temperature:.2f}",
                "clusterer": "FINCH-p2",
                **primary,
            },
            {
                "recipe": f"adapter_temp_{temperature:.2f}",
                "clusterer": "Louvain-res6",
                **primary,
            },
        ],
        "provenance": {
            "script": str(queue_screen.TEMPERATURE_RUNNER),
            "script_sha256": queue_screen.sha256_file(
                queue_screen.TEMPERATURE_RUNNER
            ),
            "trainer": str(queue_screen.TRAINER),
            "trainer_sha256": queue_screen.sha256_file(queue_screen.TRAINER),
            "protocol_source_sha256": queue_screen.sha256_file(
                Path(queue_screen.protocol.__file__).resolve()
            ),
            "scorer_bundle_sha256": context["scorer_bundle_sha256"],
            "eval_manifest": str(source_eval_manifest),
            "eval_manifest_sha256": context["eval_manifest_sha256"],
            "eval_manifest_file_sha256": queue_screen.sha256_file(
                source_eval_manifest
            ),
            "embedding_sidecars": {
                key: {
                    "path": str(sidecar),
                    "sha256": queue_screen.sha256_file(sidecar),
                }
            },
        },
    }
    queue_screen.atomic_write_json(result, payload)
    return payload


def test_missing_or_rejected_temperature_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    context, result = _configure(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="missing"):
        queue_screen.load_accepted_temperature(context)

    _write_accepted_temperature(
        tmp_path, result, context, accepted=False
    )
    with pytest.raises(RuntimeError, match="not exactly one accepted"):
        queue_screen.load_accepted_temperature(context)


def test_accepted_temperature_requires_current_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    context, result = _configure(monkeypatch, tmp_path)
    payload = _write_accepted_temperature(tmp_path, result, context)

    temperature, evidence = queue_screen.load_accepted_temperature(context)
    assert temperature == 0.07
    assert evidence["temperature_result_sha256"] == queue_screen.sha256_file(
        result
    )

    sidecar = Path(
        payload["provenance"]["embedding_sidecars"]["0.07"]["path"]
    )
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar_payload["contract"]["temperature"] = "0.10"
    queue_screen.atomic_write_json(sidecar, sidecar_payload)
    payload["provenance"]["embedding_sidecars"]["0.07"]["sha256"] = (
        queue_screen.sha256_file(sidecar)
    )
    queue_screen.atomic_write_json(result, payload)
    with pytest.raises(RuntimeError, match="sidecar mismatch: temperature"):
        queue_screen.load_accepted_temperature(context)


def test_commands_encode_exact_queue_size_and_fixed_axis(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _configure(monkeypatch, tmp_path)
    control = queue_screen.training_command(0, 0.07)
    queued = queue_screen.training_command(2048, 0.07)

    assert control[control.index("--queue-size") + 1] == "0"
    assert "--use-queue" not in control
    assert queued[queued.index("--queue-size") + 1] == "2048"
    assert "--use-queue" in queued
    for command in (control, queued):
        assert command[command.index("--temp") + 1] == "0.07"
        assert command[command.index("--seed") + 1] == "3"
        assert command[command.index("--epochs") + 1] == "4"
        assert "--fresh" in command
        assert command[command.index("--wafer-rot-deg") + 1] == "0"
        assert command[command.index("--wafer-translate") + 1] == "0"


def test_queue_sidecar_binds_queue_checkpoint_and_temperature_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    context, result = _configure(monkeypatch, tmp_path)
    _write_accepted_temperature(tmp_path, result, context)
    temperature, evidence = queue_screen.load_accepted_temperature(context)
    queue_size = 1024
    checkpoint = queue_screen.checkpoint_path(queue_size, temperature)
    embedding = queue_screen.embedding_path(queue_size, temperature)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"queue-checkpoint")
    embedding.write_bytes(b"queue-embedding")

    assert not queue_screen.embedding_is_reusable(
        queue_size, temperature, evidence, context
    )
    queue_screen.write_embedding_sidecar(
        queue_size, temperature, evidence, context
    )
    assert queue_screen.embedding_is_reusable(
        queue_size, temperature, evidence, context
    )

    sidecar = queue_screen.embedding_sidecar_path(queue_size, temperature)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["contract"]["queue_size"] = 2048
    queue_screen.atomic_write_json(sidecar, payload)
    assert not queue_screen.embedding_is_reusable(
        queue_size, temperature, evidence, context
    )

    queue_screen.write_embedding_sidecar(
        queue_size, temperature, evidence, context
    )
    checkpoint.write_bytes(b"changed-checkpoint")
    assert not queue_screen.embedding_is_reusable(
        queue_size, temperature, evidence, context
    )

    checkpoint.write_bytes(b"queue-checkpoint")
    queue_screen.write_embedding_sidecar(
        queue_size, temperature, evidence, context
    )
    temp_payload = json.loads(result.read_text(encoding="utf-8"))
    temp_payload["created_at"] = "changed"
    queue_screen.atomic_write_json(result, temp_payload)
    _, changed_evidence = queue_screen.load_accepted_temperature(context)
    assert not queue_screen.embedding_is_reusable(
        queue_size, temperature, changed_evidence, context
    )


def _metric_row(
    recipe: str,
    clusterer: str,
    *,
    p1: int,
    p2: float,
    p3: float,
    p4: float,
    ari: float,
) -> dict:
    return {
        "recipe": recipe,
        "clusterer": clusterer,
        "P1_capture_count": p1,
        "P2_noise_pct": p2,
        "P3_completeness": p3,
        "P4_homogeneity": p4,
        "fragment_ratio": 1.0,
        "ARI": ari,
        "AMI": ari,
    }


def test_gate_uses_p1_p2_p3_p4_for_finch_and_louvain_not_ari() -> None:
    rows = []
    for clusterer in sorted(queue_screen.PRIMARY_CLUSTERERS):
        rows.append(
            _metric_row(
                "frozen",
                clusterer,
                p1=31,
                p2=1.0,
                p3=0.90,
                p4=0.95,
                ari=0.99,
            )
        )
    for queue_size in queue_screen.QUEUE_SIZES:
        for clusterer in sorted(queue_screen.PRIMARY_CLUSTERERS):
            rows.append(
                _metric_row(
                    f"adapter_queue_{queue_size}",
                    clusterer,
                    p1=30 if queue_size == 0 else 31,
                    p2=1.0,
                    p3=0.91 + queue_size / 1_000_000,
                    p4=0.96 + queue_size / 1_000_000,
                    ari=0.01,
                )
            )

    gate = queue_screen.screen(rows)

    assert gate["values"][0]["accepted"] is False
    assert all(item["accepted"] for item in gate["values"][1:])
    assert gate["proposed_queue_size"] == 4096
    assert "supporting only" in gate["contract"]["ARI_AMI"]


def test_fixed_protocol_constants() -> None:
    assert queue_screen.SEED == 3
    assert queue_screen.EPOCH == 4
    assert queue_screen.QUEUE_SIZES == (0, 1024, 2048, 4096)
    assert queue_screen.HDBSCAN_CONFIG == {
        "min_cluster_size": 12,
        "min_samples": 15,
        "cluster_selection_method": "leaf",
        "cluster_selection_epsilon": 0.06,
    }
