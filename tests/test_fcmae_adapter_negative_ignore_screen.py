from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_fcmae_adapter_negative_ignore_screen as runner


def _context(tmp_path: Path) -> dict[str, str]:
    train_manifest = tmp_path / "train_manifest.json"
    eval_manifest = tmp_path / "eval_manifest.json"
    train_manifest.write_text('{"train": true}\n', encoding="utf-8")
    eval_manifest.write_text('{"eval": true}\n', encoding="utf-8")
    return {
        "protocol_id": "test-protocol",
        "scorer_bundle_sha256": "scorer-bundle",
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": runner.sha256_file(train_manifest),
        "eval_dir": str(tmp_path / "eval"),
        "eval_manifest": str(eval_manifest),
        "eval_manifest_sha256": "eval-content-manifest",
    }


def _configure_paths(monkeypatch, tmp_path: Path) -> dict[str, str]:
    monkeypatch.setattr(runner, "EMBEDDINGS", tmp_path / "embeddings")
    monkeypatch.setattr(runner, "RUN_ROOT", tmp_path / "run")
    monkeypatch.setattr(runner, "OUTPUT_JSON", tmp_path / "result.json")
    return _context(tmp_path)


def _option(command: list[str], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


def test_condition_set_and_commands_deliver_exact_trainer_overrides() -> None:
    assert [
        (item["key"], item["value"]) for item in runner.SCREEN_CONDITIONS
    ] == [
        ("off", 1.01),
        ("neg080", 0.80),
        ("neg075", 0.75),
        ("neg070", 0.70),
    ]
    assert runner.CONTROL_CONDITION == {
        "key": "b4_control_072",
        "value": 0.72,
        "label": "0.72",
    }

    commands = runner.all_commands()
    axis = runner.validate_axis_commands(commands)
    assert axis["only_variable"] == "--ignore"
    for item in runner.ALL_CONDITIONS:
        command = commands[item["key"]]
        assert command.count("--ignore") == 1
        assert float(_option(command, "--ignore")) == item["value"]
        assert command.count("--use-queue") == 1
        assert _option(command, "--queue-size") == "4096"
        assert _option(command, "--temp") == "0.05"
        assert command.count("--fresh") == 1


def test_command_validation_fails_closed_on_missing_duplicate_or_aliased_value() -> None:
    command = runner.training_command("neg080")

    missing = list(command)
    index = missing.index("--ignore")
    del missing[index : index + 2]
    with pytest.raises(RuntimeError, match="--ignore must occur exactly once"):
        runner.validate_training_command("neg080", missing)

    duplicate = [*command, "--ignore", "0.80"]
    with pytest.raises(RuntimeError, match="--ignore must occur exactly once"):
        runner.validate_training_command("neg080", duplicate)

    silently_defaulted = list(command)
    silently_defaulted[silently_defaulted.index("--ignore") + 1] = "0.72"
    with pytest.raises(RuntimeError, match="NEG override mismatch"):
        runner.validate_training_command("neg080", silently_defaulted)

    off_activated = runner.training_command("off")
    off_activated[off_activated.index("--ignore") + 1] = "0.99"
    with pytest.raises(RuntimeError, match="NEG override mismatch"):
        runner.validate_training_command("off", off_activated)


def test_axis_validation_rejects_any_second_variable() -> None:
    commands = runner.all_commands()
    changed = {key: list(value) for key, value in commands.items()}
    command = changed["neg075"]
    command[command.index("--temp") + 1] = "0.07"
    with pytest.raises(RuntimeError, match="--temp mismatch"):
        runner.validate_axis_commands(changed)


def test_actual_trainer_and_historical_b4_contracts_fail_closed(
    tmp_path: Path,
) -> None:
    trainer_contract = runner.trainer_negative_ignore_contract()
    assert trainer_contract["explicit_off_value"] == 1.01
    assert trainer_contract["active_when"] == "value < 1.0"
    assert trainer_contract["sha256"] == runner.sha256_file(runner.TRAINER)

    changed_trainer = tmp_path / "_ssl_methods_changed.py"
    source = runner.TRAINER.read_text(encoding="utf-8")
    assert "qneg * t > args.ignore" in source
    changed_trainer.write_text(
        source.replace(
            "qneg * t > args.ignore",
            "qneg * t > 0.72",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="trainer NEG implementation changed"):
        runner.trainer_negative_ignore_contract(changed_trainer)

    b4 = runner.b4_control_reference_contract()
    assert b4["queue_enabled"] is True
    assert b4["negative_ignore"] == 0.72
    assert b4["sha256"] == runner.sha256_file(runner.B4_REFERENCE_SOURCE)

    changed_b4 = tmp_path / "changed_b4.py"
    b4_source = runner.B4_REFERENCE_SOURCE.read_text(encoding="utf-8")
    needle = (
        '"B4": {"local": True, "local_weight": 1.0, '
        '"queue": True, "ignore": 0.72, "neco": 0.0}'
    )
    assert needle in b4_source
    changed_b4.write_text(
        b4_source.replace(needle, needle.replace("0.72", "0.71"), 1),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="historical B4"):
        runner.b4_control_reference_contract(changed_b4)


def test_embedding_sidecar_rejects_stale_value_checkpoint_and_eval(
    monkeypatch,
    tmp_path: Path,
) -> None:
    context = _configure_paths(monkeypatch, tmp_path)
    commands = runner.all_commands()
    key = "neg075"
    checkpoint = runner.checkpoint_path(key)
    embedding = runner.embedding_path(key)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    embedding.write_bytes(b"embedding")

    assert not runner.embedding_is_reusable(key, context, commands)
    runner.write_embedding_sidecar(key, context, commands)
    assert runner.embedding_is_reusable(key, context, commands)

    sidecar = runner.embedding_sidecar_path(key)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["contract"]["trainer_ignore_value"] = "0.72"
    runner.atomic_write_json(sidecar, payload)
    assert not runner.embedding_is_reusable(key, context, commands)

    runner.write_embedding_sidecar(key, context, commands)
    checkpoint.write_bytes(b"changed-checkpoint")
    assert not runner.embedding_is_reusable(key, context, commands)

    runner.write_embedding_sidecar(key, context, commands)
    changed_context = dict(context)
    changed_context["eval_manifest_sha256"] = "changed-eval"
    assert not runner.embedding_is_reusable(key, changed_context, commands)


def _row(
    recipe: str,
    clusterer: str,
    *,
    p1: int,
    p2: float,
    p3: float,
    p4: float,
    ari: float,
    ami: float,
) -> dict[str, object]:
    return {
        "recipe": recipe,
        "clusterer": clusterer,
        "P1_capture_count": p1,
        "P2_noise_pct": p2,
        "P3_completeness": p3,
        "P4_homogeneity": p4,
        "ARI_supporting": ari,
        "AMI_supporting": ami,
    }


def test_screen_uses_p1_p2_p3_p4_and_ignores_ari_ami() -> None:
    rows: list[dict[str, object]] = []
    for clusterer in ("FINCH-p2", "Louvain-res6"):
        rows.append(
            _row(
                "frozen",
                clusterer,
                p1=30,
                p2=0.0,
                p3=0.80,
                p4=0.90,
                ari=0.99,
                ami=0.99,
            )
        )
        rows.append(
            _row(
                runner.recipe_name("b4_control_072"),
                clusterer,
                p1=30,
                p2=0.0,
                p3=0.81,
                p4=0.91,
                ari=0.99,
                ami=0.99,
            )
        )
        for item in runner.SCREEN_CONDITIONS:
            key = item["key"]
            winner = key == "neg075"
            rows.append(
                _row(
                    runner.recipe_name(key),
                    clusterer,
                    p1=30 if winner else 29,
                    p2=0.0,
                    p3=0.82 if winner else 0.83,
                    p4=0.92 if winner else 0.93,
                    ari=-100.0 if winner else 1.0,
                    ami=-100.0 if winner else 1.0,
                )
            )

    result = runner.screen(rows)
    assert result["proposed_condition"] == "neg075"
    assert result["proposed_negative_ignore"] == "0.75"
    winner = next(
        item for item in result["values"] if item["condition"] == "neg075"
    )
    assert winner["accepted"] is True
    assert "ARI" not in json.dumps(winner)
    assert "AMI" not in json.dumps(winner)


def test_result_validator_rejects_stale_provenance(
    monkeypatch,
    tmp_path: Path,
) -> None:
    context = _configure_paths(monkeypatch, tmp_path)
    commands = runner.all_commands()
    rows: list[dict[str, object]] = [
        {"recipe": "frozen"},
        {"recipe": "frozen"},
    ]
    for item in runner.ALL_CONDITIONS:
        key = item["key"]
        checkpoint = runner.checkpoint_path(key)
        embedding = runner.embedding_path(key)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{key}".encode())
        embedding.write_bytes(f"embedding-{key}".encode())
        runner.write_embedding_sidecar(key, context, commands)
        embedding_hash = runner.sha256_file(embedding)
        rows.extend(
            [
                {
                    "recipe": runner.recipe_name(key),
                    "embedding_sha256": embedding_hash,
                },
                {
                    "recipe": runner.recipe_name(key),
                    "embedding_sha256": embedding_hash,
                },
            ]
        )
    payload = {
        "screen": {
            "values": [
                {"condition": item["key"]}
                for item in runner.SCREEN_CONDITIONS
            ]
        },
        "rows": rows,
        "provenance": runner.result_provenance(context, commands),
    }
    runner.atomic_write_json(runner.OUTPUT_JSON, payload)

    valid, reason = runner.validate_result(context, commands)
    assert valid, reason

    payload["provenance"]["axis_contract"]["common_command_sha256"] = "stale"
    runner.atomic_write_json(runner.OUTPUT_JSON, payload)
    valid, reason = runner.validate_result(context, commands)
    assert not valid
    assert "provenance is stale" in reason
