from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts import run_fcmae_adapter_queue_screen as queue_screen


def _context(tmp_path: Path) -> dict[str, str]:
    train_manifest = tmp_path / "train_manifest.json"
    eval_manifest = tmp_path / "eval_manifest.json"
    train_manifest.write_text('{"train": true}\n', encoding="utf-8")
    eval_manifest.write_text('{"eval": true}\n', encoding="utf-8")
    return {
        "protocol_id": "test-protocol",
        "scorer_bundle_sha256": "scorer-bundle",
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": queue_screen.sha256_file(train_manifest),
        "eval_dir": str(tmp_path / "eval"),
        "eval_manifest": str(eval_manifest),
        "eval_manifest_sha256": "eval-content-manifest",
    }


def _configure(
    monkeypatch: Any, tmp_path: Path
) -> tuple[Path, dict[str, str]]:
    source = tmp_path / "source.pt"
    trainer = tmp_path / "trainer.py"
    source.write_bytes(b"fixed-queue-off-checkpoint")
    trainer.write_text("# fixed trainer\n", encoding="utf-8")
    monkeypatch.setattr(queue_screen, "SOURCE_CHECKPOINT", source)
    monkeypatch.setattr(queue_screen, "TRAINER", trainer)
    monkeypatch.setattr(queue_screen, "EMBEDDINGS", tmp_path / "embeddings")
    monkeypatch.setattr(queue_screen, "OUTPUT_JSON", tmp_path / "result.json")
    return source, _context(tmp_path)


def _strip_queue_axis(command: list[str]) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(command):
        value = command[index]
        if value in {"--use-queue", "--fresh"}:
            index += 1
            continue
        if value in {"--queue-size", "--tag"}:
            index += 2
            continue
        result.append(value)
        index += 1
    return result


def _metric_row(
    recipe: str,
    clusterer: str,
    *,
    p1: int = 32,
    p2: float = 0.0,
    p3: float = 0.80,
    p4: float = 0.90,
    ari: float = 0.50,
    ami: float = 0.60,
    embedding_sha256: str = "embedding",
) -> dict[str, Any]:
    return {
        "recipe": recipe,
        "clusterer": clusterer,
        "P1_capture_count": p1,
        "P1_target_class_count": 32,
        "P1_capture": p1 / 32,
        "P2_noise_pct": p2,
        "P3_completeness": p3,
        "P4_homogeneity": p4,
        "fragment_ratio": 1.5,
        "ARI_supporting": ari,
        "AMI_supporting": ami,
        "embedding_sha256": embedding_sha256,
    }


def test_commands_change_only_queue_axis() -> None:
    commands = {
        size: queue_screen.training_command(size)
        for size in queue_screen.QUEUE_SIZES
    }
    common = _strip_queue_axis(commands[0])
    assert all(_strip_queue_axis(command) == common for command in commands.values())
    assert "--use-queue" not in commands[0]
    assert "--fresh" not in commands[0]
    for size in (1024, 2048, 4096):
        command = commands[size]
        assert "--use-queue" in command
        assert "--fresh" in command
        index = command.index("--queue-size")
        assert command[index + 1] == str(size)
    assert "--temp" in common
    assert common[common.index("--temp") + 1] == "0.05"
    assert queue_screen.fixed_recipe_contract()["adapter_inference_scale"] == "1.00"
    assert queue_screen.fixed_recipe_contract()["may_hdbscan"] == {
        "min_cluster_size": 12,
        "min_samples": 15,
        "cluster_selection_method": "leaf",
        "cluster_selection_epsilon": 0.06,
    }


def test_sidecar_binds_queue_checkpoint_commands_and_manifests(
    monkeypatch: Any, tmp_path: Path
) -> None:
    source, context = _configure(monkeypatch, tmp_path)
    for size in (0, 2048):
        checkpoint = queue_screen.checkpoint_path(size)
        embedding = queue_screen.embedding_path(size)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(
            source.read_bytes() if size == 0 else b"trained-queue-checkpoint"
        )
        embedding.write_bytes(f"embedding-{size}".encode("ascii"))
        assert not queue_screen.embedding_is_reusable(size, context)
        queue_screen.write_embedding_sidecar(size, context)
        assert queue_screen.embedding_is_reusable(size, context)

        sidecar = queue_screen.embedding_sidecar_path(size)
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        payload["contract"]["queue_size"] = 4096
        queue_screen.atomic_write_json(sidecar, payload)
        assert not queue_screen.embedding_is_reusable(size, context)

        queue_screen.write_embedding_sidecar(size, context)
        checkpoint.write_bytes(b"changed-checkpoint")
        assert not queue_screen.embedding_is_reusable(size, context)

        checkpoint.write_bytes(
            source.read_bytes() if size == 0 else b"trained-queue-checkpoint"
        )
        queue_screen.write_embedding_sidecar(size, context)
        changed_context = dict(context)
        changed_context["eval_manifest_sha256"] = "changed-eval"
        assert not queue_screen.embedding_is_reusable(size, changed_context)


def test_gate_uses_both_clusterers_and_ignores_ari_ami() -> None:
    clusterers = ("FINCH-p2", "Louvain-res6")
    rows = [
        _metric_row("frozen", clusterer)
        for clusterer in clusterers
    ]
    for clusterer in clusterers:
        rows.append(_metric_row(queue_screen.recipe_name(0), clusterer))
        rows.append(
            _metric_row(
                queue_screen.recipe_name(1024),
                clusterer,
                p3=0.82,
                p4=0.91,
                ari=-10.0,
                ami=-20.0,
            )
        )
        rows.append(
            _metric_row(
                queue_screen.recipe_name(2048),
                clusterer,
                p1=31,
                p3=0.90,
                p4=0.95,
            )
        )
        rows.append(
            _metric_row(
                queue_screen.recipe_name(4096),
                clusterer,
                p2=0.1,
                p3=0.90,
                p4=0.95,
            )
        )
    result = queue_screen.screen(rows)
    by_size = {item["queue_size"]: item for item in result["values"]}
    assert by_size[1024]["accepted"]
    assert not by_size[2048]["accepted"]
    assert not by_size[4096]["accepted"]
    assert result["proposed_queue_size"] == 1024
    assert set(by_size[1024]["clusterers"]) == set(clusterers)


def test_result_validation_fails_closed_on_stale_provenance(
    monkeypatch: Any, tmp_path: Path
) -> None:
    source, context = _configure(monkeypatch, tmp_path)
    monkeypatch.setattr(queue_screen, "QUEUE_SIZES", (0, 1024))
    rows = []
    for clusterer in ("FINCH-p2", "Louvain-res6"):
        rows.append(_metric_row("frozen", clusterer))
    sidecars: dict[str, dict[str, str]] = {}
    for size in queue_screen.QUEUE_SIZES:
        checkpoint = queue_screen.checkpoint_path(size)
        embedding = queue_screen.embedding_path(size)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(
            source.read_bytes() if size == 0 else f"checkpoint-{size}".encode()
        )
        embedding.write_bytes(f"embedding-{size}".encode())
        queue_screen.write_embedding_sidecar(size, context)
        sidecar = queue_screen.embedding_sidecar_path(size)
        sidecars[queue_screen.queue_key(size)] = {
            "path": str(sidecar),
            "sha256": queue_screen.sha256_file(sidecar),
        }
        embedding_hash = queue_screen.sha256_file(embedding)
        for clusterer in ("FINCH-p2", "Louvain-res6"):
            rows.append(
                _metric_row(
                    queue_screen.recipe_name(size),
                    clusterer,
                    embedding_sha256=embedding_hash,
                )
            )
    commands = {
        queue_screen.queue_key(size): queue_screen.training_command(size)
        for size in queue_screen.QUEUE_SIZES
    }
    fixed = queue_screen.fixed_recipe_contract()
    payload = {
        "screen": {
            "values": [
                {"queue": queue_screen.queue_key(size), "queue_size": size}
                for size in queue_screen.QUEUE_SIZES
            ]
        },
        "rows": rows,
        "provenance": {
            "script_sha256": queue_screen.sha256_file(
                Path(queue_screen.__file__).resolve()
            ),
            "trainer_sha256": queue_screen.sha256_file(queue_screen.TRAINER),
            "protocol_source_sha256": queue_screen.sha256_file(
                Path(queue_screen.protocol.__file__).resolve()
            ),
            "baseline_source_checkpoint_sha256": queue_screen.sha256_file(
                source
            ),
            "fixed_recipe": fixed,
            "fixed_recipe_sha256": queue_screen.sha256_json(fixed),
            "may_hdbscan": queue_screen.MAY_HDBSCAN,
            "commands": commands,
            "commands_sha256": queue_screen.sha256_json(commands),
            "protocol_id": context["protocol_id"],
            "scorer_bundle_sha256": context["scorer_bundle_sha256"],
            "train_manifest": context["train_manifest"],
            "train_manifest_sha256": context["train_manifest_sha256"],
            "train_manifest_file_sha256": queue_screen.manifest_file_hash(
                context, "train_manifest"
            ),
            "eval_manifest": context["eval_manifest"],
            "eval_manifest_sha256": context["eval_manifest_sha256"],
            "eval_manifest_file_sha256": queue_screen.manifest_file_hash(
                context, "eval_manifest"
            ),
            "embedding_sidecars": sidecars,
        },
    }
    queue_screen.atomic_write_json(queue_screen.OUTPUT_JSON, payload)
    valid, reason = queue_screen.validate_result(context)
    assert valid, reason

    payload["provenance"]["script_sha256"] = "stale-runner"
    queue_screen.atomic_write_json(queue_screen.OUTPUT_JSON, payload)
    valid, reason = queue_screen.validate_result(context)
    assert not valid
    assert "script_sha256" in reason

    payload["provenance"]["script_sha256"] = queue_screen.sha256_file(
        Path(queue_screen.__file__).resolve()
    )
    queue_screen.atomic_write_json(queue_screen.OUTPUT_JSON, payload)
    sidecar = queue_screen.embedding_sidecar_path(1024)
    sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
    sidecar_payload["contract"]["fixed_recipe_sha256"] = "stale-recipe"
    queue_screen.atomic_write_json(sidecar, sidecar_payload)
    valid, reason = queue_screen.validate_result(context)
    assert not valid
    assert "embedding provenance" in reason
