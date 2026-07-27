from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import run_fcmae_adapter_alpha_three_seed as runner


def _screen_payload(alpha: float | None = 0.5) -> dict:
    values = []
    for value in (0.25, 0.5, 0.75, 1.0):
        accepted = alpha is not None and value == alpha
        values.append(
            {
                "alpha": value,
                "accepted": accepted,
                "clusterers": {
                    clusterer: {
                        "P1_preserved": accepted,
                        "P2_not_worse": accepted,
                        "P3_not_worse": accepted,
                        "P4_not_worse": accepted,
                        "P3_delta": 0.01 if accepted else -0.01,
                        "P4_delta": 0.01 if accepted else -0.01,
                    }
                    for clusterer in runner.EXPECTED_CLUSTERERS
                },
            }
        )
    return {
        "gate": {"proposed_alpha": alpha, "values": values},
        "provenance": {"script_sha256": "screen-runner"},
    }


def _write_screen(monkeypatch, tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "residual_scale_screen.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(runner, "SCREEN_RESULT", path)
    return path


def _configure_checkpoints(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "SOURCE_ROOT", tmp_path / "sources")
    monkeypatch.setattr(runner, "EMBEDDINGS", tmp_path / "embeddings")
    runner.SOURCE_ROOT.mkdir(parents=True)


def _write_source(seed: int, gamma: float = 0.4) -> Path:
    path = runner.source_checkpoint(seed)
    torch.save(
        {
            "gstep": runner.EXPECTED_GSTEP,
            "model": {
                "ad_gamma": torch.tensor([gamma], dtype=torch.float32),
                "other": torch.tensor([seed], dtype=torch.float32),
            },
            "opt": {},
            "queue": torch.zeros(1, 1),
            "qptr": 0,
            "center": torch.zeros(1, 1),
            "target": None,
        },
        path,
    )
    return path


def _context(tmp_path: Path) -> dict[str, str]:
    train_manifest = tmp_path / "train_manifest.json"
    eval_manifest = tmp_path / "eval_manifest.json"
    train_manifest.write_text('{"train": true}\n', encoding="utf-8")
    eval_manifest.write_text('{"eval": true}\n', encoding="utf-8")
    return {
        "protocol_id": "test-protocol",
        "scorer_bundle_sha256": "test-scorer",
        "train_manifest": str(train_manifest),
        "train_manifest_sha256": runner.sha256_file(train_manifest),
        "eval_dir": str(tmp_path / "unknown_eval100"),
        "eval_manifest": str(eval_manifest),
        "eval_manifest_sha256": "test-eval-manifest",
    }


def _metric_row(
    recipe: str,
    clusterer: str,
    seed: int | str,
    p1: int,
    *,
    p3: float,
    p4: float,
    fragment: float,
    ari: float,
    ami: float,
) -> dict:
    return {
        "recipe": recipe,
        "clusterer": clusterer,
        "seed": seed,
        "P1_capture_count": p1,
        "P2_noise_pct": 0.0,
        "P3_completeness": p3,
        "P4_homogeneity": p4,
        "fragment_ratio": fragment,
        "recov": 0.9,
        "Sil": 0.5,
        "ARI_supporting": ari,
        "AMI_supporting": ami,
    }


def test_no_proposal_has_explicit_exit_20(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(runner, "SCREEN_RESULT", tmp_path / "missing.json")

    assert runner.main(["--dry-run"]) == runner.EXIT_NO_PROPOSAL == 20
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "no_op"
    assert payload["exit_code"] == 20

    _write_screen(monkeypatch, tmp_path, _screen_payload(alpha=None))
    assert runner.main(["--dry-run"]) == 20


def test_proposal_requires_both_clusterer_gates(
    monkeypatch, tmp_path: Path
) -> None:
    payload = _screen_payload(alpha=0.5)
    payload["gate"]["values"][1]["clusterers"]["Louvain-res6"][
        "P4_not_worse"
    ] = False
    _write_screen(monkeypatch, tmp_path, payload)

    with pytest.raises(runner.ContractError, match="Louvain"):
        runner.read_screen_proposal()


def test_valid_proposal_is_read_exactly_once(monkeypatch, tmp_path: Path) -> None:
    payload = _screen_payload(alpha=0.75)
    _write_screen(monkeypatch, tmp_path, payload)

    alpha, selected_payload = runner.read_screen_proposal()

    assert alpha == pytest.approx(0.75)
    assert selected_payload == payload


def test_derived_checkpoint_scales_gamma_without_changing_source(
    monkeypatch, tmp_path: Path
) -> None:
    _configure_checkpoints(monkeypatch, tmp_path)
    source = _write_source(seed=1, gamma=0.4)
    source_hash = runner.sha256_file(source)

    source_record = runner.derive_scaled_checkpoint(1, 0.5, "screen-hash")

    assert runner.sha256_file(source) == source_hash
    original = torch.load(source, map_location="cpu", weights_only=False)
    derived_path = runner.scaled_checkpoint(1, 0.5)
    derived = torch.load(derived_path, map_location="cpu", weights_only=False)
    assert float(original["model"]["ad_gamma"].item()) == pytest.approx(0.4)
    assert float(derived["model"]["ad_gamma"].item()) == pytest.approx(0.2)
    assert derived["alpha_three_seed_provenance"]["alpha"] == "0.50"
    assert (
        derived["alpha_three_seed_provenance"]["source_checkpoint_sha256"]
        == source_hash
    )
    assert runner.scaled_checkpoint_is_reusable(
        1, 0.5, source_record, "screen-hash"
    )
    assert not runner.scaled_checkpoint_is_reusable(
        1, 0.75, source_record, "screen-hash"
    )


def test_embedding_sidecar_binds_alpha_sources_manifest_and_scorer(
    monkeypatch, tmp_path: Path
) -> None:
    _configure_checkpoints(monkeypatch, tmp_path)
    _write_screen(monkeypatch, tmp_path, _screen_payload(alpha=0.5))
    _write_source(seed=1, gamma=0.4)
    source = runner.derive_scaled_checkpoint(1, 0.5, "screen-hash")
    context = _context(tmp_path)
    embedding = runner.embedding_path(1, 0.5)
    embedding.parent.mkdir(parents=True, exist_ok=True)
    np.save(embedding, np.zeros((3, 4), dtype=np.float32))

    runner.write_embedding_sidecar(
        1, 0.5, source, "screen-hash", context
    )

    assert runner.embedding_is_reusable(
        1, 0.5, source, "screen-hash", context
    )
    assert not runner.embedding_is_reusable(
        1, 0.5, source, "other-screen-hash", context
    )
    changed_context = dict(context)
    changed_context["scorer_bundle_sha256"] = "changed-scorer"
    assert not runner.embedding_is_reusable(
        1, 0.5, source, "screen-hash", changed_context
    )
    changed_context = dict(context)
    changed_context["eval_manifest_sha256"] = "changed-manifest"
    assert not runner.embedding_is_reusable(
        1, 0.5, source, "screen-hash", changed_context
    )


def test_three_seed_gate_uses_primary_metrics_not_ari_ami() -> None:
    alpha = 0.5
    recipe = "L0_alpha_0.50_ep4"
    rows = []
    for clusterer, frozen_p1 in (
        ("FINCH-p2", 32),
        ("Louvain-res6", 31),
    ):
        rows.append(
            _metric_row(
                "frozen",
                clusterer,
                "none",
                frozen_p1,
                p3=0.8,
                p4=0.9,
                fragment=2.0,
                ari=0.99,
                ami=0.99,
            )
        )
        for seed in runner.SEEDS:
            rows.append(
                _metric_row(
                    recipe,
                    clusterer,
                    seed,
                    frozen_p1,
                    p3=0.81,
                    p4=0.91,
                    fragment=1.9,
                    ari=-100.0,
                    ami=-100.0,
                )
            )

    gate = runner.evaluate_rows(rows, alpha)

    assert gate["accepted"] is True
    assert gate["ARI_AMI"] == "supporting only; excluded from gate"
    for clusterer in runner.EXPECTED_CLUSTERERS:
        checks = gate["clusterers"][clusterer]["checks"]
        assert all("ARI" not in name and "AMI" not in name for name in checks)


def test_valid_dry_run_never_derives_or_extracts(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    payload = _screen_payload(alpha=0.5)
    _write_screen(monkeypatch, tmp_path, payload)
    monkeypatch.setattr(
        runner,
        "validate_screen_provenance",
        lambda _: ({"protocol_id": "test"}, "screen-hash"),
    )
    monkeypatch.setattr(
        runner,
        "derive_scaled_checkpoint",
        lambda *args, **kwargs: pytest.fail("dry-run derived a checkpoint"),
    )
    monkeypatch.setattr(
        runner,
        "extract_embedding",
        lambda *args, **kwargs: pytest.fail("dry-run extracted embeddings"),
    )

    assert runner.main(["--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["mode"] == "inference_only"
    assert plan["no_tapt"] is True
    assert plan["seeds"] == [1, 3, 5]
    assert plan["alpha"] == pytest.approx(0.5)
