from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
GUARD = REPO / "scripts/run_fcmae_adapter_temperature_screen_guard.ps1"
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fake_runner(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "import os",
                "from pathlib import Path",
                "Path(os.environ['GUARD_TEST_MARKER']).write_text('ran', encoding='utf-8')",
                "raise SystemExit(int(os.environ.get('GUARD_TEST_EXIT', '0')))",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _run_guard(
    *,
    runner: Path,
    result: Path,
    lock: Path,
    marker: Path,
    child_tag_prefix: str,
    exit_code: int = 0,
    min_free_physical_gb: float = 0.0,
    min_free_virtual_gb: float = 0.0,
    resource_retry_exit_code: int = 75,
    resource_log: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    if POWERSHELL is None:
        pytest.skip("PowerShell is unavailable")
    env = os.environ.copy()
    env["GUARD_TEST_MARKER"] = str(marker)
    env["GUARD_TEST_EXIT"] = str(exit_code)
    resource_log = resource_log or lock.with_name(f"{lock.name}.resource.log")
    return subprocess.run(
        [
            POWERSHELL,
            "-NoLogo",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(GUARD),
            "-Python",
            sys.executable,
            "-Runner",
            str(runner),
            "-ResultJson",
            str(result),
            "-LockPath",
            str(lock),
            "-ChildTagPrefix",
            child_tag_prefix,
            "-MinFreePhysicalGB",
            str(min_free_physical_gb),
            "-MinFreeVirtualGB",
            str(min_free_virtual_gb),
            "-ResourceRetryExitCode",
            str(resource_retry_exit_code),
            "-RetryAfterSeconds",
            "120",
            "-ResourceLogPath",
            str(resource_log),
        ],
        cwd=REPO,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def _write_terminal_result(result: Path, payload: dict) -> None:
    result.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_complete_result(
    root: Path, runner: Path, result: Path
) -> dict[str, Path]:
    trainer = root / "trainer.py"
    protocol_source = root / "protocol.py"
    scorer = root / "scorer.py"
    baseline_checkpoint = root / "source.pt"
    train_manifest = root / "train_manifest.json"
    eval_manifest = root / "eval_manifest.json"
    eval_dir = root / "eval"
    frozen_embedding = root / "frozen.npy"
    csv_output = root / "result.csv"
    markdown_output = root / "result.md"
    eval_dir.mkdir()
    for path, content in (
        (trainer, "trainer"),
        (protocol_source, "protocol"),
        (scorer, "scorer"),
        (baseline_checkpoint, "checkpoint"),
        (train_manifest, '{"train": true}'),
        (eval_manifest, '{"eval": true}'),
        (frozen_embedding, "frozen-embedding"),
        (csv_output, "rows"),
        (markdown_output, "report"),
    ):
        path.write_text(content, encoding="utf-8")

    protocol_id = "guard-test-protocol"
    scorer_bundle_sha256 = _sha256(scorer)
    train_manifest_sha256 = _sha256(train_manifest)
    eval_manifest_sha256 = _sha256(eval_manifest)
    temperatures = ("0.05", "0.07", "0.10", "0.20")
    sidecars: dict[str, dict[str, str]] = {}
    sidecar_paths: dict[str, Path] = {}
    progress_paths: dict[str, Path] = {}
    embeddings: dict[str, Path] = {}
    commands: dict[str, list[str]] = {}
    for temperature in temperatures:
        tag = f"fcmae_ad1_t{int(round(float(temperature) * 100)):03d}_s1"
        command = ["--temp", temperature]
        commands[temperature] = command
        checkpoint = root / f"temperature_{temperature}.pt"
        checkpoint.write_bytes(
            baseline_checkpoint.read_bytes()
            if temperature == "0.05"
            else f"checkpoint-{temperature}".encode("utf-8")
        )
        embedding = root / f"temperature_{temperature}.npy"
        embedding.write_bytes(f"embedding-{temperature}".encode("utf-8"))
        sidecar = root / f"temperature_{temperature}.provenance.json"
        launch_contract = {
            "schema": 2,
            "axis": "temperature",
            "temperature": temperature,
            "seed": 1,
            "epoch": 4,
            "tag": tag,
            "command": command,
            "command_sha256": hashlib.sha256(
                json.dumps(
                    command,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "source": str(runner),
            "source_sha256": _sha256(runner),
            "baseline_source_checkpoint": str(baseline_checkpoint),
            "baseline_source_checkpoint_sha256": _sha256(
                baseline_checkpoint
            ),
            "trainer": str(trainer),
            "trainer_sha256": _sha256(trainer),
            "protocol_source": str(protocol_source),
            "protocol_source_sha256": _sha256(protocol_source),
            "protocol_id": protocol_id,
            "scorer_bundle_sha256": scorer_bundle_sha256,
            "train_manifest": str(train_manifest),
            "train_manifest_sha256": train_manifest_sha256,
            "train_manifest_file_sha256": _sha256(train_manifest),
            "eval_dir": str(eval_dir),
            "eval_manifest": str(eval_manifest),
            "eval_manifest_sha256": eval_manifest_sha256,
            "eval_manifest_file_sha256": _sha256(eval_manifest),
        }
        sidecar_payload = {
            "created_at": "2026-07-25T07:59:00+09:00",
            "contract": {
                "schema": 2,
                "axis": "temperature",
                "temperature": temperature,
                "seed": 1,
                "epoch": 4,
                "source_checkpoint": str(checkpoint),
                "source_checkpoint_sha256": _sha256(checkpoint),
                "baseline_origin_checkpoint": (
                    {
                        "path": str(baseline_checkpoint),
                        "sha256": _sha256(baseline_checkpoint),
                    }
                    if temperature == "0.05"
                    else None
                ),
                "launch_contract": launch_contract,
                "launch_contract_sha256": hashlib.sha256(
                    json.dumps(
                        launch_contract,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "runner": launch_contract["source"],
                "runner_sha256": launch_contract["source_sha256"],
                "trainer": launch_contract["trainer"],
                "trainer_sha256": launch_contract["trainer_sha256"],
                "protocol_source": launch_contract["protocol_source"],
                "protocol_source_sha256": launch_contract[
                    "protocol_source_sha256"
                ],
                "scorer_bundle_sha256": scorer_bundle_sha256,
                "train_manifest": str(train_manifest),
                "train_manifest_sha256": train_manifest_sha256,
                "eval_dir": str(eval_dir),
                "eval_manifest": str(eval_manifest),
                "eval_manifest_sha256": eval_manifest_sha256,
                "eval_manifest_file_sha256": _sha256(eval_manifest),
                "command_sha256": launch_contract["command_sha256"],
            },
            "embedding": str(embedding),
            "embedding_sha256": _sha256(embedding),
        }
        sidecar.write_text(json.dumps(sidecar_payload), encoding="utf-8")
        if temperature != "0.05":
            progress = root / f"{tag}.progress.json"
            checkpoint_after = {
                "path": str(checkpoint),
                "sha256": _sha256(checkpoint),
                "size_bytes": checkpoint.stat().st_size,
                "gstep": 3296,
                "qptr": 0,
                "queue_shape": [4096, 128],
                "center_shape": [1, 128],
            }
            progress.write_text(
                json.dumps(
                    {
                        "created_at": "2026-07-25T07:58:00+09:00",
                        "updated_at": "2026-07-25T07:59:00+09:00",
                        "state": "completed",
                        "contract": launch_contract,
                        "last_launch": {
                            "launch_mode": "fresh",
                            "exit_code": 0,
                            "checkpoint_after": checkpoint_after,
                        },
                        "attempts": [],
                    }
                ),
                encoding="utf-8",
            )
            progress_paths[temperature] = progress
        sidecars[temperature] = {"path": str(sidecar), "sha256": _sha256(sidecar)}
        sidecar_paths[temperature] = sidecar
        embeddings[temperature] = embedding

    clusterers = ("FINCH-p2", "Louvain-res6")
    values = []
    for temperature in temperatures:
        values.append(
            {
                "temperature": float(temperature),
                "accepted": False,
                "clusterers": {
                    name: {
                        "accepted": False,
                        "checks": {
                            "P1_preserved": True,
                            "P2_not_worse": True,
                            "P3_not_worse": False,
                            "P4_not_worse": True,
                        },
                        "delta": {},
                    }
                    for name in clusterers
                },
            }
        )

    rows = []
    for recipe, seed, epoch, embedding in (
        ("frozen", "none", 0, frozen_embedding),
        ("adapter_temp_0.05", 1, 4, embeddings["0.05"]),
        ("adapter_temp_0.07", 1, 4, embeddings["0.07"]),
        ("adapter_temp_0.10", 1, 4, embeddings["0.10"]),
        ("adapter_temp_0.20", 1, 4, embeddings["0.20"]),
    ):
        for clusterer in clusterers:
            rows.append(
                {
                    "protocol_id": protocol_id,
                    "recipe": recipe,
                    "seed": seed,
                    "epoch": epoch,
                    "clusterer": clusterer,
                    "P1_capture_count": 32,
                    "P2_noise_pct": 0.0,
                    "P3_completeness": 0.9,
                    "P4_homogeneity": 0.95,
                    "embedding_path": str(embedding),
                    "embedding_sha256": _sha256(embedding),
                    "scorer_bundle_sha256": scorer_bundle_sha256,
                    "eval_manifest_sha256": eval_manifest_sha256,
                    "train_manifest_sha256": train_manifest_sha256,
                }
            )

    payload = {
        "created_at": "2026-07-25T08:00:00+09:00",
        "protocol_id": protocol_id,
        "screen": {"values": values, "proposed_temperature": None},
        "rows": rows,
        "provenance": {
            "protocol_id": protocol_id,
            "script": str(runner),
            "script_sha256": _sha256(runner),
            "trainer": str(trainer),
            "trainer_sha256": _sha256(trainer),
            "protocol_source": str(protocol_source),
            "protocol_source_sha256": _sha256(protocol_source),
            "baseline_source_checkpoint": str(baseline_checkpoint),
            "baseline_source_checkpoint_sha256": _sha256(baseline_checkpoint),
            "commands": commands,
            "scorer_bundle_sha256": scorer_bundle_sha256,
            "train_manifest": str(train_manifest),
            "train_manifest_sha256": train_manifest_sha256,
            "train_manifest_file_sha256": _sha256(train_manifest),
            "eval_manifest": str(eval_manifest),
            "eval_manifest_sha256": eval_manifest_sha256,
            "eval_manifest_file_sha256": _sha256(eval_manifest),
            "embedding_sidecars": sidecars,
        },
        "outputs": {
            "json": str(result),
            "csv": str(csv_output),
            "csv_sha256": _sha256(csv_output),
            "markdown": str(markdown_output),
            "markdown_sha256": _sha256(markdown_output),
        },
    }
    _write_terminal_result(result, payload)
    return {
        "json": result,
        "csv": csv_output,
        "markdown": markdown_output,
        "frozen_embedding": frozen_embedding,
        "sidecar_0.07": sidecar_paths["0.07"],
        "progress_0.07": progress_paths["0.07"],
        "embedding_0.07": embeddings["0.07"],
    }


def test_guard_is_synchronous_and_has_no_other_queue_actions() -> None:
    source = GUARD.read_text(encoding="utf-8")

    assert "& $Python -u $Runner" in source
    assert "Start-Process" not in source
    assert "Register-ScheduledTask" not in source
    assert "Start-ScheduledTask" not in source
    assert "run_fcmae_adapter_queue" not in source
    assert "run_fcmae_adapter_negative" not in source
    assert "run_post_holdout" not in source


def test_guard_runs_runner_and_forwards_exit_code(tmp_path: Path) -> None:
    runner = tmp_path / "guard_fixture_runner.py"
    result = tmp_path / "missing_result.json"
    lock = tmp_path / "guard.lock"
    marker = tmp_path / "runner.marker"
    _write_fake_runner(runner)

    completed = _run_guard(
        runner=runner,
        result=result,
        lock=lock,
        marker=marker,
        child_tag_prefix="guard_fixture_temperature_t0",
        exit_code=23,
    )

    assert completed.returncode == 23, completed.stderr
    assert marker.read_text(encoding="utf-8") == "ran"
    assert not lock.exists()


def test_guard_exits_for_valid_complete_result(tmp_path: Path) -> None:
    runner = tmp_path / "guard_complete_fixture.py"
    result = tmp_path / "complete_result.json"
    lock = tmp_path / "guard.lock"
    marker = tmp_path / "runner.marker"
    _write_fake_runner(runner)
    _write_complete_result(tmp_path, runner, result)

    completed = _run_guard(
        runner=runner,
        result=result,
        lock=lock,
        marker=marker,
        child_tag_prefix="guard_complete_temperature_t0",
    )

    assert completed.returncode == 0, completed.stderr
    assert not marker.exists()
    assert not lock.exists()


@pytest.mark.parametrize("artifact", ("csv", "sidecar", "progress"))
def test_guard_relaunches_when_terminal_artifact_is_invalid(
    tmp_path: Path, artifact: str
) -> None:
    runner = tmp_path / "guard_tamper_fixture.py"
    result = tmp_path / "complete_result.json"
    lock = tmp_path / "guard.lock"
    marker = tmp_path / "runner.marker"
    _write_fake_runner(runner)
    paths = _write_complete_result(tmp_path, runner, result)
    if artifact == "csv":
        paths["csv"].write_text("tampered", encoding="utf-8")
    elif artifact == "sidecar":
        paths["sidecar_0.07"].write_text("{}", encoding="utf-8")
    else:
        progress = json.loads(
            paths["progress_0.07"].read_text(encoding="utf-8")
        )
        progress["state"] = "failed"
        paths["progress_0.07"].write_text(
            json.dumps(progress), encoding="utf-8"
        )

    completed = _run_guard(
        runner=runner,
        result=result,
        lock=lock,
        marker=marker,
        child_tag_prefix="guard_tamper_temperature_t0",
    )

    assert completed.returncode == 0, completed.stderr
    assert marker.read_text(encoding="utf-8") == "ran"
    assert not lock.exists()


def test_guard_exits_when_runner_process_is_active(tmp_path: Path) -> None:
    runner = tmp_path / "guard_active_runner_fixture.py"
    result = tmp_path / "missing_result.json"
    lock = tmp_path / "guard.lock"
    marker = tmp_path / "runner.marker"
    _write_fake_runner(runner)

    blocker = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            str(runner),
        ]
    )
    try:
        time.sleep(0.5)
        completed = _run_guard(
            runner=runner,
            result=result,
            lock=lock,
            marker=marker,
            child_tag_prefix="guard_active_temperature_t0",
        )
        assert completed.returncode == 0, completed.stderr
        assert not marker.exists()
        assert not lock.exists()
    finally:
        blocker.terminate()
        blocker.wait(timeout=10)


def test_guard_defers_launch_when_memory_is_below_threshold(
    tmp_path: Path,
) -> None:
    runner = tmp_path / "guard_memory_fixture.py"
    result = tmp_path / "missing_result.json"
    lock = tmp_path / "guard.lock"
    marker = tmp_path / "runner.marker"
    resource_log = tmp_path / "guard_resource.log"
    _write_fake_runner(runner)

    completed = _run_guard(
        runner=runner,
        result=result,
        lock=lock,
        marker=marker,
        child_tag_prefix="guard_memory_temperature_t0",
        min_free_physical_gb=1_000_000_000.0,
        min_free_virtual_gb=1_000_000_000.0,
        resource_retry_exit_code=75,
        resource_log=resource_log,
    )

    assert completed.returncode == 75, completed.stderr
    assert not marker.exists()
    assert not lock.exists()
    log = resource_log.read_text(encoding="utf-8")
    assert "launch_deferred" in log
    assert "retry_after_seconds=120" in log
    assert "exit_code=75" in log


def test_stale_lock_is_reclaimed_only_without_matching_process(
    tmp_path: Path,
) -> None:
    runner = tmp_path / "guard_stale_fixture.py"
    result = tmp_path / "missing_result.json"
    lock = tmp_path / "guard.lock"
    marker = tmp_path / "runner.marker"
    child_tag_prefix = "guard_stale_temperature_t0"
    _write_fake_runner(runner)
    lock.write_text("old lock", encoding="utf-8")
    stale_time = time.time() - 7200
    os.utime(lock, (stale_time, stale_time))

    blocker = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            f"{child_tag_prefix}05_s1",
        ]
    )
    try:
        time.sleep(0.5)
        blocked = _run_guard(
            runner=runner,
            result=result,
            lock=lock,
            marker=marker,
            child_tag_prefix=child_tag_prefix,
        )
        assert blocked.returncode == 0, blocked.stderr
        assert lock.exists()
        assert not marker.exists()
    finally:
        blocker.terminate()
        blocker.wait(timeout=10)

    recovered = _run_guard(
        runner=runner,
        result=result,
        lock=lock,
        marker=marker,
        child_tag_prefix=child_tag_prefix,
    )
    assert recovered.returncode == 0, recovered.stderr
    assert marker.read_text(encoding="utf-8") == "ran"
    assert not lock.exists()
