from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


CONTROLLER_PATH = Path(
    r"D:\project\unknown-contrastive\runs\campaign_state\controllers"
    r"\run_lr008_rule_c_recovery_260727_0640.py"
)
SPEC = importlib.util.spec_from_file_location("lr008_recovery_controller", CONTROLLER_PATH)
ctl = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ctl)


def passing_guard(_safety):
    return True, {
        "gpu_pid_probe_ok": True,
        "memory_query_ok": True,
        "free_mb": 12588.0,
        "required_free_mb": 7576.0,
        "alive_locks": [],
        "busy_by_lock": False,
        "busy_by_process": False,
        "busy_by_mem": False,
        "gpu_processes": [],
    }


def fake_output(_artifacts, _stages, _pool, checkpoints, _attempt):
    selected = Path(ctl.CHECKPOINTS) / "proj_ep9.pt"
    return {
        "paths": {},
        "sha256": {},
        "selected_epoch": 9,
        "selected_checkpoint_path": str(selected),
        "selected_checkpoint_sha256": checkpoints["file_sha256"]["proj_ep9.pt"],
        "stage_records": [],
        "labels_used": False,
        "offline_only": True,
    }


class ControllerContractTests(unittest.TestCase):
    def test_lock_collision_and_token_verified_release(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / ".RUNNING"
            first = ctl.ExclusiveTokenLock(path)
            first.acquire()
            with self.assertRaisesRegex(RuntimeError, "campaign_lock_collision"):
                ctl.ExclusiveTokenLock(path).acquire()
            self.assertTrue(first.release())
            self.assertFalse(path.exists())

    def test_lock_never_removes_foreign_token(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / ".RUNNING"
            lock = ctl.ExclusiveTokenLock(path)
            lock.acquire()
            path.write_text('{"owner_token":"foreign"}', encoding="utf-8")
            self.assertFalse(lock.release())
            self.assertTrue(path.exists())

    def test_bound_hash_accepts_then_rejects_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bound.txt"
            path.write_text("before", encoding="utf-8")
            expected = {str(path): ctl.sha256_file(path)}
            self.assertEqual(ctl.verify_bound(expected), expected)
            path.write_text("after", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "bound_input_drift"):
                ctl.verify_bound(expected)

    def test_checkpoint_inventory_exact_then_extra_file_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "checkpoints"
            root.mkdir()
            for epoch in range(1, 21):
                (root / f"proj_ep{epoch}.pt").write_bytes(f"epoch={epoch}".encode())
            expected_tree = ctl.sha256_tree(root)
            result = ctl.checkpoint_snapshot(root, expected_tree)
            self.assertEqual(result["count"], 20)
            (root / "extra.pt").write_bytes(b"x")
            with self.assertRaisesRegex(RuntimeError, "checkpoint_inventory_not_exact"):
                ctl.checkpoint_snapshot(root, expected_tree)

    def test_pool_pair_exact_order_and_order_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = [f"class/image_{index:04d}.png" for index in range(4178)]
            unlabeled = root / "unlabeled.json"
            labeled = root / "labeled.json"
            unlabeled.write_text(
                json.dumps({"root": r"E:\data\images\unknown", "files": paths}),
                encoding="utf-8",
            )
            labeled.write_text(
                json.dumps(
                    {
                        "root": r"E:\data\images\unknown",
                        "files": [{"path": path, "label": "x"} for path in paths],
                    }
                ),
                encoding="utf-8",
            )
            ordered_sha = ctl.hashlib.sha256(
                json.dumps(paths, separators=(",", ":"), ensure_ascii=False).encode()
            ).hexdigest()
            with mock.patch.object(ctl, "ORDERED_POOL_PATHS_SHA256", ordered_sha):
                self.assertEqual(ctl.validate_pool_pair(unlabeled, labeled)["count"], 4178)
                payload = json.loads(labeled.read_text(encoding="utf-8"))
                payload["files"][0], payload["files"][1] = payload["files"][1], payload["files"][0]
                labeled.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "pool_count_uniqueness_or_order_invalid"):
                    ctl.validate_pool_pair(unlabeled, labeled)

    def test_registry_requires_single_terminal_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoints = root / "checkpoints"
            checkpoints.mkdir()
            expected_checkpoint = str((checkpoints / "proj_ep20.pt").resolve())
            started = {"step": "strict_novel_lr008_seed42", "status": "started"}
            failed = {
                "step": "strict_novel_lr008_seed42",
                "status": "failed",
                "failure_reason": "rule_c_selector_gpu_busy",
                "produced_checkpoint_path": expected_checkpoint,
                "rule_c_artifacts": [],
                "seed": 42,
                "attempt": 1,
            }
            registry = root / "registry.jsonl"
            registry.write_text(
                "\n".join(json.dumps(item) for item in (started, failed)),
                encoding="utf-8",
            )
            with mock.patch.object(ctl, "CHECKPOINTS", checkpoints):
                self.assertEqual(ctl.validate_registry(registry)["terminal_line"], 2)
                with registry.open("a", encoding="utf-8") as stream:
                    stream.write("\n" + json.dumps(failed))
                with self.assertRaisesRegex(RuntimeError, "record_sequence_invalid"):
                    ctl.validate_registry(registry)

    def test_attempt_collision_and_reparse_chain_block(self):
        with tempfile.TemporaryDirectory() as temp:
            state = Path(temp) / "state"
            state.mkdir()
            attempt = state / "rule_c" / "attempt"
            with mock.patch.object(ctl, "STATE", state):
                ctl.assert_fresh_attempt(attempt)
                attempt.mkdir(parents=True)
                with self.assertRaisesRegex(RuntimeError, "output_collision"):
                    ctl.assert_fresh_attempt(attempt)
                attempt.rmdir()
                real_is_reparse = ctl.is_reparse

                def marked(path):
                    return Path(path) == attempt.parent or real_is_reparse(path)

                attempt.parent.mkdir(exist_ok=True)
                with mock.patch.object(ctl, "is_reparse", side_effect=marked):
                    with self.assertRaisesRegex(RuntimeError, "reparse_point_refused"):
                        ctl.assert_fresh_attempt(attempt)

    def test_config_delta_adds_only_two_temporary_names(self):
        config, step, safety = ctl.load_config_and_exact_step()
        original = config["safety"]
        self.assertEqual(
            safety["gpu_non_compute_process_allowlist"][
                len(original["gpu_non_compute_process_allowlist"]) :
            ],
            ["wudfhost.exe", "amdrssrcext.exe"],
        )
        reconstructed = json.loads(json.dumps(safety))
        reconstructed["gpu_non_compute_process_allowlist"] = original[
            "gpu_non_compute_process_allowlist"
        ]
        self.assertEqual(reconstructed, original)
        self.assertEqual(step["rule_c"]["device"], "cuda")

    def test_guard_failures_block_and_passing_guard_is_accepted(self):
        _, _, safety = ctl.load_config_and_exact_step()
        detail, identities = ctl.validate_gpu_guard(
            passing_guard, safety, identity_validator=lambda _rows: []
        )
        self.assertEqual(detail["free_mb"], 12588.0)
        self.assertEqual(identities, [])

        def low_memory(_safety):
            ok, detail = passing_guard(_safety)
            detail["free_mb"] = 7000.0
            return ok, detail

        with self.assertRaisesRegex(RuntimeError, "gpu_guard_failed"):
            ctl.validate_gpu_guard(low_memory, safety, identity_validator=lambda _rows: [])

    def test_changed_wudfhost_pid_is_refused_before_process_probe(self):
        rows = [
            {
                "pid": 999999,
                "name": "wudfhost.exe",
                "classification": "gui_allowlisted",
                "allowlisted": True,
            }
        ]
        with self.assertRaisesRegex(RuntimeError, "changed_wudfhost_pid_refused"):
            ctl.validate_temporary_gpu_identities(rows)

    def test_receipt_is_create_new_only(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "receipt.json"
            ctl.write_json_create_new(path, {"status": "completed"})
            with self.assertRaisesRegex(RuntimeError, "receipt_collision"):
                ctl.write_json_create_new(path, {"status": "second"})

    def test_postflight_drift_comparator(self):
        ctl.require_unchanged({"a": 1}, {"a": 1}, "same")
        with self.assertRaisesRegex(RuntimeError, "postflight_drift"):
            ctl.require_unchanged({"a": 1}, {"a": 2}, "changed")

    def test_stage_and_artifact_validation_reject_missing_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "stage_call_count_invalid"):
                ctl.validate_stage_records([], Path(temp))
            with self.assertRaisesRegex(RuntimeError, "artifact_path_invalid"):
                ctl.validate_outputs([], [], {"paths": []}, {"file_sha256": {}}, Path(temp))

    def test_stop_request_blocks_before_helper_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            state = Path(temp) / "state"
            state.mkdir()
            stop = state / "STOP_REQUESTED"
            stop.write_text("stop", encoding="utf-8")
            attempt = state / "rule_c" / "attempt"
            lock = state / ".RUNNING"
            calls = []
            with (
                mock.patch.object(ctl, "STATE", state),
                mock.patch.object(ctl, "STOP", stop),
            ):
                with self.assertRaisesRegex(RuntimeError, "STOP_REQUESTED_present"):
                    ctl.execute(
                        helper=lambda *args, **kwargs: calls.append(1),
                        guard=passing_guard,
                        identity_validator=lambda _rows: [],
                        output_validator=fake_output,
                        attempt=attempt,
                        receipt=attempt / "recovery_receipt.json",
                        lock_path=lock,
                    )
            self.assertEqual(calls, [])
            self.assertFalse(attempt.exists())
            self.assertFalse(lock.exists())

    def test_output_collision_blocks_before_helper_and_partial_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            state = Path(temp) / "state"
            attempt = state / "rule_c" / "attempt"
            attempt.mkdir(parents=True)
            stop = state / "STOP_REQUESTED"
            lock = state / ".RUNNING"
            calls = []
            with (
                mock.patch.object(ctl, "STATE", state),
                mock.patch.object(ctl, "STOP", stop),
            ):
                with self.assertRaisesRegex(RuntimeError, "output_collision"):
                    ctl.execute(
                        helper=lambda *args, **kwargs: calls.append(1),
                        guard=passing_guard,
                        identity_validator=lambda _rows: [],
                        output_validator=fake_output,
                        attempt=attempt,
                        receipt=attempt / "recovery_receipt.json",
                        lock_path=lock,
                    )
            self.assertEqual(calls, [])
            self.assertTrue(attempt.exists())
            self.assertFalse(lock.exists())

    def test_mock_success_calls_helper_once_and_writes_no_secret_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            state = Path(temp) / "state"
            state.mkdir()
            attempt = state / "rule_c" / "attempt"
            lock = state / ".RUNNING"
            receipt = attempt / "recovery_receipt.json"
            calls = []

            def helper(*_args, **_kwargs):
                calls.append(1)
                return [{}], None

            with (
                mock.patch.object(ctl, "STATE", state),
                mock.patch.object(ctl, "STOP", state / "STOP_REQUESTED"),
            ):
                result = ctl.execute(
                    helper=helper,
                    guard=passing_guard,
                    identity_validator=lambda _rows: [],
                    output_validator=fake_output,
                    attempt=attempt,
                    receipt=receipt,
                    lock_path=lock,
                )
            self.assertEqual(calls, [1])
            self.assertEqual(result["helper"]["call_count"], 1)
            raw = receipt.read_text(encoding="utf-8")
            self.assertNotIn("OPENAI_API_KEY", raw)
            self.assertNotIn("CUDA_VISIBLE_DEVICES", raw)
            self.assertFalse(lock.exists())


if __name__ == "__main__":
    unittest.main()
