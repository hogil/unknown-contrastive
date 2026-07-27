from __future__ import annotations

import importlib.util
import inspect
import json
import math
import os
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock


SOURCE = Path(r"D:\project\unknown-contrastive\runs\campaign_state\controllers\run_strict_novel_q32768_seed42_v3.py")
SPEC = importlib.util.spec_from_file_location("q32768_controller", SOURCE)
ctl = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ctl
assert SPEC.loader is not None
SPEC.loader.exec_module(ctl)
AUTO_IDENTITY = object()


class ControllerTests(unittest.TestCase):
    def case(self, name: str) -> Path:
        path = ctl.SCRATCH / name
        path.mkdir(parents=True, exist_ok=False)
        return path

    def paths(self, case: Path) -> ctl.Paths:
        return ctl.Paths(ctl.SCRATCH, case / "action", case / "run.lock", case / "gpu.lock", case / "receipts")

    def rows_and_lineage(self):
        rows = [(f"proj_ep{epoch}.pt", f"checkpoint-{epoch}".encode()) for epoch in range(1, 21)]
        lineage = {
            name: {"producer": "synthetic-trainer", "input_sha256": "a" * 64}
            for name, _ in rows
        }
        return rows, lineage

    def selector(self, *, eligible: bool = True, pool_sha: str = "b" * 64, ordered_paths=None):
        return {
            "schema": "q32768_selector.v1",
            "pool_kind": "unlabeled",
            "pool_sha": pool_sha,
            "ordered_paths": list(ordered_paths or ["synthetic/001.png", "synthetic/002.png"]),
            "epochs": [
                {
                    "epoch": epoch,
                    "worst_pre_reassign_noise": float(epoch),
                    "mean": float(epoch),
                    "eligible": eligible,
                }
                for epoch in range(1, 21)
            ],
        }

    def policy(
        self,
        case: Path,
        *,
        schema: str = "strict_novel_q32768_seed42_v3_launch.v1",
        label_payload: bytes = b'{"labels":[1,2]}',
        ordered_paths=None,
        label_size=None,
        label_device=AUTO_IDENTITY,
        label_inode=AUTO_IDENTITY,
        scorer_payload=None,
    ):
        ordered_paths = list(ordered_paths or ["synthetic/001.png", "synthetic/002.png"])
        scorer_payload = ctl.SCORER_BYTES if scorer_payload is None else scorer_payload
        deps = {}
        dep_root = case / "deps"
        dep_root.mkdir()
        for name in sorted(ctl.DEP_KEYS):
            path = dep_root / f"{name}.bin"
            payload = label_payload if name == "labeled" else scorer_payload if name == "scorer" else f"synthetic-{name}".encode()
            path.write_bytes(payload)
            deps[name] = (path, ctl.sha(payload))
        label = deps["labeled"][0]
        label_stat = label.stat()
        r3 = case / "launch_r3.json"
        body = {
            "schema": schema,
            "verdict": "approve",
            "launch_authorized": True,
            "dependencies": {
                name: {"path": ctl.lexical(path), "sha256": digest}
                for name, (path, digest) in deps.items()
            },
            "unlabeled_binding": {
                "manifest_sha256": deps["unlabeled"][1],
                "ordered_paths_sha256": ctl.ordered_digest(ordered_paths),
                "ordered_paths_count": len(ordered_paths),
            },
            "labeled_binding": {
                "canonical_path": ctl.lexical(label),
                "sha256": deps["labeled"][1],
                "size": len(label_payload) if label_size is None else label_size,
                "device": label_stat.st_dev if label_device is AUTO_IDENTITY else label_device,
                "inode": label_stat.st_ino if label_inode is AUTO_IDENTITY else label_inode,
            },
        }
        raw = ctl.canon(body)
        r3.write_bytes(raw)
        return ctl.LaunchPolicy(r3, ctl.sha(raw), deps)

    def state(self):
        raw = ctl.canon({"state": "WAITING_PANEL", "queue": []})
        return raw, ctl.sha(raw)

    def snapshot(self, value: str = "stable"):
        return {key: f"{value}-{key}" for key in ctl.SNAPSHOT_KEYS}

    def harness(
        self,
        case: Path,
        *,
        probes=(True, True),
        stopped=False,
        trainer_mode="one",
        selector_mode="valid",
        drift_at=None,
        release_fail_once=(),
    ):
        events = []
        paths = self.paths(case)
        policy = self.policy(case)
        state_bytes, state_sha = self.state()
        stable = self.snapshot()
        stable["state"] = state_sha
        snapshot_calls = 0

        def snapshotter():
            nonlocal snapshot_calls
            snapshot_calls += 1
            events.append(f"snapshot:{snapshot_calls}")
            return self.snapshot("drift") if drift_at is not None and snapshot_calls >= drift_at else dict(stable)

        probe_values = list(probes)

        def probe():
            value = probe_values.pop(0) if probe_values else probes[-1]
            events.append(f"probe:{value}")
            return value

        def stop():
            value = stopped() if callable(stopped) else bool(stopped)
            events.append(f"stop:{value}")
            return value

        class FakeLock:
            failures = {name: 1 for name in release_fail_once}

            def __init__(self, path):
                self.name = Path(path).name

            def acquire(self):
                events.append(f"acquire:{self.name}")

            def release(self):
                events.append(f"release:{self.name}")
                if self.failures.get(self.name, 0):
                    self.failures[self.name] -= 1
                    raise RuntimeError(f"release_failed:{self.name}")

        trainer_root = case / "trainers"
        trainer_root.mkdir()
        trainer = trainer_root / "new"

        def trainer_inventory():
            events.append("trainer_inventory")
            return list(trainer_root.iterdir())

        def train_phase():
            events.append("train")
            if trainer_mode != "zero":
                trainer.mkdir()
            if trainer_mode == "multiple":
                (trainer_root / "extra").mkdir()
            return trainer

        rows, lineage = self.rows_and_lineage()

        def checkpoint_loader(path):
            self.assertEqual(path, trainer)
            events.append("checkpoint_loader")
            return rows, lineage

        def selector_phase(path):
            self.assertEqual(path, trainer)
            events.append("selector")
            if selector_mode == "invalid":
                return {"epochs": []}
            return self.selector(pool_sha=policy.deps["unlabeled"][1])

        return {
            "paths": paths,
            "policy": policy,
            "state_bytes": state_bytes,
            "expected_state_sha": state_sha,
            "snapshotter": snapshotter,
            "probe": probe,
            "stop": stop,
            "sleep": lambda seconds: events.append(f"sleep:{seconds}"),
            "locks": FakeLock,
            "train_phase": train_phase,
            "trainer_inventory": trainer_inventory,
            "checkpoint_loader": checkpoint_loader,
            "selector_phase": selector_phase,
        }, events

    def test_01_import_and_recipe_contract(self):
        self.assertIs(sys.modules["q32768_controller"], ctl)
        self.assertEqual(
            ctl.SCRATCH,
            Path(
                r"D:\project\unknown-contrastive\runs\campaign_state\validation_scratch"
                r"\strict_novel_q32768_seed42_v3_repair_result_260727_v1"
            ),
        )
        self.assertEqual(ctl.TRAIN_COMMAND, (ctl.TRAIN_PYTHON, str(ctl.REPO / "_may_ablation.py"), "B4"))
        self.assertEqual(
            ctl.TRAIN_ENV,
            {
                "REPRO_DATA": str(ctl.TRAIN_MANIFEST),
                "REPRO_SEED": "42",
                "REPRO_BATCH": "16",
                "REPRO_WORKERS": "8",
                "REPRO_GPU_MEMORY_FRACTION": "0.40",
                "REPRO_EPOCHS": "20",
                "REPRO_SAMPLING": "0.25",
                "REPRO_LR": "0.004",
                "REPRO_TEMP": "0.20",
                "REPRO_QUEUE": "32768",
                "REPRO_IGNORE_NEG_SIM": "0.72",
                "REPRO_TAG": "_campaign_strict_q32768_s42_v3",
                "REPRO_LEAN": "1",
                "REPRO_SKIP_FINAL_EMBED": "1",
                "CUDA_VISIBLE_DEVICES": "0",
            },
        )
        self.assertEqual(ctl.SELECTOR_RATIOS, (".005", ".01", ".02"))
        self.assertEqual(ctl.SELECTOR_SEEDS, tuple(range(1, 11)))
        self.assertIn("scorer", ctl.DEP_KEYS)

    def test_02_default_main_and_policy_fail_closed(self):
        self.assertRaisesRegex(RuntimeError, "launch_unset", ctl.verify, None)
        self.assertRaisesRegex(RuntimeError, "launch_unset", ctl.main, [])

    def test_03_exact_scratch_and_escape(self):
        case = self.case("case03")
        self.paths(case).check()
        self.assertRaisesRegex(
            RuntimeError,
            "scratch_root",
            ctl.Paths(case, case / "a", case / "r", case / "g", case / "x").check,
        )
        self.assertRaisesRegex(RuntimeError, "scratch_escape", ctl.safe, Path(r"D:\outside"))
        self.assertRaisesRegex(
            RuntimeError,
            "scratch_escape",
            ctl.safe,
            ctl.REPO
            / r"runs\campaign_state\validation_scratch"
            / "strict_novel_q32768_seed42_v3_controller_review_260727_v1",
        )
        self.assertRaisesRegex(RuntimeError, "scratch_escape", ctl.safe, ctl.PRODUCTION_ROOT)

    def test_04_atomic_create_and_collision(self):
        case = self.case("case04")
        forbidden = ("os." + "link", "os." + "symlink", "Path." + "symlink_to", "Create" + "HardLink")
        source_text = SOURCE.read_text(encoding="utf-8")
        test_text = Path(__file__).read_text(encoding="utf-8")
        for token in forbidden:
            self.assertNotIn(token, source_text)
            self.assertNotIn(token, test_text)
        path = case / "artifact.json"
        expected = ctl.canon({"a": 1})
        self.assertEqual(ctl.atomic(path, {"a": 1}), ctl.sha(expected))
        self.assertEqual(path.read_bytes(), expected)
        self.assertRaisesRegex(RuntimeError, "atomic_collision", ctl.atomic, path, {"a": 2})

    def test_05_token_lock_success_and_collision(self):
        case = self.case("case05")
        first = ctl.Lock(case / "lock")
        first.acquire()
        self.assertRaisesRegex(RuntimeError, "lock_collision", ctl.Lock(case / "lock").acquire)
        first.release()
        self.assertFalse((case / "lock").exists())

    def test_06_token_lock_wrong_corrupt_and_missing(self):
        case = self.case("case06")
        wrong = ctl.Lock(case / "wrong")
        wrong.acquire()
        (case / "wrong").write_bytes(ctl.canon({"pid": os.getpid(), "owner_token": "different"}))
        self.assertRaisesRegex(RuntimeError, "lock_token", wrong.release)
        corrupt = ctl.Lock(case / "corrupt")
        corrupt.acquire()
        (case / "corrupt").write_bytes(b"{")
        self.assertRaisesRegex(RuntimeError, "lock_corrupt", corrupt.release)
        missing = ctl.Lock(case / "missing")
        missing.acquire()
        (case / "missing").unlink()
        self.assertRaisesRegex(RuntimeError, "lock_missing", missing.release)

    def test_07_policy_valid_and_raw_hash_failure(self):
        case = self.case("case07")
        policy = self.policy(case)
        binding = ctl.verify(policy)
        self.assertIs(type(binding), ctl._VerifiedBinding)
        self.assertEqual(
            ctl.nonlabel_binding(binding)["r3"],
            {"path": ctl.lexical(policy.r3), "sha256": policy.r3_sha},
        )
        bad = ctl.LaunchPolicy(policy.r3, "0" * 64, policy.deps)
        self.assertRaisesRegex(RuntimeError, "r3_raw_hash", ctl.verify, bad)

    def test_08_policy_schema_keyset_and_dependency_hash_failures(self):
        schema_case = self.case("case08_schema")
        self.assertRaisesRegex(RuntimeError, "r3_schema", ctl.verify, self.policy(schema_case, schema="wrong"))
        key_case = self.case("case08_keys")
        policy = self.policy(key_case)
        missing = dict(policy.deps)
        missing.pop("labeled")
        self.assertRaisesRegex(
            RuntimeError, "dep_keyset", ctl.verify, ctl.LaunchPolicy(policy.r3, policy.r3_sha, missing)
        )
        hash_case = self.case("case08_hash")
        policy = self.policy(hash_case)
        path, _ = policy.deps["trainer"]
        path.write_bytes(b"tampered-trainer")
        self.assertRaisesRegex(RuntimeError, "dep_hash_trainer", ctl.verify, policy)
        scorer_hash_case = self.case("case08_scorer_hash")
        policy = self.policy(scorer_hash_case)
        policy.deps["scorer"][0].write_bytes(b"tampered")
        self.assertRaisesRegex(RuntimeError, "dep_hash_scorer", ctl.verify, policy)
        for index, payload in enumerate(
            (
                b'{"schema":"q32768_scorer_program.v1","op":"count_labels"}',
                b'{"op":"other","schema":"q32768_scorer_program.v1"}',
                b'{"op":"count_labels","schema":"wrong"}',
                b'{"extra":0,"op":"count_labels","schema":"q32768_scorer_program.v1"}',
            )
        ):
            scorer_case = self.case(f"case08_scorer_program_{index}")
            self.assertRaisesRegex(
                RuntimeError,
                "scorer_program",
                ctl.verify,
                self.policy(scorer_case, scorer_payload=payload),
            )
        spy_case = self.case("case08_spy")
        policy = self.policy(spy_case)
        seen = []

        def reader(path):
            seen.append(path)
            if path == policy.deps["labeled"][0]:
                raise AssertionError("pre-seal labeled read")
            return path.read_bytes()

        probes = []

        def guarded(name):
            original = getattr(Path, name)

            def call(path, *args, **kwargs):
                if path == policy.deps["labeled"][0]:
                    probes.append(name)
                    raise AssertionError(f"pre-seal labeled probe:{name}")
                return original(path, *args, **kwargs)

            return call

        with (
            mock.patch.object(Path, "resolve", guarded("resolve")),
            mock.patch.object(Path, "exists", guarded("exists")),
            mock.patch.object(Path, "is_file", guarded("is_file")),
            mock.patch.object(Path, "is_symlink", guarded("is_symlink")),
            mock.patch.object(Path, "stat", guarded("stat")),
            mock.patch.object(Path, "lstat", guarded("lstat")),
            mock.patch.object(Path, "iterdir", guarded("iterdir")),
        ):
            ctl.verify(policy, reader)
        self.assertNotIn(policy.deps["labeled"][0], seen)
        self.assertEqual(probes, [])
        self.assertEqual(
            set(seen),
            {policy.r3} | {path for name, (path, _) in policy.deps.items() if name != "labeled"},
        )
        self.assertTrue(all(seen.count(path) == 1 for path in set(seen)))
        malformed_case = self.case("case08_malformed_label")
        policy = self.policy(malformed_case)
        body = json.loads(policy.r3.read_bytes())
        body["labeled_binding"]["size"] = -1
        raw = ctl.canon(body)
        policy.r3.write_bytes(raw)
        seen = []
        self.assertRaisesRegex(
            RuntimeError,
            "labeled_binding",
            ctl.verify,
            ctl.LaunchPolicy(policy.r3, ctl.sha(raw), policy.deps),
            lambda path: seen.append(path) or path.read_bytes(),
        )
        self.assertNotIn(policy.deps["labeled"][0], seen)
        for field, value in (("device", None), ("inode", None), ("device", True), ("inode", True)):
            identity_case = self.case(f"case08_{field}_{type(value).__name__}")
            kwargs = {f"label_{field}": value}
            policy = self.policy(identity_case, **kwargs)
            seen = []
            self.assertRaisesRegex(
                RuntimeError,
                "labeled_binding",
                ctl.verify,
                policy,
                lambda path: seen.append(path) or path.read_bytes(),
            )
            self.assertNotIn(policy.deps["labeled"][0], seen)

    def test_09_checkpoint_inventory_hash_and_lineage(self):
        rows, lineage = self.rows_and_lineage()
        result = ctl.checkpoint(rows, lineage)
        self.assertEqual(len(result), 20)
        self.assertEqual(result[0]["name"], "proj_ep1.pt")
        self.assertEqual(result[0]["sha256"], ctl.sha(b"checkpoint-1"))
        self.assertRaisesRegex(RuntimeError, "checkpoint_names", ctl.checkpoint, rows[:-1], lineage)
        self.assertRaisesRegex(RuntimeError, "checkpoint_lineage", ctl.checkpoint, rows, {})

    def test_10_selector_tie_eligibility_and_checkpoint_membership(self):
        case = self.case("case10")
        policy = self.policy(case)
        binding = ctl.verify(policy)
        rows, lineage = self.rows_and_lineage()
        checkpoints = ctl.checkpoint(rows, lineage)
        selector = self.selector(pool_sha=policy.deps["unlabeled"][1])
        for row in selector["epochs"]:
            row["worst_pre_reassign_noise"] = 5.0
            row["mean"] = 5.0
        self.assertEqual(ctl.select(selector, checkpoints, binding)["name"], "proj_ep1.pt")
        selector["epochs"][0]["eligible"] = False
        self.assertEqual(ctl.select(selector, checkpoints, binding)["name"], "proj_ep2.pt")
        self.assertRaisesRegex(
            RuntimeError,
            "selected_missing",
            ctl.select,
            self.selector(pool_sha=policy.deps["unlabeled"][1]),
            checkpoints[1:],
            binding,
        )

    def test_11_selector_schema_finite_epoch_and_recursive_firewall(self):
        case = self.case("case11")
        policy = self.policy(case)
        binding = ctl.verify(policy)
        pool_sha = policy.deps["unlabeled"][1]
        self.assertRaisesRegex(RuntimeError, "selector_schema", ctl.select, {"epochs": []}, [], binding)
        duplicate = self.selector(pool_sha=pool_sha)
        duplicate["epochs"][-1]["epoch"] = 1
        self.assertRaisesRegex(RuntimeError, "selector_epochs", ctl.select, duplicate, [], binding)
        nonfinite = self.selector(pool_sha=pool_sha)
        nonfinite["epochs"][0]["mean"] = math.nan
        self.assertRaisesRegex(RuntimeError, "selector_fields", ctl.select, nonfinite, [], binding)
        boolean = self.selector(pool_sha=pool_sha)
        boolean["epochs"][0]["mean"] = True
        self.assertRaisesRegex(RuntimeError, "selector_fields", ctl.select, boolean, [], binding)
        no_eligible = self.selector(eligible=False, pool_sha=pool_sha)
        self.assertRaisesRegex(RuntimeError, "selector_no_eligible", ctl.select, no_eligible, [], binding)
        leaked = self.selector(pool_sha=pool_sha)
        leaked["epochs"][0]["nested"] = {"ground_truth": 1}
        self.assertRaisesRegex(RuntimeError, "firewall", ctl.select, leaked, [], binding)
        wrong_pool = self.selector(pool_sha="0" * 64)
        self.assertRaisesRegex(RuntimeError, "selector_pool", ctl.select, wrong_pool, [], binding)
        for paths in (
            ["synthetic/002.png", "synthetic/001.png"],
            ["synthetic/001.png"],
            ["synthetic/001.png", "synthetic/003.png"],
            ["synthetic/001.png", "synthetic/002.png", "synthetic/003.png"],
            ["synthetic/001.png", "synthetic/001.png"],
            ["./relative.png", "synthetic/002.png"],
            ["synthetic/../escape.png", "synthetic/002.png"],
            ["synthetic//empty.png", "synthetic/002.png"],
            [r"synthetic\backslash.png", "synthetic/002.png"],
        ):
            candidate = self.selector(pool_sha=pool_sha, ordered_paths=paths)
            self.assertRaisesRegex(RuntimeError, "selector_paths", ctl.select, candidate, [], binding)
        for paths in (
            ["/absolute.png", "synthetic/002.png"],
            ["//server/share.png", "synthetic/002.png"],
            [r"C:\drive.png", "synthetic/002.png"],
        ):
            candidate = self.selector(pool_sha=pool_sha, ordered_paths=paths)
            self.assertRaisesRegex(RuntimeError, "firewall", ctl.select, candidate, [], binding)
        launch = json.loads(policy.r3.read_bytes())
        tampered = {
            "r3": {"path": ctl.lexical(policy.r3), "sha256": policy.r3_sha},
            "dependencies": launch["dependencies"],
            "unlabeled_binding": launch["unlabeled_binding"],
            "labeled_binding": dict(launch["labeled_binding"]),
            "scorer_program": ctl.SCORER_PROGRAM,
        }
        tampered["labeled_binding"]["canonical_path"] = ctl.lexical(case / "forged-label.json")
        tampered["self_sha256"] = ctl.sha(ctl.canon(tampered))
        self.assertRaisesRegex(
            RuntimeError,
            "binding_tamper",
            ctl.select,
            self.selector(pool_sha=pool_sha),
            [],
            tampered,
        )
        forged_case = self.case("case11_forged_seal")
        forged_paths = self.paths(forged_case)
        forged_paths.action.mkdir()
        self.assertRaisesRegex(
            RuntimeError,
            "binding_tamper",
            ctl.seal,
            forged_paths,
            self.selector(pool_sha=pool_sha),
            [],
            {"run": "safe"},
            tampered,
        )
        self.assertFalse((forged_paths.action / "rule_c").exists())
        self.assertFalse((forged_paths.action / "rule_c" / "label.reservation").exists())
        self.assertRaisesRegex(RuntimeError, "binding_tamper", ctl._VerifiedBinding)
        reconstructed = object.__new__(ctl._VerifiedBinding)
        self.assertRaisesRegex(
            RuntimeError,
            "binding_tamper",
            ctl.select,
            self.selector(pool_sha=pool_sha),
            [],
            reconstructed,
        )

        class BindingSubclass(ctl._VerifiedBinding):
            def __new__(cls):
                return object.__new__(cls)

        self.assertRaisesRegex(
            RuntimeError,
            "binding_tamper",
            ctl.select,
            self.selector(pool_sha=pool_sha),
            [],
            BindingSubclass(),
        )
        with self.assertRaises(AttributeError):
            binding.payload = tampered

    def test_12_seal_artifacts_and_recursive_meta_firewall(self):
        case = self.case("case12")
        paths = self.paths(case)
        paths.action.mkdir()
        policy = self.policy(case)
        binding = ctl.verify(policy)
        rows, lineage = self.rows_and_lineage()
        checkpoints = ctl.checkpoint(rows, lineage)
        selector = self.selector(pool_sha=policy.deps["unlabeled"][1])
        sealed = ctl.seal(paths, selector, checkpoints, {"launch_sha256": "c" * 64}, binding)
        self.assertIs(type(sealed), ctl._VerifiedSeal)
        self.assertRaisesRegex(RuntimeError, "seal_tamper", ctl._VerifiedSeal)
        v3 = paths.action / "rule_c" / "v3.json"
        epoch = paths.action / "rule_c" / "epoch.json"
        zero = paths.receipts / "seal.json"
        self.assertTrue(v3.is_file() and epoch.is_file() and zero.is_file())
        v3_payload = json.loads(v3.read_bytes())
        self.assertEqual(v3_payload["label_open_count"], 0)
        self.assertEqual(v3_payload["authorization"], ctl.nonlabel_binding(binding))
        self.assertEqual(
            v3_payload["authorization"]["r3"],
            {"path": ctl.lexical(policy.r3), "sha256": policy.r3_sha},
        )
        self.assertEqual(v3_payload["authorization"]["scorer_program"], ctl.SCORER_PROGRAM)
        self.assertEqual(
            set(v3_payload["authorization"]["dependencies"]),
            ctl.DEP_KEYS - {"labeled"},
        )
        self.assertEqual(
            v3_payload["authorization"]["dependencies"]["unlabeled"]["sha256"],
            policy.deps["unlabeled"][1],
        )
        self.assertEqual(
            v3_payload["selector"]["ordered_paths"],
            ["synthetic/001.png", "synthetic/002.png"],
        )
        self.assertEqual(v3_payload["selected"]["name"], "proj_ep1.pt")
        self.assertEqual(v3_payload["checkpoints"], checkpoints)
        self.assertEqual(v3_payload["meta"], {"launch_sha256": "c" * 64})
        self.assertNotIn("labeled", v3_payload["authorization"]["dependencies"])
        v3_text = v3.read_text(encoding="utf-8")
        self.assertNotIn(ctl.lexical(policy.deps["labeled"][0]), v3_text)
        self.assertNotIn(policy.deps["labeled"][1], v3_text)
        self.assertEqual(
            v3_payload["self_sha256"],
            ctl.sha(ctl.canon({key: value for key, value in v3_payload.items() if key != "self_sha256"})),
        )
        epoch_payload = json.loads(epoch.read_bytes())
        zero_payload = json.loads(zero.read_bytes())
        self.assertEqual(epoch_payload["v3_file_sha256"], ctl.sha(v3.read_bytes()))
        self.assertEqual(epoch_payload["v3_self_sha256"], v3_payload["self_sha256"])
        self.assertEqual(epoch_payload["selected"], v3_payload["selected"])
        self.assertEqual(epoch_payload["checkpoint_lineage"], checkpoints)
        self.assertEqual(zero_payload["v3_file_sha256"], ctl.sha(v3.read_bytes()))
        self.assertEqual(zero_payload["epoch_file_sha256"], ctl.sha(epoch.read_bytes()))
        self.assertEqual(
            (v3_payload["label_open_count"], epoch_payload["label_open_count"], zero_payload["label_open_count"]),
            (0, 0, 0),
        )
        blocked = self.case("case12_blocked")
        blocked_paths = self.paths(blocked)
        blocked_paths.action.mkdir()
        self.assertRaisesRegex(
            RuntimeError,
            "firewall",
            ctl.seal,
            blocked_paths,
            selector,
            checkpoints,
            {"nested": {"target": 1}},
            binding,
        )
        blocked_hash = self.case("case12_blocked_hash")
        blocked_hash_paths = self.paths(blocked_hash)
        blocked_hash_paths.action.mkdir()
        self.assertRaisesRegex(
            RuntimeError,
            "firewall",
            ctl.seal,
            blocked_hash_paths,
            selector,
            checkpoints,
            {"neutral": policy.deps["labeled"][1]},
            binding,
        )

    def test_13_offline_one_materialization_object_only_and_second_refusal(self):
        case = self.case("case13")
        paths = self.paths(case)
        paths.action.mkdir()
        payload = b'{"labels":[1,2,3]}'
        policy = self.policy(case, label_payload=payload)
        binding = ctl.verify(policy)
        rows, lineage = self.rows_and_lineage()
        selector = self.selector(pool_sha=policy.deps["unlabeled"][1])
        sealed = ctl.seal(
            paths, selector, ctl.checkpoint(rows, lineage), {"run": "safe"}, binding
        )
        zero = paths.receipts / "seal.json"
        label = policy.deps["labeled"][0]
        dependency_paths = {policy.r3} | {
            path for name, (path, _) in policy.deps.items() if name != "labeled"
        }
        counts = {"open": 0, "read": 0, "fstat": 0, "close": 0, "final_path": 0}
        original_open = Path.open
        original_read = ctl.read
        original_fstat = ctl.os.fstat
        original_final_path = ctl.final_path

        class Watched:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                self.stream.__enter__()
                return self

            def __exit__(self, *args):
                counts["close"] += 1
                return self.stream.__exit__(*args)

            def fileno(self):
                return self.stream.fileno()

            def read(self, *args):
                counts["read"] += 1
                return self.stream.read(*args)

        def open_spy(path, *args, **kwargs):
            if ctl.lexical(path) == ctl.lexical(label):
                self.assertTrue((paths.action / "rule_c" / "label.reservation").is_file())
                self.assertEqual(args[0] if args else kwargs.get("mode", "r"), "rb")
                counts["open"] += 1
                return Watched(original_open(path, *args, **kwargs))
            return original_open(path, *args, **kwargs)

        def fstat_spy(fd):
            counts["fstat"] += 1
            return original_fstat(fd)

        def final_path_spy(stream):
            counts["final_path"] += 1
            return original_final_path(stream)

        def no_dependency_reread(path):
            self.assertNotIn(path, dependency_paths)
            return original_read(path)

        with (
            mock.patch.object(Path, "open", open_spy),
            mock.patch.object(ctl.os, "fstat", fstat_spy),
            mock.patch.object(ctl, "final_path", final_path_spy),
            mock.patch.object(ctl, "read", no_dependency_reread),
        ):
            result_sha = ctl.offline(paths, sealed)
            first_counts = dict(counts)
            self.assertRaisesRegex(RuntimeError, "atomic_collision", ctl.offline, paths, sealed)
            self.assertEqual(counts, first_counts)
        self.assertEqual(counts, {"open": 1, "read": 1, "fstat": 1, "close": 1, "final_path": 1})
        receipt = json.loads((paths.receipts / "offline.json").read_bytes())
        self.assertEqual(receipt["scope"], "synthetic_governance_only")
        self.assertEqual(receipt["scorer_program"], ctl.SCORER_PROGRAM)
        self.assertEqual(receipt["scorer_program_sha256"], policy.deps["scorer"][1])
        self.assertEqual(receipt["score"], {"count": 3})
        self.assertEqual(receipt["label_open_count"], 1)
        self.assertEqual(receipt["zero_file_sha256"], ctl.sha(zero.read_bytes()))
        self.assertEqual(receipt["label_identity"]["nlink"], 1)
        reservation = paths.action / "rule_c" / "label.reservation"
        self.assertEqual(receipt["reservation_sha256"], ctl.sha(reservation.read_bytes()))
        self.assertEqual(result_sha, ctl.sha((paths.receipts / "offline.json").read_bytes()))
        receipt_text = (paths.receipts / "offline.json").read_text(encoding="utf-8").casefold()
        for forbidden_claim in ("ari", "ami", "capture", "deployment_ready", "production_ready", "launch_ready"):
            self.assertNotIn(forbidden_claim, receipt_text)
        for fn in (ctl.offline, ctl.core):
            parameters = inspect.signature(fn).parameters
            self.assertNotIn("score", parameters)
            self.assertNotIn("scoring_spec", parameters)
            self.assertNotIn("offline_spec_factory", parameters)
        for capability in ("binding", "label", "opener", "handle", "policy"):
            self.assertNotIn(capability, inspect.signature(ctl.offline).parameters)

    def test_14_offline_seal_tamper_and_crash_reservation(self):
        rows, lineage = self.rows_and_lineage()

        def sealed(
            name,
            *,
            label_size=None,
            label_device=AUTO_IDENTITY,
            label_inode=AUTO_IDENTITY,
            wrong_hash=False,
        ):
            case = self.case(name)
            paths = self.paths(case)
            paths.action.mkdir()
            policy = self.policy(
                case,
                label_payload=b'{"labels":[]}',
                label_size=label_size,
                label_device=label_device,
                label_inode=label_inode,
            )
            if wrong_hash:
                body = json.loads(policy.r3.read_bytes())
                body["dependencies"]["labeled"]["sha256"] = "0" * 64
                body["labeled_binding"]["sha256"] = "0" * 64
                raw = ctl.canon(body)
                policy.r3.write_bytes(raw)
                deps = dict(policy.deps)
                deps["labeled"] = (deps["labeled"][0], "0" * 64)
                policy = ctl.LaunchPolicy(policy.r3, ctl.sha(raw), deps)
            binding = ctl.verify(policy)
            selector = self.selector(pool_sha=policy.deps["unlabeled"][1])
            seal_token = ctl.seal(
                paths, selector, ctl.checkpoint(rows, lineage), {"run": "safe"}, binding
            )
            return case, paths, policy, binding, seal_token

        mutations = (
            ("v3_self", "v3", "self_sha256", "0" * 64),
            ("v3_count", "v3", "label_open_count", 1),
            ("epoch_v3_file", "epoch", "v3_file_sha256", "0" * 64),
            ("epoch_v3_self", "epoch", "v3_self_sha256", "0" * 64),
            ("epoch_selected", "epoch", "selected", {"name": "wrong"}),
            ("epoch_lineage", "epoch", "checkpoint_lineage", []),
            ("epoch_count", "epoch", "label_open_count", 1),
            ("zero_v3", "zero", "v3_file_sha256", "0" * 64),
            ("zero_epoch", "zero", "epoch_file_sha256", "0" * 64),
            ("zero_count", "zero", "label_open_count", 1),
        )
        for name, target_name, key, value in mutations:
            _, paths, _, _, seal_token = sealed(f"case14_tamper_{name}")
            v3 = paths.action / "rule_c" / "v3.json"
            epoch = paths.action / "rule_c" / "epoch.json"
            zero = paths.receipts / "seal.json"
            target = {"v3": v3, "epoch": epoch, "zero": zero}[target_name]
            payload = json.loads(target.read_bytes())
            payload[key] = value
            target.write_bytes(ctl.canon(payload))
            self.assertRaisesRegex(RuntimeError, "seal_tamper", ctl.offline, paths, seal_token)
            self.assertFalse((paths.action / "rule_c" / "label.reservation").exists())

        _, paths, _, _, seal_token = sealed("case14_coherent_rewrite")
        v3 = paths.action / "rule_c" / "v3.json"
        epoch = paths.action / "rule_c" / "epoch.json"
        zero = paths.receipts / "seal.json"
        v3_payload = json.loads(v3.read_bytes())
        v3_payload["meta"]["coherent"] = "rewrite"
        v3_payload["self_sha256"] = ctl.sha(
            ctl.canon({key: value for key, value in v3_payload.items() if key != "self_sha256"})
        )
        v3.write_bytes(ctl.canon(v3_payload))
        epoch_payload = json.loads(epoch.read_bytes())
        epoch_payload["v3_file_sha256"] = ctl.sha(v3.read_bytes())
        epoch_payload["v3_self_sha256"] = v3_payload["self_sha256"]
        epoch.write_bytes(ctl.canon(epoch_payload))
        zero_payload = json.loads(zero.read_bytes())
        zero_payload["v3_file_sha256"] = ctl.sha(v3.read_bytes())
        zero_payload["epoch_file_sha256"] = ctl.sha(epoch.read_bytes())
        zero.write_bytes(ctl.canon(zero_payload))
        self.assertRaisesRegex(RuntimeError, "seal_tamper", ctl.offline, paths, seal_token)
        self.assertFalse((paths.action / "rule_c" / "label.reservation").exists())

        _, paths, _, _, seal_token = sealed("case14_forged_seal")
        self.assertRaisesRegex(RuntimeError, "seal_tamper", ctl.offline, paths, {})
        reconstructed_seal = object.__new__(ctl._VerifiedSeal)
        self.assertRaisesRegex(RuntimeError, "seal_tamper", ctl.offline, paths, reconstructed_seal)

        class SealSubclass(ctl._VerifiedSeal):
            def __new__(cls):
                return object.__new__(cls)

        self.assertRaisesRegex(RuntimeError, "seal_tamper", ctl.offline, paths, SealSubclass())
        with self.assertRaises(AttributeError):
            seal_token.payload = {}
        other_case = self.case("case14_binding_mismatch")
        other_paths = self.paths(other_case)
        self.assertRaisesRegex(RuntimeError, "seal_tamper", ctl.offline, other_paths, seal_token)
        self.assertFalse((paths.action / "rule_c" / "label.reservation").exists())

        _, paths, _, _, seal_token = sealed("case14_single_capture")
        v3 = paths.action / "rule_c" / "v3.json"
        epoch = paths.action / "rule_c" / "epoch.json"
        zero = paths.receipts / "seal.json"
        reads = {v3: 0, epoch: 0, zero: 0}
        original_read = ctl.read

        def read_once(path):
            if path in reads:
                reads[path] += 1
                if reads[path] > 1:
                    raise AssertionError("second seal read")
            return original_read(path)

        with mock.patch.object(ctl, "read", read_once):
            ctl.offline(paths, seal_token)
        self.assertEqual(reads, {v3: 1, epoch: 1, zero: 1})

        _, _, policy, _, _ = sealed("case14_inert")
        label_path = ctl.lexical(policy.deps["labeled"][0])
        label_sha = policy.deps["labeled"][1]
        called = []

        class EvilList(list):
            def __iter__(self):
                called.append("list_iter")
                return super().__iter__()

            def __len__(self):
                called.append("list_len")
                return super().__len__()

        class EvilDict(dict):
            def items(self):
                called.append("dict_items")
                return super().items()

            def values(self):
                called.append("dict_values")
                return super().values()

        class EvilString(str):
            def casefold(self):
                called.append("string_casefold")
                return super().casefold()

        class EvilInt(int):
            def __int__(self):
                called.append("int_convert")
                return super().__int__()

        class EvilFloat(float):
            def __float__(self):
                called.append("float_convert")
                return super().__float__()

        bad_values = (
            lambda: called.append("called"),
            Path("labels.json"),
            b"{}",
            bytearray(b"{}"),
            math.nan,
            {1: "x"},
            {EvilString("neutral"): 1},
            {"label_path": "x"},
            EvilList([1]),
            EvilDict({"x": 1}),
            EvilString("x"),
            EvilInt(1),
            EvilFloat(1.0),
        )
        for bad in bad_values:
            self.assertRaisesRegex(RuntimeError, "offline_spec", ctl.inert, bad, label_path, label_sha)
        for bad_string in (
            label_path,
            label_sha,
            r"C:\absolute",
            "C:/absolute",
            "C:drive-relative",
            r"\\server\share",
            "//server/share",
            r"\\?\C:\device",
            r"\\.\device",
            r"\single-rooted",
            "/rooted",
            "label_path",
            "label_hash",
            "label_identity",
            "opener",
            "handle",
            "reader",
            "callback",
            "factory",
            "identity",
        ):
            self.assertRaisesRegex(
                RuntimeError,
                "offline_spec",
                ctl.inert,
                {"neutral": bad_string},
                label_path,
                label_sha,
            )
        for capability_key in ("opener", "handle", "reader", "callback", "factory", "identity"):
            self.assertRaisesRegex(
                RuntimeError,
                "offline_spec",
                ctl.inert,
                {capability_key: "relative"},
                label_path,
                label_sha,
            )
        self.assertEqual(called, [])
        self.assertEqual(
            ctl.inert({"ok": [1, True, None, 1.5, "relative/path"]}, label_path, label_sha),
            {"ok": [1, True, None, 1.5, "relative/path"]},
        )

        def permanent_failure(name, mode, **policy_kwargs):
            case, paths, policy, _, seal_token = sealed(name, **policy_kwargs)
            label = policy.deps["labeled"][0]
            opens = {"label": 0}
            original_open = Path.open

            def open_spy(path, *args, **kwargs):
                if ctl.lexical(path) == ctl.lexical(label):
                    self.assertTrue((paths.action / "rule_c" / "label.reservation").is_file())
                    opens["label"] += 1
                return original_open(path, *args, **kwargs)

            with ExitStack() as stack:
                stack.enter_context(mock.patch.object(Path, "open", open_spy))
                if mode == "final_path":
                    stack.enter_context(
                        mock.patch.object(ctl, "final_path", return_value=ctl.lexical(case / "alternate.json"))
                    )
                elif mode in ("reparse", "parent_reparse"):
                    original_lstat = ctl.lstat
                    target = label if mode == "reparse" else label.parent

                    def lstat_spy(path):
                        result = original_lstat(path)
                        if Path(path) == target:
                            return mock.Mock(
                                st_mode=result.st_mode,
                                st_file_attributes=getattr(ctl.stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400),
                                st_nlink=1,
                            )
                        return result

                    stack.enter_context(mock.patch.object(ctl, "lstat", lstat_spy))
                elif mode == "nlink":
                    original_fstat = ctl.os.fstat

                    def fstat_spy(fd):
                        result = original_fstat(fd)
                        return mock.Mock(
                            st_mode=result.st_mode,
                            st_file_attributes=0,
                            st_nlink=2,
                            st_size=result.st_size,
                            st_dev=result.st_dev,
                            st_ino=result.st_ino,
                        )

                    stack.enter_context(mock.patch.object(ctl.os, "fstat", fstat_spy))
                elif mode == "interpreter":
                    stack.enter_context(
                        mock.patch.object(ctl, "interpret", side_effect=RuntimeError("scorer_input"))
                    )
                error = "scorer_input" if mode == "interpreter" else "label_identity"
                self.assertRaisesRegex(RuntimeError, error, ctl.offline, paths, seal_token)
                first_opens = opens["label"]
                self.assertTrue((paths.action / "rule_c" / "label.reservation").is_file())
                self.assertRaisesRegex(RuntimeError, "atomic_collision", ctl.offline, paths, seal_token)
                self.assertEqual(opens["label"], first_opens)

        permanent_failure("case14_size", "size", label_size=999)
        permanent_failure("case14_hash", "hash", wrong_hash=True)
        permanent_failure("case14_device", "device", label_device=-1)
        permanent_failure("case14_inode", "inode", label_inode=-1)
        permanent_failure("case14_final_path", "final_path")
        permanent_failure("case14_reparse", "reparse")
        permanent_failure("case14_parent_reparse", "parent_reparse")
        permanent_failure("case14_nlink", "nlink")
        permanent_failure("case14_interpreter", "interpreter")

        case = self.case("case14_production_shape")
        paths = self.paths(case)
        paths.action.mkdir()
        policy = self.policy(case, label_payload=b'{"labels":[]}')
        actual_label = policy.deps["labeled"][0]
        production_shape = Path(
            r"D:\project\unknown-contrastive\data\pools\v2\unknown\strict_novel_val.json"
        )
        body = json.loads(policy.r3.read_bytes())
        body["dependencies"]["labeled"]["path"] = ctl.lexical(production_shape)
        body["labeled_binding"]["canonical_path"] = ctl.lexical(production_shape)
        raw = ctl.canon(body)
        policy.r3.write_bytes(raw)
        deps = dict(policy.deps)
        deps["labeled"] = (production_shape, deps["labeled"][1])
        policy = ctl.LaunchPolicy(policy.r3, ctl.sha(raw), deps)
        binding = ctl.verify(policy)
        selector = self.selector(pool_sha=policy.deps["unlabeled"][1])
        seal_token = ctl.seal(
            paths, selector, ctl.checkpoint(rows, lineage), {"run": "safe"}, binding
        )
        safe_label_calls = []
        original_safe = ctl.safe

        def safe_spy(path):
            if ctl.lexical(path) == ctl.lexical(production_shape):
                safe_label_calls.append(path)
            return original_safe(path)

        def target_spy(path):
            self.assertEqual(ctl.lexical(path), ctl.lexical(production_shape))
            self.assertTrue((paths.action / "rule_c" / "label.reservation").is_file())
            return actual_label

        with (
            mock.patch.object(ctl, "safe", safe_spy),
            mock.patch.object(ctl, "label_target", target_spy),
            mock.patch.object(ctl, "final_path", return_value=ctl.lexical(production_shape)),
        ):
            ctl.offline(paths, seal_token)
        self.assertEqual(safe_label_calls, [])

    def test_15_core_success_order_receipts_and_cleanup(self):
        case = self.case("case15")
        kwargs, events = self.harness(case)
        production_calls = []
        production_root = ctl.lexical(ctl.PRODUCTION_ROOT)

        def production_shaped(path):
            value = ctl.lexical(Path(path))
            return value == production_root or value.startswith(production_root + os.sep)

        def guarded_path_method(name):
            original = getattr(Path, name)

            def call(path, *args, **kwargs):
                if production_shaped(path):
                    production_calls.append((name, str(path)))
                    raise AssertionError(f"live production probe:{name}")
                return original(path, *args, **kwargs)

            return call

        original_lstat = ctl.lstat

        def guarded_lstat(path):
            if production_shaped(path):
                production_calls.append(("ctl.lstat", str(path)))
                raise AssertionError("live production probe:ctl.lstat")
            return original_lstat(path)

        with ExitStack() as stack:
            for name in (
                "exists",
                "open",
                "is_file",
                "is_dir",
                "is_symlink",
                "stat",
                "lstat",
                "mkdir",
                "iterdir",
                "rglob",
                "unlink",
            ):
                stack.enter_context(mock.patch.object(Path, name, guarded_path_method(name)))
            stack.enter_context(mock.patch.object(ctl, "lstat", guarded_lstat))
            result = ctl.core(**kwargs)
        expected = [
            "snapshot:1",
            "acquire:run.lock",
            "probe:True",
            "acquire:gpu.lock",
            "stop:False",
            "probe:True",
            "snapshot:2",
            "trainer_inventory",
            "train",
            "trainer_inventory",
            "checkpoint_loader",
            "selector",
            "release:gpu.lock",
            "release:run.lock",
            "snapshot:3",
        ]
        self.assertEqual(events, expected)
        self.assertEqual(result, ctl.sha((kwargs["paths"].receipts / "offline.json").read_bytes()))
        for name in ("started", "train", "selector", "seal", "offline", "terminal"):
            self.assertTrue((kwargs["paths"].receipts / f"{name}.json").is_file(), name)
        self.assertFalse((kwargs["paths"].receipts / "failed.json").exists())
        self.assertFalse(kwargs["paths"].run.exists())
        self.assertFalse(kwargs["paths"].gpu.exists())
        self.assertEqual(production_calls, [])

    def test_16_core_stop_and_timeout_leave_no_action_root(self):
        stop_case = self.case("case16_stop")
        kwargs, events = self.harness(stop_case, probes=(False,), stopped=True)
        self.assertRaisesRegex(RuntimeError, "^stop$", ctl.core, **kwargs)
        self.assertFalse(kwargs["paths"].action.exists())
        self.assertIn("release:run.lock", events)
        timeout_case = self.case("case16_timeout")
        kwargs, events = self.harness(timeout_case, probes=(False,), stopped=False)
        kwargs["max_wait_seconds"] = 0
        self.assertRaisesRegex(RuntimeError, "^timeout$", ctl.core, **kwargs)
        self.assertFalse(kwargs["paths"].action.exists())
        self.assertIn("release:run.lock", events)

    def test_17_core_post_lease_loss_releases_both_locks(self):
        case = self.case("case17")
        kwargs, events = self.harness(case, probes=(True, False))
        self.assertRaisesRegex(RuntimeError, "post_lease_loss", ctl.core, **kwargs)
        self.assertFalse(kwargs["paths"].action.exists())
        self.assertIn("release:gpu.lock", events)
        self.assertIn("release:run.lock", events)
        drift_case = self.case("case17_drift")
        kwargs, events = self.harness(drift_case, drift_at=2)
        self.assertRaisesRegex(RuntimeError, "post_lease_drift", ctl.core, **kwargs)
        self.assertFalse(kwargs["paths"].action.exists())
        self.assertNotIn("train", events)
        self.assertIn("release:gpu.lock", events)
        self.assertIn("release:run.lock", events)

    def test_18_core_root_collision_is_non_mutating(self):
        case = self.case("case18")
        kwargs, _ = self.harness(case)
        kwargs["paths"].action.mkdir()
        marker = kwargs["paths"].action / "marker"
        marker.write_text("preserve")
        self.assertRaisesRegex(RuntimeError, "root_collision", ctl.core, **kwargs)
        self.assertEqual(marker.read_text(), "preserve")
        self.assertFalse(kwargs["paths"].receipts.exists())

    def test_19_core_zero_and_multiple_trainer_outputs_fail(self):
        zero_case = self.case("case19_zero")
        kwargs, _ = self.harness(zero_case, trainer_mode="zero")
        self.assertRaisesRegex(RuntimeError, "trainer_collision", ctl.core, **kwargs)
        self.assertTrue((kwargs["paths"].receipts / "failed.json").is_file())
        self.assertFalse((kwargs["paths"].receipts / "terminal.json").exists())
        multiple_case = self.case("case19_multiple")
        kwargs, _ = self.harness(multiple_case, trainer_mode="multiple")
        self.assertRaisesRegex(RuntimeError, "trainer_collision", ctl.core, **kwargs)
        self.assertTrue((kwargs["paths"].receipts / "failed.json").is_file())

    def test_20_core_selector_failure_has_no_label_materialization(self):
        case = self.case("case20")
        kwargs, events = self.harness(case, selector_mode="invalid")
        self.assertRaisesRegex(RuntimeError, "selector_schema", ctl.core, **kwargs)
        self.assertNotIn("score", events)
        self.assertFalse((kwargs["paths"].receipts / "selector.json").exists())
        self.assertFalse((kwargs["paths"].action / "rule_c").exists())
        self.assertFalse((kwargs["paths"].action / "rule_c" / "label.reservation").exists())
        self.assertTrue((kwargs["paths"].receipts / "failed.json").is_file())
        self.assertFalse((kwargs["paths"].receipts / "terminal.json").exists())
        mismatch_case = self.case("case20_pool")
        kwargs, events = self.harness(mismatch_case)
        kwargs["selector_phase"] = lambda _: self.selector(pool_sha="0" * 64)
        self.assertRaisesRegex(RuntimeError, "selector_pool", ctl.core, **kwargs)
        self.assertFalse((kwargs["paths"].receipts / "selector.json").exists())
        self.assertFalse((kwargs["paths"].action / "rule_c").exists())
        self.assertNotIn("score", events)

    def test_21_core_snapshot_drift_blocks_terminal(self):
        case = self.case("case21")
        kwargs, _ = self.harness(case, drift_at=3)
        self.assertRaisesRegex(RuntimeError, "immutable_drift", ctl.core, **kwargs)
        failed = json.loads((kwargs["paths"].receipts / "failed.json").read_bytes())
        self.assertEqual(failed["error"], "immutable_drift")
        self.assertFalse((kwargs["paths"].receipts / "terminal.json").exists())

    def test_22_core_release_failure_attempts_both_and_blocks_terminal(self):
        case = self.case("case22")
        kwargs, events = self.harness(case, release_fail_once=("gpu.lock",))
        self.assertRaisesRegex(RuntimeError, "release_failed:gpu.lock", ctl.core, **kwargs)
        self.assertGreaterEqual(events.count("release:gpu.lock"), 2)
        self.assertIn("release:run.lock", events)
        self.assertTrue((kwargs["paths"].receipts / "failed.json").is_file())
        self.assertFalse((kwargs["paths"].receipts / "terminal.json").exists())
        run_case = self.case("case22_run")
        kwargs, events = self.harness(run_case, release_fail_once=("run.lock",))
        self.assertRaisesRegex(RuntimeError, "release_failed:run.lock", ctl.core, **kwargs)
        self.assertTrue((kwargs["paths"].receipts / "failed.json").is_file())
        self.assertFalse((kwargs["paths"].receipts / "terminal.json").exists())
        combined_case = self.case("case22_combined")
        kwargs, events = self.harness(
            combined_case,
            drift_at=2,
            release_fail_once=("gpu.lock", "run.lock"),
        )
        self.assertRaisesRegex(RuntimeError, "post_lease_drift", ctl.core, **kwargs)
        self.assertIn("release:gpu.lock", events)
        self.assertIn("release:run.lock", events)
        self.assertFalse(kwargs["paths"].action.exists())
        started_case = self.case("case22_combined_started")
        kwargs, events = self.harness(
            started_case,
            trainer_mode="zero",
            drift_at=3,
            release_fail_once=("gpu.lock", "run.lock"),
        )
        self.assertRaisesRegex(RuntimeError, "trainer_collision", ctl.core, **kwargs)
        failed = json.loads((kwargs["paths"].receipts / "failed.json").read_bytes())
        self.assertEqual(failed["error"], "trainer_collision")
        self.assertIn("release:gpu.lock", events)
        self.assertIn("release:run.lock", events)
        self.assertFalse((kwargs["paths"].receipts / "terminal.json").exists())

    def test_23_state_hash_state_content_and_snapshot_keyset_fail_before_locks(self):
        hash_case = self.case("case23_hash")
        kwargs, events = self.harness(hash_case)
        kwargs["expected_state_sha"] = "0" * 64
        self.assertRaisesRegex(RuntimeError, "state_hash", ctl.core, **kwargs)
        self.assertFalse(any(event.startswith("acquire:") for event in events))
        state_case = self.case("case23_state")
        kwargs, events = self.harness(state_case)
        invalid = ctl.canon({"state": "RUNNING", "queue": []})
        kwargs["state_bytes"] = invalid
        kwargs["expected_state_sha"] = ctl.sha(invalid)
        self.assertRaisesRegex(RuntimeError, "state_guard", ctl.core, **kwargs)
        self.assertFalse(any(event.startswith("acquire:") for event in events))
        snapshot_case = self.case("case23_snapshot")
        kwargs, events = self.harness(snapshot_case)
        kwargs["snapshotter"] = lambda: {"state": "only"}
        self.assertRaisesRegex(RuntimeError, "snapshot_keyset", ctl.core, **kwargs)
        self.assertFalse(any(event.startswith("acquire:") for event in events))
        firewall_case = self.case("case23_snapshot_firewall")
        kwargs, events = self.harness(firewall_case)
        snapshot = self.snapshot()
        snapshot["state"] = kwargs["expected_state_sha"]
        snapshot["action"] = "handle"
        kwargs["snapshotter"] = lambda: snapshot
        constructed = []

        class LockTrap:
            def __init__(self, path):
                constructed.append(path)

        kwargs["locks"] = LockTrap
        self.assertRaisesRegex(RuntimeError, "firewall", ctl.core, **kwargs)
        self.assertEqual(constructed, [])
        self.assertFalse(any(event.startswith(("acquire:", "probe:", "train", "selector")) for event in events))
        self.assertFalse((kwargs["paths"].action / "rule_c" / "label.reservation").exists())


if __name__ == "__main__":
    unittest.main()
