import hashlib
import inspect
import json

import pytest

from scripts import run_fusion_alpha_offline as worker2
from scripts import run_fusion_alpha_screen as controller
from scripts import run_fusion_alpha_unlabeled as worker1
from scripts.fusion_alpha_common import DIALS, INTERIOR_ALPHAS, canonical_bytes, seal_hash


def _seal(alpha=.2, screen="a" * 64):
    value = {"schema": "fusion_alpha.selection_seal.v2", "screen_sha256": screen,
             "selected_alpha": alpha, "status": "selected", "labels_used": False,
             "r3_file_sha256": controller.R3_FILE_SHA256}
    value["seal_sha256"] = seal_hash(value)
    return value


def _science(strict=True):
    value = {"P1_unique_dominant_capture": 1, "macro_image_cap": 1, "minimum_image_cap": 1,
             "purity_weighted": 1, "ARI": 1, "AMI": 1, "pre_reassign_noise": 1,
             "post_reassign_noise": 1, "fragmentation": 1, "lost_classes": [], "captured_classes": ["a"]}
    if strict:
        value["ARI"] = 2
    return value


def test_argument_and_import_firewalls_are_real():
    assert not worker1.parser()._actions[1:]
    names = {action.dest for action in worker2.parser()._actions}
    assert names == {"help", "expected_seal_sha256"}
    assert all(token not in inspect.getsource(worker2.parser) for token in ("--alpha", "--grid", "--q75", "--ranking"))
    source = inspect.getsource(controller)
    assert "import fusion_alpha_common" not in source and "import numpy" not in source
    assert "json.loads" not in inspect.getsource(controller.verified_seal)
    assert worker1.opaque_alignment(["opaque/a"], ["opaque/a"])
    assert not worker1.opaque_alignment(["opaque/a"], ["opaque/b"])


def test_worker2_real_single_label_open_success_and_scientific_failure(tmp_path, monkeypatch):
    root = tmp_path / "root"; root.mkdir()
    seal = _seal(); raw = canonical_bytes(seal); (root / "selection_seal.json").write_bytes(raw)
    labels = tmp_path / "labels.json"; labels_raw = json.dumps({"files": [{"path": "x", "label": "a"}]}).encode(); labels.write_bytes(labels_raw)
    monkeypatch.setattr(worker2, "ROOT", root); monkeypatch.setattr(worker2, "LABELED", labels)
    monkeypatch.setitem(worker2.HASHES, "labeled", hashlib.sha256(labels_raw).hexdigest())
    order = []
    monkeypatch.setattr(worker2, "preflight_offline_inputs", lambda opened: (order.append("preflight") or ["x"], None, None, opened))
    monkeypatch.setattr(worker2, "_frozen_rows", lambda opened: (order.append("frozen") or {dial: _science(False) for dial in DIALS}))
    actual_label_read = worker2._read_labeled_manifest_once
    monkeypatch.setattr(worker2, "_read_labeled_manifest_once", lambda opened: (order.append("label") or actual_label_read(opened)))
    monkeypatch.setattr(worker2, "fuse", lambda f, p, alpha: object())
    monkeypatch.setattr(worker2, "_score", lambda embedding, labels: {dial: _science(True) for dial in DIALS})
    output = worker2.run_offline(hashlib.sha256(raw).hexdigest())
    assert output["selected_alpha"] == .2
    assert order == ["preflight", "frozen", "label"]
    receipt = json.loads((root / "offline_process_receipt.json").read_text())
    assert receipt["label_open_count"] == 1 and receipt["labels_used"] is True
    failed = tmp_path / "failed"; failed.mkdir(); raw2 = canonical_bytes(_seal()); (failed / "selection_seal.json").write_bytes(raw2)
    monkeypatch.setattr(worker2, "ROOT", failed)
    monkeypatch.setattr(worker2, "_score", lambda embedding, labels: {dial: _science(False) for dial in DIALS})
    with pytest.raises(RuntimeError, match="mandatory frozen gate"):
        worker2.run_offline(hashlib.sha256(raw2).hexdigest())
    assert (failed / "offline_process_receipt.json").is_file()


def test_controller_orders_fresh_workers_after_seal_barrier(tmp_path, monkeypatch):
    root = tmp_path / "root"; order = []; launches = []
    monkeypatch.setattr(controller, "ROOT", root)
    monkeypatch.setattr(controller, "verify_controller_bindings", lambda: order.append("bindings"))
    monkeypatch.setattr(controller, "run_required_tests",
                        lambda: (order.append("tests") or controller.source_and_test_hashes()))
    monkeypatch.setattr(controller, "LABELED_MANIFEST", tmp_path / "labels.json")
    class Process:
        next_pid = 10
        def __init__(self, command, *, cwd, env):
            self.command = command; self.cwd = cwd; self.env = env
            self.worker = "worker1" if "scripts.run_fusion_alpha_unlabeled" in command else "worker2"
            self.pid = Process.next_pid; Process.next_pid += 1
            launches.append({"command": command, "cwd": cwd, "cuda": env["CUDA_VISIBLE_DEVICES"]})
            order.append(self.worker)
        def wait(self):
            if self.worker == "worker1":
                (root / "unlabeled_screen.json").write_text("{}")
                screen = hashlib.sha256((root / "unlabeled_screen.json").read_bytes()).hexdigest()
                (root / "selection_seal.json").write_bytes(canonical_bytes(_seal(screen=screen)))
                seal_digest = hashlib.sha256((root / "selection_seal.json").read_bytes()).hexdigest()
                (root / "unlabeled_process_receipt.json").write_text(json.dumps({"status": "selected", "selection_seal_sha256": seal_digest, "label_open_count": 0}))
            else:
                (root / "offline_selected_alpha.json").write_text("{}")
                output_digest = hashlib.sha256((root / "offline_selected_alpha.json").read_bytes()).hexdigest()
                seal_digest = hashlib.sha256((root / "selection_seal.json").read_bytes()).hexdigest()
                (root / "offline_process_receipt.json").write_text(json.dumps({
                    "worker": "offline", "status": "passed", "selection_seal_sha256": seal_digest,
                    "selected_alpha": .2, "label_open_count": 1, "labels_used": True,
                    "labels_used_after_selection": True,
                    "offline_selected_alpha_sha256": output_digest,
                    "opened_paths": [{"path": str(controller.LABELED_MANIFEST), "sha256": "x" * 64}],
                }))
            return 0
        def poll(self): return 0
    monkeypatch.setattr(controller.subprocess, "Popen", Process)
    controller.main([])
    assert order == ["bindings", "tests", "worker1", "worker2"]
    seal_digest = hashlib.sha256((root / "selection_seal.json").read_bytes()).hexdigest()
    assert launches == [
        {"command": [controller.sys.executable, "-B", "-m", "scripts.run_fusion_alpha_unlabeled"],
         "cwd": str(controller.PROJECT_ROOT), "cuda": ""},
        {"command": [controller.sys.executable, "-B", "-m", "scripts.run_fusion_alpha_offline",
                     "--expected-seal-sha256", seal_digest],
         "cwd": str(controller.PROJECT_ROOT), "cuda": ""},
    ]
    assert json.loads((root / "result_receipt.json").read_text())["status"] == "complete"
    created = json.loads((root / "create_new_receipt.json").read_text())
    assert created["pytest_command"] == controller.PYTEST_COMMAND and created["pytest_status"] == "passed"
    assert created["worker_environment"]["CUDA_VISIBLE_DEVICES"] == ""


def test_controller_rejects_tampered_seal_and_offline_receipt(tmp_path, monkeypatch):
    seal = _seal()
    path = tmp_path / "seal.json"
    path.write_bytes(canonical_bytes(seal))
    status, digest, _, alpha = controller.verified_seal(path)
    assert (status, alpha) == ("selected", .2)
    raw = path.read_bytes().replace(b'"labels_used":false', b'"labels_used":true')
    path.write_bytes(raw)
    with pytest.raises(RuntimeError, match="seal"):
        controller.verified_seal(path)

    output = tmp_path / "offline.json"
    output.write_text("{}")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({
        "worker": "offline", "status": "passed", "selection_seal_sha256": digest,
        "selected_alpha": .2, "label_open_count": 0, "labels_used": True,
        "labels_used_after_selection": True,
        "offline_selected_alpha_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "opened_paths": [],
    }))
    with pytest.raises(RuntimeError, match="offline receipt"):
        controller.verified_worker2_receipt(receipt, output, digest, .2)


def test_existing_root_is_refused_without_alternate_path(tmp_path, monkeypatch):
    root = tmp_path / "root"; root.mkdir(); monkeypatch.setattr(controller, "ROOT", root)
    with pytest.raises(RuntimeError, match="already exists"):
        controller.create_root()
