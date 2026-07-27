"""Standard-library-only controller for the exact R3 fusion-alpha action."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(r"D:\project\unknown-contrastive\runs\campaign_state\fusion\lr008_ep12_frozen_projection_alpha_screen_260727_v1")
PROJECT_ROOT = Path(r"D:\project\unknown-contrastive")
R3 = Path(r"D:\project\unknown-contrastive\runs\campaign_state\panels\lr008_result_260727_v3\r3_C.json")
EVIDENCE = R3.with_name("evidence_packet.json")
R3_FILE_SHA256 = "27df3ed5745aba21d62f4b83a8af06665301c88a96c80cebae664a399850ccde"
EVIDENCE_FILE_SHA256 = "6faab6a206ac343f538e3b8c1d2a6a827fbbf15c55f4467573d75fd24ba96fd8"
DEPENDENCIES = {
    Path(r"D:\project\unknown-contrastive\_grouping_eval.py"): "37435a5d59c5d2b1ef5fff8115c2d4169f4af328439a4d4de71948826199fa0c",
    Path(r"D:\project\unknown-contrastive\scripts\_common.py"): "eb0aa9433c45d024ae49f4162185d691bd44e564577c2cd6be1459af5c9490fb",
    Path(r"D:\project\unknown-contrastive\scripts\cluster_metrics.py"): "c4a5218d1df11294820818ebdca1c63c44c57c117e2223aab353f347451708c5",
    Path(r"D:\project\unknown-contrastive\scripts\eval_open_set_embeddings.py"): "e5e04686b46629ec609f01bac7bdf945cf916d4784e751d4d735f94c00923a08",
    Path(r"D:\project\unknown-contrastive\scripts\run_rule_c_selector.py"): "9b65cf4ff35b027621a273a2bd5b1ef48bc63d13e20c28be85aca3b8a08e9cc6",
    Path(r"D:\project\unknown-contrastive\scripts\run_rule_c_v3_reselector.py"): "c49ee6458148345dfc46ded29ea16e552fda53495f2c08c2c6ce930278b6dcb3",
    Path(r"D:\project\unknown-contrastive\scripts\run_rule_c_offline.py"): "3198a1da6e85a029dbdfa03211f04b3645ae241ee6dc4e879ffcdf737aa2769d",
}
TESTS = (Path(r"D:\project\unknown-contrastive\tests\test_fusion_alpha_contract.py"),
         Path(r"D:\project\unknown-contrastive\tests\test_fusion_alpha_firewall.py"))
PYTEST_COMMAND = "python -B -m pytest -p no:cacheprovider -q D:\\project\\unknown-contrastive\\tests\\test_fusion_alpha_contract.py D:\\project\\unknown-contrastive\\tests\\test_fusion_alpha_firewall.py"
LABELED_MANIFEST = Path(r"D:\project\unknown-contrastive\data\pools\v2\unknown\strict_novel_val.json")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def atomic_json_new(path, value):
    target = Path(path)
    data = _canonical(value)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data); fh.flush(); os.fsync(fh.fileno())
        os.link(temporary, target)
    except FileExistsError as exc:
        raise RuntimeError("output already exists") from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def source_and_test_hashes():
    scripts = (Path(__file__), Path(__file__).with_name("fusion_alpha_common.py"),
               Path(__file__).with_name("run_fusion_alpha_unlabeled.py"), Path(__file__).with_name("run_fusion_alpha_offline.py"), *TESTS)
    return {str(path): sha256(path) for path in scripts}


def verify_controller_bindings():
    if sha256(R3) != R3_FILE_SHA256 or sha256(EVIDENCE) != EVIDENCE_FILE_SHA256:
        raise RuntimeError("R3/evidence hash mismatch")
    binding = R3.read_text(encoding="utf-8")
    if not re.search(r'"binding_sha256"\s*:\s*"10ad4ed0b0539230385c456b8242feee6f2506a66a104dc926df4aa48528e093"', binding):
        raise RuntimeError("R3 decision-binding mismatch")
    for path, expected in DEPENDENCIES.items():
        if sha256(path) != expected:
            raise RuntimeError(f"dependency hash mismatch: {path}")


def run_required_tests():
    before = source_and_test_hashes()
    command = [sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q", *(str(path) for path in TESTS)]
    result = subprocess.run(command, cwd=str(PROJECT_ROOT), env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, check=False)
    if result.returncode:
        raise RuntimeError("required fusion tests failed")
    after = source_and_test_hashes()
    if after != before:
        raise RuntimeError("fusion source/test drift during required tests")
    return after


def create_root(root=None, tested_hashes=None):
    target = Path(ROOT if root is None else root)
    if target != ROOT:
        raise RuntimeError("controller root is not the exact R3 root")
    if target.exists():
        raise RuntimeError("exact output root already exists")
    hashes = source_and_test_hashes()
    if tested_hashes is not None and hashes != tested_hashes:
        raise RuntimeError("fusion source/test drift before output creation")
    target.mkdir(parents=True, exist_ok=False)
    atomic_json_new(target / "create_new_receipt.json", {
        "schema": "fusion_alpha.create_new.v2", "root": str(target), "root_absent_before_create": True,
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(), "controller_pid": os.getpid(),
        "r3_path": str(R3), "r3_file_sha256": R3_FILE_SHA256, "evidence_path": str(EVIDENCE),
        "evidence_file_sha256": EVIDENCE_FILE_SHA256, "source_and_test_hashes": hashes,
        "pytest_command": PYTEST_COMMAND, "pytest_status": "passed",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "worker_environment": {"CUDA_VISIBLE_DEVICES": ""},
    })
    return target


def verified_seal(path):
    """Verify the canonical seal envelope without JSON-decoding scientific fields."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    status = re.search(rb'"status":"(selected|no_candidate)"', raw)
    alpha = re.search(rb'"selected_alpha":(null|0\.02|0\.05|0\.1|0\.2|0\.4|0\.6)(?=[,}])', raw)
    seal = re.search(rb',"seal_sha256":"([0-9a-f]{64})"', raw)
    screen = re.search(rb'"screen_sha256":"([0-9a-f]{64})"', raw)
    labels = re.search(rb'"labels_used":false(?=[,}])', raw)
    r3 = re.search(rb'"r3_file_sha256":"([0-9a-f]{64})"', raw)
    if not status or not alpha or not seal or not screen or not labels or not r3:
        raise RuntimeError("malformed selection seal")
    bare, replacements = re.subn(rb',"seal_sha256":"[0-9a-f]{64}"', b"", raw)
    if replacements != 1 or hashlib.sha256(bare).hexdigest().encode("ascii") != seal.group(1):
        raise RuntimeError("selection seal self-hash mismatch")
    if r3.group(1).decode("ascii") != R3_FILE_SHA256:
        raise RuntimeError("selection seal R3 mismatch")
    status_value = status.group(1).decode("ascii")
    alpha_token = alpha.group(1).decode("ascii")
    if status_value == "selected" and alpha_token == "null":
        raise RuntimeError("selected seal lacks interior alpha")
    if status_value == "no_candidate" and alpha_token != "null":
        raise RuntimeError("no-candidate seal contains alpha")
    selected_alpha = None if alpha_token == "null" else float(alpha_token)
    return status_value, digest, screen.group(1).decode("ascii"), selected_alpha


def verified_worker1_receipt(path, seal_digest, status):
    raw = Path(path).read_text(encoding="utf-8")
    receipt_status = re.search(r'"status"\s*:\s*"(selected|no_candidate)"', raw)
    receipt_seal = re.search(r'"selection_seal_sha256"\s*:\s*"([0-9a-f]{64})"', raw)
    labels = re.search(r'"label_open_count"\s*:\s*0(?=[,}])', raw)
    if not receipt_status or not receipt_seal or not labels or receipt_status.group(1) != status or receipt_seal.group(1) != seal_digest:
        raise RuntimeError("unlabeled receipt/seal barrier mismatch")


def verified_worker2_receipt(path, offline_output, seal_digest, selected_alpha):
    try:
        receipt = json.loads(Path(path).read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("invalid offline worker receipt") from exc
    output_digest = sha256(offline_output)
    opened = receipt.get("opened_paths")
    labeled_reads = [item for item in opened or []
                     if isinstance(item, dict) and item.get("path") == str(LABELED_MANIFEST)]
    if (receipt.get("worker") != "offline" or receipt.get("status") != "passed"
            or receipt.get("selection_seal_sha256") != seal_digest
            or receipt.get("selected_alpha") != selected_alpha
            or receipt.get("label_open_count") != 1 or receipt.get("labels_used") is not True
            or receipt.get("labels_used_after_selection") is not True
            or receipt.get("offline_selected_alpha_sha256") != output_digest
            or len(labeled_reads) != 1):
        raise RuntimeError("offline receipt/output binding mismatch")
    return output_digest


def _worker(command, env):
    process = subprocess.Popen(command, cwd=str(PROJECT_ROOT), env=env)
    returncode = process.wait()
    if process.poll() is None:
        raise RuntimeError("worker PID did not terminate")
    return process.pid, returncode


def final_receipt(root, status, **extra):
    payload = {"schema": "fusion_alpha.result.v2", "status": status, "controller_pid": os.getpid(),
               "completed_utc": datetime.datetime.now(datetime.UTC).isoformat(), "root": str(root),
               "source_and_test_hashes": source_and_test_hashes(), **extra}
    atomic_json_new(Path(root) / "result_receipt.json", payload)


def main(argv=None):
    if parser().parse_args(argv).__dict__:
        raise RuntimeError("controller accepts no runtime overrides")
    root = None
    worker2_started = False
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        verify_controller_bindings()
        tested_hashes = run_required_tests()
        root = create_root(tested_hashes=tested_hashes)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        worker1 = [sys.executable, "-B", "-m", "scripts.run_fusion_alpha_unlabeled"]
        pid1, code1 = _worker(worker1, env)
        if code1:
            raise RuntimeError("unlabeled worker failed")
        receipt1 = root / "unlabeled_process_receipt.json"
        seal = root / "selection_seal.json"
        if not receipt1.is_file() or not seal.is_file():
            raise RuntimeError("unlabeled worker outputs missing")
        status, seal_digest, screen_digest, selected_alpha = verified_seal(seal)
        screen = root / "unlabeled_screen.json"
        if not screen.is_file() or sha256(screen) != screen_digest:
            raise RuntimeError("unlabeled screen hash barrier mismatch")
        verified_worker1_receipt(receipt1, seal_digest, status)
        if status == "no_candidate":
            final_receipt(root, "no_candidate", worker1_pid=pid1, worker1_terminated=True,
                          selection_seal_sha256=seal_digest, label_open_count=0)
            return
        worker2 = [sys.executable, "-B", "-m", "scripts.run_fusion_alpha_offline",
                   "--expected-seal-sha256", seal_digest]
        worker2_started = True
        pid2, code2 = _worker(worker2, env)
        if code2:
            raise RuntimeError("offline worker failed")
        offline_receipt = root / "offline_process_receipt.json"
        offline_output = root / "offline_selected_alpha.json"
        if not offline_receipt.is_file() or not offline_output.is_file():
            raise RuntimeError("offline worker outputs missing")
        offline_digest = verified_worker2_receipt(
            offline_receipt, offline_output, seal_digest, selected_alpha)
        if source_and_test_hashes() != tested_hashes:
            raise RuntimeError("fusion source/test drift during execution")
        final_receipt(root, "complete", worker1_pid=pid1, worker1_terminated=True, worker2_pid=pid2,
                      worker2_terminated=True, selection_seal_sha256=seal_digest,
                      selected_alpha=selected_alpha, offline_selected_alpha_sha256=offline_digest,
                      offline_process_receipt_sha256=sha256(offline_receipt), label_open_count=1)
    except Exception as exc:
        if root is not None and not (root / "result_receipt.json").exists():
            final_receipt(root, "failed", error=str(exc), label_open_count=1 if worker2_started else 0)
        raise


def parser():
    return argparse.ArgumentParser(description=__doc__)


if __name__ == "__main__":
    main()
