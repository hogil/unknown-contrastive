"""Worker 2: fresh-process, sealed-alpha offline scorer with one label read."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.fusion_alpha_common import (DIAL_SPECS, FROZEN_OBJECT_FINGERPRINTS, FROZEN_OFFLINE,
    HASHES, INTERIOR_ALPHAS, LABELED, ROOT, atomic_json_new, canonical_bytes, fuse, gate_all,
    label_free_raw, preflight_offline_inputs, read_json_verified, receipt_base, reject, seal_hash, sha256, utc_now)


def parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-seal-sha256", required=True)
    return parser


def load_sealed_alpha(path, expected):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        reject("selection seal hash mismatch")
    try:
        seal = json.loads(raw)
    except json.JSONDecodeError:
        reject("invalid selection seal")
    claimed = seal.get("seal_sha256")
    bare = dict(seal)
    bare.pop("seal_sha256", None)
    alpha = seal.get("selected_alpha")
    if (claimed != seal_hash(bare) or seal.get("status") != "selected" or alpha not in INTERIOR_ALPHAS
            or seal.get("labels_used") is not False or not isinstance(seal.get("screen_sha256"), str)):
        reject("invalid selection seal contract")
    return alpha, seal


def _read_labeled_manifest_once(opened):
    """The hash and parse are intentionally performed from this one bytes object."""
    raw = LABELED.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != HASHES["labeled"]:
        reject("labeled manifest hash mismatch")
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError:
        reject("invalid labeled manifest")
    opened.append({"path": str(LABELED), "sha256": digest})
    files = manifest.get("files")
    if not isinstance(files, list) or not all(isinstance(item, dict) and isinstance(item.get("path"), str) and "label" in item for item in files):
        reject("labeled manifest schema mismatch")
    return files


def _frozen_rows(opened):
    payload = read_json_verified(FROZEN_OFFLINE, HASHES["frozen_offline"], opened)
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        reject("frozen comparator metrics missing")
    rows = {}
    for ratio in (0.005, 0.01, 0.02):
        item = metrics.get(str(ratio), metrics.get(ratio))
        row = item.get("frozen", item) if isinstance(item, dict) else None
        if not isinstance(row, dict) or hashlib.sha256(canonical_bytes(row)).hexdigest() != FROZEN_OBJECT_FINGERPRINTS[ratio]:
            reject("frozen comparator object fingerprint mismatch")
        rows[ratio] = row
    return rows


def _score(embedding, labels):
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    from scripts.cluster_metrics import capture_metrics
    from scripts.eval_open_set_embeddings import purity_score, reassign_noise_to_nearest_cluster
    rows = {}
    for dial in DIAL_SPECS:
        _, prediction = label_free_raw(embedding, dial)
        capture = capture_metrics(prediction, labels)
        post, _ = reassign_noise_to_nearest_cluster(embedding, prediction, "nearest_q90")
        coverage = []
        for label in np.unique(labels):
            clusters = [cluster for cluster, dominant in capture["dominant_by_cluster"].items() if dominant == label]
            coverage.append(float(np.mean(np.isin(prediction[labels == label], clusters))))
        rows[dial["ratio"]] = {
            "P1_unique_dominant_capture": capture["capture_rate"], "captured_classes": capture["captured_classes"],
            "lost_classes": sorted(set(np.unique(labels).tolist()) - set(capture["captured_classes"]), key=str),
            "macro_image_cap": float(np.mean(coverage)), "minimum_image_cap": float(np.min(coverage)),
            "pre_reassign_noise": float(np.mean(prediction == -1) * 100),
            "post_reassign_noise": float(np.mean(post == -1) * 100),
            "k": int(len(set(prediction.tolist())) - (1 if -1 in prediction else 0)),
            "purity_weighted": float(purity_score(labels, prediction)),
            "fragmentation": float(len({int(v) for v in prediction if int(v) >= 0}) / max(1, len(set(labels.tolist())))),
            "ARI": float(adjusted_rand_score(labels, prediction)), "AMI": float(adjusted_mutual_info_score(labels, prediction)),
        }
    return rows


def run_offline(expected_seal_sha256, root=None):
    started_utc = utc_now()
    root = Path(ROOT if root is None else root)
    if root != ROOT or not root.is_dir():
        reject("worker root is not the exact R3 root")
    opened = []
    seal_path = root / "selection_seal.json"
    alpha, seal = load_sealed_alpha(seal_path, expected_seal_sha256)
    opened.append({"path": str(seal_path), "sha256": expected_seal_sha256})
    bound_paths, frozen, projection, opened = preflight_offline_inputs(opened)
    frozen_rows = _frozen_rows(opened)
    files = _read_labeled_manifest_once(opened)
    paths = [item["path"] for item in files]
    if paths != bound_paths:
        reject("labeled/bundle path order mismatch")
    rows = _score(fuse(frozen, projection, alpha), np.asarray([item["label"] for item in files]))
    passed = gate_all(rows, frozen_rows)
    output = {"schema": "fusion_alpha.offline.v2", "selection_seal_sha256": expected_seal_sha256,
              "selected_alpha": alpha, "fixed_dials": DIAL_SPECS, "metrics": rows,
              "frozen_gate_pass": passed, "labels_used_after_selection": True,
              "frozen_comparator_sha256": HASHES["frozen_offline"],
              "frozen_object_fingerprints": FROZEN_OBJECT_FINGERPRINTS}
    atomic_json_new(root / "offline_selected_alpha.json", output)
    receipt = receipt_base(opened, label_open_count=1, labels_used=True, started_utc=started_utc)
    receipt.update({"worker": "offline", "status": "passed" if passed else "scientific_gate_failed",
                    "selected_alpha": alpha, "selection_seal_sha256": expected_seal_sha256,
                    "offline_selected_alpha_sha256": sha256(root / "offline_selected_alpha.json"),
                    "fixed_dials": DIAL_SPECS, "labels_used_after_selection": True})
    atomic_json_new(root / "offline_process_receipt.json", receipt)
    if not passed:
        reject("sealed alpha failed mandatory frozen gate")
    return output


def main(argv=None):
    args = parser().parse_args(argv)
    run_offline(args.expected_seal_sha256)


if __name__ == "__main__":
    main()
