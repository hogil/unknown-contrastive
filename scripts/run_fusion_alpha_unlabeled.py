"""Worker 1: fixed-root, label-free fusion screen.

There is intentionally no argument for a manifest, alpha, grid, or output root.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from scripts.fusion_alpha_common import (DIAL_SPECS, ENDPOINT_ALPHAS, ENDPOINT_EXPECTED,
    FROZEN_PRE_NOISE_CAPS, HASHES, INTERIOR_ALPHAS, R3, ROOT, atomic_json_new, canonical_partition,
    label_free_row, ordered_paths_hash, receipt_base, reject, seal_hash, select, sha256, source_hashes,
    utc_now, validate_unlabeled_inputs)


def parser():
    return argparse.ArgumentParser(description=__doc__)


def opaque_alignment(left, right):
    """Paths are opaque strings: exact equality/order is the only operation."""
    return list(left) == list(right)


def _prediction_key(dial, endpoint):
    return f"{dial['mcs']}_{dial['min_samples']}_{'frozen' if endpoint == 0.0 else 'ep12'}"


def _bound_endpoint_schema(selection, dial):
    """Read only the R3-bound label-free schema, never historical metric values."""
    for item in selection.get("per_dial", []):
        source = item.get("dial", {})
        if source.get("mcs") == dial["mcs"] and source.get("ms", source.get("min_samples")) == dial["min_samples"]:
            schema = item.get("metric_schema", {})
            fields = schema.get("ep12") if isinstance(schema, dict) else None
            if isinstance(fields, dict):
                return set(fields)
            if isinstance(fields, (list, tuple, set)) and all(isinstance(field, str) for field in fields):
                return set(fields)
    return None


def _verify_endpoint(endpoint, embedding, dial, row, prediction, raw, predictions, selection):
    if (endpoint == 0.0 and embedding is not _verify_endpoint.frozen) or (endpoint == 1.0 and embedding is not _verify_endpoint.projection):
        reject("endpoint array was not routed directly")
    key = _prediction_key(dial, endpoint)
    bound = predictions.get(key)
    if bound is None:
        reject("bound endpoint prediction missing")
    if not np.array_equal(prediction == -1, bound == -1) or not np.array_equal(canonical_partition(prediction), canonical_partition(bound)):
        reject("endpoint noise mask or partition mismatch")
    expected_k, expected_noise = ENDPOINT_EXPECTED[endpoint][dial["ratio"]]
    if int(row["k"]) != expected_k or int(row["pre_reassign_noise"]) != expected_noise:
        reject("endpoint integer count mismatch")
    bound_schema = _bound_endpoint_schema(selection, dial)
    if bound_schema is None:
        reject("bound endpoint metric schema missing")
    if set(raw) != bound_schema:
        reject("endpoint bound metric mismatch")


def run_screen(root=None):
    started_utc = utc_now()
    root = Path(ROOT if root is None else root)
    if root != ROOT:
        reject("worker root is not the exact R3 root")
    if not root.is_dir():
        reject("exact controller root missing")
    opened = []
    paths, frozen, projection, predictions, selection, opened = validate_unlabeled_inputs(opened)
    _verify_endpoint.frozen, _verify_endpoint.projection = frozen, projection
    metrics = {}
    endpoint_evidence = {}
    for alpha in ENDPOINT_ALPHAS + INTERIOR_ALPHAS:
        embedding = frozen if alpha == 0.0 else projection if alpha == 1.0 else None
        if embedding is None:
            from scripts.fusion_alpha_common import fuse
            embedding = fuse(frozen, projection, alpha)
        metrics[alpha] = {}
        for dial in DIAL_SPECS:
            row, prediction, raw = label_free_row(embedding, dial)
            metrics[alpha][dial["ratio"]] = row
            if alpha in ENDPOINT_ALPHAS:
                _verify_endpoint(alpha, embedding, dial, row, prediction, raw, predictions, selection)
                endpoint_evidence.setdefault(alpha, {})[dial["ratio"]] = {
                    "k": int(row["k"]), "pre_reassign_noise": int(row["pre_reassign_noise"]),
                    "noise_mask_sha256": hashlib.sha256((prediction == -1).tobytes()).hexdigest(),
                    "canonical_partition_sha256": hashlib.sha256(canonical_partition(prediction).tobytes()).hexdigest(),
                }
    selection_result = select({alpha: metrics[alpha] for alpha in INTERIOR_ALPHAS})
    screen = {"schema": "fusion_alpha.unlabeled.v2", "path_count": len(paths),
              "interior_metrics": {str(alpha): metrics[alpha] for alpha in INTERIOR_ALPHAS},
              "endpoint_evidence": {str(alpha): endpoint_evidence[alpha] for alpha in ENDPOINT_ALPHAS},
              "integer_frozen_pre_noise_caps": FROZEN_PRE_NOISE_CAPS, **selection_result,
              "labels_used": False}
    screen_path = root / "unlabeled_screen.json"
    atomic_json_new(screen_path, screen)
    seal = {"schema": "fusion_alpha.selection_seal.v2", "screen_sha256": sha256(screen_path),
            "r3_file_sha256": sha256(R3),
            "selected_alpha": selection_result["selected_alpha"], "status": selection_result["status"],
            "labels_used": False, "q75": selection_result["q75"], "q75_populations": selection_result["q75_populations"],
            "q75_derivations": selection_result["q75_derivations"],
            "base_pass_membership": selection_result["base_pass_membership"],
            "eligible": selection_result["eligible"], "rank_tuples": selection_result["rank_tuples"],
            "integer_frozen_pre_noise_caps": FROZEN_PRE_NOISE_CAPS, "path_count": len(paths),
            "ordered_paths_sha256": ordered_paths_hash(paths),
            "array_shapes": {"frozen": list(frozen.shape), "projection": list(projection.shape)},
            "immutable_input_hashes": HASHES, "source_hashes": source_hashes(),
            "environment": {"cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "")},
            "endpoint_evidence": endpoint_evidence, "opened_paths": opened}
    seal["seal_sha256"] = seal_hash(seal)
    atomic_json_new(root / "selection_seal.json", seal)
    receipt = receipt_base(opened, label_open_count=0, labels_used=False, started_utc=started_utc)
    receipt.update({"worker": "unlabeled", "status": selection_result["status"],
                    "screen_sha256": sha256(screen_path), "selection_seal_sha256": sha256(root / "selection_seal.json"),
                    "selected_alpha": selection_result["selected_alpha"], "path_count": len(paths)})
    atomic_json_new(root / "unlabeled_process_receipt.json", receipt)
    return selection_result


def main(argv=None):
    if parser().parse_args(argv).__dict__:
        reject("worker accepts no runtime overrides")
    run_screen()


if __name__ == "__main__":
    main()
