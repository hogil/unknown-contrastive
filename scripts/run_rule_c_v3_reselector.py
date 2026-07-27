"""CPU-only, label-free Rule-C V3 reselector over a persisted V2 snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np


RULE_NAME = "primary_all_support_minimax_pre_reassign_noise_v3"
V2_SCHEMA = "rule_c_label_free_selection.v2"
V3_SCHEMA = "rule_c_label_free_selection.v3"
MANIFEST_SHA256 = "9f3870e2a5c5a0af5d56bc013463ce68a1308dd984e1fc0a4b4b67b60838e397"
MANIFEST_COUNT = 4178
PRIMARY_DIALS = (
    {"ratio": .005, "mcs": 21, "ms": 5, "eps": .06, "method": "leaf"},
    {"ratio": .01, "mcs": 42, "ms": 10, "eps": .06, "method": "leaf"},
    {"ratio": .02, "mcs": 84, "ms": 21, "eps": .06, "method": "leaf"},
)
METRIC_FIELDS = ("k", "pre_reassign_noise", "stability", "coherence", "over_merge")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def ordered_paths_sha256(paths: list[str]) -> str:
    return hashlib.sha256(json.dumps(paths, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def epoch_metrics(metrics: dict) -> tuple[dict[int, dict], list[str]]:
    errors: list[str] = []
    values: dict[int, dict] = {}
    for epoch in range(1, 21):
        key = f"ep{epoch:02d}"
        row = metrics.get(key)
        if not isinstance(row, dict):
            errors.append(f"missing_checkpoint_metric:{key}")
        else:
            values[epoch] = row
    extra = sorted(set(metrics) - {f"ep{i:02d}" for i in range(1, 21)} - {"frozen", *[f"z0_s{i}" for i in range(1, 11)]})
    if extra:
        errors.append("non_checkpoint_metric_records_present:" + ",".join(extra))
    return values, errors


def finite_metric(row: dict, key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def linear_p75(values: list[float]) -> float:
    ordered = sorted(values)
    h = (len(ordered) - 1) * .75
    lo, hi = math.floor(h), math.ceil(h)
    return ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo])


def evaluate_dial(entry: dict) -> dict:
    metrics = entry.get("metrics")
    if not isinstance(metrics, dict):
        return {"errors": ["metrics_not_object"], "gate_pass_epochs": [], "k_q75": None, "retained_epochs": []}
    rows, errors = epoch_metrics(metrics)
    gate: list[tuple[int, float]] = []
    metric_schema: dict[str, list[str]] = {}
    for epoch, row in rows.items():
        missing = [field for field in METRIC_FIELDS if field not in row]
        metric_schema[f"ep{epoch:02d}"] = sorted(row.keys())
        if missing:
            errors.append(f"missing_metric_fields:ep{epoch:02d}:" + ",".join(missing))
            continue
        numbers = {field: finite_metric(row, field) for field in METRIC_FIELDS}
        if any(value is None for value in numbers.values()):
            errors.append(f"nonfinite_metric:ep{epoch:02d}")
            continue
        if numbers["over_merge"] == 0 and numbers["stability"] >= .75 and numbers["coherence"] >= .80:
            gate.append((epoch, numbers["k"]))
    if not gate:
        return {"errors": errors + ["empty_gate_population"], "gate_pass_epochs": [], "k_q75": None, "retained_epochs": [], "metric_schema": metric_schema}
    q75 = linear_p75([k for _, k in gate])
    retained = [epoch for epoch, k in gate if k >= q75]
    return {"errors": errors, "gate_pass_epochs": [epoch for epoch, _ in gate], "k_q75": q75, "retained_epochs": retained, "metric_schema": metric_schema}


def reselect(snapshot: dict) -> dict:
    """Return a complete V3 decision; all malformed or incomparable inputs fail closed."""
    failures: list[str] = []
    if snapshot.get("schema_version") != V2_SCHEMA:
        failures.append("source_schema_not_v2")
    if snapshot.get("pool_sha256") != MANIFEST_SHA256:
        failures.append("manifest_sha256_mismatch")
    primary = snapshot.get("primary_dials")
    per_dial = snapshot.get("per_dial")
    if primary != list(PRIMARY_DIALS) or not isinstance(per_dial, list) or len(per_dial) != 3:
        failures.append("primary_dial_contract_mismatch")
        primary_entries: list[dict] = []
    else:
        primary_entries = per_dial
        for expected, entry in zip(PRIMARY_DIALS, primary_entries):
            if not isinstance(entry, dict) or entry.get("dial") != expected:
                failures.append("per_dial_contract_mismatch")
                break
    dial_results = [evaluate_dial(entry) for entry in primary_entries]
    for index, result in enumerate(dial_results):
        failures.extend(f"dial{index + 1}:{error}" for error in result["errors"])
    schemas = [result.get("metric_schema") for result in dial_results]
    if len(schemas) == 3 and any(schema != schemas[0] for schema in schemas[1:]):
        failures.append("persisted_metric_schema_mismatch")
    sensitivity = snapshot.get("sensitivity_audit")
    sensitivity_ok = isinstance(sensitivity, dict) and sensitivity.get("consensus_eligible") is False
    if not sensitivity_ok:
        failures.append("sensitivity_audit_contract_mismatch")

    retained = [set(result["retained_epochs"]) for result in dial_results]
    candidates = sorted(set.intersection(*retained)) if len(retained) == 3 and all(retained) else []
    scores = []
    for epoch in candidates:
        noises = []
        for entry in primary_entries:
            noise = finite_metric(entry["metrics"][f"ep{epoch:02d}"], "pre_reassign_noise")
            if noise is None:
                failures.append(f"candidate_nonfinite_noise:ep{epoch:02d}")
                break
            noises.append(noise)
        if len(noises) == 3:
            scores.append({"epoch": epoch, "pre_reassign_noise_by_ratio": {str(dial["ratio"]): noise for dial, noise in zip(PRIMARY_DIALS, noises)}, "worst_pre_reassign_noise_pct": max(noises), "mean_pre_reassign_noise_pct": sum(noises) / 3, "selection_key": [max(noises), sum(noises) / 3, epoch]})
    if not candidates:
        failures.append("empty_three_dial_intersection")
    selected = min(scores, key=lambda item: tuple(item["selection_key"])) if scores and not failures else None
    for dial, result in zip(PRIMARY_DIALS, dial_results):
        result["dial"] = dial
    return {"failures": sorted(set(failures)), "per_dial": dial_results, "candidate_scores": scores, "selected": selected,
            "comparability": {"manifest_sha256": snapshot.get("pool_sha256") == MANIFEST_SHA256, "expected_manifest_count": MANIFEST_COUNT,
                              "single_source_snapshot": True, "metric_schema_equal": len(schemas) == 3 and all(schema == schemas[0] for schema in schemas[1:]),
                              "pre_reassign_noise_fields_finite": not any("nonfinite_metric" in failure for failure in failures)}}


def immutable_write(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable output already exists: {path}")
    encoded = json.dumps(payload, sort_keys=True, indent=2).encode("utf-8")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
        os.link(temp_name, path)
    except FileExistsError:
        raise FileExistsError(f"immutable output already exists: {path}") from None
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return sha256_file(path)


def verify_cached_inputs(snapshot: dict, checkpoint_dir: Path) -> tuple[dict, list[str]]:
    """Verify the bound unlabeled cache without opening any labeled manifest."""
    failures: list[str] = []
    bundle_path = Path(snapshot.get("bundle_path", ""))
    if not bundle_path.is_file() or sha256_file(bundle_path) != snapshot.get("bundle_sha256"):
        return {}, ["source_bundle_missing_or_hash_mismatch"]
    try:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, ["source_bundle_invalid"]
    pool_path = Path(snapshot.get("pool", ""))
    if not pool_path.is_file() or sha256_file(pool_path) != MANIFEST_SHA256:
        failures.append("pool_missing_or_hash_mismatch")
        pool = {}
    else:
        try:
            pool = json.loads(pool_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            failures.append("pool_invalid")
            pool = {}
    paths = pool.get("files")
    if not isinstance(pool.get("root"), str) or not isinstance(paths, list) or not all(isinstance(item, str) for item in paths):
        failures.append("pool_not_path_only")
    elif len(paths) != MANIFEST_COUNT:
        failures.append("pool_count_mismatch")
    if bundle.get("root") != pool.get("root") or bundle.get("paths") != paths:
        failures.append("bundle_path_order_mismatch")
    npz_path = Path(bundle.get("npz_path", ""))
    if not npz_path.is_file() or sha256_file(npz_path) != bundle.get("npz_sha256"):
        failures.append("npz_missing_or_hash_mismatch")
    else:
        try:
            with np.load(npz_path, allow_pickle=False) as npz:
                if npz["paths"].tolist() != paths:
                    failures.append("npz_path_order_mismatch")
        except (OSError, KeyError, ValueError):
            failures.append("npz_invalid_or_paths_missing")
    return {"bundle_path": str(bundle_path.resolve()), "bundle_sha256": snapshot.get("bundle_sha256"),
            "npz_path": str(npz_path.resolve()), "npz_sha256": bundle.get("npz_sha256"),
            "pool_path": str(pool_path.resolve()), "pool_sha256": MANIFEST_SHA256,
            "pool_count": len(paths) if isinstance(paths, list) else None, "ordered_paths_sha256": ordered_paths_sha256(paths) if isinstance(paths, list) else None,
            "checkpoint_hashes": bundle.get("checkpoint_sha256")}, failures


def verify_selected_checkpoint(cache: dict, checkpoint_dir: Path, epoch: int | None) -> tuple[dict, list[str]]:
    if epoch is None:
        return {}, ["no_selected_checkpoint"]
    expected = cache.get("checkpoint_hashes")
    name = f"proj_ep{epoch}.pt"
    path = checkpoint_dir.resolve() / name
    if not isinstance(expected, dict) or not path.is_file() or expected.get(name) != sha256_file(path):
        return {}, ["selected_checkpoint_missing_or_hash_mismatch"]
    return {"selected_checkpoint_path": str(path), "selected_checkpoint_sha256": expected[name]}, []


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-v2-snapshot", type=Path, required=True)
    parser.add_argument("--r3-binding", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seal-out", type=Path, required=True)
    args = parser.parse_args(argv)
    source = args.source_v2_snapshot.resolve()
    binding = args.r3_binding.resolve()
    snapshot = json.loads(source.read_text(encoding="utf-8"))
    binding_payload = json.loads(binding.read_text(encoding="utf-8"))
    result = reselect(snapshot)
    repair = binding_payload.get("response", {}).get("rule_c_repair", {})
    specification = repair.get("specification", {}) if isinstance(repair, dict) else {}
    binding_errors = []
    if (repair.get("decision") != "implement_exact_v3" or specification.get("selection_name") != RULE_NAME
            or repair.get("selected_epoch_on_frozen_snapshot") != 14 or repair.get("labels_used") is not False
            or repair.get("code_only_implementation_authorized") is not True
            or not binding_payload.get("evidence_packet_sha") or not binding_payload.get("binding_sha256")):
        binding_errors.append("r3_binding_contract_mismatch")
    selected = result["selected"]
    cache, cache_failures = verify_cached_inputs(snapshot, args.checkpoint_dir)
    checkpoint, checkpoint_failures = verify_selected_checkpoint(cache, args.checkpoint_dir, selected["epoch"] if selected else None)
    result["failures"] = sorted(set(result["failures"] + binding_errors + cache_failures + checkpoint_failures))
    selected = selected if not result["failures"] else None
    valid = selected is not None
    payload = {"schema_version": V3_SCHEMA, "selection_name": RULE_NAME, "selection_valid": valid,
               "status": "selected" if valid else "diagnostic_no_consensus", "selected_epoch": selected["epoch"] if valid else None,
               "offline_labels_evaluated": False, "labels_used": False,
               "source_v2_snapshot_path": str(source), "source_v2_snapshot_sha256": sha256_file(source),
               "pool": snapshot.get("pool"), "pool_sha256": snapshot.get("pool_sha256"),
               "r3_binding_path": str(binding), "r3_binding_sha256": sha256_file(binding),
               "r3_evidence_packet_sha256": binding_payload.get("evidence_packet_sha"), "r3_decision_binding_sha256": binding_payload.get("binding_sha256"),
               "selector_source_path": str(Path(__file__).resolve()), "selector_source_sha256": source_sha256(), "selection_manifest_sha256": MANIFEST_SHA256, "selection_manifest_count": MANIFEST_COUNT,
               "source_bundle": cache, "selected_checkpoint": checkpoint,
               "metric_comparability": result["comparability"], "primary_dials": list(PRIMARY_DIALS),
               "p75_method": "linear", "gate": {"required_finite_metrics": list(METRIC_FIELDS), "over_merge": 0, "stability_gte": .75, "coherence_gte": .80},
               "per_dial": result["per_dial"], "cross_dial_support_required": 3, "candidate_scores": result["candidate_scores"],
               "selection_key": ["worst_pre_reassign_noise_pct", "mean_pre_reassign_noise_pct", "epoch"], "failures": result["failures"],
               "sensitivity_excluded": True}
    digest = immutable_write(args.out.resolve(), payload)
    if not valid:
        print(json.dumps({"selection_snapshot_path": str(args.out.resolve()), "selection_snapshot_sha256": digest,
                          "selection_valid": False, "selected_epoch": None}))
        raise SystemExit("v3 selection invalid; epoch seal not created")
    seal = {"schema_version": "rule_c_epoch_seal.v3", "selection_name": RULE_NAME,
            "selection_snapshot_path": str(args.out.resolve()), "selection_snapshot_sha256": digest,
            "selected_epoch": payload["selected_epoch"], "bundle_sha256": cache.get("bundle_sha256"),
            "npz_sha256": cache.get("npz_sha256"), "selected_checkpoint_sha256": checkpoint.get("selected_checkpoint_sha256"),
            "pool_path": cache.get("pool_path"), "pool_sha256": cache.get("pool_sha256"), "pool_count": cache.get("pool_count"), "ordered_paths_sha256": cache.get("ordered_paths_sha256"),
            "source_v2_snapshot_sha256": payload["source_v2_snapshot_sha256"], "r3_binding_sha256": payload["r3_binding_sha256"],
            "selector_source_sha256": payload["selector_source_sha256"], "labels_used": False}
    seal_digest = immutable_write(args.seal_out.resolve(), seal)
    print(json.dumps({"selection_snapshot_path": str(args.out.resolve()), "selection_snapshot_sha256": digest,
                      "epoch_seal_path": str(args.seal_out.resolve()), "epoch_seal_sha256": seal_digest,
                      "selection_valid": valid, "selected_epoch": payload["selected_epoch"]}))


if __name__ == "__main__":
    main()
