"""Immutable, label-free primitives for the R3 fusion-alpha action.

This module is deliberately imported by data workers only.  The controller must
remain a standard-library orchestrator and never import it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\project\unknown-contrastive\runs\campaign_state\fusion\lr008_ep12_frozen_projection_alpha_screen_260727_v1")
R3 = Path(r"D:\project\unknown-contrastive\runs\campaign_state\panels\lr008_result_260727_v3\r3_C.json")
EVIDENCE = R3.with_name("evidence_packet.json")
BUNDLE = Path(r"D:\project\unknown-contrastive\runs\campaign_state\rule_c\strict_novel_lr008_seed42_rulec_recovery_260727_0640\rule_c\prediction_embedding_bundle.json")
NPZ = BUNDLE.with_suffix(".npz")
SELECTION_V3 = BUNDLE.with_name("selection_snapshot_v3.json")
EPOCH_SEAL = BUNDLE.with_name("epoch_seal_v3.json")
UNLABELED = Path(r"D:\project\unknown-contrastive\data\pools\v2\unknown\strict_novel_val_unlabeled.json")
LABELED = Path(r"D:\project\unknown-contrastive\data\pools\v2\unknown\strict_novel_val.json")
FROZEN_OFFLINE = BUNDLE.with_name("offline.json")

INTERIOR_ALPHAS = (0.02, 0.05, 0.1, 0.2, 0.4, 0.6)
ENDPOINT_ALPHAS = (0.0, 1.0)
DIALS = (0.005, 0.01, 0.02)
DIAL_SPECS = ({"ratio": .005, "mcs": 21, "min_samples": 5, "method": "leaf", "epsilon": .06},
              {"ratio": .01, "mcs": 42, "min_samples": 10, "method": "leaf", "epsilon": .06},
              {"ratio": .02, "mcs": 84, "min_samples": 21, "method": "leaf", "epsilon": .06})
FROZEN_PRE_NOISE_CAPS = {.005: 835, .01: 7, .02: 18}
ENDPOINT_EXPECTED = {
    0.0: {.005: (12, 835), .01: (10, 7), .02: (10, 18)},
    1.0: {.005: (17, 1517), .01: (12, 628), .02: (11, 708)},
}
HASHES = {
    "r3_file": "27df3ed5745aba21d62f4b83a8af06665301c88a96c80cebae664a399850ccde",
    "r3_binding": "10ad4ed0b0539230385c456b8242feee6f2506a66a104dc926df4aa48528e093",
    "evidence_file": "6faab6a206ac343f538e3b8c1d2a6a827fbbf15c55f4467573d75fd24ba96fd8",
    "evidence_canonical": "e670220b7a94334fb22b7938f007bc511180dd699affb11bed8a11e30e4bb0ab",
    "bundle": "c744eceaaaeef845ebae79cd09086e9274f8f009c1b9ea4f26b45b9b19be7292",
    "npz": "2eda78ab0c513dcaffbb867cec73948cdd0a317cdff6f8f0036ff8d1985f971f",
    "selection_v3": "31b724522331d638db2b6f360095575faec34525d91f79df29363a6442cadcc7",
    "epoch_seal": "9e86cd7bf4ea2d794f8c08b24bbf7c85550cd9cdc899f60ef3ef184644a6850c",
    "unlabeled": "9f3870e2a5c5a0af5d56bc013463ce68a1308dd984e1fc0a4b4b67b60838e397",
    "labeled": "aa2da7f8ff4ef63d5fe7312c80828f443efec52cf7bf42a8ef6b6008bf8446f6",
    "frozen_offline": "29b076dd773686da73476b98df58b787f98418910faedb84149d1adc1f352b6e",
}
DEPENDENCIES = {
    Path(r"D:\project\unknown-contrastive\_grouping_eval.py"): "37435a5d59c5d2b1ef5fff8115c2d4169f4af328439a4d4de71948826199fa0c",
    Path(r"D:\project\unknown-contrastive\scripts\_common.py"): "eb0aa9433c45d024ae49f4162185d691bd44e564577c2cd6be1459af5c9490fb",
    Path(r"D:\project\unknown-contrastive\scripts\cluster_metrics.py"): "c4a5218d1df11294820818ebdca1c63c44c57c117e2223aab353f347451708c5",
    Path(r"D:\project\unknown-contrastive\scripts\eval_open_set_embeddings.py"): "e5e04686b46629ec609f01bac7bdf945cf916d4784e751d4d735f94c00923a08",
    Path(r"D:\project\unknown-contrastive\scripts\run_rule_c_selector.py"): "9b65cf4ff35b027621a273a2bd5b1ef48bc63d13e20c28be85aca3b8a08e9cc6",
    Path(r"D:\project\unknown-contrastive\scripts\run_rule_c_v3_reselector.py"): "c49ee6458148345dfc46ded29ea16e552fda53495f2c08c2c6ce930278b6dcb3",
    Path(r"D:\project\unknown-contrastive\scripts\run_rule_c_offline.py"): "3198a1da6e85a029dbdfa03211f04b3645ae241ee6dc4e879ffcdf737aa2769d",
}
FROZEN_OBJECT_FINGERPRINTS = {
    .005: "55e357928c36ca7aa8460e878b707643dd5812345f315309e3a6f1d630c49569",
    .01: "ec3a347ff5d12ae9cf36cdf42c2be407790135543c30b0e463ac1b9c697be67c",
    .02: "1323fbe50cd557c0f8741b785b429ae9f3845705010995006c1f4badb0822ec7",
}
RECOVERY_R3_FILE_SHA256 = "b6438fb3cbc54536f2bd0abaaa60fd23cbeabef7ad6aa32f5e68b78f10459e2d"
RECOVERY_R3_EVIDENCE_SHA256 = "f2c1417e263999997643c52412a3dd1db4cd35b1e1c78471d14f206736f607db"
RECOVERY_R3_DECISION_BINDING_SHA256 = "2f625a30be5ca68b925dc2f806ce473dc050da6cf3e8f372add772b814691d70"
SOURCE_V2_SNAPSHOT_SHA256 = "a6b26aa18f6465adebaf5c664a183e9a6865ea7455d748da5d627740c15093ec"
SELECTOR_SOURCE_SHA256 = "c49ee6458148345dfc46ded29ea16e552fda53495f2c08c2c6ce930278b6dcb3"
ORDERED_PATHS_SHA256 = "b821df9ac1c101f7dae68576a2ec4173b65e4de800c0be89a52f7fac0298976b"

def reject(message: str) -> None:
    raise RuntimeError(message)

def sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def canonical_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def panel_sha256(value) -> str:
    """Match the campaign panel's immutable sha256_obj encoding exactly."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()

def atomic_json_new(path: Path | str, value) -> None:
    """Atomically publish a never-before-existing JSON file without replacement."""
    target = Path(path)
    data = canonical_bytes(value)
    if not target.parent.is_dir():
        reject("output parent does not exist")
    fd = None
    temporary = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
        with os.fdopen(fd, "wb") as fh:
            fd = None
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        # link is create-new and atomic; it never replaces an existing result.
        os.link(temporary, target)
    except FileExistsError:
        reject("output already exists")
    finally:
        if fd is not None:
            os.close(fd)
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)

def read_json_verified(path: Path, expected: str, opened: list[dict] | None = None):
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected:
        reject(f"immutable hash mismatch: {path}")
    if opened is not None:
        opened.append({"path": str(Path(path)), "sha256": digest})
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        reject(f"invalid JSON: {path}: {exc}")

def ordered_paths_hash(paths) -> str:
    return hashlib.sha256(json.dumps(paths, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()

def source_hashes() -> dict[str, str]:
    here = Path(__file__).resolve().parent
    files = (here / "fusion_alpha_common.py", here / "run_fusion_alpha_unlabeled.py",
             here / "run_fusion_alpha_offline.py", here / "run_fusion_alpha_screen.py")
    return {str(p): sha256(p) for p in files}

def verify_runtime_bindings(opened: list[dict]) -> None:
    r3 = read_json_verified(R3, HASHES["r3_file"], opened)
    evidence = read_json_verified(EVIDENCE, HASHES["evidence_file"], opened)
    binding = evidence.get("binding")
    if (r3.get("binding_sha256") != HASHES["r3_binding"]
            or r3.get("evidence_packet_sha") != HASHES["evidence_canonical"]
            or not isinstance(binding, dict)
            or panel_sha256(binding) != HASHES["r3_binding"]
            or panel_sha256(evidence) != HASHES["evidence_canonical"]):
        reject("R3/evidence binding mismatch")
    for path, expected in DEPENDENCIES.items():
        actual = sha256(path)
        if actual != expected:
            reject(f"dependency hash mismatch: {path}")
        opened.append({"path": str(path), "sha256": actual})

def _norm_rows(value):
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        reject("invalid embedding array")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms == 0):
        reject("zero row norm")
    return array / norms[:, None]

def fuse(frozen, projection, alpha: float):
    if alpha == 0.0:
        return frozen
    if alpha == 1.0:
        return projection
    if alpha not in INTERIOR_ALPHAS:
        reject("alpha outside fixed interior grid")
    left, right = _norm_rows(frozen), _norm_rows(projection)
    if len(left) != len(right):
        reject("row count mismatch")
    return _norm_rows(np.concatenate((math.sqrt(1 - alpha) * left, math.sqrt(alpha) * right), axis=1))

def canonical_partition(prediction):
    pred = np.asarray(prediction)
    result = np.full(pred.shape[0], -1, dtype=np.int64)
    clusters = [(int(np.flatnonzero(pred == cluster).min()), cluster)
                for cluster in np.unique(pred) if cluster != -1]
    for canonical, (_, cluster) in enumerate(sorted(clusters)):
        result[pred == cluster] = canonical
    return result

def label_free_raw(embedding, dial):
    from scripts.run_rule_c_selector import label_free_metrics
    call = {"mcs": dial["mcs"], "ms": dial["min_samples"], "eps": dial["epsilon"], "method": dial["method"]}
    row, prediction = label_free_metrics(embedding, call)
    if not isinstance(row, dict) or not isinstance(prediction, np.ndarray):
        reject("invalid label-free result")
    return dict(row), np.asarray(prediction)

def label_free_row(embedding, dial):
    raw, prediction = label_free_raw(embedding, dial)
    row = dict(raw)
    row["pre_reassign_noise_pct"] = row.get("pre_reassign_noise")
    row["pre_reassign_noise"] = int(np.count_nonzero(prediction == -1))
    return row, canonical_partition(prediction), raw

def _lineage(selection, seal, bundle):
    source = selection.get("source_bundle", {})
    checkpoint = selection.get("selected_checkpoint", {})
    if (selection.get("schema_version") != "rule_c_label_free_selection.v3"
            or selection.get("status") != "selected" or selection.get("selection_valid") is not True
            or selection.get("selected_epoch") != 12 or selection.get("labels_used") is not False
            or selection.get("offline_labels_evaluated") is not False
            or source.get("bundle_sha256") != HASHES["bundle"] or source.get("npz_sha256") != HASHES["npz"]
            or source.get("pool_path") != str(UNLABELED) or source.get("pool_sha256") != HASHES["unlabeled"]
            or source.get("pool_count") != 4178 or source.get("ordered_paths_sha256") != ORDERED_PATHS_SHA256
            or selection.get("r3_binding_sha256") != RECOVERY_R3_FILE_SHA256
            or selection.get("r3_evidence_packet_sha256") != RECOVERY_R3_EVIDENCE_SHA256
            or selection.get("r3_decision_binding_sha256") != RECOVERY_R3_DECISION_BINDING_SHA256
            or selection.get("source_v2_snapshot_sha256") != SOURCE_V2_SNAPSHOT_SHA256
            or selection.get("selector_source_sha256") != SELECTOR_SOURCE_SHA256
            or selection.get("pool") != str(UNLABELED)
            or selection.get("pool_sha256") != HASHES["unlabeled"]
            or selection.get("selection_manifest_count") != 4178
            or selection.get("selection_manifest_sha256") != HASHES["unlabeled"]
            or checkpoint.get("selected_checkpoint_sha256") != "6db73a0ae9aefdffe8213598a79959c4908bf5616a4785ee22ab59c82d679346"
            or bundle.get("npz_path") != str(NPZ) or bundle.get("npz_sha256") != HASHES["npz"]
            or bundle.get("pool_sha256") != HASHES["unlabeled"]):
        reject("selection lineage mismatch")
    if (seal.get("schema_version") != "rule_c_epoch_seal.v3" or seal.get("selected_epoch") != 12
            or seal.get("selection_snapshot_path") != str(SELECTION_V3)
            or seal.get("selection_snapshot_sha256") != HASHES["selection_v3"]
            or seal.get("bundle_sha256") != HASHES["bundle"] or seal.get("npz_sha256") != HASHES["npz"]
            or seal.get("pool_path") != str(UNLABELED) or seal.get("pool_sha256") != HASHES["unlabeled"]
            or seal.get("pool_count") != 4178 or seal.get("ordered_paths_sha256") != ORDERED_PATHS_SHA256
            or seal.get("selected_checkpoint_sha256") != checkpoint.get("selected_checkpoint_sha256")
            or seal.get("source_v2_snapshot_sha256") != SOURCE_V2_SNAPSHOT_SHA256
            or seal.get("selector_source_sha256") != SELECTOR_SOURCE_SHA256
            or seal.get("r3_binding_sha256") != RECOVERY_R3_FILE_SHA256
            or seal.get("labels_used") is not False):
        reject("epoch seal immutable lineage mismatch")

def validate_unlabeled_inputs(opened: list[dict] | None = None):
    audit = opened if opened is not None else []
    verify_runtime_bindings(audit)
    bundle = read_json_verified(BUNDLE, HASHES["bundle"], audit)
    selection = read_json_verified(SELECTION_V3, HASHES["selection_v3"], audit)
    epoch_seal = read_json_verified(EPOCH_SEAL, HASHES["epoch_seal"], audit)
    manifest = read_json_verified(UNLABELED, HASHES["unlabeled"], audit)
    _lineage(selection, epoch_seal, bundle)
    paths = manifest.get("files")
    if (not isinstance(paths, list) or not all(isinstance(item, str) for item in paths)
            or len(paths) != 4178 or ordered_paths_hash(paths) != "b821df9ac1c101f7dae68576a2ec4173b65e4de800c0be89a52f7fac0298976b"
            or bundle.get("paths") != paths):
        reject("unlabeled manifest path order mismatch")
    npz_digest = sha256(NPZ)
    if npz_digest != HASHES["npz"]:
        reject("NPZ hash mismatch")
    audit.append({"path": str(NPZ), "sha256": npz_digest})
    with np.load(NPZ, allow_pickle=False) as archive:
        expected = {"paths", "frozen", "ep12"}
        endpoint_keys = {f"{dial['mcs']}_{dial['min_samples']}_{name}" for dial in DIAL_SPECS for name in ("frozen", "ep12")}
        if not ((expected | endpoint_keys) <= set(archive.files)) or archive["paths"].tolist() != paths:
            reject("NPZ keys or path order mismatch")
        for key in archive.files:
            value = archive[key]
            if value.dtype.hasobject or value.shape[0] != 4178 or (np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all()):
                reject("invalid source NPZ key")
        frozen, projection = archive["frozen"], archive["ep12"]
        if (frozen.shape != (4178, 1024) or projection.shape != (4178, 128)
                or frozen.dtype != np.float32 or projection.dtype != np.float32):
            reject("NPZ shape or dtype mismatch")
        _norm_rows(frozen); _norm_rows(projection)
        predictions = {key: archive[key].copy() for key in archive.files if key not in expected and key.endswith(("_frozen", "_ep12"))}
        return paths, frozen.copy(), projection.copy(), predictions, selection, audit

def preflight_offline_inputs(opened: list[dict] | None = None):
    """Worker-2 non-label preflight, performed before the single label read."""
    audit = opened if opened is not None else []
    verify_runtime_bindings(audit)
    bundle = read_json_verified(BUNDLE, HASHES["bundle"], audit)
    selection = read_json_verified(SELECTION_V3, HASHES["selection_v3"], audit)
    epoch_seal = read_json_verified(EPOCH_SEAL, HASHES["epoch_seal"], audit)
    _lineage(selection, epoch_seal, bundle)
    paths = bundle.get("paths")
    if not isinstance(paths, list) or not all(isinstance(item, str) for item in paths) or len(paths) != 4178:
        reject("bundle path order mismatch")
    npz_digest = sha256(NPZ)
    if npz_digest != HASHES["npz"]:
        reject("NPZ hash mismatch")
    audit.append({"path": str(NPZ), "sha256": npz_digest})
    with np.load(NPZ, allow_pickle=False) as archive:
        if not {"paths", "frozen", "ep12"} <= set(archive.files) or archive["paths"].tolist() != paths:
            reject("NPZ keys or labeled path order mismatch")
        for key in archive.files:
            value = archive[key]
            if value.dtype.hasobject or value.shape[0] != 4178 or (np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all()):
                reject("invalid source NPZ key")
        frozen, projection = archive["frozen"], archive["ep12"]
        if (frozen.shape != (4178, 1024) or projection.shape != (4178, 128)
                or frozen.dtype != np.float32 or projection.dtype != np.float32):
            reject("NPZ shape or dtype mismatch")
        _norm_rows(frozen); _norm_rows(projection)
        return paths, frozen.copy(), projection.copy(), audit

def validate_offline_inputs(paths, opened: list[dict] | None = None):
    """Compatibility wrapper: verify preflight then compare a supplied label path list."""
    bound_paths, frozen, projection, audit = preflight_offline_inputs(opened)
    if list(paths) != bound_paths:
        reject("labeled/bundle path order mismatch")
    return frozen, projection, audit

def q75(values):
    return q75_derivation(values)["q75"]

def q75_derivation(values):
    values = sorted(values)
    if not values:
        reject("undefined q75")
    h = (len(values) - 1) * .75
    j = math.floor(h)
    g = h - j
    return {"sorted_population": values, "n": len(values), "h": h, "j": j, "g": g,
            "q75": values[j] + g * (values[min(j + 1, len(values) - 1)] - values[j])}

def base_pass(row):
    try:
        fields = ("k", "pre_reassign_noise", "stability", "coherence", "over_merge")
        return (all(np.isfinite(row[field]) for field in fields) and row["stability"] >= .75
                and row["coherence"] >= .8 and row["over_merge"] == 0)
    except (KeyError, TypeError):
        return False

def select(metrics):
    membership = {dial: [alpha for alpha in INTERIOR_ALPHAS if base_pass(metrics[alpha][dial])] for dial in DIALS}
    populations = {dial: sorted(int(metrics[alpha][dial]["k"]) for alpha in membership[dial]) for dial in DIALS}
    if any(not value for value in populations.values()):
        derivations = {dial: {"sorted_population": populations[dial], "n": len(populations[dial]),
                              "h": None, "j": None, "g": None, "q75": None} for dial in DIALS}
        return {"q75_populations": populations, "q75_derivations": derivations, "base_pass_membership": membership,
                "q75": {}, "eligible": [], "rank_tuples": [],
                "selected_alpha": None, "status": "no_candidate", "labels_used": False}
    derivations = {dial: q75_derivation(populations[dial]) for dial in DIALS}
    thresholds = {dial: derivations[dial]["q75"] for dial in DIALS}
    ranks = []
    for alpha in INTERIOR_ALPHAS:
        rows = metrics[alpha]
        if all(base_pass(rows[dial]) and int(rows[dial]["k"]) >= thresholds[dial]
               and int(rows[dial]["pre_reassign_noise"]) <= FROZEN_PRE_NOISE_CAPS[dial] for dial in DIALS):
            counts = tuple(int(rows[dial]["pre_reassign_noise"]) for dial in DIALS)
            ranks.append((max(counts), sum(counts) / 3, alpha))
    ranks.sort()
    return {"q75_populations": populations, "q75_derivations": derivations, "base_pass_membership": membership,
            "q75": thresholds, "eligible": [row[2] for row in ranks],
            "rank_tuples": ranks, "selected_alpha": ranks[0][2] if ranks else None,
            "status": "selected" if ranks else "no_candidate", "labels_used": False}

def gate_ratio(selected, frozen):
    higher = ("P1_unique_dominant_capture", "macro_image_cap", "minimum_image_cap", "purity_weighted", "ARI", "AMI")
    lower = ("pre_reassign_noise", "post_reassign_noise", "fragmentation")
    try:
        checks = [selected[key] >= frozen[key] for key in higher] + [selected[key] <= frozen[key] for key in lower]
        checks += [set(selected["lost_classes"]) <= set(frozen["lost_classes"]),
                   set(selected["captured_classes"]) >= set(frozen["captured_classes"])]
        strict = [selected[key] > frozen[key] for key in higher] + [selected[key] < frozen[key] for key in lower]
        strict += [set(selected["lost_classes"]) < set(frozen["lost_classes"]),
                   set(selected["captured_classes"]) > set(frozen["captured_classes"])]
        return all(checks) and any(strict)
    except (KeyError, TypeError):
        return False

def gate_all(selected_by_ratio, frozen_by_ratio):
    return all(gate_ratio(selected_by_ratio[dial], frozen_by_ratio[dial]) for dial in DIALS)

def seal_hash(seal) -> str:
    return hashlib.sha256(canonical_bytes(seal)).hexdigest()

def utc_now() -> str:
    return datetime.now(UTC).isoformat()

def receipt_base(opened, *, label_open_count: int, labels_used: bool, started_utc: str) -> dict:
    return {"pid": os.getpid(), "parent_pid": os.getppid(), "started_utc": started_utc, "ended_utc": utc_now(),
            "environment": {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                            "python_executable": os.environ.get("PYTHONEXECUTABLE", ""), "cwd": os.getcwd()}, "source_hashes": source_hashes(),
            "opened_paths": opened, "label_open_count": label_open_count, "labels_used": labels_used,
            "forbidden_mutations": 0}
