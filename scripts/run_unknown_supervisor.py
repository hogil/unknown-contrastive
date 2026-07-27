#!/usr/bin/env python3
"""Durable outer-loop supervisor for the finite unknown campaign executor.

The campaign runner remains deliberately finite.  This wrapper persists its
state and waits for evidence/panel decisions instead of interpreting an empty
queue as completion.  It never reads sealed-test manifests.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.run_unknown_campaign import (
    DEFAULT_CONFIG, DEFAULT_REGISTRY, REPO_ROOT, STOP_FLAG, ExclusiveFileLock,
    Registry, Campaign, check_disk_guard, sha256_file, sha256_tree,
    atomic_write_json, sha256_obj, validate_panel_bundle, validate_panel_dispatch,
    canonical_path, check_allowlist, sealed_reference_detail,
    check_gpu_guard, recompute_contract,
)

STATE_ROOT = REPO_ROOT / "runs" / "campaign_state"
STATE_PATH = STATE_ROOT / "unknown_supervisor_state.json"
EVENT_LOG = STATE_ROOT / "unknown_supervisor_events.jsonl"
SUPERVISOR_LOCK = STATE_ROOT / ".SUPERVISOR_RUNNING"
PANEL_ROOT = STATE_ROOT / "panels"
STATES = {"REPAIR_EVIDENCE", "READY", "RUNNING_BATCH", "SCORING", "WAITING_PANEL", "QUEUED_NEXT", "STOPPED"}
PANEL_ROLES = {
    "A": {"model": "gpt-5.6-sol", "reasoning_effort": "max"},
    "B": {"model": "gpt-5.6-terra", "reasoning_effort": "ultra"},
    "C": {"model": "gpt-5.6-sol", "reasoning_effort": "ultra"},
}


def now_ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def save_state(state: dict) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, STATE_PATH)


def event(state: dict, kind: str, **detail) -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    row = {"ts": now_ts(), "kind": kind, "state": state["state"], "attempt": state["attempt"], **detail}
    with EVENT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def initial_state(cfg: dict) -> dict:
    outer = cfg.get("outer_loop", {})
    return {"schema_version": "unknown_supervisor.v2", "state": "REPAIR_EVIDENCE", "attempt": 0, "queue": [], "completed": [],
            "failed": [], "dispatch_failures": [],
            "initial_design_approved": False,
            "initial_gate_passed": False, "post_gate_materialized": False,
            "wait_backoff_sec": 60, "waiting_event_emitted": False,
            "config_path": str(DEFAULT_CONFIG), "config_snapshot_sha256": None,
            "created_at": now_ts(), "outer_loop": outer}


def materialize_initial_queue(state: dict) -> None:
    panel = state.get("panel", {})
    if not (panel.get("approved") is True and panel.get("action") == "experiment_design"
            and panel.get("r3_artifact_path") and panel.get("evidence_packet_sha")):
        return
    cells = state["outer_loop"].get("initial_queue", {}).get("cells", [])
    if not state["queue"] and not state["completed"]:
        approval = panel["r3_artifact_path"]
        evidence_sha = panel["evidence_packet_sha"]
        state["queue"] = [{**dict(cell), "attempts": 0, "approved_queue_index": index,
                           "panel_r3_artifact": approval,
                           "evidence_packet_sha": evidence_sha} for index, cell in enumerate(cells)]


def initial_screen_completed(state: dict) -> bool:
    expected = [cell.get("id") for cell in state["outer_loop"].get("initial_queue", {}).get("cells", [])]
    completed = [item.get("id") for item in state.get("completed", [])]
    return bool(expected) and completed[:len(expected)] == expected


def post_gate_completed(state: dict) -> bool:
    initial = [cell.get("id") for cell in state["outer_loop"].get("initial_queue", {}).get("cells", [])]
    expected = [cell.get("id") for cell in state["outer_loop"].get("post_gate_cells", [])]
    completed = [item.get("id") for item in state.get("completed", [])]
    return bool(expected) and completed[len(initial):len(initial) + len(expected)] == expected


def materialize_post_gate_queue(state: dict) -> None:
    panel = state.get("panel", {})
    if (state["initial_gate_passed"] and initial_screen_completed(state) and not state["post_gate_materialized"]
            and panel.get("approved") is True and panel.get("action") == "next_queue"
            and panel.get("r3_artifact_path") and panel.get("evidence_packet_sha")
            and panel.get("evidence_packet_sha") != state.get("initial_design_approval_sha")):
        approval = panel["r3_artifact_path"]
        evidence_sha = panel["evidence_packet_sha"]
        state["queue"].extend({**dict(cell), "attempts": 0, "approved_queue_index": index,
                               "panel_r3_artifact": approval,
                               "evidence_packet_sha": evidence_sha}
                              for index, cell in enumerate(state["outer_loop"].get("post_gate_cells", [])))
        state["post_gate_materialized"] = True


def materialize_provisional_base_adoption_queue(state: dict, cfg: dict, config_path: Path) -> bool:
    """Append only the pre-registered LR008 cell after an explicit base adoption."""
    panel = state.get("panel", {})
    outer = state.get("outer_loop", {})
    adoption = outer.get("provisional_base_adoption")
    queue = outer.get("provisional_base_adoption_lr008_queue")
    if (panel.get("approved") is not True or panel.get("action") != "provisional_base_adoption_lr008"
            or not isinstance(adoption, dict) or not isinstance(queue, list) or len(queue) != 1
            or state.get("provisional_base_adoption_materialized")):
        return False
    base = adoption.get("base", {})
    identity_keys = ("id", "tier", "step", "seed")
    if (not isinstance(base, dict)
            or any(isinstance(row, dict) and all(row.get(key) == base.get(key) for key in identity_keys)
                   for row in state.get("completed", []) + state.get("failed", []))):
        return False
    lr = queue[0]
    ok, _ = validate_panel_dispatch(panel.get("r3_artifact_path"), panel.get("evidence_packet_sha"), lr,
                                    "provisional_base_adoption_lr008", sha256_file(config_path), cfg, config_path)
    if not ok:
        return False
    projection = adoption["projection"]
    projection_path = Path(projection.get("path", ""))
    expected_projection_sha = projection.get("sha256")
    if (not projection_path.is_file() or not isinstance(expected_projection_sha, str)
            or sha256_file(projection_path) != expected_projection_sha):
        return False
    projection_payload = load_json(projection_path, None)
    if (not isinstance(projection_payload, dict)
            or any(projection_payload.get(key) != base.get(key) for key in identity_keys)
            or sha256_file(projection_path) != expected_projection_sha):
        return False
    state.setdefault("completed", []).append(projection_payload)
    state["queue"].append({**dict(lr), "attempts": 0, "approved_queue_index": 0,
                           "panel_r3_artifact": panel["r3_artifact_path"],
                           "evidence_packet_sha": panel["evidence_packet_sha"]})
    state.setdefault("adopted_base_projections", []).append(dict(adoption["projection"]))
    state["provisional_base_adoption_materialized"] = True
    return True


def resource_blocked(cfg: dict) -> tuple[bool, dict]:
    safety = cfg.get("safety", {})
    disk_ok, disk_detail = check_disk_guard(safety.get("min_free_gb", {}))
    gpu_ok, gpu_detail = check_gpu_guard(safety)
    return not (disk_ok and gpu_ok), {"disk": disk_detail, "gpu": gpu_detail}


def _step_for_item(cfg: dict, item: dict) -> dict | None:
    for tier_cfg in cfg.get("tiers", []):
        if int(tier_cfg.get("tier")) == int(item.get("tier")):
            found = next((step for step in tier_cfg.get("steps", [])
                          if step.get("name") == item.get("step")), None)
            if found:
                found = dict(found)
                if item.get("panel_r3_artifact"):
                    found["panel_r3_artifact"] = item["panel_r3_artifact"]
                    found["evidence_packet_sha"] = item.get("evidence_packet_sha")
                found["dispatch_item"] = {
                    key: item.get(key) for key in ("id", "tier", "step", "seed")}
                found["approved_queue_index"] = item.get("approved_queue_index")
            return found
    return None


def validate_dispatch_result(cfg: dict, config_path: Path, item: dict, dispatch_id: str,
                             returncode: int, registry_path: Path = DEFAULT_REGISTRY) -> tuple[dict | None, str | None]:
    if returncode != 0:
        return None, f"executor_returncode={returncode}"
    step = _step_for_item(cfg, item)
    if step is None:
        return None, "queue_step_missing_from_config"
    campaign = Campaign(config_path=config_path, registry_path=registry_path)
    expected_fp = campaign._step_fingerprint(int(item["tier"]), step, item["seed"])
    records = [rec for rec in Registry(registry_path).read_all() if rec.get("dispatch_id") == dispatch_id]
    if not records:
        return None, "matching_registry_record_missing"
    rec = records[-1]
    if ((rec.get("tier"), rec.get("step"), rec.get("seed"))
            != (int(item["tier"]), item["step"], item["seed"])):
        return None, "matching_registry_identity_mismatch"
    if rec.get("campaign_id") != cfg.get("campaign_id"):
        return None, "campaign_id_mismatch"
    if rec.get("config_snapshot_sha256") != sha256_file(config_path):
        return None, "config_snapshot_sha_mismatch"
    if rec.get("step_fingerprint") != expected_fp:
        return None, "step_fingerprint_mismatch"
    if rec.get("evidence_packet_sha") != item.get("evidence_packet_sha"):
        return None, "evidence_packet_sha_mismatch"
    panel_path = item.get("panel_r3_artifact")
    if not panel_path or rec.get("panel_r3_artifact_sha256") != sha256_file(Path(panel_path)):
        return None, "panel_r3_artifact_sha_mismatch"
    if rec.get("approved_queue_index") != item.get("approved_queue_index"):
        return None, "approved_queue_index_mismatch"
    if rec.get("status") != "completed":
        return None, f"registry_final_status={rec.get('status')}"
    artifact = rec.get("artifact_path")
    if not artifact or not rec.get("artifact_sha256") or not Path(artifact).exists():
        return None, "artifact_missing"
    if sha256_tree(Path(artifact)) != rec["artifact_sha256"]:
        return None, "artifact_sha_mismatch"
    return rec, None


def _git_snapshot() -> dict:
    def git(*args):
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    dirty = git("diff", "--binary", "HEAD")
    status = git("status", "--porcelain=v1")
    return {"head": git("rev-parse", "HEAD"), "dirty_diff_sha256": sha256_obj(dirty), "status": status}


def ensure_split_overlap_audits(cfg: dict, force: bool = False) -> None:
    """Create/reuse CPU-only split audits before an evidence packet is sealed."""
    tool = REPO_ROOT / "scripts" / "audit_manifest_overlap.py"
    tool_sha = sha256_file(tool)
    allowlist = cfg.get("safety", {}).get("allowlist_roots", [])
    declared_train = set()
    declared_validation = set()
    for tier in cfg.get("tiers", []):
        for step in tier.get("steps", []):
            for value in (step.get("manifest"), step.get("env", {}).get("REPRO_DATA")):
                if value:
                    declared_train.add(canonical_path(value))
            value = step.get("rule_c", {}).get("offline_pool")
            if value:
                declared_validation.add(canonical_path(value))
    audit_root = (STATE_ROOT / "audits").resolve()
    for spec in cfg.get("safety", {}).get("split_overlap_audits", []):
        train = (REPO_ROOT / spec["train_manifest"]).resolve()
        validation = (REPO_ROOT / spec["validation_manifest"]).resolve()
        artifact = (REPO_ROOT / spec["artifact"]).resolve()
        if train == validation:
            raise ValueError(f"split_overlap_audit_same_manifest:{spec['name']}")
        if (canonical_path(train) not in declared_train
                or canonical_path(validation) not in declared_validation):
            raise ValueError(f"split_overlap_audit_not_declared_by_active_steps:{spec['name']}")
        for role, manifest in (("train", train), ("validation", validation)):
            if sealed_reference_detail({"manifest": str(manifest)}, cfg):
                raise ValueError(f"split_overlap_audit_sealed_reference:{spec['name']}:{role}")
            payload = load_json(manifest, None)
            if not isinstance(payload, dict) or not isinstance(payload.get("root"), str):
                raise ValueError(f"split_overlap_audit_manifest_invalid:{spec['name']}:{role}")
            if not check_allowlist(Path(payload["root"]).resolve(), allowlist):
                raise ValueError(f"split_overlap_audit_root_not_approved:{spec['name']}:{role}")
        if audit_root != artifact.parent and audit_root not in artifact.parents:
            raise ValueError(f"split_overlap_audit_artifact_outside_state_root:{spec['name']}")
        reusable = False
        if artifact.is_file() and not force:
            report = load_json(artifact, {})
            inputs = report.get("inputs", {})
            reusable = (
                report.get("schema_version") == "manifest_overlap_audit.v1"
                and report.get("tool_sha256") == tool_sha
                and inputs.get("train", {}).get("manifest_sha256") == sha256_file(train)
                and inputs.get("validation", {}).get("manifest_sha256") == sha256_file(validation)
                and int(report.get("near", {}).get("threshold", -1)) == int(spec["near_threshold"])
            )
        if reusable:
            continue
        command = [
            sys.executable, str(tool),
            "--train-manifest", str(train),
            "--validation-manifest", str(validation),
            "--out", str(artifact),
            "--near-threshold", str(spec["near_threshold"]),
            "--max-examples", str(spec.get("max_examples", 50)),
        ]
        for root in allowlist:
            command.extend(["--allowed-root", str(root)])
        result = subprocess.run(command, cwd=REPO_ROOT)
        if result.returncode not in (0, 1) or not artifact.is_file():
            raise RuntimeError(f"split_overlap_audit_failed:{spec['name']}:returncode={result.returncode}")


def evaluate_initial_screen(state: dict) -> dict:
    """Run the non-promoting paired screen on the two completed initial arms."""
    by_id = {item.get("id"): item for item in state.get("completed", [])}
    expected = state["outer_loop"].get("initial_queue", {}).get("cells", [])
    if len(expected) != 2 or any(cell.get("id") not in by_id for cell in expected):
        raise ValueError("initial screen requires both exact initial arms")
    base, candidate = (by_id[cell["id"]] for cell in expected)

    def artifact(item: dict) -> dict:
        values = item.get("rule_c_artifacts")
        if not isinstance(values, list) or len(values) != 1:
            raise ValueError(f"exactly one Rule-C artifact required:{item.get('id')}")
        value = values[0]
        required = {
            "selection_snapshot_path", "selection_snapshot_sha256",
            "offline_path", "offline_sha256",
        }
        if not required.issubset(value):
            raise ValueError(f"Rule-C artifact fields missing:{item.get('id')}")
        for path_key, sha_key in (
            ("selection_snapshot_path", "selection_snapshot_sha256"),
            ("offline_path", "offline_sha256"),
        ):
            path = Path(value[path_key])
            if not path.is_file() or sha256_file(path) != value[sha_key]:
                raise ValueError(f"Rule-C artifact hash mismatch:{item.get('id')}:{path_key}")
        return value

    base_artifact, candidate_artifact = artifact(base), artifact(candidate)
    input_paths = {
        "base_selector": base_artifact["selection_snapshot_path"],
        "lr_selector": candidate_artifact["selection_snapshot_path"],
        "base_offline": base_artifact["offline_path"],
        "lr_offline": candidate_artifact["offline_path"],
    }
    expected_shas = {
        "base_selector": base_artifact["selection_snapshot_sha256"],
        "lr_selector": candidate_artifact["selection_snapshot_sha256"],
        "base_offline": base_artifact["offline_sha256"],
        "lr_offline": candidate_artifact["offline_sha256"],
    }
    screen_id = sha256_obj(expected_shas)[:20]
    screen_dir = STATE_ROOT / "screens" / screen_id
    sha_path = screen_dir / "expected_input_shas.json"
    output_path = screen_dir / "screen.json"
    atomic_write_json(sha_path, expected_shas)
    settings = state["outer_loop"].get("initial_screen", {})
    command = [
        sys.executable, str(REPO_ROOT / "scripts" / "evaluate_rule_c_screen.py"),
        "--base-selector", input_paths["base_selector"],
        "--lr-selector", input_paths["lr_selector"],
        "--base-offline", input_paths["base_offline"],
        "--lr-offline", input_paths["lr_offline"],
        "--expected-shas", str(sha_path),
        "--out", str(output_path),
        "--bootstrap-seed", str(settings.get("bootstrap_seed", 42)),
        "--bootstrap-iterations", str(settings.get("bootstrap_iterations", 2000)),
        "--min-explicit-blocks", str(settings.get("min_explicit_blocks", 20)),
    ]
    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode != 0 or not output_path.is_file():
        return {
            "passed": False, "reasons": [f"screen_executor_failed:returncode={result.returncode}"],
            "input_sha256": expected_shas,
        }
    payload = load_json(output_path, {})
    gates = payload.get("gates", {})
    passed = (
        payload.get("promotes") is False
        and isinstance(gates, dict)
        and bool(gates)
        and all(value is True for value in gates.values())
        and payload.get("input_sha256") == expected_shas
    )
    return {
        "passed": passed,
        "reasons": payload.get("reasons", []) if not passed else [],
        "artifact_path": str(output_path.resolve()),
        "artifact_sha256": sha256_file(output_path),
        "expected_input_shas_path": str(sha_path.resolve()),
        "expected_input_shas_sha256": sha256_file(sha_path),
        "input_sha256": expected_shas,
        "gates": gates,
        "promotes": payload.get("promotes"),
    }


def create_evidence_packet(state: dict, cfg: dict, config_path: Path, action: str,
                           panel_root: Path | None = None) -> dict:
    # A panel decides one immutable proposal.  Hash the exact scripts, manifests,
    # backbone declarations, environment and ordered queue before it sees them.
    if action == "experiment_design":
        proposed_queue = state["outer_loop"].get("initial_queue", {}).get("cells", [])
        binding_queue = proposed_queue
    elif action == "next_queue":
        proposed_queue = state["outer_loop"].get("post_gate_cells", [])
        binding_queue = proposed_queue
    elif action == "screen_review":
        proposed_queue = []
        binding_queue = state["outer_loop"].get("initial_queue", {}).get("cells", [])
    elif action == "expansion_review":
        proposed_queue = []
        binding_queue = state["outer_loop"].get("post_gate_cells", [])
    elif action == "provisional_base_adoption_lr008":
        proposed_queue = state["outer_loop"].get("provisional_base_adoption_lr008_queue", [])
        binding_queue = proposed_queue
    else:
        raise ValueError(f"unsupported evidence action:{action}")
    all_steps = [item for tier in cfg.get("tiers", []) for item in tier.get("steps", [])]
    wanted = {(q.get("tier"), q.get("step"), q.get("seed")) for q in binding_queue}
    steps = [s for tier in cfg.get("tiers", []) for s in tier.get("steps", [])
             if (tier.get("tier"), s.get("name"), next((q.get("seed") for q in binding_queue
                                                           if q.get("tier") == tier.get("tier") and q.get("step") == s.get("name")), None)) in wanted]
    if not steps:
        raise ValueError("proposed_queue_has_no_configured_steps")
    for step in steps:
        if sealed_reference_detail(step, cfg):
            raise ValueError("sealed_reference_in_proposed_queue")
    backbones = sorted({str(s.get("backbone") or s.get("rule_c", {}).get("backbone"))
                        for s in steps if s.get("backbone") or s.get("rule_c", {}).get("backbone")})
    envs = {s.get("name"): s.get("env", {}) for s in steps}
    source_files = [REPO_ROOT / "scripts" / "run_unknown_campaign.py",
                    REPO_ROOT / "scripts" / "run_unknown_supervisor.py",
                    REPO_ROOT / "_grouping_eval.py"]
    ensure_split_overlap_audits(cfg)
    try:
        binding = recompute_contract(cfg, config_path, binding_queue)
    except ValueError as exc:
        if "split overlap audit image-content binding mismatch" not in str(exc):
            raise
        ensure_split_overlap_audits(cfg, force=True)
        binding = recompute_contract(cfg, config_path, binding_queue)
    # Keep the evidence field's legacy shape, but derive it from the single
    # current-byte contract pass rather than hashing every manifest a second time.
    manifest_records = [
        {"manifest": row["manifest"], "manifest_sha256": row["manifest_sha256"],
         "root": row["root"], "image_count": row["count"],
         "image_content_sha256": row["content_sha256"]}
        for row in binding["recompute_contract"]["data"]
    ]
    overlap_reports = [
        load_json((REPO_ROOT / spec["artifact"]).resolve(), {})
        for spec in cfg.get("safety", {}).get("split_overlap_audits", [])
    ]
    payload = {
        "campaign_id": cfg.get("campaign_id"),
        "config_snapshot_sha256": sha256_file(config_path),
        "completed": state.get("completed", []),
        "failed": state.get("failed", []),
        "dispatch_failures": state.get("dispatch_failures", []),
        "initial_screen": state.get("initial_screen"),
        "action": action,
        "proposed_queue": proposed_queue,
        "provisional_base_adoption": (state["outer_loop"].get("provisional_base_adoption")
                                       if action == "provisional_base_adoption_lr008" else None),
        "review_subject_queue": binding_queue if action in {"screen_review", "expansion_review"} else None,
        "provenance": {"git": _git_snapshot(), "effective_steps": [{"name": s.get("name"), "recipe": s.get("recipe", {}),
                       "env": s.get("env", {}), "command": s.get("command", []), "transforms": s.get("transforms"),
                       "determinism": s.get("determinism"), "seed": next((q.get("seed") for q in proposed_queue if q.get("step") == s.get("name")), None)} for s in steps],
                       "python": sys.version, "platform": sys.platform,
                       "packages": subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False).stdout.splitlines()},
        "manifest_content": manifest_records,
        "split_overlap_audits": overlap_reports,
        "binding": binding, "recompute_contract": binding["recompute_contract"],
    }
    evidence_sha = sha256_obj(payload)
    panel_id = f"unknown-{evidence_sha[:16]}"
    panel_dir = (panel_root or PANEL_ROOT) / panel_id
    evidence_path = panel_dir / "evidence_packet.json"
    atomic_write_json(evidence_path, payload)
    state["panel"] = {
        "panel_id": panel_id, "panel_dir": str(panel_dir.resolve()),
        "evidence_packet_path": str(evidence_path.resolve()),
        "evidence_packet_sha": evidence_sha,
        "panel_contract": {"rounds": {"R1": PANEL_ROLES, "R2": PANEL_ROLES},
                           "R3": {"C": PANEL_ROLES["C"]}},
        "contract": "R1 blind A/B/C -> R2 cross-rebuttal A/B/C -> R3 chair C binding",
        "action": action,
    }
    return state["panel"]


def ingest_panel(state: dict) -> tuple[bool, str]:
    panel = state.get("panel")
    if not panel:
        return False, "panel_not_initialized"
    panel_dir = Path(panel["panel_dir"])
    action = panel.get("action")
    if action not in {"experiment_design", "next_queue", "screen_review", "expansion_review", "provisional_base_adoption_lr008"}:
        return False, "panel_action_invalid"
    evidence = load_json(panel_dir / "evidence_packet.json", {})
    if evidence.get("action") != action:
        return False, "panel_action_binding_invalid"
    r3_path = panel_dir / "r3_C.json"
    ok, reason, r3 = validate_panel_bundle(r3_path, panel["evidence_packet_sha"])
    if not ok:
        return False, reason
    if r3.get("panel_id") != panel["panel_id"]:
        return False, "panel_r3_panel_id_mismatch"
    artifacts = [{"path": str((panel_dir / f"{round.lower()}_{judge}.json").resolve()),
                  "sha256": sha256_file(panel_dir / f"{round.lower()}_{judge}.json"),
                  "round": round, "judge": judge}
                 for round in ("R1", "R2") for judge in "ABC"]
    artifacts.append({"path": str(r3_path.resolve()), "sha256": sha256_file(r3_path), "round": "R3", "judge": "C"})
    panel.update({
        "artifacts": artifacts, "r3_artifact_path": str(r3_path.resolve()),
        "r3_artifact_sha256": sha256_file(r3_path),
        "chair_conclusion": r3.get("conclusion"),
        "critical_objections": r3.get("critical_objections", []),
        "minority_positions": r3.get("minority_positions", []),
        "approved": True,
    })
    return True, "approved"


def tick(state: dict, cfg: dict, config_path: Path,
         registry_path: Path = DEFAULT_REGISTRY) -> bool:
    """Advance one durable state.  True means safe stop was requested."""
    if state.get("state") == "STOPPED":
        return True
    # Old RUNNING_BATCH state can neither prove panel binding nor config identity.
    # It is quarantined instead of being resumed or retried.
    current_config_sha = sha256_file(config_path)
    incomplete_running = state.get("state") == "RUNNING_BATCH" and (
        not state.get("config_snapshot_sha256") or any(
            not item.get("panel_r3_artifact") or not item.get("evidence_packet_sha")
            for item in state.get("queue", [])))
    if state.get("schema_version") != "unknown_supervisor.v2" or incomplete_running or (
            state.get("config_snapshot_sha256") not in (None, current_config_sha)):
        previous = dict(state)
        previous_sha = sha256_obj(previous)
        archive_path = STATE_ROOT / "archive" / f"supervisor_state_{previous_sha[:16]}.json"
        if not archive_path.exists():
            atomic_write_json(archive_path, previous)
        fresh = initial_state(cfg)
        fresh.update({
            "config_path": str(config_path.resolve()),
            "config_snapshot_sha256": current_config_sha,
            "quarantined_previous_state": {
                "path": str(archive_path.resolve()), "sha256": previous_sha,
                "previous_state": previous.get("state"),
            },
        })
        state.clear()
        state.update(fresh)
        event(state, "legacy_or_config_drift_quarantined", reason="REPAIR_EVIDENCE",
              archive_path=str(archive_path.resolve()), archive_sha256=previous_sha)
        save_state(state)
        return False
    state["config_snapshot_sha256"] = current_config_sha
    if STOP_FLAG.exists():
        event(state, "safe_stop", reason="STOP_REQUESTED")
        state["state"] = "STOPPED"
        save_state(state)
        return True
    blocked, resource_detail = resource_blocked(cfg)
    if state["state"] == "RUNNING_BATCH" and state.get("queue") and blocked:
        fingerprint = sha256_obj(resource_detail)
        if state.get("resource_wait_fingerprint") != fingerprint:
            event(state, "resource_wait", detail=resource_detail)
        state["resource_wait_fingerprint"] = fingerprint
        save_state(state)
        return False
    state.pop("resource_wait_fingerprint", None)
    if state["state"] == "REPAIR_EVIDENCE":
        create_evidence_packet(state, cfg, config_path, "experiment_design")
        state["state"] = "WAITING_PANEL"
        state["waiting_event_emitted"] = False
        event(state, "evidence_repaired", queued=0, panel_required=True)
    elif state["state"] == "READY":
        materialize_post_gate_queue(state)
        state["state"] = "RUNNING_BATCH" if state["queue"] else "WAITING_PANEL"
        event(state, "batch_ready", queued=len(state["queue"]))
    elif state["state"] == "RUNNING_BATCH":
        if not state["queue"]:
            state["state"] = "SCORING"
            save_state(state)
            return False
        item = state["queue"][0]
        if not item.get("panel_r3_artifact") or not item.get("evidence_packet_sha"):
            state.update({"state": "REPAIR_EVIDENCE", "queue": [], "waiting_event_emitted": False})
            event(state, "incomplete_queue_quarantined", reason="REPAIR_EVIDENCE")
            save_state(state)
            return False
        state["attempt"] += 1
        item["attempts"] = int(item.get("attempts", 0)) + 1
        dispatch_id = f"{item['id']}.{state['attempt']}.{int(time.time())}.{os.getpid()}"
        command = [sys.executable, str(REPO_ROOT / "scripts" / "run_unknown_campaign.py"),
                   "--config", str(config_path), "--resume",
                   "--dispatch-tier", str(item["tier"]),
                   "--dispatch-step", str(item["step"]),
                   "--dispatch-seed", str(item["seed"]),
                   "--dispatch-id", dispatch_id]
        if item.get("panel_r3_artifact"):
            command.extend(["--panel-r3-artifact", item["panel_r3_artifact"],
                            "--evidence-packet-sha", item["evidence_packet_sha"]])
        result = subprocess.run(command, cwd=REPO_ROOT)
        rec, failure = validate_dispatch_result(
            cfg, config_path, item, dispatch_id, result.returncode, registry_path)
        event(state, "executor_returned", returncode=result.returncode,
              dispatch_id=dispatch_id, item_id=item["id"], validated=rec is not None, failure=failure)
        if rec is not None:
            state["completed"].append({**item, "run_id": rec.get("run_id"),
                                       "artifact_sha256": rec.get("artifact_sha256"),
                                       "config_snapshot_sha256": rec.get("config_snapshot_sha256"),
                                       "rule_c_artifacts": rec.get("rule_c_artifacts")})
            state["queue"].pop(0)
            adoption = state["outer_loop"].get("provisional_base_adoption", {})
            lr_queue = state["outer_loop"].get("provisional_base_adoption_lr008_queue")
            base = adoption.get("base", {}) if isinstance(adoption, dict) else {}
            if (isinstance(lr_queue, list) and len(lr_queue) == 1
                    and all(item.get(key) == base.get(key) for key in ("id", "tier", "step", "seed"))
                    and state["queue"] == lr_queue and not state.get("provisional_base_adoption_materialized")):
                state["queue"] = []
                state["provisional_base_adoption_pending"] = True
        else:
            state["dispatch_failures"].append({
                "ts": now_ts(), "item": dict(item), "dispatch_id": dispatch_id, "reason": failure})
            max_attempts = 1  # retry is prohibited: a new attempt needs a new panel bundle.
            if item["attempts"] >= max_attempts:
                state["failed"].append({**state["queue"].pop(0), "failure_reason": failure})
                initial_ids = {cell.get("id") for cell in state["outer_loop"].get("initial_queue", {}).get("cells", [])}
                if item.get("id") in initial_ids:
                    state["queue"] = []
        state["state"] = "QUEUED_NEXT" if state["queue"] else "SCORING"
    elif state["state"] == "SCORING":
        if state.pop("provisional_base_adoption_pending", False):
            evidence_action = "provisional_base_adoption_lr008"
        elif initial_screen_completed(state) and not state.get("initial_screen"):
            try:
                state["initial_screen"] = evaluate_initial_screen(state)
            except Exception as exc:
                state["initial_screen"] = {
                    "passed": False, "reasons": [f"screen_validation_failed:{exc}"]}
            state["initial_gate_passed"] = state["initial_screen"].get("passed") is True
            event(state, "initial_screen_evaluated", screen=state["initial_screen"])
        if 'evidence_action' not in locals() and post_gate_completed(state):
            evidence_action = "expansion_review"
        elif 'evidence_action' not in locals() and state.get("initial_gate_passed"):
            evidence_action = "next_queue"
        elif 'evidence_action' not in locals():
            evidence_action = "screen_review"
        create_evidence_packet(state, cfg, config_path, evidence_action)
        state["state"] = "WAITING_PANEL"
        state["waiting_event_emitted"] = False
        event(state, "scoring_registered")
    elif state["state"] == "QUEUED_NEXT":
        state["state"] = "READY"
        event(state, "next_queue_ready", queued=len(state["queue"]))
    else:  # WAITING_PANEL: intentionally durable and non-terminal.
        panel_approved, panel_reason = ingest_panel(state)
        if panel_approved and state["panel"].get("action") == "experiment_design" and not state.get("initial_design_approved"):
            state["initial_design_approved"] = True
            state["initial_design_approval_sha"] = state["panel"]["evidence_packet_sha"]
            materialize_initial_queue(state)
            state["wait_backoff_sec"] = 60
            state["waiting_event_emitted"] = False
            state["state"] = "QUEUED_NEXT"
            event(state, "panel_chair_approved_initial_design", panel=state["panel"])
        elif (panel_approved and state["panel"].get("action") == "next_queue"
              and initial_screen_completed(state) and state.get("initial_gate_passed")
              and not state.get("post_gate_materialized")):
            materialize_post_gate_queue(state)
            state["wait_backoff_sec"] = 60
            state["waiting_event_emitted"] = False
            state["state"] = "QUEUED_NEXT"
            event(state, "panel_chair_approved", panel=state["panel"])
        elif panel_approved and state["panel"].get("action") == "provisional_base_adoption_lr008" and materialize_provisional_base_adoption_queue(state, cfg, config_path):
            state["wait_backoff_sec"] = 60
            state["waiting_event_emitted"] = False
            state["state"] = "QUEUED_NEXT"
            event(state, "panel_chair_approved_provisional_base_adoption", panel=state["panel"])
        else:
            # The binding R3 contract permits only the exact queue in the
            # current immutable panel bundle; the legacy external inbox is not
            # a dispatch authority.
            if not state.get("waiting_event_emitted"):
                event(state, "waiting_panel", queued=0, panel_reason=panel_reason,
                      panel_dir=(state.get("panel") or {}).get("panel_dir"))
                state["waiting_event_emitted"] = True
            state["wait_backoff_sec"] = min(max(60, int(state.get("wait_backoff_sec", 60)) * 2), 1800)
    save_state(state)
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--initial-gate-pass", action="store_true",
                    help="record an explicit reviewed pass and materialize the seven pre-registered cells")
    args = ap.parse_args(argv)
    config_path = Path(args.config)
    cfg = load_json(config_path, {})
    state = load_json(STATE_PATH, initial_state(cfg))
    if state.get("state") not in STATES:
        raise SystemExit("invalid durable supervisor state")
    if args.initial_gate_pass:
        raise SystemExit("--initial-gate-pass is disabled; provide complete R1/R2/R3 panel artifacts")
    lock = ExclusiveFileLock(SUPERVISOR_LOCK)
    try:
        lock.acquire()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        while True:
            if tick(state, cfg, config_path) or args.once:
                return 0
            wait_sec = max(1, args.interval)
            if state["state"] == "WAITING_PANEL":
                wait_sec = max(wait_sec, int(state.get("wait_backoff_sec", 60)))
            time.sleep(wait_sec)
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
