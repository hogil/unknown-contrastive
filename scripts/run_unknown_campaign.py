#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unmanned multi-day campaign runner for the 4-tier unknown-grouping ladder.

Ladder (Tier 1 lowest-risk -> Tier 4 highest-cost, all expressed in
configs/unknown_campaign_v1.json, not hardcoded here):
  1. existing model direct predict (frozen FCMAE + current 2-seed proj ensemble)
  2. current best recipe retrain (frozen FCMAE, seeds 1/2/42, label-free epoch pick)
  3. recipe + grouping (HDBSCAN dial) sweep, pilot -> top4 -> top2 3-seed
  4. CNN-TAPT backbone init, same sweep, only promoted if it beats frozen champion

Operating recommendation is always the LOWEST tier that has passed its
promotion gate; a higher (later) tier passing its own gate never
overwrites an already-set champion (see Campaign._maybe_promote).

Every dispatched step is recorded as one JSON line in
runs/campaign_registry.jsonl (append-only, never rewritten) --
dataset/split sha256, tier, recipe sha256, seed, checkpoint type
(scripts/campaign_checkpoint.py), metrics, gate verdict, failure
reason, retry count, and champion before/after. --resume replays this
registry to figure out what is already done and continue from there.

CLI:
    python scripts/run_unknown_campaign.py --config configs/unknown_campaign_v1.json --resume
    python scripts/run_unknown_campaign.py --config configs/unknown_campaign_v1.json --status
    python scripts/run_unknown_campaign.py --config configs/unknown_campaign_v1.json --resume --max-tier 2
    python scripts/run_unknown_campaign.py --config configs/unknown_campaign_v1.json --stop-after-current

Absolute rules enforced here (see docs/paper/... and CLAUDE.md):
  - GPU training: at most 1 concurrent process for this campaign and at most
    the configured 40% VRAM fraction.
  - Explicitly allowlisted external GPU services may coexist only when the
    40% project allocation plus configured headroom is actually free.
  - No run starts if D: free < safety.min_free_gb.D or E: free <
    safety.min_free_gb.E.
  - Only paths under safety.allowlist_roots may be used as a data pool.
  - A checkpoint/predictor pairing that scripts/campaign_checkpoint.py
    flags as incompatible is never dispatched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scripts.campaign_checkpoint import (  # noqa: E402
    classify_checkpoint,
    check_predictor_compatibility,
    sha256_file,
)

DEFAULT_CONFIG = REPO_ROOT / "configs" / "unknown_campaign_v1.json"
DEFAULT_REGISTRY = REPO_ROOT / "runs" / "campaign_registry.jsonl"
STATE_DIR = REPO_ROOT / "runs" / "campaign_state"
STOP_FLAG = STATE_DIR / "STOP_REQUESTED"
CHAMPION_FILE = STATE_DIR / "champion.json"
CAMPAIGN_LOCK = STATE_DIR / ".RUNNING"
DECISION_REQUIRED_ACTIONS = {
    "experiment_design", "next_queue", "metric_change", "gate_change", "dial_change",
    "split_change", "champion_change", "tier_escalation", "stop_strategy",
    "retry_strategy", "tapt_entry", "claim_change",
    "provisional_base_adoption_lr008",
}


def canonical_path(value: str | Path, base: Path = REPO_ROOT) -> str:
    """Canonical, case-insensitive path identity without requiring existence."""
    path = Path(str(value))
    if not path.is_absolute():
        path = base / path
    return os.path.normcase(str(path.resolve(strict=False))).casefold()


def atomic_write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


PANEL_CONTRACT = {
    "A": ("gpt-5.6-sol", "max"),
    "B": ("gpt-5.6-terra", "ultra"),
    "C": ("gpt-5.6-sol", "ultra"),
}
PANEL_BINDING_FIELDS = {
    "config_snapshot_sha256", "source_snapshot_sha256",
    "data_snapshot_sha256", "backbone_snapshot_sha256", "env_snapshot_sha256",
    "ordered_queue_sha256",
}

REPRO_SOURCE_FILES = ("scripts/run_unknown_campaign.py", "scripts/run_unknown_supervisor.py",
    "scripts/materialize_panel_round.py",
    "scripts/run_rule_c_selector.py", "scripts/run_rule_c_v3_reselector.py",
    "scripts/run_rule_c_offline.py",
    "scripts/evaluate_rule_c_screen.py", "scripts/audit_manifest_overlap.py", "_grouping_eval.py",
    "_may_ablation.py", "_may_repro_src.py", "scripts/cluster_metrics.py",
    "scripts/eval_open_set_embeddings.py", "scripts/_common.py")


def _source_tokens_in_order(path: Path, tokens: tuple[bytes, ...]) -> bool:
    """Fail closed unless the immutable entrypoint source has this exact order."""
    try:
        source = path.read_bytes()
    except OSError:
        return False
    offset = 0
    for token in tokens:
        offset = source.find(token, offset)
        if offset < 0:
            return False
        offset += len(token)
    return True


def trainer_gpu_memory_hook_contract_ok(step: dict) -> bool:
    """Approve only the campaign's B4 trainer and its early CUDA cap hook."""
    if step.get("command") != ["{python}", "_may_ablation.py", "B4"]:
        return False
    return _source_tokens_in_order(REPO_ROOT / "_may_repro_src.py", (
        b"def main():",
        b"torch.cuda.set_per_process_memory_fraction(_gpu_fraction, device=0)",
        b"seed_all(CFG[\"SEED\"])",
        b"torch.device(\"cuda\"",
        b"model = CL().to(device)",
    ))


def rule_c_selector_gpu_memory_hook_contract_ok() -> bool:
    """Require the selector cap to precede embedding/model construction."""
    path = REPO_ROOT / "scripts" / "run_rule_c_selector.py"
    return (
        _source_tokens_in_order(path, (
        b"def embeddings(",
        b"core.load_backbone(backbone)",
        b"p.load_state_dict(state);p.eval().to(device)",
        ))
        and _source_tokens_in_order(path, (
            b"def main(argv=None):",
            b"torch.cuda.set_per_process_memory_fraction(float(raw),device=0)",
            b"emb=embeddings(",
        ))
    )

def _git_binding():
    def call(*args):
        p=subprocess.run(["git",*args],cwd=REPO_ROOT,capture_output=True,text=True,encoding="utf-8",errors="replace",check=False)
        if p.returncode: raise RuntimeError("git binding unavailable")
        return p.stdout
    dirty=call("diff","--binary","HEAD"); status=call("status","--porcelain=v1")
    return {"head":call("rev-parse","HEAD").strip(),"dirty_diff_sha256":hashlib.sha256(dirty.encode()).hexdigest(),"status_sha256":hashlib.sha256(status.encode()).hexdigest()}

def recompute_contract(cfg: dict, config_path: str | Path, proposed_queue: list[dict]) -> dict:
    """Deterministic launch binding, shared by evidence creation and dispatch."""
    # This cache lasts for one contract recomputation only.  Every dispatch
    # therefore still hashes the current bytes afresh, while aliases such as
    # validation and validation-unlabeled do not reread the same physical file.
    content_sha_cache: dict[str, str] = {}

    def cached_sha256(path: str | Path) -> str:
        resolved = Path(path).resolve()
        key = canonical_path(resolved)
        if key not in content_sha_cache:
            content_sha_cache[key] = sha256_file(resolved)
        return content_sha_cache[key]

    steps=[]
    for item in proposed_queue:
        found=None
        for tier in cfg.get("tiers",[]):
            if tier.get("tier")==item.get("tier"): found=next((s for s in tier.get("steps",[]) if s.get("name")==item.get("step")),None)
        if not found: raise ValueError("approved queue step missing")
        steps.append({"item":{k:item[k] for k in ("id","tier","step","seed")},"recipe":found.get("recipe",{}),"env":found.get("env",{}),"command":found.get("command",[]),"rule_c":found.get("rule_c",{})})
    sources={}
    for rel in REPRO_SOURCE_FILES:
        p=REPO_ROOT/rel
        if not p.is_file(): raise ValueError(f"required source missing:{p}")
        sources[str(p.resolve())]=cached_sha256(p)
    manifests=[]
    for step in steps:
        for value in (cfgstep:=step).get("rule_c",{}).get("unlabeled_pool"), cfgstep.get("rule_c",{}).get("offline_pool"):
            if value: manifests.append(value)
    for step in steps:
        # training manifest is represented by REPRO_DATA when present
        if step["env"].get("REPRO_DATA"): manifests.append(step["env"]["REPRO_DATA"])
    data=[]
    for raw in sorted(set(manifests)):
        mp=(REPO_ROOT/raw).resolve(); payload=read_json(mp); root=Path(payload["root"]).resolve()
        if not check_allowlist(root,cfg.get("safety",{}).get("allowlist_roots",[])): raise ValueError("manifest root outside approved roots")
        rows=[]
        for e in payload["files"]:
            rel=e["path"] if isinstance(e,dict) else e; f=(root/rel).resolve()
            if root not in f.parents or not f.is_file(): raise ValueError("manifest image missing/outside root")
            rows.append((str(rel),cached_sha256(f)))
        data.append({"manifest":str(mp),"root":str(root),"count":len(rows),"content_sha256":sha256_obj(rows),"manifest_sha256":cached_sha256(mp)})
    data_by_manifest={canonical_path(row["manifest"]):row for row in data}
    split_audits=[]
    for spec in cfg.get("safety",{}).get("split_overlap_audits",[]):
        required={"name","train_manifest","validation_manifest","artifact","near_threshold"}
        if not required.issubset(spec): raise ValueError("split overlap audit config incomplete")
        train_path=(REPO_ROOT/spec["train_manifest"]).resolve()
        validation_path=(REPO_ROOT/spec["validation_manifest"]).resolve()
        train_record=data_by_manifest.get(canonical_path(train_path))
        validation_record=data_by_manifest.get(canonical_path(validation_path))
        # Ignore audit declarations unrelated to the exact proposed queue.
        if train_record is None and validation_record is None: continue
        if train_record is None or validation_record is None:
            raise ValueError("split overlap audit pair incomplete for proposed queue")
        artifact=(REPO_ROOT/spec["artifact"]).resolve()
        if not artifact.is_file(): raise ValueError(f"split overlap audit missing:{artifact}")
        report=read_json(artifact)
        inputs=report.get("inputs",{})
        train_input,validation_input=inputs.get("train",{}),inputs.get("validation",{})
        if (report.get("schema_version")!="manifest_overlap_audit.v1"
                or canonical_path(train_input.get("path",""))!=canonical_path(train_path)
                or canonical_path(validation_input.get("path",""))!=canonical_path(validation_path)
                or train_input.get("manifest_sha256")!=train_record["manifest_sha256"]
                or validation_input.get("manifest_sha256")!=validation_record["manifest_sha256"]):
            raise ValueError("split overlap audit manifest binding mismatch")
        if (train_input.get("image_content_sha256")!=train_record["content_sha256"]
                or validation_input.get("image_content_sha256")!=validation_record["content_sha256"]):
            raise ValueError("split overlap audit image-content binding mismatch")
        exact=report.get("exact",{})
        if (report.get("status")=="overlap_found" or int(exact.get("content_pair_count",-1))!=0
                or int(exact.get("same_resolved_path_count",-1))!=0):
            raise ValueError("split overlap audit exact duplicate/path overlap")
        if int(report.get("near",{}).get("threshold",-1))!=int(spec["near_threshold"]):
            raise ValueError("split overlap audit threshold mismatch")
        audit_tool=REPO_ROOT/"scripts"/"audit_manifest_overlap.py"
        if report.get("tool_sha256")!=cached_sha256(audit_tool):
            raise ValueError("split overlap audit tool binding mismatch")
        split_audits.append({"name":spec["name"],"artifact":str(artifact),
            "artifact_sha256":cached_sha256(artifact),"status":report.get("status"),
            "review_required":report.get("review_required"),
            "near_candidate_pair_count":report.get("near",{}).get("candidate_pair_count"),
            "train_content_sha256":train_record["content_sha256"],
            "validation_content_sha256":validation_record["content_sha256"]})
    backs=[]
    for s in steps:
        p=s["rule_c"].get("backbone")
        if p:
            bp=(REPO_ROOT/p).resolve();
            if not bp.is_file(): raise ValueError("backbone missing")
            # Keep one record per configured step for a byte-compatible
            # backbone_snapshot_sha256; only the file read is deduplicated.
            backs.append({"path":str(bp),"sha256":cached_sha256(bp)})
    try:
        import torch
        torch_env={"version":torch.__version__,"cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),
                   "cuda_available":torch.cuda.is_available(),"device_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except Exception as exc:
        raise RuntimeError(f"torch environment unavailable:{exc}") from exc
    pip=subprocess.run([sys.executable,"-m","pip","freeze"],capture_output=True,text=True,encoding="utf-8",errors="replace",check=False)
    if pip.returncode: raise RuntimeError("pip freeze unavailable")
    env={"steps":steps,"python":sys.version,"platform":platform.platform(),"pip_freeze":sorted(pip.stdout.splitlines()),"torch":torch_env,"git":_git_binding(),"repro_contract":cfg.get("repro_contract",{})}
    return {"source_snapshot_sha256":sha256_obj(sources),
            "data_snapshot_sha256":sha256_obj({"manifests":data,"split_overlap_audits":split_audits}),
            "backbone_snapshot_sha256":sha256_obj(backs),"env_snapshot_sha256":sha256_obj(env),
            "config_snapshot_sha256":cached_sha256(config_path),"ordered_queue_sha256":sha256_obj(proposed_queue),
            "recompute_contract":{"sources":sources,"data":data,"split_overlap_audits":split_audits,
                                  "backbones":backs,"environment":env,"queue":proposed_queue}}


def validate_panel_bundle(path: str | Path | None,
                          expected_evidence_sha: str | None = None) -> tuple[bool, str, dict | None]:
    """Fail closed unless the complete A/B/C R1+R2+C R3 bundle is bound.

    R3 is deliberately never a standalone approval token: each R2 must bind all
    R1 response hashes and R3 must bind all R2 response hashes.  The evidence
    packet's immutable binding is also copied by hash into every round.
    """
    if not path or not Path(path).is_file():
        return False, "panel_r3_artifact_missing", None
    r3_path = Path(path)
    panel_dir = r3_path.parent
    try:
        r3 = read_json(r3_path)
        evidence = read_json(panel_dir / "evidence_packet.json")
    except Exception as exc:
        return False, f"panel_bundle_invalid:{exc}", None
    evidence_sha = sha256_obj(evidence)
    evidence_file_sha = sha256_file(panel_dir / "evidence_packet.json")
    if expected_evidence_sha and evidence_sha != expected_evidence_sha:
        return False, "panel_evidence_packet_sha_mismatch", r3
    if r3.get("panel_id") != f"unknown-{evidence_sha[:16]}":
        return False, "panel_id_not_derived_from_evidence", r3
    binding = evidence.get("binding")
    if not isinstance(binding, dict) or not PANEL_BINDING_FIELDS.issubset(binding):
        return False, "panel_binding_fields_missing", r3
    binding_sha = sha256_obj(binding)
    panel_id = r3.get("panel_id")
    # These receipts are structural orchestration evidence, not cryptographic
    # attestation.  The collaboration service logs remain the external authority.
    responses: dict[str, dict[str, str]] = {"R1": {}, "R2": {}}
    receipt_times: dict[str, dict[str, tuple[datetime, datetime]]] = {"R1": {}, "R2": {}}
    agent_ids: dict[str, str] = {}
    for round_name in ("R1", "R2"):
        for judge, contract in PANEL_CONTRACT.items():
            artifact = panel_dir / f"{round_name.lower()}_{judge}.json"
            try:
                payload = read_json(artifact)
            except Exception:
                return False, f"panel_{round_name}_{judge}_missing", r3
            required = {"panel_id", "judge", "round", "model", "reasoning_effort", "prompt_sha",
                        "prompt_path", "prompt_sha256", "receipt_path", "receipt_sha256",
                        "evidence_packet_sha", "evidence_packet_file_sha256", "binding_sha256", "response_sha", "timestamp", "response"}
            if not required.issubset(payload):
                return False, f"panel_{round_name}_{judge}_fields_missing", r3
            if (payload.get("panel_id"), payload.get("judge"), payload.get("round"),
                    payload.get("model"), payload.get("reasoning_effort")) != (panel_id, judge, round_name, *contract):
                return False, f"panel_{round_name}_{judge}_identity_invalid", r3
            if (payload.get("evidence_packet_sha") != evidence_sha or payload.get("evidence_packet_file_sha256") != evidence_file_sha or
                    payload.get("binding_sha256") != binding_sha or
                    payload.get("response_sha") != sha256_obj(payload.get("response"))):
                return False, f"panel_{round_name}_{judge}_sha_invalid", r3
            try:
                prompt_path = Path(payload["prompt_path"])
                receipt_path = Path(payload["receipt_path"])
                prompt = read_json(prompt_path)
                receipt = read_json(receipt_path)
                if sha256_file(prompt_path) != payload["prompt_sha256"] or sha256_file(receipt_path) != payload["receipt_sha256"]:
                    return False, f"panel_{round_name}_{judge}_prompt_or_receipt_sha_invalid", r3
                if payload["prompt_sha"] != payload["prompt_sha256"]:
                    return False, f"panel_{round_name}_{judge}_prompt_sha_invalid", r3
                if (prompt.get("panel_id"), prompt.get("judge"), prompt.get("round"), prompt.get("evidence_packet_sha"), prompt.get("evidence_packet_file_sha256")) != (panel_id, judge, round_name, evidence_sha, evidence_file_sha):
                    return False, f"panel_{round_name}_{judge}_prompt_identity_invalid", r3
                if (receipt.get("model"), receipt.get("reasoning_effort"), receipt.get("judge"), receipt.get("round"), receipt.get("prompt_sha256")) != (*contract, judge, round_name, payload["prompt_sha256"]):
                    return False, f"panel_{round_name}_{judge}_receipt_identity_invalid", r3
                agent_id = receipt.get("agent_id")
                if not isinstance(agent_id, str) or not agent_id or not isinstance(receipt.get("task_name"), str):
                    return False, f"panel_{round_name}_{judge}_receipt_agent_invalid", r3
                if judge in agent_ids and agent_ids[judge] != agent_id:
                    return False, f"panel_{round_name}_{judge}_agent_inconsistent", r3
                agent_ids[judge] = agent_id
                started, completed = datetime.fromisoformat(receipt["started_at"]), datetime.fromisoformat(receipt["completed_at"])
                datetime.fromisoformat(payload["timestamp"])
                if completed < started: return False, f"panel_{round_name}_{judge}_receipt_time_invalid", r3
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                return False, f"panel_{round_name}_{judge}_prompt_or_receipt_invalid", r3
            receipt_times[round_name][judge] = (started, completed)
            responses[round_name][judge] = payload["response_sha"]
            if round_name == "R1":
                if any(key in prompt for key in ("r1_response_shas", "r2_response_shas", "peer_response_shas")):
                    return False, f"panel_R1_{judge}_not_blind", r3
            elif payload.get("r1_response_shas") != responses["R1"] or prompt.get("r1_response_shas") != responses["R1"]:
                return False, f"panel_R2_{judge}_r1_binding_invalid", r3
        if len(set(responses[round_name].values())) != 3:
            return False, f"panel_{round_name}_responses_not_distinct", r3
    if len(set(agent_ids.values())) != 3:
        return False, "panel_agents_not_distinct", r3
    if max(value[1] for value in receipt_times["R1"].values()) > min(value[0] for value in receipt_times["R2"].values()):
        return False, "panel_round_order_R1_R2_invalid", r3
    required_r3 = {"panel_id", "judge", "round", "model", "reasoning_effort", "prompt_sha", "prompt_path", "prompt_sha256", "receipt_path", "receipt_sha256",
                   "evidence_packet_sha", "evidence_packet_file_sha256", "binding_sha256", "response_sha", "timestamp", "response",
                   "r2_response_shas", "conclusion"}
    if not required_r3.issubset(r3):
        return False, "panel_r3_fields_missing", r3
    if (r3.get("panel_id"), r3.get("judge"), r3.get("round"), r3.get("model"),
            r3.get("reasoning_effort")) != (panel_id, "C", "R3", *PANEL_CONTRACT["C"]):
        return False, "panel_r3_identity_invalid", r3
    if (r3.get("evidence_packet_sha") != evidence_sha or r3.get("evidence_packet_file_sha256") != evidence_file_sha or r3.get("binding_sha256") != binding_sha or
            r3.get("response_sha") != sha256_obj(r3.get("response")) or
            r3.get("r2_response_shas") != responses["R2"]):
        return False, "panel_r3_binding_invalid", r3
    try:
        prompt_path, receipt_path = Path(r3["prompt_path"]), Path(r3["receipt_path"])
        prompt, receipt = read_json(prompt_path), read_json(receipt_path)
        if (sha256_file(prompt_path) != r3["prompt_sha256"] or sha256_file(receipt_path) != r3["receipt_sha256"] or r3["prompt_sha"] != r3["prompt_sha256"]):
            return False, "panel_r3_prompt_or_receipt_sha_invalid", r3
        if (prompt.get("panel_id"), prompt.get("judge"), prompt.get("round"), prompt.get("evidence_packet_sha"), prompt.get("evidence_packet_file_sha256"), prompt.get("r2_response_shas")) != (panel_id, "C", "R3", evidence_sha, evidence_file_sha, responses["R2"]):
            return False, "panel_r3_prompt_binding_invalid", r3
        if (receipt.get("agent_id"), receipt.get("model"), receipt.get("reasoning_effort"), receipt.get("judge"), receipt.get("round"), receipt.get("prompt_sha256")) != (agent_ids["C"], *PANEL_CONTRACT["C"], "C", "R3", r3["prompt_sha256"]):
            return False, "panel_r3_receipt_identity_invalid", r3
        started, completed = datetime.fromisoformat(receipt["started_at"]), datetime.fromisoformat(receipt["completed_at"])
        datetime.fromisoformat(r3["timestamp"])
        if completed < started or max(value[1] for value in receipt_times["R2"].values()) > started:
            return False, "panel_round_order_R2_R3_invalid", r3
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False, "panel_r3_prompt_or_receipt_invalid", r3
    if str(r3.get("conclusion")).casefold() != "approve":
        return False, "panel_r3_not_approved", r3
    return True, "ok", r3


def validate_panel_r3_artifact(path: str | Path | None,
                               expected_evidence_sha: str | None = None) -> tuple[bool, str, dict | None]:
    """Compatibility entrypoint; R3-only validation is intentionally impossible."""
    return validate_panel_bundle(path, expected_evidence_sha)


def validate_near_audit_adjudications(r3: dict, evidence: dict) -> tuple[bool, str]:
    """Require C to explicitly close every evidence-bound near-duplicate review."""
    bound = evidence.get("binding", {}).get("recompute_contract", {}).get("split_overlap_audits", [])
    required = [item for item in bound if item.get("review_required") is True]
    if not required:
        return True, "ok"
    adjudications = r3.get("near_audit_adjudications")
    if not isinstance(adjudications, list):
        return False, "panel_near_audit_adjudication_missing"
    for audit in required:
        path = Path(str(audit.get("artifact", "")))
        expected_sha = audit.get("artifact_sha256")
        if not path.is_file() or not isinstance(expected_sha, str) or sha256_file(path) != expected_sha:
            return False, "panel_near_audit_artifact_drift"
        report = read_json(path)
        near = report.get("near", {})
        count, examples = near.get("candidate_pair_count"), near.get("examples", [])
        match = next((item for item in adjudications if isinstance(item, dict)
                      and item.get("audit_path") == str(path) and item.get("audit_sha256") == expected_sha), None)
        if not match:
            return False, "panel_near_audit_adjudication_unbound"
        examples_bound = (match.get("near_candidate_examples") == examples or
                          match.get("near_candidate_examples_sha256") == sha256_obj(examples))
        if (match.get("near_candidate_pair_count") != count or not examples_bound or
                match.get("verdict") != "approve_no_material_leakage" or
                not isinstance(match.get("rationale"), str) or not match["rationale"].strip()):
            return False, "panel_near_audit_adjudication_invalid"
        provenance = report.get("provenance", {})
        block_count = provenance.get("cross_split_block_overlap_count", 0)
        if not isinstance(block_count, int) or block_count < 0:
            return False, "panel_block_overlap_report_invalid"
        if block_count > 0:
            block_examples = provenance.get("cross_split_block_overlap_examples", [])
            if (match.get("cross_split_block_overlap_count") != block_count or
                    match.get("cross_split_block_overlap_examples_sha256") != sha256_obj(block_examples) or
                    match.get("block_verdict") != "approve_no_material_source_leakage" or
                    not isinstance(match.get("block_rationale"), str) or
                    not match["block_rationale"].strip()):
                return False, "panel_block_overlap_adjudication_invalid"
    return True, "ok"


def validate_panel_dispatch(path: str | Path | None, evidence_sha: str | None,
                            item: dict, required_action: str,
                            config_snapshot_sha256: str, cfg: dict | None = None,
                            config_path: str | Path | None = None) -> tuple[bool, str]:
    """Bind a launch to one exact, ordered item from its approved evidence."""
    ok, reason, r3 = validate_panel_bundle(path, evidence_sha)
    if not ok:
        return False, reason
    try:
        evidence = read_json(Path(path).parent / "evidence_packet.json")
    except Exception as exc:
        return False, f"panel_evidence_unreadable:{exc}"
    binding = evidence.get("binding", {})
    queue = evidence.get("proposed_queue")
    audit_ok, audit_reason = validate_near_audit_adjudications(r3 or {}, evidence)
    if not audit_ok:
        return False, audit_reason
    if evidence.get("action") != required_action:
        return False, "panel_action_mismatch"
    if cfg is None:
        return False, "panel_config_required_for_exact_queue"
    if required_action == "experiment_design":
        expected_queue = cfg.get("outer_loop", {}).get("initial_queue", {}).get("cells", [])
    elif required_action == "next_queue":
        expected_queue = cfg.get("outer_loop", {}).get("post_gate_cells", [])
    elif required_action == "provisional_base_adoption_lr008":
        outer = cfg.get("outer_loop", {})
        adoption = outer.get("provisional_base_adoption")
        expected_queue = outer.get("provisional_base_adoption_lr008_queue")
        if not isinstance(adoption, dict) or not isinstance(expected_queue, list) or len(expected_queue) != 1:
            return False, "panel_provisional_base_adoption_config_invalid"
        if evidence.get("provisional_base_adoption") != adoption:
            return False, "panel_provisional_base_adoption_evidence_mismatch"
        receipt, projection = adoption.get("receipt"), adoption.get("projection")
        source_config_snapshot_sha256 = adoption.get("source_config_snapshot_sha256")
        if (not isinstance(receipt, dict) or not isinstance(projection, dict)
                or not isinstance(source_config_snapshot_sha256, str) or not source_config_snapshot_sha256):
            return False, "panel_provisional_base_adoption_receipt_or_projection_missing"
        response = (r3 or {}).get("response")
        decision = response.get("provisional_base_adoption") if isinstance(response, dict) else None
        if not isinstance(decision, dict) or decision.get("decision") != "adopt":
            return False, "panel_provisional_base_adoption_not_adopted"
        for label, configured in (("receipt", receipt), ("projection", projection)):
            path_value, expected_sha = configured.get("path"), configured.get("sha256")
            if (not isinstance(path_value, str) or not isinstance(expected_sha, str)
                    or decision.get(f"{label}_path") != path_value
                    or decision.get(f"{label}_sha256") != expected_sha):
                return False, f"panel_provisional_base_adoption_{label}_binding_mismatch"
            source = Path(path_value)
            if not source.is_file() or sha256_file(source) != expected_sha:
                return False, f"panel_provisional_base_adoption_{label}_drift"
        try:
            receipt_payload = read_json(Path(receipt["path"]))
        except Exception as exc:
            return False, f"panel_provisional_base_adoption_receipt_unreadable:{exc}"
        if (isinstance(receipt_payload, dict)
                and "source_config_snapshot_sha256" in receipt_payload
                and receipt_payload["source_config_snapshot_sha256"] != source_config_snapshot_sha256):
            return False, "panel_provisional_base_adoption_receipt_source_config_mismatch"
        base = adoption.get("base")
        completed, failed = evidence.get("completed", []), evidence.get("failed", [])
        if not isinstance(base, dict) or any(not isinstance(rows, list) for rows in (completed, failed)):
            return False, "panel_provisional_base_adoption_base_state_invalid"
        identity_keys = ("id", "tier", "step", "seed")
        if any(isinstance(row, dict) and all(row.get(key) == base.get(key) for key in identity_keys)
               for row in completed + failed):
            return False, "panel_provisional_base_adoption_base_state_preexisting"
        try:
            projection_payload = read_json(Path(projection["path"]))
        except Exception as exc:
            return False, f"panel_provisional_base_adoption_projection_unreadable:{exc}"
        if not isinstance(projection_payload, dict) or any(projection_payload.get(key) != base.get(key) for key in identity_keys):
            return False, "panel_provisional_base_adoption_projection_identity_mismatch"
        required_projection = {"run_id", "artifact_path", "artifact_sha256", "source_config_snapshot_sha256", "rule_c_artifacts"}
        if not required_projection.issubset(projection_payload):
            return False, "panel_provisional_base_adoption_projection_fields_missing"
        if projection_payload["source_config_snapshot_sha256"] != source_config_snapshot_sha256:
            return False, "panel_provisional_base_adoption_projection_source_config_mismatch"
        artifact_path = Path(str(projection_payload["artifact_path"]))
        if not artifact_path.exists() or sha256_tree(artifact_path) != projection_payload["artifact_sha256"]:
            return False, "panel_provisional_base_adoption_projection_artifact_drift"
        artifacts = projection_payload.get("rule_c_artifacts")
        if not isinstance(artifacts, list) or len(artifacts) != 1:
            return False, "panel_provisional_base_adoption_rule_c_artifact_invalid"
        artifact = artifacts[0]
        for path_key, sha_key in (("selection_snapshot_path", "selection_snapshot_sha256"),
                                  ("offline_path", "offline_sha256")):
            source, expected_sha = Path(str(artifact.get(path_key, ""))), artifact.get(sha_key)
            if not source.is_file() or not isinstance(expected_sha, str) or sha256_file(source) != expected_sha:
                return False, "panel_provisional_base_adoption_rule_c_artifact_drift"
    else:
        return False, "panel_dispatch_action_not_supported"
    if queue != expected_queue:
        return False, "panel_queue_not_exact_config_order"
    queue_ids = [row.get("id") for row in queue if isinstance(row, dict)]
    if len(queue_ids) != len(queue) or None in queue_ids or len(set(queue_ids)) != len(queue_ids):
        return False, "panel_queue_ids_invalid"
    if required_action == "next_queue":
        screen = evidence.get("initial_screen")
        if not isinstance(screen, dict) or screen.get("passed") is not True or screen.get("promotes") is not False:
            return False, "panel_initial_screen_not_passed"
        gates = screen.get("gates")
        if not isinstance(gates, dict) or not gates or not all(value is True for value in gates.values()):
            return False, "panel_initial_screen_gates_invalid"
        screen_path = Path(str(screen.get("artifact_path", "")))
        if (not screen_path.is_file() or not screen.get("artifact_sha256")
                or sha256_file(screen_path) != screen["artifact_sha256"]):
            return False, "panel_initial_screen_artifact_drift"
        try:
            screen_payload = read_json(screen_path)
        except Exception as exc:
            return False, f"panel_initial_screen_unreadable:{exc}"
        if (screen_payload.get("promotes") is not False
                or screen_payload.get("gates") != gates
                or screen_payload.get("input_sha256") != screen.get("input_sha256")):
            return False, "panel_initial_screen_payload_mismatch"
        sidecar_path = Path(str(screen.get("expected_input_shas_path", "")))
        if (not sidecar_path.is_file() or not screen.get("expected_input_shas_sha256")
                or sha256_file(sidecar_path) != screen["expected_input_shas_sha256"]):
            return False, "panel_initial_screen_sidecar_drift"
        try:
            sidecar = read_json(sidecar_path)
        except Exception as exc:
            return False, f"panel_initial_screen_sidecar_unreadable:{exc}"
        if sidecar != screen.get("input_sha256"):
            return False, "panel_initial_screen_sidecar_mismatch"
        initial_cells = (cfg or {}).get("outer_loop", {}).get("initial_queue", {}).get("cells", [])
        completed_by_id = {row.get("id"): row for row in evidence.get("completed", [])}
        if len(initial_cells) != 2 or any(cell.get("id") not in completed_by_id for cell in initial_cells):
            return False, "panel_initial_screen_completed_arms_missing"
        bound_inputs = {}
        for prefix, cell in zip(("base", "lr"), initial_cells):
            completed = completed_by_id[cell["id"]]
            artifacts = completed.get("rule_c_artifacts")
            if not isinstance(artifacts, list) or len(artifacts) != 1:
                return False, "panel_initial_screen_rule_c_artifacts_invalid"
            artifact = artifacts[0]
            source_payloads = {}
            for kind, path_key, sha_key in (
                ("selector", "selection_snapshot_path", "selection_snapshot_sha256"),
                ("offline", "offline_path", "offline_sha256"),
            ):
                source_path = Path(str(artifact.get(path_key, "")))
                expected_sha = artifact.get(sha_key)
                if not source_path.is_file() or not expected_sha or sha256_file(source_path) != expected_sha:
                    return False, "panel_initial_screen_source_artifact_drift"
                bound_inputs[f"{prefix}_{kind}"] = expected_sha
                try:
                    source_payloads[kind] = read_json(source_path)
                except Exception as exc:
                    return False, f"panel_initial_screen_source_unreadable:{exc}"
            configured_step = None
            for tier_cfg in cfg.get("tiers", []):
                if tier_cfg.get("tier") == cell.get("tier"):
                    configured_step = next(
                        (row for row in tier_cfg.get("steps", []) if row.get("name") == cell.get("step")),
                        None,
                    )
            rule_c = (configured_step or {}).get("rule_c", {})
            unlabeled_path = (REPO_ROOT / str(rule_c.get("unlabeled_pool", ""))).resolve()
            labeled_path = (REPO_ROOT / str(rule_c.get("offline_pool", ""))).resolve()
            if not unlabeled_path.is_file() or not labeled_path.is_file():
                return False, "panel_initial_screen_current_manifest_missing"
            selector_payload, offline_payload = source_payloads["selector"], source_payloads["offline"]
            if (canonical_path(selector_payload.get("pool", "")) != canonical_path(unlabeled_path)
                    or selector_payload.get("pool_sha256") != sha256_file(unlabeled_path)):
                return False, "panel_initial_screen_unlabeled_manifest_drift"
            if (offline_payload.get("selection_snapshot_sha256") != artifact["selection_snapshot_sha256"]
                    or canonical_path(offline_payload.get("unlabeled_pool_path", "")) != canonical_path(unlabeled_path)
                    or offline_payload.get("unlabeled_pool_sha256") != sha256_file(unlabeled_path)
                    or canonical_path(offline_payload.get("labeled_pool_path", "")) != canonical_path(labeled_path)
                    or offline_payload.get("labeled_pool_sha256") != sha256_file(labeled_path)):
                return False, "panel_initial_screen_offline_manifest_binding_mismatch"
            if (offline_payload.get("bundle_path") != selector_payload.get("bundle_path")
                    or offline_payload.get("bundle_sha256") != selector_payload.get("bundle_sha256")):
                return False, "panel_initial_screen_bundle_binding_mismatch"
            for path_key, sha_key in (
                ("bundle_path", "bundle_sha256"), ("npz_path", "npz_sha256")):
                provenance_path = Path(str(offline_payload.get(path_key, "")))
                if (not provenance_path.is_file() or not offline_payload.get(sha_key)
                        or sha256_file(provenance_path) != offline_payload[sha_key]):
                    return False, "panel_initial_screen_provenance_artifact_drift"
        if bound_inputs != screen.get("input_sha256"):
            return False, "panel_initial_screen_completed_binding_mismatch"
    if not isinstance(queue, list) or binding.get("ordered_queue_sha256") != sha256_obj(queue):
        return False, "panel_ordered_queue_invalid"
    if binding.get("config_snapshot_sha256") != config_snapshot_sha256:
        return False, "panel_config_binding_drift"
    identity = {key: item.get(key) for key in ("id", "tier", "step", "seed")}
    if None in identity.values() or identity not in [{key: q.get(key) for key in identity} for q in queue]:
        return False, "panel_item_not_approved"
    if cfg is not None:
        try: current=recompute_contract(cfg,config_path or DEFAULT_CONFIG,queue)
        except Exception as exc: return False, f"panel_recompute_failed:{exc}"
        for key, reason in (("source_snapshot_sha256","panel_source_binding_drift"),("data_snapshot_sha256","panel_data_binding_drift"),("backbone_snapshot_sha256","panel_backbone_binding_drift"),("env_snapshot_sha256","panel_env_binding_drift"),("ordered_queue_sha256","panel_queue_binding_drift")):
            if binding.get(key)!=current.get(key): return False, reason
    return True, "ok"


class ExclusiveFileLock:
    """Atomic O_EXCL lock with token-verified release."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.token = uuid.uuid4().hex
        self.acquired = False

    def acquire(self) -> dict:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"pid": os.getpid(), "started_at": now_ts(), "host": platform.node(),
                   "owner_token": self.token}
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            info = read_json(self.path) if self.path.exists() else {}
            raise RuntimeError(f"lock already held: {self.path} owner={info}")
        try:
            os.write(fd, json.dumps(payload).encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        self.acquired = True
        return payload

    def release(self) -> bool:
        if not self.acquired or not self.path.exists():
            return False
        try:
            info = read_json(self.path)
        except Exception:
            return False
        if info.get("owner_token") != self.token:
            return False
        self.path.unlink()
        self.acquired = False
        return True


def now_ts() -> str:
    return datetime.now().strftime("%y%m%d_%H%M%S")


def sha256_obj(obj: Any) -> str:
    """sha256 of a canonical (sorted-key) JSON encoding -- used for recipe fingerprints."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def sha256_tree(path: Path) -> str:
    """Stable digest of a file or directory artifact."""
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = item.relative_to(path).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(sha256_file(item).encode("ascii"))
    return digest.hexdigest()


def read_json(path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj) -> None:
    atomic_write_json(Path(path), obj)


# ===================== safety guards =====================
def disk_free_gb(drive_root: str) -> float:
    return shutil.disk_usage(drive_root).free / 1e9


def check_disk_guard(min_free_gb: dict) -> tuple[bool, dict]:
    """min_free_gb: {"D": 200, "E": 500} -- drive letter -> minimum free GB."""
    detail = {}
    ok = True
    for drive, min_gb in min_free_gb.items():
        root = f"{drive}:/"
        try:
            free = disk_free_gb(root)
        except OSError as e:
            detail[drive] = {"free_gb": None, "min_gb": min_gb, "ok": False, "error": str(e)}
            ok = False
            continue
        drive_ok = free >= float(min_gb)
        detail[drive] = {"free_gb": round(free, 1), "min_gb": min_gb, "ok": drive_ok}
        ok = ok and drive_ok
    return ok, detail


def check_allowlist(pool_path, allowlist_roots: list[str]) -> bool:
    """A pool path is safe only if it resolves inside one of allowlist_roots."""
    try:
        p = Path(pool_path).resolve()
    except OSError:
        return False
    for root in allowlist_roots:
        try:
            r = Path(root).resolve()
        except OSError:
            continue
        if p == r or r in p.parents:
            return True
        # a manifest .json is allowed to describe files rooted under an allowlisted dir
        # even though the manifest file itself may live under data/pools/ (in-repo);
        # callers that hold a manifest should pass its "root" field here instead.
    return False


def gpu_free_mb() -> float | None:
    """None if nvidia-smi is unavailable (treated as 'unknown', caller decides fallback)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def gpu_total_mb() -> float | None:
    """Total VRAM on CUDA device 0, or None when it cannot be queried."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def scan_running_locks(search_roots: list[Path]) -> list[dict]:
    """Find every */.RUNNING lock file under the given roots (recursive) and parse it.

    Mirrors the lock convention already used by _may_repro_src.py (pid +
    started_at + data paths). A stale lock (pid no longer alive on this
    host) is reported with alive=False rather than silently dropped --
    the caller decides whether that still counts as "GPU busy".
    """
    import psutil  # already a project dependency (scripts/_resource_monitor.py)
    found = []
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for lock_path in root.rglob(".RUNNING"):
            if lock_path.resolve(strict=False) == CAMPAIGN_LOCK.resolve(strict=False):
                continue  # orchestration lock is not a GPU-training lock
            try:
                info = json.loads(lock_path.read_text(encoding="utf-8"))
            except Exception:
                info = {}
            pid = info.get("pid")
            alive = bool(pid) and psutil.pid_exists(int(pid))
            found.append({"lock_path": str(lock_path), "pid": pid, "alive": alive, **info})
    return found


def check_gpu_guard(cfg_safety: dict) -> tuple[bool, dict]:
    """True == GPU is available for this campaign to start a training step on."""
    free_mb = gpu_free_mb()
    total_mb = gpu_total_mb()
    locks = scan_running_locks([REPO_ROOT / "runs"])
    alive_locks = [l for l in locks if l["alive"]]
    max_concurrent = int(cfg_safety.get("gpu_max_concurrent", 1))
    # Locks are advisory only; query actual GPU compute PIDs as the authority.
    gpu_pids = []; gpu_processes = []
    probe_ok = False
    try:
        probe = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=10, check=False)
        if probe.returncode == 0:
            lines = [x.strip() for x in probe.stdout.splitlines() if x.strip()]
            if all(x.split(",", 1)[0].strip().isdigit() for x in lines):
                import psutil
                gui_allow = {str(x).casefold() for x in cfg_safety.get("gpu_non_compute_process_allowlist", [])}
                coexist_allow = {str(x).casefold() for x in cfg_safety.get("gpu_coexist_process_allowlist", [])}
                for row in lines:
                    pid=int(row.split(",",1)[0].strip())
                    try: name=psutil.Process(pid).name().casefold()
                    except (psutil.Error, OSError): name="<unreadable>"
                    classification = ("gui_allowlisted" if name in gui_allow else
                                      "coexist_allowlisted" if name in coexist_allow else "blocked")
                    gpu_pids.append(pid); gpu_processes.append(
                        {"pid":pid,"name":name,"classification":classification,
                         "allowlisted":classification != "blocked"})
                probe_ok = True
    except (OSError, subprocess.SubprocessError):
        gpu_pids = []
    busy_by_lock = len(alive_locks) >= max_concurrent
    busy_by_process = (not probe_ok) or any(not p["allowlisted"] for p in gpu_processes)
    try:
        min_free_mb = float(cfg_safety.get("gpu_min_free_mb", 1024))
        fraction = float(cfg_safety.get("gpu_memory_fraction", 0.40))
        headroom_mb = float(cfg_safety.get("gpu_headroom_mb", 0))
    except (TypeError, ValueError):
        min_free_mb, fraction, headroom_mb = float("nan"), float("nan"), float("nan")
    memory_query_ok = total_mb is not None and free_mb is not None and total_mb > 0 and min_free_mb >= 0 and 0.0 < fraction <= 1.0 and headroom_mb >= 0
    required_free_mb = max(min_free_mb, total_mb * fraction + headroom_mb) if memory_query_ok else None
    busy_by_mem = (not memory_query_ok) or free_mb < required_free_mb
    detail = {"total_mb": total_mb, "free_mb": free_mb, "min_free_mb": min_free_mb,
               "gpu_memory_fraction": fraction, "headroom_mb": headroom_mb,
               "required_free_mb": required_free_mb, "memory_query_ok": memory_query_ok,
               "alive_locks": alive_locks,
               "max_concurrent": max_concurrent, "gpu_compute_pids": gpu_pids, "gpu_processes":gpu_processes, "gpu_pid_probe_ok": probe_ok,
               "busy_by_lock": busy_by_lock, "busy_by_process": busy_by_process, "busy_by_mem": busy_by_mem}
    ok = not (busy_by_lock or busy_by_process or busy_by_mem)
    return ok, detail


# ===================== registry =====================
class Registry:
    """Append-only JSONL ledger. Never rewritten -- --resume replays it."""

    FIELDS_DOC = (
        "ts, run_id, tier, tier_name, step, seed, dataset_sha256, manifest_path, "
        "recipe, recipe_sha256, checkpoint_path, checkpoint_type, checkpoint_sha256, "
        "predictor, metrics, gate, status, failure_reason, retry_count, attempt, "
        "champion_before, champion_after, duration_sec"
    )

    def __init__(self, path=DEFAULT_REGISTRY):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, record: dict) -> None:
        record = dict(record)
        record.setdefault("ts", now_ts())
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a torn last line from a killed process
        return out

    def completed_steps(self) -> set[tuple]:
        """Set of (tier, step, seed) keys whose latest attempt has status == 'completed'."""
        latest: dict[tuple, dict] = {}
        for rec in self.read_all():
            key = (rec.get("tier"), rec.get("step"), rec.get("seed"))
            latest[key] = rec  # later lines override earlier -> "latest attempt wins"
        return {k for k, rec in latest.items() if rec.get("status") == "completed"}

    def current_champion(self) -> dict | None:
        champ = None
        for rec in self.read_all():
            if rec.get("champion_after"):
                champ = rec
        return champ

    def latest(self, tier, step, seed) -> dict | None:
        """Latest record for one logical step; used for artifact dependencies."""
        found = None
        for rec in self.read_all():
            if (rec.get("tier"), rec.get("step"), rec.get("seed")) == (tier, step, seed):
                found = rec
        return found

    def latest_matching(self, *, campaign_id, config_snapshot_sha256, step_fingerprint,
                        tier, step, seed) -> dict | None:
        found = None
        for rec in self.read_all():
            if (rec.get("campaign_id") == campaign_id
                    and rec.get("config_snapshot_sha256") == config_snapshot_sha256
                    and rec.get("step_fingerprint") == step_fingerprint
                    and (rec.get("tier"), rec.get("step"), rec.get("seed")) == (tier, step, seed)):
                found = rec
        return found


# ===================== promotion gate (hardcoded thresholds) =====================
# Values below are the literal numbers from the approved campaign plan (team-lead,
# 260726). Do not loosen without an explicit user decision -- record any change in
# configs/unknown_campaign_v1.json's "gates" block (this function only reads it).
DEFAULT_GATES = {
    "capture_drop_pp_max": 1.0,
    "far_increase_max": 0.5,  # units: alarms per held-out background batch (NOT a percentage -- 260726 team-lead fix)
    "far_or_noise_improve_pp_min": 3.0,
    "far_or_noise_improve_datasets_min": 2,
    "ari_ami_drop_max": 0.02,
    "n_seeds_min": 3,
    "temporal_far_events_max": 0,
    "temporal_detect_batches_max": 2,
    "temporal_novel_min_per_batch": 20,
    "clean546_capture_required": "7/7",
    "clean546_noise_pct_target": 2.57,
    "clean546_noise_pct_tolerance_pp": 1.0,
    "severstal_capture_required": "4/4",
    "severstal_noise_pct_baseline": 77.74,
    "severstal_noise_pct_improve_min_pp": 20.0,
}


def _frac(capture_str: str) -> float:
    num, den = capture_str.split("/")
    return float(num) / float(den)


def evaluate_gate(gates_cfg: dict, metrics: dict, n_seeds_min_override: int | None = None) -> tuple[bool, list[str]]:
    """Pure function: (gate thresholds, candidate metrics) -> (passed, reasons).

    `metrics` schema (any missing top-level key is treated as "insufficient
    data for that check" -- the check fails closed, i.e. counts as NOT
    passed, with a reason explaining what is missing):
      n_seeds: int
      primary_wafer: {name: {"capture_drop_pp": float}}   # baseline - candidate, in pp
      background_far: {"far_per_batch_delta": float}  # alarms/batch, candidate minus baseline
      far_or_noise_improvements: {name: {"far_improve_pp": float, "noise_improve_pp": float}}
      ari: {"drop": float}   # baseline - candidate
      ami: {"drop": float}
      temporal: {"far_events": int, "detect_batches": int, "novel_per_batch": int}
      clean546: {"capture": "7/7", "noise_pct": float}
      severstal: {"capture": "4/4", "noise_pct": float}
    """
    g = {**DEFAULT_GATES, **(gates_cfg or {})}
    reasons: list[str] = []
    passed = True

    def fail(msg: str) -> None:
        nonlocal passed
        passed = False
        reasons.append(msg)

    n_seeds_min = n_seeds_min_override if n_seeds_min_override is not None else g["n_seeds_min"]
    n_seeds = metrics.get("n_seeds")
    if n_seeds is None:
        fail("missing n_seeds (need >= %d seeds + bootstrap CI to rule out a lucky run)" % n_seeds_min)
    elif n_seeds < n_seeds_min:
        fail(f"n_seeds={n_seeds} < required {n_seeds_min}")

    primary = metrics.get("primary_wafer")
    if not primary:
        fail("missing primary_wafer capture metrics")
    else:
        for name, m in primary.items():
            drop = m.get("capture_drop_pp")
            if drop is None:
                fail(f"primary_wafer[{name}]: missing capture_drop_pp")
            elif drop > g["capture_drop_pp_max"]:
                fail(f"primary_wafer[{name}]: capture dropped {drop:.2f}pp > "
                     f"{g['capture_drop_pp_max']}pp max")

    bg = metrics.get("background_far")
    if not bg or bg.get("far_per_batch_delta") is None:
        fail("missing background_far.far_per_batch_delta")
    elif bg["far_per_batch_delta"] > g["far_increase_max"]:
        fail(f"background_far increased {bg['far_per_batch_delta']:.3f} alarms/batch > "
             f"{g['far_increase_max']} alarms/batch max")

    improved = metrics.get("far_or_noise_improvements") or {}
    n_improved = 0
    for name, m in improved.items():
        far_imp = m.get("far_improve_pp") or 0.0
        noise_imp = m.get("noise_improve_pp") or 0.0
        if far_imp >= g["far_or_noise_improve_pp_min"] or noise_imp >= g["far_or_noise_improve_pp_min"]:
            n_improved += 1
    if n_improved < g["far_or_noise_improve_datasets_min"]:
        fail(f"only {n_improved}/{g['far_or_noise_improve_datasets_min']} datasets improved FAR/noise "
             f">= {g['far_or_noise_improve_pp_min']}pp")

    ari = metrics.get("ari")
    if not ari or ari.get("drop") is None:
        fail("missing ari.drop")
    elif ari["drop"] > g["ari_ami_drop_max"]:
        fail(f"ARI dropped {ari['drop']:.4f} > {g['ari_ami_drop_max']} max")
    ami = metrics.get("ami")
    if not ami or ami.get("drop") is None:
        fail("missing ami.drop")
    elif ami["drop"] > g["ari_ami_drop_max"]:
        fail(f"AMI dropped {ami['drop']:.4f} > {g['ari_ami_drop_max']} max")

    temporal = metrics.get("temporal")
    if not temporal:
        fail("missing temporal held-out replay metrics")
    else:
        if temporal.get("far_events") is None or temporal["far_events"] > g["temporal_far_events_max"]:
            fail(f"temporal background FAR events={temporal.get('far_events')} > "
                 f"{g['temporal_far_events_max']} allowed")
        if temporal.get("novel_per_batch") is None or temporal["novel_per_batch"] < g["temporal_novel_min_per_batch"]:
            fail(f"temporal novel_per_batch={temporal.get('novel_per_batch')} < "
                 f"{g['temporal_novel_min_per_batch']} required")
        if temporal.get("detect_batches") is None or temporal["detect_batches"] > g["temporal_detect_batches_max"]:
            fail(f"temporal detect_batches={temporal.get('detect_batches')} > "
                 f"{g['temporal_detect_batches_max']} allowed")

    c546 = metrics.get("clean546")
    if not c546 or c546.get("capture") is None or c546.get("noise_pct") is None:
        fail("missing clean546 regression-test metrics")
    else:
        if _frac(c546["capture"]) < _frac(g["clean546_capture_required"]):
            fail(f"clean546 capture {c546['capture']} < required {g['clean546_capture_required']}")
        target = g["clean546_noise_pct_target"]
        tol = g["clean546_noise_pct_tolerance_pp"]
        if abs(c546["noise_pct"] - target) > tol:
            fail(f"clean546 noise_pct={c546['noise_pct']:.2f}% not within {tol}pp of "
                 f"target {target}% (transductive reproduction)")

    sev = metrics.get("severstal")
    if sev is not None:  # only enforced once a Severstal milestone metric is supplied
        if sev.get("capture") is None or sev.get("noise_pct") is None:
            fail("severstal metrics present but incomplete (capture/noise_pct)")
        else:
            if _frac(sev["capture"]) < _frac(g["severstal_capture_required"]):
                fail(f"severstal capture {sev['capture']} < required {g['severstal_capture_required']}")
            improve = g["severstal_noise_pct_baseline"] - sev["noise_pct"]
            if improve < g["severstal_noise_pct_improve_min_pp"]:
                fail(f"severstal noise_pct improved only {improve:.2f}pp < "
                     f"{g['severstal_noise_pct_improve_min_pp']}pp required vs "
                     f"{g['severstal_noise_pct_baseline']}% baseline")

    return passed, reasons


# ===================== best-effort metrics adapter =====================
def extract_metrics_from_run(run_dir: Path) -> dict:
    """Best-effort mapping from a grouping_deploy.py / offline-eval output dir
    into a partial evaluate_gate() metrics dict. Only fills what is actually
    on disk (summary.json / offline_summary.json) -- everything else is left
    absent so evaluate_gate() fails closed on it rather than guessing.

    This is deliberately thin: the campaign plan's per-dataset gate fields
    (primary_wafer, background_far, far_or_noise_improvements, temporal) need
    a baseline to diff against, which this generic adapter does not have. A
    tier's step config may point "metrics_path" at a hand-assembled JSON that
    already matches evaluate_gate()'s schema -- that takes priority over this
    adapter when present.
    """
    out: dict = {}
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        s = read_json(summary_path)
        out["_label_free_summary"] = {"n": s.get("n"), "k": s.get("k"), "noise_pct": s.get("noise_pct"),
                                        "mean_coherence": s.get("mean_coherence"),
                                        "mean_stability": s.get("mean_stability")}
    offline_path = run_dir / "offline_summary.json"
    if offline_path.exists():
        o = read_json(offline_path)
        out["_offline_summary"] = o
        if "ARI" in o:
            out.setdefault("ari", {})["candidate_raw"] = o["ARI"]
        if "AMI" in o:
            out.setdefault("ami", {})["candidate_raw"] = o["AMI"]
    return out

def _capture_frac(capture_str) -> float | None:
    """"31/31" -> 0.9686... style fraction parse; returns None for anything else."""
    if not capture_str or "/" not in str(capture_str):
        return None
    try:
        num, den = str(capture_str).split("/")
        return float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        return None


def load_temporal_metrics(report_dir, operating_point: dict) -> dict | None:
    """Reads a perf-temporal summary_tables.csv (schema: metric,arm,P,K,size05,
    size10,size20,size30,unit) for one (arm,P,K,size_col) cell and returns
    {"far_events": int, "detect_batches": int, "novel_per_batch": int,
    "held_out_batches": int}. held_out_batches is parsed out of the
    "FAR_alarms_over_N_bg_batches" metric name itself (N), not hardcoded --
    if the sim's held-out window changes, that name changes with it and this
    keeps following it (260726 team-lead fix). Returns None (fail-closed
    upstream) if the file or the requested row is missing -- never estimates.
    """
    path = Path(report_dir) / "summary_tables.csv"
    if not path.exists():
        return None
    import csv
    import re
    p_val, k_val = str(operating_point.get("P")), str(operating_point.get("K"))
    size_col, arm = operating_point.get("size_col"), operating_point.get("arm")
    if not (p_val and k_val and size_col and arm):
        return None
    rows = {}
    far_metric_name = None
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("arm") == arm and row.get("P") == p_val and row.get("K") == k_val:
                metric = row.get("metric")
                rows[metric] = row.get(size_col)
                if metric and metric.startswith("FAR_alarms_over_"):
                    far_metric_name = metric
    far = next((v for k, v in rows.items() if k and k.startswith("FAR_alarms_over_")), None)
    lag = rows.get("true_positive_detection_lag")
    if far is None or lag is None or far_metric_name is None:
        return None
    m = re.search(r"over_(\d+)_bg_batches", far_metric_name)
    if not m:
        return None
    held_out_batches = int(m.group(1))
    digits = "".join(ch for ch in size_col if ch.isdigit())
    return {"far_events": int(far), "detect_batches": int(lag),
            "novel_per_batch": int(digits) if digits else None,
            "held_out_batches": held_out_batches}


def load_background_far_delta(report_dir, operating_point: dict, baseline_arm: str = "frozen") -> dict | None:
    """held-out background FAR rate delta = candidate arm's (alarms / held-out
    batches) minus the baseline (frozen) arm's, at the same (P,K,size_col)
    cell. held-out batch count is parsed dynamically from the metric name
    (see load_temporal_metrics), never hardcoded. Returns
    {"far_per_batch_delta": float} -- alarms per batch, matching
    evaluate_gate()'s "far_increase_max" threshold unit (260726 team-lead
    fix; the earlier "increase_pp" event-count-delta version conflated units
    with the gate's percentage-point threshold and is retired). Returns None
    if either side is missing.
    """
    cand = load_temporal_metrics(report_dir, operating_point)
    base = load_temporal_metrics(report_dir, {**operating_point, "arm": baseline_arm})
    if cand is None or base is None:
        return None
    cand_rate = cand["far_events"] / cand["held_out_batches"]
    base_rate = base["far_events"] / base["held_out_batches"]
    return {"far_per_batch_delta": round(cand_rate - base_rate, 4),
            "_unit_note": "alarms per held-out background batch, candidate minus baseline"}



def build_gate_metrics(step: dict, cfg: dict, run_dir: Path) -> dict:
    """Assembles evaluate_gate()'s metrics dict by diffing this step's candidate
    result against a config-declared frozen baseline (cfg["baselines"][dataset])
    and, if configured, a perf-temporal summary_tables.csv. Every field is only
    set when a real number was found on both sides -- no estimates, so
    evaluate_gate() keeps failing closed on anything this adapter can't fill.
    """
    metrics: dict = {}
    if step.get("n_seeds_reported") is not None:
        metrics["n_seeds"] = step["n_seeds_reported"]
    elif step.get("seeds"):
        metrics["n_seeds"] = len(step["seeds"])

    dataset = step.get("baseline_dataset")
    baselines = cfg.get("baselines", {})
    base = baselines.get(dataset) if dataset else None
    offline_path = run_dir / "offline_summary.json"
    cand = read_json(offline_path) if offline_path.exists() else None

    if cand and base and dataset:
        cand_cap, base_cap = _capture_frac(cand.get("P1_capture")), _capture_frac(base.get("P1_capture"))
        if cand_cap is not None and base_cap is not None:
            metrics.setdefault("primary_wafer", {})[dataset] = {
                "capture_drop_pp": round((base_cap - cand_cap) * 100.0, 4)}
        if "ARI" in cand and "ARI" in base:
            metrics["ari"] = {"drop": round(base["ARI"] - cand["ARI"], 4)}
        if "AMI" in cand and "AMI" in base:
            metrics["ami"] = {"drop": round(base["AMI"] - cand["AMI"], 4)}
        if cand.get("P2_noise_pct") is not None and base.get("noise_pct") is not None:
            improve = base["noise_pct"] - cand["P2_noise_pct"]
            metrics.setdefault("far_or_noise_improvements", {})[dataset] = {"noise_improve_pp": round(improve, 4)}
        if dataset == "clean546":
            metrics["clean546"] = {"capture": cand.get("P1_capture"), "noise_pct": cand.get("P2_noise_pct")}
        if dataset == "severstal":
            metrics["severstal"] = {"capture": cand.get("P1_capture"), "noise_pct": cand.get("P2_noise_pct")}

    temporal_dir, op = step.get("temporal_report_path"), step.get("operating_point")
    if temporal_dir and op:
        t = load_temporal_metrics(temporal_dir, op)
        if t:
            metrics["temporal"] = t
        bg = load_background_far_delta(temporal_dir, op)
        if bg:
            metrics["background_far"] = bg

    return metrics




def resolve_trainer_output_dir(search_root: Path, tag: str, cell: str, after_ts: float) -> Path:
    """Find _may_ablation.py's own timestamped output dir for one (REPRO_TAG, cell)
    pair (it computes CFG["OUTPUT_DIR"] + "_" + RUN_TS internally -- there is no
    --out flag to hand it a path directly). Pattern: abl{tag}_{cell}_<RUN_TS>,
    searched recursively under search_root, restricted to directories created at
    or after `after_ts` (the step's own start time) so a stale dir from an
    earlier attempt with the same REPRO_TAG is never picked up.

    Raises RuntimeError if there are zero or more-than-one candidates -- per
    260726 team-lead directive, silently picking one when there are multiple
    could aggregate a different seed's results into this step's record.
    """
    pattern = f"abl{tag}_{cell}_*"
    # 2s tolerance: OS/filesystem timestamp precision can otherwise let a
    # directory created a few ms after after_ts read back with a slightly
    # earlier ctime (observed on NTFS) -- real subprocess dispatch always
    # has far more latency than this, so it never risks matching a genuinely
    # stale run from an earlier attempt (those are seconds-to-minutes old).
    cutoff = after_ts - 2.0
    candidates = sorted(
        p for p in search_root.rglob(pattern)
        if p.is_dir() and p.stat().st_ctime >= cutoff
    )
    if not candidates:
        raise RuntimeError(f"no trainer output dir found matching {search_root}/**/{pattern} "
                            f"created at or after step start ({after_ts})")
    if len(candidates) > 1:
        raise RuntimeError(f"ambiguous trainer output dir: {len(candidates)} candidates match "
                            f"{pattern} created after step start -- {[str(c) for c in candidates]}")
    return candidates[0]


# ===================== step dispatch =====================
def build_command(step: dict, context: dict) -> list[str]:
    """step["command"] is an argv list of strings; "{placeholders}" are filled
    from context (python executable, run_dir, seed, manifest path, ...)."""
    py = context.get("python", sys.executable)
    fmt_ctx = {"python": py, **context}
    argv = []
    for tok in step["command"]:
        try:
            argv.append(str(tok).format(**fmt_ctx))
        except (KeyError, IndexError) as e:
            raise ValueError(f"unresolved placeholder in command token {tok!r}: {e}") from e
    return argv


def build_env(step: dict, context: dict) -> dict:
    """step["env"] (extra env vars, e.g. _may_repro_src.py's REPRO_* contract) with the
    same "{placeholder}" substitution as build_command. Returns {} if step has no "env"."""
    fmt_ctx = {"python": context.get("python", sys.executable), **context}
    env = {}
    for k, v in step.get("env", {}).items():
        try:
            env[k] = str(v).format(**fmt_ctx)
        except (KeyError, IndexError) as e:
            raise ValueError(f"unresolved placeholder in env {k}={v!r}: {e}") from e
    return env


def terminate_process_tree(pid: int, timeout_sec: float = 10.0) -> None:
    """Terminate descendants before the parent (important for Windows workers)."""
    try:
        import psutil
        parent = psutil.Process(pid)
        processes = parent.children(recursive=True) + [parent]
        for process in reversed(processes):
            try:
                process.terminate()
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(processes, timeout=timeout_sec)
        for process in alive:
            try:
                process.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(alive, timeout=timeout_sec)
    except Exception:
        try:
            import psutil
            psutil.Process(pid).kill()
        except Exception:
            pass


def run_subprocess_step(argv: list[str], run_dir: Path, timeout_sec: int,
                         heartbeat_sec: int = 60, cwd: Path = REPO_ROOT, env: dict | None = None) -> dict:
    """Runs argv to completion with a heartbeat file + hard timeout.

    Returns {"status": "completed"|"failed"|"timeout", "returncode": int|None,
    "duration_sec": float, "log_path": str}. Never raises for a normal
    failure/timeout -- callers decide retry policy.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "step.log"
    heartbeat_path = run_dir / "step.heartbeat"
    full_env = os.environ.copy()
    if env:
        full_env.update({k: str(v) for k, v in env.items()})
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"[cmd] {' '.join(argv)}\n[cwd] {cwd}\n[started] {now_ts()}\n\n")
        logf.flush()
        try:
            proc = subprocess.Popen(
                argv, cwd=str(cwd), stdout=logf, stderr=subprocess.STDOUT, env=full_env,
                creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
            )
        except Exception as exc:
            duration = time.time() - t0
            logf.write(f"\n[popen_error] {type(exc).__name__}: {exc}\n")
            logf.flush()
            return {"status": "failed", "returncode": None, "duration_sec": round(duration, 1),
                    "log_path": str(log_path), "popen_error": f"{type(exc).__name__}:{exc}"}
        last_size = -1
        status = "completed"
        while True:
            try:
                rc = proc.wait(timeout=heartbeat_sec)
                break
            except subprocess.TimeoutExpired:
                elapsed = time.time() - t0
                cur_size = log_path.stat().st_size
                heartbeat_path.write_text(json.dumps(
                    {"pid": proc.pid, "elapsed_sec": round(elapsed, 1),
                     "log_bytes": cur_size, "progressing": cur_size != last_size}), encoding="utf-8")
                last_size = cur_size
                if elapsed > timeout_sec:
                    terminate_process_tree(proc.pid)
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        pass
                    status = "timeout"
                    rc = proc.returncode
                    break
        duration = time.time() - t0
    heartbeat_path.unlink(missing_ok=True)
    if status == "completed" and rc != 0:
        status = "failed"
    return {"status": status, "returncode": rc, "duration_sec": round(duration, 1),
            "log_path": str(log_path), "pid": proc.pid}


# ===================== preflight =====================
def _sealed_paths(cfg: dict) -> set[str]:
    return {
        canonical_path(value)
        for key, value in cfg.get("sealed_test_pools", {}).items()
        if not str(key).startswith("_") and isinstance(value, str)
    }


def _value_hits_sealed(value: Any, sealed: set[str]) -> bool:
    if not isinstance(value, (str, Path)):
        return False
    raw = str(value).strip().strip("\"'")
    candidates = [raw]
    if "=" in raw:
        candidates.append(raw.split("=", 1)[1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            if canonical_path(candidate) in sealed:
                return True
        except (OSError, ValueError):
            continue
    return False


def sealed_reference_detail(step: dict, cfg: dict) -> dict | None:
    """Return the first sealed reference without opening the sealed manifest."""
    sealed = _sealed_paths(cfg)
    for field in ("manifest", "pool"):
        if _value_hits_sealed(step.get(field), sealed):
            return {"source": field, "value": str(step.get(field))}
    for index, token in enumerate(step.get("command", [])):
        if _value_hits_sealed(token, sealed):
            return {"source": f"command[{index}]", "value": str(token)}
    for key, value in step.get("env", {}).items():
        if _value_hits_sealed(value, sealed):
            return {"source": f"env.{key}", "value": str(value)}
    for key, value in step.get("rule_c", {}).items():
        if isinstance(value, (str, Path)) and _value_hits_sealed(value, sealed):
            return {"source": f"rule_c.{key}", "value": str(value)}

    manifest = step.get("manifest")
    if manifest and Path(manifest).exists():
        try:
            payload = read_json(manifest)
        except Exception:
            payload = {}
        if isinstance(payload, dict) and _value_hits_sealed(payload.get("root"), sealed):
            return {"source": "manifest.root", "value": str(payload.get("root"))}
    return None


def preflight(step: dict, cfg: dict) -> tuple[bool, list[str], dict]:
    """All guards a step must clear before it is ever dispatched.

    Checks (any failure blocks the step, none of them raise):
      - disk free space (D/E, from cfg["safety"]["min_free_gb"])
      - GPU single-concurrency + free-mem, only if step.get("needs_gpu")
      - pool/manifest path is under an allowlisted root
      - checkpoint type is compatible with the declared predictor
        (scripts/campaign_checkpoint.py) -- this is what stops today's
        SystemExit incident from ever being dispatched again
    """
    blockers: list[str] = []
    detail: dict = {}

    safety = cfg.get("safety", {})
    # Do this before Path.exists(), hashing, or manifest parsing.  A sealed test
    # manifest is intentionally opaque to this executor until an explicit final
    # scoring configuration is supplied.
    sealed_detail = sealed_reference_detail(step, cfg)
    if sealed_detail:
        return False, ["sealed_test_reference"], {"sealed_test": "fail_closed", **sealed_detail}
    action = step.get("decision_required_action")
    if action in DECISION_REQUIRED_ACTIONS:
        panel_ok, panel_reason = validate_panel_dispatch(
            step.get("panel_r3_artifact"), step.get("evidence_packet_sha"),
            step.get("dispatch_item", {}), action, sha256_file(cfg.get("_config_path", DEFAULT_CONFIG)),cfg,cfg.get("_config_path",DEFAULT_CONFIG))
        detail["panel"] = {
            "action": action, "ok": panel_ok, "reason": panel_reason,
            "panel_id": None,
        }
        if not panel_ok:
            blockers.append("panel_r3_required")
    disk_ok, disk_detail = check_disk_guard(safety.get("min_free_gb", {"D": 200, "E": 500}))
    detail["disk"] = disk_detail
    if not disk_ok:
        blockers.append("disk_low")

    if step.get("needs_gpu"):
        configured_fraction = safety.get("gpu_memory_fraction")
        env_fraction = step.get("env", {}).get("REPRO_GPU_MEMORY_FRACTION")
        try:
            fraction_ok = configured_fraction is not None and env_fraction is not None and \
                float(env_fraction) == float(configured_fraction) and 0.0 < float(env_fraction) <= 1.0
        except (TypeError, ValueError):
            fraction_ok = False
        detail["gpu_memory_fraction"] = {
            "configured": configured_fraction, "child_env": env_fraction, "supported": fraction_ok}
        if not fraction_ok:
            blockers.append("gpu_memory_fraction_unsupported")
        hook_ok = trainer_gpu_memory_hook_contract_ok(step)
        detail["gpu_memory_hook"] = {"entrypoint": "_may_ablation.py B4", "supported": hook_ok}
        if not hook_ok:
            blockers.append("gpu_memory_hook_missing_or_late")
        gpu_ok, gpu_detail = check_gpu_guard(safety)
        detail["gpu"] = gpu_detail
        if not gpu_ok:
            blockers.append("gpu_busy")

    pool = step.get("pool") or step.get("manifest")
    if pool:
        allowlist = safety.get("allowlist_roots", [])
        pool_for_check = pool
        if str(pool).lower().endswith(".json") and Path(pool).exists():
            try:
                manifest_root = read_json(pool).get("root")
                if manifest_root:
                    from scripts._common import resolve_path  # local import: avoid hard dep at module load
                    pool_for_check = resolve_path(manifest_root)
            except Exception:
                pass
        detail["pool_allowlisted"] = check_allowlist(pool_for_check, allowlist)
        if not detail["pool_allowlisted"]:
            blockers.append("pool_not_allowlisted")
    else:
        detail["pool_allowlisted"] = None

    prohibited = ("cca", "my-lot", "mylot", "nolocal", "no_local", "alias", "copy")
    declared_paths = [str(step.get(key, "")) for key in ("pool", "manifest", "backbone", "checkpoint")]
    if any(token in value.casefold() for value in declared_paths for token in prohibited):
        blockers.append("prohibited_data_or_alias_reference")

    if step.get("checkpoint") and step.get("predictor"):
        ckpt_path = step["checkpoint"] if isinstance(step["checkpoint"], str) else step["checkpoint"][0]
        ck_info = classify_checkpoint(ckpt_path)
        detail["checkpoint_type"] = ck_info
        ok, msg = check_predictor_compatibility(step["predictor"], ck_info,
                                                  has_backbone_arg=bool(step.get("backbone")))
        detail["checkpoint_predictor_compat"] = {"ok": ok, "message": msg}
        if not ok:
            blockers.append("checkpoint_predictor_incompatible")

    manifest = step.get("manifest")
    if manifest and not Path(manifest).exists():
        blockers.append("manifest_missing")
        detail["manifest_missing"] = str(manifest)

    return (len(blockers) == 0), blockers, detail


# ===================== campaign orchestrator =====================
def _ckpt_entries(step: dict) -> list[dict]:
    """Normalize step['checkpoint'] (str or list[str]) into [{"path":..., "sha256":...}]."""
    raw = step.get("checkpoint")
    if not raw:
        return []
    paths = raw if isinstance(raw, list) else [raw]
    out = []
    for p in paths:
        p = Path(p)
        out.append({"path": str(p), "sha256": sha256_file(p) if p.exists() else None})
    return out


def run_rule_c_evaluations(step: dict, trainer_output: Path, attempt_dir: Path,
                           python_exe: str, env: dict, heartbeat_sec: int,
                           safety: dict | None = None) -> tuple[list[dict], str | None]:
    """Run label-free selection first, then a separate offline-only process."""
    cfg = step.get("rule_c")
    if not cfg:
        return [], None
    checkpoints = sorted((trainer_output / "checkpoints").glob("proj_ep*.pt"))
    if not checkpoints:
        return [], "rule_c_no_proj_checkpoints"
    output_dir = attempt_dir / "rule_c"
    if len(checkpoints) != 20 or {p.name for p in checkpoints} != {f"proj_ep{i}.pt" for i in range(1, 21)}:
        return [], "rule_c_requires_exact_proj_ep1_to_ep20"
    if str(cfg.get("device", "cpu")).casefold() == "cuda":
        if STOP_FLAG.exists():
            return [], "rule_c_selector_refused_STOP_REQUESTED"
        if not isinstance(safety, dict):
            return [], "rule_c_selector_gpu_safety_missing"
        configured_fraction = safety.get("gpu_memory_fraction")
        try:
            fraction_ok = (
                configured_fraction is not None
                and float(env.get("REPRO_GPU_MEMORY_FRACTION")) == float(configured_fraction)
                and 0.0 < float(configured_fraction) <= 0.40
            )
        except (TypeError, ValueError):
            fraction_ok = False
        if not fraction_ok:
            return [], "rule_c_selector_gpu_fraction_invalid"
        if not rule_c_selector_gpu_memory_hook_contract_ok():
            return [], "rule_c_selector_gpu_memory_hook_missing_or_late"
        gpu_ok, _ = check_gpu_guard(safety)
        if not gpu_ok:
            return [], "rule_c_selector_gpu_busy"
    r3_binding = cfg.get("v3_r3_binding")
    if not isinstance(r3_binding, str) or not r3_binding:
        return [], "rule_c_v3_r3_binding_missing"
    selector = [python_exe, str(REPO_ROOT / "scripts" / "run_rule_c_selector.py"), "--pool", str(cfg["unlabeled_pool"]), "--backbone", str(cfg["backbone"]),
                "--proj-dir", str(trainer_output / "checkpoints"), "--out-dir", str(output_dir), "--ratios",
                *[str(x) for x in cfg.get("ratios", [])], "--z0-seeds", *[str(x) for x in cfg.get("z0_seeds", [])],
                "--device", str(cfg.get("device", "cpu")), "--batch-size", str(cfg.get("batch_size", 16))]
    selected = run_subprocess_step(selector, attempt_dir / "rule_c_selector", int(cfg.get("timeout_sec", 21600)), heartbeat_sec, env=env)
    snapshot = output_dir / "selection_snapshot.json"
    def valid_v2_snapshot() -> bool:
        try:
            payload = read_json(snapshot)
            bundle = Path(payload["bundle_path"])
            if (payload.get("schema_version") != "rule_c_label_free_selection.v2"
                    or payload.get("offline_labels_evaluated") is not False
                    or not bundle.is_file() or sha256_file(bundle) != payload.get("bundle_sha256")):
                return False
            cache = read_json(bundle)
            npz = Path(cache["npz_path"])
            return npz.is_file() and sha256_file(npz) == cache.get("npz_sha256")
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            return False
    # A V2 no-consensus exit is diagnostic only: its immutable snapshot and
    # bound cache remain the required input to the deterministic V3 repair.
    if not snapshot.is_file() or (selected["status"] != "completed" and not valid_v2_snapshot()):
        return [], "rule_c_selector_failed"
    if not valid_v2_snapshot():
        return [], "rule_c_v2_snapshot_or_bundle_invalid"
    snapshot_sha = sha256_file(snapshot)
    v3_snapshot = output_dir / "selection_snapshot_v3.json"
    seal_path = output_dir / "epoch_seal_v3.json"
    cpu_env = {**env, "CUDA_VISIBLE_DEVICES": ""}
    cpu_env.pop("REPRO_GPU_MEMORY_FRACTION", None)
    reselector = [python_exe, str(REPO_ROOT / "scripts" / "run_rule_c_v3_reselector.py"),
                  "--source-v2-snapshot", str(snapshot), "--r3-binding", str(r3_binding),
                  "--checkpoint-dir", str(trainer_output / "checkpoints"), "--out", str(v3_snapshot),
                  "--seal-out", str(seal_path)]
    v3 = run_subprocess_step(reselector, attempt_dir / "rule_c_v3_reselector",
                             int(cfg.get("timeout_sec", 21600)), heartbeat_sec, env=cpu_env)
    if v3["status"] != "completed" or not v3_snapshot.is_file() or not seal_path.is_file():
        return [], "rule_c_v3_reselector_failed"
    try:
        v3_payload = read_json(v3_snapshot)
        seal_payload = read_json(seal_path)
        if (v3_payload.get("schema_version") != "rule_c_label_free_selection.v3"
                or v3_payload.get("status") != "selected" or v3_payload.get("selection_valid") is not True
                or v3_payload.get("labels_used") is not False
                or seal_payload.get("schema_version") != "rule_c_epoch_seal.v3"):
            return [], "rule_c_v3_snapshot_or_seal_invalid"
    except (OSError, ValueError, json.JSONDecodeError):
        return [], "rule_c_v3_snapshot_or_seal_invalid"
    v3_snapshot_sha, seal_sha = sha256_file(v3_snapshot), sha256_file(seal_path)
    offline_path = output_dir / "offline.json"
    offline = [python_exe, str(REPO_ROOT / "scripts" / "run_rule_c_offline.py"), "--selection-snapshot", str(v3_snapshot),
               "--expected-selection-sha256", v3_snapshot_sha, "--epoch-seal", str(seal_path),
               "--expected-epoch-seal-sha256", seal_sha, "--labeled-pool", str(cfg["offline_pool"]), "--out", str(offline_path)]
    result = run_subprocess_step(
        offline, attempt_dir / "rule_c_offline", int(cfg.get("timeout_sec", 21600)),
        heartbeat_sec, env=cpu_env)
    if result["status"] != "completed" or not offline_path.is_file(): return [], "rule_c_offline_failed"
    return [{"selection_snapshot_path": str(v3_snapshot.resolve()), "selection_snapshot_sha256": v3_snapshot_sha,
             "epoch_seal_path": str(seal_path.resolve()), "epoch_seal_sha256": seal_sha,
             "source_v2_snapshot_path": str(snapshot.resolve()), "source_v2_snapshot_sha256": snapshot_sha,
             "offline_path": str(offline_path.resolve()), "offline_sha256": sha256_file(offline_path), "checkpoint_count": 20}], None


def _numeric_leaves(obj: Any, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], float]:
    out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key not in {"n_seeds", "aggregate_seed_run_ids", "three_seed_aggregate"}:
                out.update(_numeric_leaves(value, prefix + (str(key),)))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = float(obj)
    return out


def _set_nested(root: dict, path: tuple[str, ...], value: Any) -> None:
    node = root
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value


def build_three_seed_aggregate(records: list[dict]) -> tuple[dict | None, str | None]:
    """Build mean/std metrics and paired deltas from exactly three real seeds."""
    seeds = [rec.get("seed") for rec in records]
    if len(records) != 3 or len(set(seeds)) != 3:
        return None, "aggregate_requires_exactly_three_distinct_seeds"
    metric_maps = [_numeric_leaves(rec.get("metrics") or {}) for rec in records]
    common = set(metric_maps[0]).intersection(*(set(item) for item in metric_maps[1:]))
    paired_maps = [_numeric_leaves((rec.get("metrics") or {}).get("paired_deltas") or {})
                   for rec in records]
    paired_common = set(paired_maps[0]).intersection(*(set(item) for item in paired_maps[1:]))
    if not common:
        return None, "aggregate_missing_common_numeric_metrics"
    if not paired_common:
        return None, "aggregate_missing_paired_deltas"

    mean_metrics: dict = {}
    statistics_block: dict = {}
    for path in sorted(common):
        values = [item[path] for item in metric_maps]
        _set_nested(mean_metrics, path, statistics.fmean(values))
        statistics_block[".".join(path)] = {
            "values": values, "mean": statistics.fmean(values), "std": statistics.stdev(values)}
    paired_block = {}
    for path in sorted(paired_common):
        values = [item[path] for item in paired_maps]
        paired_block[".".join(path)] = {
            "values": values, "mean": statistics.fmean(values), "std": statistics.stdev(values)}
    return {
        "n_seeds": 3, "seeds": sorted(seeds),
        "run_ids": [rec.get("run_id") for rec in records],
        "mean_metrics": mean_metrics, "metric_statistics": statistics_block,
        "paired_deltas": paired_block,
    }, None


class Campaign:
    def __init__(self, config_path=DEFAULT_CONFIG, registry_path=DEFAULT_REGISTRY, python_exe=None):
        self.config_path = Path(config_path)
        self.cfg = read_json(self.config_path)
        self.registry = Registry(registry_path)
        self.python_exe = python_exe or sys.executable
        self.config_snapshot_sha256 = sha256_file(self.config_path)
        self.campaign_id = self.cfg.get("campaign_id")
        self._lock = ExclusiveFileLock(CAMPAIGN_LOCK)

    def status(self) -> str:
        lines = [f"campaign: {self.cfg.get('campaign_id')}  config={self.config_path}"]
        champ = self.registry.current_champion()
        lines.append(f"champion: tier={champ.get('tier')} run_id={champ.get('run_id')} "
                      f"checkpoint={champ.get('checkpoint_path')}" if champ else "champion: none yet")
        completed = self.registry.completed_steps()
        for tier_cfg in self.cfg.get("tiers", []):
            t = tier_cfg["tier"]
            steps = tier_cfg.get("steps", [])
            n_done = sum(1 for s in steps for seed in (s.get("seeds") or [None])
                         if (t, s["name"], seed) in completed)
            n_total = sum(len(s.get("seeds") or [None]) for s in steps)
            lines.append(f"tier {t} [{tier_cfg.get('name')}]: {n_done}/{n_total} steps completed")
        disk_ok, disk_detail = check_disk_guard(self.cfg.get("safety", {}).get("min_free_gb", {}))
        lines.append(f"disk: ok={disk_ok} {disk_detail}")
        gpu_ok, gpu_detail = check_gpu_guard(self.cfg.get("safety", {}))
        lines.append(f"gpu: available={gpu_ok} free_mb={gpu_detail.get('free_mb')} "
                      f"alive_locks={len(gpu_detail.get('alive_locks', []))}")
        lock_holder = None
        if CAMPAIGN_LOCK.exists():
            try:
                lock_holder = json.loads(CAMPAIGN_LOCK.read_text(encoding="utf-8"))
            except Exception:
                lock_holder = {"raw": "unreadable"}
        lines.append(f"campaign_lock: {lock_holder}")
        lines.append(f"stop_requested: {STOP_FLAG.exists()}")
        n_records = len(self.registry.read_all())
        lines.append(f"registry: {self.registry.path} ({n_records} records)")
        return "\n".join(lines)

    def _maybe_promote(self, tier: int, run_id: str, checkpoint_entries: list[dict], gate_passed) -> str | None:
        if not gate_passed:
            return None
        if self.registry.current_champion() is not None:
            return None
        write_json(CHAMPION_FILE, {"tier": tier, "run_id": run_id, "checkpoints": checkpoint_entries,
                                     "promoted_at": now_ts()})
        return run_id

    def _acquire_lock(self) -> None:
        try:
            self._lock.acquire()
        except RuntimeError as exc:
            raise SystemExit(f"{exc} -- run with --status to inspect it") from exc

    def _release_lock(self) -> None:
        self._lock.release()

    def _dependency_context(self, step: dict, seed) -> tuple[dict, str | None]:
        """Resolve a same-seed completed producer without guessing a checkpoint.

        A consumer can declare ``depends_on: {tier: 2, step: train_frozen_recipe}``.
        The dependency's recorded produced checkpoint must still exist; this makes
        resume deterministic and prevents evaluation from silently using another
        seed's newest checkpoint.
        """
        dep = step.get("depends_on")
        if not dep:
            return {}, None
        dep_seed = seed if dep.get("same_seed", True) else dep.get("seed")
        producer = None
        for tier_cfg in self.cfg.get("tiers", []):
            if tier_cfg.get("tier") == dep.get("tier"):
                producer = next((item for item in tier_cfg.get("steps", [])
                                 if item.get("name") == dep.get("step")), None)
                break
        if producer is None:
            return {}, "dependency_step_missing_from_config"
        rec = self.matching_final_record(dep.get("tier"), producer, dep_seed)
        checkpoint = (rec or {}).get("produced_checkpoint_path")
        if not rec or rec.get("status") != "completed":
            return {}, "dependency_not_completed"
        if not checkpoint or not Path(checkpoint).is_file():
            return {}, "dependency_checkpoint_missing"
        return {"dependency_checkpoint": str(Path(checkpoint).resolve()),
                "dependency_run_id": rec.get("run_id"),
                "dependency_seed": dep_seed}, None

    def _aggregate_seed_records(self, spec: dict) -> list[dict]:
        """Return only real, completed records for distinct requested seeds."""
        tier, name = spec.get("tier"), spec.get("step")
        try:
            _, producer = self._find_step(tier, name)
        except KeyError:
            return []
        records = [self.matching_final_record(tier, producer, seed) for seed in spec.get("seeds", [])]
        return [rec for rec in records if rec is not None]

    def _step_fingerprint(self, tier: int, step: dict, seed) -> str:
        return sha256_obj({"tier": tier, "step": step, "seed": seed})

    @staticmethod
    def _artifact_valid(rec: dict) -> bool:
        path, expected = rec.get("artifact_path"), rec.get("artifact_sha256")
        if not path or not expected or not Path(path).exists():
            return False
        return sha256_tree(Path(path)) == expected

    def matching_final_record(self, tier: int, step: dict, seed) -> dict | None:
        fingerprint = self._step_fingerprint(tier, step, seed)
        rec = self.registry.latest_matching(
            campaign_id=self.campaign_id,
            config_snapshot_sha256=self.config_snapshot_sha256,
            step_fingerprint=fingerprint,
            tier=tier, step=step["name"], seed=seed,
        )
        if rec and rec.get("status") == "completed" and self._artifact_valid(rec):
            return rec
        return None

    def _reconcile_started_orphan(self, tier: int, step: dict, seed) -> None:
        fingerprint = self._step_fingerprint(tier, step, seed)
        rec = self.registry.latest_matching(
            campaign_id=self.campaign_id,
            config_snapshot_sha256=self.config_snapshot_sha256,
            step_fingerprint=fingerprint,
            tier=tier, step=step["name"], seed=seed,
        )
        if rec and rec.get("status") == "started":
            self.registry.append({
                **rec, "ts": now_ts(), "status": "failed",
                "failure_reason": "orphaned_started_reconciled",
                "reconciled_from_attempt": rec.get("attempt"),
            })

    # ---- one atomic step: preflight -> dispatch -> retry(<=2) -> registry append ----
    def _run_one_step(self, tier_cfg: dict, step: dict, seed, dispatch_id: str | None = None) -> dict:
        tier = tier_cfg["tier"]
        max_retries = int(self.cfg.get("safety", {}).get("max_retries", 2))
        run_tag = f"t{tier}_{step['name']}" + (f"_seed{seed}" if seed is not None else "")
        run_id = f"{run_tag}_{now_ts()}_{uuid.uuid4().hex[:8]}"
        run_root = REPO_ROOT / "runs" / "campaign" / run_id

        if STOP_FLAG.exists():
            return {"status": "blocked", "failure_reason": "STOP_REQUESTED", "gate": {"passed": None}}
        recipe = dict(step.get("recipe", {}))
        if seed is not None:
            recipe["seed"] = seed
        recipe_sha = sha256_obj(recipe) if recipe else None
        manifest = step.get("manifest")
        is_sealed = sealed_reference_detail(step, self.cfg) is not None
        dataset_sha = sha256_file(manifest) if manifest and not is_sealed and Path(manifest).exists() else None
        ckpt_entries = _ckpt_entries(step)

        dependency_context, dependency_error = self._dependency_context(step, seed)
        step_fingerprint = self._step_fingerprint(tier, step, seed)
        panel_path = step.get("panel_r3_artifact")
        queue_index = step.get("approved_queue_index")

        base = {"campaign_id": self.campaign_id, "config_snapshot_sha256": self.config_snapshot_sha256,
                "step_fingerprint": step_fingerprint, "dispatch_id": dispatch_id,
                "run_id": run_id, "tier": tier, "tier_name": tier_cfg.get("name"), "step": step["name"],
                "seed": seed, "dataset_sha256": dataset_sha, "manifest_path": str(manifest) if manifest else None,
                "recipe": recipe or None, "recipe_sha256": recipe_sha,
                 "checkpoint_path": step.get("checkpoint"), "checkpoint_sha256": ckpt_entries or None,
                "evidence_packet_sha": step.get("evidence_packet_sha"),
                "panel_r3_artifact_sha256": sha256_file(panel_path) if panel_path and Path(panel_path).is_file() else None,
                "approved_queue_index": queue_index,
                "predictor": step.get("predictor"),
                "dependency": dependency_context or step.get("depends_on")}

        attempt = 0
        while True:
            attempt += 1
            attempt_dir = run_root / f"attempt_{attempt:02d}"
            champion_before = (self.registry.current_champion() or {}).get("run_id")
            if dependency_error:
                rec = {**base, "status": "blocked", "failure_reason": dependency_error,
                       "retry_count": attempt - 1, "attempt": attempt,
                       "attempt_dir": str(attempt_dir),
                       "champion_before": champion_before, "champion_after": None,
                       "metrics": None, "gate": {"evaluated": False, "passed": None, "reasons": None}}
                self.registry.append(rec)
                return rec
            try:
                ok, blockers, pf_detail = preflight(step, self.cfg)
            except Exception as e:  # a broken step config must never crash the whole campaign
                rec = {**base, "status": "blocked", "failure_reason": f"preflight_error:{e}",
                       "retry_count": attempt - 1, "attempt": attempt,
                       "attempt_dir": str(attempt_dir),
                       "champion_before": champion_before, "champion_after": None,
                       "metrics": None, "gate": {"evaluated": False, "passed": None, "reasons": None}}
                self.registry.append(rec)
                return rec
            if not ok:
                rec = {**base, "status": "blocked", "failure_reason": ",".join(blockers),
                       "retry_count": attempt - 1, "attempt": attempt, "preflight": pf_detail,
                       "attempt_dir": str(attempt_dir),
                       "champion_before": champion_before, "champion_after": None,
                       "metrics": None, "gate": {"evaluated": False, "passed": None, "reasons": None}}
                self.registry.append(rec)
                return rec

            try:
                context = {"python": self.python_exe, "run_dir": str(attempt_dir), "seed": seed,
                           "manifest": manifest, **dependency_context, **step.get("context", {})}
                argv = build_command(step, context)
                env = build_env(step, context)
            except ValueError as e:
                rec = {**base, "status": "blocked", "failure_reason": f"command_build_error:{e}",
                       "retry_count": attempt - 1, "attempt": attempt,
                       "attempt_dir": str(attempt_dir),
                       "champion_before": champion_before, "champion_after": None,
                       "metrics": None, "gate": {"evaluated": False, "passed": None, "reasons": None}}
                self.registry.append(rec)
                return rec

            timeout_sec = int(step.get("timeout_sec", self.cfg.get("safety", {}).get("run_timeout_sec", 43200)))
            heartbeat_sec = int(self.cfg.get("safety", {}).get("heartbeat_sec", 60))
            step_start_ts = time.time()
            self.registry.append({**base, "status": "started", "attempt": attempt,
                                  "retry_count": attempt - 1, "started_at": now_ts(),
                                  "attempt_dir": str(attempt_dir), "argv": argv,
                                  "champion_before": champion_before, "champion_after": None,
                                  "metrics": None, "gate": {"evaluated": False, "passed": None, "reasons": None}})
            result = run_subprocess_step(argv, attempt_dir, timeout_sec, heartbeat_sec, env=env)

            resolved_output_dir, produced_checkpoint_path, resolve_error = None, None, None
            rule_c_artifacts: list[dict] = []
            resolve_cell = step.get("resolve_output_glob_cell")
            if resolve_cell and result["status"] == "completed":
                try:
                    resolved = resolve_trainer_output_dir(REPO_ROOT / "runs", env.get("REPRO_TAG", ""),
                                                           resolve_cell, step_start_ts)
                    resolved_output_dir = str(resolved)
                    epochs = recipe.get("epochs")
                    if epochs:
                        produced_checkpoint_path = str(resolved / "checkpoints" / f"proj_ep{epochs}.pt")
                        if not Path(produced_checkpoint_path).is_file():
                            raise RuntimeError(f"trainer did not produce expected checkpoint: {produced_checkpoint_path}")
                    rule_c_artifacts, rule_c_error = run_rule_c_evaluations(
                        step, resolved, attempt_dir, self.python_exe, env, heartbeat_sec,
                        safety=self.cfg.get("safety", {}))
                    if rule_c_error:
                        raise RuntimeError(rule_c_error)
                except RuntimeError as e:
                    resolve_error = str(e)
                    result = {**result, "status": "failed"}

            metrics = None
            metrics_path = step.get("metrics_path")
            if metrics_path:
                mp = Path(str(metrics_path).format(run_dir=attempt_dir))
                if mp.exists():
                    metrics = read_json(mp)
            if metrics is None:
                metrics = extract_metrics_from_run(attempt_dir)
                metrics.update(build_gate_metrics(step, self.cfg, attempt_dir))
            if rule_c_artifacts:
                metrics["_rule_c"] = rule_c_artifacts

            gate_passed, gate_reasons = None, None
            if step.get("evaluate_gate", False):
                gate_cfg = tier_cfg.get("gates", self.cfg.get("gates", {}))
                # A per-seed record is evidence only.  Promotion requires an
                # explicit aggregate step carrying three distinct seed records.
                required = tier_cfg.get("n_seeds_min")
                aggregate = step.get("aggregate_seeds_from")
                if aggregate:
                    records = self._aggregate_seed_records(aggregate)
                    aggregate_payload, aggregate_error = build_three_seed_aggregate(records)
                    if aggregate_error:
                        gate_passed, gate_reasons = False, [aggregate_error]
                    else:
                        metrics = {**aggregate_payload["mean_metrics"],
                                   "n_seeds": 3, "three_seed_aggregate": aggregate_payload}
                        gate_passed, gate_reasons = evaluate_gate(gate_cfg, metrics, required)
                elif required and step.get("seeds"):
                    gate_passed, gate_reasons = False, ["per_seed_gate_forbidden: aggregate distinct seeds first"]
                else:
                    gate_passed, gate_reasons = evaluate_gate(gate_cfg, metrics, required)

            failure_reason = None if result["status"] == "completed" else \
                (resolve_error or result.get("popen_error") or
                 f"{result['status']}:returncode={result.get('returncode')}")

            artifact_path, artifact_sha = None, None
            if result["status"] == "completed":
                configured_artifact = step.get("artifact_path")
                if configured_artifact:
                    artifact = Path(str(configured_artifact).format(run_dir=attempt_dir, **dependency_context))
                elif resolved_output_dir:
                    artifact = Path(resolved_output_dir) / "checkpoints"
                else:
                    artifact = attempt_dir
                if artifact.exists():
                    artifact_path, artifact_sha = str(artifact.resolve()), sha256_tree(artifact)
                else:
                    result = {**result, "status": "failed"}
                    failure_reason = f"artifact_missing:{artifact}"

            champion_after = None
            if result["status"] == "completed" and artifact_sha and gate_passed:
                champion_after = self._maybe_promote(tier, run_id, ckpt_entries, gate_passed)

            rec = {**base, "metrics": metrics, "resolved_output_dir": resolved_output_dir,
                   "gate": {"evaluated": step.get("evaluate_gate", False), "passed": gate_passed,
                             "reasons": gate_reasons},
                   "status": result["status"], "failure_reason": failure_reason,
                   "retry_count": attempt - 1, "attempt": attempt, "duration_sec": result["duration_sec"],
                   "attempt_dir": str(attempt_dir),
                   "log_path": result["log_path"], "produced_checkpoint_path": produced_checkpoint_path,
                   "artifact_path": artifact_path, "artifact_sha256": artifact_sha,
                   "rule_c_artifacts": rule_c_artifacts,
                   "champion_before": champion_before,
                   "champion_after": champion_after}
            self.registry.append(rec)

            if result["status"] == "completed" or attempt > max_retries:
                return rec
            time.sleep(min(30 * attempt, 120))  # brief backoff before retry

    def _find_step(self, tier: int, step_name: str) -> tuple[dict, dict]:
        for tier_cfg in self.cfg.get("tiers", []):
            if int(tier_cfg.get("tier")) != int(tier):
                continue
            for step in tier_cfg.get("steps", []):
                if step.get("name") == step_name:
                    return tier_cfg, step
        raise KeyError(f"unknown campaign step tier={tier} step={step_name}")

    def run_selected(self, tier: int, step_name: str, seed, dispatch_id: str,
                     panel_r3_artifact: str | None = None,
                     evidence_packet_sha: str | None = None) -> dict:
        """Dispatch exactly one concrete configured (tier, step, seed)."""
        if STOP_FLAG.exists():
            raise ValueError("STOP_REQUESTED: selected dispatch refused")
        tier_cfg, configured_step = self._find_step(tier, step_name)
        step = dict(configured_step)
        if panel_r3_artifact:
            step["panel_r3_artifact"] = panel_r3_artifact
            step["evidence_packet_sha"] = evidence_packet_sha
        queue_id = next((q.get("id") for q in (self.cfg.get("outer_loop", {}).get("initial_queue", {}).get("cells", []) +
                                                self.cfg.get("outer_loop", {}).get("post_gate_cells", []))
                         if (q.get("tier"), q.get("step"), q.get("seed")) == (tier, step_name, seed)), None)
        step["dispatch_item"] = {"id": queue_id, "tier": tier, "step": step_name, "seed": seed}
        self.cfg["_config_path"] = str(self.config_path)
        if not panel_r3_artifact or not evidence_packet_sha:
            raise ValueError("selected dispatch requires a complete panel bundle")
        evidence = read_json(Path(panel_r3_artifact).parent / "evidence_packet.json")
        approved_queue = evidence.get("proposed_queue", [])
        current_index = next((i for i, q in enumerate(approved_queue)
                              if q == step["dispatch_item"]), None)
        if current_index is None:
            raise ValueError("selected dispatch item is not in approved ordered queue")
        step["approved_queue_index"] = current_index
        panel_sha = sha256_file(Path(panel_r3_artifact))
        records = self.registry.read_all()
        missing_prior = []
        for prior_index, prior in enumerate(approved_queue[:current_index]):
            identity = (prior.get("tier"), prior.get("step"), prior.get("seed"))
            valid_prior = any(
                (record.get("tier"), record.get("step"), record.get("seed")) == identity
                and record.get("status") == "completed"
                and record.get("evidence_packet_sha") == evidence_packet_sha
                and record.get("config_snapshot_sha256") == self.config_snapshot_sha256
                and record.get("panel_r3_artifact_sha256") == panel_sha
                and record.get("approved_queue_index") == prior_index
                and self._artifact_valid(record)
                for record in records
            )
            if not valid_prior:
                missing_prior.append(prior)
        if missing_prior:
            raise ValueError("selected dispatch violates approved queue order")
        declared = step.get("seeds") or [None]
        if seed not in declared:
            raise ValueError(f"seed {seed!r} is not declared for tier={tier} step={step_name}")
        self._acquire_lock()
        try:
            self._reconcile_started_orphan(tier, step, seed)
            prior = self.matching_final_record(tier, step, seed)
            if prior:
                rec = {**prior, "ts": now_ts(), "dispatch_id": dispatch_id,
                       "reused_from_run_id": prior.get("run_id")}
                self.registry.append(rec)
                return rec
            return self._run_one_step(tier_cfg, step, seed, dispatch_id=dispatch_id)
        finally:
            self._release_lock()

    # ---- full ladder loop ----
    def run(self, resume: bool, max_tier: int | None, stop_after_current: bool) -> None:
        raise SystemExit("full campaign CLI is disabled; dispatch only through the panel-bound supervisor")
        if not resume and self.registry.read_all():
            raise SystemExit(f"registry {self.registry.path} already has entries -- pass --resume to "
                              f"continue this campaign (or point at a fresh config/registry).")
        self._acquire_lock()
        try:
            for tier_cfg in self.cfg.get("tiers", []):
                tier = tier_cfg["tier"]
                if max_tier is not None and tier > max_tier:
                    print(f"[campaign] stopping before tier {tier} (--max-tier {max_tier})")
                    break
                for step in tier_cfg.get("steps", []):
                    for seed in (step.get("seeds") or [None]):
                        self._reconcile_started_orphan(tier, step, seed)
                        if self.matching_final_record(tier, step, seed):
                            print(f"[campaign] skip tier={tier} step={step['name']} seed={seed} (completed)")
                            continue
                        print(f"[campaign] tier={tier} step={step['name']} seed={seed} -> dispatching")
                        rec = self._run_one_step(tier_cfg, step, seed)
                        print(f"[campaign] tier={tier} step={step['name']} seed={seed} -> "
                              f"status={rec['status']} gate_passed={rec['gate']['passed']}")
                        if STOP_FLAG.exists():
                            print(f"[campaign] STOP_REQUESTED honored after tier={tier} step={step['name']}")
                            return
                        if stop_after_current:
                            print("[campaign] --stop-after-current: exiting after this one step")
                            return
        finally:
            self._release_lock()


# ===================== CLI =====================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="campaign config JSON")
    ap.add_argument("--resume", action="store_true",
                     help="continue an existing registry (required if runs/campaign_registry.jsonl "
                          "already has entries for this config)")
    ap.add_argument("--status", action="store_true",
                     help="print campaign state (champion, per-tier progress, disk/GPU, lock) and exit; "
                          "never dispatches anything")
    ap.add_argument("--stop-after-current", action="store_true",
                     help="if another runner instance holds the lock, just signal it to stop after its "
                          "current step and exit. If no instance is running, run exactly one more step "
                          "(via --resume) then exit.")
    ap.add_argument("--max-tier", type=int, default=None, help="cap the ladder at this tier number")
    ap.add_argument("--dispatch-tier", type=int, default=None,
                    help="with --dispatch-step/--dispatch-seed/--dispatch-id, execute exactly one item")
    ap.add_argument("--dispatch-step", default=None)
    ap.add_argument("--dispatch-seed", type=int, default=None)
    ap.add_argument("--dispatch-id", default=None)
    ap.add_argument("--panel-r3-artifact", default=None)
    ap.add_argument("--evidence-packet-sha", default=None)
    a = ap.parse_args(argv)

    campaign = Campaign(config_path=a.config)

    if a.status:
        print(campaign.status())
        return 0

    selected = [a.dispatch_tier is not None, a.dispatch_step is not None,
                a.dispatch_seed is not None, a.dispatch_id is not None]
    if any(selected) and not all(selected):
        ap.error("selected dispatch requires --dispatch-tier, --dispatch-step, --dispatch-seed, and --dispatch-id")
    if all(selected):
        try:
            rec = campaign.run_selected(
                a.dispatch_tier, a.dispatch_step, a.dispatch_seed, a.dispatch_id,
                panel_r3_artifact=a.panel_r3_artifact,
                evidence_packet_sha=a.evidence_packet_sha)
        except (KeyError, ValueError) as exc:
            print(f"[campaign] selected dispatch rejected: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(rec, ensure_ascii=False, default=str))
        return 0 if rec.get("status") == "completed" and campaign._artifact_valid(rec) else 2

    if a.stop_after_current and CAMPAIGN_LOCK.exists():
        try:
            info = json.loads(CAMPAIGN_LOCK.read_text(encoding="utf-8"))
            import psutil
            if info.get("pid") and psutil.pid_exists(int(info["pid"])):
                STATE_DIR.mkdir(parents=True, exist_ok=True)
                STOP_FLAG.write_text(json.dumps({"requested_at": now_ts(), "by_pid": os.getpid()}),
                                       encoding="utf-8")
                print(f"[campaign] STOP_REQUESTED written -- active runner (pid={info['pid']}) will "
                      f"exit after its current step. This process is not starting a second runner.")
                return 0
        except (json.JSONDecodeError, KeyError, ValueError):
            pass  # stale/unreadable lock -- fall through to a normal (single-step) run below

    campaign.run(resume=a.resume, max_tier=a.max_tier, stop_after_current=a.stop_after_current)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
