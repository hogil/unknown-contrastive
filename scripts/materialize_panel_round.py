#!/usr/bin/env python3
"""Prepare an immutable panel prompt, then finalize its real agent response."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from scripts.run_unknown_campaign import PANEL_CONTRACT, sha256_file, sha256_obj

JUDGES = tuple(PANEL_CONTRACT)
ROUNDS = ("R1", "R2", "R3")
R3_PRESERVED_FIELDS = ("conclusion", "critical_objections", "minority_positions", "near_audit_adjudications")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError(f"JSON object required: {path}")
    return value


def _bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8")


def _atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_bytes(_bytes(value)); os.replace(temp, path)
    finally:
        if temp.exists(): temp.unlink()


def _iso(value: str, label: str) -> datetime:
    try: return datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc: raise ValueError(f"{label} must be ISO-8601") from exc


def _evidence(source: str | Path, panel_dir: Path) -> tuple[Path, str, str, str]:
    source = Path(source).resolve(); value = _read(source); binding = value.get("binding")
    if not isinstance(binding, dict): raise ValueError("evidence binding object is required")
    target = panel_dir / "evidence_packet.json"; canonical_sha = sha256_obj(value)
    if target.exists():
        if sha256_obj(_read(target)) != canonical_sha: raise ValueError("output evidence differs from supplied evidence")
    else: _atomic(target, value)
    return target.resolve(), canonical_sha, sha256_obj(binding), sha256_file(target)


def _peer_bindings(panel_dir: Path, prior: str, evidence_sha: str, evidence_file_sha: str,
                   binding_sha: str) -> tuple[dict[str, str], dict[str, dict[str, str]], datetime]:
    response_shas, bindings, completions = {}, {}, []
    for judge in JUDGES:
        artifact_path = (panel_dir / f"{prior.lower()}_{judge}.json").resolve()
        if not artifact_path.is_file(): raise ValueError(f"missing required {prior} peer artifact for {judge}")
        artifact = _read(artifact_path); receipt_path = Path(str(artifact.get("receipt_path", "")))
        if (artifact.get("judge"), artifact.get("round"), artifact.get("evidence_packet_sha"),
                artifact.get("evidence_packet_file_sha256"), artifact.get("binding_sha256")) != (
                    judge, prior, evidence_sha, evidence_file_sha, binding_sha):
            raise ValueError(f"invalid {prior} peer artifact for {judge}")
        response_sha = artifact.get("response_sha")
        if not isinstance(response_sha, str) or response_sha != sha256_obj(artifact.get("response")):
            raise ValueError(f"invalid {prior} response hash for {judge}")
        if not receipt_path.is_file(): raise ValueError(f"missing {prior} receipt for {judge}")
        receipt = _read(receipt_path)
        response_shas[judge] = response_sha
        bindings[judge] = {"artifact_path": str(artifact_path), "artifact_sha256": sha256_file(artifact_path), "response_sha": response_sha}
        completions.append(_iso(receipt["completed_at"], f"{prior} completed_at"))
    if len(set(response_shas.values())) != 3: raise ValueError(f"{prior} peer responses must be distinct")
    return response_shas, bindings, max(completions)


def prepare_panel_prompt(*, evidence_packet: str | Path, panel_dir: str | Path, task_file: str | Path,
                         judge: str, round_name: str, model: str, reasoning_effort: str,
                         prepared_at: str) -> Path:
    """Write the real invocation prompt before spawning an agent; never rewrite it."""
    if judge not in JUDGES or round_name not in ROUNDS or (round_name == "R3" and judge != "C"):
        raise ValueError("invalid judge/round")
    if (model, reasoning_effort) != PANEL_CONTRACT[judge]: raise ValueError("model/reasoning_effort does not match panel contract")
    _iso(prepared_at, "prepared_at")
    panel_dir = Path(panel_dir).resolve(); task_path = Path(task_file).resolve()
    if not task_path.is_file(): raise ValueError("task_file missing")
    task_bytes = task_path.read_bytes()
    evidence_path, evidence_sha, binding_sha, evidence_file_sha = _evidence(evidence_packet, panel_dir)
    prompt: dict[str, Any] = {"panel_id": f"unknown-{evidence_sha[:16]}", "judge": judge, "round": round_name,
        "model": model, "reasoning_effort": reasoning_effort, "evidence_packet_path": str(evidence_path),
        "evidence_packet_sha": evidence_sha, "evidence_packet_file_sha256": evidence_file_sha,
        "binding_sha256": binding_sha, "task_file_path": str(task_path),
        "task_file_sha256": hashlib.sha256(task_bytes).hexdigest(), "task_instructions": task_bytes.decode("utf-8"), "prepared_at": prepared_at}
    if round_name in ("R2", "R3"):
        prior = "R1" if round_name == "R2" else "R2"
        shas, peers, _ = _peer_bindings(panel_dir, prior, evidence_sha, evidence_file_sha, binding_sha)
        prompt[f"{prior.lower()}_response_shas"] = shas
        prompt["peer_artifacts"] = peers
    path = (panel_dir / f"prompt_{round_name}_{judge}.json").resolve(); expected = _bytes(prompt)
    if path.exists():
        if path.read_bytes() != expected: raise ValueError("existing prompt is not byte-identical to requested prompt")
    else: _atomic(path, prompt)
    return path


def finalize_panel_round(*, panel_dir: str | Path, prompt_path: str | Path, prompt_sha256: str,
                         agent_id: str, task_name: str, model: str, reasoning_effort: str,
                         started_at: str, completed_at: str, timestamp: str, raw_response_path: str | Path) -> Path:
    """Wrap a real response using an existing immutable prompt; it is never rewritten."""
    panel_dir = Path(panel_dir).resolve(); prompt_path = Path(prompt_path).resolve()
    if not prompt_path.is_file() or sha256_file(prompt_path) != prompt_sha256: raise ValueError("prompt path/SHA mismatch")
    prompt = _read(prompt_path); judge, round_name = prompt.get("judge"), prompt.get("round")
    if judge not in JUDGES or round_name not in ROUNDS: raise ValueError("prompt identity invalid")
    if (model, reasoning_effort) != PANEL_CONTRACT[judge] or (prompt.get("model"), prompt.get("reasoning_effort")) != (model, reasoning_effort):
        raise ValueError("prompt or supplied model/effort violates panel contract")
    if not agent_id or not task_name: raise ValueError("persistent agent_id and task_name are required")
    started, completed = _iso(started_at, "started_at"), _iso(completed_at, "completed_at"); _iso(timestamp, "timestamp")
    if completed < started or started <= _iso(prompt.get("prepared_at"), "prepared_at"): raise ValueError("agent times invalid relative to prompt")
    evidence_path = Path(str(prompt.get("evidence_packet_path", "")))
    if not evidence_path.is_file(): raise ValueError("prompt evidence path missing")
    evidence_file_sha = sha256_file(evidence_path)
    if prompt.get("evidence_packet_file_sha256") != evidence_file_sha:
        raise ValueError("prompt evidence file SHA invalid")
    evidence = _read(evidence_path); evidence_sha, binding_sha = sha256_obj(evidence), sha256_obj(evidence.get("binding"))
    if (prompt.get("evidence_packet_sha"), prompt.get("binding_sha256")) != (evidence_sha, binding_sha): raise ValueError("prompt evidence binding invalid")
    task_path = Path(str(prompt.get("task_file_path", "")))
    if not task_path.is_file() or hashlib.sha256(task_path.read_bytes()).hexdigest() != prompt.get("task_file_sha256"):
        raise ValueError("task file drift")
    if prompt.get("task_instructions") != task_path.read_bytes().decode("utf-8"):
        raise ValueError("prompt task instructions drift")
    if round_name == "R1" and any(key in prompt for key in ("r1_response_shas", "r2_response_shas", "peer_artifacts")):
        raise ValueError("R1 prompt is not blind")
    previous_shas = None
    if round_name in ("R2", "R3"):
        prior = "R1" if round_name == "R2" else "R2"
        shas, peers, predecessor_completed = _peer_bindings(panel_dir, prior, evidence_sha, evidence_file_sha, binding_sha)
        if prompt.get(f"{prior.lower()}_response_shas") != shas or prompt.get("peer_artifacts") != peers:
            raise ValueError("prompt predecessor bindings drift")
        if started <= predecessor_completed: raise ValueError("agent started before predecessor completion")
        previous_shas = shas
    raw_path = Path(raw_response_path).resolve(); raw = _read(raw_path)
    receipt_path = (panel_dir / f"receipt_{round_name}_{judge}.json").resolve()
    receipt = {"agent_id": agent_id, "task_name": task_name, "model": model, "reasoning_effort": reasoning_effort,
               "judge": judge, "round": round_name, "prompt_sha256": prompt_sha256, "started_at": started_at, "completed_at": completed_at}
    _atomic(receipt_path, receipt)
    artifact: dict[str, Any] = {"panel_id": prompt["panel_id"], "judge": judge, "round": round_name, "model": model,
        "reasoning_effort": reasoning_effort, "prompt_sha": prompt_sha256, "prompt_path": str(prompt_path), "prompt_sha256": prompt_sha256,
        "receipt_path": str(receipt_path), "receipt_sha256": sha256_file(receipt_path), "evidence_packet_sha": evidence_sha,
        "evidence_packet_file_sha256": evidence_file_sha,
        "binding_sha256": binding_sha, "response_sha": sha256_obj(raw), "timestamp": timestamp, "response": raw,
        "raw_response_path": str(raw_path), "raw_response_sha256": sha256_file(raw_path)}
    if round_name == "R2": artifact["r1_response_shas"] = previous_shas
    if round_name == "R3":
        artifact["r2_response_shas"] = previous_shas
        for key in R3_PRESERVED_FIELDS:
            if key in raw: artifact[key] = raw[key]
        if "conclusion" not in artifact: raise ValueError("R3 raw response must contain top-level conclusion")
    artifact_path = (panel_dir / f"{round_name.lower()}_{judge}.json").resolve(); _atomic(artifact_path, artifact)
    return artifact_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--evidence-packet", required=True); p.add_argument("--panel-dir", required=True); p.add_argument("--task-file", required=True); p.add_argument("--judge", required=True, choices=JUDGES); p.add_argument("--round", dest="round_name", required=True, choices=ROUNDS); p.add_argument("--model", required=True); p.add_argument("--reasoning-effort", required=True); p.add_argument("--prepared-at", required=True)
    f = sub.add_parser("finalize"); f.add_argument("--panel-dir", required=True); f.add_argument("--prompt-path", required=True); f.add_argument("--prompt-sha256", required=True); f.add_argument("--agent-id", required=True); f.add_argument("--task-name", required=True); f.add_argument("--model", required=True); f.add_argument("--reasoning-effort", required=True); f.add_argument("--started-at", required=True); f.add_argument("--completed-at", required=True); f.add_argument("--timestamp", required=True); f.add_argument("--raw-response-path", required=True)
    args = vars(parser.parse_args()); command = args.pop("command")
    print(prepare_panel_prompt(**args) if command == "prepare" else finalize_panel_round(**args))


if __name__ == "__main__": main()
