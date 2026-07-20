#!/usr/bin/env python3
"""Run one read-only Codex analysis agent for a completed experiment result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = Path(__file__).with_name("result_analysis_decision.schema.json")
DEFAULT_STATE_ROOT = ROOT / "runs" / "result_analysis_agent"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-file", type=Path, required=True)
    parser.add_argument(
        "--context",
        choices=("may_source", "hard42", "cross_dataset"),
        required=True,
    )
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="0 retries until the agent produces a valid decision",
    )
    parser.add_argument("--attempt-timeout", type=int, default=1800)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_id(event_file: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{event_file.parent.name}_{event_file.stem}")
    return f"{stem}_{sha256(event_file)[:12]}"


def codex_command() -> list[str]:
    wrapper = shutil.which("codex.cmd") or shutil.which("codex")
    if not wrapper:
        raise FileNotFoundError("Codex CLI is not on PATH")
    npm_dir = Path(wrapper).resolve().parent
    script = npm_dir / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
    node = npm_dir / "node.exe"
    if not node.exists():
        resolved = shutil.which("node")
        if not resolved:
            raise FileNotFoundError("Node.js is required to run Codex CLI")
        node = Path(resolved)
    if not script.exists():
        raise FileNotFoundError(f"Codex CLI script is unavailable: {script}")
    return [str(node), str(script)]


def prompt_for(event_file: Path, context: str, identifier: str, state_root: Path) -> str:
    return f"""
You are the read-only result-analysis agent for the unknown-contrastive experiment loop.

Completed event:
- event_id: {identifier}
- context: {context}
- event_file: {event_file}
- decision_history: {state_root / 'decisions'}

Objective:
Find a contrastive-learning recipe that improves over the matching frozen
backbone on multiple datasets. The May experiment is a mechanism control, not
the final claim. Hard-42 strict novel must beat its own frozen baseline without
using train/eval target-class overlap.

Required analysis:
1. Inspect the event file, its run directory, config, canonical metrics, prior
   cells for the same backbone/protocol, and the matching frozen baseline.
2. Report canonical P1 as unique dominant/main target classes divided by all
   target classes. Enforce capture_count <= dominant_cluster_count <= k.
   Never call legacy image presence/coverage P1.
3. Record and compare P1, P2 noise, P3 completeness, P4 homogeneity, ARI,
   Silhouette, k, and fragment ratio. Decide in this order: P1 first, then
   P2/P3/P4 and k/fragmentation. ARI and Silhouette are supporting metrics and
   must never override a P1/P3/P4 regression. Use the protocol's pre-registered
   primary partition and Louvain as a sanity check when available.
4. Diagnose overmerge, fragmentation, noise, capture regression,
   representation regression, clusterer artifact, or invalid protocol.
5. Recommend exactly one next one-axis experiment or state that the required
   source-control sequence should continue. Do not recommend a multi-axis
   combo until its individual components have evidence.
6. Distinguish same-pool/transductive rediscovery from strict-novel discovery.
7. Use absolute filesystem paths for every artifact.

Operational constraints:
- Read only. Do not edit files, launch/stop processes, or consume the GPU.
- Do not browse the web.
- Do not infer success from loss alone.
- If evidence is missing or mismatched, return invalid/inconclusive and ask for
  the smallest controlled rerun.
- Return only JSON that satisfies the supplied schema. Put the concise Korean
  user-facing explanation in analysis_markdown.
""".strip()


def append_state(state_root: Path, record: dict) -> None:
    state_path = state_root / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {"processed": []}
    processed = [item for item in state.get("processed", []) if item.get("event_id") != record["event_id"]]
    processed.append(record)
    state["processed"] = processed
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_decision(decision: object, identifier: str, context: str) -> dict:
    if not isinstance(decision, dict):
        raise ValueError("agent output is not a JSON object")
    if decision.get("event_id") != identifier:
        raise ValueError(f"event_id mismatch: {decision.get('event_id')} != {identifier}")
    if decision.get("context") != context:
        raise ValueError(f"context mismatch: {decision.get('context')} != {context}")
    if not isinstance(decision.get("analysis_markdown"), str):
        raise ValueError("analysis_markdown is missing")
    if not isinstance(decision.get("next_experiment"), dict):
        raise ValueError("next_experiment is missing")
    return decision


def main() -> None:
    args = parse_args()
    event_file = args.event_file.resolve()
    state_root = args.state_root.resolve()
    if not event_file.exists():
        raise FileNotFoundError(f"result event is unavailable: {event_file}")
    if not SCHEMA.exists():
        raise FileNotFoundError(f"decision schema is unavailable: {SCHEMA}")

    identifier = event_id(event_file)
    decisions = state_root / "decisions"
    reports = state_root / "reports"
    logs = state_root / "logs"
    for directory in (state_root, decisions, reports, logs):
        directory.mkdir(parents=True, exist_ok=True)
    decision_path = decisions / f"{identifier}.json"
    report_path = reports / f"{identifier}.md"
    if decision_path.exists() and report_path.exists() and not args.force:
        print(f"[RESULT_AGENT] reuse decision={decision_path}", flush=True)
        return

    prompt = prompt_for(event_file, args.context, identifier, state_root)
    command_prefix = codex_command()
    attempt = 0
    while True:
        attempt += 1
        temporary = state_root / f".{identifier}.attempt{attempt}.json"
        log_path = logs / f"{identifier}.attempt{attempt}.log"
        command = command_prefix + [
            "--ask-for-approval",
            "never",
            "--sandbox",
            "read-only",
            "--cd",
            str(ROOT),
            "exec",
            "--ephemeral",
            "--color",
            "never",
            "--output-schema",
            str(SCHEMA),
            "--output-last-message",
            str(temporary),
            prompt,
        ]
        started = datetime.now().isoformat(timespec="seconds")
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=args.attempt_timeout,
                check=False,
            )
            log_path.write_text(
                f"started={started}\nexit={completed.returncode}\n"
                f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}\n",
                encoding="utf-8",
            )
            if completed.returncode != 0:
                raise RuntimeError(f"Codex exited {completed.returncode}; log={log_path}")
            decision = validate_decision(
                json.loads(temporary.read_text(encoding="utf-8")), identifier, args.context
            )
            decision["event_file"] = str(event_file)
            decision["event_sha256"] = sha256(event_file)
            decision["analyzed_at"] = datetime.now().isoformat(timespec="seconds")
            decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
            report_path.write_text(decision["analysis_markdown"].rstrip() + "\n", encoding="utf-8")
            append_state(
                state_root,
                {
                    "event_id": identifier,
                    "context": args.context,
                    "event_file": str(event_file),
                    "decision": str(decision_path),
                    "report": str(report_path),
                    "verdict": decision["verdict"],
                    "next_action": decision["next_experiment"]["action"],
                    "analyzed_at": decision["analyzed_at"],
                },
            )
            temporary.unlink(missing_ok=True)
            print(
                f"[RESULT_AGENT] analyzed event={identifier} verdict={decision['verdict']} "
                f"next={decision['next_experiment']['action']} decision={decision_path}",
                flush=True,
            )
            return
        except (OSError, ValueError, RuntimeError, json.JSONDecodeError, subprocess.TimeoutExpired) as error:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"\nERROR: {type(error).__name__}: {error}\n")
            temporary.unlink(missing_ok=True)
            print(f"[RESULT_AGENT] attempt={attempt} failed: {error}", flush=True)
            if args.max_attempts > 0 and attempt >= args.max_attempts:
                raise
            time.sleep(min(300, 30 * attempt))


if __name__ == "__main__":
    main()
