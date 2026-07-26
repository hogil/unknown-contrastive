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
  - CPU only while any GPU training is active elsewhere (>=1 free GB is
    the known "other training is running" signature on this box).
  - GPU training: at most 1 concurrent process for this campaign.
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
import subprocess
import sys
import time
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


def now_ts() -> str:
    return datetime.now().strftime("%y%m%d_%H%M%S")


def sha256_obj(obj: Any) -> str:
    """sha256 of a canonical (sorted-key) JSON encoding -- used for recipe fingerprints."""
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def read_json(path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


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
    locks = scan_running_locks([REPO_ROOT / "runs"])
    alive_locks = [l for l in locks if l["alive"]]
    max_concurrent = int(cfg_safety.get("gpu_max_concurrent", 1))
    busy_by_lock = len(alive_locks) >= max_concurrent
    min_free_mb = float(cfg_safety.get("gpu_min_free_mb", 1024))
    busy_by_mem = (free_mb is not None) and (free_mb < min_free_mb)
    detail = {"free_mb": free_mb, "min_free_mb": min_free_mb, "alive_locks": alive_locks,
               "max_concurrent": max_concurrent, "busy_by_lock": busy_by_lock, "busy_by_mem": busy_by_mem}
    ok = not (busy_by_lock or busy_by_mem)
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


# ===================== promotion gate (hardcoded thresholds) =====================
# Values below are the literal numbers from the approved campaign plan (team-lead,
# 260726). Do not loosen without an explicit user decision -- record any change in
# configs/unknown_campaign_v1.json's "gates" block (this function only reads it).
DEFAULT_GATES = {
    "capture_drop_pp_max": 1.0,
    "far_increase_pp_max": 0.5,
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
      background_far: {"increase_pp": float}
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
    if not bg or bg.get("increase_pp") is None:
        fail("missing background_far.increase_pp")
    elif bg["increase_pp"] > g["far_increase_pp_max"]:
        fail(f"background_far increased {bg['increase_pp']:.2f}pp > {g['far_increase_pp_max']}pp max")

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
    {"far_events": int, "detect_batches": int, "novel_per_batch": int}.
    Returns None (fail-closed upstream) if the file or the requested row is
    missing -- never estimates.
    """
    path = Path(report_dir) / "summary_tables.csv"
    if not path.exists():
        return None
    import csv
    p_val, k_val = str(operating_point.get("P")), str(operating_point.get("K"))
    size_col, arm = operating_point.get("size_col"), operating_point.get("arm")
    if not (p_val and k_val and size_col and arm):
        return None
    rows = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("arm") == arm and row.get("P") == p_val and row.get("K") == k_val:
                rows[row.get("metric")] = row.get(size_col)
    far = rows.get("FAR_alarms_over_4_bg_batches")
    lag = rows.get("true_positive_detection_lag")
    if far is None or lag is None:
        return None
    digits = "".join(ch for ch in size_col if ch.isdigit())
    return {"far_events": int(far), "detect_batches": int(lag),
            "novel_per_batch": int(digits) if digits else None}


def load_background_far_delta(report_dir, operating_point: dict, baseline_arm: str = "frozen") -> dict | None:
    """held-out background FAR delta = candidate arm's FAR-alarm COUNT minus the
    baseline (frozen) arm's, at the same (P,K,size_col) cell. Per team-lead
    260726: this reuses the temporal report's held-out batches (t=5-8), not a
    separate per-image percentage -- the "increase_pp" field name is kept for
    evaluate_gate() schema compatibility, but the value here is an event-count
    delta, not a true percentage-point. Returns None if either side is missing.
    """
    cand = load_temporal_metrics(report_dir, operating_point)
    base = load_temporal_metrics(report_dir, {**operating_point, "arm": baseline_arm})
    if cand is None or base is None:
        return None
    return {"increase_pp": cand["far_events"] - base["far_events"],
            "_unit_note": "event-count delta over held-out background batches, not a true percentage-point"}


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
        proc = subprocess.Popen(argv, cwd=str(cwd), stdout=logf, stderr=subprocess.STDOUT, env=full_env)
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
                    proc.kill()
                    proc.wait(timeout=30)
                    status = "timeout"
                    rc = proc.returncode
                    break
        duration = time.time() - t0
    heartbeat_path.unlink(missing_ok=True)
    if status == "completed" and rc != 0:
        status = "failed"
    return {"status": status, "returncode": rc, "duration_sec": round(duration, 1), "log_path": str(log_path)}


# ===================== preflight =====================
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
    disk_ok, disk_detail = check_disk_guard(safety.get("min_free_gb", {"D": 200, "E": 500}))
    detail["disk"] = disk_detail
    if not disk_ok:
        blockers.append("disk_low")

    if step.get("needs_gpu"):
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


class Campaign:
    def __init__(self, config_path=DEFAULT_CONFIG, registry_path=DEFAULT_REGISTRY, python_exe=None):
        self.config_path = Path(config_path)
        self.cfg = read_json(self.config_path)
        self.registry = Registry(registry_path)
        self.python_exe = python_exe or sys.executable

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
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        if CAMPAIGN_LOCK.exists():
            try:
                info = json.loads(CAMPAIGN_LOCK.read_text(encoding="utf-8"))
                import psutil
                if info.get("pid") and psutil.pid_exists(int(info["pid"])):
                    raise SystemExit(
                        f"another campaign runner is already active (pid={info['pid']}, "
                        f"lock={CAMPAIGN_LOCK}) -- run with --status to inspect it, or use "
                        f"--stop-after-current to ask it to exit after its current step.")
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
        CAMPAIGN_LOCK.write_text(json.dumps(
            {"pid": os.getpid(), "started_at": now_ts(), "host": platform.node()}), encoding="utf-8")

    def _release_lock(self) -> None:
        CAMPAIGN_LOCK.unlink(missing_ok=True)

    # ---- one atomic step: preflight -> dispatch -> retry(<=2) -> registry append ----
    def _run_one_step(self, tier_cfg: dict, step: dict, seed) -> dict:
        tier = tier_cfg["tier"]
        max_retries = int(self.cfg.get("safety", {}).get("max_retries", 2))
        run_tag = f"t{tier}_{step['name']}" + (f"_seed{seed}" if seed is not None else "")
        run_id = f"{run_tag}_{now_ts()}"
        run_dir = REPO_ROOT / "runs" / "campaign" / run_id

        recipe = dict(step.get("recipe", {}))
        if seed is not None:
            recipe["seed"] = seed
        recipe_sha = sha256_obj(recipe) if recipe else None
        manifest = step.get("manifest")
        dataset_sha = sha256_file(manifest) if manifest and Path(manifest).exists() else None
        ckpt_entries = _ckpt_entries(step)

        base = {"run_id": run_id, "tier": tier, "tier_name": tier_cfg.get("name"), "step": step["name"],
                "seed": seed, "dataset_sha256": dataset_sha, "manifest_path": str(manifest) if manifest else None,
                "recipe": recipe or None, "recipe_sha256": recipe_sha,
                "checkpoint_path": step.get("checkpoint"), "checkpoint_sha256": ckpt_entries or None,
                "predictor": step.get("predictor")}

        attempt = 0
        while True:
            attempt += 1
            champion_before = (self.registry.current_champion() or {}).get("run_id")
            try:
                ok, blockers, pf_detail = preflight(step, self.cfg)
            except Exception as e:  # a broken step config must never crash the whole campaign
                rec = {**base, "status": "blocked", "failure_reason": f"preflight_error:{e}",
                       "retry_count": attempt - 1, "attempt": attempt,
                       "champion_before": champion_before, "champion_after": None,
                       "metrics": None, "gate": {"evaluated": False, "passed": None, "reasons": None}}
                self.registry.append(rec)
                return rec
            if not ok:
                rec = {**base, "status": "blocked", "failure_reason": ",".join(blockers),
                       "retry_count": attempt - 1, "attempt": attempt, "preflight": pf_detail,
                       "champion_before": champion_before, "champion_after": None,
                       "metrics": None, "gate": {"evaluated": False, "passed": None, "reasons": None}}
                self.registry.append(rec)
                return rec

            try:
                argv = build_command(step, {"python": self.python_exe, "run_dir": str(run_dir),
                                             "seed": seed, "manifest": manifest, **step.get("context", {})})
                env = build_env(step, {"python": self.python_exe, "run_dir": str(run_dir),
                                     "seed": seed, "manifest": manifest, **step.get("context", {})})
            except ValueError as e:
                rec = {**base, "status": "blocked", "failure_reason": f"command_build_error:{e}",
                       "retry_count": attempt - 1, "attempt": attempt,
                       "champion_before": champion_before, "champion_after": None,
                       "metrics": None, "gate": {"evaluated": False, "passed": None, "reasons": None}}
                self.registry.append(rec)
                return rec

            timeout_sec = int(step.get("timeout_sec", self.cfg.get("safety", {}).get("run_timeout_sec", 43200)))
            heartbeat_sec = int(self.cfg.get("safety", {}).get("heartbeat_sec", 60))
            result = run_subprocess_step(argv, run_dir, timeout_sec, heartbeat_sec, env=env)

            metrics = None
            metrics_path = step.get("metrics_path")
            if metrics_path:
                mp = Path(str(metrics_path).format(run_dir=run_dir))
                if mp.exists():
                    metrics = read_json(mp)
            if metrics is None:
                metrics = extract_metrics_from_run(run_dir)
                metrics.update(build_gate_metrics(step, self.cfg, run_dir))  # baseline-diffed gate fields

            gate_passed, gate_reasons = None, None
            if step.get("evaluate_gate", False):
                gate_cfg = tier_cfg.get("gates", self.cfg.get("gates", {}))
                gate_passed, gate_reasons = evaluate_gate(gate_cfg, metrics, tier_cfg.get("n_seeds_min"))

            champion_after = None
            if result["status"] == "completed" and gate_passed:
                champion_after = self._maybe_promote(tier, run_id, ckpt_entries, gate_passed)

            failure_reason = None if result["status"] == "completed" else \
                f"{result['status']}:returncode={result.get('returncode')}"

            rec = {**base, "metrics": metrics,
                   "gate": {"evaluated": step.get("evaluate_gate", False), "passed": gate_passed,
                             "reasons": gate_reasons},
                   "status": result["status"], "failure_reason": failure_reason,
                   "retry_count": attempt - 1, "attempt": attempt, "duration_sec": result["duration_sec"],
                   "log_path": result["log_path"], "champion_before": champion_before,
                   "champion_after": champion_after}
            self.registry.append(rec)

            if result["status"] == "completed" or attempt > max_retries:
                return rec
            time.sleep(min(30 * attempt, 120))  # brief backoff before retry

    # ---- full ladder loop ----
    def run(self, resume: bool, max_tier: int | None, stop_after_current: bool) -> None:
        if not resume and self.registry.read_all():
            raise SystemExit(f"registry {self.registry.path} already has entries -- pass --resume to "
                              f"continue this campaign (or point at a fresh config/registry).")
        self._acquire_lock()
        try:
            completed = self.registry.completed_steps()
            for tier_cfg in self.cfg.get("tiers", []):
                tier = tier_cfg["tier"]
                if max_tier is not None and tier > max_tier:
                    print(f"[campaign] stopping before tier {tier} (--max-tier {max_tier})")
                    break
                for step in tier_cfg.get("steps", []):
                    for seed in (step.get("seeds") or [None]):
                        if (tier, step["name"], seed) in completed:
                            print(f"[campaign] skip tier={tier} step={step['name']} seed={seed} (completed)")
                            continue
                        print(f"[campaign] tier={tier} step={step['name']} seed={seed} -> dispatching")
                        rec = self._run_one_step(tier_cfg, step, seed)
                        print(f"[campaign] tier={tier} step={step['name']} seed={seed} -> "
                              f"status={rec['status']} gate_passed={rec['gate']['passed']}")
                        if rec["status"] == "completed":
                            completed.add((tier, step["name"], seed))
                        if STOP_FLAG.exists():
                            print(f"[campaign] STOP_REQUESTED honored after tier={tier} step={step['name']}")
                            STOP_FLAG.unlink(missing_ok=True)
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
    a = ap.parse_args(argv)

    campaign = Campaign(config_path=a.config)

    if a.status:
        print(campaign.status())
        return 0

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
