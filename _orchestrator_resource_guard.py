#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resource watchdog for the running obj_id_maps build (and later compound training).

Polls every POLL_SEC. Reads RAM via psutil, GPU via nvidia-smi.
- If RAM > 80% sustained for SUSTAIN_SEC -> abort
- If GPU mem-used / mem-total > 90% sustained for SUSTAIN_SEC -> abort

When abort:
  - kill the watched python process (graceful then hard)
  - rename log/_obj_id_maps_full_build.log -> log/_obj_id_maps_full_build_PAUSED_<TS>.log
  - DO NOT touch any output dir (obj_id_maps/, log/<run>/) — never delete

Reads watched PID from CLI arg or auto-detect by command pattern.
"""
from __future__ import annotations
import argparse, os, sys, time, subprocess, signal
from pathlib import Path

# stdout/stderr 를 UTF-8 로 (Windows cp949 환경에서 em-dash, 한글 print 깨짐 방지)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import psutil

POLL_SEC = 30
SUSTAIN_SEC = 90
RAM_LIMIT = 0.80
GPU_MEM_LIMIT = 0.90


def gpu_mem_ratio() -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        )
        used, total = [int(x) for x in out.strip().splitlines()[0].split(",")]
        return used / max(1, total)
    except Exception:
        return 0.0


def find_target_pid(pattern: str) -> int | None:
    self_pid = os.getpid()
    self_script = Path(__file__).name
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            if p.info["pid"] == self_pid:
                continue
            cl = p.info.get("cmdline") or []
            joined = " ".join(cl)
            # skip processes that look like the guard itself (e.g. when this script
            # is invoked with --pattern referencing the target script name).
            if self_script in joined:
                continue
            if pattern in joined and "python" in joined.lower():
                return p.info["pid"]
        except Exception:
            continue
    return None


def kill_proc(pid: int) -> None:
    try:
        p = psutil.Process(pid)
        for ch in p.children(recursive=True):
            try: ch.terminate()
            except Exception: pass
        p.terminate()
        try: p.wait(timeout=15)
        except psutil.TimeoutExpired: p.kill()
    except psutil.NoSuchProcess:
        pass


def rename_paused(log_path: Path) -> Path | None:
    if not log_path.exists():
        return None
    ts = time.strftime("%y%m%d_%H%M%S")
    new = log_path.with_name(log_path.stem + f"_PAUSED_{ts}" + log_path.suffix)
    try:
        log_path.rename(new)
        return new
    except Exception as e:
        print(f"[guard] rename failed: {e}", flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", required=True,
                    help="substring to find python process (e.g. _build_obj_id_maps.py)")
    ap.add_argument("--log", required=True, help="path to log file to rename on abort")
    ap.add_argument("--ram-limit", type=float, default=RAM_LIMIT,
                    help=f"RAM percent limit 0..1 (default: {RAM_LIMIT})")
    ap.add_argument("--gpu-mem-limit", type=float, default=GPU_MEM_LIMIT,
                    help=f"GPU mem percent limit 0..1 (default: {GPU_MEM_LIMIT}). "
                         f"Inference workload typically holds model+cache at ~90%, "
                         f"bump to 0.95-0.97 to avoid false positives.")
    ap.add_argument("--sustain-sec", type=float, default=SUSTAIN_SEC,
                    help=f"sustained over-limit before kill, seconds (default: {SUSTAIN_SEC})")
    ap.add_argument("--poll-sec", type=float, default=POLL_SEC,
                    help=f"poll interval seconds (default: {POLL_SEC})")
    args = ap.parse_args()

    log_path = Path(args.log)
    ram_limit = float(args.ram_limit)
    gpu_mem_limit = float(args.gpu_mem_limit)
    sustain_sec = float(args.sustain_sec)
    poll_sec = float(args.poll_sec)
    print(f"[guard] watching pattern={args.pattern!r} log={log_path}", flush=True)
    print(f"[guard] limits: ram={ram_limit:.0%} gpu_mem={gpu_mem_limit:.0%} "
          f"sustain={sustain_sec:.0f}s poll={poll_sec:.0f}s", flush=True)

    over_since: float | None = None
    while True:
        pid = find_target_pid(args.pattern)
        if pid is None:
            print("[guard] target process not found - exiting watchdog", flush=True)
            return 0
        ram = psutil.virtual_memory().percent / 100.0
        gpu = gpu_mem_ratio()
        over = (ram > ram_limit) or (gpu > gpu_mem_limit)
        ts = time.strftime("%H:%M:%S")
        print(f"[guard {ts}] pid={pid} ram={ram:.0%} gpu_mem={gpu:.0%} over={over}", flush=True)
        if over:
            if over_since is None:
                over_since = time.time()
            elif time.time() - over_since > sustain_sec:
                print(f"[guard] LIMIT EXCEEDED for {sustain_sec:.0f}s - killing pid={pid}", flush=True)
                kill_proc(pid)
                renamed = rename_paused(log_path)
                print(f"[guard] log renamed to: {renamed}", flush=True)
                return 2
        else:
            over_since = None
        time.sleep(poll_sec)


if __name__ == "__main__":
    sys.exit(main())
