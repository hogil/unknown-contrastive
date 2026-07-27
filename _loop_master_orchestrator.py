#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""contrastive chain orchestrator — 60-min polling loop, 8 iters (8h total).

매 iter:
  1) chain alive: _dispatch_logs/_loop_n500_full_*.log 최근 mtime (5min 이내 = alive)
                  + _loop_n500_full.py python proc 존재
  2) cell progress: outputs_contrastive_*/tier1_*_n500.json glob count
  3) 자매 GPU (chain dead 일 때만): sister mem < 2 GB AND util < 10% 이면 re-dispatch
  4) _loop_master.log 한 줄 append
  5) 60min sleep (time.sleep)

절대 금지: python proc kill, 결과 폴더 삭제, 새 cmd 창.
"""
from __future__ import annotations
import glob
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = Path("D:/project/unknown-contrastive")
LOG_DIR = REPO / "_dispatch_logs"
MASTER_LOG = REPO / "_loop_master.log"
SUMMARY_MD = REPO / "_loop_master_summary.md"

CHAIN_LOG_GLOB = "_loop_n500_full_*.log"
CELL_GLOB = "outputs_contrastive_*/tier1_*_n500.json"
ALIVE_MAX_AGE_SEC = 5 * 60      # 5 min
SLEEP_SEC = 60 * 60             # 60 min
ITERS = 8

SISTER_MEM_OK_MB = 2 * 1024     # < 2 GB
SISTER_UTIL_OK_PCT = 10         # < 10%


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_master(line: str) -> None:
    with MASTER_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def chain_log_age() -> tuple[float, Path | None]:
    """가장 최근 chain log 의 mtime 으로부터 경과 초. (age_sec, path)."""
    candidates = sorted(LOG_DIR.glob(CHAIN_LOG_GLOB),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    # filter out *_E_* sub-logs — main loop log 만
    mains = [p for p in candidates if "_E_" not in p.name and "_status" not in p.name]
    if not mains:
        return float("inf"), None
    p = mains[0]
    return time.time() - p.stat().st_mtime, p


def loop_proc_alive() -> bool:
    try:
        import psutil
        for p in psutil.process_iter(['name', 'cmdline']):
            cl = p.info.get('cmdline') or []
            if any('_loop_n500_full.py' in c for c in cl):
                return True
        return False
    except Exception:
        return False


def cell_count() -> int:
    return len(list(REPO.glob(CELL_GLOB)))


def gpu_state() -> tuple[int, int]:
    """전체 GPU mem_used_MB, util_pct. nvidia-smi 직접 read (sister + ours 합산)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT, timeout=10,
        ).decode("utf-8", errors="replace").strip()
        # 한 줄: "10401, 72"
        first = out.splitlines()[0]
        parts = [x.strip() for x in first.split(",")]
        mem_mb = int(parts[0])
        util = int(parts[1])
        return mem_mb, util
    except Exception as e:
        return -1, -1


def sister_gpu_state() -> tuple[int, int]:
    """sister (non-our) GPU 사용량 추정.
    chain alive 면 chain 이 GPU 점유 중이므로 의미 없음. dead 일 때만 호출."""
    return gpu_state()  # chain dead 일 때만 호출되므로 sister 만 남음


def redispatch_chain() -> str:
    """_loop_n500_full.py 를 background 로 띄움. PID 반환."""
    try:
        # Popen 으로 새 console 없이 background 실행
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        # CREATE_NO_WINDOW = 0x08000000 (Windows) — 새 console 창 안 만듦
        creationflags = 0
        if sys.platform == "win32":
            creationflags = 0x08000000
        proc = subprocess.Popen(
            [sys.executable, "_loop_n500_full.py"],
            cwd=str(REPO),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
            env=env,
        )
        return f"redispatched(pid={proc.pid})"
    except Exception as e:
        return f"redispatch_fail({e})"


def write_summary(records: list[dict]) -> None:
    lines = [
        f"# contrastive orchestrator summary",
        f"",
        f"- started: {records[0]['ts'] if records else '?'}",
        f"- finished: {records[-1]['ts'] if records else '?'}",
        f"- iters: {len(records)}",
        f"",
        f"| iter | ts | chain | cells | gpu_mem_MB | gpu_util | action |",
        f"|---|---|---|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| {r['i']} | {r['ts']} | {r['chain']} | {r['cells']} | "
            f"{r['gpu_mem']} | {r['gpu_util']} | {r['action']} |"
        )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    append_master(f"=== loop_master_orchestrator start {now_iso()} (iters={ITERS}, sleep={SLEEP_SEC}s) ===")
    records: list[dict] = []
    for i in range(1, ITERS + 1):
        age, log_path = chain_log_age()
        proc_alive = loop_proc_alive()
        chain_alive = (age <= ALIVE_MAX_AGE_SEC) and proc_alive
        chain_str = (
            f"alive(log_age={int(age)}s,proc=1)" if chain_alive
            else f"dead(log_age={int(age) if age != float('inf') else 'inf'}s,proc={1 if proc_alive else 0})"
        )
        cells = cell_count()
        mem_mb, util = gpu_state()

        if chain_alive:
            action = "chain_running_skip"
        else:
            # chain dead → sister GPU 점검
            if mem_mb < SISTER_MEM_OK_MB and util < SISTER_UTIL_OK_PCT and mem_mb >= 0:
                action = redispatch_chain()
            else:
                action = f"sister_busy_skip(mem={mem_mb}MB,util={util}%)"

        line = (f"[iter {i}/{ITERS}] {now_iso()} chain={chain_str} "
                f"cells={cells}/6 gpu_mem={mem_mb}MB util={util}% action={action}")
        append_master(line)
        records.append({
            "i": i, "ts": now_iso(), "chain": chain_str, "cells": f"{cells}/6",
            "gpu_mem": mem_mb, "gpu_util": util, "action": action,
        })
        write_summary(records)
        if i < ITERS:
            time.sleep(SLEEP_SEC)

    append_master(f"=== loop_master_orchestrator done {now_iso()} ===")
    write_summary(records)


if __name__ == "__main__":
    main()
