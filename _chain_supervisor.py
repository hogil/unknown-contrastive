#!/usr/bin/env python3
"""Chain supervisor — 학습 안 멈추게 보장.

매 60s 마다:
1. `_loop_n500_full.py` chain process alive 확인
2. dead 시 즉시 respawn (detached, no console)
3. run_contrastive child alive 확인 (학습 active 여부)
4. 모두 dead 시 alert + respawn

종료 조건: 6 cells 완료 (tier1_B0/B1/B3/B4/B5/NEW_n500.json 모두 존재) 또는 사용자 kill.

사용자 명시 (260520 05:30): 학습 안 멈추게 보장 + 오류 시 새로 돌리기.
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

REPO = Path(__file__).resolve().parent
LOG = REPO / "_chain_supervisor.log"
TARGET_CELLS = ["B0_n500", "B1_n500", "B3_n500", "B4_n500", "B5_n500", "NEW_n500"]
SPAWN_FLAGS = 0x00000200 | 0x00000008 | 0x08000000   # NEW_PG | DETACHED | NO_WINDOW
INTERVAL_SEC = 60


def log(msg: str):
    ts = datetime.now().isoformat(timespec="seconds")
    line = f"[{ts}] {msg}\n"
    print(line.strip(), flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def find_chain_pids() -> list[int]:
    """`_loop_n500_full.py` 살아있는 PID 모음."""
    pids = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info.get("name") != "python.exe":
                continue
            cmd = " ".join(p.info.get("cmdline") or [])
            if "_loop_n500_full" in cmd:
                pids.append(p.info["pid"])
        except Exception:
            pass
    return pids


def find_child_pids() -> list[int]:
    """run_contrastive.py 살아있는 PID 모음 (학습 active)."""
    pids = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info.get("name") != "python.exe":
                continue
            cmd = " ".join(p.info.get("cmdline") or [])
            if "run_contrastive" in cmd:
                pids.append(p.info["pid"])
        except Exception:
            pass
    return pids


def cells_done() -> set[str]:
    """tier1_*_n500.json 존재 cell 모음."""
    done = set()
    for cell in TARGET_CELLS:
        for f in REPO.glob(f"outputs_contrastive_*/tier1_{cell}.json"):
            done.add(cell)
            break
    return done


def spawn_chain() -> int:
    """chain detached spawn."""
    log_dir = REPO / "_dispatch_logs"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    fp = (log_dir / f"_supervisor_chain_{ts}.log").open("ab")
    proc = subprocess.Popen(
        [sys.executable, "-u", str(REPO / "_loop_n500_full.py")],
        cwd=str(REPO),
        stdout=fp,
        stderr=subprocess.STDOUT,
        creationflags=SPAWN_FLAGS,
        close_fds=True,
    )
    return proc.pid


def main():
    log(f"=== chain supervisor START interval={INTERVAL_SEC}s ===")
    log(f"target cells: {TARGET_CELLS}")
    n = 0
    while True:
        n += 1
        try:
            chain_pids = find_chain_pids()
            child_pids = find_child_pids()
            done = cells_done()
            remaining = [c for c in TARGET_CELLS if c not in done]
            log(
                f"[iter {n}] chain={chain_pids} child_train={child_pids} "
                f"done={len(done)}/6 remaining={remaining}"
            )

            # 종료 조건
            if not remaining:
                log(f"✓ ALL 6 cells DONE — supervisor exit")
                return

            # chain dead → respawn
            if not chain_pids:
                log(f"⚠ chain DEAD — respawning")
                pid = spawn_chain()
                log(f"  respawned PID={pid}")
                time.sleep(10)   # spawn 안정화
            elif not child_pids:
                # chain alive but no run_contrastive — dispatch 단계 또는 transition
                log(f"  (chain alive but no run_contrastive — transition or dispatch init)")

        except Exception as e:
            log(f"[iter {n}] err: {e}")

        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
