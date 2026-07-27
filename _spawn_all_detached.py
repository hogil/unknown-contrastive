#!/usr/bin/env python3
"""모든 daemon (monitor + master + analyzer + recorder + chain) 을
새창 없이 (CREATE_NO_WINDOW + DETACHED_PROCESS) inline spawn.

사용자 절대규칙 (260517 23:30): 새창 절대 금지.
agent SKILL/dispatch 거치지 않고 직접 Python subprocess.Popen 만 사용.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path("D:/project/unknown-contrastive")
LOG_DIR = REPO / "_dispatch_logs"
LOG_DIR.mkdir(exist_ok=True)

# Windows detached + no-window flags (사용자 절대 규칙)
DETACHED_PROCESS = 0x00000008
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
FLAGS = CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS | CREATE_NO_WINDOW


def spawn(name: str, script: str, log: str) -> int:
    log_path = LOG_DIR / log
    fp = log_path.open("ab")
    proc = subprocess.Popen(
        [sys.executable, "-u", str(REPO / script)],
        cwd=str(REPO),
        stdout=fp,
        stderr=subprocess.STDOUT,
        creationflags=FLAGS,
        close_fds=True,
    )
    print(f"  {name:25s}  PID={proc.pid:>6}  log={log_path.name}", flush=True)
    return proc.pid


def main():
    print("=== spawn_all_detached (no console windows) ===", flush=True)
    pids = {}
    pids["monitor"] = spawn("resource-monitor (10s)", "_loop_resource_1min.py",
                            "_spawn_monitor.log")
    pids["master"] = spawn("contrastive-master", "_loop_master_orchestrator.py",
                           "_spawn_master.log")
    pids["analyzer"] = spawn("cluster-analyzer", "_loop_analyzer_30min.py",
                             "_spawn_analyzer.log")
    pids["recorder"] = spawn("paper-recorder", "_paper_recorder_loop.py",
                             "_spawn_recorder.log")
    pids["chain"] = spawn("n500 chain", "_loop_n500_full.py",
                          "_spawn_chain.log")
    print(f"\n=== {len(pids)} processes spawned (detached + no_window) ===", flush=True)
    print(f"PIDs: {pids}", flush=True)


if __name__ == "__main__":
    main()
