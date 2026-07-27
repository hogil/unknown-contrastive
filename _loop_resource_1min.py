#!/usr/bin/env python3
"""1-minute resource ENFORCER — strict 30% policy (사용자 명시 260517 21:55).

제1 규칙 (절대 규칙):
- 1분마다 GPU / RAM / CPU 측정
- 우리 프로젝트 사용량 > 30% 시 chain process kill + throttle 강화 + re-dispatch
- 에이전트 4개 daemon 은 절대 kill 안 함 (cmdline 기반 식별)
- 학습 cell 진행 vs 자원 한계 — 자원 한계 우선

agent daemon 보존 (kill list 에서 제외):
- _loop_resource_1min.py (self)
- _loop_master.py / _loop_master_orchestrator.py
- _paper_recorder_loop.py / _paper_recorder_run_2to8.py
- cluster_analyzer_iter.py / cluster_analyzer_summary.py

kill 대상 (chain only):
- _loop_n500_full.py
- _dispatch_iter.py
- run_contrastive.py
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = Path("D:/project/unknown-contrastive")
LOG_JSONL = REPO / "_loop_resource_1min.log"
THROTTLE_OVERRIDE = REPO / "_chain_throttle_override.txt"
ENFORCE_LOG = REPO / "_enforcer_actions.log"

# 사용자 명시 (260519 14:50) — 학습 안정성 우선 95% 완화 (enforce 사실상 off)
LIMIT_PCT = 95.0
THROTTLE_STEP_MS = 20
THROTTLE_MAX_MS = 200
THROTTLE_INIT_MS = 0

AGENT_KEEP = [
    "_loop_resource_1min.py",
    "_loop_master.py", "_loop_master_orchestrator.py",
    "_paper_recorder_loop.py", "_paper_recorder_run_2to8.py",
    "cluster_analyzer_iter.py", "cluster_analyzer_summary.py",
    "_loop_analyzer_30min", "_loop_recorder",
]
CHAIN_KILL = [
    "_loop_n500_full.py",
    "_dispatch_iter.py",
    "run_contrastive.py",
]


def gpu_snapshot():
    # CREATE_NO_WINDOW=0x08000000 — nvidia-smi console popup 방지 (사용자 명시 260518 15:24)
    out = subprocess.check_output(
        ["nvidia-smi",
         "--query-gpu=utilization.gpu,memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        text=True, timeout=5, creationflags=0x08000000,
    ).strip().split("\n")[0]
    parts = [int(x.strip()) for x in out.split(",")]
    return {"util": parts[0], "mem_used_mb": parts[1], "mem_total_mb": parts[2]}


def proc_snapshot():
    import psutil
    ram = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.5)
    pys = []
    for p in psutil.process_iter(['name', 'memory_info']):
        try:
            if p.info.get('name') == 'python.exe':
                pys.append(p.info['memory_info'].rss / (1024 * 1024))
        except Exception:
            pass
    return {
        "ram_used_mb": int(ram.used / (1024 * 1024)),
        "ram_total_mb": int(ram.total / (1024 * 1024)),
        "ram_pct": round(ram.percent, 1),
        "cpu_pct": round(cpu, 1),
        "py_count": len(pys),
        "py_ram_total_mb": round(sum(pys), 1),
    }


def classify_pythons():
    """chain processes vs agent daemons 구분."""
    import psutil
    chain = []
    agent = []
    other = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
        try:
            if p.info.get('name') != 'python.exe':
                continue
            cmd = ' '.join(p.info.get('cmdline') or [])
            mb = p.info['memory_info'].rss / (1024 * 1024)
            row = {"pid": p.info['pid'], "ram_mb": round(mb, 1), "cmd": cmd[:120]}
            # xeval (read-only inference) 은 보존 — env LOAD_CHECKPOINT 있는 run_contrastive
            try:
                envs = p.environ()
                if envs.get('LOAD_CHECKPOINT'):
                    agent.append(row)  # xeval = agent 보존
                    continue
            except Exception:
                pass
            if any(k in cmd for k in AGENT_KEEP):
                agent.append(row)
            elif any(k in cmd for k in CHAIN_KILL):
                chain.append(row)
            else:
                other.append(row)
        except Exception:
            pass
    return chain, agent, other


def measure():
    gpu = gpu_snapshot()
    proc = proc_snapshot()
    # 내 (Claude spawn) python RAM 만 계산 — 시스템 자체 + 다른 app 제외
    sys_total_mb = proc["ram_total_mb"]
    my_ram_pct = (proc["py_ram_total_mb"] / sys_total_mb * 100) if sys_total_mb > 0 else 0
    rec = {
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec='seconds'),
        "gpu_util_pct": gpu["util"],
        "gpu_mem_used_mb": gpu["mem_used_mb"],
        "gpu_mem_total_mb": gpu["mem_total_mb"],
        "gpu_mem_pct": round(gpu["mem_used_mb"] / gpu["mem_total_mb"] * 100, 2),
        **proc,
        "my_ram_pct": round(my_ram_pct, 2),
    }
    # enforce 기준: CPU 도 포함 (사용자 명시 260517 23:24)
    rec["max_pct"] = max(rec["gpu_util_pct"], rec["gpu_mem_pct"],
                        rec["my_ram_pct"], rec["cpu_pct"])
    rec["over_limit"] = rec["max_pct"] > LIMIT_PCT
    return rec


def read_throttle():
    try:
        if THROTTLE_OVERRIDE.exists():
            return int(THROTTLE_OVERRIDE.read_text().strip())
    except Exception:
        pass
    return THROTTLE_INIT_MS


def write_throttle(v):
    THROTTLE_OVERRIDE.write_text(str(v), encoding='utf-8')


def append_enforce_log(msg):
    ts = datetime.now(timezone.utc).isoformat(timespec='seconds')
    with ENFORCE_LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


def kill_chain_only():
    """chain processes kill, agent daemon 보존."""
    import psutil
    chain, agent, other = classify_pythons()
    killed = []
    for row in chain:
        try:
            p = psutil.Process(row['pid'])
            p.kill()
            killed.append(row['pid'])
        except Exception:
            pass
    return killed, len(agent), len(chain), len(other)


def spawn_chain():
    """chain restart with current throttle override."""
    DETACHED_PROCESS = 0x00000008
    CREATE_NO_WINDOW = 0x08000000
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS | CREATE_NO_WINDOW
    log_path = REPO / "_dispatch_logs" / f"_chain_enforcer_{datetime.now().strftime('%y%m%d_%H%M%S')}.log"
    log_path.parent.mkdir(exist_ok=True)
    fp = log_path.open("ab")
    proc = subprocess.Popen(
        [sys.executable, "-u", str(REPO / "_loop_n500_full.py")],
        cwd=str(REPO), stdout=fp, stderr=subprocess.STDOUT,
        creationflags=flags,
    )
    return proc.pid, log_path


def enforce(rec):
    """30% 초과 시 chain kill + throttle 증가 + re-dispatch."""
    cur_throttle = read_throttle()
    new_throttle = min(THROTTLE_MAX_MS, cur_throttle + THROTTLE_STEP_MS)
    write_throttle(new_throttle)
    killed, n_agent, n_chain, n_other = kill_chain_only()
    msg = (f"OVER_LIMIT max={rec['max_pct']:.1f}% "
           f"(gpu_util={rec['gpu_util_pct']}% gpu_mem={rec['gpu_mem_pct']:.0f}% "
           f"ram={rec['ram_pct']}% cpu={rec['cpu_pct']}%) — "
           f"killed chain pids={killed} (kept {n_agent} agents, {n_other} others) "
           f"throttle {cur_throttle}→{new_throttle}ms")
    append_enforce_log(msg)
    print(f"  ⚠ ENFORCE: {msg}", flush=True)
    # 30s 후 chain re-spawn (RAM 회수 시간)
    time.sleep(30)
    pid, log = spawn_chain()
    append_enforce_log(f"RESPAWN chain pid={pid} throttle={new_throttle}ms log={log}")
    return killed, new_throttle


def main():
    print(f"=== loop_resource_1min ENFORCER START {datetime.now().isoformat(timespec='seconds')} ===", flush=True)
    print(f"log: {LOG_JSONL}", flush=True)
    print(f"enforce_log: {ENFORCE_LOG}", flush=True)
    print(f"interval: 10s, SUSTAINED-3 enforce (학습 step peak 무시) max(GPU_util/GPU_mem/my_ram) < {LIMIT_PCT}%", flush=True)
    print(f"agent daemon 보존: {AGENT_KEEP}", flush=True)
    print(f"chain kill target: {CHAIN_KILL}", flush=True)
    if not THROTTLE_OVERRIDE.exists():
        write_throttle(THROTTLE_INIT_MS)
    n = 0
    over_streak = 0   # sustained over_limit counter (260517 23:12 patch)
    SUSTAIN_REQUIRED = 12   # 12 × 10s = 2min sustained (사용자 명시 260518 04:24)
    while True:
        try:
            rec = measure()
            with LOG_JSONL.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            n += 1
            if rec["over_limit"]:
                over_streak += 1
            else:
                over_streak = 0
            print(
                f"[{rec['ts_utc']}] iter={n} max={rec['max_pct']:>5.1f}% "
                f"gpu={rec['gpu_util_pct']:>3}% mem={rec['gpu_mem_pct']:>4.1f}% "
                f"my_ram={rec['my_ram_pct']:>4.1f}% cpu={rec['cpu_pct']:>4.1f}% "
                f"py={rec['py_count']:>3} over={rec['over_limit']} streak={over_streak}",
                flush=True,
            )
            if over_streak >= SUSTAIN_REQUIRED:
                enforce(rec)
                over_streak = 0
        except Exception as e:
            print(f"err iter {n}: {e}", flush=True)
        time.sleep(10)


if __name__ == "__main__":
    sys.exit(main())
