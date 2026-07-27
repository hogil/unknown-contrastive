#!/usr/bin/env python3
"""1-minute resource ENFORCER v2 — my_* 기준 (260526 사용자 명시).

사용자 명시 (260526):
- 한계: my_gpu_mem 40% / my_ram 20% / my_cpu 20%
- 측정 기준: 우리 process 만 (system 전체 X)
- 초과 시: chain kill (re-dispatch 는 사용자 결정 시 별도 spawn)

v1 (_loop_resource_1min.py) 와 차이:
- v1: system gpu_util / gpu_mem / my_ram / system cpu, LIMIT 95%
- v2: my_gpu_mem / my_ram / my_cpu, LIMITS dict per-metric
- v2: enforce 는 kill 만 (auto re-dispatch X)

agent daemon 보존 (cmdline 기반):
- _loop_resource_1min*.py (self)
- _loop_master*.py / _loop_analyzer*.py / _loop_recorder*.py
- _paper_recorder*.py / cluster_analyzer*.py

kill 대상 (chain only):
- _loop_n500_full.py / _dispatch_iter.py / run_contrastive.py
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
LOG_JSONL = REPO / "_loop_resource_1min_v2.log"
ENFORCE_LOG = REPO / "_enforcer_actions_v2.log"

# Per-metric 한계 (사용자 명시 260526 — 두 번째 결정 후 확정).
#
# 사용자 명시:
#   "학습 process 본체 = system 45.8% 그냥 이거 돌리는 만큼 하자.
#    메모리 gpu 잘 지키고 왜냐면 오류나서 움직이지 않을때도 있어서"
#
# 결정:
#   - my_cpu_pct: enforce 제외 (torch MKL multi-thread 가 자연 사용)
#   - my_gpu_mem_pct, my_ram_pct: enforce (오류로 hang 시 메모리 leak 방지)
LIMITS = {
    "my_gpu_mem_pct": 40.0,
    "my_ram_pct":     20.0,
    # my_cpu_pct: 측정만, enforce 제외 (사용자 명시 260526)
}
INTERVAL_SEC = 10
SUSTAIN_REQUIRED = 12   # 12 × 10s = 2min sustained

AGENT_KEEP = [
    "_loop_resource_1min",
    "_loop_master", "_loop_analyzer", "_loop_recorder",
    "_paper_recorder", "cluster_analyzer",
]
CHAIN_KILL = [
    "_loop_n500_full.py",
    "_dispatch_iter.py",
    "run_contrastive.py",
]


def gpu_compute_apps():
    """nvidia-smi --query-compute-apps → {pid: used_memory_mb}."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5, creationflags=0x08000000,
        ).strip()
    except Exception:
        return {}
    pid_mem = {}
    for line in out.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                pid_mem[int(parts[0])] = int(parts[1])
            except ValueError:
                pass
    return pid_mem


def gpu_total():
    """nvidia-smi → total VRAM (MB)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5, creationflags=0x08000000,
        ).strip().split("\n")[0]
        return int(out)
    except Exception:
        return 16380


def is_our_python(p):
    """psutil.Process → True if cmdline contains repo path or our scripts."""
    try:
        cmd = ' '.join(p.cmdline() or [])
    except Exception:
        return False
    if str(REPO).replace("\\", "/").lower() in cmd.replace("\\", "/").lower():
        return True
    for k in AGENT_KEEP + CHAIN_KILL:
        if k in cmd:
            return True
    return False


def measure():
    """우리 process 만 측정 — my_gpu_mem / my_ram / my_cpu."""
    import psutil

    ram_total_mb = psutil.virtual_memory().total / (1024 * 1024)
    gpu_total_mb = gpu_total()
    pid_gpu = gpu_compute_apps()

    my_pids = []
    my_ram_mb = 0.0
    my_gpu_mb = 0
    my_cpu_pct = 0.0

    # 1차 pass — our python.exe 식별, cpu_percent 초기화
    procs = []
    for p in psutil.process_iter(['pid', 'name', 'memory_info', 'cmdline']):
        try:
            if p.info.get('name') != 'python.exe':
                continue
            if not is_our_python(p):
                continue
            procs.append(p)
            p.cpu_percent(None)  # 1st call → returns 0, initializes
        except Exception:
            continue

    # CPU sampling window
    time.sleep(0.5)

    # 2차 pass — RAM/GPU/CPU 합산
    for p in procs:
        try:
            pid = p.pid
            my_pids.append(pid)
            my_ram_mb += p.memory_info().rss / (1024 * 1024)
            if pid in pid_gpu:
                my_gpu_mb += pid_gpu[pid]
            my_cpu_pct += p.cpu_percent(None)
        except Exception:
            continue

    # WDDM fallback — Windows consumer GPU 는 process 별 mem query 안 되므로
    # nvidia-smi --query-compute-apps 가 0 리턴. system gpu mem 으로 대체
    # (자매 known-cnn 학습 안 도는 가정 — 자매 동시 학습 시 분리 측정 불가)
    if my_gpu_mb == 0 and my_pids:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=5, creationflags=0x08000000,
            ).strip()
            my_gpu_mb = int(out.split("\n")[0])
        except Exception:
            pass

    n_cpu = psutil.cpu_count(logical=True) or 1
    my_cpu_pct_norm = my_cpu_pct / n_cpu  # normalize to 0-100 across all cores

    my_gpu_mem_pct = (my_gpu_mb / gpu_total_mb * 100) if gpu_total_mb > 0 else 0
    my_ram_pct = (my_ram_mb / ram_total_mb * 100) if ram_total_mb > 0 else 0

    rec = {
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec='seconds'),
        "n_our_py": len(my_pids),
        "my_gpu_mem_mb": int(my_gpu_mb),
        "gpu_total_mb": int(gpu_total_mb),
        "my_gpu_mem_pct": round(my_gpu_mem_pct, 2),
        "my_ram_mb": round(my_ram_mb, 1),
        "ram_total_mb": int(ram_total_mb),
        "my_ram_pct": round(my_ram_pct, 2),
        "my_cpu_pct": round(my_cpu_pct_norm, 1),
        "n_cpu": n_cpu,
    }

    # over_limit per metric
    over_metrics = []
    for k, lim in LIMITS.items():
        if rec.get(k, 0) > lim:
            over_metrics.append(f"{k}={rec[k]:.1f}>{lim}")
    rec["over_metrics"] = over_metrics
    rec["over_limit"] = len(over_metrics) > 0
    return rec


def classify_pythons():
    import psutil
    chain = []
    agent = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
        try:
            if p.info.get('name') != 'python.exe':
                continue
            cmd = ' '.join(p.info.get('cmdline') or [])
            mb = p.info['memory_info'].rss / (1024 * 1024)
            row = {"pid": p.info['pid'], "ram_mb": round(mb, 1), "cmd": cmd[:120]}
            if any(k in cmd for k in AGENT_KEEP):
                agent.append(row)
            elif any(k in cmd for k in CHAIN_KILL):
                chain.append(row)
        except Exception:
            pass
    return chain, agent


def append_enforce_log(msg):
    ts = datetime.now(timezone.utc).isoformat(timespec='seconds')
    with ENFORCE_LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


def kill_chain_only():
    import psutil
    chain, agent = classify_pythons()
    killed = []
    for row in chain:
        try:
            p = psutil.Process(row['pid'])
            p.kill()
            killed.append(row['pid'])
        except Exception:
            pass
    return killed, len(agent), len(chain)


def enforce(rec):
    """over_limit sustained → chain kill (re-dispatch 사용자 결정 시 별도 spawn)."""
    killed, n_agent, n_chain = kill_chain_only()
    msg = (
        f"OVER_LIMIT {rec['over_metrics']} — "
        f"killed chain pids={killed} (kept {n_agent} agents)"
    )
    append_enforce_log(msg)
    print(f"  ENFORCE: {msg}", flush=True)
    return killed


def main():
    print(f"=== loop_resource_1min v2 START {datetime.now().isoformat(timespec='seconds')} ===", flush=True)
    print(f"log: {LOG_JSONL}", flush=True)
    print(f"enforce_log: {ENFORCE_LOG}", flush=True)
    print(f"interval: {INTERVAL_SEC}s, SUSTAINED-{SUSTAIN_REQUIRED} ({SUSTAIN_REQUIRED*INTERVAL_SEC}s) sustained", flush=True)
    print(f"LIMITS (my_* only): {LIMITS}", flush=True)
    print(f"agent KEEP: {AGENT_KEEP}", flush=True)
    print(f"chain KILL: {CHAIN_KILL}", flush=True)

    n = 0
    over_streak = 0
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
                f"[{rec['ts_utc']}] iter={n} "
                f"gpu={rec['my_gpu_mem_pct']:>5.1f}% "
                f"ram={rec['my_ram_pct']:>4.1f}% "
                f"cpu={rec['my_cpu_pct']:>5.1f}% "
                f"n_py={rec['n_our_py']} "
                f"over={rec['over_limit']} streak={over_streak} "
                f"({','.join(rec['over_metrics']) if rec['over_metrics'] else 'OK'})",
                flush=True,
            )
            if over_streak >= SUSTAIN_REQUIRED:
                enforce(rec)
                over_streak = 0
        except Exception as e:
            print(f"err iter {n}: {e}", flush=True)
        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    sys.exit(main())
