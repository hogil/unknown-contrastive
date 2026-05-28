#!/usr/bin/env python3
"""unknown-contrastive 프로젝트 전용 resource monitor daemon.

사용자 directive (260528): 이 프로젝트의 python process 가
  CPU > 30% (시스템 전체 대비)
  RAM > 30% (시스템 total 대비)
  GPU mem > 40% (시스템 total 대비, 우리 process 가 GPU 쓸 때만)
초과 시 우리 process 전부 kill.

실행 시 1회 점검 + 60s 주기 watchdog. 다른 프로젝트(자매 known-cnn 등) 는 절대 안 건드림.

usage:
    python scripts/_resource_monitor.py            # 60s 주기
    python scripts/_resource_monitor.py --interval 30
"""
from __future__ import annotations
import argparse, os, sys, time, subprocess
from pathlib import Path

try:
    import psutil
except ImportError:
    print("psutil 필요: pip install psutil", file=sys.stderr); sys.exit(2)

# === LIMITS (사용자 명시) ===
CPU_LIMIT_PCT = 30.0
RAM_LIMIT_PCT = 30.0
GPU_LIMIT_PCT = 40.0

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_CWD_LOWER = str(PROJECT_ROOT).replace("\\", "/").lower()


def our_procs() -> list[psutil.Process]:
    """cwd 가 unknown-contrastive 인 python process 만 (monitor 자기 자신 + 자매 repo 제외)."""
    self_pid = os.getpid()
    out = []
    for p in psutil.process_iter(["pid", "name"]):
        try:
            if p.info["pid"] == self_pid:                                              # ★ 자기 자신 제외
                continue
            if p.info["name"] not in ("python.exe", "python"):
                continue
            try:
                cwd = p.cwd().replace("\\", "/").lower()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue
            if PROJECT_CWD_LOWER in cwd or "unknown-contrastive" in cwd:
                out.append(psutil.Process(p.info["pid"]))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return out


def gpu_mem_pct(our_pids: set) -> float:
    """우리 process 의 GPU mem 합계 / 시스템 GPU total. 우리가 CUDA 안 쓰면 0."""
    if not our_pids:
        return 0.0
    try:
        r1 = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        total = int(r1.stdout.strip().split("\n")[0])
        r2 = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        used = 0
        for line in r2.stdout.strip().split("\n"):
            if not line:
                continue
            parts = [x.strip() for x in line.split(",")]
            try:
                pid = int(parts[0])
                mem = parts[1]
                if pid in our_pids and mem not in ("[N/A]", "N/A"):
                    used += int(mem)
            except (ValueError, IndexError):
                continue
        return 100.0 * used / max(1, total)
    except Exception:
        return 0.0


def measure(procs: list[psutil.Process]) -> tuple[float, float, float, int]:
    """우리 procs 의 cpu%(시스템대비), ram%(시스템대비), gpu%(시스템 전체), n."""
    # cpu_percent(interval=None) 은 직전 호출 이후 누적률 — 미리 baseline 필요
    cpus = []
    for p in procs:
        try:
            cpus.append(p.cpu_percent(interval=None))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    cpu_total_pct = sum(cpus) / max(1, psutil.cpu_count())
    rss = 0
    for p in procs:
        try:
            rss += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    ram_pct = 100.0 * rss / psutil.virtual_memory().total
    our_pids = {p.pid for p in procs}
    gpu_pct = gpu_mem_pct(our_pids)
    return cpu_total_pct, ram_pct, gpu_pct, len(procs)


def kill_ours(procs: list[psutil.Process]) -> None:
    for p in procs:
        try:
            p.kill()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--once", action="store_true", help="1회만 측정 후 종료")
    args = ap.parse_args()

    print(f"[mon] start  CPU<{CPU_LIMIT_PCT:.0f}%  RAM<{RAM_LIMIT_PCT:.0f}%  "
          f"GPU<{GPU_LIMIT_PCT:.0f}%  project={PROJECT_CWD_LOWER}", flush=True)

    # baseline cpu_percent 초기화 (첫 호출은 0 반환, 두 번째부터 실제값)
    procs = our_procs()
    for p in procs:
        try:
            p.cpu_percent(interval=None)
        except Exception:
            pass

    # 시작 시 1회 점검 (1s 대기 후 측정 — baseline 후 변화량 확보)
    time.sleep(1.0)
    procs = our_procs()
    for p in procs:
        try:
            p.cpu_percent(interval=None)
        except Exception:
            pass
    cpu, ram, gpu, n = measure(procs)
    ts = time.strftime("%H:%M:%S")
    print(f"[mon {ts}] start-check n={n} cpu={cpu:.1f}% ram={ram:.1f}% gpu={gpu:.1f}%",
          flush=True)

    if args.once:
        return

    while True:
        time.sleep(args.interval)
        procs = our_procs()
        # baseline 후 1s 측정
        for p in procs:
            try:
                p.cpu_percent(interval=None)
            except Exception:
                pass
        time.sleep(1.0)
        cpu, ram, gpu, n = measure(procs)
        ts = time.strftime("%H:%M:%S")
        print(f"[mon {ts}] n={n} cpu={cpu:.1f}% ram={ram:.1f}% gpu={gpu:.1f}%",
              flush=True)
        if cpu > CPU_LIMIT_PCT or ram > RAM_LIMIT_PCT or gpu > GPU_LIMIT_PCT:
            print(f"[mon {ts}] !! LIMIT EXCEEDED -> killing {n} our procs", flush=True)
            kill_ours(procs)


if __name__ == "__main__":
    main()
