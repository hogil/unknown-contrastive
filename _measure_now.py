"""Snapshot — unknown-contrastive 프로젝트의 현재 CPU / RAM / GPU 사용량."""
import psutil
import subprocess
import time
import datetime
from pathlib import Path

REPO = str(Path("D:/project/unknown-contrastive")).replace("\\", "/").lower()
PROJECT_KEYS = [
    "run_contrastive", "contrastive.py", "_loop_resource_1min",
    "_dispatch_iter", "_loop_n500_full", "_loop_master", "_loop_analyzer",
    "_loop_recorder", "_paper_recorder", "cluster_analyzer",
    "_measure_now.py",  # self
]

n_cpu = psutil.cpu_count(logical=True)
ram_total = psutil.virtual_memory().total / 1024 / 1024

ours = []
for p in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
    try:
        if p.info.get("name") != "python.exe":
            continue
        cmd = " ".join(p.info.get("cmdline") or [])
        cmd_l = cmd.lower().replace("\\", "/")
        if REPO in cmd_l or any(k in cmd for k in PROJECT_KEYS):
            p.cpu_percent(None)
            ours.append((p, cmd))
    except Exception:
        pass

time.sleep(1.0)

print(f"=== unknown-contrastive project process only ===")
print(f"cores={n_cpu}, RAM total={ram_total/1024:.1f} GB\n")

total_cpu = 0
total_ram = 0
print(f"{'PID':>6} {'CPU%raw':>8} {'CPU%sys':>8} {'RAM_MB':>9} {'create':>9}  cmdline")
print(f"{'-'*6} {'-'*8} {'-'*8} {'-'*9} {'-'*9}  {'-'*60}")
for p, cmd in ours:
    try:
        cpu = p.cpu_percent(None)
        mb = p.memory_info().rss / 1024 / 1024
        ct = datetime.datetime.fromtimestamp(p.create_time()).strftime("%H:%M:%S")
        total_cpu += cpu
        total_ram += mb
        cmd_short = cmd[:80]
        print(f"{p.pid:>6} {cpu:>7.1f}% {cpu/n_cpu:>7.1f}% {mb:>8.1f} {ct:>9}  {cmd_short}")
    except Exception:
        pass

print()
print(f"=== PROJECT TOTAL (unknown-contrastive only) ===")
print(f"CPU sum:  {total_cpu:>6.1f}% raw  =  {total_cpu/n_cpu:>5.1f}% of {n_cpu}-core system")
print(f"RAM sum:  {total_ram:>8.1f} MB  =  {total_ram/ram_total*100:>5.2f}% of {ram_total/1024:.1f} GB")

# GPU mem (compute-apps) by PID
print()
print("=== GPU mem (project PIDs only) ===")
out = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
     "--format=csv,noheader,nounits"],
    text=True, timeout=5, creationflags=0x08000000,
).strip()
our_pids = set(p.pid for p, _ in ours)
total_gpu = 0
for line in out.split("\n"):
    parts = [x.strip() for x in line.split(",")]
    if len(parts) >= 2:
        try:
            pid = int(parts[0])
            mb = int(parts[1])
            if pid in our_pids:
                print(f"  PID={pid} mem={mb} MB")
                total_gpu += mb
        except Exception:
            pass
print(f"GPU mem sum (project): {total_gpu} MB")

# system GPU
gpu_used, gpu_total = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=memory.used,memory.total",
     "--format=csv,noheader,nounits"],
    text=True, creationflags=0x08000000,
).strip().split(",")
gpu_used = int(gpu_used.strip())
gpu_total = int(gpu_total.strip())
print()
print(f"=== System GPU (reference) ===")
print(f"system used: {gpu_used} MB / {gpu_total} MB ({gpu_used/gpu_total*100:.1f}%)")
print(f"  project share: {total_gpu} MB ({total_gpu/gpu_total*100:.1f}% of GPU)")
print(f"  other (자매 chip_multilabel 등): {gpu_used - total_gpu} MB ({(gpu_used - total_gpu)/gpu_total*100:.1f}%)")
