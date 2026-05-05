import json, subprocess, shutil, sys, time, os
import psutil

PID = 2396
INTERVAL = 60
MAX_MIN = 30
RAM_WARN = 70.0
GPU_WARN = 80.0
RAM_ABORT = 80.0
GPU_ABORT = 90.0

def measure():
    ram = psutil.virtual_memory().percent
    cpu = psutil.cpu_percent(interval=1)
    gpu = None
    if shutil.which('nvidia-smi'):
        try:
            out = subprocess.check_output(
                ['nvidia-smi','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits'],
                timeout=5, stderr=subprocess.DEVNULL).decode().strip().splitlines()
            if out:
                u, t = (int(x) for x in out[0].split(','))
                gpu = round(100.0*u/max(1,t), 2)
        except Exception:
            pass
    return ram, cpu, gpu

def find_rogue_canvas():
    rogues = []
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            cl = p.info.get('cmdline') or []
            if any('sample_canvas_gen' in str(x) for x in cl):
                rogues.append(p.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return rogues

def pid_alive(pid):
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False

warnings = 0
aborts = 0
rogue_kills = 0
end_status = "NORMAL_END"

start = time.time()
deadline = start + MAX_MIN * 60

# initial measurement immediately
iteration = 0
while True:
    iteration += 1
    if not pid_alive(PID):
        print("TRAINING_ENDED", flush=True)
        end_status = "NORMAL_END"
        break

    rogues = find_rogue_canvas()
    for rpid in rogues:
        if rpid == PID:
            continue
        print(f"ROGUE_CANVAS_GEN pid={rpid}", flush=True)
        try:
            psutil.Process(rpid).kill()
            rogue_kills += 1
        except Exception as e:
            print(f"ROGUE_KILL_FAIL pid={rpid} err={e}", flush=True)

    ram, cpu, gpu = measure()
    gpu_v = gpu if gpu is not None else 0.0
    if ram >= RAM_ABORT or gpu_v >= GPU_ABORT:
        print(f"ABORT_RESOURCE ram={ram}% gpu={gpu}% pid={PID}", flush=True)
        aborts += 1
        end_status = "ABORT_TRIGGERED"
        break
    elif ram > RAM_WARN or gpu_v > GPU_WARN:
        print(f"WARN ram={ram}% gpu={gpu}% pid={PID}", flush=True)
        warnings += 1
    else:
        print(f"ok ram={ram}% gpu={gpu}% pid={PID}", flush=True)

    if time.time() >= deadline:
        end_status = "TIMEOUT"
        break

    # sleep until next interval, but check deadline
    sleep_left = INTERVAL
    while sleep_left > 0:
        chunk = min(5, sleep_left)
        time.sleep(chunk)
        sleep_left -= chunk
        if time.time() >= deadline:
            break

summary = {
    "warnings": warnings,
    "aborts": aborts,
    "rogue_kills": rogue_kills,
    "end_status": end_status,
    "iterations": iteration,
    "elapsed_sec": int(time.time() - start),
}
print("SUMMARY " + json.dumps(summary), flush=True)
