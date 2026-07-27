#!/usr/bin/env python3
"""Lightweight resource guard for paper contrastive experiment loops."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "result_grouping" / "paper_contrastive_supervisor_260617"


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run(cmd: list[str], timeout: int = 20) -> str:
    try:
        p = subprocess.run(cmd, cwd=REPO, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout)
        return p.stdout.strip()
    except Exception as exc:
        return f"[ERR] {exc}"


def process_json() -> str:
    ps = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -match '_paper_contrastive|run_paper_contrastive_grid|_ssl_methods|_score_umapfree|export_best_groupings|paper_resource_monitor|paper_contrastive_supervisor' } | "
        "Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Compress"
    )
    return run(["powershell", "-NoProfile", "-Command", ps])


def gpu_csv() -> str:
    return run([
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ])


def compute_csv() -> str:
    return run([
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader",
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--min-free-gb", type=float, default=30.0)
    ap.add_argument("--critical-free-gb", type=float, default=20.0)
    ap.add_argument("--max-vram-pct", type=float, default=92.0)
    ap.add_argument("--max-temp-c", type=float, default=82.0)
    ap.add_argument("--max-ckpt-gb", type=float, default=80.0)
    ap.add_argument("--guard-file", default=str(OUT / "STOP_RESOURCE_GUARD.txt"))
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "resource_monitor.log"
    status = OUT / "resource_status.json"
    guard = Path(args.guard_file)
    while True:
        disk = shutil.disk_usage(REPO.anchor or "D:\\")
        free_gb = disk.free / (1024 ** 3)
        total_gb = disk.total / (1024 ** 3)
        gpu = gpu_csv()
        compute = compute_csv()
        ckpts = list((REPO / "result_grouping").glob("paper_contrastive*_260617/**/*.pt"))
        ckpt_gb = sum(p.stat().st_size for p in ckpts if p.exists()) / (1024 ** 3)
        warning = []
        critical = []
        if free_gb < args.min_free_gb:
            warning.append(f"LOW_DISK free_gb={free_gb:.1f}")
        if free_gb < args.critical_free_gb:
            critical.append(f"CRITICAL_DISK free_gb={free_gb:.1f}")
        if ckpt_gb > args.max_ckpt_gb:
            warning.append(f"HIGH_CKPT_BYTES ckpt_gb={ckpt_gb:.1f}")
        try:
            used, total, util, temp = [float(x.strip()) for x in gpu.splitlines()[0].split(",")[:4]]
            if total > 0 and (used / total * 100.0) > args.max_vram_pct:
                warning.append(f"HIGH_VRAM used={used:.0f}/{total:.0f}MB")
            if temp > args.max_temp_c:
                warning.append(f"HIGH_TEMP temp_c={temp:.0f}")
                critical.append(f"CRITICAL_TEMP temp_c={temp:.0f}")
        except Exception:
            pass
        if critical:
            guard.parent.mkdir(parents=True, exist_ok=True)
            guard.write_text(
                f"{now()} resource guard active: {'; '.join(critical)}\n",
                encoding="utf-8",
            )
        elif guard.exists():
            try:
                guard.unlink()
            except OSError:
                pass
        payload = {
            "updated": now(),
            "repo": str(REPO),
            "disk_free_gb": round(free_gb, 2),
            "disk_total_gb": round(total_gb, 2),
            "checkpoint_count": len(ckpts),
            "checkpoint_gb": round(ckpt_gb, 2),
            "gpu": gpu,
            "compute_apps": compute,
            "processes": process_json(),
            "warnings": warning,
            "critical": critical,
            "guard_file": str(guard.resolve()),
            "guard_active": guard.exists(),
        }
        status.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        with log.open("a", encoding="utf-8", errors="replace") as f:
            f.write(
                f"[{now()}] disk_free={free_gb:.1f}GB ckpt={ckpt_gb:.1f}GB "
                f"gpu=`{gpu}` warnings={';'.join(warning) or '-'} critical={';'.join(critical) or '-'}\n"
            )
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
