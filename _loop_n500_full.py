#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""n500 full chain — anchor skip, 합성 폴더(E:/data/images/unknown) 직접 학습.

A chain (_loop_30class_v2.py) 가 Stage D 완료 후 자동 시작.

Stage E. ablation 6 cell (B0/B1/B3/B4/B5/NEW) on 합성 폴더 직접
Stage F. tier1 결과 표 → docs/paper/RESULTS_30class_full_<ts>.md

학습 데이터: 15,000 wafer (30 class × 500 + Normal × 2000)
ETA: 1 cell ~3.7h × 6 ≈ 22h
"""
from __future__ import annotations
import json
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

REPO = Path(__file__).parent.resolve()
LOG_DIR = REPO / "_dispatch_logs"
LOG_DIR.mkdir(exist_ok=True)
TS = datetime.now().strftime("%y%m%d_%H%M%S")
MAIN_LOG = LOG_DIR / f"_loop_n500_full_{TS}.log"
STATUS_JSON = LOG_DIR / f"_loop_n500_full_{TS}_status.json"

E_DATA = "E:/data/images/unknown"
HEATMAP = "D:/project/known-cnn/dist_learn/_dist_heatmaps"
# backbone: 옛 260511 학습이 사용한 TAPT-init backbone (ImageNet FCMAE + wafer-domain finetune)
BACKBONE_CKPT = "C:/Users/hgcho/AppData/Local/Temp/contrastive_init_best_model.pth"

ENV_BASE = {
    "WAFER_PNG_OUT_DIR": E_DATA,
    "WAFER_JSON_OUT_DIR": "E:/data/positions/unknown",
    "CLASSIFICATION_CHIPS_ROOT": "E:/data/images/classification_chips",
    "HEATMAP_DIR": HEATMAP,
    "BACKBONE_CKPT": BACKBONE_CKPT,
    "PYTHONUNBUFFERED": "1",
    "PYTHONIOENCODING": "utf-8",
}

GPU_HIGH = 95       # 사용자 명시 260519 14:50: 95% 완화 (학습 안정성 우선)
GPU_LOW = 50        # 50% 미만이면 throttle--
POLL_SEC = 120
BATCH_INIT = 8
BATCH_MIN = 2    # BatchNorm1d 호환성

# A chain status — wait until Stage D done
A_STATUS_PATTERN = "_loop_v2_*_status.json"


def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with MAIN_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def status_set(stage: str, cell: str, state: str, **kw) -> None:
    rec = {"stage": stage, "cell": cell, "state": state,
           "ts": datetime.now().isoformat(timespec="seconds"), **kw}
    cur = {}
    if STATUS_JSON.exists():
        try:
            cur = json.loads(STATUS_JSON.read_text(encoding="utf-8"))
        except Exception:
            cur = {}
    cur.setdefault("history", []).append(rec)
    cur["current"] = rec
    STATUS_JSON.write_text(json.dumps(cur, indent=2, ensure_ascii=False),
                           encoding="utf-8")


def gpu_util() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5, creationflags=0x08000000,   # CREATE_NO_WINDOW
        ).strip().split("\n")[0]
        return int(out)
    except Exception:
        return 0


def resource_snapshot() -> dict:
    """GPU + 내 python 전용 RAM/CPU% 측정 (사용자 명시 260517 22:01).
    시스템 전체 RAM/CPU 아님 — Claude spawn (chain + agent) python.exe 만 합산."""
    my_ram_pct = 0.0
    my_cpu_pct = 0.0
    try:
        import psutil
        sys_total_mb = psutil.virtual_memory().total / (1024 * 1024)
        my_ram_mb = 0.0
        n_cpu = psutil.cpu_count() or 1
        my_cpu_raw = 0.0
        for p in psutil.process_iter(['name', 'memory_info', 'cpu_percent']):
            try:
                if p.info.get('name') == 'python.exe':
                    my_ram_mb += p.info['memory_info'].rss / (1024 * 1024)
                    my_cpu_raw += (p.info.get('cpu_percent') or 0.0)
            except Exception:
                pass
        my_ram_pct = (my_ram_mb / sys_total_mb) * 100 if sys_total_mb > 0 else 0
        my_cpu_pct = my_cpu_raw / n_cpu  # process cpu% / core count = system-relative
    except Exception:
        pass
    g = gpu_util()
    try:
        gmem = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5, creationflags=0x08000000,   # CREATE_NO_WINDOW
        ).strip().split(",")
        gmem_pct = int(gmem[0]) / int(gmem[1]) * 100
    except Exception:
        gmem_pct = 0.0
    return {"gpu_util": g, "gpu_mem_pct": gmem_pct,
            "ram_pct": my_ram_pct, "cpu_pct": my_cpu_pct,
            "max_pct": max(g, gmem_pct, my_ram_pct, my_cpu_pct)}


def wait_for_a_chain_done() -> None:
    """Poll A chain status json — wait until Stage D 'done'."""
    log("=== WAITING for A chain Stage D done ===")
    status_set("WAIT", "a_chain", "polling")
    last_print = 0
    while True:
        candidates = sorted(LOG_DIR.glob(A_STATUS_PATTERN),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            try:
                data = json.loads(candidates[0].read_text(encoding="utf-8"))
                cur = data.get("current", {})
                stage = cur.get("stage")
                cell = cur.get("cell")
                state = cur.get("state")
                if stage == "D" and state == "done":
                    log(f"  ✓ A chain Stage D done — proceed n500 full")
                    return
                now = time.time()
                if now - last_print > 600:  # 10 min progress
                    log(f"  A chain still on stage={stage} cell={cell} state={state}")
                    last_print = now
            except Exception as e:
                log(f"  parse err: {e}")
        time.sleep(POLL_SEC)


CELLS = [
    # B0_n500, B1_n500 — 이미 완료 (260521 09:00) tier1 산출 → skip
    ("B3_n500", {"use_local": "true", "use_queue": "true",
                 "local_weight": 0.5, "ignore_neg_sim": 1.0, "neco_weight": 0.0}),
    ("B4_n500", {"use_local": "true", "use_queue": "true",
                 "local_weight": 0.5, "ignore_neg_sim": 0.72, "neco_weight": 0.0}),
    ("B5_n500", {"use_local": "true", "use_queue": "true",
                 "local_weight": 0.5, "ignore_neg_sim": 0.72, "neco_weight": 0.2}),
    ("NEW_n500", {"use_local": "false", "use_queue": "true",
                  "local_weight": 0, "ignore_neg_sim": 0.72, "neco_weight": 0.2}),
]


def run_with_watchdog(cmd: list[str], stage_log: str, batch: int = BATCH_INIT) -> int:
    log_path = LOG_DIR / f"_loop_n500_full_{TS}_{stage_log}.log"
    env = os.environ.copy()
    env.update(ENV_BASE)
    env["BATCH"] = str(batch)
    # ★ GPU throttle (사용자 정책 strict 30%): enforcer 의 throttle override file 우선
    override_path = REPO / "_chain_throttle_override.txt"
    if override_path.exists():
        try:
            throttle_ms = float(override_path.read_text().strip())
            log(f"  ★ throttle override file: {throttle_ms}ms")
        except Exception:
            throttle_ms = float(env.get("GPU_THROTTLE_MS", 300))
    else:
        throttle_ms = float(env.get("GPU_THROTTLE_MS", 300))
    env["GPU_THROTTLE_MS"] = str(throttle_ms)
    low_count = 0  # sustained GPU<LOW iter counter for BATCH++
    log(f"  cmd: {' '.join(cmd)}")
    log(f"  log: {log_path}")
    log(f"  initial throttle: {throttle_ms}ms (target GPU {GPU_LOW}-{GPU_HIGH}%)")
    fp = log_path.open("ab")
    DETACHED_PROCESS = 0x00000008
    CREATE_NO_WINDOW = 0x08000000  # cmd 창 안 띄움 (사용자 정책)
    pflags = subprocess.CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS | CREATE_NO_WINDOW
    proc = subprocess.Popen(cmd, cwd=str(REPO), env=env,
                            stdout=fp, stderr=subprocess.STDOUT,
                            creationflags=pflags)
    # ★ 260520 watchdog kill 비활성화 — die 진짜 원인 추적 + 학습 안 멈추게
    # proc 끝까지 wait. timeout 없음.
    rc = proc.wait()
    fp.close()
    log(f"  rc={rc} BATCH={batch} throttle={throttle_ms}ms (no-watchdog)")
    return rc

    # ===== 아래 watchdog kill 로직 사용 안 함 (보존만) =====
    while True:
        try:
            rc = proc.wait(timeout=POLL_SEC)
            fp.close()
            log(f"  rc={rc} BATCH={batch} throttle={throttle_ms}ms")
            return rc
        except subprocess.TimeoutExpired:
            snap = resource_snapshot()
            u = snap["max_pct"]
            log(f"  watchdog: max={u:.0f}% (gpu_util={snap['gpu_util']}% "
                f"gpu_mem={snap['gpu_mem_pct']:.0f}% "
                f"ram={snap['ram_pct']:.0f}% cpu={snap['cpu_pct']:.0f}%) "
                f"BATCH={batch} throttle={throttle_ms}ms")
            if u > GPU_HIGH:
                if batch > BATCH_MIN:
                    log(f"  ⚠ GPU>{GPU_HIGH}% → kill + BATCH--")
                    proc.terminate()
                    try: proc.wait(timeout=10)
                    except subprocess.TimeoutExpired: proc.kill()
                    fp.close()
                    batch -= 1
                    env["BATCH"] = str(batch)
                    fp = log_path.open("ab")
                    fp.write(f"\n=== RESTART BATCH={batch} ===\n".encode())
                    proc = subprocess.Popen(cmd, cwd=str(REPO), env=env,
                                            stdout=fp, stderr=subprocess.STDOUT,
                                            creationflags=pflags)
                elif throttle_ms < 500:  # cap 500ms
                    throttle_ms = min(500, throttle_ms + 20)
                    log(f"  ⚠ GPU>{GPU_HIGH}% BATCH=MIN → throttle++={throttle_ms}ms (kill+restart)")
                    proc.terminate()
                    try: proc.wait(timeout=10)
                    except subprocess.TimeoutExpired: proc.kill()
                    fp.close()
                    env["GPU_THROTTLE_MS"] = str(throttle_ms)
                    fp = log_path.open("ab")
                    fp.write(f"\n=== RESTART throttle={throttle_ms}ms ===\n".encode())
                    proc = subprocess.Popen(cmd, cwd=str(REPO), env=env,
                                            stdout=fp, stderr=subprocess.STDOUT,
                                            creationflags=pflags)
            elif u < GPU_LOW:
                low_count += 1
                if throttle_ms >= 10:
                    # GPU < LOW → throttle-- (속도 ↑)
                    throttle_ms = max(0, throttle_ms - 10)
                    log(f"  ↑ GPU<{GPU_LOW}% → throttle--={throttle_ms}ms (next restart)")
                    env["GPU_THROTTLE_MS"] = str(throttle_ms)
                elif low_count >= 2 and batch < BATCH_INIT:
                    # throttle 0 도달 + 2min sustained → BATCH++ (사용자 정책 "모자라면 더")
                    log(f"  ↑ GPU<{GPU_LOW}% sustained → BATCH++={batch+1} (kill+restart)")
                    proc.terminate()
                    try: proc.wait(timeout=10)
                    except subprocess.TimeoutExpired: proc.kill()
                    fp.close()
                    batch += 1
                    env["BATCH"] = str(batch)
                    low_count = 0
                    fp = log_path.open("ab")
                    fp.write(f"\n=== RESTART BATCH={batch} (BATCH++) ===\n".encode())
                    proc = subprocess.Popen(cmd, cwd=str(REPO), env=env,
                                            stdout=fp, stderr=subprocess.STDOUT,
                                            creationflags=pflags)
            else:
                low_count = 0  # in target range, reset sustained counter


def stage_e_ablation_n500() -> None:
    log("=== STAGE E: ablation 6 cell on 합성 폴더 직접 (n500 full) ===")
    for tag, cfg in CELLS:
        log(f"--- cell {tag} ---")
        status_set("E", tag, "start", cfg=cfg)
        cmd = [sys.executable, "-u", "_dispatch_iter.py",
               "--tag", tag, "--data", E_DATA,
               "--epochs", "5", "--batch", str(BATCH_INIT),
               "--image-size", "384", "--warmup", "1", "--seed", "42",
               "--mcs", "12", "--ms", "3",
               "--cluster-method", "eom", "--cluster-eps", "0",
               "--freeze-backbone", "true",
               "--nce-temp", "0.07", "--lr-head", "1e-3",
               "--local-pos-topk", "12", "--queue-size", "4096",
               "--use-local", cfg["use_local"],
               "--use-queue", cfg["use_queue"],
               "--local-weight", str(cfg["local_weight"]),
               "--ignore-neg-sim", str(cfg["ignore_neg_sim"]),
               "--neco-weight", str(cfg["neco_weight"]),
               # ★ n500 protocol: defect 500 + Normal 2000 (사용자 명시 260517 21:01)
               "--per-class-cap", "500",
               "--normal-cap", "2000",
               # ★ 70% 완화 (260517 23:01): throttle 30ms 시작
               "--wait", "true",
               "--gpu-throttle-ms", "30"]
        rc = run_with_watchdog(cmd, stage_log=f"E_{tag}", batch=BATCH_INIT)
        status_set("E", tag, "done", rc=rc)


def stage_f_table() -> None:
    log("=== STAGE F: tier1 results → markdown table (n500 full) ===")
    metric_keys = ["completeness", "homogeneity", "ami", "nmi", "ari",
                   "silhouette_cosine", "noise_pct", "class_capture_rate",
                   "n_clusters"]
    rows = []
    for tag, cfg in CELLS:
        candidates = sorted(REPO.glob(f"outputs_contrastive_*/tier1_{tag}.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            rows.append({"tag": tag, "cfg": cfg, "status": "MISSING"})
            continue
        try:
            data = json.loads(candidates[0].read_text(encoding="utf-8"))
        except Exception as e:
            rows.append({"tag": tag, "cfg": cfg, "status": f"PARSE_ERR {e}"})
            continue
        if isinstance(data, list):
            dicts = [x for x in data if isinstance(x, dict)]
            data = dicts[0] if dicts else {}
        if not isinstance(data, dict):
            rows.append({"tag": tag, "cfg": cfg, "status": f"NOT_DICT type={type(data).__name__}"})
            continue
        flat = {}
        for k in metric_keys:
            for cand in [k, k.upper(), k.replace("_", " "),
                         f"defect_only.{k}", f"all.{k}"]:
                if "." in cand:
                    a, b = cand.split(".", 1)
                    val_a = data.get(a)
                    if isinstance(val_a, dict) and b in val_a:
                        flat[k] = val_a[b]
                        break
                if cand in data:
                    flat[k] = data[cand]
                    break
        rows.append({"tag": tag, "cfg": cfg, "status": "OK",
                     "src": str(candidates[0]), "metrics": flat})

    md = ["# 30-class n500 full ablation table (autogen)", "",
          f"timestamp: {datetime.now().isoformat(timespec='seconds')}",
          f"anchor: SKIPPED — 학습 input = {E_DATA} 직접 (~15,000 wafer)", "",
          "| cell | Local | LW | Queue | NEG | NeCo | "
          "ARI | Comp | Hom | AMI | noise % | cap | n_cl |",
          "|---|:-:|:-:|:-:|:-:|:-:|"
          "---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        c = r["cfg"]
        m = r.get("metrics", {})
        def fm(k, fmt="{:.4f}"):
            v = m.get(k)
            return fmt.format(v) if isinstance(v, (int, float)) else "—"
        md.append(
            f"| {r['tag']} | {c['use_local']} | {c['local_weight']} | "
            f"{c['use_queue']} | {c['ignore_neg_sim']} | {c['neco_weight']} | "
            f"{fm('ari')} | {fm('completeness')} | {fm('homogeneity')} | "
            f"{fm('ami')} | {fm('noise_pct', '{:.2f}')} | "
            f"{fm('class_capture_rate', '{:.4f}')} | "
            f"{fm('n_clusters', '{:.0f}')} |"
        )

    out_path = REPO / "docs" / "paper" / f"RESULTS_30class_full_{TS}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    log(f"  table written: {out_path}")
    status_set("F", "table", "done", out=str(out_path), rows=rows)


def main() -> int:
    log(f"=== LOOP n500 FULL START TS={TS} ===")
    log(f"main: {MAIN_LOG}")
    log(f"status: {STATUS_JSON}")
    log(f"data: {E_DATA} (anchor skip)")
    # wait_for_a_chain_done() — A chain 정리됨 (260519 cleanup). 직접 skip.
    log("[skip] A chain wait — 데이터 이미 E:/data/images/unknown 준비")
    stage_e_ablation_n500()
    stage_f_table()
    log("=== LOOP n500 FULL DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
