#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resource guard -- 학습/생성 스크립트 시작 전 자동 점검 + 학습 중 모니터링.

규칙:
- RAM 80% 초과: 시작 거부 / 학습 중에는 즉시 abort (메모리는 OOM 직격)
- GPU memory 90% 초과: cuda 사용 거부 -> CPU fallback
- CPU 90% 초과: 경고만 (학습 자체가 CPU 올림)
- GPU 사용 가능하지만 다른 프로세스가 점유 중이면 그대로 사용 (PyTorch가 알아서 분배)

사용:
    from _resource_guard import assess_start, ResourceMonitor

    a = assess_start()
    if not a["ok"]: sys.exit(1)
    device = torch.device(a["device"])

    monitor = ResourceMonitor()
    monitor.start()
    for ep in ...:
        if monitor.should_abort():
            # save partial, break
            break
    monitor.stop()
"""
from __future__ import annotations
import os, sys, time, threading, subprocess, shutil
from typing import Optional, List, Dict

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore

# ---- thresholds (percent) ----
RAM_LIMIT       = 80.0   # 시작 차단 + 학습 중 abort
GPU_MEM_LIMIT   = 90.0   # 시작 시 cuda -> cpu fallback
CPU_LIMIT       = 90.0   # 경고만
MONITOR_INTERVAL_SEC = 30


def _gpu_mem_pct() -> Optional[float]:
    """nvidia-smi로 GPU memory used / total %. 호출 실패 시 None."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=5, stderr=subprocess.DEVNULL,
        ).decode("ascii", errors="ignore").strip().splitlines()
        if not out:
            return None
        used_mb, total_mb = (int(x) for x in out[0].split(","))
        return 100.0 * used_mb / max(1, total_mb)
    except Exception:
        return None


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore
        return torch.cuda.is_available()
    except Exception:
        return False


def assess_start(
    ram_limit: float = RAM_LIMIT,
    gpu_mem_limit: float = GPU_MEM_LIMIT,
    cpu_limit: float = CPU_LIMIT,
    require_gpu: bool = False,
) -> Dict:
    """
    시작 가능 여부 + 권장 device 판단. dict:
      {ok: bool, device: 'cuda'|'cpu', reasons: [str], snapshot: {...}}
    """
    reasons: List[str] = []
    snapshot: Dict[str, float] = {}

    if psutil is not None:
        ram_pct = float(psutil.virtual_memory().percent)
        cpu_pct = float(psutil.cpu_percent(interval=0.5))
        snapshot.update({"ram_pct": ram_pct, "cpu_pct": cpu_pct})
        if ram_pct >= ram_limit:
            reasons.append(f"RAM {ram_pct:.1f}% >= {ram_limit:.0f}% (한계 초과 -- 시작 차단)")
        if cpu_pct >= cpu_limit:
            # 경고만, 차단 아님
            reasons.append(f"[warn] CPU {cpu_pct:.1f}% (학습 중 더 올라갈 것)")
    else:
        reasons.append("[warn] psutil 미설치 -- RAM/CPU 미점검")

    gpu_pct = _gpu_mem_pct()
    if gpu_pct is not None:
        snapshot["gpu_mem_pct"] = gpu_pct

    has_cuda = _cuda_available()
    snapshot["cuda_available"] = has_cuda

    # device 결정
    if not has_cuda:
        device = "cpu"
        if require_gpu:
            reasons.append("CUDA 사용 불가인데 require_gpu=True -- 시작 차단")
    elif gpu_pct is not None and gpu_pct >= gpu_mem_limit:
        device = "cpu"
        reasons.append(f"GPU mem {gpu_pct:.1f}% >= {gpu_mem_limit:.0f}% -- CPU fallback")
        if require_gpu:
            reasons.append("require_gpu=True -- 시작 차단")
    else:
        device = "cuda"

    # ok 판정 -- 차단 사유 (ram 한계 / require_gpu 실패) 있는지
    ok = True
    for r in reasons:
        if r.startswith("[warn]"):
            continue
        if "시작 차단" in r:
            ok = False
            break

    return {"ok": ok, "device": device, "reasons": reasons, "snapshot": snapshot}


def format_assessment(a: Dict) -> str:
    s = a["snapshot"]
    lines = [
        f"[guard] device={a['device']} ok={a['ok']}",
        f"  RAM={s.get('ram_pct', '?')}% / CPU={s.get('cpu_pct', '?')}% / GPU_mem={s.get('gpu_mem_pct', '?')}% / cuda={s.get('cuda_available', '?')}",
    ]
    for r in a["reasons"]:
        lines.append(f"  - {r}")
    return "\n".join(lines)


class ResourceMonitor:
    """
    백그라운드 thread로 RAM 주기적 점검. 한계 초과 시 should_abort() True.
    """
    def __init__(
        self,
        ram_limit: float = RAM_LIMIT,
        interval_sec: float = MONITOR_INTERVAL_SEC,
        on_abort=None,  # callable(reason: str) -- optional
        logger=None,
    ):
        self.ram_limit = ram_limit
        self.interval_sec = interval_sec
        self.on_abort = on_abort
        self.logger = logger
        self._stop_event = threading.Event()
        self._abort_event = threading.Event()
        self._abort_reason: Optional[str] = None
        self._thread: Optional[threading.Thread] = None

    def _log(self, msg: str):
        if self.logger is not None:
            try:
                self.logger.info(msg)
                return
            except Exception:
                pass
        print(msg, flush=True)

    def _loop(self):
        while not self._stop_event.is_set():
            if psutil is not None:
                ram = float(psutil.virtual_memory().percent)
                if ram >= self.ram_limit:
                    reason = f"RAM {ram:.1f}% >= {self.ram_limit:.0f}% -- 학습 중단"
                    self._abort_reason = reason
                    self._log(f"[guard] {reason}")
                    self._abort_event.set()
                    if self.on_abort is not None:
                        try:
                            self.on_abort(reason)
                        except Exception:
                            pass
                    return
            self._stop_event.wait(self.interval_sec)

    def start(self):
        if self._thread is not None:
            return
        if psutil is None:
            self._log("[guard] psutil 미설치 -- monitor 스킵")
            return
        self._thread = threading.Thread(target=self._loop, name="ResourceMonitor", daemon=True)
        self._thread.start()
        self._log(f"[guard] monitor started (RAM<={self.ram_limit:.0f}%, every {self.interval_sec:.0f}s)")

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def should_abort(self) -> bool:
        return self._abort_event.is_set()

    @property
    def abort_reason(self) -> Optional[str]:
        return self._abort_reason


# ---- CLI smoke test ----
if __name__ == "__main__":
    a = assess_start()
    print(format_assessment(a))
    if not a["ok"]:
        sys.exit(1)
    print("\n[guard] 5초간 모니터 동작 확인...")
    m = ResourceMonitor(interval_sec=2)
    m.start()
    time.sleep(5)
    m.stop()
    print(f"abort={m.should_abort()} reason={m.abort_reason}")
