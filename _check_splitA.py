"""CNN backbone (Split A) 학습 진행 확인 도구.

사용자 호출:  python _check_splitA.py

기능:
- PID 11916 alive 확인
- 학습 log 최근 진행
- best_model.pth 산출 여부
- 끝나면 contrastive Split B dispatch 명령 출력
"""
import datetime
import os
import re
from pathlib import Path

import psutil

CNN_PID_FILE = Path("D:/project/unknown-contrastive/_dispatch_logs/_splitA_cnn_pid.txt")
LOGS_WAFER = Path("D:/project/known-cnn/logs_wafer")
DISPATCH_LOGS = Path("D:/project/unknown-contrastive/_dispatch_logs")


def main():
    pid = int(CNN_PID_FILE.read_text().strip()) if CNN_PID_FILE.exists() else None
    alive = psutil.pid_exists(pid) if pid else False
    print(f"=== CNN backbone (Split A) status ===")
    print(f"PID file: {CNN_PID_FILE}")
    print(f"PID:      {pid}")
    print(f"alive:    {alive}")

    # latest log
    logs = sorted(DISPATCH_LOGS.glob("_splitA_cnn_*.log"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if logs:
        log = logs[0]
        st = log.stat()
        mtime = datetime.datetime.fromtimestamp(st.st_mtime)
        print()
        print(f"=== log ({log.name}) ===")
        print(f"size: {st.st_size:,} bytes,  mtime: {mtime.strftime('%H:%M:%S')}")
        content = log.read_text(encoding="utf-8", errors="replace")
        lines = content.strip().split("\n")
        # last epoch progress
        epoch_lines = [ln for ln in lines if re.search(r"epoch\s*\d+", ln, re.I)]
        if epoch_lines:
            print()
            print("Latest 8 epoch-related lines:")
            for ln in epoch_lines[-8:]:
                print(f"  {ln[:200]}")
        else:
            print()
            print("Latest 10 lines:")
            for ln in lines[-10:]:
                print(f"  {ln[:200]}")

    # logs_wafer output check
    print()
    print(f"=== outputs (logs_wafer/splitA_21cls_*) ===")
    candidates = sorted(LOGS_WAFER.glob("splitA_21cls_*"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        print("  no splitA_21cls_* folder yet")
    else:
        for c in candidates[:3]:
            best = c / "best_model.pth"
            best_exists = best.exists()
            mtime = datetime.datetime.fromtimestamp(c.stat().st_mtime).strftime("%H:%M:%S")
            sz = f"{best.stat().st_size/1024/1024:.0f} MB" if best_exists else "-"
            print(f"  {c.name}  mtime={mtime}  best_model.pth={'EXISTS' if best_exists else 'pending'}  {sz}")

    # if done → contrastive Split B dispatch command
    if not alive and candidates:
        best = candidates[0] / "best_model.pth"
        if best.exists():
            print()
            print("================================================================")
            print("CNN BACKBONE DONE — dispatch Split B contrastive:")
            print("================================================================")
            print(f"""
cd /d/project/unknown-contrastive && \\
nohup env \\
  BACKBONE_CKPT="{best.as_posix()}" \\
  ACTIVE_CLASSES_YAML="experiments/split_b_contrastive_22.yaml" \\
  DATA_DIR="E:/data/images/unknown" \\
  NUM_WORKERS=0 BATCH=4 EPOCHS=5 WARMUP_EPOCHS=1 \\
  TRAIN_SAMPLING_RATIO=0.25 \\
  USE_QUEUE=true QUEUE_SIZE=4096 USE_LOCAL=false \\
  IGNORE_NEG_SIM=0.72 NCE_TEMP=0.07 LR_HEAD=0.001 \\
  FREEZE_BACKBONE=true SEED=42 \\
  CLUSTER_SELECTION_METHOD=eom CLUSTER_SELECTION_EPSILON=0.0 \\
  MIN_CLUSTER_SIZE=12 MIN_SAMPLES=3 \\
  PER_CLASS_CAP=500 NORMAL_CAP=2000 \\
  NECO_WEIGHT=0.2 NECO_TAU=0.1 \\
  python run_contrastive.py > _dispatch_logs/_splitB_contrastive.log 2>&1 < /dev/null &
""")


if __name__ == "__main__":
    main()
