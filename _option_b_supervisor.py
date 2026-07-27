"""Option B supervisor — sequential chain orchestrator.

Stage 1: Wait ImageNet embed (PID 59500) done.
Stage 2: Compute tier1 for ImageNet baseline → save to outputs/.../tier1_imagenet.json
Stage 3: Dispatch CNN backbone training (known-cnn, split A 21 class).
Stage 4: Wait CNN training done → contrastive backbone path 확정.
Stage 5: Dispatch contrastive learning (split B 22 class) with new backbone.
Stage 6: Compute tier1 for SplitB → 3-way 비교 (TAPT vs ImageNet vs SplitB).

nohup detach 으로 spawn. 모니터링은 _option_b_supervisor.log 확인.
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

REPO = Path("D:/project/unknown-contrastive")
KNOWN_CNN = Path("D:/project/known-cnn")
LOG = REPO / "_option_b_supervisor.log"

# Stage 1 — ImageNet embed PID
IMAGENET_PID = 59500
IMAGENET_OUTPUT = REPO / "outputs_contrastive_260527_064042"

# Stage 3 — CNN backbone training
CNN_TRAIN_SCRIPT = KNOWN_CNN / "wafer_train" / "cnn_train_wafer.py"
SPLIT_A_YAML = KNOWN_CNN / "experiments" / "split_a_cnn_21.yaml"
SPLIT_B_YAML = REPO / "experiments" / "split_b_contrastive_22.yaml"


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def wait_pid_dead(pid: int, label: str, poll_sec: int = 60, max_hours: float = 24):
    """Poll until PID dies. Max wait."""
    log(f"[wait] {label} PID={pid} alive→dead polling (every {poll_sec}s, max {max_hours}h)")
    started = time.time()
    while time.time() - started < max_hours * 3600:
        if not psutil.pid_exists(pid):
            log(f"[wait] {label} PID={pid} DEAD after {(time.time()-started)/60:.1f}min")
            return True
        time.sleep(poll_sec)
    log(f"[wait] {label} PID={pid} STILL ALIVE after {max_hours}h, abort")
    return False


def stage_1_wait_imagenet_embed():
    log("=== Stage 1: wait ImageNet embed done ===")
    if not wait_pid_dead(IMAGENET_PID, "ImageNet embed", poll_sec=60, max_hours=3):
        log("Stage 1 timeout")
        return False
    # Check clusters_global_list.txt exists
    clusters_file = IMAGENET_OUTPUT / "clusters_global_list.txt"
    if clusters_file.exists():
        log(f"[ok] {clusters_file} exists ({clusters_file.stat().st_size} bytes)")
        return True
    log(f"[warn] {clusters_file} NOT FOUND — embed may have crashed")
    return False


def stage_2_compute_imagenet_tier1():
    log("=== Stage 2: compute tier1 for ImageNet baseline ===")
    # Use the same logic as _compute_tier1_remain.py inline
    try:
        from collections import Counter, defaultdict
        import numpy as np
        from sklearn.metrics import (adjusted_mutual_info_score,
                                     adjusted_rand_score,
                                     completeness_score,
                                     homogeneity_score)
        TOTAL = 19250
        lst = IMAGENET_OUTPUT / "clusters_global_list.txt"
        lines = lst.read_text(encoding="utf-8").strip().split("\n")
        rows = [ln.split("\t") for ln in lines[1:]]
        pred = np.array([int(r[0]) for r in rows])
        true_cls = np.array([r[1] for r in rows])
        classes = sorted(set(true_cls))
        cl2i = {c: i for i, c in enumerate(classes)}
        true = np.array([cl2i[c] for c in true_cls])
        n_clustered = len(rows)
        noise = TOTAL - n_clustered
        noise_pct = noise / TOTAL * 100

        cluster_classes = defaultdict(Counter)
        for p, c in zip(pred, true_cls):
            cluster_classes[int(p)][c] += 1
        class_total = Counter(true_cls)
        capture = {}
        for cls, total in class_total.items():
            max_in = max(
                (cnt for cl, ccnt in cluster_classes.items() for c, cnt in ccnt.items() if c == cls),
                default=0,
            )
            capture[cls] = max_in / total
        capture_rate = float(np.mean(list(capture.values())))
        ari = float(adjusted_rand_score(true, pred))
        ami = float(adjusted_mutual_info_score(true, pred))
        hom = float(homogeneity_score(true, pred))
        comp = float(completeness_score(true, pred))
        n_clusters = len(set(pred))

        result = {
            "cell": "ImageNet_only_n500",
            "total_samples": TOTAL,
            "clustered_samples": int(n_clustered),
            "noise_count": int(noise),
            "noise_pct": round(noise_pct, 2),
            "n_clusters": int(n_clusters),
            "class_capture_rate": round(capture_rate, 4),
            "completeness": round(comp, 4),
            "homogeneity": round(hom, 4),
            "ari": round(ari, 4),
            "ami": round(ami, 4),
        }
        out = IMAGENET_OUTPUT / "tier1_ImageNet_only_n500.json"
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"[ok] tier1 saved: {out}")
        for k, v in result.items():
            log(f"  {k:30s} {v}")
        return True
    except Exception as e:
        log(f"[err] stage_2 fail: {e}")
        import traceback
        log(traceback.format_exc())
        return False


def stage_3_train_cnn_backbone():
    log("=== Stage 3: dispatch CNN backbone (split A 21 class) ===")
    if not CNN_TRAIN_SCRIPT.exists():
        log(f"[err] {CNN_TRAIN_SCRIPT} not found")
        return None
    if not SPLIT_A_YAML.exists():
        log(f"[err] {SPLIT_A_YAML} not found")
        return None

    # Output dir name 으로 추적
    model_tag = "splitA_21cls_b16_e30"
    out_log = REPO / "_dispatch_logs" / f"_cnn_splitA_{datetime.now().strftime('%y%m%d_%H%M%S')}.log"
    cmd = [
        sys.executable,
        str(CNN_TRAIN_SCRIPT),
        "--epochs", "30",
        "--batch", "16",
        "--model-tag", model_tag,
        "--active-classes-yaml", str(SPLIT_A_YAML),
        "--allow-missing-active-classes",
    ]
    log(f"[exec] {' '.join(cmd)}")
    log(f"[log]  {out_log}")
    DETACHED_PROCESS = 0x00000008
    CREATE_NO_WINDOW = 0x08000000
    flags = (subprocess.CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS | CREATE_NO_WINDOW
             if os.name == "nt" else 0)
    with open(out_log, "wb") as fp:
        proc = subprocess.Popen(
            cmd, cwd=str(KNOWN_CNN), stdout=fp, stderr=subprocess.STDOUT,
            creationflags=flags,
        )
    log(f"[spawned] CNN backbone PID={proc.pid}")
    return proc.pid


def stage_4_wait_cnn_done(pid: int):
    log(f"=== Stage 4: wait CNN backbone PID={pid} done ===")
    return wait_pid_dead(pid, "CNN backbone", poll_sec=300, max_hours=10)


def stage_5_find_best_cnn_backbone():
    """Find latest logs_wafer/splitA_*/best_model.pth in known-cnn."""
    log("=== Stage 5: locate best CNN backbone ===")
    logs_wafer = KNOWN_CNN / "logs_wafer"
    candidates = sorted(logs_wafer.glob("splitA_*/best_model.pth"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        candidates = sorted(logs_wafer.glob("*/best_model.pth"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        log("[err] no best_model.pth found in known-cnn/logs_wafer")
        return None
    best = candidates[0]
    log(f"[ok] selected backbone: {best}")
    return best


def stage_6_dispatch_split_b_contrastive(backbone_path: Path):
    """Dispatch run_contrastive.py with new backbone + split B class filter."""
    log("=== Stage 6: dispatch contrastive Split B 22 class ===")
    out_log = REPO / "_dispatch_logs" / f"_splitB_contrastive_{datetime.now().strftime('%y%m%d_%H%M%S')}.log"
    env = {
        **os.environ,
        "DATA_DIR": "E:/data/images/unknown",
        # ★ backbone path env — run_contrastive.py line 401 가 BACKBONE_CKPT 읽음
        "BACKBONE_CKPT": str(backbone_path),
        # ★ class filter env — run_contrastive.py FilteredImageFolder 가 yaml 의 classes 만 keep
        "ACTIVE_CLASSES_YAML": str(SPLIT_B_YAML),
        "NUM_WORKERS": "0",
        "BATCH": "4",
        "EPOCHS": "5",
        "WARMUP_EPOCHS": "1",
        "TRAIN_SAMPLING_RATIO": "0.25",
        "USE_QUEUE": "true",
        "QUEUE_SIZE": "4096",
        "USE_LOCAL": "false",
        "IGNORE_NEG_SIM": "0.72",
        "NCE_TEMP": "0.07",
        "LR_HEAD": "0.001",
        "FREEZE_BACKBONE": "true",
        "SEED": "42",
        "CLUSTER_SELECTION_METHOD": "eom",
        "CLUSTER_SELECTION_EPSILON": "0.0",
        "MIN_CLUSTER_SIZE": "12",
        "MIN_SAMPLES": "3",
        "PER_CLASS_CAP": "500",
        "NORMAL_CAP": "2000",
        "NECO_WEIGHT": "0.2",
        "NECO_TAU": "0.1",
    }
    DETACHED_PROCESS = 0x00000008
    CREATE_NO_WINDOW = 0x08000000
    flags = (subprocess.CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS | CREATE_NO_WINDOW
             if os.name == "nt" else 0)
    log(f"[exec] python run_contrastive.py (env BACKBONE_STATE_PATH={backbone_path.name})")
    log(f"[log]  {out_log}")
    with open(out_log, "wb") as fp:
        proc = subprocess.Popen(
            [sys.executable, "run_contrastive.py"],
            cwd=str(REPO), stdout=fp, stderr=subprocess.STDOUT,
            env=env, creationflags=flags,
        )
    log(f"[spawned] Split B contrastive PID={proc.pid}")
    return proc.pid, out_log


def stage_7_wait_and_compute_tier1(pid: int, out_log_path: Path):
    log(f"=== Stage 7: wait Split B contrastive PID={pid} ===")
    if not wait_pid_dead(pid, "Split B contrastive", poll_sec=120, max_hours=4):
        return False
    # find newest outputs_contrastive_* with clusters_global_list.txt 생성된 것
    log("=== Stage 7b: compute tier1 for Split B ===")
    candidates = sorted(REPO.glob("outputs_contrastive_*/clusters_global_list.txt"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        log("[err] no clusters_global_list.txt found")
        return False
    target = candidates[0].parent
    log(f"[target] {target}")

    from collections import Counter, defaultdict
    import numpy as np
    from sklearn.metrics import (adjusted_mutual_info_score,
                                 adjusted_rand_score,
                                 completeness_score,
                                 homogeneity_score)
    lst = target / "clusters_global_list.txt"
    lines = lst.read_text(encoding="utf-8").strip().split("\n")
    rows = [ln.split("\t") for ln in lines[1:]]
    pred = np.array([int(r[0]) for r in rows])
    true_cls = np.array([r[1] for r in rows])
    classes = sorted(set(true_cls))
    cl2i = {c: i for i, c in enumerate(classes)}
    true = np.array([cl2i[c] for c in true_cls])
    n_clustered = len(rows)

    cluster_classes = defaultdict(Counter)
    for p, c in zip(pred, true_cls):
        cluster_classes[int(p)][c] += 1
    class_total = Counter(true_cls)
    capture = {}
    for cls, total in class_total.items():
        max_in = max(
            (cnt for cl, ccnt in cluster_classes.items() for c, cnt in ccnt.items() if c == cls),
            default=0,
        )
        capture[cls] = max_in / total
    capture_rate = float(np.mean(list(capture.values())))
    result = {
        "cell": "SplitB_contrastive",
        "clustered_samples": int(n_clustered),
        "n_clusters": int(len(set(pred))),
        "class_capture_rate": round(capture_rate, 4),
        "completeness": round(completeness_score(true, pred), 4),
        "homogeneity": round(homogeneity_score(true, pred), 4),
        "ari": round(adjusted_rand_score(true, pred), 4),
        "ami": round(adjusted_mutual_info_score(true, pred), 4),
    }
    out = target / "tier1_SplitB_contrastive.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"[ok] tier1 saved: {out}")
    for k, v in result.items():
        log(f"  {k:25s} {v}")
    return True


def main():
    log("================================================================")
    log("OPTION B SUPERVISOR v3 — CNN backbone wait + Split B auto")
    log("================================================================")
    log(f"REPO        = {REPO}")
    log(f"KNOWN_CNN   = {KNOWN_CNN}")

    # v3: ImageNet baseline skip (반복 실패), CNN backbone 이미 dispatched
    cnn_pid_file = REPO / "_dispatch_logs" / "_splitA_cnn_pid.txt"
    if not cnn_pid_file.exists():
        log(f"[err] {cnn_pid_file} not found — CNN backbone not dispatched")
        return 1
    cnn_pid = int(cnn_pid_file.read_text().strip())
    log(f"[v3] using pre-dispatched CNN PID={cnn_pid}")

    if not stage_4_wait_cnn_done(cnn_pid):
        log("Stage 4 timeout — abort")
        return 1
    backbone_path = stage_5_find_best_cnn_backbone()
    if not backbone_path:
        log("Stage 5 fail — abort")
        return 1
    splitb_pid, splitb_log = stage_6_dispatch_split_b_contrastive(backbone_path)
    stage_7_wait_and_compute_tier1(splitb_pid, splitb_log)
    log("================================================================")
    log("OPTION B SUPERVISOR DONE (TAPT / ImageNet / SplitB 3-way 비교 가능)")
    log("================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
