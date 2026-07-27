#!/usr/bin/env python3
"""Build batch-sequence pool manifests for the temporal novelty-detection simulation.

사용자/팀리드 지시 (260726): "사내 실데이터에서 unknown grouping 으로 신규불량 발생 시
감지" — 정적 지표(P1~P4)만으로는 시간축(감지 지연/오경보율)을 측정할 수 없다. 이 스크립트는
그 시뮬레이션에 쓸 배치 시퀀스를 **manifest 로만** 구성한다 (폴더 생성/복사/링크 없음 — 이미지는
E:/data/images/unknown 마스터를 상대경로로 가리킬 뿐).

설계 (data/pools/temporal/<out-name>/sim_config.json 에 그대로 기록됨):
  - background = Normal + known 클래스 목록. 기본은 하드코드 10 class(=구 unknown_train_defectaware_260710
    학습 클래스). --known-pool 로 다른 manifest(예: v2 strict_novel_train.json)를 주면 그
    manifest 의 라벨 집합(Normal 제외)에서 **자동 derive** — 클래스 리스트 하드코드 회피.
  - novel = --novel-class (기본 CrossScratch — 마스터에만 존재, 기본 known pool 의 학습 클래스
    어디에도 없음 — 진짜 미학습 클래스). t<9 배치엔 단 한 장도 넣지 않는다 (누수 방지, 자체검증).
  - t=1..4  : calibration (기준 REF 클러스터링을 t=4 누적 풀에서 1회 확정)
  - t=5..8  : FAR-test (calibration 에 쓰이지 않은 순수 background, 오경보율 측정 구간)
  - t=9..14 : novel-window (배치당 신규불량 5/10/20/30장 sweep — size05/10/20/30 변형,
              배경 구성은 변형 간 동일 재사용, novel 이미지만 변형별로 겹치지 않게 분리)
  - 모든 이미지는 클래스별로 seed=42 로 1회 shuffle 후 앞에서부터 겹치지 않게 슬라이스
    (배치 간 재사용 없음 — "새 lot" 가정에 맞춤).

★ 260726 팀리드 지시: 2차 novel 클래스(DiagonalSmear/BrokenRing) 교차검증 시 배경 스트림을
완전히 동일하게 유지해야 "novel 클래스만 바꾼 1-축 변경"이 된다. background 생성 로직은
seed/BATCH_BG/NORMAL_FRAC/known-class-목록/N_CALIB/N_FAR/N_NOVEL_WINDOW 에만 의존하고
--novel-class 와 무관하므로, 이 값들을 안 건드리는 한 batch_01..08.json 과 각 사이즈의
배경 부분은 novel 클래스를 바꿔도 바이트 단위로 동일하게 재생성된다(재현성 검증 가능).

★ 260726 (leakage-fix v2): --known-pool data/pools/v2/unknown/strict_novel_train.json 처럼
주면 known 클래스 목록이 그 manifest 에서 자동 파생된다 (champion 을 그 pool 로 새로 학습한
경우, "배경=학습 도메인" 이 실제로 성립 — 구 unknown_train_defectaware_260710 은 SHA 감사로
unknown_eval100/holdout/anchor 와 겹침이 확정돼 background=known 가정이 새 champion 에는
안 맞음).

출력 (--out-name 기본값 = "unknown_novelty_sim"):
  data/pools/temporal/<out-name>/sim_config.json
  data/pools/temporal/<out-name>/batch_{01..08}.json         (background-only, 변형 공유)
  data/pools/temporal/<out-name>/size{05,10,20,30}/batch_{09..14}.json (배경+novel 합본)
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = Path(__file__).resolve().parent.parent
MASTER = Path("E:/data/images/unknown")

SEED = 42
DEFAULT_BATCH_BG = 200   # 배치당 background 이미지 수 (기본 10-class 목록 기준)
NORMAL_FRAC = 0.70       # background 중 Normal 비중
N_CALIB = 4              # t=1..4  (calibration, REF 는 t=4 누적 풀에서 확정)
N_FAR = 4                # t=5..8  (FAR-test, 순수 background held-out)
N_NOVEL_WINDOW = 6       # t=9..14 (novel 주입 구간)
NOVEL_SIZES = [5, 10, 20, 30]

DEFAULT_KNOWN_DEFECT_CLASSES = [
    "Center_bank_boundary", "Center_scratch", "Donut_bank_boundary", "Donut_fork",
    "Edge-Ring_bank_boundary", "Edge-Ring_scratch", "Edge-Top_fork", "Full_scratch",
    "ParallelScratches", "RingDots",
]  # == 구 unknown_train_defectaware_260710.json 의 defect 클래스 (SHA 감사로 leak 확정 —
   # v2 champion 검증엔 --known-pool 로 반드시 override할 것)

T_TOTAL = N_CALIB + N_FAR + N_NOVEL_WINDOW  # 14


def list_class_files(cls: str) -> list[str]:
    d = MASTER / cls
    files = sorted(p.name for p in d.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    rng = random.Random(SEED)
    rng.shuffle(files)
    return files


def known_classes_from_pool(pool_path: Path) -> list[str]:
    """learned-pool manifest 의 라벨 집합(Normal 제외)에서 known 클래스 목록을 derive."""
    manifest = json.loads(pool_path.read_text(encoding="utf-8"))
    labels = sorted({f["label"] for f in manifest["files"] if f.get("label") != "Normal"})
    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--novel-class", default="CrossScratch",
                     help="held-out novel class (must be absent from the known/background pool"
                          " for a clean leakage check)")
    ap.add_argument("--known-pool", default=None,
                     help="manifest whose label set (minus Normal) becomes the background known-class"
                          " list (e.g. data/pools/v2/unknown/strict_novel_train.json). Default: "
                          "hardcoded 10-class list (legacy unknown_train_defectaware_260710, now known"
                          " to leak into unknown_eval100/holdout/anchor — pass --known-pool for v2 work).")
    ap.add_argument("--batch-bg", type=int, default=None,
                     help="background images per batch (default: 200 for the legacy 10-class list, or"
                          " 210 when --known-pool yields 21 classes — auto-picked for clean per-class"
                          " integer counts; override only if you know the class count divides evenly)")
    ap.add_argument("--out-name", default=None,
                     help="output dir name under data/pools/temporal/ (default: unknown_novelty_sim"
                          " for CrossScratch, else unknown_novelty_sim_<novelclass-lower>)")
    a = ap.parse_args()
    novel_class = a.novel_class
    out_name = a.out_name or (
        "unknown_novelty_sim" if novel_class == "CrossScratch"
        else f"unknown_novelty_sim_{novel_class.lower()}"
    )
    out = REPO / "data/pools/temporal" / out_name

    if a.known_pool:
        known_classes = known_classes_from_pool(REPO / a.known_pool)
        known_pool_source = a.known_pool
    else:
        known_classes = DEFAULT_KNOWN_DEFECT_CLASSES
        known_pool_source = None
    batch_bg = a.batch_bg or (DEFAULT_BATCH_BG if not a.known_pool else 210)

    out.mkdir(parents=True, exist_ok=True)
    for s in NOVEL_SIZES:
        (out / f"size{s:02d}").mkdir(parents=True, exist_ok=True)

    # ── background 파일 풀 (클래스별 셔플 후 앞에서부터 배치 순서로 겹치지 않게 소비) ──
    n_defect_per_class_per_batch = round(batch_bg * (1 - NORMAL_FRAC) / len(known_classes))
    n_normal_per_batch = batch_bg - n_defect_per_class_per_batch * len(known_classes)
    assert n_normal_per_batch > 0
    assert n_defect_per_class_per_batch > 0

    normal_pool = list_class_files("Normal")
    defect_pools = {c: list_class_files(c) for c in known_classes}
    novel_pool = list_class_files(novel_class)
    assert novel_class not in known_classes, f"novel class {novel_class!r} is in the known/background list — pick another"

    need_normal = n_normal_per_batch * T_TOTAL
    need_defect = n_defect_per_class_per_batch * T_TOTAL
    need_novel = sum(NOVEL_SIZES) * N_NOVEL_WINDOW
    assert need_normal <= len(normal_pool), (need_normal, len(normal_pool))
    for c, pool in defect_pools.items():
        assert need_defect <= len(pool), (c, need_defect, len(pool))
    assert need_novel <= len(novel_pool), (novel_class, need_novel, len(novel_pool))

    cursor_normal = 0
    cursor_defect = {c: 0 for c in known_classes}
    cursor_novel = 0

    def take_normal(n: int) -> list[dict]:
        nonlocal cursor_normal
        chunk = normal_pool[cursor_normal:cursor_normal + n]
        cursor_normal += n
        return [{"path": f"Normal/{f}", "label": "Normal"} for f in chunk]

    def take_defect(cls: str, n: int) -> list[dict]:
        pool = defect_pools[cls]
        c0 = cursor_defect[cls]
        chunk = pool[c0:c0 + n]
        cursor_defect[cls] = c0 + n
        return [{"path": f"{cls}/{f}", "label": cls} for f in chunk]

    def take_novel(n: int) -> list[dict]:
        nonlocal cursor_novel
        chunk = novel_pool[cursor_novel:cursor_novel + n]
        cursor_novel += n
        return [{"path": f"{novel_class}/{f}", "label": novel_class} for f in chunk]

    def background_batch() -> list[dict]:
        entries = take_normal(n_normal_per_batch)
        for c in known_classes:
            entries += take_defect(c, n_defect_per_class_per_batch)
        return entries

    def write_manifest(path: Path, entries: list[dict]) -> None:
        manifest = {
            "root": "E:/data/images/unknown",
            "source_pool": None,
            "n_files": len(entries),
            "verify": "temporal-sim (make_temporal_novelty_pools.py)",
            "files": entries,
        }
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # ── t=1..8: 배경 전용 (calibration+FAR-test) ──
    bg_batches: dict[int, list[dict]] = {}
    for t in range(1, N_CALIB + N_FAR + 1):
        entries = background_batch()
        bg_batches[t] = entries
        write_manifest(out / f"batch_{t:02d}.json", entries)

    # ── t=9..14: novel-window 배경은 ★한 번만★ 뽑아 4개 변형이 그대로 공유 ──
    # (배경=배경이므로 변형 간 동일 스트림 위에서 novel 주입량만 바꾸는 통제실험 설계.
    #  변형마다 새로 뽑으면 4배 소모되어 소량 클래스 풀이 바닥나 뒷 배치가 조용히
    #  잘리는 버그가 났었다 — 반드시 1회만 호출.)
    novel_window_bg: dict[int, list[dict]] = {}
    for t in range(N_CALIB + N_FAR + 1, T_TOTAL + 1):
        novel_window_bg[t] = background_batch()

    leakage_report = {}
    for s in NOVEL_SIZES:
        novel_used_this_variant = []
        for t in range(N_CALIB + N_FAR + 1, T_TOTAL + 1):
            entries = list(novel_window_bg[t]) + take_novel(s)
            assert len(entries) == batch_bg + s, (s, t, len(entries))
            novel_used_this_variant.extend(e["path"] for e in entries if e["label"] == novel_class)
            write_manifest(out / f"size{s:02d}" / f"batch_{t:02d}.json", entries)
        leakage_report[f"size{s:02d}"] = len(novel_used_this_variant)

    # ── 누수 자체검증: t=1..8 배경 manifest 중 novel_class 가 단 한 장도 없어야 한다 ──
    for t in range(1, N_CALIB + N_FAR + 1):
        assert all(e["label"] != novel_class for e in bg_batches[t]), f"LEAK: novel class present in batch_{t:02d} (t<9)"

    sim_config = {
        "seed": SEED,
        "master_root": str(MASTER),
        "batch_size_background": batch_bg,
        "normal_frac": NORMAL_FRAC,
        "n_normal_per_batch": n_normal_per_batch,
        "n_defect_per_class_per_batch": n_defect_per_class_per_batch,
        "known_classes": known_classes,
        "known_pool_source": known_pool_source,
        "novel_class": novel_class,
        "phases": {
            "calibration": {"t_range": [1, N_CALIB], "purpose": "REF 클러스터 확정 (t=4 누적) + noise-floor 분포"},
            "far_test": {"t_range": [N_CALIB + 1, N_CALIB + N_FAR], "purpose": "held-out 순수 background, FAR 측정"},
            "novel_window": {"t_range": [N_CALIB + N_FAR + 1, T_TOTAL], "purpose": "novel 주입, detection lag 측정"},
        },
        "novel_sizes_per_batch": NOVEL_SIZES,
        "t_total": T_TOTAL,
        "t0_novel_start": N_CALIB + N_FAR + 1,
        "novel_images_used_per_variant": leakage_report,
        "leakage_check": "PASS — novel class absent from all batch_01..batch_08 manifests (asserted above)",
        "note_background_reuse": (
            "t=9..14 배경 구성은 4개 novel-size 변형 간 동일 이미지 재사용 (동일 배경 스트림 위에서 "
            "주입량만 바꾸는 통제실험 설계 — 배경=배경이므로 데이터 누수 아님, 재현성 위해 명시)."
        ),
        "note_cross_novel_reproducibility": (
            "background 생성(batch_01..08 + novel-window 배경)은 seed/batch-bg/known 클래스 목록/"
            "N_CALIB/N_FAR/N_NOVEL_WINDOW 에만 의존 — --novel-class 를 바꿔도 이 값들이 동일하면 "
            "background manifest 내용은 바이트 단위로 동일 (교차검증 시 1축 변경 보장)."
        ),
    }
    (out / "sim_config.json").write_text(json.dumps(sim_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[OUT] {out}")
    print(json.dumps({k: v for k, v in sim_config.items() if k not in ("known_classes",)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
