#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""site/ 단일 설정 파일 — **여기만 고치면 된다.**

  - 경로는 전부 **프로젝트 루트 기준 상대경로**. 절대경로를 써도 되지만 권장하지 않는다.
  - 각 값은 `env("SITE_XXX", default)` — **환경변수가 있으면 그것, 없으면 default.**
    서버에 고정 경로가 있으면 아래 default 를 직접 고쳐두는 게 편하다.
  - step0~5 가 전부 이 파일을 import 한다. 같은 값을 여러 파일에서 고칠 일이 없다.

★ 절대 넣지 않는 것: **불량 종수 k.** 모르는 게 전제이고 HDBSCAN 을 쓰는 이유가 k-free 라서다.
  클러스터 개수를 입력으로 주는 건 치팅이다. 다이얼은 `Cluster.MIN_GROUP_SIZE`
  ("몇 장 이상 뭉쳐야 그룹인가" = 운영 판단) 또는 `Cluster.AUTO_DIAL` (bootstrap 안정성)로 정한다.

★ UMAP 도 쓰지 않는다. raw 임베딩 -> HDBSCAN 직접.
  UMAP 은 평가 데이터 자체에 fit 하는 transductive 변환이라 (a) 배포에서 이미지 1장을 판정할 수
  없고 (b) 배치마다 좌표계가 바뀌어 시간축 REF 비교가 무너진다. 실측상 UMAP-free 가 contrastive
  이득도 더 컸다 (`docs/paper/ABLATION_TABLE_UMAPFREE.md`).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _site_common import env  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
class Paths:
    """모델 · 데이터 · 산출 경로."""

    # ── 데이터 ────────────────────────────────────────────────────────────
    # 사내 이미지 루트. 하위폴더 전부 재귀 수집(깊이 무관). 라벨 폴더 구조 불필요.
    IMAGE_ROOT = env("SITE_IMAGE_ROOT", "data/site_images")
    # 이미 골라둔 목록이 있으면 여기에. 지정하면 IMAGE_ROOT 스캔을 건너뛴다.
    POOL_MANIFEST = env("SITE_POOL_MANIFEST", "")
    EXTS = env("SITE_EXTS", "png,jpg,jpeg,bmp,tif,tiff")

    # ── 산출 ──────────────────────────────────────────────────────────────
    OUT_ROOT = env("SITE_OUT_ROOT", "runs/site")

    # ── 모델 ──────────────────────────────────────────────────────────────
    # 기준선 backbone. step2/3/5 학습·임베딩의 기본값이기도 하다.
    BACKBONE = env("SITE_BACKBONE", "weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth")

    # champion projection head — 2개 concat+L2 앙상블이 배포 기본값. 콤마 구분.
    CHAMPION_PROJ = env("SITE_CHAMPION_PROJ",
                        "weights/champion/proj_s42_ep20.pt,weights/champion/proj_s1_ep18.pt")

    # May 배포본 contrastive_b4 — ★ 자체 backbone 을 갖는 독립 arm (FCMAE 와 섞으면 안 됨).
    #   `python site/extract_b4.py` 로 아래 두 파일을 만든다.
    B4_BACKBONE = env("SITE_B4_BACKBONE", "weights/b4_may/b4_backbone.pth")
    B4_PROJ = env("SITE_B4_PROJ", "weights/b4_may/b4_proj.pt")
    B4_BUNDLE = env("SITE_B4_SRC", "weights/b4_may/contrastive_b4.pt")   # extract 입력

    # 사다리 ④ TAPT backbone (없으면 step4 는 안내만 하고 skip)
    TAPT_BACKBONE = env("SITE_TAPT_BACKBONE", "weights/tapt/backbone_tapt.pth")


# ═══════════════════════════════════════════════════════════════════════════
class Runtime:
    """실행 환경.

    ★ H200(141GB) 기준 default. 추론 배치는 **자원 파라미터**라 올려도 결과가 안 바뀐다.
      VRAM 이 작은 장비면 SITE_BATCH 를 32~64 로 낮춰라.
      반면 `Recipe.BATCH`(학습 배치)는 **레시피 파라미터**다 — 결과가 바뀌므로
      자원이 남는다고 임의로 올리지 마라(올리려면 sweep 셀로 비교할 것).
    """
    DEVICE = env("SITE_DEVICE", "cuda")        # cuda | cpu
    BATCH = env("SITE_BATCH", 128)             # 추론/채점 배치 (H200 여유)


# ═══════════════════════════════════════════════════════════════════════════
class Cluster:
    """HDBSCAN 다이얼 + 후처리. ★ k 는 입력하지 않는다."""

    # "몇 장 이상 뭉쳐야 하나의 그룹으로 볼 것인가" = HDBSCAN min_cluster_size 의 원래 의미.
    # 클래스 수가 아니라 **보고 가치가 있는 최소 그룹 크기**라 k 를 몰라도 정할 수 있다.
    MIN_GROUP_SIZE = env("SITE_MIN_GROUP_SIZE", 20)

    # 1 이면 MIN_GROUP_SIZE 주변을 스캔해 **bootstrap 안정성 최대**인 다이얼을 자동 선택.
    # 라벨도 k 도 쓰지 않는다. 실측: severstal 의 아는 정답(mcs20)을 정확히 집었다
    # (DBCV 는 15 로 빗나갔고, ARI 최대화는 mcs60 = k2 병합 치팅으로 걸어갔다).
    AUTO_DIAL = env("SITE_AUTO_DIAL", False)

    # 수동 고정 (0 이면 위 규칙으로 자동 계산). 튜닝 목적 외에는 건드리지 마라.
    MCS = env("SITE_MCS", 0)
    MS = env("SITE_MS", 0)

    METHOD = env("SITE_METHOD", "leaf")        # leaf | eom
    EPS = env("SITE_EPS", 0.06)

    # noise 재배정. ★ 판정은 재배정 **전**(seed_noise)으로 한다 —
    # 재배정 후 noise 는 어떤 임베딩에든 나오는 바닥값이라(랜덤 head 와 0.03pp 차) 판정에 못 쓴다.
    REASSIGN = env("SITE_REASSIGN", "nearest_q90")   # none|nearest_q90|nearest_q80|assign_all

    # Rule C 의 k 하한 퍼센타일. ★ 이건 k 입력이 아니라 **자기 run 안에서의 상대 기준**이다.
    # 이게 없으면 noise 최소화가 "전부 한 덩어리" 축퇴 해로 걸어간다.
    RULE_C_K_PERCENTILE = env("SITE_K_PERCENTILE", 75)


# ═══════════════════════════════════════════════════════════════════════════
class Recipe:
    """contrastive 학습 레시피 (B4). step2 기본값이자 step3 스윕의 base."""

    EPOCHS = env("SITE_EPOCHS", 20)
    SEEDS = env("SITE_SEEDS", "42,1,2")        # 3-seed 권장. 급하면 "42"

    TEMP = env("SITE_TEMP", 0.20)
    QUEUE = env("SITE_QUEUE", 16384)
    IGNORE_NEG = env("SITE_IGNORE_NEG", 0.72)  # ★ env 이름은 REPRO_IGNORE_NEG_SIM (REPRO_NEG 아님)
    LR_HEAD = env("SITE_LR_HEAD", 0.004)
    # ★ 학습 배치는 레시피다(결과가 바뀐다). H200 이라 256 도 무리 없지만
    #   B4 기본은 64 다. 256 을 쓰려면 MayPreset 또는 sweep 셀 `may` 로 **비교해서** 채택해라.
    BATCH = env("SITE_TRAIN_BATCH", 64)
    SAMPLING = env("SITE_SAMPLING", 0.25)
    USE_LOCAL = env("SITE_USE_LOCAL", True)

    # ★ residual adapter — GAP 직후, projection head **앞**에 들어간다:
    #     pool = pool + gamma * adapt(pool)   (gamma=0 초기화 -> 시작점이 정확히 frozen)
    #   frozen 하한이 보장되므로 도움이 될 때만 벗어난다. 시간축 champion 이 이걸 썼다
    #   (`fcmae_ad1_t010_s1_ep4` 의 ad1).
    ADAPTER = env("SITE_ADAPTER", False)


# ═══════════════════════════════════════════════════════════════════════════
class MayPreset:
    """★ 5월 배포 SOTA 재현 조건 (`project_may_source_recovered_260722`).

    현재 B4 와 다른 점이 batch 다 — 재현 실패의 근본 원인이 batch 8 이었고,
    원본은 **256** 이었다. VRAM 이 부족하면 GPU 가 감당하는 최대로 낮추되
    64 미만으로는 내리지 마라(발산한다).

    ⚠ 5월 SOTA 헤드라인(ARI 0.8588 / capture 1.000 / noise 0.61%)은
      **UMAP + HDBSCAN** 조건이었다. 지금 파이프라인은 UMAP-free 라 그 숫자는
      직접 비교 대상이 아니다. 여기서 재현하는 건 **학습 레시피**뿐이다.
    """
    ENABLE = env("SITE_MAY_PRESET", False)     # 1 이면 Recipe 를 아래 값으로 덮어쓴다

    EPOCHS = 20
    BATCH = 256          # ← 핵심. batch 8 로 재현 시도했다가 발산했다. H200 이면 여유.
    TEMP = 0.20
    QUEUE = 16384
    IGNORE_NEG = 0.72
    LR_HEAD = 0.004
    SAMPLING = 0.25
    USE_LOCAL = True     # local loss 가중 0.5 (트레이너 내부 상수)
    ADAPTER = False
    # 그 외 트레이너 고정값: normalize ON / global_pool "" / manual GAP /
    #                        proj bias=False / backbone frozen


def recipe() -> dict:
    """현재 유효 레시피. MayPreset.ENABLE 이면 5월 조건으로 덮어쓴다."""
    r = {"temp": Recipe.TEMP, "queue": Recipe.QUEUE, "ignore_neg": Recipe.IGNORE_NEG,
         "lr_head": Recipe.LR_HEAD, "batch": Recipe.BATCH, "sampling": Recipe.SAMPLING,
         "use_local": bool(Recipe.USE_LOCAL), "adapter": bool(Recipe.ADAPTER)}
    if MayPreset.ENABLE:
        r.update({"temp": MayPreset.TEMP, "queue": MayPreset.QUEUE,
                  "ignore_neg": MayPreset.IGNORE_NEG, "lr_head": MayPreset.LR_HEAD,
                  "batch": MayPreset.BATCH, "sampling": MayPreset.SAMPLING,
                  "use_local": MayPreset.USE_LOCAL, "adapter": MayPreset.ADAPTER})
    return r


def epochs() -> int:
    return int(MayPreset.EPOCHS if MayPreset.ENABLE else Recipe.EPOCHS)


# ═══════════════════════════════════════════════════════════════════════════
class Sweep:
    """step3 레시피 스윕. 각 셀은 base 에서 **정확히 1축**만 바꾼다."""

    # 실측 승자가 LR 2배(lr008)라 LR 축을 앞에 둔다 — 3셀만 돌려도 방향이 보인다.
    CELLS = env("SITE_CELLS", "lr008,lr002,q32768,q4096,t030,t010,neg060,neg085,nolocal")
    ROUND1_SEED = env("SITE_ROUND1_SEED", 42)
    TOP_N = env("SITE_TOP_N", 3)               # 상위 N 셀만 multi-seed 재확인 (0=생략)
    ROUND2_SEEDS = env("SITE_ROUND2_SEEDS", "1,2")

    AXES = {
        "base":    {},
        "lr002":   {"lr_head": 0.002},
        "lr008":   {"lr_head": 0.008},
        "lr016":   {"lr_head": 0.016},
        "t010":    {"temp": 0.10},
        "t030":    {"temp": 0.30},
        "neg060":  {"ignore_neg": 0.60},
        "neg085":  {"ignore_neg": 0.85},
        "q4096":   {"queue": 4096},
        "q32768":  {"queue": 32768},
        "nolocal": {"use_local": False},
        "adapter": {"adapter": True},          # residual adapter on
        "may":     {"batch": 256},             # 5월 원본 batch
    }


# ═══════════════════════════════════════════════════════════════════════════
class Temporal:
    """step5 시간축 신규불량 감지."""

    # 쓸 head. 비우면 step3 -> step2 -> frozen 순으로 자동 탐색. 콤마 구분(앙상블).
    PROJ = env("SITE_TEMPORAL_PROJ", "")

    # 시간 배치: by_dir = 상위 폴더명(날짜 권장) / by_mtime = 파일 수정시각 순
    TIME_MODE = env("SITE_TIME_MODE", "by_dir")
    TIME_BATCH_SIZE = env("SITE_TIME_BATCH_SIZE", 200)      # by_mtime 일 때만
    TIME_DIR_REGEX = env("SITE_TIME_DIR_REGEX", r"(\d{6,8})")

    # ★ 배치 단위 클러스터링의 최소 그룹 크기. pool 전체 다이얼과 **별개**다 —
    #   배치는 pool 보다 훨씬 작아서 같은 mcs 를 쓰면 클러스터가 하나도 안 잡힌다.
    #   0 이면 배치 크기의 5% (최소 3).
    BATCH_MIN_GROUP = env("SITE_BATCH_MIN_GROUP", 0)

    N_CALIB = env("SITE_N_CALIB", 4)   # 앞 N 배치로 REF centroid + 임계 캘리브레이션
    N_BG = env("SITE_N_BG", 4)         # 그 다음 N 배치 = "신규불량 없음" 구간 -> FAR 측정

    # ★ P1/P2 는 쓰지 마라 — 표본이 적으면 1st 퍼센타일은 통계량이 아니라 관측 최솟값이고,
    #   캘리브레이션 창을 한 배치 옮기면 임계가 전체 범위의 절반만큼 튄다(실측).
    #   P10/P20 은 5~15배 안정적이다.
    P_LIST = env("SITE_P_LIST", "10,20")
    K_LIST = env("SITE_K_LIST", "1,2,3")
    MIN_SIZE_PCT = env("SITE_MIN_SIZE_PCT", "25,50,75")


# ═══════════════════════════════════════════════════════════════════════════
class Judge:
    """판정 규격 — 실측으로 도출. 바꾸려면 근거가 있어야 한다."""

    # 동일 config 4회 반복에서 나온 측정 잡음폭. 이 안의 차이는 "차이 없음".
    BAND = {"noise": 2.28, "ARI": 0.019, "Hom": 0.005, "Comp": 0.033, "P1": 0.0}

    # 민감도 (평균 |diff| / 잡음폭). Comp 가 가장 관대 -> **단독 통과 판정 금지**.
    SENSITIVITY = ("P1", "ARI", "Hom", "noise", "Comp")

    # 우선순위 (사전식). frag 는 비용. 잡음폭 안이면 동률 처리 후 다음 축으로.
    PRIORITY = ("P1", "noise", "Comp", "Hom")


def show() -> dict:
    """전 설정을 한 번에 출력하고 dict 로 반환 (재현성 기록용)."""
    out = {}
    for cls in (Paths, Runtime, Cluster, Recipe, MayPreset, Sweep, Temporal):
        d = {k: getattr(cls, k) for k in vars(cls)
             if k.isupper() and not k.startswith("_")}
        out[cls.__name__] = d
        print(f"[{cls.__name__}]")
        for k in sorted(d):
            v = d[k]
            if isinstance(v, dict):
                v = f"<{len(v)} entries>"
            print(f"    {k:<20} = {v}")
    out["recipe_effective"] = recipe()
    out["epochs_effective"] = epochs()
    if MayPreset.ENABLE:
        print("  ★ MayPreset ON -> recipe =", recipe())
    print(flush=True)
    return out


if __name__ == "__main__":
    show()
