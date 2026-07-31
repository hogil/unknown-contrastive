#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""site/ 단일 설정 파일 — **여기만 고치면 된다.**

  - 경로는 전부 **프로젝트 루트 기준 상대경로**. 절대경로를 써도 되지만 권장하지 않는다.
  - 각 값은 `env("SITE_XXX", default)` — **환경변수가 있으면 그것, 없으면 default.**
    서버에 고정 경로가 있으면 아래 default 를 직접 고쳐두는 게 편하다.
  - step0~5 가 전부 이 파일을 import 한다. 같은 값을 여러 파일에서 고칠 일이 없다.

★ 절대 넣지 않는 것: **불량 종수 k.** 모르는 게 전제이고 HDBSCAN 을 쓰는 이유가 k-free 라서다.
  클러스터 개수를 입력으로 주는 건 치팅이다. 다이얼은 `Cluster.MCS`
  ("몇 장 이상 뭉쳐야 그룹인가" = 운영 판단)로 정한다 — k 와 무관하게 답할 수 있는 값이다.

★ UMAP 도 쓰지 않는다. raw 임베딩 -> HDBSCAN 직접.
  UMAP 은 평가 데이터 자체에 fit 하는 transductive 변환이라 (a) 배포에서 이미지 1장을 판정할 수
  없고 (b) 배치마다 좌표계가 바뀌어 시간축 REF 비교가 무너진다. 실측상 UMAP-free 가 contrastive
  이득도 더 컸다 (`docs/paper/ABLATION_TABLE_UMAPFREE.md`).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _site_common import env  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# ★★★ composite 색상표 — 파일 경로 + 쓸 스킴 이름. 여기서 바로 보고 바꾼다.
#   실제 색상 값은 COLOR_LEGENDS_PATH 가 가리키는 JSON 안(composite.<scheme>)에 있다.
#   스킴 목록: default / anonymous / light_low_035 / light_low_070 /
#             fast_red_060 / fast_red_040 / fast_red_030 / fast_red_020
#   (아래 `class Composite` 가 이 두 값을 읽어 쓴다 — env 로 override 하려면
#    SITE_ENV_OVERRIDE=1 SITE_COLOR_LEGENDS=... SITE_COLOR_SCHEME=...)
#
# ★ 상대경로 / 절대경로 **둘 다** 된다 (`composite._read_stops`):
#     상대 -> 프로젝트 루트 기준으로 푼다. 폴더를 통째로 어디에 옮겨도 돈다.
#     절대 -> 그대로 쓴다.  예) mapviewer 정본을 그대로 가리키려면
#             "D:/project/mapviewer/logs/color-legends.json"
#             (단 그 파일의 composite.default 는 원본 빨강 램프라 "아래쪽 연하게"가 없다)
#   기본값이 상대경로인 이유: 절대경로를 박아두면 그 경로가 없는 서버(Ubuntu24)에서
#   색이 조용히 계산 기본값으로 바뀐다. 폴백이 있어 죽지는 않지만 색이 달라진다.
COLOR_LEGENDS_PATH = "deploy/color-legends.json"
COLOR_SCHEME_NAME = "fast_red_020"


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
    #   `python deploy/extract_b4.py` 로 아래 두 파일을 만든다.
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

    # ── ★ 디코드 캐시 — H200 을 놀리지 않는 핵심 스위치 ────────────────────
    # 병목은 GPU 가 아니라 6400x6400 palette PNG 디코드다. 실측(4060 Ti):
    #     캐시 없음  7.6 img/s   디코드대기 62~79% / GPU 21~38%
    #     캐시 사용 19.7 img/s   디코드대기  0.4% / GPU 99.5%      ← 2.6x
    # GPU 가 빠를수록 격차가 커진다. H200 은 forward 가 6~8배 빨라서 캐시 없이는
    # GPU 비중이 6% 근처로 떨어진다(= 보고된 nvidia-smi 값). 캐시가 있어야 GPU 가 산다.
    # 게다가 step1 은 arm 이 6개라 **같은 이미지를 6번 다시 디코드**했다 —
    # 캐시는 그 6번을 1번으로 만든다.
    # 결과는 원본과 **비트 단위로 동일**하다 (groups.csv/summary.json 완전 일치 확인).
    # ⚠ 학습에는 쓰지 마라 — RandomResizedCrop 이 원본 해상도를 필요로 한다.
    CACHE = env("SITE_CACHE", True)            # step0 이 만들고 step1/3/5 가 쓴다
    CACHE_DIR = env("SITE_CACHE_DIR", "")      # 비우면 <OUT_ROOT>/_cache
    # 디코드 프로세스 수. 0 = cpu_count 전부. 코어가 많을수록 캐시 생성이 빨라진다.
    DECODE_WORKERS = env("SITE_DECODE_WORKERS", 0)

    # ★ **학습** DataLoader 워커 수 — **배포 대상(Ubuntu24) 기준으로 켜 둔다.**
    #   실측(260730, anchor): 학습이 epoch 당 213s 인데 그 epoch 은 565장(2260×sampling
    #   0.25)이라 **2.65 img/s** — GPU 연산 한계가 아니라 6400x6400 **단일스레드 디코드에
    #   GPU 가 굶는 것**이다 (그 사이 nvidia-smi 는 100% 로 보이지만 처리량이 이렇다).
    #   그래서 사내 서버에서는 워커를 반드시 켜야 한다.
    #   ⚠ 플랫폼별로 갈린다 — Windows 는 spawn 으로 학습이 죽은 전례가 있어 0 이다
    #     (`_site_common.train_env` 경고, task #25). Linux 는 fork 라 안전하다.
    #     `_may_repro_src` 는 persistent_workers=False 라 워커 leak 도 없다.
    #   수동 고정하려면 SITE_TRAIN_WORKERS (SITE_ENV_OVERRIDE=1 필요) 또는 이 줄을 고쳐라.
    #   ★ 서버 첫 run 은 학습이 **끝까지 도는지** 확인하고, epoch 시간이 실제로 줄었는지
    #     `time=NNNs` 로그로 확인하라 (안 줄면 디코드가 아니라 다른 병목이라는 뜻).
    TRAIN_WORKERS = env("SITE_TRAIN_WORKERS",
                        0 if sys.platform.startswith("win") else min(16, (os.cpu_count() or 8)))

    # ── 정밀도 (기본 끔) ──────────────────────────────────────────────────
    # 실측 배속/오차 (4060 Ti, 임베딩 최대 절대차):
    #     channels_last  1.03x  9.0e-4
    #     bf16 + CL      1.54x  6.4e-3   <- 오차가 크다
    #     fp16 + CL      1.50x  8.5e-4   <- 같은 속도에 오차 1/7
    # ★ 기본 off 인 이유: 5.5e-4 오차만으로도 경계 근처 점의 HDBSCAN 클러스터가
    #   바뀐 실측이 있다. 측정 잡음폭이 ARI ±0.019 / Hom ±0.005 인 프로젝트에서
    #   1.5배 때문에 지표를 흔들 수 없다.
    #   속도가 급하고 arm 끼리 상대비교만 하면 되면 fp16 을 켜라 (전 arm 동일 설정 필수).
    AMP = env("SITE_AMP", "off")               # off | fp16 | bf16
    CHANNELS_LAST = env("SITE_CHANNELS_LAST", False)


# ═══════════════════════════════════════════════════════════════════════════
class Cluster:
    """HDBSCAN 다이얼 + 후처리. ★ k 는 입력하지 않는다."""

    # "몇 장 이상 뭉쳐야 하나의 그룹으로 볼 것인가" = HDBSCAN min_cluster_size 의 원래 의미.
    # 클래스 수가 아니라 **보고 가치가 있는 최소 그룹 크기**라 k 를 몰라도 정할 수 있다.
    # HDBSCAN min_cluster_size — **몇 장 이상 뭉쳐야 하나의 그룹으로 볼 것인가.**
    #   작으면 잘게 나뉘고 k 가 많아진다. 크면 작은 그룹이 통째로 noise 로 버려진다.
    #   ★ 불량 종수 k 가 아니다. k 는 모르는 게 전제이고 HDBSCAN 을 쓰는 이유가 k-free 라서다.
    #     이 값은 "보고 가치가 있는 최소 그룹 크기"라는 **운영 판단**이라 k 없이 정할 수 있다.
    #   ⚠ pool 마다 다시 정하라. 다른 pool 값 이식 금지 (mcs6 이식으로 결론이 뒤집힌 전례).
    #   ⚠ PARTITION_BY 를 켜면 **파티션마다** 적용된다. 작은 파티션이 있으면 낮춰라
    #     (step1 로그의 `[partition] ★ 경고` 확인).
    #   ★ 260731 사용자 지정: 10. (앞선 severstal 실측 승자는 mcs20/ms5 였지만 그건
    #     그 pool 의 값이고, 다이얼은 pool 마다 다시 정한다 — 이식 금지가 원칙이다.)
    MCS = env("SITE_MCS", 10)

    # HDBSCAN min_samples — **얼마나 보수적으로 볼 것인가.**
    #   한 점이 "밀집 지역에 있다"고 인정받는 데 필요한 이웃 수.
    #   크면 경계의 점을 쉽게 noise 로 던져 클러스터가 단단해지고, 작으면 잘 안 버리는 대신
    #   서로 다른 그룹이 붙는다. (severstal 실측 승자 조합은 mcs20/ms5 = mcs 의 1/4.)
    #   ★ 260731 사용자 지정: 3.
    MS = env("SITE_MS", 3)

    METHOD = env("SITE_METHOD", "leaf")        # leaf | eom
    # cluster_selection_epsilon. 이 거리 아래로 붙은 클러스터는 합친다. 0 이면 순수 HDBSCAN.
    #   ★ 260731 사용자 지정: 0 = 병합 없는 순수 HDBSCAN.
    EPS = env("SITE_EPS", 0.0)
    # 거리 척도. 임베딩이 L2 정규화돼 있어 euclidean 이 cosine 과 단조 동치다.
    METRIC = env("SITE_METRIC", "euclidean")

    # ★ GPU HDBSCAN(RAPIDS cuML) — opt-in, 기본 off. Windows 개발 머신은 cuML 자체가
    #   import 안 됨(RAPIDS 는 Linux/WSL2 전용). 사내 서버는 Ubuntu24 라 거기선 동작할
    #   수 있지만, CPU `hdbscan` 패키지와 클러스터 결과가 동일한지 아직 검증 안 했다.
    #   켜기 전에 같은 pool·다이얼로 CPU 결과(groups.csv)와 비교부터 하고 켜라.
    HDBSCAN_GPU = env("SITE_HDBSCAN_GPU", False)

    # ★ 260728: bootstrap group_stability(HDBSCAN 을 5회 더 돌려 재는 안정성 지표)는
    #   사용자 지시로 제거했다 — "사내는 답이 없다, 결과를 빨리 눈으로 보고 값을
    #   튜닝해가야 한다." HDBSCAN 은 이제 1회만 돈다. 무라벨 판단은 seed_noise/k/coherence.

    # 과병합 경고 임계 — 한 그룹이 자기 파티션의 이 비율 이상이면 review 표시.
    # ★ 전체 n 이 아니라 **자기 파티션** 기준이다 (분리 그룹핑을 켜면 전체 기준은 안 걸린다).
    OVER_MERGE_FRAC = env("SITE_OVER_MERGE_FRAC", 0.20)

    # noise 재배정. ★ 판정은 재배정 **전**(seed_noise)으로 한다 —
    # 재배정 후 noise 는 어떤 임베딩에든 나오는 바닥값이라(랜덤 head 와 0.03pp 차) 판정에 못 쓴다.
    REASSIGN = env("SITE_REASSIGN", "nearest_q90")   # none|nearest_q90|nearest_q80|assign_all

    # ★ palette 전처리 — wafer PNG(mode="P")는 index 0~7 이 결함 grade,
    #   그 위 index 는 border/background/invalid 다. 그 색이 clustering shortcut 이 될 수 있어
    #   흰색으로 지우는 전처리가 있다 (scripts/_common.mask_palette_non_grade_to_white).
    #   ⚠ **학습 때와 반드시 같아야 한다.** champion 과 May b4 는 마스킹 **없이** 학습됐다 —
    #     그 head 에 마스킹을 켜면 학습 때와 다른 입력이 되어 결과가 무의미해진다.
    #   PALETTE_PROBE=1 이면 step1 이 head 없는 frozen arm 을 마스킹 켠 판으로 하나 더 만들어
    #   **그 pool 에서 효과를 실측**한다. 이득이 크면 step2/3 를 마스킹 켜고 학습하면 된다
    #   (그 경우 UC_PALETTE_MASK=1 을 학습·채점 양쪽에 동일하게 준다).
    PALETTE_PROBE = env("SITE_PALETTE_PROBE", True)

    # ★ step1 전용 두 값 — 예전엔 `step1_zeroshot.py` 안에 **하드코딩**돼 있어서
    #   config 로도 env 로도 못 바꿨다 (SITE_SKIP_CHAMPION=1 을 줘도 champion 이 그대로
    #   돌았다, 260730 실측). config 로 끌어올린다.
    #   SKIP_CHAMPION: champion head 를 아예 안 돌린다(체크포인트가 없거나, frozen/z0
    #     기준선만 빠르게 보고 싶을 때). False 면 파일이 있을 때만 돈다.
    SKIP_CHAMPION = env("SITE_SKIP_CHAMPION", False)
    #   Z0_SEED: 랜덤 head(z0) 대조군의 시드. ★ 대조군이므로 arm 비교 중엔 고정해야 한다 —
    #     바꾸면 "학습이 이겼나"의 기준선 자체가 움직인다.
    Z0_SEED = env("SITE_Z0_SEED", 42)

    # ★ palette 마스킹 모드. 260728 ablation(6조건 x 5 disjoint subset x 2 다이얼 = 60셀,
    #   라벨 채점) 승자 = unify_border.
    #     unify_border    0~7 유지 + 경계 마커(11~30)를 **기본 칩경계색(10)으로 통일** + 배경 흰색
    #                     ARI 0.8300±0.0270  <- 채택
    #     grade_only      0~7 + 경계10 유지, 마커는 **흰색으로 삭제** (경계의 10.2% 가 끊긴다)
    #                     ARI 0.8089±0.0393
    #     unify_border_bg 위 unify_border + 배경 유지        ARI 0.7701±0.0285
    #     grade_noborder  경계까지 삭제                       ARI 0.7601±0.0485
    #     raw             원본                                ARI 0.5460±0.0749
    #   상세: docs/paper/BORDER_ABLATION_260728.md
    #   ⚠ 학습과 채점에 **같은 값**을 줘야 한다. 다르면 입력이 갈려 결과가 무의미해진다.
    PALETTE_MODE = env("UC_PALETTE_MODE", "unify_border")

    # ★ 위 PALETTE_MODE 를 **실제로 켜는 스위치**. 이게 없으면 MODE 는 죽은 설정이다
    #   (`grouping_deploy._PALETTE_MASK` 기본 "0", 학습기는 마스킹 코드 자체가 없었다).
    #   260731 실측 이전까지 step2/3/4/5 는 전부 `raw` 로 학습·채점하고 있었다 —
    #   그런데 같은 pool 에서 마스킹만 켠 arm 이 frozen 을 **24.84pp** 이겼고
    #   (seed_noise 33.67 -> 8.83, 잡음폭 2.28 의 10배), 60셀 ablation 도 같은 방향이었다
    #   (unify_border ARI 0.8300 vs raw 0.5460). 그래서 기본을 켬으로 둔다.
    #   ⚠ step1 은 여기에 **영향받지 않는다** — arm 마다 명시적으로 켜고 끈다.
    #     champion / May b4 head 는 마스킹 **없이** 학습된 것이라 켜면 입력 불일치가 되고,
    #     frozen 은 그것들의 기준선이라 같이 raw 여야 한다. `frozen_masked` 가 probe 다.
    #     반면 step2/3/4 는 **여기서 새로 학습**하므로 학습·채점 양쪽에 같이 켜면 불일치가 없다.
    PALETTE_MASK = env("SITE_PALETTE_MASK", True)

    # ★ 분리 그룹핑 — 파일명 코드마다 **따로** 클러스터링한다.
    #   사내 파일명: `AAQ729_00C_20_20260501_010000_83.0_17_PE_ENGINEER.png`
    #     `_` 로 나눈 필드 -> [AAQ729, 00C, 20, 20260501, ...]  코드는 **필드 1**.
    #     실측 분포 00C 1,259장 / 00P 1,001장.
    #   서로 다른 제품/라인은 애초에 같은 그룹이 될 수 없다. 한 덩어리로 돌리면
    #   다이얼 하나가 두 모집단을 동시에 맞춰야 해서 양쪽 다 나빠진다.
    #   summary.json 에 키별 표(partitions)가 따로 남는다 — 합산값만 보면
    #   한쪽 코드가 나빠도 다른 쪽에 묻힌다.
    #   ⚠ "언더바 사이 3글자"를 길이로 잡으면 안 된다 — 같은 파일명의 `PWQ` 도 3글자다
    #     (2,260장 중 742장이 3글자 필드를 2개 갖는다). 그래서 위치로 고정한다.
    #   "" = 분리 안 함 / "field:1" = 1번 필드 / 정규식(캡처그룹 1)도 가능
    PARTITION_BY = env("SITE_PARTITION_BY", "field:1")

    # Rule C 의 k 하한 퍼센타일. ★ 이건 k 입력이 아니라 **자기 run 안에서의 상대 기준**이다.
    # 이게 없으면 noise 최소화가 "전부 한 덩어리" 축퇴 해로 걸어간다.
    RULE_C_K_PERCENTILE = env("SITE_K_PERCENTILE", 75)


# ═══════════════════════════════════════════════════════════════════════════
class Composite:
    """그룹 composite map. `D:/project/mapviewer` 공식 규격.

    구조: palette index 의 **grade 빈도**를 쌓아 스칼라 맵을 만들고, 11색 색상표로 칠한다.
      값 = Σ(count[g]·NUM[g]) / Σ(count[g]·DEN[g])
      색 = quantile 0/10/…/100 에 색 하나씩 -> 선형 보간 -> min-max 로 정규화한 값에 매핑

    ★ 리사이즈 금지. chip 경계는 1px 이라 축소하면 통째로 사라진다 (1024/1600/800 실측).
    ★ chip 경계는 전부 idx10 회색 한 색. chip 안 bin 번호(idx9)는 제거된다.
    """

    # ── 색 ────────────────────────────────────────────────────────────────
    # mapviewer 와 **같은 스키마**의 JSON. 그쪽 파일을 그대로 가리켜도 된다:
    #   SITE_COLOR_LEGENDS=D:/project/mapviewer/logs/color-legends.json
    # 파일/스킴이 없으면 계산된 기본색(gb=round(255*(1-step/100)))으로 되돌아간다.
    # ★ 절대경로. 사내 색 정본이 따로 있으면 이 줄을 그 파일로 바꾸면 된다.
    #   ⚠ 이 프로젝트의 다른 경로와 달리 여기만 절대경로다. 그래서 **없으면 폴백한다**:
    #     ① 아래 경로 → ② 동봉 `deploy/color-legends.json` → ③ 계산 기본색
    #     폴백이 없으면 이 경로가 없는 서버에서 색이 조용히 바뀐다.
    #   mapviewer 정본을 그대로 쓰려면:
    #     "D:/project/mapviewer/logs/color-legends.json"
    #     단 그 파일의 composite.default 는 **원본 빨강 램프**라 "아래쪽 연하게"가 사라진다
    #     (그 파일에서 흑백인 건 composite 가 아니라 measure.default 다).
    COLOR_LEGENDS = env("SITE_COLOR_LEGENDS", COLOR_LEGENDS_PATH)
    # 위 파일의 composite.<scheme>. 색을 바꾸려면 **JSON 의 항목을 고치면 된다**
    # (또는 파일 맨 위 COLOR_LEGENDS_PATH/COLOR_SCHEME_NAME 을 바로 고쳐도 된다).
    #   default        = 아래쪽(낮은 quantile)만 흰쪽으로 당겨 바닥을 죽인 것
    #   anonymous      = mapviewer 기본 빨강 램프 그대로
    #   light_low_035 / light_low_070 = default 세기 변형 (비교용)
    #   fast_red_060 / fast_red_040 / fast_red_030 / fast_red_020 = light_low 의 반대 —
    #     아래~중간 quantile 을 더 빨리 빨갛게 당김 (숫자가 작을수록 세게). 260728 채택.
    COLOR_SCHEME = env("SITE_COLOR_SCHEME", COLOR_SCHEME_NAME)

    # ── 값 계산 ───────────────────────────────────────────────────────────
    # sq  = 원본 mapviewer   : 분자 grade^2      [0,1,4,9,16,25,36,49]
    #                          분모 WT_FACTORS   [1,1,2,3,4,5,6,7]
    # uc  = 채택안(method 3) : 분자 0,1 은 제곱 / 2 부터 2g  [0,1,4,6,8,10,12,14]
    #                          분모 전부 1 (WT0 만 손잡이)
    #       -> grade7/grade2 증폭이 12.25배에서 3.5배로 완화된다.
    #   ⚠ uc 분자에 WT_FACTORS 분모를 쓰면 2g/g = 2.000 으로 grade 2 이상이 전부 같아진다.
    # invalid 칩 처리. 테두리 idx11 + 내부 idx31.
    #   grade0  ★ invalid 를 grade 0 으로 카운트 (mapviewer 의 `>=14 -> grade0` 과 같은 취지).
    #           "측정 안 된 정상"이므로 그 자리가 가장 옅게 나온다.
    #   border  테두리=경계회색 / 내부=남의 웨이퍼 heat.  ← 예전 동작.
    #           invalid 클래스에서 결함 정보가 하나도 안 남는다 (실측: idx11 100% border,
    #           idx31 100% 남의 heat 로 흡수).
    #   exclude 카운트에서 제외(기권). 나머지 장으로만 평균.
    INVALID = env("SITE_COMPOSITE_INVALID", "grade0")

    METHOD = env("SITE_COMPOSITE_METHOD", "uc")      # uc | sq
    # 정상(grade0) 픽셀이 분모를 얼마나 채울지. 1.0 = 한 표, 0.0 = 결함 픽셀만
    # (작을수록 드문 결함이 드러나지만 바닥 노이즈도 같이 올라온다).
    WT0 = env("SITE_COMPOSITE_WT0", 1.0)

    # composite 에 쓸 **최대 멤버 수** (그룹 중심에 가까운 순).
    # composite 는 384 캐시를 못 쓰고 원본 6400x6400 을 멤버 수만큼 다시 디코드한다
    # (장당 0.68초). 상한이 없으면 2,000장 그룹 하나에 22.7분이 걸린다.
    # 중심 10장만 써도 그림이 거의 같다 — 평균이라 멤버가 늘어도 수렴한다.
    # 0 이면 전부 사용 (느리다).
    MAX_MEMBERS = env("SITE_COMPOSITE_MAX_MEMBERS", 10)

    # 비교용 square_average 도 같이 저장할지
    ALSO_SQUARE_AVERAGE = env("SITE_COMPOSITE_ALSO_AVG", False)

    # step1 에서 composite 를 **전 arm** 에 만들지. ★ 기본 켬.
    # composite 는 384 캐시를 못 쓰고 원본 6400x6400 을 다시 읽어 arm 당 ~70초를 먹는다
    # (임베딩은 5초). 한때 "이긴 arm 하나만" 으로 두어 step1 을 486->90초로 줄였지만,
    # 그러면 나머지 arm 폴더에는 medoid 만 남고 전 arm 이 k=0 이면 **하나도 안 생긴다**.
    # 그림이 없으면 결과를 눈으로 확인할 수가 없어서 기본을 켬으로 되돌린다.
    # 속도가 급하면 0 으로 꺼라 (그때는 이긴 arm 하나만 만든다).
    ALL_ARMS = env("SITE_COMPOSITE_ALL_ARMS", True)


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
