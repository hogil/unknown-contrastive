# deploy/ — 사내 서버 무라벨 배포 파이프라인

배포 사다리 ①~④ 를 **순서대로** 실행하는 단계별 스크립트.
목적: 사내 실데이터에서 **unknown grouping 으로 신규불량 발생 시 감지**.

- 경로는 **전부 프로젝트 루트 기준 상대경로**. 폴더를 통째로 어디에 두든 그대로 돈다.
- 각 실행 파일 맨 위에 **`class Config`** 가 있다. **환경변수가 있으면 그것, 없으면 default.**
- 라벨은 **한 줄도 안 쓴다.** 사내엔 없으니까.

---

## 0. 보내야 할 체크포인트

코드만 먼저 올렸다. 아래 파일들을 프로젝트 폴더 안에 그대로 두면 된다.

### 필수 1개 — backbone

| 두는 위치 (상대경로) | 원본 | 크기 |
|---|---|---|
| `weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth` | ImageNet FCMAE ConvNeXtV2-base | ~350MB |

**모든 step 이 이걸 쓴다.** 없으면 아무것도 안 돈다.

### ★ 최우선 — May 배포본 `contrastive_b4.pt`

**260727 실측 결과 이게 가장 중요한 체크포인트다.** 아래 champion head 보다 우선순위가 높다.

| 두는 위치 | 원본 | 크기 |
|---|---|---|
| `weights/b4_may/contrastive_b4.pt` | `<failure_agent>/checkpoints/contrastive_b4.pt` | ~356MB |

```bash
python deploy/extract_b4.py    # backbone / proj 분리 + 검증 -> step1 이 자동 인식
```

2개 pool 에서 **b4 의 backbone 을 head 없이 쓰는 arm 이 전부 1위**였다:

| pool | arm | P1 | seed_noise | ARI |
|---|---|---|--:|--:|
| clean546 | champion (2-head) | 8/9 | 37.18 | 0.711 |
| clean546 | **b4_backbone only** | 8/9 | 41.94 | **0.904** |
| anchor | champion (2-head) | 30/43 | 63.50 | 0.692 |
| anchor | **b4_backbone only** | **43/43** | **47.30** | **0.807** |

anchor 에서 champion 은 **frozen(39/43)보다도 나쁘다**. b4 backbone 은 **P1 만점**이다.
b4 자신의 head 를 붙이면 오히려 나빠지므로(ARI 0.904→0.706) **backbone 만 쓰는 arm 이 본命**이다.

가설: b4 backbone 은 wafer 데이터로 TAPT 된 것이라 ImageNet FCMAE 보다 도메인이 가깝다.
사내도 wafer 이므로 이 이점이 유지될 가능성이 높다 — 다만 step1 이 그 pool 에서 다시 판정한다.

### 선택 2개 — champion projection head (**보내려면 2개 다**)

| 두는 위치 (상대경로) | 원본 경로 | 크기 |
|---|---|---|
| `weights/champion/proj_s42_ep20.pt` | `runs/sweep/abl_sw_t20_B4_260724_102757/checkpoints/proj_ep20.pt` | ~5MB |
| `weights/champion/proj_s1_ep18.pt` | `runs/sweep/abl_best_s1_B4_260724_111053/checkpoints/proj_ep18.pt` | ~5MB |

**왜 2개인가**: 배포 champion 은 이 두 head 의 **concat + L2 앙상블**이다.
실제 배포 지점(reassign 후)에서 비교했을 때 앙상블이 단일 head 를 이겼다
(P1 동률·잡는 클래스도 동일, noise 차이는 잡음폭 안, 그다음 축 Comp +0.039 / Hom +0.040 / ARI +0.074).
**하나만 보내면 step1(사다리 ①)이 반쪽이 된다.**

- **없어도 step0/2/3/5 는 다 돈다.** step1 만 `frozen`/`z0` 비교로 축소된다.
  그 경우 `SITE_SKIP_CHAMPION=1` 로 두면 경고 없이 진행한다.
- 나중에 받으면 step1 만 다시 돌리면 된다.

### 선택 — May 배포본 `contrastive_b4.pt` (사다리 ① 비교용)

| 두는 위치 | 원본 | 크기 |
|---|---|---|
| `weights/b4_may/contrastive_b4.pt` | `<failure_agent>/checkpoints/contrastive_b4.pt` | ~356MB |

```bash
python deploy/extract_b4.py      # backbone / proj 로 분리 -> step1 이 자동 인식
```

★ **b4 는 자체 backbone 을 갖는 독립 arm 이다.** 번들 안의 backbone 378개 텐서가
같은 폴더 `backbone.pth` 와 **전부 다르다**(실측). 그래서 b4 proj 를 우리 FCMAE 위에
올리면 안 된다 — 학습 때와 다른 feature 분포를 먹이게 된다.
`extract_b4.py` 가 이 검증을 자동으로 해준다.

champion(FCMAE + 2 head) 과 b4(자체 backbone + 1 head) 중 어느 쪽이 사내 데이터에서
나은지는 **재봐야 안다**. step1 이 두 arm 을 같은 다이얼로 비교해서 판정까지 출력한다.

### 선택 1개 — TAPT backbone (사다리 ④ 용, 라벨 필요)

| 두는 위치 | 비고 |
|---|---|
| `weights/tapt/backbone_tapt.pth` | 없으면 step4 는 안내만 출력하고 skip |

사내 라벨이 없으면 못 만든다. **①~③ 으로 충분하면 안 보내도 된다.**
(우리 실측: new-domain 에서는 TAPT 가 오히려 불리했다.)

---

## 1. 실행 순서

```bash
# 사내 이미지를 프로젝트 안에 두거나 경로를 지정
export SITE_IMAGE_ROOT=data/site_images     # 기본값
export SITE_MCS=20                          # 몇 장 이상 뭉치면 그룹으로 볼지 (k 아님)
export SITE_MS=5                            # min_samples (보통 mcs 의 1/4)

python deploy/step0_prepare.py     # manifest + pool 기하 -> 권장 다이얼
python deploy/step1_zeroshot.py    # 사다리 ① 학습 0.  frozen / z0 / champion 3-arm
python deploy/step2_recipe.py      # 사다리 ② B4 레시피 학습 + Rule C epoch 선택
python deploy/step3_sweep.py       # 사다리 ③ 레시피 sweep + 무라벨 셀 선택
python deploy/step4_tapt.py        # 사다리 ④ (TAPT backbone 있을 때만)
python deploy/step5_temporal.py    # ★ 최종 산출물: 시간축 신규불량 감지
```

각 step 은 앞 step 의 결과(`runs/deploy/stepN_result.json`)를 자동으로 물려 쓴다.
중간에 끊겨도 이미 끝난 step 은 다시 안 돌려도 된다.

### 가장 빠른 확인 경로

시간이 없으면 **step0 → step1 → step5** 만 돌려도 된다.
학습 없이 frozen 기준선으로 신규불량 감지가 되는지부터 본다.

---

## 2. 불량 종수(k)는 입력하지 않는다

**k 는 모르는 게 전제다.** HDBSCAN 을 쓰는 이유가 바로 k-free 라서고, k 를 입력으로 받으면
그 이유가 사라진다. 대신 정하는 건 이것 하나다:

```
SITE_MCS = 20   # "몇 장 이상 뭉쳐야 하나의 그룹으로 볼 것인가" = HDBSCAN min_cluster_size
SITE_MS  = 5    # min_samples — 보수성. 보통 mcs 의 1/4
```

이건 클래스 수가 아니라 **보고 가치가 있는 최소 그룹 크기**다. "20장쯤 모이면 들여다볼
가치가 있다" 는 운영 판단이고, 불량이 몇 종인지 몰라도 답할 수 있다.
HDBSCAN `min_cluster_size` 의 원래 의미 그대로다.

### 그것도 정하기 싫으면 자동

| pool | 아는 정답 | DBCV | **stability(채택)** | ARI 최대(라벨 필요) |
|---|--:|--:|--:|--:|
| severstal | 20 | 15 | **20 ✓** | 60 (k=2 병합 치팅) |

ARI 최대화는 **전부 한 덩어리로 병합하는 축퇴 해**로 걸어간다(mcs60 에서 k=2).
안정성 기준은 그 함정을 피하면서 정답을 집었다.

⚠ 다이얼이 결과에 거의 영향 없는 pool 도 있다 — clean546 은 mcs 5~44 에서 ARI 가
0.63~0.70 으로 평탄했다. AUTO 모드는 스캔 표를 함께 출력하니 민감도를 눈으로 확인하라.
표가 평탄하면 아무 값이나 써도 되고, 급격하면 그 pool 은 다이얼이 중요하다는 뜻이다.

## 3. 주요 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `SITE_IMAGE_ROOT` | `data/site_images` | 사내 이미지 루트 (flat/중첩 둘 다) |
| `SITE_MCS` | `20` | HDBSCAN min_cluster_size = 그룹으로 볼 최소 장수 (k 아님) |
| `SITE_MS` | `5` | HDBSCAN min_samples (보수성) |
| `SITE_OUT_ROOT` | `runs/site` | 산출 루트 |
| `SITE_BACKBONE` | `weights/convnextv2_base...pth` | frozen backbone |
| `SITE_CHAMPION_PROJ` | `weights/champion/proj_s42_ep20.pt,weights/champion/proj_s1_ep18.pt` | 콤마 구분 2개 |
| `SITE_DEVICE` | `cuda` | `cpu` 가능 (느림) |
| `SITE_SEEDS` | `42,1,2` | step2 학습 seed. 급하면 `42` |
| `SITE_EPOCHS` | `20` | 학습 epoch |
| `SITE_CELLS` | `lr008,lr002,q32768,...` | step3 스윕 셀 순서 |
| `SITE_MCS` / `SITE_MS` | `0` (자동) | 다이얼 수동 고정 |
| `SITE_TIME_MODE` | `by_dir` | step5 시간 배치: 폴더명 / `by_mtime` |
| `SITE_N_CALIB` / `SITE_N_BG` | `4` / `4` | 캘리브레이션 / 배경(FAR 측정) 배치 수 |

---

## 3-1. step 별 사용 모델

| step | backbone | head (projection) | 모델을 실제로 로드하나? |
|---|---|---|---|
| **step0** prepare | — | — | ❌ 안 함. manifest 스캔 + 다이얼 결정(config 값 그대로) + 디코드 캐시 생성만. |
| **step1** zeroshot | `frozen`/`z0`/`champion`/`frozen_masked` arm = **FCMAE backbone**(`SITE_BACKBONE`). `contrastive_b4(May)`/`b4_backbone_only` arm = **b4 자체 backbone**(`SITE_B4_BACKBONE`, FCMAE 와 다른 독립 가중치) | `frozen`/`frozen_masked`: 없음. `z0_xN`: 랜덤 초기화(학습 안 함, champion 과 head 수만 맞춤). `champion`: `SITE_CHAMPION_PROJ` 2개 concat+L2 앙상블. `contrastive_b4(May)`: `SITE_B4_PROJ`. `b4_backbone_only`: 없음 | ✅ arm 마다 로드만 하고 **아무것도 학습 안 함**(사다리① = 학습 0) |
| **step2** recipe | **FCMAE backbone**(`SITE_BACKBONE`) 고정 — 모든 seed 가 여기서 시작 | `_may_ablation.py`(B4 레시피)로 **새로 학습**한 head. seed 마다 독립 학습 후 Rule C 로 epoch 선택. 최종은 seed 앙상블(concat+L2) | ✅ 학습(사다리②) — FCMAE 위에 head 만 새로 얹는다, backbone 은 그대로 |
| **step3** sweep | **FCMAE backbone**(`SITE_BACKBONE`) 고정 — step2 와 동일 | 레시피 hparam(`SITE_CELLS`, lr/temp/queue 등)만 셀마다 바꿔 **새로 학습**. 무라벨 규칙으로 셀 선택 | ✅ 학습(사다리③) — step2 와 같은 backbone, recipe 만 다르다 |
| **step4** TAPT | **TAPT backbone**(`SITE_TAPT_BACKBONE`, 없으면 안내만 하고 skip) — FCMAE 대신 이걸로 교체 | step3 와 동일한 sweep 을 이 backbone 위에서 재실행 | ⚠ TAPT backbone 이 있을 때만(사다리④, 라벨 필요) |
| **step5** temporal | **FCMAE backbone**(`SITE_BACKBONE`) 고정 | `SITE_TEMPORAL_PROJ` 미지정 시 **자동 탐색**: step3 승자 체크포인트 → (없으면) step2 체크포인트 앙상블 → (그것도 없으면) head 없이 frozen backbone 만 | ✅ step2/3 산출을 물려 씀. 직접 학습은 안 함 |

- **`SITE_BACKBONE`(FCMAE) 은 step0 을 제외한 전 step 의 기본값**이다. b4/TAPT 는 각 arm/step 안에서 **명시적으로 교체**될 때만 다르다.
- step1 의 `contrastive_b4(May)` 와 `b4_backbone_only` 는 FCMAE 와 **섞이지 않는 독립 arm** — `extract_b4.py` 가 검증한 별도 backbone 가중치를 쓴다(`weights/b4_may/`).
- step2/step3 는 **head 만 새로 학습**한다. backbone 자체(FCMAE)를 fine-tune 하지 않는다(frozen).
- step4 만 backbone 을 바꾼다. 나머지 step 은 전부 같은 FCMAE 위에서 비교된다.

---

## 4. 결과 읽는 법 — 지키지 않으면 결론이 틀어진다

1. **판정은 `seed_noise`(reassign 전)로 한다.**
   reassign 후 noise 는 **어떤 임베딩에든 나오는 바닥값**이다. 랜덤 head 에 같은 후처리를
   걸면 champion 과 0.03pp 차이까지 붙는다.
2. **대조군은 frozen 이 아니라 `z0`(랜덤 head)다.** step1 이 자동으로 만든다.
   랜덤 투영만으로 지표가 잡음폭 밖으로 움직이는 pool 이 실제로 있다.
   z0 를 못 이기면 **"학습이 개선했다"고 쓸 수 없다.**
3. **잡음폭 안 차이는 차이가 아니다** — noise ±2.28pp / ARI ±0.019 / Hom ±0.005 / Comp ±0.033.
   Comp 가 가장 관대하니 **Comp 단독으로 통과 판정하지 마라.**
4. **사다리는 낮은 순위가 우선.** ④가 ③과 잡음폭 안이면 **③을 쓴다.**
5. **step5 의 FAR 은 "배경 구간에 신규불량이 없다"는 가정 위의 값**이다.
   그 구간에 실제 신규불량이 있었으면 과대평가된다 — 현업과 확인해라.

---

## 5. 알려진 함정

- **`REPRO_NEG` 아니라 `REPRO_IGNORE_NEG_SIM`** — 오타 시 조용히 무시되고 셀이 중복 실행된다.
  (`_site_common.train_env()` 가 올바른 이름을 쓴다.)
- **`REPRO_WORKERS` 넘기지 마라** — Windows spawn 으로 학습이 죽는다.
- **cp949 `UnicodeEncodeError`** — `run()` 이 `PYTHONIOENCODING=utf-8` 을 강제한다.
  `logger` 경로는 exit code 0 으로 삼켜지지만 `print` 경로는 exit 1 을 낸다.
  그래서 **성공 판정을 exit code 로 하지 않고 산출 파일 존재로 한다.**
- **결과 폴더를 지우지 마라.** 같은 이름으로 다시 돌리지 말고 `SITE_OUT_ROOT` 를 바꿔라.

---

## 6. 근거 문서

| 문서 | 내용 |
|---|---|
| `docs/ABSOLUTE_RULES.md` | 정본 — 목적·사다리·데이터 경계 |
| `docs/GOAL_AND_LADDER.md` | 측정 규율 + 에이전트 설계 구조도 |
| `docs/HANDOFF_260726.md` | 실행 상태 스냅샷 + 실측 수치 |
| `runs/severstal/dial_sweep/REPORT.md` | 168셀 다이얼 진단 |
| `docs/audit/z0_control_audit_260726.md` | z0 대조군 감사 |
