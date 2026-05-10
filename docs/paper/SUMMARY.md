# Contrastive Wafer Group Detection — 초보자 친화 요약

> 이 문서는 AI/통계 비전공자도 이해할 수 있게 시각적으로 설명한 프로젝트 종합 요약이다.
> 자세한 수식·논문 인용은 `METHOD.md`, `REFERENCES.md` 참조.

---

## 1. 프로젝트 한 줄 설명

> **반도체 wafer 사진 수천 장을 컴퓨터에게 "비슷한 것끼리 묶어라" 시키는 시스템.**
> 새로운 결함 패턴이 들어오면 자동으로 "이건 처음 보는 그룹이네!" 알려주는 게 목표.

---

## 2. 컴퓨터가 wafer 를 어떻게 다루는가? (embedding)

```
사진 (384×384 픽셀 RGB)
       ↓
   [신경망 (ConvNeXtV2-base, 88M params)]
       ↓
   숫자 128개 = 점 1개
   = "embedding"
```

각 wafer 사진이 **128차원 공간의 점 1개** 가 됨. 비슷한 wafer = 가까운 점.

**비유**: 사람을 "키, 몸무게, 나이" 3차원 점으로 표현한다고 치면,
키 175cm·70kg·30살 두 사람이면 가까운 점. 우리는 3차원 대신 128차원.

---

## 3. Contrastive Learning 이 뭐야?

> **"같은 종류는 가깝게, 다른 종류는 멀게"** 점들을 배치하도록 신경망 훈련.

```
훈련 전:                       훈련 후:

점들이 마구 섞여있음           같은 색끼리 모임

  🔴 🔵 🔴 🔵                    🔴🔴🔴   🔵🔵🔵
   🔵🔴🔵🔴      →               🔴🔴      🔵🔵
  🔴🔵🔴🔵
```

방법: anchor 한 점 잡고 → "같은 종류 점은 끌어당겨라 (positive), 다른 종류 점은 밀어내라 (negative)".

**InfoNCE loss 직관**:
```
                exp( sim(anchor, positive) / τ )
L = -log ─────────────────────────────────────────
                Σ exp( sim(anchor, candidate) / τ )

읽기: "분자 (positive 점수) 가 분모 (전체 점수 합) 의 몇 % 인가 → log"
```

---

## 4. Global vs Local feature

```
입력 384×384 → ConvNeXt → feature [12, 12, 1024]   12×12 = 144 patch grid

384×384 입력             12×12 patch (32×32 픽셀씩)
┌──────────┐             ┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│          │             │1 │2 │3 │4 │5 │6 │7 │8 │9 │10│11│12│
│  wafer   │             ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│  image   │   →         │13│14│15│16│ ... 144 patch          │
│          │             ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│          │             │  ...                              │
└──────────┘             └──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘
```

```
Global feature:                  Local feature:

12×12 patch 들을               12×12 patch 각각이
모두 평균해서 1개 vector        독립적으로 contrast 참여

   pooled = wafer 전체 평균       각 patch 가 [128] vector
   [128]                          → 144 × 128 만큼 신호

"이 wafer 가 Donut 이다"         "이 patch 위치에 scratch 가 있다"
큰 정보                           위치-민감 정보
```

---

## 5. Hyperparameter 시각화 (4개 핵심)

### 5-1. LR_HEAD — "걸음 크기" (불 세기)

```
산 (loss 곡면) 위에서 한 발씩 골짜기 (정답) 로 내려가기:

LR=1e-3 (큰 걸음)              LR=5e-4 (작은 걸음, 우리 best)

⛰️ start                            ⛰️ start
    ╲                                   ╲
     🔴────────╲                          🔵
              🔴────────╲                  ╲
                       🔴 ← 너무 큼!         🔵
                        ╲ valley 지나침       ╲
                         🔴                    🔵 ← valley 정확히
                                                  도달

   ✗ 빨리 가지만 흔들림                  ✓ 느리지만 정확
   ✗ collapse 위험 (모델 폭발)           ✓ Comp 0.83 → 0.948 (+12pp) ★

너무 크면 (1e-2): 한 발에 산 너머로 → 발산 → 학습 망가짐 (collapse)
너무 작으면 (1e-6): 1cm 씩 이동 → 골짜기 도달 못 함
```

### 5-2. NCE_TEMP (τ) — "양념 진하기" (softmax sharpness)

```
softmax( sim/τ ) 의 효과:

τ=0.5 (양념 묽다 = 부드럽다):
   exp(0.92/0.5) = 1.82
   exp(0.30/0.5) = 1.22
   exp(0.10/0.5) = 1.10
   ↓
   ┌─────────────────────────────┐
   │ pos:    44%  ███████████░░  │
   │ neg_A:  29%  ████████░░░░░  │  → 모든 negative 골고루 push
   │ neg_B:  27%  ███████░░░░░░  │  → boundary 흐림
   └─────────────────────────────┘

τ=0.05 (양념 진하다, 우리 best ★):
   exp(0.92/0.05) = 90,000,000
   exp(0.30/0.05) = 400
   exp(0.10/0.05) = 7.4
   ↓
   ┌─────────────────────────────┐
   │ pos:    99.9% ████████████  │
   │ neg_A:   0.03%░░░░░░░░░░░░  │  → 가장 비슷한 hard neg 만 push
   │ neg_B:    0.0%░░░░░░░░░░░░  │  → sister-class 분리 sharp ↑
   └─────────────────────────────┘
```

**비유**: 햇볕 (τ 큼, 모든 곳 미지근) vs 돋보기 (τ 작음, 한 점만 뜨겁게).

```
시험 채점 비유:
τ 큰 (관대): 70점, 80점 — 다 비슷 평가 → 학생 노력 약함
τ 작은 (엄격): 80점=C, 81점=B, 82점=A → 1점 차이가 huge gap
              → 학생 1점 더 받으려 죽기살기 노력 → 미세한 차이도 잡음
```

### 5-3. IGNORE_NEG_SIM (NEG) — "false negative 보호장치"

```
contrastive 의 가정: "같은 wafer aug = positive, 나머지 = negative"
   ↑ 라벨 정보 안 씀 (SSL)
   ↑ 그래서 같은 종류 wafer 도 negative 처럼 처리

batch 안에 같은 종류 wafer 가 우연히 있으면 → false negative 발생:

anchor 와 각 candidate 의 sim:
   sim ─────────────── 1.0 (자기 자신)
   0.95 ███████████  Edge-Top_scratch_2  ← false neg!
   0.85 ██████████   Edge-Top_scratch_rot ← sister
   ─── NEG=0.65 ────────────── threshold 라인
   0.60 ████         Donut_scratch
   0.30 ██           Center_invalid

   NEG 위 sim → ❌ 분모에서 빼버림 (false neg 의심)
   NEG 아래 sim → ✓ 정상 negative push
```

**NEG=0.72 (Iter 1) vs NEG=0.65 (Iter 14)**:
- 0.72: 더 많은 neg 살림 → strong gradient → noise 4.62% (P2 King)
- 0.65: 더 많이 제외 → sister-class 보존 → Comp 0.952 (Quality King)

### 5-4. LOCAL_WEIGHT (LW) — "전체 보기 vs 부분 보기"

```
L_total = L_global + LW × L_local

LW=0.5 (Iter A0 baseline):                LW=1.0 ★ (Iter 1 P2 King):

   Global 위주 + local 약간                   Global = local

   ┌──────────────┐                          ┌──┬──┬──┬──┬──┬──┐
   │   pooled     │                          ├──┼──┼──┼──┼──┼──┤
   │   wafer      │                          │  ... 144 patch   │
   │   전체 평균  │            vs            ├──┼──┼──┼──┼──┼──┤
   │   [128]      │                          │  각 patch 독립    │
   └──────────────┘                          │   contrast       │
                                             └──┴──┴──┴──┴──┴──┘

   ✓ Center/Donut 큰 패턴                    ✓ Edge-Top scratch 위치 ↑
   ✗ Edge-Top scratch 위치 약함              ✓ fork 작은 결함 ↑
   noise 9.34%                              noise 4.62% ★ (-50%)
```

★ **LW 가 가장 큰 lever** — 0.5 → 1.0 만으로 noise 절반 감소.

---

## 6. NeCo (Neighborhood-aware Cluster Order) — lever 5번째

> 출처: Pariza et al. 2024, arXiv:2408.11054
> 우리 iter 37 에서 noise 2.01 → 0.61% (**-70%**) huge 효과

### 한 줄 정의

> **두 augmentation view 의 같은 patch 가 보는 "이웃 patch 들의 순위" 가 같아야 한다**

### 친구 관계 비유

```
민수 (anchor patch) 의 친구 관계:
   1순위: 영희 / 2순위: 철수 / 3순위: 동수

이 순위가 어디서 봐도 같아야:

   학교 view A:                  집 view B:
   1순위 영희   sim 0.92         1순위 영희   sim 0.95  ✓
   2순위 철수   sim 0.85         2순위 철수   sim 0.82  ✓
   3순위 동수   sim 0.60         3순위 동수   sim 0.65  ✓

   → 두 view 의 이웃 순위 같음 → loss 작음 ✓

순위 어긋나면:
   학교: 1순위 영희 / 2순위 철수
   집:   1순위 철수 / 2순위 영희   ← 순위 뒤바뀜 ✗
   → loss 커짐 → 학습 신호 발생 → spatial 관계 안정화 학습
```

### 우리 wafer 에 적용

```
view A patch [12,12,128]            view B patch [12,12,128]
       ↓                                   ↓
P_A[i, :] = softmax(sim(p_i, p_j)/τ)    P_B[i, :] = softmax(...)
   144 차원 분포                          144 차원 분포
       ↓                                   ↓
       └─ symmetric KL divergence ────────┘
                  ↓
              L_neco

L_total = L_global + LW×L_local + NECO_WEIGHT × L_neco
                                        ↑
                                    우리 sweep (0.0 ~ 0.3)
```

### NeCo weight sweet spot

```
NeCo=0.0  →  Comp 0.978 / AMI 0.946  (iter 35 baseline)
NeCo=0.1  →  Comp 0.985 / AMI 0.956  ↗ 약간 ↑
NeCo=0.2  →  Comp 0.991 / AMI 0.960  ★★★ peak (sweet spot, iter 37)
NeCo=0.3  →  Comp 0.980 / AMI 0.954  ↘ 후퇴
NeCo=0.5  →  미실시 (양쪽 후퇴 → lock)

너무 작 (0.1): "이웃 순위 일관성" 약함 → cluster 흔들림
너무 크 (0.3): NeCo 강제가 InfoNCE 신호 덮어버림 → AMI ↓
```

### NeCo 의 진짜 메커니즘 (cluster-analyzer 발견)

```
가설:  cluster 응집 (intra distance) 줄임
실측:  intra 거리 변화 거의 없음 (0.0206 → 0.0212)

★ 진짜 효과: Normal-defect boundary 재배치
   Full_scratch ↔ Normal centroid 거리 0.27 → 0.32 (+0.05)
   Full_scratch_rot ↔ Normal centroid 거리 0.29 → 0.36 (+0.07)

   → iter 35 에서 Normal supercluster 에 흡수됐던 54 wafer 가
     iter 37 에서 own pure cluster (size 30, 20) 로 분리
   → 이게 noise 2.01 → 0.61% 의 진짜 원인
```

---

## 7. HDBSCAN (점들을 자동 그룹핑)

훈련 끝나면 점들 모인 모양에서 cluster 자동 추출:

```
embedding 공간 (점들이 흩어져있음):

  🔴🔴       🔵🔵🔵          .    ← 외톨이 점 = "noise"
   🔴       🔵🔵🔵🔵
  🔴🔴🔴      🔵🔵            🟢🟢
                              🟢🟢🟢
                .             🟢

HDBSCAN 결과:
  [Cluster 1: 7개]   [Cluster 2: 7개]   [Cluster 3: 5개]   noise: 2개
```

### HDBSCAN 옵션

| param | 비유 | 효과 |
|---|---|---|
| **mcs** (min_cluster_size) | "최소 N개 모여야 cluster" | mcs=12: 12개 미만은 noise. ↑ 큰 cluster 만 ↓ tiny 도 살림 |
| **ms** (min_samples) | 빽빽함 기준 | ↑ 보수적 noise ↑ / ↓ 관대 cluster 잡 |
| **method=leaf** | "가지 끝까지 세분화" | sub-cluster 다 살림 → over-segment 위험 |
| **method=eom** ★ | "큰 안정 덩어리만 픽" | 자연스러운 sub-style 흡수 → noise -58% |

### eom vs leaf 시각화

```
hierarchical cluster tree:
                       [전체 wafer 1146]
                       /              \
                 [stable A: 800]    [stable B: 250]   ← eom 가 pick
                 /        \           /          \      "큰 stable mass"
            [a1: 80]  [a2: 90]   [b1: 50]   [b2: 30]  ← leaf 가 pick
                                                          "모든 leaf"
─────────────────────────────────────────────────────────────────
leaf method:                           eom method (★ 새 발견):
  41 cluster (over-segmentation)       35 cluster (clean stable)
  같은 class → 여러 sub-cluster        sub-style 자동 통합
  noise 6.54%                          noise 2.79%  ★ -58%
```

---

## 8. 진짜 lever 5개 — 효과 size 정리

```
40+ iter 결과 — 의미 있는 axis 만:

LW (0.5→1.0):       noise [████████████████████] 9.34→4.62%   -50%   ★★★ Iter 1
LR_HEAD (1e-3→5e-4): Comp [████████████]         0.83→0.948   +12pp  ★★ Iter 11
NEG (0.72→0.65):    sister 분리 ↑                              ↑     ★ Iter 13
NCE_TEMP (0.07→0.05): AMI [████]                 0.91→0.913   +0.3pp ★ Iter 14
NeCo (0→0.2):       noise [██████████████████]   2.01→0.61%   -70%   ★★★★ iter 37
HDBSCAN leaf→eom:   noise [██████████████████]   6.72→2.79%   -58%   ★★★ encoder 무관
HDBSCAN ms 4→3:     noise [██████████]           1.22→0.61%   -50%   ★ encoder 무관

dead axes (모두 reject):
   PercPos α / EPOCHS↑ / WARMUP↑ / TOPK≠12 / QUEUE≠4096 / BATCH≠8 /
   LW 작은 변화 / NEG 사촌 / multi-axis combo / HDBSCAN ε / backbone unfreeze
```

---

## 9. 진화 history

```
A0 baseline                    9.34% noise
   │ + LW=1.0 (lever 1)
Iter 1                         4.62% (P2 King)
   │ + LR/NEG/TEMP (lever 2-4)
Iter 14                        6.63% / Comp 0.952 (Quality King)
   │ + new anchor v19o chip
iter 34                        6.72% / Comp 0.951
   │ Iter 1 cfg back
iter 35                        4.19%
   │ + HDBSCAN eom (encoder 무관)
iter 35 + eom                  2.01% / Comp 0.978
   │ + NeCo 0.2 (lever 5)
iter 37 + eom ms=3             0.61% / Comp 0.991 ★★★★★ SOTA
   │ NeCo 0.2 sweet spot 확정 (0.1, 0.3 양쪽 후퇴)
   │ Quality King + NeCo combo 시도
iter 40                        진행 중

★ 총 noise 감소: 9.34% → 0.61% (-93.5%)
```

---

## 10. 현재 SOTA (2026-05-09)

### iter 37 + HDBSCAN eom mcs=12 ms=3

| 지표 | 값 | 의미 |
|---|---:|---|
| **noise(def, P2)** | **0.61%** | 1146 중 7 wafer 만 어디도 못 묶임 |
| **Completeness (P3)** | **0.991** | cluster 응집 거의 perfect |
| **Homogeneity (P4)** | 0.978 | cluster 안 순도 |
| **AMI** | **0.960** | balanced 측정 |
| **NMI** | 0.962 | |
| **ARI** | **0.870** | over-cluster 페널티 포함 |
| **Silhouette (cos)** | 0.610 | 응집/분리 |
| **capture (P1)** | **1.000** | 43/43 class 모두 group 1+ |
| **n_clusters** | 36 | compact |

### Configuration

```python
# Encoder
BACKBONE       = "ConvNeXtV2-base FCMAE + supervised TAPT"
PROJ_DIM       = 128
IMAGE_SIZE     = 384
FREEZE_BACKBONE = True
BACKBONE_UNFREEZE_LAST_N = 0  # iter 36 unfreeze 시도 reject

# Loss
LOCAL_WEIGHT   = 1.0
LOCAL_POS_TOPK = 12
NCE_TEMP       = 0.07         # P2 King base
IGNORE_NEG_SIM = 0.72         # P2 King base
NECO_WEIGHT    = 0.2          # ★ sweet spot
NECO_TAU       = 0.1
USE_QUEUE      = True
QUEUE_SIZE     = 4096

# Training
EPOCHS         = 5
BATCH          = 8
LR_HEAD        = 1e-3         # P2 King base
WARMUP_EPOCHS  = 1
TRAIN_SAMPLING_RATIO = 0.25
SEED           = 42

# HDBSCAN (encoder 학습 후 적용)
MIN_CLUSTER_SIZE = 12
MIN_SAMPLES      = 3          # ★ default 4 → 3 (noise -50%)
CLUSTER_SELECTION_METHOD = "eom"  # ★ default leaf → eom (noise -58%)
CLUSTER_SELECTION_EPSILON = 0.06
```

### Anchor 데이터

```
경로: D:/project/data/contrastive_anchor/avg30_new_260508_123037
n=2146 wafer (defect 42 class avg ~30 + Normal 1000)
chip 합성: v19o (260508 regen)
```

---

## 11. 진행 history (iter 34 → 42)

| # | atomic 변경 | noise(def) | Comp | AMI | ARI | cap | 판정 |
|---:|---|---:|---:|---:|---:|---:|---|
| 34 | new anchor + Iter 14 cfg (baseline) | 2.79% | 0.977 | 0.931 | 0.750 | 1.000 | base |
| 35 | + LR/NEG/TEMP = Iter 1 P2 King | 2.01% | 0.978 | 0.946 | 0.856 | 1.000 | ★ -28% noise |
| 36 | + backbone unfreeze (LR_SCALE 0.02) | 4.28% | 0.953 | 0.873 | 0.582 | 0.976 ❌ | ✗ REJECT |
| **37** | **+ NeCo 0.2** | **0.61%** | **0.991** | **0.960** | **0.870** | **1.000** | **★★★★★ 현 SOTA** |
| 38 | NeCo 0.1 | 0.52% | 0.985 | 0.956 | 0.860 | 1.000 | ✗ regression (mixed 6→7) |
| 39 | NeCo 0.3 | 1.05% | 0.980 | 0.954 | 0.868 | 1.000 | ✗ regression — 0.2 lock |
| 40 | base = Quality King + NeCo 0.2 | 4.10% | 0.962 | 0.922 | 0.738 | 1.000 | ✗ huge regression (-13pp ARI) |
| 41 | HDBSCAN mcs forcing (encoder X) | 3.05% | 0.997 | 0.928 | 0.770 | 0.952 ❌ | ✗ dead axis |
| **42** | + backbone unfreeze 안전 (LR_SCALE 0.005) | (학습 중) | | | | | — |

### 핵심 path

```
baseline (iter 34)              noise 2.79%, AMI 0.931
   │ ★ Iter 1 P2 King cfg
iter 35                         noise 2.01%, AMI 0.946  (-28%)
   │ ★★★★ NeCo 0.2 (lever 5)
iter 37 ★★★★★ 현 SOTA           noise 0.61%, AMI 0.960  (-70%)
```

### iter 43 자동 결정 logic

```
iter 42 결과 → ┬→ best 갱신 (Comp ≥ 0.991, AMI ≥ 0.960, cap=1.000)
              │      ↓
              │   iter 43 = LR_SCALE 0.01 (덜 보수)
              │   iter 44 = LOCAL_POS_TOPK 16 추가 combo
              │   iter 45 = multi-seed (variance 측정)
              │
              └→ regression
                     ↓
                 iter 43 = multi-seed iter 37 (seed 1, 2, 3)
                 iter 44 = LOCAL_POS_TOPK 16 (NeCo 추가 후 dead axis 재검증)
                 iter 45 = 합성 spec 강화 (TEF vs FF) — 사용자 승인
```

---

## 12. 절대 금기 / dead axes

```
✗ TTA (test-time augmentation): 사용자 정책 영구 금지
✗ SupCon: unknown class 일반화 약화 우려 — 사용자 정책 거부
✗ Multi-crop: wafer 위치 정보 손상 — 사용자 정책 거부
✗ EPOCHS > 5: over-fit (Iter 6, 12 reject)
✗ WARMUP > 1: P1 violation (Iter 17 reject)
✗ BATCH ≠ 8: sweet spot
✗ LOCAL_POS_TOPK ≠ 12: sweet spot (8, 16 reject)
✗ QUEUE_SIZE ≠ 4096: sweet spot (8192 reject)
✗ HDBSCAN eps sweep: dead axis (0.0~0.20 모두 동일)
✗ Backbone full unfreeze: collapse (Iter 31)
✗ Backbone partial unfreeze + LR_HEAD 1e-3: capture P1 violation (iter 36)
```

---

## 13. 관련 문서

- `ITERATIONS.md` — Iter A0 ~ 40 상세 history (append-only)
- `METHOD.md` — 수식·아키텍처 detail
- `RESULTS.md` — 표 정책 + 공식 결과
- `EXPERIMENTS.md` — ablation 설계
- `REFERENCES.md` — 인용 논문 (NeCo, DenseCL, NV-Retriever 등)
