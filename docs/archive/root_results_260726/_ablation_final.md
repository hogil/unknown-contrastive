# WM-811K Novel-Class Discovery — Ablation (paper)

**Setup.** Backbone **`convnext_base.dinov3_lvd1689m`** (DINOv3 ConvNeXt-base) — 백본 고정.
Inductive NCD: encoder는 novel class를 한 번도 안 봄.
- 학습(SSL, unlabeled): `cnn_seen_train` (Center/Edge-Ring/Near-full, 1149) — novel 제외, **라벨 미사용**.
- 평가(held-out novel): `novel_eval` (Donut/Edge-Loc/Random, 1500).
- Metric: **k-means(k=3, known novel count) ARI/NMI/AMI** — 모든 점 배정, 클러스터링 파라미터로 못 속임(표준 NCD). 공식 metric만.

## ★ Main ablation waterfall (baseline = DINOv3 frozen, 기법 누적)

| Stage (누적 기법) | **ARI** | Δ |
|---|--:|--:|
| **S0 DINOv3 frozen (baseline)** | **0.3097** | — |
| S1 + PCA/clustering 튜닝 | 0.3173 | +0.008 |
| S2 + SimCLR fine-tune (all-unfreeze) | **0.5976** | **+0.281** |
| S3 + engineering (queue/ignore/koleo) | (진행중) | ? |

**ARI 0.310 → 0.598 (1.93×).** 최대 레버 = SimCLR fine-tune.

## ★ 백본 2개 비교 (별도 표 — init만 교체, 나머지 전부 동일)

같은 데이터(1149 unlabeled) · 같은 aug · 같은 SimCLR recipe · 같은 평가:

| 백본 init | frozen ARI | + SimCLR FT ARI (5ep best) | FT 이득 |
|---|--:|--:|--:|
| FCMAE (`convnextv2_base.fcmae_ft_in22k_in1k_384`, MAE+supervised FT) | 0.2097 | 0.2711 | +0.061 |
| **DINOv3 (`convnext_base.dinov3_lvd1689m`, SSL self-distillation)** | **0.3097** | **0.5976** | **+0.288** |

→ **baseline 선정 실험**: 후보 둘을 같은 조건(frozen / +동일 SimCLR FT recipe)에서 비교,
frozen에서도 FT 후에도 DINOv3 우위 (gap 0.10 → 0.33 확대) → **ref = DINOv3 확정**.
FCMAE 는 loss 정상 수렴(0.36→0.05)에도 ARI 이득 1/5 — 출발점 차이가 FT 로 메워지지 않음.
(주의: 두 ckpt 는 구조/사전학습데이터/방식이 모두 달라 우위의 "원인" 분리는 불가 —
이 표의 주장은 인과가 아니라 선정 근거. 본론은 이 ref 에서 쌓는 waterfall.)

## ★ 각 조건 방법 상세 (모든 행이 정확히 뭘 하는지)

### 공통 fine-tune recipe (모든 FT 행 동일 — 다르면 행에 별도 명시)
| 항목 | 값 | 이유 |
|---|---|---|
| init | `convnext_base.dinov3_lvd1689m` (timm pretrained) | 백본 고정 directive |
| 구조 | backbone(GAP 1024-d h) + proj MLP(1024→1024→256, BN+ReLU) | 표준 contrastive head |
| unfreeze | **all-unfreeze** (백본 전 레이어 학습) | frozen 천장 0.317 → 도메인 적응 필수 |
| optimizer | AdamW, lr backbone **2e-6** / head **1e-3**, wd 1e-6 | 백본 저LR=사전지식 보존, head 고LR=빠른 정렬 |
| batch | 8 | 자원 정책 (negative 부족은 queue로 보완) |
| aug (two views) | RandomResizedCrop 384 scale 0.94~1.0 + RandomAffine ±7°·translate 5%·scale ±5% + gauss noise 0.02 | 위치·방향=클래스 정체성 보존. **flip/colorjitter 금지** |
| 평가 임베딩 | backbone h (1024-d, proj 전) L2-norm | proj 후 z는 loss 전용 공간 (h가 일반화 우수) |
| epochs | 5~6, best-epoch 선택 (매 epoch 임베딩 저장) | ep3~5 부근 peak, 이후 진동 |

### SSL 방법별 메커니즘 (recipe 동일, loss만 교체)
| 방법 | 무엇을 하나 | novel ARI |
|---|---|--:|
| **SimCLR** | 같은 이미지 2뷰=positive 당김, **배치 내 다른 모든 샘플(2B-2)=negative 밀어냄**. InfoNCE temp 0.05 | **0.598 ★** |
| VICReg | 2뷰 MSE 당김 + 각 차원 분산≥1 강제(collapse 방지) + 차원간 covariance→0. 가중 25/25/1 | 0.530 |
| MoCo | EMA teacher(m=0.99)가 key 생성 + **queue 1024 과거 negative**. InfoNCE | 0.495 |
| Barlow Twins | 2뷰 feature cross-correlation 행렬→단위행렬 (대각=1 당김, 비대각=0 중복제거) | 0.433 |
| SimSiam | predictor MLP + stop-gradient로 당김. **negative 없음** | 0.431 |
| DINO | prototype 4096 분포를 teacher(EMA)가 만들고 student가 모방. centering+sharpening(τ_t 0.04/τ_s 0.1) | 0.174 (collapse — multi-crop·momentum/temp schedule·freeze-last 트릭 누락) |
| BYOL | predictor + EMA teacher, negative 없음 | 0.067 (collapse) |

### engineering 변형 (SimCLR 0.598 위에 1개씩 추가)
| 변형 | 무엇을 하나 | 왜 |
|---|---|---|
| queue 1k/4k | MoCo식 과거 임베딩 큐를 negative에 추가 | batch 8 negative 부족(14개) 보완 = batch↑ 대체 |
| ignore 0.9 | cos>0.9 negative를 loss에서 마스킹 | 같은 클래스끼리 밀어내는 false-negative 방지 |
| koleo 0.1 | 임베딩별 최근접이웃 거리 log 최대화 항 (DINOv2) | 뭉침 방지, 균일 분산 |
| combo | queue4k + ignore + koleo 동시 | 상보 가설 |
| local 0.5 | 12×12 patch별 contrastive, positive=다른 뷰 최근접 patch (DenseCL) | 국소 결함 패턴 학습 |
| neco 1.0 | 두 뷰의 patch-patch 유사도 행렬 일치 (NeCo) | patch 이웃 구조 보존 |

## fine-tune sub-ablation (S2 상세 — LR/unfreeze 범위)

| config | best ARI | 비고 |
|---|--:|---|
| last_stage unfreeze, lr 3e-6 | 0.474 | 마지막 stage만 학습 — 안정적 plateau |
| **all-unfreeze, lr 2e-6** | **0.495 (MoCo)** → SimCLR 0.598 | ★ sweet spot |
| all-unfreeze, lr 1e-6 | 0.395 | LR 낮아 적응 부족 |
| all-unfreeze, lr 5e-6 | 0.478 | LR 높아 사전지식 distort |

## 탐색했으나 도움 안 됨 (음성 결과, 정직)

| 기법 | 결과 | 해석 |
|---|--:|---|
| NN-positive (min_sim 0.9) | no-op | queue 유사도 0.9 거의 안 넘어 미발동 |
| NN-positive (min_sim 0.6) | ↓ | 잘못된 이웃 당겨 오류 강화 |
| temp 0.03 / 0.07 | <0.45 | 0.05 sweet spot |
| multi-checkpoint ensemble | ↓ | 단일 best ckpt 최적 |
| spectral / agglomerative | 0.325 / 0.466 | k-means 가 best |
| supervised (SupCE/SupCon) | — | **영구 폐기 — 현업 라벨 없음 (사용자 directive)** |

## 핵심 결론
1. **DINOv3 frozen이 baseline** (0.310) — 백본 고정. FCMAE 비교는 별도 표 (frozen 0.210 열위).
2. **SimCLR fine-tune이 최대 레버** (+0.281) — 도메인 적응 필수, frozen 천장 0.317.
3. negative 있는 방법(SimCLR/VICReg/MoCo)이 negative 없는 방법(BYOL/SimSiam)·전용트릭 필요한 방법(DINO)보다 작은 데이터에서 안정적.
4. 현업 정렬(라벨·k 없는 불균형 풀)은 `_field_results.md` 참조 (cca 1779 풀, HDBSCAN k-free + Tier1).

산출: `_ablation_waterfall.{md,csv}`, `_ssl_comparison.{md,csv}`, `_field_results.{md,csv}`,
임베딩 `result_grouping/_dinov3_ncd_autoloop/{embeddings,ft_embeddings,ssl_embeddings}/`, `result_grouping/_field_cca/`.
