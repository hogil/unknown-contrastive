# Wafer Defect Contrastive Learning — 종합 보고서

> **anchor data 기준**: `D:/project/data/contrastive_anchor/avg30_260505_203615/` (defect 42 class avg 30 + Normal_bank_boundary 1000 = **2,146 wafer**)
> 작성: 2026-05-07

---

## ⓪ Dataset 미리보기 (실제 wafer 이미지)

각 wafer 는 **6400×6400 픽셀** PNG, 32×32 grid 의 chip 으로 구성. 결함 패턴별 class 명시 (anchor data, 42 defect class + Normal):

| Class | 패턴 설명 | 실제 sample |
|---|---|---|
| **Normal_bank_boundary** | 결함 없는 baseline (전체 grid 균일) | ![Normal](../../../data/contrastive_anchor/avg30_260505_203615/Normal_bank_boundary/AAJ350_00C_03_20260501_010000_92.6_6_PT_ENGINEER.png) |
| **CrescentArc** | 상단 호(arc) 형태 결함 | ![CrescentArc](../../../data/contrastive_anchor/avg30_260505_203615/CrescentArc/AOQ302_00P_03_20260501_010000_99.0_1_PE_NORMAL.png) |
| **BrokenRing** | 끊어진 ring 형태 | ![BrokenRing](../../../data/contrastive_anchor/avg30_260505_203615/BrokenRing/CDQ107_00P_13_20260501_010000_84.3_16_EE_NORMAL.png) |
| **Starburst** | 중심에서 방사형 라인 | ![Starburst](../../../data/contrastive_anchor/avg30_260505_203615/Starburst/AND515_00P_03_20260501_010000_86.3_14_PE_NORMAL.png) |
| **RingDots** | 원형 dot 클러스터 | ![RingDots](../../../data/contrastive_anchor/avg30_260505_203615/RingDots/BRW697_00C_20_20260501_010000_90.6_9_EE_ENGINEER.png) |
| **Edge-Bottom_scratch_rot** | 하단 + -21° 기울 scratch | ![EBSR](../../../data/contrastive_anchor/avg30_260505_203615/Edge-Bottom_scratch_rot/ACY549_00P_04_20260501_010000_99.0_1_PT_NORMAL.png) |
| **Edge-Top_fork** | 상단 + fork 모양 (가로 + 4-6 vertical legs) | ![ETF](../../../data/contrastive_anchor/avg30_260505_203615/Edge-Top_fork/ANW377_00P_02_20260501_010000_99.0_1_EE_NORMAL.png) |
| **Center_scratch** | 중심 + vertical scratch | ![CS](../../../data/contrastive_anchor/avg30_260505_203615/Center_scratch/AIG882_00C_16_20260501_010000_97.6_2_EE_PWQ.png) |
| **Donut_bank_boundary** | 도넛 + grid line | ![DB](../../../data/contrastive_anchor/avg30_260505_203615/Donut_bank_boundary/CEV315_00C_16_20260501_010000_95.9_4_PT_NORMAL.png) |
| **Full_invalid_main** | 전 영역 + invalid pattern | ![FI](../../../data/contrastive_anchor/avg30_260505_203615/Full_invalid_main/EBQ630_00C_03_20260501_010000_67.0_0_PE_ENGINEER.png) |

**Class 명명 규칙**: `<spatial>_<obj>` — spatial (어디?) + obj (어떤 모양?).
- spatial: Center / Donut / Edge-Ring / Edge-Top / Edge-Bottom / Full / Thick-Edge
- obj: bank_boundary (grid 라인) / fork (포크 모양) / scratch (직선) / scratch_rot (-21° 기울) / invalid_main (불량 영역 표시)
- standalone class: BrokenRing, CrescentArc, Starburst, RingDots, CrossScratch, DiagonalSmear, ParallelScratches, CenterCircle, CenterDonut, Row

→ **42 defect class** (combination 별 + standalone) + **1 Normal class** = 43 GT label

---

## 0. Contrastive Learning 이란? (Tutorial)

### 0.1 핵심 개념 — "닮은건 가까이, 다른건 멀리"

지도학습 (Supervised Learning) 은 **정답 label 이 필요**. 하지만 fab 환경에서:
- 새로운 결함 (unknown defect) 이 매일 등장
- 라벨링 비용 큼 + 라벨 없는 데이터가 99%

**Contrastive Learning (대조 학습)** = **label 없이** 이미지간 유사도 학습:
- 같은 wafer 의 두 변형 (augmentation) → **positive pair** (가까이)
- 다른 wafer 들 → **negative pair** (멀리)
- 모델이 "wafer X 와 wafer Y 가 비슷한가/다른가" 만 학습 — class 이름 모름

### 0.2 학습 흐름 (한 step)

```
   [Wafer A]                       [Wafer B]
       │                              │
       ├── view1 (random aug)         ├── view1
       └── view2 (random aug)         └── view2
       │       │                      │       │
       ↓       ↓                      ↓       ↓
   [Encoder ConvNeXtV2 + Projection 128-dim]
       ↓       ↓                      ↓       ↓
       z_A1    z_A2                   z_B1    z_B2
       └────┬────┘                    └────┬────┘
            │                              │
        positive pair                  positive pair
            │                              │
            └──────────── negative ────────┘
                 (멀리 밀어내야)

Loss:
   L = -log[ exp(sim(z_A1, z_A2) / τ) / Σ exp(sim(z_A1, z_*) / τ) ]
       └────────────┬─────────────┘   └─────────────┬──────────────┘
            positive 가 분자          모든 negative 가 분모
```
- `sim(a, b) = a·b / (|a||b|)` — cosine similarity
- `τ (TEMP)` — softmax sharpness, 작을수록 sharp
- 학습 끝나면 **유사 wafer 끼리 embedding space 에서 가까이 모임**

### 0.3 학습된 후 어떻게 군집화 (HDBSCAN)?

```
2,146 wafer × 384×384 PNG
      ↓ (학습된 encoder 통과)
2,146 × 128-dim embedding
      ↓ (HDBSCAN: density 기반 군집화)
43 cluster + 일부 noise (격리된 wafer)
      ↓
각 cluster 마다 medoid (중심에 가장 가까운 wafer) 시각화
```

**HDBSCAN** (Hierarchical Density-Based Spatial Clustering):
- K-means 와 달리 **cluster 개수 미리 지정 X**
- density 기반 — 밀집 영역만 cluster, 외딴 점은 noise (label = -1)
- 핵심 hparam: `min_cluster_size=12` (12 wafer 이상이어야 cluster), `cluster_selection_method='leaf'` (하위 leaf cluster 까지)

### 0.4 시각화 — Cluster 결과 예시 (Iter 14)

| Cluster | Class | 시각화 |
|---|---|---|
| 010 | **CrescentArc** (43 wafer) | 상단 호(arc) 패턴 — wafer 위쪽에 곡선 결함 |
| 018 | **Edge-Top_scratch** (58 wafer) | 상단에 vertical scratch 라인 |
| 014 | **RingDots** (27 wafer) | 원형 dot 패턴 |
| 003 | **CrossScratch** (31 wafer) | 가로+세로 scratch 교차 |

→ **같은 결함 패턴 wafer 들이 자동으로 같은 cluster** 에 들어감 — label 한 번도 안 줬는데도.

### 0.5 왜 ConvNeXtV2 + TAPT?

- **ConvNeXtV2-base** (Woo et al. CVPR 2023): ImageNet 에서 강력한 backbone
- **TAPT (Task-Adaptive Pre-Training)**: ImageNet 의 일반 이미지 → 같은 wafer 데이터로 supervised 33-class 학습 (`known-cnn` repo) → 그 backbone state_dict 추출 → contrastive 시작점으로 사용
  - 이유: ImageNet 직접 init 보다 같은 도메인 fit 이 contrastive convergence 빠름
- **Backbone freeze + projection head 만 학습** (FREEZE_BACKBONE=True): 작은 dataset (2146) 에서 over-fit 방지

---

## 1. 문제 정의 (Problem Setup)

### 1.1 목표
- **Self-Supervised Contrastive Learning** + **HDBSCAN** 으로 wafer-level defect map 군집화
- 지도 학습 없이 (label 무관) wafer 합성 결함 패턴별 자동 그룹핑
- Production 시나리오: unknown defect 출현 시 누락 없이 신규 cluster 형성 (**P1: capture rate**)

### 1.2 데이터
- **합성 wafer 6400×6400** PNG (sister repo `known-cnn` 의 `dist_apply/_sample_gen.py` 가 생성)
- 42 defect class (Center/Donut/Edge-Ring/Edge-Top/Edge-Bottom/Full/Thick-Edge × bank_boundary/fork/scratch/scratch_rot/invalid_main + standalone classes BrokenRing/CrescentArc/...)
- Normal class (1000 sample, defect 없는 baseline distribution)
- anchor subset: defect class 별 평균 30 (random distribution 15-45) + Normal 전체 — **method ablation 의 same-data 기준점**

### 1.3 평가 metric 정의 + 의미 (이미지 예시 포함)

cluster 결과 평가 — **비지도 군집화** (label 없이 학습) but **GT label** (anchor 의 class 폴더명) 으로 검증.

---

#### **P1 = class_capture_rate** (recall 느낌, 1순위)

**정의**: `(잡힌 defect class 수) / (전체 defect class 수)`

**계산 예시**:
- 42 defect class 중 어떤 class 의 wafer 들이 **모두 noise (-1)** 가 되면 그 class "놓침" — P1 깎임
- Iter 1: 42/42 = 1.000 → 모든 class 잡음
- Iter 2 (REJECT): 41/42 = 0.976 → Donut_scratch_rot (n=15) 전부 noise → P1 violation

**시각 예시**:
- ✅ Donut_scratch_rot 15 wafer 가 Cluster #X 에 모임 → capture
  ![DSR cluster](../../outputs_contrastive_260506_235250/cluster_summary/cluster_001_size_32__Donut_invalid_main__medoid_dist0.0415.png)
- ❌ Donut_scratch_rot 15 wafer 모두 noise 처리 → P1 violation (cluster medoid PNG 안 만들어짐)

**왜 1순위?**: production 에서 결함 누락 = 큰 사고 (검사 라인 통과 후 chip 출하 → 고객 불만)

---

#### **P2 = noise_pct (defect only)** (false alarm 느낌)

**정의**: `(defect 중 HDBSCAN noise=-1 로 분류된 비율) / (defect 전체)`

**계산 예시** (Iter 14, anchor 1146 defect wafer):
- HDBSCAN noise = -1 인 defect wafer 수 / 1146
- Iter 14: 76 / 1146 = 6.63%
- Iter 1: 53 / 1146 = 4.62% (best)

**시각 예시 (Iter 14 cluster #18 Edge-Top_scratch, sz 58)**:
![ETS](../../outputs_contrastive_260506_235250/cluster_summary/cluster_018_size_58__Edge-Top_scratch__medoid_dist0.1210.png)
- 58 wafer 가 같은 cluster 에 모임 → cluster 형성 성공
- 만약 58 중 5 wafer 가 cluster 에 못 들어가서 noise (-1) 처리됐다면 → P2 +0.4%pp (5/1146)

낮을수록 좋음 — defect 격리 잘 됨.

---

#### **P3 = Completeness** (Rosenberg & Hirschberg 2007)

**정의**: 같은 class 의 wafer 들이 같은 cluster 에 모이는 정도

**수식**: `Completeness = 1 - H(C|K) / H(C)`
- C = GT class label (e.g., "Edge-Top_scratch"), K = HDBSCAN cluster id
- `H(C|K)` = cluster 알 때 class 의 entropy — cluster 가 class 를 잘 split 하면 0 → Comp = 1
- `H(C)` = class 분포 entropy

**계산 worked example** (4 wafer 단순화):

GT:        `[A, A, A, B]`
Cluster:   `[1, 1, 1, 2]` → A 모두 cluster 1, B 단독 cluster 2 → **Comp = 1.0** (perfect)

GT:        `[A, A, A, B]`
Cluster:   `[1, 2, 1, 1]` → A 가 cluster 1+2 로 split → Comp 낮음 (≈ 0.4)

**시각 예시 — Iter 14 의 high Completeness (0.952)**:
- Edge-Top_scratch (n=18) wafer 들이 거의 **모두 cluster #18** 에 모임 (사진처럼 같은 패턴 wafer 들이 한 그룹) → Comp 높음
- Edge-Top_scratch 가 cluster #18 에 9 + cluster #20 에 9 분산되면 → Comp 낮음

**1.0 = 모든 class 가 단일 cluster (split 절대 X)**

---

#### **P4 = Homogeneity** (Rosenberg & Hirschberg 2007)

**정의**: 한 cluster 안에 한 class 만 있는 정도 (Completeness 의 dual)

**수식**: `Homogeneity = 1 - H(K|C) / H(K)`

**계산 worked example**:

GT:        `[A, A, B, B]`
Cluster:   `[1, 1, 2, 2]` → cluster 1 = A 만, cluster 2 = B 만 → **Hom = 1.0**

GT:        `[A, A, B, B]`
Cluster:   `[1, 1, 1, 2]` → cluster 1 = A+A+B (mixed) → Hom 낮음

**시각 예시 — Iter 14 cluster #14 RingDots (sz 27)**:
![RD](../../outputs_contrastive_260506_235250/cluster_summary/cluster_014_size_27__RingDots__medoid_dist0.1441.png)
- 27 wafer 모두 같은 RingDots 패턴 → cluster purity 1.0 → Hom 기여 ↑
- 만약 27 중 5 가 다른 class (CrescentArc) 면 → mixed cluster → Hom 깎임

**Iter 14 Hom = 0.908** = 대부분 cluster 가 단일 class dominant

---

> ### Completeness vs Homogeneity 비교 (직관)
> - 모든 wafer 가 1 cluster → Comp 1.0 / Hom 0.0 (한 class 가 흩어지진 않지만 다른 class 와 섞임)
> - 모든 wafer 가 각각 cluster → Comp 0.0 / Hom 1.0 (한 cluster 에 한 wafer 만, 하지만 같은 class 가 흩어짐)
> - **둘 다 높아야** 좋은 cluster

---

#### **AMI** (Adjusted Mutual Information, Vinh et al. 2010)

**정의**: cluster ↔ class 의 mutual info, random baseline 보정

**수식**: `AMI = (MI - E[MI]) / (max(H(C), H(K)) - E[MI])`
- MI = mutual information between cluster K and class C
- E[MI] = random baseline expected MI
- 0 = random clustering, 1 = perfect 매칭

**의미**: Completeness + Homogeneity 의 **종합 지표**.
- random 으로 cluster 생성 시 0 근처 (보정 효과)
- AMI = 0.913 (Iter 14) = 높은 cluster-class 일치도

---

#### **ARI** (Adjusted Rand Index, Hubert & Arabie 1985)

**정의**: 같은 pair (i,j) 가 cluster + class 에서 모두 같이 묶이거나 모두 떨어진 비율 (random 보정)

**계산 worked example** (4 wafer):

GT classes:  `[A, A, B, B]` → 같은 class pair: (0,1), (2,3)
Cluster:     `[1, 1, 2, 2]` → 같은 cluster pair: (0,1), (2,3)
→ 모두 일치 → ARI = 1.0

GT:          `[A, A, B, B]`
Cluster:     `[1, 2, 1, 2]` → cluster pair: (0,2), (1,3) — class pair 와 일치 없음
→ ARI 낮음

**의미**: pair-level 평가 (Completeness/Homogeneity 가 entropy-based 인 반면).

---

#### **Silhouette (cosine)** (Rousseeuw 1987)

**정의**: 한 점이 자기 cluster 와 가장 가까운 다른 cluster 사이 거리 비율

**수식**: `s(i) = (b(i) - a(i)) / max(a(i), b(i))`
- `a(i)` = i 가 자기 cluster 안 다른 점들과 평균 거리 (응집)
- `b(i)` = i 가 가장 가까운 다른 cluster 점들과 평균 거리 (분리)
- s(i) ≈ 1 → 자기 cluster 안 가깝고 다른 cluster 와 멀음 → 좋음
- s(i) ≈ 0 → 경계 위
- s(i) < 0 → 잘못 클러스터됨

**Iter 14 Sil = 0.725** = 평균적으로 cluster 응집/분리 좋음 (cosine distance 기반)

**Intrinsic** = GT label 없이도 측정 가능. 새 wafer 도입 시 정성 평가용.

---

#### **Sister-class centroid distance** (도메인 특화 추가 metric)

**문제 의식**: contrastive 학습이 "Edge-Bottom 위치 인지" 는 잘 학습 하지만 "Edge-Bottom 안에서 fork vs scratch_rot 구분" 은 약함.

**정의**: 같은 spatial region (e.g., Edge-Bottom) 의 다른 obj cluster 의 centroid cosine distance:
```
sister_pair = (Edge-Bottom_fork, Edge-Bottom_scratch_rot)
centroid(EBF) = mean(embeddings of EBF cluster members)
centroid(EBSR) = mean(embeddings of EBSR cluster members)
sister_cos_dist = 1 - cos(centroid(EBF), centroid(EBSR))
```

**해석**:
- 0.007 (Iter A0): 두 centroid 가 거의 같은 vector — **collapse** (모델이 둘을 같은 것으로 인식)
- 0.260 (Iter 14): 분리 잘 됨

**4 sister pair 추적**:
- EB_fork ↔ EB_scratch_rot
- D_scratch ↔ D_scratch_rot
- ET_fork ↔ ET_scratch_rot
- Cen_fork ↔ Cen_scratch_rot

**시각 예시** — Iter A0 의 Edge-Bottom_fork 와 Edge-Bottom_scratch_rot 가 같은 mega-cluster (sz 104, purity 0.39) 에 흡수 → cluster medoid 가 mixed pattern → sister cos-dist 0.007

이 metric 은 **wafer-defect 도메인 특화** — 표준 sklearn metric 으로는 안 보이는 collapse 진단.

---

### 1.4 평가 우선순위 lock-in

**P1 → P2 → P3 → P4** (`docs/contrastive-eval/DECISIONS.md` D-3):
- 사용자 명시 "불량 1개라도 group 으로 나오는거 제일 중요 - recall 느낌, noise group 나오지 않는게 두번째"
- production 운영 직결 — 누락 (P1) > false alarm (P2) > clustering quality (P3-P4)

**보조 (Tier 2)**: AMI / Silhouette (cosine) / ARI

**금지** (사용자 명시 "전부 cnn 성능지표잖아"): weighted_isolation, pure_rate, binary_*, precision/recall/F1/FPR — 일체 출력 X

---

## 2. 아키텍처 (Architecture)

### 2.1 Encoder
- **ConvNeXtV2-base** (`convnextv2_base.fcmae_ft_in22k_in1k_384`)
- **TAPT (Task-Adaptive Pre-Training)**: ImageNet FCMAE pretrain → 같은 wafer 데이터로 sister repo `known-cnn` 의 33-class supervised CNN 학습 → state_dict 추출 → contrastive backbone 으로 inject
- **Backbone frozen** (FREEZE_BACKBONE=True) — projection head 만 학습 → 합성 데이터 supervised collapse 회피
- 입력 IMAGE_SIZE = **384** (사용자 GPU 작게)

### 2.2 Projection head
- **PROJ_DIM = 128** — embedding 차원
- L2-normalized → cosine similarity 기반 contrastive loss

### 2.3 Loss (InfoNCE + Local + Queue)

```
L = L_global (InfoNCE 2 view) + L_queue + LOCAL_WEIGHT × L_local
```

- **Global InfoNCE** (Wang & Isola 2020): batch 내 다른 wafer 와 contrast
- **Queue (momentum bank, MoCo style)**: QUEUE_SIZE=4096, 과거 view embedding 누적 → 더 큰 negative pool
- **Local InfoNCE (USE_LOCAL=True)**: wafer 를 **6×6 grid** 로 나눠 grid cell 단위 contrast — wafer 내 spatial pattern (Edge-Top vs Edge-Bottom 위치 정보) 학습 (DenseCL-style)

### 2.4 IGNORE_NEG_SIM (false-negative 완화)
- 같은 view 인 wafer 끼리 contrastive 시 너무 비슷한 (e.g., 같은 distribution) wafer 가 negative 로 잘못 처리 — semantic 손상
- threshold 이상 cosine sim 인 negative 는 mask (drop from contrast)
- 기본 0.72

---

## 3. 채택/거부한 기법 (Decisions)

### 3.1 채택 ✅
| 기법 | 이유 |
|---|---|
| **USE_LOCAL grid contrast** | wafer 내 위치 정보 (Edge-Top vs Edge-Bottom) 가 class identity — local contrast 가 spatial pattern 학습 |
| **USE_QUEUE (momentum bank)** | batch 작아도 (BATCH=8) 4096 negative pool 확보 |
| **TAPT backbone** | ImageNet 직접 init 보다 같은 도메인 supervised fit 이 contrastive 시작점으로 우수 |
| **IGNORE_NEG_SIM** | 같은 distribution 의 다른 wafer 가 false-negative 되는 문제 mitigate |

### 3.2 거부 ❌
| 기법 | 이유 (사용자 명시) |
|---|---|
| **Multi-crop (SwAV)** | "wafer 내 불량 분포가 같은 것들 잡아야 하는데 random crop 이라니 안 되지" — 위치 정보 손상 |
| **SupCon (Supervised Contrastive)** | "여기 등록되어있지 않은 이미지들 나오면 성능 저하" — unknown defect generalization 위험 |
| **Rotation/Flip augmentation** | scratch_rot 등 angle 이 class identity (`feedback_no_rotation_aug_chip.md`) |
| **NV-Retriever PercPos α (D-14)** | 4 step sweep all dead — Iter 1 baseline 못 이김 |
| **EPOCHS↑ (10/8)** | over-fit, 모든 P metric 후퇴 |

---

## 4. Method Ablation Chain (19 iter)

### 4.1 전체 chain

| Iter | atomic change | Comp P3 | AMI | **noise(def) P2** | **capture P1** | Hom P4 | ARI | Sil | 판정 |
|---|---|---|---|---|---|---|---|---|---|
| **A0** | baseline (LW=0.5, LR=1e-3, NEG=0.72, TEMP=0.07, EPOCHS=5) | 0.938 | 0.895 | 9.34% | 1.000 | 0.893 | 0.704 | 0.791 | base |
| **Iter 1** ★ | **LW 0.5 → 1.0** | 0.948 | 0.904 | **4.62%** | 1.000 | 0.898 | 0.733 | 0.777 | **★ P2 winner** |
| Iter 2 | NEG 0.72 → 0.65 | 0.816 | 0.821 | 7.16% | **0.976** ❌ | 0.875 | 0.639 | 0.737 | reject (P1) |
| Iter 3 | LW 1.0 → 0.7 | 0.936 | 0.893 | 9.42% | 1.000 | 0.893 | 0.701 | 0.785 | NULL ≈ A0 |
| Iter 4 | LW 1.0 → 1.5 | 0.945 | 0.899 | 5.93% | 1.000 | 0.893 | 0.721 | 0.761 | LW=1.0 sweet spot 확정 |
| Iter 6 | EPOCHS 5 → 10 | 0.798 | 0.806 | 9.34% | 1.000 | 0.867 | 0.557 | 0.758 | reject (over-fit) |
| Iter 7 | NV-Retriever PercPos α=0.95 | 0.829 | 0.812 | 10.9% | 0.976 ❌ | 0.844 | 0.643 | 0.777 | reject (P1) |
| Iter 8 | α 0.95 → 0.90 | 0.816 | 0.811 | 8.64% | 1.000 | 0.855 | 0.626 | 0.780 | partial |
| Iter 9 | α 0.90 → 0.85 | 0.813 | 0.826 | 5.15% | 1.000 | 0.885 | 0.601 | 0.771 | partial |
| Iter 10 | α 0.85 → 0.80 | 0.793 | 0.792 | 10.82% | 1.000 | 0.848 | 0.568 | 0.781 | reject (PercPos axis dead) |
| **Iter 11** | **LR_HEAD 1e-3 → 5e-4** | **0.948** | **0.905** | 6.11% | 1.000 | 0.899 | **0.734** | 0.784 | accept |
| Iter 12 | + EPOCHS 5 → 8 | 0.783 | 0.801 | 8.81% | 1.000 | 0.875 | 0.491 | 0.749 | reject (over-fit) |
| **Iter 13** | + NEG 0.72 → 0.65 | **0.949** | **0.906** | 5.32% | 1.000 | 0.901 | 0.743 | 0.752 | accept |
| **Iter 14** ★ | + **TEMP 0.07 → 0.05** | **0.952** | **0.913** | 6.63% | 1.000 | **0.908** | **0.763** | 0.725 | **★ Quality winner** |
| Iter 15 | TEMP 0.05 → 0.06 | 0.940 | 0.894 | 7.42% | 1.000 | 0.890 | 0.713 | 0.733 | reject |
| Iter 16 | LR 5e-4 → 3e-4 | 0.940 | 0.902 | 6.46% | 1.000 | 0.903 | 0.752 | 0.716 | reject (LR sweep 종결) |
| Iter 17 | WARMUP 1 → 2 | 0.944 | 0.890 | 7.94% | 0.976 ❌ | 0.879 | 0.698 | 0.745 | reject (P1) |
| Iter 18 | WARMUP 1 → 0 | 0.952 | 0.913 | 6.63% | 1.000 | 0.908 | 0.763 | 0.725 | == Iter 14 (no effect) |
| Iter 19 | LOCAL_POS_TOPK 12 → 8 | 0.945 | 0.899 | 6.54% | 1.000 | 0.892 | 0.732 | 0.732 | reject |

### 4.2 진짜 큰 lever 발견

| Axis | iter | 효과 |
|---|---|---|
| **LOCAL_WEIGHT** (0.5→1.0) | Iter 1 | noise 9.34→4.62% (-50%) ★ |
| **LR_HEAD** (1e-3→5e-4) | Iter 11 | LR 절반 = head 안정 ★ |
| **NCE_TEMP** (0.07→0.05) | Iter 14 | softmax sharper, AMI/ARI 큰 폭 ↑ |
| **IGNORE_NEG_SIM** (0.72→0.65) | Iter 13 (Iter 11 base) | LR 5e-4 와 결합 시 P1 violation 회피, sister 분리 ↑ |

### 4.3 Dead axes (4 step sweep all reject)
- **PercPos α** (NV-Retriever) — Iter 7-10 모두 baseline 미달
- **EPOCHS** (8/10 모두 over-fit) — 5 sweet spot
- **WARMUP** (0/2 의미 없거나 P1 violation)
- **LOCAL_POS_TOPK** (8 무효)

---

## 5. Best 설정 (Two-King Trade-off)

### 5.1 Iter 1 ★ — **P2 King (Production safety)**
```python
LW = 1.0
LR_HEAD = 1e-3
IGNORE_NEG_SIM = 0.72
NCE_TEMP = 0.07
EPOCHS = 5, BATCH = 8, IMAGE_SIZE = 384
USE_LOCAL = True, LOCAL_WEIGHT = 1.0, LOCAL_POS_TOPK = 12
USE_QUEUE = True, QUEUE_SIZE = 4096
```
- noise(def) 4.62% — 가장 strict defect 격리
- Production 운영 (false alarm 최소화) 우선

### 5.2 Iter 14 ★ — **Quality King (Cluster purity)**
```python
LW = 1.0
LR_HEAD = 5e-4         # ← Iter 11 변경
IGNORE_NEG_SIM = 0.65  # ← Iter 13 변경
NCE_TEMP = 0.05        # ← Iter 14 변경
EPOCHS = 5, BATCH = 8, IMAGE_SIZE = 384
USE_LOCAL = True, LOCAL_WEIGHT = 1.0, LOCAL_POS_TOPK = 12
USE_QUEUE = True, QUEUE_SIZE = 4096
```
- Completeness 0.952 / AMI 0.913 / ARI 0.763 — cluster quality 최고
- Sister-class 분리 best (mean cos-dist 0.260)
- 학술 / 분석 / 발표 우선

### 5.3 Trade-off 정량

| | **Iter 1** | **Iter 14** | Δ |
|---|---|---|---|
| capture P1 | 100% | 100% | tie |
| **noise(def) P2** | **4.62%** | 6.63% | -2.01pp |
| **Completeness P3** | 0.948 | **0.952** | +0.004 |
| **AMI** | 0.904 | **0.913** | +0.009 |
| Homogeneity P4 | 0.898 | **0.908** | +0.010 |
| **ARI** | 0.733 | **0.763** | +0.030 |
| Silhouette | **0.777** | 0.725 | -0.052 |
| Sister mean | 0.261 | **0.260** | ≈ |

**해석**: harder hard mining (NEG↓, TEMP↓) → cluster 응집 ↑ but defect-only noise 흡수 ↑. P2 우선 정책 → Iter 1, P3-P4-AMI-ARI 우선 → Iter 14.

---

## 6. Cluster 예시 (Iter 14 medoid composite)

각 cluster 의 medoid (centroid 와 가장 가까운 wafer) + 인접 멤버 시각화:

| Cluster | Class | Sample |
|---|---|---|
| 000 | Starburst (37 wafer) | ![](../../outputs_contrastive_260506_235250/cluster_summary/cluster_000_size_37__Starburst__medoid_dist0.1247.png) |
| 003 | CrossScratch (31) | ![](../../outputs_contrastive_260506_235250/cluster_summary/cluster_003_size_31__CrossScratch__medoid_dist0.0813.png) |
| 010 | CrescentArc (43) | ![](../../outputs_contrastive_260506_235250/cluster_summary/cluster_010_size_43__CrescentArc__medoid_dist0.0726.png) |
| 014 | RingDots (27) | ![](../../outputs_contrastive_260506_235250/cluster_summary/cluster_014_size_27__RingDots__medoid_dist0.1441.png) |
| 018 | Edge-Top_scratch (58) | ![](../../outputs_contrastive_260506_235250/cluster_summary/cluster_018_size_58__Edge-Top_scratch__medoid_dist0.1210.png) |

**전체 43 cluster** medoid 이미지: `outputs_contrastive_260506_235250/cluster_summary/`.

---

## 7. 학습 / 분석 파이프라인

```
[wafer 합성 (sister repo)]
       ↓
[anchor subset hardlink]  ← _build_anchor_subset.py (avg30 random + Normal 전체)
       ↓
[contrastive 학습]  ← _dispatch_iter.py + run_contrastive.py + contrastive.py
   - DETACHED_PROCESS Popen (Bash 60min timeout 회피)
   - GPU throttle 5000ms (NVIDIA TDR crash 방지)
   - CFG override only (contrastive.py 무수정 정책)
       ↓
[embedding extraction] (2146 × 128)
       ↓
[HDBSCAN clustering] (mcs=12, ms=4, leaf, eps=0.06)
       ↓
[evaluation skill]  ← Tier 1+2 official metric (sklearn) + per-cluster purity + sister centroid distance
       ↓
[cluster-analyzer agent] / [image-analyzer agent] / [performance-research agent]
       ↓
[paper-recorder]  ← docs/paper/ 8 section 누적 update
       ↓
[Iter N+1 atomic dispatch]  ← method ablation strict same-data + 1 axis change
```

### 7.1 운영 안정성

| 이슈 | 해결책 |
|---|---|
| Bash run_in_background 60min timeout → python 동반 사망 | `subprocess.Popen(creationflags=DETACHED_PROCESS|CREATE_NEW_PROCESS_GROUP)` 진짜 detach |
| NVIDIA driver TDR crash (3 process 동시 시) | `torch.optim.Optimizer.step` 후 `time.sleep(5000ms)` injection |
| 한 process 만 GPU 30% target | `pynvml.nvmlDeviceGetProcessUtilization` 으로 per-process 측정 |

### 7.2 핵심 파일

```
contrastive.py             # main training engine (사용자 명시 무수정 정책)
run_contrastive.py         # CFG override wrapper (env-driven)
_dispatch_iter.py          # detached dispatch (1 iter = 1 atomic change)
_build_anchor_subset.py    # avg30 random anchor builder (file_list.parquet 자동 저장)
_eval_contrastive_n50.py   # Tier 2 (ARI/NMI/silhouette + per_class_noise)
_iter*_addons.py           # Tier 1 + per-cluster purity + sister centroid (eval skill 추가 산출)
```

---

## 8. Lessons Learned

### 8.1 효율적 axis 발견
- **2 lever (LW + LR_HEAD)** 가 90% 효과
- 다른 axis (PercPos/EPOCHS/WARMUP/LOCAL_POS_TOPK 등) 모두 marginal 또는 negative

### 8.2 Sister-class collapse 본질적 한계
- Edge-Bottom 의 fork ↔ scratch_rot centroid 거리 = 0.007 (Iter A0) — 사실상 동일 representation
- **encoder 가 spatial position dominant 학습** → defect type 구분 약함
- Iter 14 까지 mean sister cos-dist 0.260 — 개선했지만 근본 해결 X
- 처방: encoder method change 가 진짜 해결책 (ConvNeXtV2 backbone 자체 변경 또는 supervised pretrain 데이터 추가)

### 8.3 Trade-off 직면
- **harder mining (NEG↓ TEMP↓)** → cluster 응집 ↑ but **noise 흡수 ↑** (defect 일부가 noise 로)
- production safety (P2) ↔ academic quality (P3-P4-AMI-ARI) 양극화

### 8.4 검증 method ablation 정책
- **strict same-data** (anchor avg30_260505_203615 고정)
- **1 atomic axis** per iter (BATCH 변경은 same-condition 으로 간주)
- **append-only ITERATIONS log** (과거 결과 수정 X)
- HDBSCAN cfg sweep 은 별개 track (encoder 학습 ablation 과 분리)

---

## 9. References

- **InfoNCE / Alignment-Uniformity**: Wang & Isola, "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere", ICML 2020
- **DenseCL (local feature)**: Wang et al. 2021 — 본 repo 의 USE_LOCAL grid contrast 의 motivation
- **ConvNeXtV2 backbone**: Woo et al. CVPR 2023, FCMAE pretrain
- **HDBSCAN**: Campello et al. 2013, sklearn-contrib 구현
- **Tier 1+2 metric** (sklearn / paper):
  - Completeness/Homogeneity: Rosenberg & Hirschberg 2007
  - AMI: Vinh et al. 2010
  - ARI: Hubert & Arabie 1985
  - Silhouette (cosine): Rousseeuw 1987 + cosine variant
- **NV-Retriever PercPos** (Iter 7-10 시도, axis dead): Moreira et al. 2024 (arXiv:2407.15831)
- **TAPT backbone**: 자매 repo `D:/project/known-cnn/` 의 33-class supervised CNN best_model.pth

---

## 10. 다음 단계

### 10.1 즉시 가능 (env-only, 추가 dispatch 가치 낮음)
- LR sweep 종결 (5e-4 sweet spot 확정)
- TEMP sweep 종결 (0.05 best)
- WARMUP/LOCAL_POS_TOPK 효과 없음 확인

### 10.2 큰 변화 (code change 또는 정책 review 필요)
- **Backbone unfreeze + low LR**: head only → backbone joint training (FREEZE_BACKBONE=False, LR_BACKBONE=1e-5)
- **Patch ordering loss** (NeCo, arXiv:2408.11054) — sister-class 정공법 (constrastive.py 코드 변경 필요)
- **CrossDataset Augmentation**: 다른 wafer dataset (real fab data) 합쳐 generalization 향상

### 10.3 Production 적용
- Iter 1 (P2 king) + Iter 14 (Quality king) **2-track ensemble** 가능
- daily wafer 폴더 → predict_contrastive_daily.py (이미 구현) 으로 cluster + medoid + review 산출
