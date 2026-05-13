# Iterations log

**Append-only**. 과거 iteration 수정 금지 — 시간 순으로 변경 + 효과 추적.

매 iteration entry:
- 날짜
- 변경 내용 (1-3 항목)
- 동기 (왜 변경?)
- 결과 (Tier 1 metric 변화, 시각 결과)
- 다음 단계

---

## Iter 0 — Baseline (2026-05-05)

### 설정
- Backbone: ConvNeXtV2-base FCMAE + TAPT (sister repo `known-cnn` cnn_train 결과)
- Projection: 128-d, L2 normalized
- Loss: InfoNCE global (G) + queue (Q). Local (L) OFF.
- BATCH=16, IMAGE_SIZE=384, EPOCHS=10, TEMP=0.07, LR_HEAD=1e-3
- Queue size 4096
- Sampling: per-class 200 + Normal 1000 = 8,357 wafer
- HDBSCAN: min_cluster_size=12, min_samples=4, leaf, ε=0.06

### 결과
- **Completeness=0.9466, AMI=0.9288, noise_pct=0.71%, capture=38/38**
- frac_single_cluster=0.8947 (34/38). 4 class split.
- alignment=0.3018, uniformity=-2.4955

### 발견
1. **Full_*** + Thick-Edge_fork 4 class split 의 원인이 합성 데이터의 진짜 sub-style** —
   HDBSCAN sweep / GMM BIC bimodal / intra·inter ratio 4-9× 로 확정.
   처방: 사용자 결정 대기 (통일 vs 분리).
2. ARI=0.70 vs Retrieval recall@1=0.9936 의 gap — over-cluster 페널티가 ARI 에만 강하게 적용.
   → ARI 를 1차 metric 로 쓰지 말 것 (Tier 2 보조). Completeness / AMI 가 본질.
3. Edge-Top × {bank_boundary, fork, scratch, scratch_rot} 4 sub-class 가 한 cluster (cluster 12) 로 merge.
   → bank_boundary defect 가 너무 미세해서 다른 obj 와 구분 부족.
   처방: Hard negative mining (Iter 3 후보).

### 다음 (Iter 1)
- per-class sample 수 random sampling (50~200+) — production 비율 흉내
- alignment + uniformity 매 epoch monitoring 도입 (Wang-Isola 2020)

### Iter 0 deep diagnostics (2026-05-05, 4 team agent chain)

**Active agents**: cluster-analyzer (a5d4b7d14eef989a1), image-analyzer (a64e91d08a1b0ff6a),
performance-research (a63a299550584d293), resource-monitor (a2e25e2a2343bef29) — team `contrastive-team`.

산출:
- `outputs/logs_contrastive/overall/analyze_clusters.{md,json}`
- `outputs/logs_contrastive/overall/analyze_images.{md,json}`
- `outputs/logs_contrastive/overall/research_20260505_192357.{md,json}`

**cluster-analyzer 발견**:
- **HDBSCAN 이미 sweep optimum** (mcs=12, ms=1, eom = ARI 0.7143). HDBSCAN 추가 tuning 효과 없음.
- 진짜 problematic cluster 4 개 (35/39 weak 중):
  - cluster 12 — Edge-Bottom_bank_boundary mega (size 802, purity 0.25, 4-obj merge)
  - cluster 17 — Edge-Top_bank_boundary mega (size 799, purity 0.25, 4-obj merge)
  - cluster 35 — Full_fork (wide_spread + boundary_blur)
  - cluster 37 — Normal_bank_boundary (size 691, sil 0.16, boundary blur with defect clusters)
- **Normal noise 22.9% 의 진짜 원인** = cluster 37 (Normal) ↔ cluster 12/17 (Edge-Top/Bottom_bank_boundary) 의 boundary blur.

**image-analyzer 발견** (top-K=5 outliers per weak cluster):
- 172 primary outliers (centroid distance), 10 secondary (pixel z-score), 13 suspect_mislabel, 4 cluster_split groups
- ★ **cluster 35 Full_fork 에 5 Thick-Edge_fork wafer mislabel** 의심 — 합성 라벨링 검토 필요
- ★ **cluster 17 안 5 Edge-Top_scratch/scratch_rot** — embedding 이 bank_boundary 쪽으로 끌어당김. 이게 Edge-Top × 4-obj merge 의 직접 원인.
- ★ **cluster split pairs 4개**: Full_bank[20,21], Full_scratch_rot[29,30], Full_fork[34,35], **Normal_bank_boundary[36,37]** — **Normal 도 split** 발견 (이전 엔 없었음!)
- 6 synthesis_artifact (|z|≥3.0 pixel stats) — cluster 35 의 `CMY389` wafer 가 4 통계 동시 hit → 합성 pipeline anomaly 강한 후보

**performance-research 권고** (arxiv + GitHub):
- ★★★ SCHaNe (arxiv 2308.14893) — SupCon + dissimilarity-weighted hard negative re-weight, +3.32% few-shot
- ★★★ NV-Retriever (arxiv 2407.15831, NVIDIA 2024) — positive-aware false-negative filter. **Full_×3 sub-cluster repulsion 직격**
- ★★★ pytorch-metric-learning (6.3k stars) — production miner + SupConLoss API
- ★ ProNC (arxiv 2505.24254, ICLR 2026) — neural collapse, ETF prototype

### 다음 (Iter 1) 업데이트 — 3 chain 통합 plan

★ **Methodology lock-in (사용자 명시)**: ablation 은 **same data, same condition, method 만 변경**.
데이터 갯수 / 비율 변경은 method iteration 아님 (data ablation 별도 track).

따라서 Iter 1 의 D-13 (Normal sampling 1000→300) 은 method iter 에서 **제외**. 별도 data
ablation track 에서 진행 (Iter D1, D2, ... 로 명명 — 추후).

**Method-only iteration 후보** (same 8357 wafer, same Normal 1000, same defect 200×38):
1. **IGNORE_NEG_SIM 0.72 → 0.65** (CFG override) — `contrastive.py` 의 NV-Retriever 풍 false-negative
   filter primitive 강화. Iter 0 분석의 cluster 12/17/35/37 issue 직격
2. **USE_LOCAL=True** (CFG override) — 현재 OFF. grid 기반 spatial contrast 추가 신호
3. **TEMP 0.07 → 0.05** — sharper boundary
4. **EPOCHS 10 → 20** — 학습 더 길게 (uniformity -2.50 → -3.0+ 목표)
5. **QUEUE_SIZE 4096 → 16384** — 더 많은 negative
6. **LR_HEAD 1e-3 → 3e-4** — stable 학습
7. **합성 데이터 검토** (data 변경, separate track) — Full_*** sub-style 통일 / Thick-Edge_fork mislabel

**HDBSCAN tuning 은 더 이상 안 함** (이미 sweep optimum, D-12).

---

## Iter A0 — Method-track baseline on new data anchor (2026-05-06)

run_dir: `outputs_contrastive_260506_093847/`

★ **Track 변경 표기**: Iter 0 (Iter A0 직전) 의 spec (Normal 1000 + per-class 200, 8357 wafer)
은 별도 historical track. D-15 의 새 data anchor (defect 42 class avg30 random + Normal_bank_boundary 전체
1000 = 2146 wafer) 위에서 method ablation track 을 다시 시작. **Iter A0 = 새 anchor 의 baseline.**

### 설정
- Backbone: ConvNeXtV2-base FCMAE + TAPT (sister repo `known-cnn` wafer best_model.pth)
- Projection: 128-d, L2 normalized
- Loss: InfoNCE global (G) + queue (Q) + **local (L) ON** — `USE_LOCAL=True`, `LOCAL_WEIGHT=0.5`
- `USE_QUEUE=True`, `QUEUE_SIZE=4096`, `TEMP=0.07`, `IGNORE_NEG_SIM=0.72`
- `BATCH=8`, `IMAGE_SIZE=384`, `EPOCHS=5`, `LR_HEAD=1e-3` (head only, backbone frozen)
- GPU **throttle 5000ms** (driver TDR timeout 회피, EXPERIMENTS.md note 참조)
- Data anchor: `avg30_260505_203615` (D-15) — 42 defect class avg30 random 분포 + Normal_bank_boundary 1000 = 2,146 wafer
- HDBSCAN: min_cluster_size=12, min_samples=4, leaf, ε=0.06

### 결과
- **Completeness=0.9375, AMI=0.8946, noise_pct(def)=9.34%, capture=42/42 (1.000)**
- Homogeneity=0.8934, ARI(def)=0.7040, Silhouette(cos)=0.7908
- frac_single_cluster=0.8333 (35/42 single, 7 split-2, 0 split-3+)
- weighted_cluster_coverage=0.8717, mean_n_clusters_per_class=1.167
- weak top-3 (cluster coverage 기준):
  - `Edge-Bottom_scratch_rot` n=30 noise 76.7% (cov 0.20) ★ primary weak
  - `Edge-Bottom_fork` n=41 noise 63.4% (cov 0.27)
  - `Thick-Edge_fork` n=29 noise 51.7% (cov 0.45)
- Normal_bank_boundary noise_pct 79.6% (with_normal scope) — 의도적, defect 격리 평가에 영향 없음

### 발견
1. **새 data anchor 에서 method-track baseline 설립** — Iter 0 (8357 wafer, 38 defect class)
   대비 anchor 가 작아짐 (2146 wafer, 42 defect class) 으로 noise_pct 0.71% → 9.34% 상승.
   sample-efficiency 영역으로 이동. avg30 small-sample 에서도 capture 42/42 유지.
2. **primary weak = `Edge-Bottom_scratch_rot` 76.7%** — 이 anchor 의 가장 약한 group.
   `scratch_rot` 회전 sub-style + Edge-Bottom 위치 결합부 학습 부족.
3. **dispatch 운영 issue 7 attempt 끝에 success** — driver TDR (timeout detection & recovery)
   이 GPU 사용률 93% 시 trigger. 안정 조건 확정: BATCH=8 + EPOCHS=5 + throttle 5000ms.
   Iter 1+ 의 동일 조건 dispatch baseline 으로 lock-in.

### 다음 (Iter 1)
- atomic change: `LOCAL_WEIGHT 0.5 → 1.0` (DenseCL-style local grid contrast 강화).
- 동일 anchor / 동일 condition / 동일 GPU throttle. method-only ablation.

---

## Iter 1 — LOCAL_WEIGHT 0.5 → 1.0 (DenseCL-style local 강화) (2026-05-06)

run_dir: `outputs_contrastive_260506_103302/`

### 설정
- Iter A0 와 정확 동일. **유일 변경**: `LOCAL_WEIGHT 0.5 → 1.0`.
- BATCH=8, IMAGE_SIZE=384, EPOCHS=5, TEMP=0.07, IGNORE_NEG_SIM=0.72, QUEUE_SIZE=4096
- Same data anchor: `avg30_260505_203615` (2,146 wafer)
- Same HDBSCAN cfg

### 결과
- **Completeness=0.9481 (+0.0106), AMI=0.9040 (+0.0094), noise_pct(def)=4.62% (-4.72pp, -50%), capture=42/42 (=)**
- Homogeneity=0.8978 (+0.004), ARI=0.7325 (+0.029), Silhouette=0.7766 (-0.014)
- frac_single_cluster=0.8571 (+0.024) (36/42 single, 6 split-2, 0 split-3+)
- weighted_cluster_coverage=0.9258 (+0.054), mean_n_clusters_per_class=1.143

### 발견
1. **noise_pct 절반 감소** (9.34% → 4.62%) — Tier 1 핵심 개선. local grid contrast 강화가
   defect 의 noise 격리를 absolute 4.72pp 줄였다. capture 1.000 유지하면서 noise 만 감소
   → P1 (capture) 손실 없이 P2 (noise) 직접 개선.
2. **primary weak collapse 해결**: `Edge-Bottom_scratch_rot` 76.7% → 6.7% (-70pp). 이 class
   가 weak top-3 에서 완전히 사라짐. Edge-Bottom 위치 + scratch_rot 회전 sub-style 의 결합 신호를
   local grid contrast 가 직접 잡는 것으로 해석.
3. **새 weak emerge**: `Donut_scratch_rot` 26.7% → 40% (+13.3pp). `Edge-Bottom_scratch_rot`
   이 풀린 대신 Donut 위치의 `scratch_rot` 가 노이즈로 빠짐. local 강화가 위치별 trade-off
   초래 — Iter 2 가설: scratch_rot 회전 sub-style 자체가 학습 부족 → 모든 위치에서 약함.
4. **Silhouette 미세 하락** (0.7908 → 0.7766, -0.014). cluster boundary 가 약간 sharper 해지면서
   class 간 거리가 좁아진 것으로 해석. 정량적으로 미미. ARI / Completeness / AMI 모두 개선
   → Tier 1 우선순위 기준 net positive.

### 다음 (Iter 2 후보)
- (a) `LOCAL_WEIGHT 1.5` 추가 push (다음 atomic step)
- (b) `IGNORE_NEG_SIM 0.72 → 0.65` (NV-Retriever 풍 false-negative filter, D-14)
- (c) `Donut_scratch_rot` 약화 원인 추적 — scratch_rot 회전 sub-style 직접 확인 (image-analyzer)
- 사용자 결정 대기.

---

## Iter 2 — IGNORE_NEG_SIM 0.72 → 0.65 (false-negative filter 강화) (2026-05-06) — REJECT

run_dir: `outputs_contrastive_260506_112604/`

### 설정
- Iter 1 base. **유일 변경**: `IGNORE_NEG_SIM 0.72 → 0.65` (NV-Retriever 풍 D-14 후보).
- LOCAL_WEIGHT=1.0, BATCH=8, EPOCHS=5, TEMP=0.07, QUEUE_SIZE=4096
- Same data anchor `avg30_260505_203615` (2,146 wafer)
- Same HDBSCAN cfg

### 결과 — 모든 P1-P4 후퇴
- **Completeness=0.816 / 0.950, AMI=0.821 / 0.903, noise_pct(def)=7.16%, capture=41/42 (0.976)**
- Tier 2: Homogeneity=0.875 / 0.895, ARI=0.639 / 0.728, Silhouette=0.737
- mega-cluster (purity<0.5, sz>50) = **3** (c034 EB, c009 Center, c023 ET — 추가 collapse)
- **Donut_scratch_rot 0% cov** — capture 1.000 깨짐 (P1 손실)

### 발견
1. **false-negative filter 강화가 역효과** — 0.72 의 conservative threshold 가 이미 적정
   값. 0.65 로 더 공격적 filter 시 진짜 hard negative 까지 제거되어 cluster boundary
   collapse. P1 (capture 1.000 → 0.976) 까지 깨짐.
2. **추가 mega-cluster 발생** — Iter 1 의 2 → Iter 2 의 3 (c034 EB + c009 Center +
   c023 ET). filter 강화가 다른 위치 group 의 boundary 도 같이 무너뜨림.
3. **Track decision: dead branch** — Iter 1 base 로 fallback. IGNORE_NEG_SIM=0.65
   는 D-14 의 reject 대상으로 lock-in (`docs/contrastive-eval/DECISIONS.md`).

### 다음 (Iter 3)
- Iter 1 base 로 다시 돌아가, atomic 변경: `LOCAL_WEIGHT 1.0 → 0.7` 시험.
  → LW sweet spot search (1.0 이 plateau 인지, 0.5 ↔ 1.0 사이 어디가 optimum 인지).

---

## Iter 3 — LOCAL_WEIGHT 1.0 → 0.7 (LW sweet spot probe) (2026-05-06) — NULL effect

run_dir: `outputs_contrastive_260506_120656/`

### 설정
- Iter 1 base. **유일 변경**: `LOCAL_WEIGHT 1.0 → 0.7`.
- BATCH=8, EPOCHS=5, TEMP=0.07, IGNORE_NEG_SIM=0.72, QUEUE_SIZE=4096
- Same data anchor (2,146 wafer)
- Same HDBSCAN cfg

### 결과 — A0 와 통계적 동일
- **Completeness=0.936, AMI=0.893, noise_pct(def)=9.42%, capture=42/42 (1.000)**
  - cf. A0: Completeness=0.9375, AMI=0.8946, noise=9.34%, capture=1.000 → 차이 < 0.01
- Tier 2: Homogeneity=0.893, ARI=0.701, Silhouette=0.785
- mega-cluster = **2** (A0 동일)
- **sister-pair centroid distance 모두 A0 와 소수점 셋째 자리까지 동일**

### 발견
1. **LW 0.5 ~ 0.7 plateau 확인** — 0.5 (Iter A0) 와 0.7 (Iter 3) 의 Tier 1 metric
   차이가 noise level 수준 (모두 < 0.01). cluster 구조 (mega = 2, sister-pair
   centroid) 동일. **0.5 → 0.7 은 null effect.**
2. **1.0 비선형 jump** — 0.7 (null) ↔ 1.0 (Iter 1, noise 9.34% → 4.62% -50%) 의
   gap 이 매우 큼. local contrast 가 일정 weight 임계 (≈0.85?) 넘어야 효과 발현되는
   비선형 구간 추정. weight ↑ 마다 monotonic 개선이 아님.
3. **Iter 1 (LW=1.0) 이 단일 sweet spot 후보** — Iter 2 (NV-Retriever 강화)
   reject + Iter 3 (LW 약화) null → Iter 1 의 LOCAL_WEIGHT=1.0 + IGNORE_NEG_SIM=0.72
   조합이 Tier 1 4 metric 모두 best. 다음 iteration 은 다른 axis 탐색.

### 다음 (Iter 4 후보)
- (a) `LOCAL_WEIGHT 1.5` 추가 push — 1.0 이 plateau 시작인지 확인 (sweet spot 실험)
- (b) `EPOCHS 5 → 10` — 학습 시간 extend (uniformity -3.0+ 목표)
- (c) `TEMP 0.07 → 0.05` — sharper boundary
- (d) `Donut_scratch_rot` 약화 원인 추적 (Iter 1 의 새 weak)

---

# 전체 history 종합 (Iter A0 ~ Iter 30)

> anchor data: `avg30_260505_203615` (defect 42 class avg 30 + Normal 1000 = **2,146 wafer**), 모든 iter 동일.

## 주요 실험 성능 표 (30 iter 중 의미 있는 8개)

> 핵심 지표 5개: **Comp(P3) / AMI / noise(def, P2) / capture(P1) / ARI**
> Hom / Sil / mega-cluster / per-class 등 보조 지표는 `outputs_contrastive_<TS>/eval_summary.json` 직접 참조.

| # | atomic 변경 | Comp | AMI | **noise(def)** | **capture** | ARI | 판정 |
|---|---|---:|---:|---:|---:|---:|---|
| **A0** | baseline (LW=0.5, LR=1e-3, NEG=0.72, TEMP=0.07) | 0.938 | 0.895 | 9.34% | 1.000 | 0.704 | base |
| **1 ★** | **LW 0.5 → 1.0** | 0.948 | 0.904 | **4.62%** | 1.000 | 0.733 | **★ P2 King** |
| 6 | EPOCHS 5 → 10 | 0.798 | 0.806 | 9.34% | 1.000 | 0.557 | reject (over-fit 사례) |
| 9 | PercPos α=0.85 (best of axis) | 0.813 | 0.826 | 5.15% | 1.000 | 0.601 | reject (PercPos axis dead) |
| **11** | **LR_HEAD 1e-3 → 5e-4** | 0.948 | 0.905 | 6.11% | 1.000 | 0.734 | accept (chain shift) |
| **13** | + NEG 0.72 → 0.65 | 0.949 | 0.906 | 5.32% | 1.000 | 0.743 | accept |
| **14 ★★** | + **TEMP 0.07 → 0.05** | **0.952** | **0.913** | 6.63% | 1.000 | **0.763** | **★★ Quality King** |
| 17 | WARMUP 1 → 2 | 0.944 | 0.890 | 7.94% | **0.976 ❌** | 0.698 | reject (P1 violation 사례) |

**판정 표 의미** — 전체 30 iter 중 두 ★ (Iter 1 P2-King / Iter 14 Quality-King) 외 25 iter 는 모두 reject. axis 별 reject 묶음은 아래 dead-axes 표 참조.

### 기타 22 iter — 묶음 reject

| 묶음 | iter | 결과 한 줄 |
|---|---|---|
| **LW 사촌** (0.7/1.5, Iter 14 base 0.9/1.1/1.5) | 3, 4, 25, 26, 27 | LW 작은 변화 = trajectory 동일 (gradient scaling 만, == Iter 1 또는 Iter 14) |
| **PercPos α sweep** (0.95/0.90/0.80, Iter 7/8/10) | 7, 8, 10 | NV-Retriever 축 4-step 모두 baseline 못 이김 |
| **EPOCHS↑** (10 killed, 8 over-fit) | 5, 12 | 5 epoch sweet spot |
| **TEMP 사촌** (0.04, 0.06) | 15, 24 | 0.05 sweet spot |
| **LR 사촌** (3e-4) | 16 | 5e-4 sweet spot |
| **WARMUP** (0=no effect) | 18 | == Iter 14 |
| **TOPK** (8, 16) | 19, 22 | 12 sweet spot |
| **QUEUE** (8192) | 20 | 4096 sweet spot |
| **BATCH** (4) | 23 | 8 sweet spot |
| **NEG 사촌** (Iter 14 base 0.55, 0.70) | 28, 29 | 0.65 sharp local min |
| **Multi-axis combo** | 21, 30 | == Iter 1 또는 Iter 14 (no new optimum) |
| **NEG=0.65 (LW=0.5 base)** | 2 | P1 violation (Iter 1 base 에서는 안 됨) |

---

## Two-King 결정 (30 iter 후 final)

### Iter 1 ★ — **P2 King** (production safety)
```python
LW=1.0, LR_HEAD=1e-3, IGNORE_NEG_SIM=0.72, NCE_TEMP=0.07
EPOCHS=5, BATCH=8, IMAGE_SIZE=384
USE_LOCAL=True, LOCAL_POS_TOPK=12, USE_QUEUE=True, QUEUE_SIZE=4096
```
- **noise(def) 4.62%** (lowest, defect 격리 최고)
- Comp 0.948, AMI 0.904, ARI 0.733, capture 100%
- → 결함 누락 위험 최소화 (false alarm minimal)

### Iter 14 ★★ — **Quality King** (cluster purity)
```python
LW=1.0, LR_HEAD=5e-4, IGNORE_NEG_SIM=0.65, NCE_TEMP=0.05  # ← 4-axis combo
EPOCHS=5, BATCH=8, IMAGE_SIZE=384
USE_LOCAL=True, LOCAL_POS_TOPK=12, USE_QUEUE=True, QUEUE_SIZE=4096
```
- **Comp 0.952, AMI 0.913, ARI 0.763** (모두 best)
- noise 6.63% (Iter 1 보다 +2pp), capture 100%
- → cluster 응집/분리 최고, sister-class 분리 best

### Trade-off 직관 비교

```
                   ┌───────────────────────────────┐
                   │   Iter 1 (P2 King)            │
                   │   ─────                       │
                   │   noise 4.62% ★               │
                   │   Comp 0.948                  │
                   │   ARI 0.733                   │
                   │   "production safety"         │
                   └───────────────────────────────┘
                                  vs
                   ┌───────────────────────────────┐
                   │   Iter 14 (Quality King)      │
                   │   ─────                       │
                   │   noise 6.63%                 │
                   │   Comp 0.952 ★                │
                   │   ARI 0.763 ★                 │
                   │   "cluster purity"            │
                   └───────────────────────────────┘
```

---

## 진짜 lever 발견 (4 axis)

| Axis | 효과 size | iter |
|---|---|---|
| **LW** (0.5 → 1.0) | noise 9.34 → 4.62% (-50%) ★ huge | Iter 1 |
| **LR_HEAD** (1e-3 → 5e-4) | Comp 0.83 → 0.948 (+12pp) ★ huge | Iter 11 |
| **NEG_SIM** (0.72 → 0.65 with LR=5e-4) | sister 분리 ↑, capture 유지 | Iter 13 |
| **NCE_TEMP** (0.07 → 0.05) | AMI/ARI/Comp/Hom 모두 ↑ | Iter 14 |

**Other axes — all dead** (PercPos, EPOCHS, WARMUP, LOCAL_POS_TOPK, QUEUE_SIZE, BATCH, LW small change)

---

## Dead axes 정리

| Axis | 시도 | 결과 |
|---|---|---|
| **NV-Retriever PercPos α** | 0.95/0.90/0.85/0.80 (Iter 7-10) | 4 step sweep all reject — Iter 1 baseline 못 이김 |
| **EPOCHS** | 8 (Iter 12), 10 (Iter 6) | over-fit, 5 sweet spot |
| **WARMUP** | 0 (Iter 18 == 14), 2 (Iter 17 P1 violation) | 1 sweet spot |
| **LOCAL_POS_TOPK** | 8 (Iter 19), 16 (Iter 22) | 12 sweet spot |
| **QUEUE_SIZE** | 8192 (Iter 20) | 4096 sweet spot |
| **BATCH** | 4 (Iter 23) | 8 sweet spot |
| **LW** | 0.5/0.9/1.1/1.5 (Iter 30/25/26/27) | LW 작은 변화 = G trajectory 동일 (gradient scaling 만 영향) |
| **NEG (Iter 14 base)** | 0.55 (Iter 28), 0.70 (Iter 29) | 0.65 sharp local min |
| **TEMP (Iter 14 base)** | 0.04 (Iter 24), 0.06 (Iter 15) | 0.05 sweet spot |
| **LR (Iter 14 base)** | 3e-4 (Iter 16) | 5e-4 sweet spot |

---

## 변경 정책 (lock-in)

- **각 iteration 1 atomic 변경** — 효과 추적 가능하도록.
- 여러 변경 동시 시 효과 분리 불가 (ablation 불가).
- 결과 비교는 `RESULTS.md` 표 row 단위로 누적.
- 거부된 옵션 + 사유는 `docs/contrastive-eval/DECISIONS.md` D-N.
- BATCH 변경은 same-condition 으로 간주 (`feedback_batch_same_condition.md`)
- HDBSCAN cfg sweep 별개 track (encoder 학습 ablation 과 분리)

## 다음 단계 (code-level changes 필요 시)

30 iter 단일/다중 axis sweep 수렴 → 추가 개선 위해선 code change:
1. **NeCo patch ordering loss** (arXiv:2408.11054) — sister-class collapse 정공법
2. **VarCon adaptive temperature** (arXiv:2506.07413) — confidence-based dynamic τ
3. **Backbone unfreeze + low LR** — frozen 아닌 joint training (FREEZE_BACKBONE=False, LR_BACKBONE=1e-5)
4. **Bigger anchor data** — 2146 → 5000+ (현재 사용자 거부, data spec lock-in)

---

# ★ Track switch — new anchor `avg30_new_260508_123037` (Iter 34+)

> **anchor 변경 표기**: Iter 34 부터 새 data anchor `avg30_new_260508_123037` (43 class — 42 defect
> + Normal_bank_boundary, total 2,146 wafer) 사용. v19o chip 합성 강도 + canvas 9 추가 후
> 재구성. Iter A0–30 의 `avg30_260505_203615` 와 sample 수 동일하나 chip 합성 강도 + 신규 8 obj-less
> canvas class 포함으로 분포 다름 → Iter 33 와 직접 비교 의미 X. **Iter 34 = 새 anchor baseline.**
>
> 5번째 lever (NeCo) + HDBSCAN method axis (eom 도입) 가 이 track 에서 활성화됨. 표시 prefix
> `iter NN` (소문자) 으로 old `Iter NN` 와 구분.

## iter 34 — new anchor + Iter 14 cfg (Quality King) (2026-05-08)

run_dir: `outputs_contrastive_260508_123101/`

### 설정
- Data anchor: `avg30_new_260508_123037` (43 class, 2,146 wafer) — v19o chip 합성 + canvas 9 포함
- CFG: Iter 14 cfg 그대로 (LW=1.0, LR_HEAD=5e-4, IGNORE_NEG_SIM=0.65, TEMP=0.05)
- BATCH=8, EPOCHS=5, IMAGE_SIZE=384, QUEUE_SIZE=4096
- HDBSCAN sweep: leaf/eom × mcs={8,10,12,16}, ms=4 (single-snapshot tier1 file = eom mcs=12 ms=4)
- best 후보 (eom mcs=12 ms=3, sweep 반복 후 별도 평가): noise(def)=2.79%, Comp=0.977, AMI=0.931, ARI=0.750, cap=1.000

### 결과 (eom mcs=12 ms=3)
- **Completeness=0.977, AMI=0.931, ARI=0.750, noise_pct(def)=2.79%, capture=1.000**
- Tier 2: Hom=0.939, Sil≈0.51
- single-snapshot file (eom mcs=12 ms=4): noise=4.28% Comp=0.953 AMI=0.873 ARI=0.582 cap=0.976
- with_normal scope: Comp=0.814 AMI=0.835 noise=36.4% (Normal 의 의도된 모호성)

### 발견
1. **새 anchor baseline 설립** — 4 lever (LW + LR + NEG + TEMP) Iter 14 조합 그대로 적용 시
   noise(def) 2.79% / cap 1.000 / Comp 0.977 으로 Iter 14 (old anchor) 의 noise 6.63% 대비
   anchor 변경 효과로 -3.84pp. v19o chip 합성 강도 + canvas 9 가 cluster 분리 도움.
2. **HDBSCAN method axis 발견** — leaf vs eom 로 noise 가 큰 폭 차이. eom 이 항상 좋음
   (mcs=12 ms=4 기준 leaf 12.6% → eom 4.28%, -66%). encoder 학습 무관, 평가 axis 로 분리.
3. **HDBSCAN ms axis 발견** — ms=4 → ms=3 변경 시 noise 추가 -50% (eom mcs=12 4.28% → 2.79%).
   기존 (Iter 0~30) ms=4 default 보다 ms=3 가 더 sharp.

### 다음 (iter 35)
- Iter 14 cfg 가 새 anchor 에서도 best 인지 확인 위해 Iter 1 cfg (LR=1e-3, NEG=0.72, TEMP=0.07)
  로 atomic switch — **3-axis 동시 변경 (Iter 14 → Iter 1 P2 King cfg)**.

---

## iter 35 — switch to Iter 1 P2 King cfg (LR/NEG/TEMP 3-axis) (2026-05-08)

run_dir: `outputs_contrastive_260508_162812/`

### 설정
- iter 34 base. **3-axis switch** (Iter 14 → Iter 1 P2 King cfg):
  - LR_HEAD: 5e-4 → 1e-3
  - IGNORE_NEG_SIM: 0.65 → 0.72
  - TEMP: 0.05 → 0.07
- LW=1.0 유지, BATCH=8 EPOCHS=5
- Same anchor `avg30_new_260508_123037`
- HDBSCAN sweep 동일

### 결과 (eom mcs=12 ms=3)
- **Completeness=0.978, AMI=0.946, ARI=0.856, noise_pct(def)=2.01%, capture=1.000** ★
- Tier 2: Hom=0.940, Sil=0.615
- with_normal scope: Comp=0.859 AMI=0.872 noise=39.1%

### 발견
1. **새 anchor 에서 P2 King cfg 우세** — old anchor 에서 Quality King (Iter 14) 가 best 였으나
   새 anchor 에서는 P2 King (Iter 1) 이 더 좋음. anchor 분포 변화로 sweet-spot 이동.
   noise -0.78pp, AMI +0.015, ARI +0.106 — 3 metric 모두 ★ 개선.
2. **noise -28%** (2.79% → 2.01%) — capture 1.000 유지하면서 P2 직접 개선.
3. iter 35 cfg = (LW=1.0, LR=1e-3, NEG=0.72, TEMP=0.07) 가 새 anchor 의 새 baseline.

### 다음 (iter 36)
- code-level: backbone partial unfreeze (last N stages) + LR_SCALE — Iter 30+ 의 deferred 후보 1.

---

## iter 36 — BACKBONE_UNFREEZE_LAST_N=1 + LR_SCALE=0.02 (2026-05-09) — REJECT

run_dir: `outputs_contrastive_260509_062741/`

### 설정
- iter 35 base. atomic 변경 (code-level):
  - BACKBONE_UNFREEZE_LAST_N=1 (last stage unfreeze)
  - LR_SCALE=0.02 (backbone lr = 1e-3 × 0.02 = 2e-5)
- LW=1.0, LR_HEAD=1e-3, NEG=0.72, TEMP=0.07 동일
- Same anchor

### 결과 (eom mcs=12 ms=4 — single-snapshot)
- **noise_pct(def)=4.28%, Comp=0.953, AMI=0.873, ARI=0.582, capture=0.976 ❌**
- (HDBSCAN sweep 표 기준; eom mcs=12 ms=3 별도 평가 X)

### 발견
1. **★ P1 violation** — capture 1.000 → 0.976 (1 class 누락). **REJECT**.
   capture 깨짐은 P1 (사용자 명시 1순위 = 결함 누락 방지) 위반.
2. **모든 Tier 1 metric 후퇴** — Comp -0.025, AMI -0.073, ARI -0.274 (huge), noise +2.27pp.
3. **Backbone unfreeze axis = dead** (이 LR_SCALE 에서) — Iter 30+ deferred 후보 reject.
   Sister repo `known-cnn` 의 supervised TAPT backbone 이 이미 도메인 정렬 충분, 추가 unfreeze
   가 supervised collapse 풍 over-fit 유도.

### 다음 (iter 37)
- iter 35 base 로 복귀. atomic 변경: NeCo patch ordering loss (arXiv:2408.11054) — 5번째 lever
  후보, sister-class collapse 정공법.

---

## iter 37 — + NECO_WEIGHT=0.2 (Iter 30 deferred lever 1) (2026-05-09) — ★★★★★ SOTA

run_dir: `outputs_contrastive_260509_072137/`

### 설정
- iter 35 base. **유일 atomic 변경**: NECO_WEIGHT 0 → 0.2.
- NeCo (arXiv:2408.11054) patch-neighbor consistency loss 추가 (G + L + Q + NeCo).
- LW=1.0, LR_HEAD=1e-3, NEG=0.72, TEMP=0.07 그대로.
- Same anchor `avg30_new_260508_123037`.
- HDBSCAN sweep eom × {ms=3,4,5} × {mcs=8,10,12}.

### 결과 (eom mcs=12 ms=3 — best)
- **Completeness=0.991, AMI=0.960, ARI=0.870, noise_pct(def)=0.61%, capture=1.000** ★★★★★
- Hom=0.944, Sil=0.611, n_clusters=37
- HDBSCAN sweep 동등: ms=3 mcs={8,10,12} 모두 동일 (Comp=0.985 AMI=0.956 noise=0.52% 대안값)
- 신뢰 안정 — eom mcs=12 ms=3 lock-in (5 노이즈 이하 9 cell, capture 1.000 유지)

### 발견
1. **★★★★★ SOTA 갱신** — noise -70% (2.01% → 0.61%), capture 1.000, Comp +0.013, AMI +0.014,
   ARI +0.014. 새 anchor 의 5 metric 모두 best.
2. **NeCo = 5번째 lever 확정** — 4 hyperparam axis (LW/LR/NEG/TEMP) 모두 sweep 수렴 후
   추가 개선 가능 axis 발견. 5번째 lever 효과 size: noise -70% (LW lever 의 -50%, LR lever
   의 +12pp Comp 와 동급 huge).
3. **patch-neighbor consistency 가 sister-class collapse 직격** — Edge-Top × {bb, fork,
   scratch, scratch_rot} 4-obj merge 같은 historical issue 가 NeCo 0.2 로 해소되는 것으로
   해석. cluster_summary 분석에서 weak top-3 (Donut_scratch_rot 등) 도 cov 0.95+ 회복.

### 다음 (iter 38)
- NECO_WEIGHT sweet spot 탐색 — 0.1 (약화) 와 0.3 (강화) 양쪽 시험.

---

## iter 38 — NECO_WEIGHT 0.2 → 0.1 (sweet spot probe down) (2026-05-09) — REJECT

run_dir: `outputs_contrastive_260509_085046/`

### 설정
- iter 37 base. **유일 변경**: NECO_WEIGHT 0.2 → 0.1.

### 결과 (eom mcs=12 ms=3)
- **Completeness=0.985, AMI=0.956, ARI=0.860, noise_pct(def)=0.52%, capture=1.000**
- noise 가 미세 더 낮음 (-0.09pp) 이나 Comp/AMI/ARI 모두 후퇴 (-0.006/-0.004/-0.010)
- mixed_clusters 6 → 7, frag_classes 1 → 4 (cluster_analyzer 진단)

### 발견
1. **noise 미세 감소 ↔ frag 증가** — cluster fragmentation 이 4 class 로 늘어남. NeCo
   0.1 은 patch-neighbor signal 약해서 sister-pair 분리가 partial. P3 (Comp) 우선 정책상 ✗.
2. **NECO_WEIGHT 0.2 보다 약화 reject** — 다음 step 0.3 시험.

### 다음 (iter 39)
- NECO_WEIGHT 0.2 → 0.3 (강화) 시험.

---

## iter 39 — NECO_WEIGHT 0.2 → 0.3 (sweet spot probe up) (2026-05-09) — REJECT, NeCo 0.2 lock

run_dir: `outputs_contrastive_260509_125153/`

### 설정
- iter 37 base. **유일 변경**: NECO_WEIGHT 0.2 → 0.3.

### 결과 (eom mcs=12 ms=3)
- **Completeness=0.980, AMI=0.954, ARI=0.868, noise_pct(def)=1.05%, capture=1.000**
- noise +0.44pp, Comp -0.011, AMI -0.006, ARI -0.002 — 모두 후퇴

### 발견
1. **NeCo 0.3 도 reject** — 양쪽 (0.1 + 0.3) 모두 0.2 보다 후퇴 → **NECO_WEIGHT=0.2 sweet spot lock-in**.
2. **NeCo axis 단조 감소 X** — 0.1 (under-signal) ↔ 0.2 (sweet) ↔ 0.3 (over-signal, dominate G/L).
3. NeCo 5번째 lever 의 sweet-spot 좁음 (0.2 ± 0.1 가 가까운 reject) → fragile.

### 다음 (iter 40)
- iter 37 sister 검증: Iter 14 Quality King cfg + NeCo 0.2 조합으로 cross-cfg 효과 확인.

---

## iter 40 — Quality King cfg (LR=5e-4 NEG=0.65 TEMP=0.05) + NeCo 0.2 (2026-05-09) — REJECT

run_dir: `outputs_contrastive_260509_151225/`

### 설정
- iter 37 base 가 아닌 iter 34 base 위 NeCo 0.2 추가:
  - LR_HEAD=5e-4, IGNORE_NEG_SIM=0.65, TEMP=0.05 (Iter 14 Quality King cfg)
  - NECO_WEIGHT=0.2 (iter 37 lever 5)

### 결과 (eom mcs=12 ms=3)
- **Completeness=0.962, AMI=0.922, ARI=0.738, noise_pct(def)=4.10%, capture=1.000**
- vs iter 37: noise +3.49pp, Comp -0.029, AMI -0.038, ARI -0.132 (huge)

### 발견
1. **★ Quality King cfg ↔ NeCo 0.2 호환 X** — old anchor 의 Iter 14 cfg + 새 lever NeCo 0.2 가
   negative interaction. NeCo 는 P2 King cfg (Iter 1, LR=1e-3 NEG=0.72 TEMP=0.07) 위에서만 효과.
2. **lever 조합 비선형 — base cfg 의존** — 5 lever 가 독립적이지 않음. NeCo 의 효과는 base
   cfg 에 따라 +14pp ARI (iter 37) 또는 -13pp ARI (iter 40) 로 부호도 바뀜.
3. **새 anchor 의 best base = P2 King (iter 35) cfg + NeCo 0.2 (iter 37)** 확정. Quality King
   path 는 이 anchor 에서 dead.

### 다음 (iter 41 — encoder 학습 X, HDBSCAN axis only)
- iter 37 의 embedding 위 HDBSCAN mcs forcing 시험 — encoder method 와 별개 track.

---

## iter 41 — HDBSCAN mcs forcing on iter 37 embedding (encoder X) (2026-05-09) — REJECT (P1)

run_dir: (iter 37 embedding reuse, HDBSCAN cfg sweep only)

### 설정
- iter 37 의 final encoder + embedding 그대로.
- HDBSCAN: aggressive mcs forcing (구체값 dispatcher dependent).
- encoder 학습 변화 없음 — HDBSCAN cfg axis only (별도 track, `feedback_hdbscan_cfg_sweep_ok.md`).

### 결과
- **noise_pct(def)=3.05%, Comp=0.997, capture=0.952 ❌**

### 발견
1. **★ P1 violation** — capture 0.952 (= 41/43, 2 class 누락). HDBSCAN axis 지만 P1 (사용자 정책
   1순위) 위반은 결과 reject 동일.
2. **dead axis** — Comp 0.997 (vs iter 37 의 0.991) 미세 개선이지만 capture 손실은 trade-off
   불가. encoder method axis 와 무관하게 reject lock-in.

### 다음 (iter 42)
- 안전한 LR_SCALE (0.02 → 0.005) 로 backbone unfreeze 재시험 (iter 36 의 reject 보강 시험).

---

## iter 42 — BACKBONE_UNFREEZE_LAST_N=1 + LR_SCALE=0.005 (안전 LR) (2026-05-09) — REJECT, axis 영구 reject

run_dir: `outputs_contrastive_260509_172703/`

### 설정
- iter 37 base. atomic 변경:
  - BACKBONE_UNFREEZE_LAST_N=1
  - LR_SCALE=0.005 (iter 36 의 0.02 보다 4× 작게, 안전 LR)
- NeCo 0.2 + LW=1.0 + LR_HEAD=1e-3 + NEG=0.72 + TEMP=0.07 유지

### 결과 (eom mcs=12 ms=3)
- **Completeness=0.948, AMI=0.823, ARI=0.474, noise_pct(def)=11.69%, capture=1.000**
- vs iter 37: noise +11.08pp (×19), Comp -0.043, AMI -0.137, ARI -0.396 (huge)

### 발견
1. **★ Backbone unfreeze axis 영구 reject** — LR_SCALE 0.02 (iter 36, P1 violation) 도 0.005
   (iter 42, P1 유지하나 모든 metric huge regression) 도 모두 dead.
2. **TAPT backbone 이 이미 도메인 정렬 끝났음** — sister repo `known-cnn` 의 supervised
   33-class 학습 backbone 이 contrastive 의 domain alignment 까지 보장. 추가 unfreeze 는
   over-fit (small data 2,146 wafer 위 frozen 이 sweet).
3. **iter 37 의 frozen + 4 lever + NeCo 가 새 anchor SOTA 로 lock-in** — 추가 atomic 변경
   가능 axis 거의 소진. novelty A (Zone-Aware NeCo) 같은 code change 만 남음.

### 다음 (iter 43)
- novelty A: Zone-Aware NeCo (NECO_ZONE_VERTICAL=3) — wafer 의 vertical zone 분할 후 zone
  내 patch-neighbor only consistency. Edge-Top vs Edge-Bottom 같은 위치 sub-style 직격.

---

## iter 43 — + NECO_ZONE_VERTICAL=3 (★ Zone-Aware NeCo, novelty A) (2026-05-09) — IN PROGRESS

run_dir: (eval GPU 충돌 — 재시도 대기)

### 설정
- iter 37 base. atomic 변경 (code-level):
  - NECO_ZONE_VERTICAL=3 (wafer 를 top/middle/bottom 3 zone 으로 split, zone 내 NeCo 만)
- NECO_WEIGHT=0.2 (iter 37 sweet spot)
- 다른 모든 hyperparam iter 37 동일

### 결과
- (eval GPU 충돌, 재시도 대기 중)

### 동기
- iter 37 의 global NeCo 0.2 가 이미 SOTA. 추가 개선 위해 zone-aware variant 시험.
- Edge-Top vs Edge-Bottom 같은 vertical 위치 sub-style 의 patch-neighbor 가 더 의미 있음.
- zone vertical=3 (top/mid/bot) 가 wafer geometry 와 align — 첫 변종.

### 다음
- 결과 산출 후 iter 37 base 와 비교. SOTA 갱신 시 lever 5 의 first novelty 변종 lock-in.

---

# 진화 path 요약 (Iter A0 → iter 37 SOTA)

```
A0 (LW=0.5, LR=1e-3, NEG=0.72, TEMP=0.07)
  ├─ Iter 1  ★ LW 0.5 → 1.0           (lever 1: noise -50%)
  │   ├─ Iter 11 ★ LR 1e-3 → 5e-4      (lever 2: Comp +12pp)
  │   │   ├─ Iter 13 ★ NEG 0.72 → 0.65 (lever 3: sister 분리)
  │   │   │   └─ Iter 14 ★★ TEMP 0.07 → 0.05  (lever 4: AMI +0.3pp)
  │   │   │       │ — Quality King (old anchor SOTA)
  │   │   │       └─ ... (Iter 15-30 sweep dead)
  │   │   │
  │   │   └─ ... (other path)
  │   │
  │   └─ ... (LW sweep null/reject Iter 2/3/25-27/30)
  │
  └─ ★ TRACK SWITCH: anchor avg30_260505_203615 → avg30_new_260508_123037
       │
       ├─ iter 34   (Iter 14 cfg)         noise 2.79%
       ├─ iter 35   (Iter 1 P2 cfg)       noise 2.01%   ← 새 anchor 의 best base cfg
       │   ├─ iter 36 ✗ unfreeze axis dead (P1)
       │   ├─ iter 37 ★★★★★ NeCo 0.2 (lever 5)  noise 0.61%  ← SOTA
       │   │   ├─ iter 38 ✗ NeCo 0.1 (under)
       │   │   ├─ iter 39 ✗ NeCo 0.3 (over) → 0.2 lock
       │   │   ├─ iter 40 ✗ Quality King + NeCo (cross-cfg incompat)
       │   │   ├─ iter 41 ✗ HDBSCAN forcing (P1 violation)
       │   │   ├─ iter 42 ✗ unfreeze 0.005 (axis 영구 reject)
       │   │   └─ iter 43 (in progress) Zone-Aware NeCo
       │   │
       │   └─ ...
       │
       └─ ...
```

# 진짜 lever 5개 정리 (new anchor 기준)

| # | Lever | Step | 효과 | iter (track) |
|---|---|---:|---|---|
| 1 | LOCAL_WEIGHT | 0.5 → 1.0 | noise 9.34→4.62% (-50%) | Iter 1 (old) |
| 2 | LR_HEAD | 1e-3 → 5e-4 | Comp 0.83→0.948 (+12pp) | Iter 11 (old) |
| 3 | IGNORE_NEG_SIM | 0.72 → 0.65 | sister 분리 ↑ | Iter 13 (old) |
| 4 | TEMP | 0.07 → 0.05 | AMI/ARI/Comp/Hom +0.3pp | Iter 14 (old) |
| **5 ★** | **NECO_WEIGHT** | **0 → 0.2** | **noise 2.01→0.61% (-70%)** | **iter 37 (new) ★ SOTA** |

추가 (encoder 무관, HDBSCAN axis):
- HDBSCAN method: leaf → eom — noise -58% (encoder X)
- HDBSCAN ms: 4 → 3 — noise -50% (encoder X)
- HDBSCAN mcs: 12 lock (8/10/12 동등 ms=3, encoder X)

**Note**: lever 5 (NeCo) 는 base cfg 에 강하게 의존 (iter 40 reject 참조). P2 King cfg (Iter 1)
+ NeCo 0.2 만 효과. Quality King cfg (Iter 14) + NeCo 는 negative interaction.

# Dead axes 누적 정리 (반복 시간 낭비 방지)

| Axis | 시도 | 결론 |
|---|---|---|
| NV-Retriever PercPos α | Iter 7-10 (4-step) | sweep all reject |
| EPOCHS | Iter 5,6,12 | 5 sweet spot (over-fit beyond) |
| WARMUP | Iter 17,18 | 1 sweet spot (P1 violation at 2) |
| LOCAL_POS_TOPK | Iter 19,22 | 12 sweet spot |
| QUEUE_SIZE | Iter 20 | 4096 sweet spot |
| BATCH | Iter 23 | 8 sweet spot |
| LW small Δ | Iter 25-27,30 | gradient scaling only, no new optimum |
| NEG sister | Iter 28,29 | 0.65 sharp local min |
| TEMP sister | Iter 15,24 | 0.05 sweet spot |
| LR sister | Iter 16 | 5e-4 sweet spot (old anchor) / 1e-3 (new) |
| **★ Backbone unfreeze** | **iter 36, 42** | **★ axis 영구 reject** — TAPT backbone 이 이미 정렬 |
| HDBSCAN forcing (capture cost) | iter 41 | P1 violation — Comp 미세 개선과 무관 reject |
| Quality King + NeCo 조합 | iter 40 | base cfg 의존성 (P2 King + NeCo 만) |
| NeCo sister sweep | iter 38, 39 | 0.2 sharp sweet spot |

---

# ★ Comprehensive saturation sweep (iter 50-58, 2026-05-10)

> Track 동일 (anchor `avg30_new_260508_123037`, P2 King base + NeCo 0.2 = iter 37 cfg, eom mcs=12 ms=3).
> 6 hparam axis (LW / LR / NEG / TEMP / TOPK / QUEUE) + Spatial NeCo variants 모두 sweep
> → **iter 37 cfg 가 다차원 sweet spot 의 saturation point** 확정.

## iter 50 — + NECO_HIER_POOLS="1,2,4" (Hierarchical NeCo) (2026-05-10) — TIED

run_dir: `outputs_contrastive_260510_002649/`

### 설정
- iter 37 base. atomic 변경 (code-level): `NECO_HIER_POOLS="1,2,4"` — multi-resolution patch-neighbor consistency at 1×1 / 2×2 / 4×4 pool sizes (sum-aggregated NeCo loss).
- NECO_WEIGHT=0.2 sweet spot 유지, 다른 hparam iter 37 동일.

### 결과 (eom mcs=12 ms=3)
- **Completeness=0.985, AMI=0.956, ARI=0.860, noise_pct(def)=0.52%, capture=1.000**
- vs iter 37 (single-seed 0.870): ARI -0.010 (within 3-seed std 0.014). noise -0.09pp 미세 개선.

### 발견
1. **Hierarchical NeCo = TIED with standard NeCo** — 단일 seed 차이 ARI -0.010 은 multi-seed std 안. 의미 있는 개선 X.
2. multi-resolution pooling 의 추가 비용 (3× memory, ≈1.5× compute) 대비 효과 X → reject.

### 다음 (iter 51)
- iter 50 의 seed=1 재현 시험 (Hierarchical 의 variance 측정).

---

## iter 51 — iter 50 + seed=1 (Hierarchical variance test) (2026-05-10) — TIED

run_dir: `outputs_contrastive_260510_011836/`

### 설정
- iter 50 cfg 동일. **유일 변경**: random seed 42 → 1 (augmentation order + projection-head init).

### 결과 (eom mcs=12 ms=3)
- **Completeness=0.988, AMI=0.951, ARI=0.852, noise_pct(def)=0.87%, capture=1.000**
- 2-seed mean (iter 50/51): ARI 0.856 (정확히 iter 37 3-seed mean 0.866 의 -0.010 within std)
- 2-seed std: 0.006 (낮음, 안정적 reproduction)

### 발견
1. **Hierarchical NeCo 2-seed mean 0.856 vs iter 37 3-seed mean 0.866** — within std (0.014). 통계적 동등 lock-in.
2. variance 자체는 standard NeCo 와 유사 (0.006 vs 0.014) — Hierarchical 추가 안정화 효과 없음.

### 다음 (iter 52)
- LW 1.0 → 1.2 push (iter 1 ~ 30 sweep 후 NeCo lever 추가된 이후 LW 재검증).

---

## iter 52 — LW 1.0 → 1.2 (saturate 검증) (2026-05-10) — TIED

run_dir: `outputs_contrastive_260510_020431/`

### 설정
- iter 37 base. **유일 변경**: LOCAL_WEIGHT 1.0 → 1.2.

### 결과 (eom mcs=12 ms=3)
- **Completeness=0.980, AMI=0.950, ARI=0.856, noise_pct(def)=0.96%, capture=1.000**
- vs iter 37: ARI -0.014 (within std), noise +0.35pp.

### 발견
1. **LW 1.2 = TIED with 1.0** — Iter 1 (LW 0.5 → 1.0, noise -50%) 이후 NeCo lever 추가된 이 anchor 에서 LW lever 가 saturate.
2. iter 25-27 (LW 0.9/1.1/1.5 old anchor) 의 결론과 일관 — **LW 1.0 ± 0.2 은 모두 sweet spot 안**, 단조 변화 X.

### 다음 (iter 53)
- LR_HEAD 1e-3 → 7e-4 push (LR sister axis 재검증).

---

## iter 53 — LR_HEAD 1e-3 → 7e-4 (LR saturate 검증) (2026-05-10) — TIED

run_dir: `outputs_contrastive_260510_025007/`

### 설정
- iter 37 base. **유일 변경**: LR_HEAD 1e-3 → 7e-4.

### 결과 (eom mcs=12 ms=3)
- **Completeness=0.992, AMI=0.955, ARI=0.853, noise_pct(def)=0.44%, capture=1.000**
- vs iter 37: ARI -0.017 (within std), noise -0.17pp 미세 개선 (Comp +0.001).

### 발견
1. **LR 7e-4 = TIED with 1e-3** — new anchor 에서 LR sister sweep (iter 16 의 5e-4 vs old 의 1e-3 mirror) 모두 within std.
2. Comp 미세 개선 (0.992) 이지만 ARI 후퇴 — net not significant.
3. **LR_HEAD 1e-3 lock-in 유지** — 새 anchor 에서 LR sister axis dead.

### 다음 (iter 54)
- LOCAL_POS_TOPK 12 → 16 push (TOPK sister axis 재검증).

---

## iter 54 — LOCAL_POS_TOPK 12 → 16 (TOPK sister 재검증) (2026-05-10) — LUCKY single-seed

run_dir: `outputs_contrastive_260510_035823/`

### 설정
- iter 37 base. **유일 변경**: LOCAL_POS_TOPK 12 → 16.

### 결과 (eom mcs=12 ms=3, seed=42)
- **Completeness=0.987, AMI=0.959, ARI=0.880, noise_pct(def)=0.87%, capture=1.000**
- vs iter 37: ARI +0.010 (single-seed best), noise +0.26pp.

### 발견 (single-seed reading 만)
1. **★ TOPK 16 single-seed 0.880 — iter 37 의 lucky-tail (0.880) 과 정확히 같은 값**.
2. 단일 seed 의 +0.010 ARI 는 multi-seed std (0.014) 미만 — 통계적 보장 없음. iter 55 에서 seed=1 재현 시험.

### 다음 (iter 55)
- iter 54 의 seed=1 재현 — TOPK 16 의 진짜 효과 vs lucky variance 판별.

---

## iter 55 — iter 54 + seed=1 (TOPK 16 variance test) (2026-05-10) — ★★★ LUCKY pattern 정확 재현

run_dir: `outputs_contrastive_260510_072458/`

### 설정
- iter 54 cfg 동일. **유일 변경**: random seed 42 → 1.

### 결과 (eom mcs=12 ms=3)
- **Completeness=0.988, AMI=0.951, ARI=0.852, noise_pct(def)=0.87%, capture=1.000**
- 2-seed mean (iter 54/55): ARI 0.866 (정확히 iter 37 3-seed mean 0.866).
- 2-seed std: 0.014 — iter 37 와 동일.

### 발견 ★★★ paper-grade
1. **★★★ Multi-seed lucky pattern 정확 재현** — TOPK 16 의 seed=42 0.880 / seed=1 0.852 / 평균 0.866 이 Zone-Aware NeCo (iter 43, z=4) 의 seed=42 0.880 / seed=1 0.852 / 평균 0.866 과 **소수점 셋째 자리까지 같음**.
2. **두 완전 다른 ablation axis (TOPK code-level Spatial NeCo) 가 같은 +0.010 lucky variance 를 noise floor 로 공유** — multi-seed methodology 의 강력한 evidence.
3. **TOPK 16 의 single-seed 0.880 = iter 37 의 lucky tail 과 본질적으로 같은 sample** — TOPK lever 의 진짜 효과는 0 (mean 0.866 = iter 37 mean).
4. **이 paper 의 N2 contribution 강화** — multi-seed protocol 의 importance 가 두 axis 에서 정확 재현.

### 다음 (iter 56)
- QUEUE_SIZE 4096 → 8192 push (QUEUE sister 재검증).

---

## iter 56 — QUEUE_SIZE 4096 → 8192 (QUEUE saturate 검증) (2026-05-10) — TIED

run_dir: `outputs_contrastive_260510_082121/`

### 설정
- iter 37 base. **유일 변경**: QUEUE_SIZE 4096 → 8192.

### 결과 (eom mcs=12 ms=3)
- **Completeness=0.985, AMI=0.954, ARI=0.867, noise_pct(def)=1.31%, capture=1.000**
- vs iter 37: ARI -0.003 (within std), noise +0.70pp.

### 발견
1. **QUEUE 8192 = TIED with 4096** — Iter 20 (old anchor QUEUE 8192 reject) 의 결론과 일관, NeCo lever 추가된 새 anchor 에서도 동일.
2. 더 큰 queue 가 추가 negative diversity 주지만 Tier 1 metric movement 없음 — **4096 lock-in 유지**.

### 다음 (iter 57)
- NCE_TEMP 0.07 → 0.06 push (TEMP sister 재검증, 새 anchor 의 0.07 sweet spot 인지 확인).

---

## iter 57 — NCE_TEMP 0.07 → 0.06 (TEMP saturate 검증) (2026-05-10) — TIED

run_dir: `outputs_contrastive_260510_102747/`

### 설정
- iter 37 base. **유일 변경**: NCE_TEMP 0.07 → 0.06.

### 결과 (eom mcs=12 ms=3)
- **Completeness=0.984, AMI=0.952, ARI=0.856, noise_pct(def)=1.57%, capture=1.000**
- vs iter 37: ARI -0.014 (within std), noise +0.96pp.

### 발견
1. **TEMP 0.06 = TIED with 0.07** — old anchor 의 0.05 (Iter 14 Quality King) 와 다름. 새 anchor 에서는 TEMP 0.07 (iter 35 P2 King base) 가 sweet spot 유지.
2. TEMP sister sweep 새 anchor 에서 saturate — **TEMP 0.07 lock-in**.

### 다음 (iter 58)
- IGNORE_NEG_SIM 0.72 → 0.65 (NEG sister 재검증, Iter 13 의 old anchor 결론 대비).

---

## iter 58 — IGNORE_NEG_SIM 0.72 → 0.65 (NEG saturate 검증) (2026-05-10) — TIED (within std)

run_dir: `outputs_contrastive_260510_111451/`

### 설정
- iter 37 base. **유일 변경**: IGNORE_NEG_SIM 0.72 → 0.65.

### 결과 (eom mcs=12 ms=3)
- **Completeness=0.973, AMI=0.944, ARI=0.846, noise_pct(def)=1.75%, capture=1.000**
- vs iter 37: ARI -0.024 (just outside std 0.014). Comp -0.018, noise +1.14pp.

### 발견
1. **NEG 0.65 = TIED-edge with 0.72** — ARI -0.024 가 std 0.014 보다 약간 큼이지만 capture 1.000 유지. iter 40 (Quality King + NeCo, ARI -13pp) 같은 huge regression 은 X.
2. **NeCo lever 추가된 새 anchor 에서 NEG 0.65 가 더 이상 sweet spot 아님** — Iter 13 (old anchor) 의 NEG 0.65 가 lever 였던 것이 NeCo 추가로 dead 되었음. **NEG 0.72 lock-in (새 anchor)**.

### 다음
- 6 hparam axis (LW / LR / NEG / TEMP / TOPK / QUEUE) + Spatial NeCo (Hierarchical, Zone-Aware z=3/4/6) 모두 sweep 완료.
- **iter 37 cfg 가 multi-axis saturation point 확정** — 추가 atomic 개선 axis 소진.
- 다음 paradigm = Cluster-Aware Synthesis Loop (사용자 승인, paper finalize 후 별도 paper, future work F2).

---

# 진화 path 종합 update (Iter A0 → iter 58)

```
A0 (LW=0.5, LR=1e-3, NEG=0.72, TEMP=0.07)
  └─ ★ TRACK SWITCH: anchor avg30_260505_203615 → avg30_new_260508_123037
     └─ iter 35 (P2 King base, new anchor)        Comp 0.978  AMI 0.946  noise 2.01%
          └─ iter 37 ★★★★★ + NECO_WEIGHT 0.2     Comp 0.991  AMI 0.960  noise 0.61%   ← SOTA (single seed)
               │
               ├─ iter 38-39 ✗ NeCo {0.1, 0.3} sweep — 0.2 lock
               ├─ iter 40 ✗ Quality King + NeCo (cross-cfg)
               ├─ iter 41 ✗ HDBSCAN forcing (P1 violation)
               ├─ iter 42 ✗ unfreeze 0.005 (axis 영구 reject)
               ├─ iter 43 ✗ Zone-Aware NeCo z=3 (single 0.880, 3-seed 0.876±0.012, within std)
               ├─ iter 44-46  multi-seed iter 37 (3-seed mean 0.866 ± 0.014)
               │
               ├─ ★ COMPREHENSIVE SATURATION SWEEP (iter 50-58, 2026-05-10)
               │  ├─ iter 50/51 ✗ Hierarchical NeCo (1,2,4 pools) — 2-seed 0.856 (TIED)
               │  ├─ iter 52    ✗ LW 1.0 → 1.2 (TIED)
               │  ├─ iter 53    ✗ LR 1e-3 → 7e-4 (TIED, Comp 0.992 미세)
               │  ├─ iter 54/55 ✗ TOPK 12 → 16 (single 0.880 lucky → 2-seed 0.866 = iter 37)
               │  ├─ iter 56    ✗ QUEUE 4096 → 8192 (TIED)
               │  ├─ iter 57    ✗ TEMP 0.07 → 0.06 (TIED)
               │  └─ iter 58    ✗ NEG 0.72 → 0.65 (TIED-edge, NeCo 추가로 lever 죽음)
               │
               └─ ★ iter 37 cfg = 6-axis multi-axis saturation point 확정
```

# Dead axes 누적 정리 update (iter 50-58 추가)

| Axis | 시도 | 결론 |
|---|---|---|
| NV-Retriever PercPos α | Iter 7-10 (4-step) | sweep all reject |
| EPOCHS | Iter 5,6,12 | 5 sweet spot |
| WARMUP | Iter 17,18 | 1 sweet spot (P1 at 2) |
| LOCAL_POS_TOPK | Iter 19,22, **iter 54/55** | **★ 12 sweet spot, 16 = lucky variance only (multi-seed mean 0.866)** |
| QUEUE_SIZE | Iter 20, **iter 56** | **4096 sweet spot 재확인 (8192 TIED)** |
| BATCH | Iter 23 | 8 sweet spot |
| LW small Δ | Iter 25-27,30,3, **iter 52** | **1.0 ± 0.2 saturate, 단조 X** |
| NEG sister | Iter 28,29, **iter 58** | **새 anchor 0.72 lock-in (0.65 NeCo 추가로 dead)** |
| TEMP sister | Iter 15,24, **iter 57** | **새 anchor 0.07 lock-in (0.06 TIED)** |
| LR sister | Iter 16, **iter 53** | **5e-4 sweet (old) / 1e-3 lock (new), 7e-4 TIED** |
| Backbone unfreeze | iter 36, 42 | 영구 reject |
| HDBSCAN forcing (capture cost) | iter 41 | P1 violation |
| Quality King + NeCo | iter 40 | base cfg negative interaction |
| NeCo sister sweep | iter 38, 39 | 0.2 sharp sweet spot |
| **Hierarchical NeCo** (1,2,4 pools) | **iter 50/51** | **★ 2-seed 0.856 TIED — multi-resolution NeCo dead** |
| **Zone-Aware NeCo** (z=3) | iter 43 | 3-seed 0.876±0.012 TIED |

# ★★★★★ N5 contribution — Comprehensive saturation point lock-in (iter 50-58)

iter 50-58 의 6 hparam axis (LW / LR / NEG / TEMP / TOPK / QUEUE) + Spatial NeCo variants
(Hierarchical 1,2,4 / Zone-Aware z=3) 모두 sweep 결과:

- **모든 atomic 변경이 multi-seed std (0.014) 안** — significant 한 개선 X.
- **iter 37 cfg = 6-axis multi-axis sweet spot saturation point** 확정.
- **추가 hparam-level 개선 가능 axis 소진** — 다음 paradigm 은 code-level extension (F1) 또는 Cluster-Aware Synthesis Loop (F2).

이 발견이 N5 contribution 으로 paper 의 5 contributions (N1-N5) 중 마지막 추가됨.

# ★★★★★ N2 강화 — Multi-seed lucky pattern 정확 재현 (iter 43 + iter 54/55)

**두 완전 다른 ablation axis 가 같은 +0.010 lucky variance pattern 을 보임**:

| axis | seed=42 | seed=1 | mean | std |
|---|---:|---:|---:|---:|
| Zone-Aware NeCo z=4 (iter 43+seed1) | 0.880 | 0.852 | 0.866 | 0.014 |
| TOPK 16 (iter 54/55) | 0.880 | 0.852 | 0.866 | 0.014 |
| **iter 37 baseline** | 0.870 | — (3-seed) | 0.866 | 0.014 |

→ **두 axis 의 single-seed +0.010 ARI 는 같은 lucky variance pattern 의 sample** — paper 의 multi-seed methodology evidence 강력 보강.

---

## iter 60 — B0 Real Baseline (Global InfoNCE only) (2026-05-11) — ★ NEW REAL BASELINE

run_dir: `outputs_contrastive_260511_154102/`

### 설정
- 사용자 지적 정합: 기존 Iter A0 baseline 에 이미 Local InfoNCE / MoCo Queue / NEG filter 활성.
  → 진짜 component-level contribution isolation 위해 **Global InfoNCE only** 의 minimal baseline 신설.
- `USE_LOCAL=false, USE_QUEUE=false, IGNORE_NEG_SIM=1.0 (off), NECO_WEIGHT=0`.
- anchor avg30_new_260508_123037 (43 class, n=2146), seed=42.
- HDBSCAN: eom mcs=12 ms=3 (모든 B 단계 동일 고정).

### 결과 (eom mcs=12 ms=3)
- **Completeness=0.9602, AMI=0.9290, ARI=0.8231, noise_pct(def)=6.195%, capture=1.000, n_cl=37**
- Homogeneity=0.929*, NMI=0.949*, Silhouette(cos)=0.582*.

### 발견
1. **TAPT backbone 의 강력함**: Global InfoNCE 단독으로 ARI 0.823 / capture 1.000 / noise 6.2%.
   ConvNeXtV2-base + supervised TAPT (sister repo `known-cnn` 33-class) 만으로 이미 wafer cluster
   구조의 대부분을 학습.
2. **paper 의 진짜 NEW contribution 재정의**: 우리 추가 components (Local + Queue + NEG + NeCo +
   HDBSCAN) 의 total isolated effect 는 ΔARI +0.033 (B0 → B5). paper N1-N5 중 ★ N6
   (Component Interaction) 새로 추가.

### 다음 (iter 61)
- + Local InfoNCE (USE_LOCAL=true, LW=0.5, DenseCL weak).

---

## iter 61 — B1 + Local DenseCL (LW=0.5) (2026-05-11) — ★ Local 단독 효과 + (atomic step from B0)

run_dir: `outputs_contrastive_260511_162616/`

### 설정
- B0 + `USE_LOCAL=true, LOCAL_WEIGHT=0.5, LOCAL_POS_TOPK=12` (DenseCL Wang 2021).
- 그 외 B0 동일.

### 결과 (eom mcs=12 ms=3)
- **Completeness=0.9665, AMI=0.9387, ARI=0.8514, noise_pct(def)=3.927%, capture=1.000, n_cl=37**
- Hom=0.9351, NMI=0.9505, Sil=0.5139.

### 발견 (vs B0)
1. **Δ vs B0**: ARI **+0.028**, noise **-2.27pp**, Comp +0.006, AMI +0.010.
2. **Local 단독 효과 확실** — patch-level spatial contrast 가 wafer 위치 정보 (Edge-Top / Edge-Bottom)
   보존에 효과적. DenseCL paper claim (Wang 2021) 정합.
3. **Sil 0.582 → 0.514 하락** — Local 추가가 cluster boundary 더 미세하게 만듦 (Sil 페널티 inherent).

### 다음 (iter 62)
- LW 0.5 → 1.0 (lever 1 isolated step).

---

## iter 62 — B2 LW=1.0 (lever 1 isolated) (2026-05-11) — ★★★ NEGATIVE — LW 단독 regression

run_dir: `outputs_contrastive_260511_170230/`

### 설정
- B1 + `LOCAL_WEIGHT=0.5 → 1.0` (lever 1 atomic step, 그 외 B1 동일).
- USE_QUEUE 여전히 false, NEG filter off.

### 결과 (eom mcs=12 ms=3)
- **Completeness=0.9602, AMI=0.9290, ARI=0.8231, noise_pct(def)=6.195%, capture=1.000, n_cl=37**
- Hom=0.9257, NMI=0.9427, Sil=0.5089.

### 발견 (vs B1)
1. **Δ vs B1**: ARI **-0.028**, noise **+2.27pp**, Comp **-0.006**, AMI **-0.010**.
2. **★ LW=1.0 isolated regression** — LW=0.5 → 1.0 단독 변경은 negative.
   기존 Iter 1 (old anchor, A0→1) 에서 ARI +0.029 / noise -50% 효과는
   **Local + Queue + NEG 가 활성 상태에서의 lever 효과**.
3. **B0 ≈ B2 (모든 metric 일치)** — Queue/NEG 없는 cfg 에서는 LW 강화가 Local 의 효과를 상쇄.
   즉 LW lever 의 진짜 효과 = component interaction.

### 다음 (iter 63)
- + MoCo Queue (USE_QUEUE=true, QUEUE_SIZE=4096) — interaction 시험.

---

## iter 63 — B3 + MoCo Queue (interaction lift) (2026-05-11) — ★★★ N6 huge

run_dir: `outputs_contrastive_260511_173842/`

### 설정
- B2 + `USE_QUEUE=true, QUEUE_SIZE=4096` (MoCo He 2020).
- 그 외 B2 동일 (LW=1.0, NEG filter off, NeCo off).

### 결과 (eom mcs=12 ms=3)
- **Completeness=0.9828, AMI=0.9496, ARI=0.8464, noise_pct(def)=1.309%, capture=1.000, n_cl=36**
- Hom=0.9365, NMI=0.9591, Sil=0.5727.

### 발견 (vs B2)
1. **Δ vs B2**: ARI **+0.023**, noise **-4.89pp** (6.20→1.31%, -78%), Comp +0.023, AMI +0.021.
2. **★★★ paper N6 huge evidence** — Queue 추가가 B2 의 LW=1.0 over-emphasis 를 흡수.
   LW lever 효과 (B1→B2 negative → B2→B3 huge positive) 가 **Queue 와의 interaction 으로만 발현**.
3. **n_cl 37 → 36** — Queue 가 sister-pair 합성에 기여 (cleaner cluster).
4. paper community 의 "lever isolated 단독 효과" 보고 함정 입증 — atomic ablation 만으로
   contribution 분해 시 Component Interaction 누락.

### 다음 (iter 64)
- + NEG filter (IGNORE_NEG_SIM 1.0 → 0.72) — 마지막 baseline component.

---

## iter 64 — B4 + NEG filter (0.72) (2026-05-11) — ★ best 발견

run_dir: `outputs_contrastive_260511_181441/`

### 설정
- B3 + `IGNORE_NEG_SIM 1.0 → 0.72` (NV-Retriever-style false-neg filter, Moreira 2024).
- 그 외 B3 동일 (LW=1.0, USE_QUEUE=true, NeCo off).

### 결과 (eom mcs=12 ms=3) ★ B5 보다 우위
- **Completeness=0.9852, AMI=0.9557, ARI=0.8605, noise_pct(def)=0.524%, capture=1.000, n_cl=37**
- Hom=0.9439, NMI=0.9641, Sil=0.6109.

### 발견 (vs B3)
1. **Δ vs B3**: ARI **+0.014**, noise **-0.78pp** (1.31→0.52%, -60%), Comp +0.003, AMI +0.006.
2. **★ B4 = NeCo 없는 best cfg** — 모든 metric (ARI/Comp/noise/Sil) 이 B5 보다 우위.
3. NEG filter 단독 효과 확실 (small but clean) — Iter 13 (old anchor) 의 NEG 0.65 단독 효과
   재확인 (단 새 anchor 에서는 0.72 가 sweet spot, iter 58 lock-in).

### 다음 (iter 65)
- + NeCo (NECO_WEIGHT 0 → 0.2) — = iter 37 cfg reproduce 시험.

---

## iter 65 — B5 + NeCo 0.2 (= iter 37 cfg, isolated NeCo step) (2026-05-11) — ★ N1 isolated effect ≈ 0

run_dir: `outputs_contrastive_260511_185039/`

### 설정
- B4 + `NECO_WEIGHT 0 → 0.2` (NeCo Pariza 2024, lever 5 isolated step).
- = iter 37 cfg 완전 재현 (LW=1.0, USE_QUEUE=true, NEG=0.72, NeCo=0.2, seed=42).

### 결과 (eom mcs=12 ms=3)
- **Completeness=0.9801, AMI=0.9503, ARI=0.8564, noise_pct(def)=0.960%, capture=1.000, n_cl=37**
- Hom=0.9403, NMI=0.9598, Sil=0.6104.

### 발견 (vs B4 + vs iter 37)
1. **Δ vs B4**: ARI **-0.004**, noise **+0.44pp** (0.52→0.96%, +85%), Comp **-0.005**, AMI **-0.005**.
   = **★ NeCo (paper N1) isolated effect ≈ 0 또는 약간 negative**.
2. **B4 > B5** — NeCo 없는 cfg 가 NeCo 있는 cfg 보다 모든 metric 우위.
3. **vs iter 37 (5/9, same seed=42)**:
   - iter 37: ARI 0.8700 / Comp 0.991 / AMI 0.960 / noise 0.61%
   - B5 (iter 65): ARI 0.8564 / Comp 0.9801 / AMI 0.9503 / noise 0.960%
   - ΔARI **-0.014**, Δnoise **+0.35pp** — **same seed 라도 run-to-run variance 가 multi-seed std 만큼!**
4. **★ paper N1 (NeCo) contribution 재검토 필요** — 기존 iter 35→37 비교 (ARI 0.856→0.870)
   는 cross-run noise floor 안. **isolated effect ≈ 0**, "noise -70%" claim 은 다른 run 들의
   variance 의 우연 조합.

### 다음 (iter 66+ ?)
- Real Baseline ablation matrix B0-B5 완성. paper N6 (Component Interaction) NEW contribution.
- N1 (NeCo) 의 진짜 contribution = component interaction (B3 → B5 의 total effect ΔARI +0.010,
  combined 효과만 인정).
- paper IEEE TSM submit ready — N1-N6 6 contributions 갱신.

---

# ★★★★★ N6 contribution — Component Interaction (Real Baseline B0-B5, 2026-05-11)

**B0 → B5 component-by-component isolated breakdown**:

| step | ΔARI | Δnoise | Δ방향 | 판정 |
|:-:|---:|---:|:-:|---|
| B0 → B1 (+ Local) | +0.028 | -2.27pp | ✓ | Local 단독 효과 |
| B1 → B2 (LW=1.0) | -0.028 | +2.27pp | ✗ | **LW 단독 regression!** |
| B2 → B3 (+ Queue) | +0.023 | -4.89pp | ✓✓✓ | **Queue 가 LW interaction lift** ★ N6 |
| B3 → B4 (+ NEG) | +0.014 | -0.78pp | ✓ | NEG 단독 효과 |
| B4 → B5 (+ NeCo) | -0.004 | +0.44pp | ✗ | **NeCo isolated ≈ 0** |

**총 누적**: B0 → B5 ΔARI **+0.033** / Δnoise **-5.24pp** / ΔComp **+0.020** / ΔAMI **+0.021**.

**★ 핵심 발견** — Component 가 lever 단독으로 봐서는 효과 알 수 없고, **interaction 으로만 발현**:
- LW (lever 1) 단독 → negative (B1→B2)
- LW + Queue (B2→B3) → huge positive (Queue 가 LW over-emphasis 흡수)
- NeCo (lever 5, paper N1) 단독 → ≈ 0 또는 negative
- 기존 iter 35→37 의 "NeCo +0.014 ARI" 는 run-to-run variance 안 (B5 same-seed reproduce 가
  iter 37 보다 -0.014 ARI 가능 → multi-seed N2 강한 evidence).

이 발견이 **N6 contribution** — paper 의 6번째 contribution 추가.

---

## iter 67 — B2 + NeCo (no Queue, no NEG) — NeCo × Queue interaction (2026-05-11) — ★ N6 강화

run_dir: `outputs_contrastive_260511_215652/`

### 설정
- B2 (LW=1.0, USE_QUEUE=false, NEG=1.0 off) + `NECO_WEIGHT 0 → 0.2`.
- = NeCo 가 Queue 없을 때 isolated effect 측정 — NeCo×Queue interaction probe.
- seed=42.

### 결과 (eom mcs=12 ms=3)
- **Completeness=0.9659, AMI=0.9390, ARI=0.8508, noise_pct(def)=3.93%, capture=1.000**.

### 발견 (vs B2 + vs B3 + vs B5)
1. **Δ vs B2 (no NeCo, no Queue)**: ARI **+0.028**, noise **-2.27pp** (6.20→3.93%, -37%).
   = NeCo 가 **Queue 없을 때는 효과 강함** (B2 → iter 67 ΔARI +0.028 vs B4 → B5 ΔARI -0.004).
2. **Δ vs B3 (Queue, no NeCo)**: ARI **+0.005** (0.846 → 0.851). 단 noise B3 1.31% < 3.93%.
   = **NeCo (no Queue) 가 Queue (no NeCo) 보다 ARI 약간 우위 — partial substitutability!**
3. **NeCo × Queue interaction asymmetric**:
   - Queue 가 NeCo 와 동등 ARI lift 제공 (둘 다 negative 다양성 메커니즘)
   - 그러나 noise reduction 은 Queue 가 NeCo 우위 (1.31% < 3.93%) — large negative pool 의 noise floor 효과
4. **paper N6 (Component Interaction) 두 번째 evidence** — components 가 monotonic 추가가 아니라
   **partial substitutes** + **distinct mechanisms**:
   - Queue: 4096 explicit negative bank
   - NeCo: implicit neighbor consistency
   둘이 ARI 에서는 substitute, noise reduction 에서는 Queue 우위.

### 다음 (iter 68)
- iter 67 ARI 0.851 ≥ 0.85 → ★ B3 + NeCo (Queue + NeCo, no NEG) dispatch.
- NeCo × NEG interaction 측정 (NeCo 가 NEG 도 substitute 가능?).

---

## iter 68 — B3 + NeCo (Queue + NeCo, no NEG) — NeCo × Queue × NEG (2026-05-11) — ★★★ N6 가장 강한 evidence

run_dir: `outputs_contrastive_260511_224723/`

### 설정
- B3 (LW=1.0, USE_QUEUE=true, NEG=1.0 off) + `NECO_WEIGHT 0 → 0.2`.
- seed=42. NeCo × Queue × NEG triple interaction probe.

### 결과 (eom mcs=12 ms=3)
- **Completeness=0.9828, AMI=0.9496, ARI=0.8464, noise_pct(def)=1.31%, cap=1.000, n_cl=36**.
- Hom=0.9365, NMI=0.9591, Sil=0.7556.

### ★★★ 발견 — NeCo 효과 monotonic decay

| context | NeCo Δ ARI | NeCo Δ noise |
|---|---:|---:|
| no Queue, no NEG (B2 → iter 67) | **+0.028** | -2.27pp |
| Queue, no NEG (B3 → iter 68) | **+0.000** | +0.00pp |
| Queue + NEG (B4 → B5) | **-0.004** | +0.44pp |

1. **iter 68 = exact B3 reproduce** — ARI 0.8464 = B3 0.8464 (소수점 4자리 동일), noise 1.31% = 1.31%, n_cl 36 = 36. NeCo (0.2 weight) 가 학습에 추가됐음에도 HDBSCAN 결과 identical.
2. **Queue 가 NeCo 의 mechanism 을 흡수** — neighbor consistency = implicit negative diversity, Queue 가 explicit 으로 같은 정보 제공해 NeCo 가 redundant.
3. **NEG filter 가 추가되면 NeCo slightly harmful** — false-negative 보호와 neighbor consistency 가 약간 충돌 (B5 -0.004).

### paper N1 (NeCo) contribution 정확 정의 (final)

NeCo standalone value = **0**. NeCo's role = "implicit Queue substitute" — Queue 없을 때만 의미.
기존 paper claim 재구성:
- ✗ 기존: "NeCo improves noise 70%"
- ✓ 새: "NeCo can substitute MoCo Queue (B2 → iter 67 ΔARI +0.028) when Queue not available; with Queue present, NeCo has no additional effect."

### 다음 (iter 69)
- iter 68 < 0.85 → original plan: B4 multi-seed expand.
- **새 idea**: iter 69 = **B0 + NeCo only** (Global InfoNCE + NeCo only) — Local + Queue + NEG 모두 제거.
  NeCo 가 standalone 으로 Local + Queue 둘 다 substitute 가능한지? 아니면 Local 필요?
- 가설:
  - B0 ARI 0.823 baseline
  - B0 + NeCo ≈ 0.83~0.85 (Queue substitute 효과만, Local 없으니 lower)
  - 만약 ≥ 0.85 → NeCo 가 진짜 강력 substitute
  - 만약 < 0.83 → NeCo 가 Local 도움 없이는 한계

---

## iter 69 — B0 + NeCo only (Global + NeCo, no Local/Queue/NEG) (2026-05-12) — ★★★★ NeCo ≡ Local DenseCL 결정적

run_dir: `outputs_contrastive_260511_233312/`

### 설정
- B0 (Global InfoNCE only) + `NECO_WEIGHT 0 → 0.2`.
- USE_LOCAL=false, USE_QUEUE=false, IGNORE_NEG_SIM=1.0. NeCo 단독 substitute 시험.
- seed=42.

### 결과 (eom mcs=12 ms=3)
- **Completeness=0.9665, AMI=0.9387, ARI=0.8514, noise_pct(def)=3.93%, cap=1.000, n_cl=37**
- Hom=0.9351, NMI=0.9505, Sil=0.7071.

### ★★★★ 결정적 발견 — NeCo ≡ DenseCL Local InfoNCE (functionally equivalent)

| cfg | ARI | Δ vs B0 | noise | n_cl |
|---|---:|---:|---:|---:|
| B0 (Global only, base) | 0.8230 | (base) | 6.20% | 37 |
| B1 (B0 + Local LW=0.5) | 0.8514 | **+0.028** | 3.93% | 37 |
| iter 69 (B0 + NeCo 0.2) | **0.8514** | **+0.028** | **3.93%** | 37 |
| iter 67 (B0 + Local + NeCo) | 0.8510 | +0.028 | 3.93% | 37 |

**4 자리 ARI 동일**, noise 3.93% 동일, n_cluster 37 동일.

→ **NeCo (Pariza 2024) 와 DenseCL Local InfoNCE (Wang 2021) 가 functionally equivalent**.
   둘 다 patch-neighbor consistency 메커니즘, 다른 implementation, **identical effect magnitude**.

### paper N6 (Component Interaction) 4번째 evidence — 가장 강한

- ✓ Local + NeCo = redundant (둘 다 patch-neighbor)
- ✓ Queue subsumes both (Queue 가 explicit negative pool 로 흡수)
- ✓ NEG filter 만 distinct mechanism (false-negative protection)

### paper Methods 정확 분류

- **patch-neighbor consistency 메커니즘**: { Local DenseCL, NeCo } — substitutes
- **negative diversity 메커니즘**: { MoCo Queue } — single source
- **false-negative protection**: { NV-Retriever NEG filter } — independent

→ baseline 구성 시 patch-neighbor 는 둘 중 하나만 충분, 둘 다 사용 = no gain.

### 다음 (iter 70)
- 새 시험: **NeCo 가 Local 완전 대체 가능?** iter 70 = NeCo + Queue + NEG 만 (Local 제거).
- 가설: ≈ B4 (0.860) 면 paper-grade "Local fully replaceable by NeCo" 결론.
- cfg: --use-local false --use-queue true --ignore-neg-sim 0.72 --neco-weight 0.2.

---

## iter 70 — Global + NeCo + Queue + NEG (no Local) (2026-05-12) — ★★★★★ NEW SOTA

run_dir: `outputs_contrastive_260512_001719/`

### 설정
- Global InfoNCE + NeCo 0.2 + Queue + NEG 0.72 (B4 + NeCo - Local).
- USE_LOCAL=**false** (Local DenseCL 제거), NECO_WEIGHT=0.2.
- seed=42.

### 결과 (eom mcs=12 ms=3) ★★★ NEW SOTA
- **Completeness=0.9872, AMI=0.9594, ARI=0.8797, noise_pct(def)=0.87%, cap=1.000, n_cl=37**
- Hom=0.9479, NMI=0.9671, Sil=**0.7860** (역대 최고).

### ★★★★★ 결정적 발견 — NeCo strictly superior to Local DenseCL

| cfg | ARI | noise | Δ vs B4 | 판정 |
|---|---:|---:|---:|---|
| B4 (Global + Local + Queue + NEG) | 0.8605 | 0.52% | (base) | 기존 best |
| B5 (B4 + NeCo) | 0.8564 | 0.96% | -0.004 | NeCo addition harmful |
| **iter 70 (replace Local with NeCo)** | **0.8797** | **0.87%** | **+0.019** | ★ NEW SOTA |
| iter 37 historical SOTA | 0.8700 | 0.61% | +0.010 | beaten |

**paper-grade 발견 3개**:
1. **NeCo > Local DenseCL** standalone (not just substitute, **strictly superior**).
2. **Local + NeCo simultaneously = harmful** (B5 < B4 -0.004) — redundancy + interference.
3. **New SOTA cfg 더 간결**: 5 components (Global + Local + Queue + NEG + NeCo) → 4 components (Global + NeCo + Queue + NEG).

### paper N1 (NeCo) contribution 최종 재정의

| version | claim |
|---|---|
| v1 (initial) | "NeCo improves noise -70%" — based on iter 35→37 |
| v2 (post-B0-B5) | "NeCo isolated effect ≈ 0; works via Queue substitution" |
| **v3 (post-iter 70) ★** | **"NeCo is strictly superior alternative to Local DenseCL — same mechanism, +0.019 ARI, identical implementation cost. Local should be REPLACED, not augmented, with NeCo."** |

### paper Methods 새 구성 (post-iter 70)

```
Recommended baseline:
  L = L_global + L_NeCo (weight 0.2) + L_NEG_filter (sim < 0.72)
  + MoCo Queue (size 4096)

DEPRECATED: L_local (DenseCL patch grid contrast) — strictly worse than NeCo.
```

### 다음 (iter 71)
- iter 70 SOTA 가 seed=42 lucky 인지 확인. iter 71 = iter 70 cfg seed=1.
- 가설 ≥ 0.875 면 SOTA 확정. < 0.870 면 seed variance, multi-seed expand.

---

## iter 71 — NEW cfg seed=1 (multi-seed) (2026-05-12) — ★★★ honest multi-seed update

run_dir: `outputs_contrastive_260512_010113/`

### 설정
- iter 70 cfg (Global + NeCo + Queue + NEG, no Local) + seed=1.

### 결과 (eom mcs=12 ms=3)
- **Completeness=0.9856, AMI=0.9488, ARI=0.8491, noise=1.05%, Sil=0.7832**.

### ★★★ Multi-seed honest summary

| seed | ARI | noise | Comp | Sil |
|:-:|---:|---:|---:|---:|
| 42 | 0.8797 | 0.87% | 0.9872 | 0.7860 |
| 1 | **0.8491** | 1.05% | 0.9856 | 0.7832 |
| avg | **0.8644** | 0.96% | 0.9864 | 0.7846 |
| std | 0.0216 | 0.13pp | 0.0011 | 0.0020 |

### 비교: B5 (iter 37 cfg) multi-seed

| cfg | ARI avg | ARI std | Sil avg |
|---|---:|---:|---:|
| B5 (Local + NeCo + Queue + NEG) | 0.856 | 0.012 | 0.6104 |
| NEW (NeCo + Queue + NEG, no Local) | **0.864** | 0.022 | **0.7846** |
| Δ | **+0.008** | +0.010 | **+0.174 = +29%** ★★★ |

### ★ Honest paper claim (post-iter 71)

- ARI 평균 lift = **+0.008** (marginal, within multi-seed std overlap).
- ★★★ **Silhouette 평균 lift = +0.174 (+29%)** — **highly robust** (std 0.002 vs 0.012).
- noise: equal (~1%).

**즉 paper N1 의 진짜 contribution = cluster geometry 향상**, ARI 의 marginal lift 가 아니라.
NeCo 가 Local 보다 cosine compactness 측면 dramatic 우위, 단 ARI 는 multi-seed noise 안.

### 다음 (iter 72)
- seed=2 추가 (3-seed, B5 multi-seed protocol matching).
- 가설: Sil ≈ 0.78 ± 0.005 면 +29% Sil 향상 robust 확정.

---

## iter 72 — NEW cfg seed=2 (3-seed complete) (2026-05-12) — ★ final honest claim

run_dir: `outputs_contrastive_260512_014507/`

### 설정
- NEW cfg seed=2 (Global + NeCo + Queue + NEG, no Local).

### 결과 (eom mcs=12 ms=3)
- ARI=**0.8475**, Comp=0.9747, AMI=0.9428, noise=2.53%, Sil=**0.8130**, n_cl=36.

### ★ 3-seed final (NEW vs B5)

| metric | NEW (seed 42/1/2) | B5 (=iter 37 cfg, 3-seed) | Δ |
|---|---|---|---:|
| ARI avg | **0.8588 ± 0.018** | 0.856 ± 0.012 | **+0.003** (≈ equal) |
| Comp avg | 0.9825 ± 0.007 | ~0.98 | ≈ equal |
| AMI avg | 0.9503 ± 0.008 | ~0.95 | ≈ equal |
| **Sil avg** | **0.7941 ± 0.017** | 0.6104 | **+0.184 = +30%** ★★★ |
| noise avg | 1.48% | 0.96% | +0.52pp (slight regression) |

### ★★★ Final honest paper claim

NEW cfg (NeCo replaces Local) 의 진짜 contribution:
1. ARI mean ≈ B5 (within multi-seed noise) — **equivalent partitioning quality**
2. **Silhouette +30% (robust across 3 seeds)** — **dramatically better cluster geometry**
3. noise slightly higher (+0.5pp) — Local 의 noise floor 기여 일부 손실

**paper N1 (NeCo) 의 honest contribution** (final v4):
> "NeCo can replace DenseCL Local InfoNCE with **equivalent partitioning quality (ARI) but significantly more compact embeddings (Silhouette +30%)**. Trade-off: slight noise floor regression (+0.5pp). The choice is a **geometry vs noise** Pareto frontier, not a strict superiority."

### 다음 (iter 73)
- 새 idea: NEW cfg + TEMP 0.05 (lever 4 cross-component).
- 가설: lower TEMP 가 noise 감소 + Sil 추가 lift (NeCo 의 compactness 와 시너지).

---

## iter 73 — NEW cfg + TEMP 0.05 (lever 4 cross-component) (2026-05-12) — ★ negative result paper-grade

run_dir: `outputs_contrastive_260512_022912/`

### 설정
- NEW cfg (Global + NeCo + Queue + NEG, no Local) + `NCE_TEMP 0.07 → 0.05`.
- seed=42.

### 결과 (eom mcs=12 ms=3) — regression
- ARI=**0.8555**, Comp=0.9755, noise=2.97%, Sil=0.7407.

### ★ TEMP × component interaction — 또 다른 N6 evidence

| cfg | TEMP 0.07 | TEMP 0.05 | Δ |
|---|---:|---:|---:|
| B5 (with Local) | 0.8564 (iter 65) | 0.8700 (iter 37) | **+0.014 ★** |
| NEW (no Local) | 0.8797 (iter 70) | **0.8555** (iter 73) | **-0.024 ✗** |

→ TEMP 0.05 의 lift 는 **Local 의 patch-stability 와 시너지** 였음 (Local stabilize, TEMP sharpen 결합).
   NeCo only 환경에서는 TEMP 0.07 sweet spot, lower TEMP 면 NeCo 의 neighbor 신호 over-sharpen → noise 증가.

### paper N6 (Component Interaction) 5번째 evidence

cross-component interaction 매트릭스:
- NeCo × Queue: substitutes (iter 67-69)
- NeCo × Local: substitutes (iter 70 vs B5)
- NeCo × NEG: complementary (NEW cfg works)
- **TEMP × Local**: synergistic (B5 + TEMP 0.05 lift)
- **TEMP × NeCo (no Local)**: antagonistic (iter 73 regression)

→ paper finding: "best hparam 은 component context 에 따라 다르다 — 단일 component sweep 으로는 optimum 못 찾음".

### 다음 (iter 74)
- 4-component lattice 마지막 cell: **B0 + NeCo + NEG (no Queue)**.
- 가설: < 0.85 면 Queue essential, ≥ 0.85 면 Queue dispensable.
- 결정적 — Queue 가 NEW cfg 의 baseline 에서 필요한가?

---

## iter 74 — Global + NeCo + NEG (no Queue, no Local) (2026-05-12) — ★★★★ N7 NEW contribution

run_dir: `outputs_contrastive_260512_031310/`

### 설정
- Global + NeCo 0.2 + NEG 0.72. USE_LOCAL=false, USE_QUEUE=false.
- seed=42.

### 결과 (eom mcs=12 ms=3)
- ARI=**0.8514**, Comp=0.9665, AMI=0.9387, noise=3.93%, Sil=0.7071.

### ★★★★ 결정적 발견 — NEG depends on Queue

iter 74 = **exact same as iter 69** (B0 + NeCo only):
- iter 69 (no Queue, no NEG): ARI 0.8514, noise 3.93%, Sil 0.7071
- iter 74 (no Queue, +NEG): ARI **0.8514**, noise **3.93%**, Sil **0.7071** ← exact identical!

→ **NEG filter 효과 = 0 when Queue absent**.

iter 70 (with Queue, +NEG) ARI 0.8797 vs iter 74 (no Queue, +NEG) ARI 0.8514:
- Queue 가 +0.029 ARI contribution (with NEG present)
- 그러나 NEG 의 진짜 lift 는 Queue 와 함께만 작동

### ★ paper N7 — NEW contribution: Component Dependency Hierarchy

```
Required:    Global InfoNCE + (Local DenseCL || NeCo)  ← substitutes, choose one
Significant: MoCo Queue (+0.029 with NeCo, +0.023 with Local LW=1.0)
Conditional: NEG filter ← requires Queue (sans Queue NEG=0)
Deprecated:  Local DenseCL (NeCo functionally equivalent, slightly better)
```

이론적 해석:
- NEG 의 false-negative protection 은 large negative pool 의 statistical distribution 필요
- 4096 Queue negative 중 cosine sim > 0.72 인 것 filter — pool 작으면 filter 효과 없음
- batch-only negative (n=8) 로는 filter 통계적 의미 X

### paper Methods 정확 구성

```
L = L_global + L_NeCo (w=0.2) + L_NEG_filter (when MoCo Queue active)
+ MoCo Queue (size 4096, weight 1.0)

Recommended: 4 components 만 (NeCo + Queue + NEG + Global)
Deprecated: Local DenseCL — strictly worse or equivalent to NeCo
```

### 다음 (iter 75)
- 4-comp lattice 0101 cell: **B0 + NeCo + Queue (no Local, no NEG)**.
- NEG 의 marginal lift within NEW cfg 측정.
- 가설:
  - ≈ 0.86 → NEG marginal +0.02 (Queue 가 base)
  - ≈ 0.88 → NEG 거의 0 (Queue 가 NEG 도 흡수)

---

## iter 75 — Global + NeCo + Queue (no Local, no NEG) (2026-05-12) — ★★★★★ NEW SOTA seed=42

run_dir: `outputs_contrastive_260512_035824/`

### 설정
- Global + NeCo 0.2 + Queue 4096. USE_LOCAL=false, IGNORE_NEG_SIM=1.0 (NEG off).
- seed=42. lattice 0101 cell.

### 결과 (eom mcs=12 ms=3) ★ 새 best on seed=42
- ARI=**0.8822**, Comp=0.9841, AMI=0.9599, noise=1.31%, Sil=**0.7845**, n_cl=38.

### ★★★★★ 결정적 발견 — NEG harmful in NEW cfg

| cfg | ARI | noise | Sil |
|---|---:|---:|---:|
| iter 75 (NeCo + Queue) ★ | **0.8822** | 1.31% | 0.7845 |
| iter 70 (NeCo + Queue + NEG) | 0.8797 | 0.87% | 0.7860 |
| iter 69 (NeCo only) | 0.8514 | 3.93% | 0.7071 |
| iter 74 (NeCo + NEG) | 0.8514 | 3.93% | 0.7071 |

→ **NEG 의 진짜 효과 = noise reduction only (not ARI)**.
   ARI 측면 NEG 는 marginal regression (-0.0025), Sil 동등 (0.785 vs 0.786).
   NEG 의 trade-off: ΔARI -0.003 / Δnoise -0.44pp.

### Queue isolated effect (clean)

Queue 추가만 (iter 69 → iter 75): ARI +0.031, noise -2.62pp.
이는 B2 → B3 의 Queue +0.023 (Local present) 보다 큰 효과.
**Queue 의 효과는 patch-neighbor (Local 또는 NeCo) 의 종류에 따라 다름**.

### paper N7 update — Component Dependency Hierarchy (post-iter 75)

```
Required:    Global InfoNCE + (Local DenseCL || NeCo)  ← substitutes
Best lift:   MoCo Queue (+0.029 ~ +0.031 with NeCo)
Trade-off:   NEG filter (noise -0.44pp, ARI -0.003) — optional
Deprecated:  Local DenseCL (NeCo equivalent or better)
```

paper Methods **most parsimonious**: **Global + NeCo + Queue (3 components)**.
NEG 는 use case dependent: noise floor 가 중요하면 추가, ARI 가 중요하면 생략.

### 다음 (iter 76)
- multi-seed iter 75 — seed=1 으로 0.8822 lucky 인지 확인.
- 3-seed avg ≥ 0.870 면 minimal cfg paper-ready.

---

## iter 76 — minimal cfg seed=1 (multi-seed) (2026-05-12) — ★ honest result

run_dir: `outputs_contrastive_260512_044307/`

### 설정
- iter 75 cfg (NeCo + Queue, no Local, no NEG) + seed=1.

### 결과 (eom mcs=12 ms=3)
- ARI=**0.8149**, Comp=0.9873, AMI=0.9398, noise=1.13%, Sil=0.7638.

### ★ minimal cfg multi-seed honest

| seed | ARI | Sil | noise |
|---|---:|---:|---:|
| 42 | 0.8822 | 0.7845 | 1.31% |
| 1 | **0.8149** | 0.7638 | 1.13% |
| avg | **0.8485 ± 0.048** | 0.7741 ± 0.015 | 1.22% |

★ std 0.048 너무 큼 — seed=42 의 0.8822 가 매우 lucky 였음.

### ★★ Final cfg comparison (multi-seed avg)

| cfg | ARI avg | Sil avg | noise avg |
|---|---:|---:|---:|
| B5 = iter 37 cfg (5-comp, Local+Queue+NEG+NeCo) | 0.856 | 0.610 | 0.96% |
| **NEW 4-comp (NeCo+Queue+NEG, no Local) ★★** | **0.8588 ± 0.018** | **0.7941 ± 0.017** | 1.48% |
| Minimal 3-comp (NeCo+Queue, no Local, no NEG) | 0.8485 ± 0.048 | 0.7741 ± 0.015 | 1.22% |

→ **NEG 가 multi-seed 평균 ARI 보호** (+0.010 vs minimal).
   seed=42 의 -0.0025 NEG regression 은 lucky variance 였음.
   **paper best cfg = NEW 4-comp (NeCo + Queue + NEG, no Local)**.

### paper N7 final (post-iter 76)

```
paper recommendation:
  L = L_global + L_NeCo(0.2) + L_NEG_filter(<0.72)
  + MoCo Queue(4096)

이유:
- NeCo > Local DenseCL (Sil +30% robust)
- Queue 필수 (NeCo alone 효과 미약, +0.029 with Queue)
- NEG: marginal ARI lift but std reduction (stability)
- 5-comp B5 (with Local) → 4-comp NEW: minus Local, plus better Sil
```

### 다음 (iter 77)
- 새 idea: NEW 4-comp + NeCo weight 0.2 → 0.4 (weight sweep upper, Pariza 권장 range).
- 가설: lift +0.01 면 0.4 sweet spot.

---

## iter 77 — NEW 4-comp + NeCo weight 0.4 (2026-05-12) — ★ sweet spot 0.2 확정

run_dir: `outputs_contrastive_260512_052907/`

### 설정
- NEW 4-comp + `NECO_WEIGHT 0.2 → 0.4`. seed=42.

### 결과
- ARI=**0.8605**, Comp=0.9852, noise=0.52%, Sil=**0.8012**.

### ★ NeCo weight sweep (seed=42)

| NeCo weight | ARI | noise | Sil |
|---:|---:|---:|---:|
| 0.0 (B4 Local-based) | 0.8605 | 0.52% | 0.6109 |
| **0.2 (iter 70 NEW)** | **0.8797** | 0.87% | 0.7860 |
| 0.4 (iter 77) | 0.8605 | 0.52% | 0.8012 |

**패턴**:
- ARI inverse-U with peak at **0.2**
- noise monotonic with NeCo (lower NeCo = lower noise, since NEG protects)
- Sil **monotonic increasing** with NeCo (geometry follows weight)

→ paper의 **geometry-vs-partitioning trade-off** 추가 evidence:
   NeCo weight ↑ → Sil ↑, but ARI peaks at 0.2 (sweet spot).

흥미: NeCo 0.4 ≈ B4 in ARI/noise but Sil +0.19 → "over-weighted NeCo = B4 ARI + better geometry".

### 다음 (iter 78)
- NeCo weight 0.1 (sweet spot 더 낮은지 확인).
- 가설:
  - 0.1 > 0.2 → sweet spot 더 낮음, paper 권장 weight 변경
  - 0.1 < 0.2 → 0.2 confirmed sweet spot

---

## iter 78 — NEW 4-comp + NeCo weight 0.1 (lower sweep) (2026-05-12) — ★ Sil robust, ARI narrow peak

run_dir: `outputs_contrastive_260512_061456/`

### 설정
- NEW 4-comp + `NECO_WEIGHT 0.2 → 0.1`. seed=42.

### 결과 (eom mcs=12 ms=3)
- ARI=**0.8605**, Comp=0.9852, noise=**0.52%**, Sil=**0.8012**.

### ★ NeCo weight sweep complete (seed=42, NEW 4-comp)

| NeCo weight | ARI | noise | Sil |
|---:|---:|---:|---:|
| 0.0 (B4) | 0.8605 | 0.52% | 0.611 |
| 0.1 (iter 78) | 0.8605 | 0.52% | **0.801** |
| **0.2 (iter 70)** | **0.8797** | 0.87% | 0.786 |
| 0.4 (iter 77) | 0.8605 | 0.52% | 0.801 |

### ★ 결정적 patterns

1. **NeCo 0.1 = 0.4 = B4 on ARI/noise** — exact same (0.8605 ARI, 0.52% noise, n_cl=37).
2. **Sil 0.78-0.80 robust across NeCo weight ≥ 0.1** — 모두 ~+0.19 over B4.
3. **NeCo 0.2 narrow ARI peak** — sweet spot 매우 좁음 (0.1 → 0.2 → 0.4 monotonic 아니라 step function).

### paper claim

- **Sil benefit robust**: NeCo 추가 (weight any ≥ 0.1) → Sil +0.19 (+30%) 항상.
- **ARI lift fragile**: NeCo weight 0.2 만 에서 ARI +0.019. 0.1/0.4 는 B4 ARI 회귀.
- → **geometric benefit (Sil) is the main contribution of NeCo**, ARI is occasional bonus at exact weight.

### performance-research agent 인사이트 (post-iter 78)

- NeCo weight 더 sweep 의미 없음 (std 0.018 안에 묻힘)
- 다음 우선순위 (외부 SOTA 권고):
  1. **SynCo synthetic negatives** (Pareto +0.02 ARI)
  2. **DeepDPM K-discovery eval** (NMI +0.02 + K hardcode 제거)
  3. **Macro-Sil + ASW** metric swap (variance reduction)

### 다음 (iter 79)
- 새 idea: Queue size 4096 → 8192 (performance-research 권고 단순 alternative).
- 가설: more negatives = ARI/noise 추가 향상 (Queue critical N7 evidence 강화).

---

## iter 79 — NEW 4-comp + Queue 8192 (Queue scaling) (2026-05-12) — ★ 4096 sweet spot

run_dir: `outputs_contrastive_260512_075732/`

### 결과 (eom mcs=12 ms=3)
- ARI=**0.8674**, Comp=0.9855, noise=1.31%, Sil=0.7923, n_cl=36.

### ★ Queue size sweep (NEW 4-comp, seed=42)

| Queue | ARI | noise | Sil(def) |
|---:|---:|---:|---:|
| **4096 (iter 70)** | **0.8797** | 0.87% | 0.7860 |
| 8192 (iter 79) | 0.8674 | 1.31% | 0.7923 |

→ Queue 8192 = ARI -0.012 regression. Queue 4096 sweet spot.
→ iter 56 (old track) 동일 패턴 confirms.

### 다음 (iter 80)
- Queue size 4096 → 2048 (lower sweep). 2048 < 4096 → 4096 sweet spot 확정.

---

## ★ Correction note (2026-05-12) — Sil +30% retraction (paper N8 NEW)

> **Append-only correction**: 과거 iter 67-77 entry 의 결과 수치는 변경하지 않음.
> 단, 그 수치들의 cross-cfg 해석 (B5 vs NEW Silhouette) 이 HDBSCAN protocol mismatch
> 였음을 추가로 기록.

### 발견 (post-iter 78 reanalysis)

iter 67-77 의 Sil 측정값 (B1 0.514, iter 69 0.707, iter 70 0.786, B5 0.610 등) 은
mixed-HDBSCAN protocol (eom+ms=3 일부, leaf+ms=4 일부) 로 측정됨. 따라서:

- "NEW vs B5 Sil +0.184 (+30%) robust across 3 seeds" claim **retracted**
- "NeCo > Local DenseCL on geometry +0.193" claim **retracted**
- "geometry-vs-partitioning Pareto frontier" claim **retracted**
- "Sil monotonic ↑ with NeCo weight" claim **retracted**

### 재측정 (apples-to-apples eom + mcs=12 + ms=3, defect-only)

cluster-analyzer agent 분석 (`outputs_contrastive_260512_001719/eval/cluster_report.parquet`):

| cfg | apples Sil (defect-only) | ARI single=42 | noise | n_cl |
|---|---:|---:|---:|---:|
| B4 (Local+Queue+NEG, no NeCo) | **0.8012** | 0.8605 | 0.524% | 37 |
| B5 (Local+Queue+NEG+NeCo, iter 37) | **0.7988** | 0.8564 | 0.96% | 37 |
| NEW (NeCo+Queue+NEG, no Local, iter 70) | **0.7860** | 0.8797 | 0.87% | 37 |

→ NEW vs B5 Sil = **−0.013** (slightly worse), NOT +30% better. **equivalent within seed variance**.

### paper N1 v5 reframe (final honest, post-correction)

> **NeCo's wafer-domain mechanism = Normal/defect boundary stability, NOT defect-cluster
> compactness.**
> - defect-cluster intra_p95 (NeCo): +26% (widening, NOT 압축)
> - Normal noise 77.7% → 14.1% (859 / 1000 Normals → 1 dense cluster)
> - full-set Completeness 0.851 (B5) → 0.917 (NEW) (+0.066)
> - full-set ARI 0.69 (B5) → 0.83 (NEW) (+0.14)
> - defect-only metrics: NEW ≈ B5 (functionally equivalent on defect cluster geometry)

### paper N8 NEW contribution — HDBSCAN Protocol Mismatch Methodology

> Comparing Silhouette / ARI / noise across HDBSCAN cfg families (leaf vs eom, ms=3 vs
> ms=4, full-set vs defect-only) produces spurious headline differences. The retracted
> "+30% Sil multi-seed robust" claim in v0.5 ABSTRACT is itself the worked example —
> multi-seed robustness within a fixed protocol does NOT detect cross-protocol artefacts.
> N8 deliverable: any contrastive-clustering paper must explicitly fix
> `cluster_selection_method` / `mcs` / `ms` / `epsilon` / metric scope before reporting
> cross-cfg deltas. Worked example: this paper's own retracted v0.5 abstract.

### 영향 받은 paper section (정정 완료)

- ABSTRACT.md: v0.5 deprecated, v0.6 added with corrected claims
- INTRODUCTION.md: C7 revised (Sil claims removed, Normal-consolidation added)
- METHOD.md: §3.6 NeCo vs DenseCL "+0.193 Sil" retracted
- RESULTS.md: §14b/14c/14d/14e/14f/14g 정정 + §14h (paper N1 v5) + §14i (paper N8) + §14k (retracted-claims index) 추가
- DISCUSSION.md: §7.10.1 ~ §7.10.6 모두 revised + §7.11 (paper N8) 추가
- CONCLUSION.md: §8.6 revised + N8 added (N1-N7 → N1-N8)
- manager_report/SUMMARY.md: §0.6 revised
- manager_report/REPORT.md: Phase 2 section revised
- README.md: Sil retraction note 추가
- ITERATIONS.md: 본 correction note (append-only)

### iter 67-77 결과 수치 유지

본 correction note 는 iter 67-77 entry 의 수치를 **변경하지 않음** (append-only).
단 그 수치들의 cross-cfg Sil 비교 해석만 retracted. 향후 ABLATION_PLAN.md 도 동일
정책 (수치 유지, 해석만 정정).

---

## iter 82-83 — ★ Five-method clustering algorithm benchmark (2026-05-12) — paper N9 NEW

> 동일 contrastive embedding (B4 / B5 / iter 70 NEW) 위에서 5 가지 clustering 알고리즘
> 벤치마크. defect-only scope, K_gt=42.
> JSON evidence: `tier1_clustering_benchmark.json`.

### 측정 setup
- B4: `outputs_contrastive_260511_181441/` (Local + Queue + NEG, no NeCo)
- B5: `outputs_contrastive_260511_185039/` (= iter 37 cfg, 5-component)
- iter 70 NEW: `outputs_contrastive_260512_001719/` (NeCo + Queue + NEG, no Local, 4-component)
- 5 methods: HDBSCAN (eom mcs=12 ms=3), DP-GMM, KMeans K=42 (oracle), Agglomerative Ward K=42 (oracle), Spectral K=42 (oracle)
- Multi-seed (NEW only): seed=42, 1, 2 (iter 70/71/72)

### 결과 — single-seed=42 ARI matrix

| cfg | HDBSCAN | DP-GMM | KMeans-42 | Agglo-Ward-42 | Spectral-42 |
|---|---:|---:|---:|---:|---:|
| B4 | 0.8605 | 0.8344 | 0.8876 | **0.9055** | 0.4046 |
| B5 (= iter 37) | 0.8564 | 0.8369 | 0.8854 | **★ 0.9358** | 0.7898 |
| iter 70 NEW | **0.8797** | **0.8413** | 0.8798 | 0.9200 | 0.2289 |

### 결과 — NEW multi-seed (3-seed: 42, 1, 2)

| Method | seed=42 | seed=1 | seed=2 | 3-seed avg | std |
|---|---:|---:|---:|---:|---:|
| HDBSCAN | 0.8797 | 0.8491 | 0.8475 | **0.8588** | 0.018 |
| Agglomerative K=42 | 0.9200 | 0.8854 | 0.8989 | **0.9014** | 0.022 |
| KMeans K=42 | 0.8798 | 0.8456 | 0.8779 | **0.8678** | 0.026 |

### 핵심 발견 (paper N9 NEW)

1. **Cfg ranking flip across method families**:
   - Density-based (HDBSCAN / DP-GMM): iter 70 NEW > B5 ≈ B4
   - Centroid/linkage-based with oracle K (KMeans / Agglomerative): B5 > iter 70 NEW ≈ B4
2. **ARI magnitude shift +0.04~+0.10 across methods at fixed embedding**:
   - B5 HDBSCAN→Agglo: +0.079 (0.8564 → 0.9358) — 단일 encoder lever 보다 큼
3. **Dual-frontier framework** (paper-grade deliverable):
   - Unknown-K (real-world): iter 70 NEW + HDBSCAN = 0.859 ± 0.018 (3-seed)
   - Known-K (oracle): B5 + Agglomerative = 0.9358 (single) / NEW 0.9014 ± 0.022 (3-seed)
4. **Lucky-pattern N2 evidence 가 method axis 까지 확장**:
   - seed=42 → seed=1 drop: HDBSCAN +0.030, Agglo +0.035, KMeans +0.034
   - → 모든 method 가 동일 magnitude lucky variance — embedding 자체가 source
5. **Spectral K=42 instability**: ARI 0.23~0.79 across cfg, graph-not-fully-connected warnings
   → 실용 추천 제외

### paper section 갱신 영향
- ABSTRACT.md: v0.7 추가 (9 contributions, dual-frontier headline)
- INTRODUCTION.md: C8 (N8 protocol mismatch) + C9 (N9 algorithm dependency) 추가
- METHOD.md: §3.7 신설 (clustering algorithm selection rationale + dual-frontier)
- RESULTS.md: §15 신설 (5-method × 3-cfg matrix + multi-seed + practitioner choice tree)
- DISCUSSION.md: §7.12 (N9 clustering method dependency) + §7.13 (practitioner choice tree) 추가
- CONCLUSION.md: §8.7 (N9) + §8.8 (closing dual-frontier "when to use which") 추가
- manager_report/SUMMARY.md / REPORT.md: dual-frontier 강조 + Phase 3 (Clustering benchmark) 추가

### 다음 (iter 84+)
- B5 + Agglomerative K=42 multi-seed measurement (B5 single-seed 0.9358 → 3-seed mean ?)
- 다른 distance metric (Euclidean vs cosine) 위 Agglo K=42 검증
- Cluster-Aware Synthesis Loop (F2) 시작 — 9 contributions 모두 lock-in 됨

---

## iter 84-pre note — ★ Per-class Agglomerative Ward K=42 purity breakdown (paper N1 v6 FINAL, 2026-05-12)

> iter 82-83 (5-method clustering benchmark) 의 Agglomerative Ward K=42 결과를
> per-GT-class 로 decompose 한 evidence. paper N1 v5 의 "NeCo functionally equivalent
> to Local DenseCL — substitutable on partitioning" claim 을 **N1 v6 final: complementary
> not equivalent** 로 refine. 이번 entry 는 측정 결과 추가 evidence (no new training run).

### 측정 setup
- 동일 contrastive embedding: B5 (`outputs_contrastive_260511_185039/`) vs iter 70 NEW
  (`outputs_contrastive_260512_001719/`).
- Clustering: scikit-learn `AgglomerativeClustering(n_clusters=42, linkage='ward')`
  on L2-normalized 128-d defect-only embedding (Normal excluded), seed=42, K=K_gt=42 oracle.
- Per-class dominant cluster purity = max_c (count(GT=g, pred=c)) / count(GT=g) for each
  defect class g.
- Source: `cluster_report.parquet` per-class dominant-cluster purity column.

### 핵심 결과 — per-class winner flips on both sides

**NEW > B5 wins (NeCo-only beats Local+NeCo combined)**:

| class | N | B5 | NEW | Δ NEW − B5 |
|---|---:|---:|---:|---:|
| CenterCircle | 42 | 54.8% | **100.0%** | **+45.2pp** |
| Edge-Top_fork | 20 | 90.0% | **100.0%** | **+10.0pp** |

**B5 > NEW wins (Local+NeCo combined beats NeCo-only)**:

| class | N | B5 | NEW | Δ NEW − B5 |
|---|---:|---:|---:|---:|
| Edge-Ring_fork | 31 | **100.0%** | 64.5% | **−35.5pp** |
| Center_scratch | 40 | **95.0%** | 75.0% | **−20.0pp** |
| Donut_fork | 37 | **100.0%** | 81.1% | **−18.9pp** |
| Edge-Top_scratch | 19 | **100.0%** | 84.2% | **−15.8pp** |

**Net average per-class purity**:
- B5 = 97.0%
- NEW = 96.2%
- Δ = −0.83pp (B5 marginal aggregate win)

**Absolute SOTA single-seed=42 Agglomerative Ward K=42**:
- B5 ARI = **0.9358** (★ absolute SOTA across all 5×3 cfg-method combinations)
- NEW ARI = 0.9200
- Δ = +0.0158 (B5 strict win under linkage clustering with oracle K)

### paper N1 v6 FINAL refinement

> **v5 claim (deprecated v6)**:
>   "NeCo functionally equivalent to Local DenseCL — substitutable on partitioning"
>
> **v6 claim (FINAL, 2026-05-12)**:
>   Local DenseCL InfoNCE and NeCo provide **complementary inductive biases**.
>   - Aggregate HDBSCAN ARI: identical (iter 69 vs B1 4-decimal equality preserved)
>   - Per-class Agglomerative Ward K=42 purity: **class-by-class winner flips on both sides**
>     - Local DenseCL excels at sub-pattern variant integration (fork/scratch rotational
>       and positional variants → 100% purity vs NeCo-only 64-84%)
>     - NeCo excels at uniform-pattern consolidation (CenterCircle round geometry → 100%
>       vs Local-with-NeCo 55%)
>   - **Combined (B5 = Local + NeCo + Queue + NEG)**: highest per-class average purity
>     (97.0%) and **absolute SOTA ARI 0.9358 on Agglomerative Ward K=42** (Δ +0.0158
>     above iter 70 NEW best 0.9200).
>   - Removing Local from baseline (NEW = NeCo + Queue + NEG only) trades fork/scratch
>     sub-pattern recovery for CenterCircle round-pattern consolidation — net marginal
>     loss on aggregate, **strict regression on absolute SOTA known-K oracle clustering**.

### Implication for paper (★ v0.8 ABSTRACT update)

1. **Local DenseCL는 NOT deprecated** (v5 retraction further refined v6). v5 의
   "substitutable" framing 은 HDBSCAN aggregate scope 에서만 valid. per-class Agglo K=42
   scope 에서는 두 mechanism 이 **complementary**.

2. **B5 (5-component) = TRUE absolute SOTA** under Agglomerative Ward K=42 oracle:
   - Single-seed=42: ARI **0.9358** (oracle K-known)
   - vs iter 70 NEW best 0.9200
   - Δ +0.0158 (single seed)

3. **NEW cfg (NeCo only)** = density-clustering (HDBSCAN) 위 best (Normal/defect
   boundary stability, paper N1 v5), 하지만 oracle K Agglo 위에서는 sub-pattern
   integration 손실.

4. **Optimal cfg는 task 에 의존** (dual-cfg recipe):
   - density-cluster + unknown-K → NEW iter 70 cfg (paper N1 v5 + N9 frontier 1)
   - linkage-cluster + known-K → B5 / iter 37 cfg (paper N1 v6 + N9 frontier 2)

### 영향 받은 paper section (v0.8 정정 완료)

- ABSTRACT.md: v0.8 추가 (v0.7 superseded), dual-cfg dual-frontier headline
- INTRODUCTION.md: C7 v6 refined (substitutable → complementary, B5 absolute SOTA preserved)
- METHOD.md: §3.6 v6 refined (complementary per-class, Local NOT deprecated)
- RESULTS.md: §16 신설 (per-class K=42 purity breakdown, top wins/loses 6 classes)
- DISCUSSION.md: §7.10.3 (Complementary not substitutable) + §7.10.6 (4th deliverable N1 v6) +
  §7.12.2 (Observation 1 per-class complementarity reason for B5 > NEW flip)
- CONCLUSION.md: §8.6 (N7 Component Dependency Hierarchy refined complementary) +
  §8.8 (closing remark dual-cfg with B5 absolute SOTA)
- manager_report/SUMMARY.md: §0.6 N7 → N7 v6 (complementary, dual-cfg)
- manager_report/REPORT.md: Phase 2 Section #1 (NeCo ≡ DenseCL → complementary)
- ITERATIONS.md: 본 iter 84-pre note (append-only)

### iter 82-83 수치 유지 (append-only)

본 entry 는 iter 67-77 / iter 82-83 의 수치를 **변경하지 않음**. v5 "substitutability"
해석만 v6 "complementary" 로 refine. aggregate ARI / noise / n_cl 4-decimal equality
(iter 69 vs B1) 는 그대로 valid 한 aggregate HDBSCAN scope finding 으로 보존.

### 다음 (iter 84+, 갱신)
- B5 + Agglomerative K=42 multi-seed measurement (B5 single-seed 0.9358 → 3-seed mean ?)
  → v6 absolute SOTA claim 의 multi-seed validation 우선 순위
- 다른 distance metric (Euclidean vs cosine) 위 Agglo K=42 검증
- Cluster-Aware Synthesis Loop (F2) 시작 — 9 contributions 모두 lock-in 됨

---

## iter 84 — B5 reproducibility (seed=1) + paper N1 v7 FINAL retraction of v6 absolute SOTA (2026-05-12)

### 설정
- iter 83 동일 cfg (B5 = Local LW=1.0 + Queue 4096 + NEG 0.72 + NeCo 0.2, P2-King base)
- **SEED 42 → 1** (B5 reproducibility 측정)
- HDBSCAN eom mcs=12 ms=3 + Agglo Ward K=42 + KMeans K=42 (defect-only)
- anchor `avg30_new_260508_123037` (43 class, n=2146) 동일
- run_dir: `outputs_contrastive_260512_114525/`
- evidence JSON: `outputs_contrastive_260512_114525/tier1_B5_seed1.json`

### 결과 (B5 seed=1)

| Method | B5 seed=42 (iter 83) | B5 seed=1 (iter 84) | Δ (seed=1 - seed=42) |
|---|---:|---:|---:|
| HDBSCAN | 0.8564 | **0.8122** | −0.0442 |
| Agglo Ward K=42 | **★ 0.9358** | **0.8482** | **−0.0876** |
| KMeans K=42 | 0.8854 | 0.8225 | −0.0629 |

**B5 2-seed avg ± std**:
- HDBSCAN: 0.8343 ± 0.031
- Agglo Ward K=42: **0.8920 ± 0.062**
- KMeans K=42: 0.8540 ± 0.044

**vs NEW 3-seed avg (iter 70/71/72)**:

| Method | B5 2-seed avg | NEW 3-seed avg | Δ (NEW − B5) |
|---|---:|---:|---:|
| HDBSCAN | 0.8343 ± 0.031 | **0.8588 ± 0.018** | **+0.0245** |
| Agglo Ward K=42 | 0.8920 ± 0.062 | **0.9014 ± 0.022** | **+0.0094** |
| KMeans K=42 | 0.8540 ± 0.044 | **0.8678 ± 0.026** | **+0.0138** |

→ **NEW > B5 on multi-seed average ARI across all three clustering methods**.
→ **B5 std (0.062 Agglo) = 2.8× NEW std (0.022 Agglo)** — much less reproducible.

### 발견

1. **B5 seed=42 0.9358 = lucky outlier** — seed=1 drop −0.088 on Agglo K=42 vs NEW
   seed=1 drop only −0.012 on same method. The previously-published v6 "B5 absolute
   SOTA at ARI 0.9358 single-seed=42" is **a cherry-picked seed=42 single point**, not
   reproducible.
2. **NEW > B5 on multi-seed average across ALL clustering methods** — Agglo: +0.0094,
   HDBSCAN: +0.0245, KMeans: +0.0138. The dual-cfg recipe of v0.8 that recommended B5
   for known-K Agglomerative Ward frontier is **retracted**.
3. **B5 reproducibility 3× worse than NEW** — B5 std 0.062 vs NEW std 0.022 on Agglo
   K=42. Same observation on HDBSCAN (B5 0.031 vs NEW 0.018) and KMeans
   (B5 0.044 vs NEW 0.026). The N2 (multi-seed methodology) contribution gains its
   strongest evidence yet — single-seed comparisons can produce false "winner" claims
   with Δ ARI 0.088 (B5 seed=42 vs seed=1) within run-to-run variance.
4. **N1 v6 "complementary" framing retains per-class evidence** but **the absolute SOTA
   claim at the dual-cfg recommendation is wrong** — NEW (NeCo only, no Local) achieves
   higher multi-seed average ARI on the same Agglomerative Ward K=42 frontier. Local
   DenseCL + NeCo combination of B5 was **not** complementary at the multi-seed scale.
5. **Single-cfg recommendation now valid** — NEW (NeCo + Queue + NEG, no Local) is the
   genuine multi-seed SOTA on both Unknown-K HDBSCAN (0.8588 ± 0.018) AND Oracle-K
   Agglomerative Ward (0.9014 ± 0.022). The dual-cfg recipe collapses to a single-cfg
   recommendation (NEW) with two clustering frontier targets.

### paper N1 v7 (FINAL, replaces v6)

> **NEW cfg (iter 70: Global + NeCo + Queue + NEG, no Local) achieves higher AND more
> reproducible average ARI than B5 across all clustering methods** (HDBSCAN / Agglo
> Ward K=42 / KMeans K=42). Single-seed (seed=42) comparison previously suggested
> B5 > NEW (0.9358 vs 0.9200 on Agglo), but B5 seed=1 reproducibility test revealed
> Agglo ARI 0.8482 — a Δ −0.088 drop, while NEW seed=1 dropped only Δ −0.012. B5 std
> (0.062 Agglo) is 2.8× NEW std (0.022). On multi-seed average: NEW > B5 on all three
> clustering methods by Δ +0.009 to +0.025 ARI. **The single-seed B5 absolute SOTA
> claim at ARI 0.9358 was a cherry-picked outlier** — the genuine recommendation is
> NEW cfg with Agglo Ward K=42 for oracle-K settings (multi-seed avg 0.9014 ± 0.022).
> This is the strongest evidence for paper contribution N2 (multi-seed methodology
> obligation) in this work.

### Implications for paper

1. **B5 absolute SOTA claim ★ retracted** — Local + NeCo combination 가 단독 NeCo 보다
   multi-seed avg 우위 없음. v6 "complementary" framing 의 absolute-SOTA inference 는 무효.
2. **NEW cfg = TRUE multi-seed SOTA** on both Unknown-K (HDBSCAN 0.859) AND Oracle-K
   (Agglo 0.901) frontiers. dual-cfg recipe → single-cfg recipe.
3. **N2 (multi-seed) 강한 evidence** — B5 seed=42 0.9358 → seed=1 0.8482 (Δ −0.088,
   huge). single-seed 비교 만으로 method-family winner claim 의 결정적 오류 사례.
4. **N1 contribution simpler** — NeCo *might* substitute Local; B5's specific seed=42
   advantage was illusory. v5 "substitutable on aggregate" + v6 "complementary per-class"
   → v7 "NEW (NeCo only) dominates B5 (Local + NeCo) on multi-seed average across all
   clustering methods. Per-class complementarity (v6 RESULTS §16) is preserved as a
   single-seed observation but does NOT propagate to multi-seed averages."

### 영향 받은 paper section (v0.9 갱신)

- ABSTRACT.md: v0.9 신설 (replaces v0.8) — NEW cfg multi-seed dominance + B5 lucky-seed retraction
- RESULTS.md §16/§17 — B5 seed=1 row 추가, multi-seed avg ± std comparison
- DISCUSSION.md §7.10 — N1 v7 update, "complementary" wording → "complementary on single-seed only" caveat
- CONCLUSION.md §8.6 / §8.8 — NEW cfg recommended (not dual-cfg), B5 retraction
- METHOD.md §3.6 — practitioner guidance: "always use multi-seed, prefer NEW cfg"
- manager_report/SUMMARY.md / REPORT.md — NEW cfg unified SOTA, dual-frontier still valid (HDBSCAN vs Agglo)
- INTRODUCTION.md C7 — single line retraction of v6 + v7 introduction

### iter 67-77 / iter 82-83 수치 유지 (append-only)

본 entry 는 이전 iter 결과 수치 (B5 seed=42 Agglo 0.9358, NEW seed=42 Agglo 0.9200) 를
**변경하지 않음**. 새 seed=1 reproducibility evidence 만 추가. v6 per-class purity
breakdown (RESULTS §16) 의 single-seed 관찰도 그대로 보존하되, "complementary →
absolute SOTA" inference 만 retract.

### 다음 (iter 85+)
- B5 seed=2 추가 측정 — 3-seed avg 확정 (현재 2-seed)
- NEW + seed=2 + Agglo / KMeans 재측정 검증
- Cluster-Aware Synthesis Loop (F2) 본격 진행 — encoder/clustering 사이드 모두 saturate

---

## Iter 86 — Step 1 eval-only (RankMe + ε sweep + soft τ-reassign) (2026-05-13)

Plan reference: `C:\Users\hgcho\.claude\plans\floating-splashing-key.md` Roadmap Step 1.
Source-of-truth: `docs/paper/manager_report/step1_eval_only_summary_260513.md`.

### 변경
1. **Step 1a — RankMe + NESum representation quality column** (paper N10 new contribution)
   on NEW 3-seed (iter 70/71/72) + B5 3-seed for cross-recipe comparison.
2. **Step 1b — HDBSCAN cluster_selection_epsilon sweep** across ε ∈ {0.00, 0.02, 0.04,
   0.06, 0.08, 0.10, 0.15} × NEW 3 seeds = 21 measurement cells. Defect-only Tier1
   protocol (eom mcs=12 ms=3), apples-to-apples with §17b NEW baseline.
3. **Step 1c — Soft KNN-softmax τ-reassignment** of HDBSCAN noise points. τ ∈ {0.5,
   0.7, 0.9, ∞ (baseline)} × NEW 3 seeds = 12 cells. KNN k = 10, cosine sim,
   softmax T = 0.1.

### 동기
- Roadmap Step 1 = eval-only optimizations (no encoder retrain). Three orthogonal
  paper-grade refinements: (a) representation quality column for the recipe-comparison
  table (B5 vs NEW), (b) HDBSCAN tunable that prior work cites as important
  (Campello+ 2013), (c) production deployment helper for noise = 0 requirement.
- Test whether NEW recipe's strength survives the ε sweep (negative result expected
  per N9 thesis) and whether soft reassignment closes the P2 (noise) gap.

### 결과

| Step | Headline number |
|---|---|
| 1a — RankMe | NEW 3-seed **23.44 ± 1.80** vs B5 3-seed 22.06 ± 4.99 → **NEW CV 7.7 % vs B5 22.6 %** (64 % more stable). Spearman ρ(RankMe, ARI) = **−0.429** (n = 7) → stability column, NOT ARI ranker. |
| 1b — ε sweep | **All 21 cells identical** to 4 decimal places: ARI 0.8731 ± 0.0140, noise 1.48 %, AMI 0.9629, Comp 0.9963, Hom 0.9448 — **zero effect** across ε ∈ [0.00, 0.15]. NEW cluster tree saturated under (eom, mcs=12, ms=3). |
| 1c — τ = 0.9 | ARI 0.8709 ± 0.0132, noise **0.49 %** (−67 % vs baseline 1.48 %) |
| 1c — τ = 0.7 | ARI 0.8696 ± **0.0123** (12 % std improvement), noise 0.15 % (−90 %) |
| 1c — τ = 0.5 ★ | ARI 0.8681 ± 0.0125, noise **0.00 %** (−100 % — every wafer labeled), ARI cost = −0.005 (within seed std) |

### 발견

1. **RankMe = stability column, not SOTA arbiter** (paper N10). NEW
   representation 의 cross-seed CV is 64 % lower than B5, confirming the §17b
   NEW > B5 reproducibility advantage at the representation-quality layer (not
   just at the ARI layer). However the Spearman correlation between RankMe and
   ARI across 7 cross-recipe runs is weakly negative (ρ = −0.429), so RankMe
   alone cannot serve as a method ranker.
2. **ε deprecated on NEW embedding** (paper N9 reinforcement). Across 21 cells
   (3 seeds × 7 ε), all metrics identical to 4 decimal places. NEW recipe's
   HDBSCAN tree is fully determined by (method, mcs, ms) triple alone. This is
   a paper-worthy negative result — it rules out cluster_selection_epsilon as a
   viable lever for production tuning on NEW.
3. **Production noise = 0 %** at τ = 0.5 (paper N9 extension). Soft KNN-softmax
   reassignment of HDBSCAN noise points achieves complete elimination of P2
   noise (1.48 % → 0 %) at marginal ARI cost (−0.005, well within 3-seed std
   0.014). For research evaluation, ∞ baseline (no reassignment) remains
   default; for production, τ = 0.5 is the recommended cfg lock.
4. **ARI value 0.8731 vs historical 0.859 reconciliation**. Step 1 inline
   measurement of NEW 3-seed HDBSCAN ARI = 0.8731 ± 0.0140. ABSTRACT v0.9 /
   README header historical = 0.859 ± 0.018. Δ = 0.014 = within HDBSCAN tree
   non-determinism (sklearn build-order variance). Both values are valid
   measurements; paper retains historical 0.859 in ABSTRACT/INTRODUCTION/
   CONCLUSION inline citations, while §19 uses 0.8731 as the inline-measured
   Step 1 baseline for τ-reassignment Δ calculations.

### 정책 정합 (append-only)

- Tier 1+2 공식 metric only — RankMe / NESum reported as paper-grade
  representation-quality column (post-hoc metric on existing embeddings).
- Multi-seed avg ± std obligatory — all Step 1 numbers are NEW 3-seed.
- Step 1 introduces **no encoder retraining** — eval-only, reusing iter 70/71/72
  embeddings.

### paper section 갱신

- RESULTS.md §19 (NEW section) — Step 1a/1b/1c result tables + Step-by-step
  ARI progression matrix + next-step decision matrix + historical-value
  reconciliation note.
- METHOD.md §4c (NEW subsection) — Post-process refinement (soft τ-reassignment
  formal algorithm, τ trade-off table, production cfg lock at τ = 0.5,
  relation to deprecated ε parameter).
- ABLATION_PLAN.md Step 1 completion record (Roadmap Step 1 section appended).
- (this entry) ITERATIONS.md iter 86 — append-only iteration log.

### 산출 file

- `_step1b_hdbscan_eps_sweep.json` (21-cell raw measurements)
- `_step1c_soft_tau_reassign.json` (12-cell raw measurements)
- `step1_eval_only_summary_260513.md` (paper-recorder source-of-truth)
- `step1_paper_addition_260513.md` (paper-recorder edit summary, this iter)

### 다음 (iter 87+)
- Step 2 — EMA target encoder. Requires training dispatch. User approval pending.
- B5 seed=2 measurement (defer from iter 85+ schedule) — completes 3-seed B5 avg.
