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
