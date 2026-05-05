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

위 분석 통합 후 우선순위 재정렬:

1. **Normal sampling 비율 조정** (cluster 37 boundary blur 해결 1순위) — Normal 1000 → 200~500 으로 줄이거나 oversample policy 변경
2. **NV-Retriever positive-aware false-negative filter** 도입 — Full_×3 + Normal split 직격
3. **SCHaNe dissimilarity-weighted hard mining** — Edge-Top × 4-obj merge (cluster 12/17) 분리
4. **합성 데이터 검토**:
   - Full_*** sub-style 4 pair 통일 또는 split label
   - Thick-Edge_fork 안에 Full_fork 와 비슷한 sample 5장 검토 (mislabel 의심)
   - cluster 35 `CMY389` 합성 anomaly 검토

**HDBSCAN tuning 은 더 이상 안 함** (이미 sweep optimum).
**Hard negative mining (Robinson 2021) → 더 진보된 NV-Retriever / SCHaNe 로 update**.

---

## Iter 1 — (계획) Normal sampling 조정 + NV-Retriever filter 도입

(미진행. 도입 시 paper-recorder 가 entry 추가)

---

## Iter 2 — (계획) Hard Negative Mining

(미진행)

---

## 변경 정책

- **각 iteration 1-3 변경만 적용** — 효과 추적 가능하도록 atomic.
- 여러 변경 동시 시 효과 분리 불가 (ablation 불가).
- 결과 비교는 `RESULTS.md` 표 row 단위로 누적.
- 거부된 옵션 + 사유는 `docs/contrastive-eval/DECISIONS.md` D-N.
