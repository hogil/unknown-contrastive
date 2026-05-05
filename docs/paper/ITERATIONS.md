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

---

## Iter 1 — (계획) Production-realistic sampling + monitoring

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
