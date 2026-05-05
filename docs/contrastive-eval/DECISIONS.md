# DECISIONS — 사용자 결정 history (감사 trail)

이 문서는 각 결정의 **사용자 명시 사유 + 거부된 대안 + 채택된 답** 을 기록.
새 세션에서 같은 논의 반복 방지.

---

## D-1. 평가 metric — 공식만 사용 (커스텀 폐기)

**시점**: 사용자 "전부 cnn 성능지표잖아 장난하나?" → "공식 지표를 써야 논문이나 대외 발표에 사내 발표에 신뢰성이 높아질 것 같다".

**거부**:
- 자체 정의 metric: `weighted_isolation`, `pure_rate`, `mixed_rate`, `isolation`, `contamination_rate`, `binary_homogeneity`
- 분류기 style metric: precision / recall / F1 / FPR / accuracy
- TP/FP/FN/TN 표

**채택**:
- Tier 1 (4 + class_fragmentation_summary): Completeness, AMI, noise_pct, class_capture_rate
- Tier 2 (3): Homogeneity, Silhouette (cosine), ARI
- Tier 3 (skip): NMI, V-measure, FMI, Davies-Bouldin, Calinski-Harabasz

**rationale**: 학술 출처 명확한 metric (sklearn / 1970s~2010s 논문) 만 발표/논문 인용 가능. 자체 metric 은 비교 불가.

---

## D-2. B-Cubed metric — drop

**시점**: D-1 후 사용자 "Recall 0.9444 / 보조 4 — 뭔지 모르겠는데 없어도 될듯".

**채택**: B-Cubed Precision / Recall / F1 (Bagga & Baldwin 1998) 모두 Tier 1 에서 제외.

**rationale**: Completeness 가 사용자 goal "같은 class → 같은 group" 의 표준 답이고, B-Cubed Recall 도 같은 정보 item-level 이라 중복. 하나만 (Completeness) 으로 충분.

---

## D-3. 우선순위 P1-P4 lock-in

**시점**: 사용자 "불량이 1개라도 group 으로 나오는거 제일 중요하고 - recall 느낌, noise group 나오지 않는게 두번째".

**채택**:
- **P1** = `class_capture_rate` (모든 defect class 가 ≥1 group)
- **P2** = `noise_pct` (defect only)
- **P3** = Completeness (같은 class → 같은 group)
- **P4** = Homogeneity (group 안 한 class 만)

**rationale**: production 운영 직결. 누락 (P1) > false alarm (P2) > clustering quality (P3-P4).

---

## D-4. Multi-crop (SwAV) — NO

**시점**: 사용자 "wafer 내의 불량 분포가 같은 것들 잡아야하는데 random crop 이라니 안 되지".

**거부 사유**: SwAV multi-crop = global 384 + local 128 random crop. 우리 wafer 는 **위치 정보 (Edge-Top vs Edge-Bottom) 가 class identity** — random crop 으로 위치 정보 손상.

**채택**: `USE_LOCAL=True` (`contrastive.py` 기존 spec) 의 grid 기반 spatial contrast 만. 같은 wafer 의 다른 grid cell embedding 끼리 contrastive — 위치 보존.

---

## D-5. SupCon (Supervised Contrastive) — 주력 X

**시점**: 사용자 "여기 등록되어있지 않은 이미지들 나오면 성능 저하 되는걸 확인해서 안되겠다".

**거부 사유**: SupCon 은 label 직접 사용 → 학습 본 class manifold 만 sharp → unknown defect 가 known class 로 끌려감 → cluster 안 만들고 흡수됨. production 에 unknown defect 출현 가능성 있어 부적합.

**채택**: SSL 유지 (현재 contrastive.py InfoNCE 그대로). 옵션으로 2-stage hybrid (SSL → 가벼운 SupCon fine-tune) 가능하나 우선순위 낮음.

---

## D-6. Hard Negative Mining (Robinson 2021) — 채택

**시점**: 사용자 "일단 지금은 이거로 해보자 실전 현업 데이터 이미지로는 label 이 없어서 될지 모르겠네 아 infonce 가지고 하는거면 되려나".

**채택**: InfoNCE 위에 importance weighting 추가. β=1.0 시작.

```python
weights = exp(β · sim_neg)
weights = weights / mean(weights)
denom = score_pos + Σ weights · score_neg
loss = -log(score_pos / denom)
```

**rationale**: label 무관 (cosine similarity 만 사용) → production unlabeled data 에서도 작동.
효과: confusing sub-class (Edge-Top × 4 obj) 분리 ↑ 기대.

---

## D-7. Class size — 다음 학습부터 random

**시점**: 사용자 "class 갯수 다 200개씩 쓰지 말고 좀 랜덤하게 써라 학습이나 pred 에서".

**채택**: `_contrastive_n50.py` subset hardlink builder 의 per-class sample 수를 random 분포로 (e.g., 50 ~ 200+). production class imbalance 흉내.

**rationale**: 현실 production 은 class 마다 sample 수 매우 다름 (Normal 80%, defect_a 15%, defect_b 5% 식). 균등 학습이 비현실적.

---

## D-8. 이미지 데이터 보존 — 추가 합성 / augment 변경 X

**시점**: 사용자 "이미지는 건드리지말자".

**거부**:
- Thick-Edge_fork 50 짜리 class 합성 데이터 추가
- augmentation 강도 / 종류 변경

**채택**: 합성 데이터 그대로. 학습 측만 변경 (sampling, hard mining, monitoring).

---

## D-9. GPU 작게 — BATCH=16, IMAGE_SIZE=384

**시점**: 사용자 "gpu 를 쓸 때는 좀 작게 써라".

**채택**: BATCH=16 (overall 학습 시), IMAGE_SIZE=384 유지. queue size 4096 유지.

**rationale**: 사용자 GPU 자원 보호. SimCLR 권고 batch 4096 같은 건 우리 setup 에 부적합.

---

## D-10. 합성 데이터 sub-style 발견 (Full_*** + Thick-Edge_fork)

**진단 결과**: Full_scratch_rot / Full_fork / Full_bank_boundary / Thick-Edge_fork 4 class 가 2 cluster 로 split. 진단 4 가지 모두 통과:
- HDBSCAN sweep: 모든 hyperparameter 에서 동일 split (HDBSCAN 문제 X)
- intra-class within/cross ratio: 4.4 ~ 8.7 (진짜 두 무리)
- GMM bimodality BIC: -19,698 (강한 bimodal)
- → **encoding 정확. 합성 데이터에 진짜 두 sub-style 있음**

**처방** (사용자 결정 대기):
- (a) 합성 코드 검토 후 두 종류 통일 → frac_single_cluster ↑
- (b) GT 두 sub-class 로 분리 (e.g., Full_scratch_rot_A / Full_scratch_rot_B)

**현재 상태**: 결정 보류. 학습 측 변경 X.

---

## D-11. 평가 monitoring — alignment+uniformity 항상, k-NN 옵션

**시점**: 사용자 "label 이 있을 때만 하자".

**채택**:
- **alignment + uniformity** (Wang & Isola 2020) — 매 epoch, label 무관, 항상
- **k-NN top-1** — label 있는 작은 subset 만, 옵션
- **periodic HDBSCAN + Tier 1** — 5 epoch 마다, label 있을 때

production (label 0%): alignment + uniformity 만.

---

---

## D-12. HDBSCAN tuning 추가 X — embedding 개선 우선

**시점**: Iter 0 cluster-analyzer 진단 (2026-05-05).

**결과**: 현재 HDBSCAN cfg (mcs=12, ms=1, eom) 가 sweep 결과 ARI 최대 (0.7143) 와 일치.
모든 hyperparameter 조합에서 같은 split 발생 (특히 Full_*** 4 class).

**채택**: HDBSCAN tuning 추가 안 함. 학습 측 (encoder / loss / sampling) 변경으로 만 개선.

**rationale**: cluster-analyzer Test 3 sweep (min_cs 8~30, ms 1/4, eps 0/0.1) 에서 Full_scratch_rot
가 항상 [101, 97] split. encoding 자체가 두 sub-style 인지. HDBSCAN 은 정확히 두 cluster 산출.

---

## D-13. Normal sampling 비율 조정 — 1순위 (Iter 1 계획)

**시점**: Iter 0 chain 분석 (2026-05-05).

**문제**: cluster 37 (Normal_bank_boundary, size 691) 가 cluster 12, 17 (Edge-Top/Bottom_bank_boundary)
와 boundary blur. Normal noise_pct 22.9% 의 직접 원인. cluster 36, 38 도 Normal split 으로 발견됨.

**채택 (계획)**: Normal 1000 → 200~500 으로 줄이고, 또는 oversampling 정책 도입.

**rationale**: 합성 baseline 의 Normal 비율 12% (1000/8357) 가 너무 높아 Edge × bank_boundary
와 manifold 가 겹침. production (Normal 80%) 흉내 위해서라도 dedicated Normal anchor 또는
class-balanced sampling 필요.

---

## D-14. Hard mining 진화 — Robinson 2021 → NV-Retriever / SCHaNe

**시점**: Iter 0 performance-research (2026-05-05).

**기존 (D-6)**: Robinson 2021 InfoNCE β param.

**업데이트**:
- **NV-Retriever** (arxiv 2407.15831, NVIDIA 2024) — positive-aware false-negative filter.
  Full_*** sub-cluster repulsion 직격 가능 (image-analyzer 의 cluster_split pairs 와 일치).
- **SCHaNe** (arxiv 2308.14893) — dissimilarity-weighted hard negative re-weight on SupCon.
  +3.32% few-shot 입증.

**채택 (계획)**: Iter 1 부터 NV-Retriever filter 우선 시도. 효과 부족 시 SCHaNe 추가.
구현 베이스: `pytorch-metric-learning` (6.3k stars, v2.9.0) 의 miner API drop-in.

**rationale**: D-5 (SupCon 주력 X) 정책 유지하면서 SSL InfoNCE 위 hard mining 만 추가.
SupCon-style filter (positive-aware) 는 SSL 에서도 augmentation positive 활용 가능.

---

## 참고

- 모든 metric 정의 → `METRICS.md`
- 학습 monitoring 상세 → `MONITORING.md`
- Hard mining 수학 → `HARD_NEGATIVE.md`
- production 시나리오 → `PRODUCTION.md`
