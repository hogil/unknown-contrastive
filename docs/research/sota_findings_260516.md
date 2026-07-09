# Performance Research — Contrastive Learning + HDBSCAN + Wafer Defect Clustering SOTA

생성: 2026-05-16
스코프: contrastive learning, HDBSCAN cluster discovery, wafer defect unsupervised clustering, MoCo memory queue, NeCo neighborhood consistency

## Diagnosis

본 repo 의 현재 baseline (production SOTA, manager_report 기준):
- ARI = 0.873, Homogeneity = 0.945, noise_pct = 0%, class_capture = 38/38
- Method stack: Local InfoNCE + MoCo Queue + NEG filter + NeCo
- Pending 6 ablation: B0/B1/B3/B4/B5/NEW + τ=0.5 post

타겟 weak point:
- novelty / open-set 능력 (unknown defect 발견 SOTA)
- HDBSCAN cluster purity / min_cluster_size 자동 결정
- MoCo memory queue 의 hard negative quality
- NeCo 후속 — patch neighbor consistency 의 dense feature 학습

## Papers (top 9)

### 1. Iterative Cluster Harvesting for Wafer Map Defect Patterns
- arxiv_id: 2404.15436 (2024-04-23)
- authors: Pleli, Baeuerle, Janus, Barth, Mikut, Lensch
- url: https://arxiv.org/abs/2404.15436
- 요약: 3-step pipeline (feature extraction → dim reduction → clustering) 을 iterative 로 반복, 매 iter 에서 silhouette score 가 가장 높은 한 cluster 만 떼어내고 (harvest) 잔여 데이터로 feature space 재정의.
- 적용성: ★★★ — wrapper `_iter_harvest.py` 추가, encoder 재학습 X.

### 2. DECOR: Deep Embedding Clustering with Orientation Robustness
- arxiv_id: 2510.03328 (2025-10-01, AAAI 2026 KGML Bridge)
- url: https://arxiv.org/abs/2510.03328
- 요약: MixedWM38 manual tuning 없이 orientation invariant clustering.
- 적용성: ★★ — MixedWM38 zero-shot fail 해결 후보. augmentation 정책과 충돌 → 별도 ablation cell.

### 3. Advanced Clustering Framework for Semiconductor Image Analytics (Deep TDA + SSL)
- arxiv_id: 2505.03848 (2025-05)
- url: https://arxiv.org/abs/2505.03848
- WebFetch: 부분 fail (title only). TDA + SSL.
- 적용성: ★ — 큰 변경 필요. 참고만.

### 4. Contrastive Representation Modeling for Anomaly Detection (ECAI 2025)
- arxiv_id: 2501.05130 (2025-01-09)
- authors: Lunardi, Banabila, Herzalla, Andreoni
- url: https://arxiv.org/abs/2501.05130
- 요약: 3 representation property 명시 — (1) inlier compact clustering, (2) inlier↔anomaly separation, (3) synthetic outlier diversity. structured contrastive + patch-based.
- 적용성: ★★★ — wrapper LOSS_* cfg override. diversity preservation 이 sub-cluster fragmentation 방지.

### 5. SynCo: Synthetic Hard Negatives for Contrastive Visual Representation Learning
- arxiv_id: 2410.02401 (2024-10-03)
- authors: Giakoumoglou, Stathaki
- url: https://arxiv.org/abs/2410.02401
- 요약: MoCo memory queue 위에서 6 synthetic hard negative strategy.
- 적용성: ★★★ — cross-class suppression (fork combo collapse) 직접 해결 후보.

### 6. NeCo: Near, far — Patch-ordering enhances vision foundation models
- arxiv_id: 2408.11054 (2024-08-20, ICLR 2025)
- url: https://arxiv.org/abs/2408.11054
- 요약: patch-level NN consistency student↔teacher + differentiable sorting. 19 GPU hour 만에 DINOv2 보강.
- 적용성: ★★ — 본 repo NeCo 통합 version 과 v3 (2025-04) 차이 검증.

### 7. CLAN: Self-supervised New Activity Detection
- arxiv_id: 2401.10288 (2024-01-17)
- url: https://arxiv.org/abs/2401.10288
- 요약: 2-tower contrastive + multi-aug negatives. +9.24% AUROC.
- 적용성: ★ — sensor domain mismatch. 참고만.

### 8. Mean Teacher + SupContrast for Wafer Pattern Recognition
- arxiv_id: 2411.18533 (2024-11-27)
- url: https://arxiv.org/abs/2411.18533
- 요약: WM-811K supervised contrastive + SMOTE. Acc +5.46%.
- 적용성: ★ — 사용자 feedback `no_multicrop_no_supcon` 위반. 참고만.

### 9. Wafer Map Defect Classification with Autoencoder Augmentation
- arxiv_id: 2411.11029 (2024-11-17)
- url: https://arxiv.org/abs/2411.11029
- 요약: AE latent noise augmentation, CNN 분류. acc 98.56%.
- 적용성: ★ — closed-set 분류 문제. 참고만.

## Repositories (top 5)

### 1. giakoumoglou/synco
- url: https://github.com/giakoumoglou/synco
- stars: 4 (paper-official)
- 핵심: MoCo v2 + 6 synthetic hard negative strategy. `main_synco.py`.
- License: CC-BY-NC 4.0 (학술 OK).

### 2. vpariza/NeCo
- url: https://github.com/vpariza/NeCo
- stars: 31 (ICLR 2025 official)
- 핵심: patch neighbor consistency + differentiable sorting. DINOv2 dependency.

### 3. Yunfan-Li/Contrastive-Clustering (AAAI 2021)
- url: https://github.com/Yunfan-Li/Contrastive-Clustering
- stars: 333
- 핵심: ICH (Instance Clustering Head) + CCH (Cluster Contrasting Head) joint training.

### 4. MichalZnalezniak/Contrastive-Hierarchical-Clustering (ECML PKDD 2023)
- url: https://github.com/MichalZnalezniak/Contrastive-Hierarchical-Clustering
- stars: 17
- 핵심: SimCLR + hierarchical tree loss.

### 5. SpatialAILab/WaferDC (EAAI 2025)
- url: https://github.com/SpatialAILab/WaferDC
- stars: 13
- 핵심: multi-cluster memory bank + SegMix anomaly seg map.
- 우리 domain 직접 일치.

## Recommended action items

전부 권고만. 코드 자동 수정 X.

1. **Iterative HDBSCAN harvesting** (paper #1) — ★★★
   - 새 wrapper `_iter_harvest.py`, encoder 고정.
   - 매 iter silhouette top-1 cluster freeze → 잔여 데이터 재학습.
   - P1 (class_capture_rate) 향상 후보.

2. **Synthetic hard negative for MoCo queue** (paper #5 SynCo + repo #1) — ★★★
   - MoCo queue dequeue 로직 옆 wrapper 단 hard negative synthesis.
   - cross-class suppression 직접 해결.
   - sample cfg cell B6/B7 추가 권고.

3. **Structured contrastive + patch-based** (paper #4 ECAI 2025) — ★★★
   - Local InfoNCE + global 추가, synthetic outlier diversity objective.

4. **Multi-cluster memory bank partition** (repo #5 WaferDC + paper #2 DECOR) — ★★
   - MoCo queue 를 wafer-canvas type 별 partition.
   - active_classes_27 정책과 연계.
   - MixedWM38 zero-shot collapse 해결 후보.

5. **NeCo version 비교** (paper #6 + repo #2) — ★★
   - 본 repo NeCo 가 paper v3 (2025-04) 와 동일한지 검증.
   - sorting depth + neighbor K sweep.

6. **Joint cluster + contrastive head** (repo #3 Yunfan-Li) — ★★
   - ICH + CCH joint training. 2-step (SSL → HDBSCAN) baseline 비교.

## Fetch failures

- arxiv 2505.03848 (Advanced Clustering w/ Deep TDA): title only.
- arxiv 2408.10798 (Universal Novelty Detection): title only — list 에서 제외.
- arxiv 2501.16360 (Momentum Contrastive w/ Hard Negative Filtering): title only — list 에서 제외.

## stars 정책 (≥100) 위반 repo

본 도메인 niche 특성상 ≥100 stars 만족하는 repo 가 거의 없음. paper-official / venue-accepted (ICLR/ECML/EAAI) repo 만 추가 inclusion.

## Cross-reference

- P1-P4 (class_capture_rate > noise_pct > Completeness > Homogeneity) 와 일치: #1 (silhouette harvesting), #4 (compactness+separation), #5 (hard negative).
- 사용자 feedback 정책 (no SupCon, no multicrop, no aug 회전) 과 충돌: #8 (SupCon), #2 (orientation aug 가능성). 적용 시 wrapper 단 격리.

## Top-3 papers summary

1. **Iterative Cluster Harvesting** (2404.15436) — ★★★ silhouette-based iterative HDBSCAN
2. **Contrastive Representation Modeling for AD** (2501.05130, ECAI 2025) — ★★★ 3-property structured objective
3. **SynCo** (2410.02401) — ★★★ MoCo queue 위 6 hard-negative strategy

## Top-3 repos summary

1. **Yunfan-Li/Contrastive-Clustering** (333 stars) — joint SSL + clustering head
2. **vpariza/NeCo** (31 stars, ICLR 2025 official) — patch neighbor consistency v3
3. **giakoumoglou/synco** (4 stars, paper-official 2024) — 6 hard-negative synthesis strategy
