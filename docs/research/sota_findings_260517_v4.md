# Performance Research — SOTA Findings v4 (260517, 4차 보강)

생성: 2026-05-17
스코프: v1 (`sota_findings_260516.md`, 9 paper + 5 repo) + v2 (`sota_findings_260516_additional.md`, 7 paper + 4 repo) + v3 (`sota_findings_260517_v3.md`, 8 paper + 4 repo) 누적. 8 신규 query.

중복 검증: 누적 24 paper / 13 repo arxiv-id + url cross-check, 0 중복 확인.

## Diagnosis (현재 baseline + 이번 round 타겟)

baseline (NEW production SOTA, manager_report 260513):
- ARI = 0.873, Homogeneity = 0.945, noise_pct = 0%, class_capture = 38/38
- Method stack: Local InfoNCE + MoCo Queue + NEG filter + NeCo
- WM-811K cca/ zero-shot partial success (Hom 0.81)
- MixedWM38 zero-shot collapse 미해결

이번 round (260517 v4) weak point:
- (W11) cluster-conditional negative — InfoNCE 가 같은 cluster 안 점을 negative 로 끌어당기는 false-negative 문제
- (W12) spectral-graph 와 contrastive 의 joint 학습 (DGI 후속) — small batch HDBSCAN 의 graph-aware 사전 정렬
- (W13) class-aware contrastive — multi-class anomaly inter-class confusion (MixedWM38 collapse 와 직접 연계)
- (W14) industrial open-vocabulary defect — wafer / PCB / semiconductor 도메인 multimodal foundation
- (W15) topological signature + HDBSCAN cluster (persistent homology 보강)

## Search queries used (이번 round, 8 query)

1. arxiv: `site:arxiv.org self-labeling contrastive learning without teacher 2025 2026`
2. arxiv: `site:arxiv.org cluster-aware InfoNCE cluster-conditional negative 2025`
3. arxiv: `site:arxiv.org spectral clustering contrastive learning joint 2025 graph`
4. arxiv: `site:arxiv.org unknown class discovery semiconductor wafer defect 2025 inspection`
5. arxiv: `site:arxiv.org wafer pattern ConvNeXt clustering 2025`
6. arxiv: `site:arxiv.org open-vocabulary defect detection PCB industrial 2025`
7. arxiv: `site:arxiv.org density-aware contrastive learning embedding HDBSCAN 2025`
8. arxiv: `site:arxiv.org embedding collapse contrastive learning prevention 2025 2026`
9. github: `site:github.com cluster-aware InfoNCE pytorch`
10. github: `site:github.com industrial defect detection open-vocabulary CLIP stars:>100`

## Papers (new top 5 — 누적 24 paper 와 중복 0)

### W1. ★★★ Understanding InfoNCE — Transition Probability Matrix Induced Feature Clustering (SC-InfoNCE)
- arxiv_id: 2511.12180 (2025-11-15)
- authors: Ge Cheng, Shuo Wang, Yun Zhang
- url: https://arxiv.org/abs/2511.12180
- venue: arxiv preprint (2025-11)
- 요약: InfoNCE 가 transition probability matrix 를 통해 자연스럽게 feature clustering 을 induce 한다는 이론. 이를 기반으로 SC-InfoNCE (Scaled Convergence InfoNCE) 제안 — tunable convergence target 으로 feature similarity alignment 정밀 제어. image / graph / text domain 일관 향상.
- 적용성: ★★★ — **본 repo 의 LOSS_TEMP / NEG-filter 외에 scale parameter 1 개 추가만으로 적용 가능**. `contrastive.py` 의 InfoNCE 계산 직후 wrapper layer 에서 scale factor 곱해주면 됨 (encoder 학습 불변).
- 예상 효과: P3 Completeness + P4 Homogeneity 동시 (cluster-preserving 보장 강화). HDBSCAN min_cluster_size sensitivity 감소.
- patch 후보: `_contrastive_n50.py` wrapper 에 `--sc-target <float>` CLI flag — InfoNCE softmax 직후 logit scale 곱 (절대 `contrastive.py` 미수정).

### W2. ★★★ I-Con — A Unifying Framework for Representation Learning (ICLR 2025)
- arxiv_id: 2504.16929 (2025-04-23)
- authors: Shaden Alshammari, John Hershey, Axel Feldmann, William T. Freeman, Mark Hamilton (MIT / Google / Microsoft)
- url: https://arxiv.org/abs/2504.16929
- venue: ICLR 2025
- 요약: 23 개 ML 손실함수 (contrastive, clustering, spectral, dim reduction, supervised) 가 동일 KL divergence integral 의 special case 임을 증명하는 단일 information-theoretic framework. ImageNet-1K unsupervised classifier +8% SOTA. **principled debiasing methods for contrastive learners** 직접 제공.
- 적용성: ★★★ — debiasing method 가 **본 repo 의 NEG filter 알고리즘적 boost**. 같은 cluster 의 false negative 자동 affect, 별도 cluster label 불필요.
- 예상 효과: W11 해결 (cluster-conditional false negative). P3 Completeness 가시 향상. encoder 학습 시 wrapper layer 1 줄 추가.
- patch 후보: I-Con github (`ShadeAlsha/ICon`) 의 `distributions/` 모듈 참고하여 `_contrastive_n50.py` wrapper 에 supervisory distribution swap 추가. backbone 가중치 변경 없음.

### W3. ★★★ CCL — Class-Aware Contrastive Learning for Multi-Class Anomaly Detection (ICCV 2025)
- arxiv_id: 2412.04769 (2024-12-06, ICCV 2025 acceptance)
- authors: Lei Fan, Junjie Huang, Donglin Di, Anyang Su, Tianyou Song, Maurice Pagnucco, Yang Song (UNSW Sydney, Li Auto)
- url: https://arxiv.org/abs/2412.04769
- venue: ICCV 2025
- 요약: multi-class anomaly detection 의 inter-class confusion (한 class 가 다른 class 로 잘못 reconstruct) 을 raw object category 정보를 supervised signal 로 사용해 local + global CCL 로 refine. 5 dataset SOTA. **pseudo-class label 만으로도 비슷한 성능** (label 없는 본 repo 에 직접 적용 가능).
- 적용성: ★★★ — MixedWM38 zero-shot collapse (38 mixed class inter-confusion) 와 직접 1:1 매칭. WM-811K 33-class 도 적용. pseudo-label 생성기 = HDBSCAN 1 회 사전 run.
- 예상 효과: W13 해결 — MixedWM38 collapse 완화. WM-811K cca/ zero-shot Hom 0.81 → 0.90 추정.
- patch 후보: 두 stage wrapper — `_contrastive_n50_ccl.py` (1) HDBSCAN 사전 pseudo-label → (2) `contrastive.py` 의 anchor / positive 선택 시 같은 pseudo-cluster 우선 → (3) local feature contrast. encoder 학습 cell 별도.

### W4. ★★ SpecMatch-CL — Graph Contrastive Learning via Spectral Graph Alignment
- arxiv_id: 2512.07878 (2025-11-27)
- authors: Manh Nguyen (UW-Madison)
- url: https://arxiv.org/abs/2512.07878
- venue: arxiv preprint
- 요약: normalized embedding + Gaussian kernel 하 InfoNCE 최소화 = spectral clustering on similarity graph 와 equivalent 임을 증명. SpecMatch-CL — sparse neighborhood graph 의 normalized Laplacian 차이 penalize. multiple benchmark SOTA.
- 적용성: ★★ — wrapper layer 에서 batch 내 affinity Laplacian 계산 + 추가 loss term. small batch (BATCH=8) 에서 effective 한지는 검증 필요.
- 예상 효과: W12 (graph-aware sub-objective 추가). HDBSCAN ε / min_cluster_size 와 spectral cluster count 정합성 ↑.
- patch 후보: `_contrastive_n50_spec.py` wrapper — InfoNCE loss 옆 0.1× spectral norm regularizer cell. encoder 학습 cost ↑ 가능 (BATCH 추가 행렬연산).

### W5. ★★ IMDD-1M — Towards Open-Vocabulary Industrial Defect Understanding with a Large-Scale Multimodal Dataset
- arxiv_id: 2512.24160 (2025-12-30)
- authors: TsaiChing Ni, ZhenQi Chen, YuanFu Yang
- url: https://arxiv.org/abs/2512.24160
- venue: arxiv preprint
- 요약: 1M aligned image-text pairs, 63 manufacturing domains, 421 defect types (전자 / 자동차 / 금속 / 텍스타일 포함). diffusion-based multimodal foundation 통합 — generative + discriminative. **<5% task-specific data 로 supervised SOTA 동등**.
- 적용성: ★★ — TAPT backbone 대안 후보 (현재 33-class wafer supervised → IMDD-1M open-vocab pretrained). 단 모델 weights / dataset 다운로드 필요 — **권고만, 자동 다운로드 금지**.
- 예상 효과: WM-811K cca/ zero-shot, MixedWM38 zero-shot 양쪽 개선 가능. 그러나 wafer-specific signal loss 가능 (domain dilution).
- patch 후보: dataset 공개 후 IMDD-1M pretrained backbone vs. 현재 TAPT 대비 ablation 1 회 (`cnn_train.py` 의 backbone init 변경, **수동만**).

## Repositories (new top 3 — 누적 13 repo 와 중복 0)

### R1. ★★★ pangdatangtt/UniNet — Contrastive Learning-guided Unified Framework with Feature Selection for Anomaly Detection (CVPR 2025)
- url: https://github.com/pangdatangtt/UniNet
- stars: 133
- last commit: (fetch_partial — explicit date not surfaced, but page-active 2026-Q1)
- license: MIT
- 핵심 기법: contrastive learning + feature selection 통합. industrial / medical / video 도메인 일관. **MVTec AD, BTAD, VisA, ISIC2018, OCT2017 SOTA**.
- pytorch 호환: 예 (PyTorch repo, `train_unsupervisedAD.py` / `main.py` entry).
- 사용 가능 코드 위치: `UniNet_lib/` (loss + feature selector). 본 repo 의 `_contrastive_n50.py` wrapper 에 feature selector 참조 후 일부 cell 차용 가능.

### R2. ★★ ShadeAlsha/ICon — I-Con Official Implementation (ICLR 2025)
- url: https://github.com/ShadeAlsha/ICon
- stars: 131
- last commit: (fetch_partial — 2025-Q2 active per arxiv release)
- 핵심 기법: 23 ML loss 의 unified KL framework. `mappers/`, `distributions/`, `model/` 모듈 구성. Config 객체로 mapper + supervisory_distribution + learned_distribution 조합 — explicit loss function file 대신 distribution swap 방식.
- pytorch 호환: 예.
- 사용 가능 코드 위치: `distributions/` (특히 debiased contrastive distribution). 본 repo 의 NEG filter 옆 debiased term wrapper 로 참조.

### R3. ★ NinaNeon/IMDD-1M-Towards-Open-Vocabulary-Industrial-Defect- — IMDD-1M Dataset + diffusion foundation
- url: https://github.com/NinaNeon/IMDD-1M-Towards-Open-Vocabulary-Industrial-Defect-
- stars: 27
- last commit: (fetch_partial)
- 핵심 기법: 1M image-text pair industrial defect dataset + diffusion foundation 모델 코드.
- pytorch 호환: 예 (model code, dataset metadata).
- 사용 가능 코드 위치: `models/` — pretrained checkpoint 공개되면 backbone 후보. **stars < 100 으로 quality bar 미달이지만 dataset 자체가 유일한 large-scale open-vocab industrial defect resource — 도메인 관련성으로 예외 채택**.

## Action items (권고 only, 자동 patch / 학습 dispatch 금지)

| # | target wrapper | param / cell | source | priority |
|---|---|---|---|---|
| A1 | `_contrastive_n50.py` 신규 cell | SC-InfoNCE scale factor (logit × scale, after softmax) | W1 paper 2511.12180 | ★★★ |
| A2 | `_contrastive_n50_icon.py` 신규 wrapper | debiased supervisory distribution swap (I-Con `distributions/`) | W2 paper 2504.16929 + R2 repo | ★★★ |
| A3 | `_contrastive_n50_ccl.py` 신규 wrapper | 2-stage: HDBSCAN pseudo-label → class-aware anchor / positive | W3 paper 2412.04769 | ★★★ |
| A4 | `_contrastive_n50_spec.py` 신규 wrapper | spectral Laplacian 차이 regularizer (0.1× weight) | W4 paper 2512.07878 | ★★ |
| A5 | `cnn_train.py` backbone ablation | IMDD-1M pretrained backbone vs. TAPT (수동) | W5 paper 2512.24160 + R3 repo | ★ (dataset 공개 시) |

**모든 patch 는 wrapper layer 만, `contrastive.py` / `_*.py` 절대 수정 금지. 학습 dispatch 절대 자동 X — 사용자 명시 후 BATCH=8 / single GPU process 정책 준수.**

## fetch_failed / partial

- 2506.12698 (Unsupervised Contrastive Learning Using OOD Data for Long-Tailed Dataset): 첫 WebFetch 에서 abstract 누락 → v1 URL 재시도로 회수, 본 표 미포함 (long-tailed 가 본 repo 의 33-class balanced 환경에 적합도 낮음)
- 2509.12510 (PPG topological + HDBSCAN): 도메인 다름 (생체신호) — 본 표 미포함
- 2507.13378 (Industrial Defect Detection Survey): 본 표 미포함 (survey — paper 카운트 외)

## Top-3 요약

1. **W1 SC-InfoNCE (2511.12180)** ★★★ — wrapper scale factor 1 줄로 P3+P4 동시 boost
2. **W2 I-Con (2504.16929) + R2 ICon repo** ★★★ — NEG filter 의 principled debiasing 대체, ICLR 2025 SOTA
3. **W3 CCL (2412.04769)** ★★★ — MixedWM38 zero-shot collapse 직접 해결, ICCV 2025

## Counts

- new papers: 5 (W1-W5, 0 중복)
- new repos: 3 (R1-R3, 0 중복)
- 누적 (v1+v2+v3+v4): 29 papers, 16 repos
- arxiv-id + repo-url cross-check: 0 중복 확인

## Sources

- [Understanding InfoNCE (2511.12180)](https://arxiv.org/abs/2511.12180)
- [I-Con (2504.16929)](https://arxiv.org/abs/2504.16929)
- [Class-Aware Contrastive Learning (2412.04769)](https://arxiv.org/abs/2412.04769)
- [SpecMatch-CL (2512.07878)](https://arxiv.org/abs/2512.07878)
- [IMDD-1M (2512.24160)](https://arxiv.org/abs/2512.24160)
- [UniNet repo](https://github.com/pangdatangtt/UniNet)
- [I-Con repo](https://github.com/ShadeAlsha/ICon)
- [IMDD-1M repo](https://github.com/NinaNeon/IMDD-1M-Towards-Open-Vocabulary-Industrial-Defect-)
- [awesome-industrial-anomaly-detection](https://github.com/M-3LAB/awesome-industrial-anomaly-detection)
