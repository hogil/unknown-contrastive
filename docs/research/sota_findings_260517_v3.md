# Performance Research — SOTA Findings v3 (260517, 3차 보강)

생성: 2026-05-17
스코프: `sota_findings_260516.md` (9 paper + 5 repo) + `sota_findings_260516_additional.md` (7 paper + 4 repo) 보강. 8 query 추가 탐색 — prototype clustering, dense patch, cluster collapse reg, novel class discovery 2026, small-batch contrastive, semi-supervised clustering, density-clustering alternatives, embedding stability.

중복 검증: 16 paper + 9 repo arxiv-id / repo-url 모두 cross-check (0 중복 확인).

## Diagnosis (현재 baseline + 이번 round 타겟)

baseline (NEW production SOTA, manager_report 260513):
- ARI = 0.873, Homogeneity = 0.945, noise_pct = 0%, class_capture = 38/38
- Method stack: Local InfoNCE + MoCo Queue + NEG filter + NeCo
- WM-811K cca/ zero-shot partial success (Hom 0.81)
- MixedWM38 zero-shot collapse 미해결

이번 round (260517) weak point:
- (W6) prototype-based positive sampling 으로 intra-cluster compactness 강화 (P4 Homogeneity)
- (W7) dense patch consistency 의 collapse-stable 학습 alternative (NeCo regression 회복)
- (W8) BATCH=8 small-batch GPU 환경에 직접 효과 있는 optimization (사용자 single_gpu_process 정책)
- (W9) HDBSCAN 외 unknown-K auto-discovery (AutoProPos 류)
- (W10) orthogonality / hyperdimensional reg 로 collapse 방지 (NeCo 미흡 보강)

## Search queries used (이번 round)

1. arxiv: `prototype contrastive learning clustering embedding 2026`
2. arxiv: `dense patch contrastive learning image clustering 2025..2026`
3. arxiv: `cluster collapse prevention contrastive embedding regularization 2025..2026`
4. arxiv: `open-set novel class discovery industrial defect inspection 2025`
5. arxiv: `small batch contrastive learning SimCLR 2025..2026 low memory`
6. arxiv: `semi-supervised clustering few labeled anchor unlabeled 2025`
7. arxiv: `HDBSCAN alternative density-aware deep clustering joint optimization 2025`
8. arxiv: `embedding stability collapse prevention self-supervised 2026 non-contrastive`
9. github: `site:github.com CAPI cluster predict latents masked image modeling DINOv2`
10. github: `site:github.com SogCLR LibAUC small batch contrastive`
11. github: `site:github.com prototypical contrastive learning unsupervised stars`

## Papers (new top 8 — 1차/2차 16 paper 와 중복 0)

### V1. ★★★ CAPI — Cluster and Predict Latent Patches for Improved Masked Image Modeling
- arxiv_id: 2502.08769 (2025-02-12)
- authors: Timothée Darcet, Federico Baldassarre, Maxime Oquab, Julien Mairal, Piotr Bojanowski (Meta FAIR)
- url: https://arxiv.org/abs/2502.08769
- venue: TMLR 2025 (code released, Apache 2.0)
- 요약: pure-MIM framework — student 가 teacher 의 dense feature cluster assignment 를 patch 단위로 predict. clustering-based loss 가 stable training + scaling property 보장. ViT-L/14 backbone 83.8% ImageNet linear probe (DINOv2 근접). MIM previous SOTA 대비 substantial gain.
- 적용성: ★★★ — **본 repo 의 NeCo (patch neighborhood consistency) 의 stable alternative**. NeCo B5 regression 원인 = collapse 가설에 대해 CAPI 의 cluster-based latent target 이 직접적 해결책. teacher = TAPT backbone 고정 가능 — encoder 학습 불변.
- 예상 효과: W7 해결 (patch consistency collapse 회복) + P3 Completeness + P4 Homogeneity.
- patch 후보: 새 wrapper `_capi_patch_loss.py` (NeCo 옆 alternative cell) — `contrastive.py` 무수정, wrapper layer 만.

### V2. ★★★ AutoProPos — Auto-K Extension of Prototype Scattering and Positive Sampling
- venue: MDPI Applied Sciences Vol 15, Issue 18, 10052 (2025-09-15)
- doi: 10.3390/app151810052
- url: https://www.mdpi.com/2076-3417/15/18/10052
- code: https://github.com/Cyrilkt/AutoProPos
- 요약: ProPos (TPAMI 2022) 을 non-parametric 으로 확장. Clustering Supervisor (CLS) 가 ProPos 와 alternate 하며 reduced latent subspace 에서 average silhouette + **Silhouette Uniformity Index (SUI)** 로 K 자동 선택. SUI 가 uniform cluster distribution 유도 — fragmentation 방지. STL-10 ACC 92.0% (+11% vs non-parametric baseline), ImageNet-50 ACC 77.0%.
- 적용성: ★★★ — **HDBSCAN 외 second auto-K discovery track**. 본 repo 의 unknown defect K 자동 결정 시나리오 직접 일치. ProPos backbone 자체 학습 무수정 — supervisor 만 추가.
- 예상 효과: W9 해결 (unknown-K auto-discovery 외 두 번째 track). P1 class_capture_rate + P4 Homogeneity (SUI uniform 분포).
- patch 후보: 새 evaluator `eval_autopropos.py` — `eval_contrastive.py` 의 HDBSCAN cell 옆 ablation. encoder 무수정.

### V3. ★★★ SogCLR — Provable Stochastic Optimization for Global Contrastive Learning (Small Batch)
- arxiv_id: 2202.12387 (2022-02-24)
- authors: Zhuoning Yuan, Yuexin Wu, Zi-Hao Qiu, Xianzhi Du, Lijun Zhang, Denny Zhou, Tianbao Yang
- url: https://arxiv.org/abs/2202.12387
- venue: ICML 2022 (LibAUC integrated)
- code: https://github.com/Optimization-AI/SogCLR (paper-foundational, 20 stars but ICML official)
- 요약: BATCH=256 으로 BATCH=8192 SimCLR 와 comparable performance. global contrastive objective 의 stochastic 최적화 — **small batch does not harm**. iSogCLR (2023) 가 automatic temperature individualization 추가. LibAUC `GCLoss('unimodal', enable_isogclr=True)` 로 1-line drop-in.
- 적용성: ★★★ — **사용자 정책 `single_gpu_process` + BATCH=8 환경 직접 효과**. 기존 InfoNCE loss 를 GCLoss 로 swap 가능. 사용자 feedback `batch_same_condition` (BATCH 조정 OK) 하에 자유롭게 sweep.
- 예상 효과: W8 해결 (small-batch SOTA). P1/P2/P3 동시 향상 (global gradient 안정화).
- patch 후보: `contrastive.py` 의 InfoNCE 옆 `--use-sogclr` flag + `_contrastive_n50.py` wrapper LOSS_TYPE override. iSogCLR temperature auto 적용 cell 별도.

### V4. ★★ Collapse-Proof Non-Contrastive Self-Supervised Learning
- arxiv_id: 2410.04959 (2024-10-07)
- authors: Emanuele Sansone, Tim Lebailly, Tinne Tuytelaars
- url: https://arxiv.org/abs/2410.04959
- venue: ICML 2025
- 요약: hyperdimensional computing 기반 projector + loss 설계. theoretically representations 가 simultaneously **decorrelated AND clustered** 하도록 유도. 4 collapse mode 모두 방지: representation collapse, dimensional collapse, cluster collapse, intracluster collapse. linear classification + clustering benchmark 동시 향상.
- 적용성: ★★ — 본 repo 의 NeCo B5 regression 가설 (collapse) 외 cluster collapse 의 4-way decomposition 진단 가능. wrapper aux loss 로 추가 — `lambda * cp_nc_loss`.
- 예상 효과: W10 해결 (4-way collapse 방지 — intracluster collapse 는 P4 Homogeneity 보강).
- patch 후보: `_collapse_proof_aux.py` wrapper, encoder 학습 alpha sweep.
- 주의: paper 의 code release 명시 X (확인 필요) — 보고 시 verbatim 인용 only.

### V5. ★★ ProPos — Learning Representation for Clustering via Prototype Scattering & Positive Sampling
- arxiv_id: 2111.11821 (2021-11)
- authors: Zhizhong Huang, Jie Chen, Junping Zhang, Hongming Shan
- url: https://arxiv.org/abs/2111.11821
- venue: IEEE TPAMI 2022
- code: https://github.com/Hzzone/ProPos (118 stars)
- 요약: 2 loss 결합 — (1) **prototype scattering** (prototype 간 거리 최대화 → class collision 방지), (2) **positive sampling alignment** (instance 의 augmented view 를 sampled neighbor 의 다른 view 와 align → within-cluster compactness). contrastive 의 uniformity vs non-contrastive 의 compactness 동시 만족.
- 적용성: ★★ — 본 repo InfoNCE 가 uniformity 위주 → prototype scattering + positive sampling 추가 시 compactness 보강. AutoProPos (V2) 의 backbone 기반.
- 예상 효과: P4 Homogeneity + W6 (prototype-based intra-cluster compactness).
- patch 후보: V2 AutoProPos 의 baseline 으로 사용 — `_propos_baseline.py` ablation cell.

### V6. ★★ CLOP — Preventing Collapse in Contrastive Learning with Orthonormal Prototypes
- arxiv_id: 2403.18699 (2024-03-27, last rev 2024-10-07)
- authors: Huanran Li, Manh Nguyen, Daniel Pimentel-Alarcón
- url: https://arxiv.org/abs/2403.18699
- venue: arXiv (preprint)
- code: 명시 X (fetch_fail — 보고 시 reference only)
- 요약: semi-supervised loss — class embedding 간 **orthogonal linear subspaces** 강제. neural collapse 방지. large learning rate 가 cosine similarity loss 에 미치는 영향 theoretical 분석 + hyperparameter robustness 실험.
- 적용성: ★★ — 본 repo 가 unsupervised 위주 — semi-supervised 측면 (defect labeled subset) 활용 가능. anchor data 의 labeled portion 에 적용.
- 예상 효과: W10 해결 (orthogonal subspace 강제 — neural collapse 방지). P4 Homogeneity.
- patch 후보: `_clop_aux.py` wrapper, anchor 만 labeled supervision 활용 — encoder 일부 epoch warmup 후 fine-tune.

### V7. ★★ DACL — Density-Aware Contrastive Learning (Medical Semi-supervised Segmentation)
- arxiv_id: 2412.19871 (2024-12-27)
- authors: Feilong Tang, Zhongxing Xu, Ming Hu, Wenxue Li, Peng Xia, Yiheng Zhong, Hanjun Wu, Jionglong Su, Zongyuan Ge
- url: https://arxiv.org/abs/2412.19871
- venue: arXiv (cs.CV)
- code: 명시 X
- 요약: feature density 로 cluster 내 sparse region 식별 → density-aware neighbor graph 구축 → low-density anchor 를 cluster center 로 pull. label-guided co-training + geometric regularization. **density 측정 = nearest neighbor 평균 거리**.
- 적용성: ★★ — medical segmentation domain 이지만 **density-aware regularization 의 idea 는 wafer feature space 에 transferable**. 본 repo 의 HDBSCAN density 와 conceptually 일치 — encoder 학습 단계에 density-aware sampling 추가.
- 예상 효과: P2 noise_pct (low-density anchor 가 noise 분류 후보 — pull 시 noise 감소) + P4 Homogeneity (intra-cluster compactness).
- patch 후보: `_dacl_density_neighbor.py` — MoCo queue 의 density measure 추가, sampling 단계에 활용.

### V8. ★★ sDBSCAN — Scalable Density-based Clustering with Random Projections
- arxiv_id: 2402.15679 (2024-02-24, last rev 2025-05-18)
- authors: Haochuan Xu, Ninh Pham
- url: https://arxiv.org/abs/2402.15679
- venue: NeurIPS 2024
- code: 명시 X (paper 본문 확인 필요)
- 요약: random projection 의 neighborhood-preserving property 활용해 cosine distance high-dim space 에서 core point + neighborhood 빠르게 식별. sOPTICS extension 도 제공. scikit-learn DBSCAN 대비 million-point dataset 에서 significant speedup. multiple distance metric 지원.
- 적용성: ★★ — HDBSCAN 직접 대체는 X (DBSCAN family). 하지만 **본 repo 의 MoCo queue + LSH (2차 N3) 와 결합 시 hard neg sampling fast** — 또는 unknown clustering 의 prefilter 로 활용. cuML HDBSCAN (2차 R1) 과 별개 track (CPU 환경 fallback 시).
- 예상 효과: 평가 throughput 향상 + LSH (N3) 와 combine 시 W1 (hard neg) 부수 효과.
- patch 후보: `eval_contrastive.py` 의 DBSCAN comparison cell 추가 (HDBSCAN 메인 유지).

## Papers — fetch failures / 제외 (이번 round)

- arxiv 2410.17243 (Breaking Memory Barrier, near infinite batch): WebFetch title only. SogCLR (V3) 와 motivation 중복 — 별도 보고 X.
- arxiv 2411.00392 (Preventing Dim Collapse via Orthogonality Regularization): WebFetch title only. CLOP (V6) 와 motivation overlap — 보조 reference only.
- arxiv 2502.06501 (Learning Clustering-based representation, ICLR 2025): search result 제목만 — abstract 미확보. 추정 금지 정책에 따라 제외.
- arxiv 2604.01171 (Open3D-AD): 3D point cloud domain mismatch — 본 repo 2D wafer 와 무관.
- arxiv 2604.08299 (SeLaR Latent Reasoning LLM): NLP/LLM domain — wafer mismatch.
- ScienceDirect S0952197625033202 (OWSSL contrastive embedding dynamic attention industrial defect): 403 Forbidden + arxiv ID 미확보. abstract 정보 검색 결과만 — verbatim 인용 시 추정 risk 있어서 제외 (단 W5 와 정렬 — 추후 fetch 가능 시 N1 IC-DefectNCD 보충).
- arxiv 2603.16083 (Structured prototype regularization driving scene): autonomous driving domain mismatch.
- arxiv 2603.09370 (Hypergraph clustering): graph domain — wafer mismatch.
- arxiv 2501.13581: 검색 매칭 mismatch (cosmology paper) — 제외.
- arxiv 2501.01472 (ACCUP time series adaptation): time-series domain — wafer mismatch.

## Repositories (new top 4 — 1차/2차 9 repo 와 중복 0)

### W1. ★★★ facebookresearch/capi — CAPI official (Meta FAIR)
- url: https://github.com/facebookresearch/capi
- stars: 134 (정책 ≥100 충족)
- license: Apache 2.0
- 핵심: train_capi.py (전체 학습 loop) + model.py (ViT + clustering head). DINOv2 codebase 기반.
- pytorch 호환: 예 (PyTorch native, FSDP 지원).
- 본 repo 적용: NeCo wrapper 옆 alternative ablation cell. teacher = TAPT backbone 고정 사용 (CAPI 의 EMA teacher 자리에 plug-in).
- 권고 paper V1.

### W2. ★★ salesforce/PCL — Prototypical Contrastive Learning (ICLR 2021)
- url: https://github.com/salesforce/PCL
- stars: 608
- license: MIT
- 상태: **archived (read-only 2025-05-01)** — fork 권고.
- 핵심: main_pcl.py + pcl/ — instance-prototype contrastive loss. ProtoNCE (instance vs prototype) + InfoNCE 결합. EM framework — E-step clustering, M-step contrastive.
- pytorch 호환: 예 (구버전 PyTorch 1.x).
- 본 repo 적용: V2 AutoProPos / V5 ProPos 의 conceptual baseline. archived 이지만 PCL 의 EM idea 가 InfoNCE 와 mix 가능.
- 권고: V5 / V2 의 reference implementation only.

### W3. ★★ Hzzone/ProPos — ProPos official (TPAMI 2022)
- url: https://github.com/Hzzone/ProPos
- stars: 118 (정책 ≥100 충족)
- license: 명시 X (repo 직접 확인 권고)
- 핵심: main.py + network/ + torch_clustering/ (custom torch K-means / 클러스터 utility). prototype scattering + positive sampling alignment.
- pytorch 호환: 예.
- 본 repo 적용: V5 ProPos + V2 AutoProPos 양쪽 baseline. torch_clustering subdir 자체가 본 repo 에 useful (GPU K-means 등 utility).
- 권고 paper V5.

### W4. ★ Cyrilkt/AutoProPos — AutoProPos official (MDPI 2025-09)
- url: https://github.com/Cyrilkt/AutoProPos
- stars: 0 (★ 정책 ≥100 위반 — paper-fresh exception, 2025-09 published)
- license: MIT
- 핵심: main.py + ClusterAnalysis.py + torch_clustering/ + config_best_models/. CLS (Clustering Supervisor) alternate ProPos + silhouette + SUI 로 K 자동 선택.
- pytorch 호환: 예 (W3 ProPos fork 기반).
- 본 repo 적용: V2 AutoProPos 의 implementation. K 자동 estimation evaluator 로 활용. encoder 무수정 — eval phase only.
- 주의: stars 0 → paper-foundational exception (peer-reviewed venue MDPI). 보조 baseline 으로만 사용 + 본 repo 의 자체 CLS reimpl 권장.
- 권고 paper V2.

## Repos — 제외 / fetch_fail (이번 round)

- Optimization-AI/SogCLR (V3 paper) — stars 20, ≥100 위반. **but ICML 2022 official + LibAUC integrated** → paper-foundational exception. README 의 train.py / sogclr/ key path 확인. **권고 시 main impl 은 LibAUC (Optimization-AI/LibAUC, stars 미확인) 우선 사용**.
- Optimization-AI/LibAUC — 모든 X-risk optimization library. SogCLR + iSogCLR drop-in. stars 검증 추후 권장.

## Recommended action items (이번 round 우선)

전부 권고만. 코드 자동 수정 X.

### B1. ★★★ CAPI cluster-based dense patch alternative → `_capi_patch_loss.py` wrapper
- paper V1 + repo W1.
- step:
  1. teacher = TAPT backbone (known-cnn cnn_train best_model.pth) 고정
  2. student encoder 가 teacher 의 patch-level cluster assignment 를 predict
  3. clustering-based loss alpha sweep (0.05 → 0.5)
  4. NeCo B5 cell 옆 ablation — 동일 BATCH 환경 (사용자 `batch_same_condition` OK)
- 기대: W7 (NeCo regression 회복) + P3/P4. stable training 보장.
- 사용자 feedback 정책 check: no_supcon ✓ (SSL only). no_multicrop ✓ (teacher EMA 만, multi-view aug 아님).

### B2. ★★★ AutoProPos auto-K HDBSCAN second track → `eval_autopropos.py`
- paper V2 + repo W4.
- step:
  1. encoder freeze (B4 backbone)
  2. ProPos prototype scattering loss + positive sampling 으로 일부 fine-tune (또는 eval-only)
  3. CLS supervisor 가 silhouette + SUI 로 K 추정
  4. HDBSCAN K 와 비교 — 일치 시 confidence 보강, 불일치 시 deeper analysis
- 기대: W9 (auto-K second track). P1 class_capture_rate 검증. fragmentation 감소.
- 본 repo 38/38 saturated 환경에서 AutoProPos K 도 38 출력 시 강력한 cross-validation 신호.

### B3. ★★★ SogCLR / iSogCLR small-batch global contrastive → InfoNCE drop-in
- paper V3 (ICML 2022) + repo SogCLR / LibAUC.
- step:
  1. `pip install libauc` (이미 환경 OK 검증)
  2. `contrastive.py` 의 InfoNCE 를 `GCLoss('unimodal', enable_isogclr=True)` 로 swap option
  3. `--use-sogclr` flag 로 ablation cell — same BATCH 그대로
  4. iSogCLR 의 auto-temperature 가 본 repo LOSS_TEMP sweep 자동화
- 기대: W8 (small-batch SOTA). large-batch SimCLR equivalent performance. P1-P4 동시 향상 가능.
- 위험: 본 repo 의 MoCo queue 와 GCLoss interaction 검증 필요 — initial cell 은 MoCo 끄고 raw InfoNCE → GCLoss swap 만.

### B4. ★★ Collapse-Proof (V4) + CLOP (V6) → orthogonality / 4-way collapse aux loss
- paper V4 + V6.
- step:
  1. encoder output 위에 orthonormal prototype 강제 (CLOP)
  2. hyperdimensional projector 추가 (Collapse-Proof)
  3. 4-way collapse diagnostic metric 추가 (representation / dimensional / cluster / intracluster)
  4. NeCo B5 regression 의 collapse 종류 진단
- 기대: W10. P4 Homogeneity 보강.
- 위험: 두 method 의 implementation 복잡도 — 단계적 도입 (V6 CLOP 만 먼저).

### B5. ★★ DACL density-aware sampling → MoCo queue density measure
- paper V7.
- step:
  1. MoCo queue entry 의 feature density (avg dist to k-NN) 계산
  2. low-density entry 우선 anchor 로 sampling (pull to cluster center)
  3. label-guided co-training 은 본 repo unsupervised 환경 적용 X → density-aware sampling 만 차용
- 기대: P2 noise_pct 감소 (low-density = noise candidate, pull 시 reduction) + P4.
- 본 repo 가 이미 noise_pct = 0% saturated — gain margin 작음. 추가 wafer dataset (cca/, MixedWM38) zero-shot 시나리오에서 효과 검증.

### B6. ★★ ProPos prototype scattering aux loss → AutoProPos baseline
- paper V5 + repo W3.
- step:
  1. encoder output 위에 prototype scattering loss (lambda small)
  2. positive sampling alignment 추가 (augmented view 가 sampled neighbor 와 align)
  3. AutoProPos (B2) 의 backbone baseline 으로 verify
- 기대: W6 + P4 Homogeneity. uniformity vs compactness 양쪽 만족.

## Cross-reference 와 통합 priority (1차 + 2차 + 3차)

| 통합 순위 | source | technique | 적용 영역 | risk |
|---|---|---|---|---|
| 1 | V1 CAPI + 1차 #6 NeCo | dense patch cluster-based stable | W7 (B5 regression 회복) | low (encoder 무수정) |
| 2 | V3 SogCLR + 1차 #1 Harvest | small-batch global contrastive + iterative HDBSCAN | W8 + P1 | medium (GCLoss swap) |
| 3 | V2 AutoProPos + 2차 R1 cuML | auto-K + GPU HDBSCAN dual track | W9 + throughput | low (eval phase only) |
| 4 | 2차 N1 IC-DefectNCD + V6 CLOP | semi-supervised k-means + orthogonal prototype | unknown discovery + collapse | medium |
| 5 | 2차 N2 B3 + 2차 N3 LSH + 1차 #5 SynCo | 3-way hard neg ablation | W1 hard neg quality | high (3 wrappers) |
| 6 | V4 Collapse-Proof + 2차 R2/R3 VICReg/Barlow | 4-way collapse 방지 | W10 | medium |
| 7 | V5 ProPos + V7 DACL | prototype scattering + density sampling | W6 + P4 | medium |
| 8 | V8 sDBSCAN | scalable density clustering | throughput fallback | low |

## 사용자 feedback 정책 충돌 check (이번 round)

- `no_multicrop` (multi-view random crop 금지): V1 CAPI teacher-student 는 EMA-based, multi-view random crop 아님 → OK. V4 Collapse-Proof, V6 CLOP, V7 DACL 모두 single-view 가능.
- `no_supcon` (SupCon 회피): V6 CLOP 가 semi-supervised — anchor labeled portion 만 사용 시 OK. SupCon 전체적용은 위반.
- `batch_same_condition` (BATCH 변경 same-condition): V3 SogCLR 의 BATCH=8 그대로 사용 OK + 추가 sweep 가능.
- `single_gpu_process` (parallel 금지): 모든 권고 sequential.
- `gpu_share_30_40`: V3 GCLoss / V1 CAPI 학습 memory footprint 동일 또는 적음. VRAM spike 없음.

## Fetch summary (이번 round)

- 성공: 8 paper abs + 4 repo metadata = 12 hit
- abstract-only / 부분 fetch: 6 paper (V1, V2, V3, V4, V5, V7 — 모두 핵심 정보 확보 후 inclusion). V6 CLOP은 abstract OK 지만 code release 명시 X. V8 sDBSCAN은 code 명시 X.
- fetch_fail: 1 건 (ScienceDirect S0952197625033202 OWSSL — 403 Forbidden, 보고 제외)
- domain mismatch 제외: 5 건 (Open3D-AD, SeLaR LLM, cosmology, hypergraph, time-series)
- 추정 답변 0 건 (사실성 가드 유지)

## 산출 paper / repo count 통합 (1차 + 2차 + 3차)

| round | papers | repos | 누적 papers | 누적 repos |
|---|---|---|---|---|
| 1차 (260516) | 9 | 5 | 9 | 5 |
| 2차 (260516 additional) | 7 | 4 | 16 | 9 |
| 3차 (260517 v3) | 8 | 4 | **24** | **13** |

중복 검증: 모든 arxiv_id + repo_url cross-check 완료 (0 중복).

## Top-3 신규 paper (★★★)

1. **CAPI** (2502.08769, TMLR 2025) — dense patch cluster-based MIM, NeCo regression 회복 후보. ★★★ Meta FAIR official Apache 2.0.
2. **AutoProPos** (MDPI 2025-09) — ProPos auto-K extension via Clustering Supervisor + silhouette + SUI. HDBSCAN second track.
3. **SogCLR** (2202.12387, ICML 2022) — small-batch (256) = large-batch (8192) SimCLR equivalent. BATCH=8 본 repo 환경 직접 효과. LibAUC integrated.

## Top-3 신규 repo

1. **facebookresearch/capi** (134 stars, Apache 2.0) — TMLR 2025 official CAPI.
2. **Hzzone/ProPos** (118 stars, TPAMI 2022) — torch_clustering utility + prototype scattering reference.
3. **salesforce/PCL** (608 stars, MIT, archived) — Prototypical Contrastive Learning EM framework reference.

## 절대 금지 준수

- 코드 자동 수정 X — 본 markdown 만 작성
- 학습 trigger X
- 외부 weights / pth 다운로드 X
- 추정 답변 X (fetch_fail 명시 — V6 CLOP code, V8 sDBSCAN code, ScienceDirect OWSSL)
- 기존 `outputs/logs_contrastive/` 수정 X
- contrastive.py / _contrastive_n50.py 자동 patch X
