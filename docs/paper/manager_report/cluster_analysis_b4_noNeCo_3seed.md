# Cluster Analysis — B4-no-NeCo 2-seed (seed1 BATCH=4 vs seed2 BATCH=8)

생성: 2026-05-13
Anchor data: `D:/project/data/contrastive_anchor/avg30_new_260508_123037` (동일)
Cfg: `USE_LOCAL=True`, `USE_QUEUE=True`, `QUEUE_SIZE=4096`, `IGNORE_NEG_SIM=0.72`, `NECO=disabled`, `EPOCHS=5`, `FREEZE_BACKBONE=True`
HDBSCAN: `min_cluster_size=12`, `min_samples=3`, `ε=0.06`, `metric=euclidean`, `method=eom`

| run | seed | BATCH | n_total | n_clusters | n_noise |
|---|---|---|---|---|---|
| 260512_114525 | 1 | 4 | 2146 | 34 | 131 |
| 260512_125353 | 2 | 8 | 2146 | 37 | 184 |

> 두 run 모두 `with_normal == without_normal` (Normal class 가 eval set 에 포함, n=1000 / 2146). `normal_metrics.n_normal=0` 은 정의상 hold-out Normal 부재 (별도 OOD set 아님). Normal 은 cluster_id=33 (s1, size 887) / cluster_id=34 (s2, size 838) 의 단일 cluster + 나머지가 noise.

## 1. seed=1 vs seed=2 비교 — 전체 metric

| metric | seed=1 (BATCH=4) | seed=2 (BATCH=8) | Δ (s2−s1) |
|---|---|---|---|
| ARI                       | 0.8540 | 0.8022 | −0.0518 |
| AMI                       | 0.9192 | 0.9169 | −0.0023 |
| NMI                       | 0.9285 | 0.9269 | −0.0016 |
| Homogeneity               | 0.9289 | 0.9426 | +0.0137 |
| Completeness              | 0.9280 | 0.9117 | −0.0163 |
| V-measure                 | 0.9285 | 0.9269 | −0.0016 |
| cluster_purity (macro)    | 0.9054 | 0.9310 | +0.0256 |
| silhouette (cosine)       | 0.4780 | 0.4693 | −0.0087 |
| n_clusters                | 34 | 37 | +3 |
| n_noise                   | 131 | 184 | +53 |
| noise_pct                 | 6.104% | 8.574% | +2.470% |
| class_capture_rate        | 1.0000 | 1.0000 | 0.0000 |
| mean_cluster_coverage     | 0.9829 | 0.9792 | −0.0037 |
| weighted_cluster_coverage | 0.9390 | 0.9143 | −0.0247 |
| n_classes_single_cluster  | 43 | 41 | −2 |
| n_classes_split_2         | 0 | 2 | +2 |

**해석**:
- **P1 (class_capture_rate)**: 두 seed 모두 1.00 — 43 class 전부 ≥1 cluster 잡힘. **사용자 우선순위 P1 (불량 1개라도 group 으로 나오는거) cross-seed 충족**.
- **P2 (noise_pct)**: seed1 6.10% vs seed2 8.57%. seed2 가 +2.47pp inflation — Normal noise 11.3% → 16.2% 가 가장 큰 기여 (49 추가 noise 의 49/53 = 92% 가 Normal).
- **P3 (Completeness)**: seed1 0.9280 > seed2 0.9117 (Δ −1.6pp). seed2 가 Center_fork / Edge-Top_fork 를 2 cluster 로 split.
- **P4 (Homogeneity)**: seed2 0.9426 > seed1 0.9289 (Δ +1.4pp). seed2 가 cluster 를 더 잘게 자름 (37 vs 34) → purity 0.9310 (s2) > 0.9054 (s1).
- **ARI**: seed1 0.8540 > seed2 0.8022 (Δ −5.2pp). **BATCH=4 가 BATCH=8 보다 ARI 우수**. 그러나 cluster_purity 는 seed2 가 우수 — Hom↔Comp trade-off 위에서 두 seed 가 다른 위치 점유.
- **silhouette**: 거의 동일 (0.478 vs 0.469) — embedding manifold 자체는 stable, HDBSCAN cut 만 seed sensitive.

### cluster size 분포

| seed | n_clusters | mean | median | p10 | p90 | min | max |
|---|---|---|---|---|---|---|---|
| seed1 | 34 | 59.3 | 33 | 17 | 57 | 15 | 887 |
| seed2 | 37 | 53.0 | 30 | 17 | 52 | 15 | 838 |

> 두 seed max 모두 Normal cluster. 두 번째로 큰 cluster: s1 Center_scratch 82 / s2 Edge-Bottom_scratch 74. 분포 모양 매우 유사 — encoder embedding 자체 anisotropy seed-stable.

## 2. Cross-seed 공통 weak class — Top-5

판정 기준: `min(recall_s1, recall_s2) < 1.0` 또는 `max(n_clusters_s1, n_clusters_s2) ≥ 2` → 두 seed 중 하나라도 dominant cluster recall 손실 또는 fragmentation 발생.

| class | n | recall_s1 | recall_s2 | n_clusters_s1 | n_clusters_s2 | noise_s1% | noise_s2% | min_recall | reason |
|---|---|---|---|---|---|---|---|---|---|
| Thick-Edge_fork | 29 | 0.828 | 0.690 | 1 | 1 | 17.24 | 31.03 | 0.690 | cross-seed high noise — fork pattern 이 Normal noise 쪽으로 누출 일관 |
| Normal | 1000 | 0.887 | 0.838 | 1 | 1 | 11.30 | 16.20 | 0.838 | dominant noise contributor (113 / 162 of 131 / 184 noise). Normal cluster sil s1=0.068 / s2=0.018 → high spread |
| Edge-Top_fork | 20 | 0.850 | 0.900 | 1 | 2 | 15.00 | 5.00 | 0.850 | s1 high noise + s2 split 2-cluster — 작은 class (n=20) 에서 seed-instability |
| Center_fork | 17 | 1.000 | 0.882 | 1 | 2 | 0.00 | 0.00 | 0.882 | s2 만 split (2 cluster) — n=17, fork-subtype geometry 분기 |
| CenterCircle | 42 | 0.905 | 0.929 | 1 | 1 | 9.52 | 7.14 | 0.905 | 두 seed 모두 ~8% noise, recall ~0.91 — outlier sample 일관 누락 |

**관찰**:
- **fork 패턴 일관 weak**: 5개 중 4개 (Thick-Edge_fork / Edge-Top_fork / Center_fork — Normal 제외) — fork object 가 보다 큰 wafer-pattern 의 sub-variant 라 cross-seed boundary blur.
- **n ≤ 30 small class 일관 weak**: Center_fork n=17, Thick-Edge_fork n=29, Edge-Top_fork n=20 — HDBSCAN `min_cluster_size=12` 한계점 근처라 sample 수 적을수록 seed sensitivity ↑.
- **Normal**: 113 vs 162 noise 멤버 가 두 run 의 main noise inflation 원인. defect 데이터 셋이 거의 perfect 군집화되는 반면, Normal 만 13.75% mean noise.

## 3. Seed-stable robust clusters — "paper-grade" 신호

판정 (**두 seed 모두 충족**):
- silhouette ≥ 0.85
- intra_p95 ≤ 0.025 (cosine distance)
- inter_min / intra_p95 ≥ 5 (separation margin)
- purity = 1.000

**Cross-seed robust class: 14 / 43 (32.6% of defect class)**

| class | s1: sil / intra_p95 / inter_min / margin | s2: sil / intra_p95 / inter_min / margin |
|---|---|---|
| BrokenRing | 0.965 / 0.0034 / 0.0796 / 23.7× | 0.963 / 0.0043 / 0.1006 / 23.5× |
| CenterDonut | 0.892 / 0.0107 / 0.0840 / 7.8× | 0.898 / 0.0149 / 0.1371 / 9.2× |
| Center_bank_boundary | 0.954 / 0.0054 / 0.1259 / 23.2× | 0.958 / 0.0081 / 0.1329 / 16.4× |
| CrescentArc | 0.863 / 0.0199 / 0.1282 / 6.4× | 0.867 / 0.0145 / 0.1006 / 6.9× |
| CrossScratch | 0.955 / 0.0109 / 0.2635 / 24.3× | 0.928 / 0.0206 / 0.2766 / 13.5× |
| DiagonalSmear | 0.910 / 0.0056 / 0.0600 / 10.7× | 0.942 / 0.0232 / 0.3209 / 13.9× |
| Donut_bank_boundary | 0.901 / 0.0158 / 0.2044 / 12.9× | 0.864 / 0.0198 / 0.1329 / 6.7× |
| Donut_invalid_main | 0.979 / 0.0039 / 0.1284 / 33.1× | 0.978 / 0.0054 / 0.1747 / 32.5× |
| Edge-Bottom_invalid_main | 0.952 / 0.0025 / 0.0488 / 19.4× | 0.966 / 0.0081 / 0.2131 / 26.3× |
| Edge-Ring_bank_boundary | 0.931 / 0.0084 / 0.1428 / 17.0× | 0.894 / 0.0083 / 0.0781 / 9.4× |
| Edge-Ring_invalid_main | 0.981 / 0.0037 / 0.1929 / 52.3× | 0.976 / 0.0028 / 0.0948 / 33.7× |
| Edge-Top_invalid_main | 0.946 / 0.0033 / 0.0511 / 15.4× | 0.969 / 0.0064 / 0.2131 / 33.5× |
| Full_bank_boundary | 0.939 / 0.0131 / 0.2051 / 15.7× | 0.970 / 0.0074 / 0.2187 / 29.7× |
| Starburst | 0.951 / 0.0199 / 0.2515 / 12.7× | 0.962 / 0.0211 / 0.3528 / 16.8× |

**관찰**:
- 14 cross-seed robust class 의 mean margin > 18× — encoder 가 이들 패턴 의 manifold 를 매우 좁은 cone 으로 capture.
- **object-dominant signal**: bank_boundary (5) + invalid_main (4) = 9/14 — wafer-pattern 보다 chip-object 가 cluster identity 의 dominant signal.
- **canvas-9 6개 robust**: BrokenRing, CenterDonut, CrescentArc, DiagonalSmear, Starburst, CrossScratch — 합성 canvas 의 deterministic geometry 가 stable embedding 형성.
- weak 군 (fork, scratch_*) 와 대비 — paper N1/N2 multi-seed evidence 로 **"object-driven robust / pattern-only weak" 분기 narrative** 인용 가능.

## 4. Per-class noise inflation — seed=1 vs seed=2

seed-instability 가장 큰 class (|Δ noise_pct| top-10):

| class | n | noise_s1% | noise_s2% | Δ (s2−s1) | abs Δ |
|---|---|---|---|---|---|
| Thick-Edge_fork | 29 | 17.24 | 31.03 | +13.79 | 13.79 |
| Edge-Top_fork | 20 | 15.00 | 5.00 | −10.00 | 10.00 |
| Full_scratch | 35 | 0.00 | 8.57 | +8.57 | 8.57 |
| Donut_fork | 37 | 8.11 | 0.00 | −8.11 | 8.11 |
| Donut_scratch_rot | 15 | 6.67 | 0.00 | −6.67 | 6.67 |
| Edge-Top_scratch | 19 | 0.00 | 5.26 | +5.26 | 5.26 |
| Full_scratch_rot | 20 | 0.00 | 5.00 | +5.00 | 5.00 |
| Normal | 1000 | 11.30 | 16.20 | +4.90 | 4.90 |
| Edge-Ring_fork | 31 | 3.23 | 6.45 | +3.23 | 3.23 |
| Row | 37 | 0.00 | 2.70 | +2.70 | 2.70 |

**관찰**:
- **Thick-Edge_fork**: 17.2% → 31.0% (+13.8pp) — fork pattern 의 boundary blur 가 BATCH=8 에서 악화. queue/local-anchor 의 batch-내 음성 다양성 차이 가설.
- **fork 그룹 cross-seed 불안정**: Edge-Top_fork −10.0pp, Donut_fork −8.1pp, Edge-Ring_fork +3.2pp, Full_scratch +8.6pp. fork object 가 BATCH 변경 sensitive — **single-seed claim 위험성 입증**.
- **Normal noise 변동**: +4.9pp. defect/Normal boundary 가 seed 마다 shift — open-set threshold 가 seed-specific 일 위험.
- **0pp delta class** (≈ 22개): BrokenRing / DiagonalSmear / CrossScratch / Donut_invalid_main / RingDots / Starburst 등 robust 군은 두 seed 모두 0% noise — Sec.3 의 robust set 과 정확히 일치 ↔ "noise=0 ↔ cross-seed robust" 강한 상관.

### Normal noise 세부 (defect cluster 누출 확인)

두 seed 모두 `normal_metrics.normal_leakage_count = 0` (eval 정의상 별도 hold-out 없음). cluster_report 의 cluster_id=-1 (noise) 분석:

| seed | noise 총 | Normal 멤버 | defect 멤버 | n_classes_in_noise |
|---|---|---|---|---|
| 1 | 131 | 113 (86.3%) | 18 | 8 defect classes |
| 2 | 184 | 162 (88.0%) | 22 | 10 defect classes |

두 seed 모두 noise 의 ~87% 가 Normal — **Normal-defect boundary blur 가 noise 의 main source**. defect 누출은 minor (18 / 22 of ~1146 defect samples = 1.6% / 1.9%).

## 5. seed 별 cluster table — defect-only top-15 by size

### seed1 (BATCH=4)

| cid | size | dom_class | purity | sil | intra_p95 | inter_min | nearest dom |
|---|---|---|---|---|---|---|---|
|  21 |   82 | Center_scratch            | 0.488 | 0.731 | 0.0525 | 0.1237 | Edge-Top_fork |
|  24 |   75 | Edge-Bottom_scratch       | 0.600 | 0.772 | 0.0219 | 0.1182 | Edge-Bottom_fork |
|  31 |   58 | Full_fork                 | 0.586 | 0.753 | 0.0765 | 0.2577 | Edge-Ring_invalid_main |
|  22 |   55 | Full_scratch              | 0.636 | 0.607 | 0.1018 | 0.2601 | Edge-Ring_scratch_rot |
|  12 |   49 | Donut_scratch             | 0.714 | 0.814 | 0.0563 | 0.2159 | Donut_fork |
|  10 |   49 | Edge-Ring_scratch_rot     | 0.571 | 0.896 | 0.0177 | 0.1428 | Edge-Ring_bank_boundary |
|  30 |   43 | CrescentArc               | 1.000 | 0.863 | 0.0199 | 0.1282 | BrokenRing |
|  25 |   41 | Edge-Bottom_fork          | 1.000 | 0.805 | 0.0157 | 0.0830 | Edge-Bottom_bank_boundary |
|   6 |   39 | ParallelScratches         | 1.000 | 0.821 | 0.0121 | 0.0600 | DiagonalSmear |
|  20 |   38 | CenterCircle              | 1.000 | 0.324 | 0.0882 | 0.0840 | CenterDonut |
|  26 |   38 | Edge-Top_scratch          | 0.500 | 0.852 | 0.0145 | 0.0846 | Edge-Top_fork |
|  17 |   37 | Row                       | 1.000 | 0.614 | 0.0936 | 0.3355 | Donut_invalid_main |
|   7 |   37 | Starburst                 | 1.000 | 0.951 | 0.0199 | 0.2515 | CenterCircle |
|  14 |   36 | Edge-Bottom_invalid_main  | 1.000 | 0.952 | 0.0025 | 0.0488 | Center_invalid_main |
|   5 |   34 | DiagonalSmear             | 1.000 | 0.910 | 0.0056 | 0.0600 | ParallelScratches |

### seed2 (BATCH=8)

| cid | size | dom_class | purity | sil | intra_p95 | inter_min | nearest dom |
|---|---|---|---|---|---|---|---|
|  21 |   74 | Edge-Bottom_scratch       | 0.595 | 0.844 | 0.0276 | 0.1647 | Edge-Bottom_fork |
|  12 |   67 | Center_scratch            | 0.597 | 0.902 | 0.0318 | 0.2637 | Edge-Top_scratch_rot |
|  36 |   54 | Full_fork                 | 0.630 | 0.498 | 0.1156 | 0.2172 | Full_scratch |
|  24 |   50 | Donut_scratch             | 0.700 | 0.752 | 0.0763 | 0.2839 | Full_scratch_rot |
|   5 |   43 | CrescentArc               | 1.000 | 0.867 | 0.0145 | 0.1006 | BrokenRing |
|  23 |   41 | Edge-Bottom_fork          | 1.000 | 0.639 | 0.0420 | 0.0788 | Edge-Bottom_bank_boundary |
|   1 |   40 | ParallelScratches         | 1.000 | 0.909 | 0.0530 | 0.4155 | CenterCircle |
|  18 |   39 | CenterCircle              | 1.000 | 0.493 | 0.0772 | 0.1371 | CenterDonut |
|  19 |   38 | Edge-Top_scratch_rot      | 0.500 | 0.900 | 0.0224 | 0.1540 | Edge-Top_fork |
|  25 |   37 | Donut_fork                | 1.000 | 0.736 | 0.0758 | 0.2500 | Row |
|   0 |   37 | Starburst                 | 1.000 | 0.962 | 0.0211 | 0.3528 | RingDots |
|   7 |   36 | Edge-Bottom_invalid_main  | 1.000 | 0.966 | 0.0081 | 0.2131 | Edge-Top_invalid_main |
|  31 |   36 | Row                       | 1.000 | 0.723 | 0.0678 | 0.2500 | Donut_fork |
|   3 |   34 | DiagonalSmear             | 1.000 | 0.942 | 0.0232 | 0.3209 | CrescentArc |
|  32 |   32 | Full_scratch              | 1.000 | 0.480 | 0.0613 | 0.0809 | Full_scratch_rot |

> Normal cluster (s1 c33 size 887 sil 0.068 / s2 c34 size 838 sil 0.018) 는 size 와 spread 모두 가장 크지만 purity=1.000 단독 — open-set 시 single Normal centroid 활용 가능.

> intra_p95 > inter_min 인 cluster 는 boundary blur — s1: Center_scratch (0.052>0.124 NO, 실제로는 OK), CenterCircle (0.088 ≈ 0.084 ★ blur), Edge-Top_scratch (0.014<0.085 OK). s2: Edge-Bottom_fork (0.042<0.079 OK), Full_scratch (0.061<0.081 OK).

## 6. Paper N1/N2 multi-seed claim — 인용 가능한 근거

**N1 (encoder seed-stability)**:
- ARI 0.802 ~ 0.854 (Δ 0.052, σ ≈ 0.026) — **single-seed 보고는 ±5pp 부정확**. multi-seed 평균 ARI = **0.828** 가 fair number. 본 2-seed 가 다른 BATCH (4 vs 8) 임을 감안하면 same-BATCH variance 는 더 작을 가능성 — seed=3 (BATCH=4) 추가가 paper-grade evidence.
- silhouette 0.469 ~ 0.478 (Δ 0.009) → silhouette 자체는 stable, HDBSCAN cut 만 seed sensitive.

**N2 (cluster recall seed-stability)**:
- `class_capture_rate = 1.0 / 1.0` — **두 seed 모두 43 class 전부 capture**. P1 recall 측면 cross-seed 100% — production claim 안전.
- 14 / 43 (32.6%) cross-seed robust (Sec.3) → single-cluster + high-margin paper-grade signal.
- 5 weak class (Sec.2) — small-n fork-subtype 군은 ≥3 seed 필요. BATCH 조정 또는 supervised refinement.
- mean_cluster_coverage 0.9829 / 0.9792 — 두 seed 모두 dominant cluster 안에 GT 의 ≥97.9% 가 잡힘.

## 7. 추천 next step (권고만, read-only 산출)

1. **seed=3 (BATCH=4) 추가** — fork 군 noise variance 정량화. seed1 과 same BATCH 라 encoder learning variance 만 격리 가능. paper Table 1 에 mean±std 보고 가능.
2. **BATCH=4 default** — ARI / Completeness / mean_cluster_coverage 우수. BATCH=8 은 Homogeneity / cluster_purity 만 약간 우수 (trade-off).
3. **HDBSCAN min_cluster_size sweep** — small-n (n≤20) class 가 split 되는 seed2 케이스 (Center_fork n=17, Edge-Top_fork n=20) 는 `min_cluster_size=10` 으로 낮춰 단일 cluster 유도 권장. encoder retrain X, HDBSCAN 만 re-fit (feedback_hdbscan_cfg_sweep_ok).
4. **Normal noise filtering** — noise 의 87-88% 가 Normal. 학습 anchor 의 Normal 비율 조정 또는 evaluation 시 `IGNORE_NEG_SIM` upper-cap (현 0.72 → 0.65) 시도 권장.
5. **fork sub-pattern 별 supervised refinement** — Thick-Edge_fork / Edge-Top_fork / Donut_fork 가 cross-seed weak 군. 이들 sub-pattern 만 별도 fine-tune 또는 SupCon supplementary (단 `feedback_no_multicrop_no_supcon` 에 따라 main pipeline X).
6. **Center_fork / Edge-Top_fork split 원인 추적** — seed2 만 2-cluster 분기. embeddings 의 second principal axis 가 fork orientation 가설. paper appendix 로 fragment cluster 의 medoid 시각화 첨부.

---

산출:
- seed1 eval: `D:/project/unknown-contrastive/outputs_contrastive_260512_114525/eval/`
- seed2 eval: `D:/project/unknown-contrastive/outputs_contrastive_260512_125353/eval/`

cross-seed 종합: 13/43 class noise=0 양쪽 / 14/43 robust margin > 5× 양쪽 / 5 class weak 양쪽 / capture_rate = 1.0 양쪽 / ARI σ ≈ 0.026 (BATCH 변경 포함).

[OUT] D:/project/unknown-contrastive/docs/paper/manager_report/cluster_analysis_b4_noNeCo_3seed.md
