# Figures

paper 에 들어갈 figure list + caption + 위치.

## Fig. 1 — Pipeline overview (TBD)

flowchart: 합성 wafer → ConvNeXtV2 backbone (frozen) → projection head → InfoNCE loss
+ HDBSCAN → cluster → composite map.

위치: TBD (직접 그려야 함).

## Fig. 2 — Cluster medoid samples

per-cluster medoid PNG 모음. cluster 0~38 의 single representative wafer.

위치: `outputs/logs_contrastive/overall/cluster_summary/cluster_*_medoid_dist*.png`

## Fig. 3 — Cluster composite (binary + grademean, 6400 × 6400)

각 cluster 의 K=20 medoid aggregate PNG. 두 가지 방식:
- (a) binary: pixel 별 defect 발생 빈도, 0=white → 1=red
- (b) grademean: pixel 별 평균 grade, 0=white → 1=red

위치: `outputs/logs_contrastive/overall/cluster_summary/composite/{binary,grademean}/cluster_*.png`

caption 후보: "Per-cluster K=20 medoid aggregate composite. Binary (a) shows defect
frequency, grademean (b) shows defect intensity. Invalid pixels (all 20 wafers grade≥8)
fill white."

## Fig. 4 — t-SNE / UMAP of embeddings (TBD)

8,357 wafer embedding 의 2D projection, GT class color.

위치: TBD (`compute_tsne.py` 별도 작성 필요).

## Fig. 5 — Loss + alignment + uniformity trajectory (TBD, Iter 2 도입 후)

학습 epoch vs (G loss, Q loss, alignment, uniformity) trajectory.

위치: 학습 도중 자동 plot (`run.log` parse) — Iter 2 에 도입.

## Fig. 6 — Cluster size histogram

39 cluster 의 size 분포 histogram. + class purity heatmap.

위치: `outputs/logs_contrastive/overall/eval/plots/cluster_*.png` (자동 생성됨).

## Fig. 7 — Per-class fragmentation barplot

x: class (38), y: n_clusters (1 best). 4 split class 강조.

위치: TBD (별도 plot 작성).

## Fig. 8 — GMM bimodality 시각 (Full_scratch_rot 진단)

Full_scratch_rot 200 wafer 의 pairwise cosine similarity distribution + 1-component
vs 2-component GMM fit. BIC 차이 시각화.

위치: TBD (Iter 0 진단의 보조 figure).

## Fig. F-N7-lattice — 4-component lattice diagram (N7 evidence)

paper DISCUSSION §7.10 (N7 Component Dependency Hierarchy) 핵심 시각화. 4-component
on/off 16-cell lattice 중 12 cell 측정 (iter 67-77).

- **rows** (cfg label, 4-bit Local/Queue/NEG/NeCo): `0000` (B0), `1000` (B1), `1*000`
  (B2 LW=1.0), `0001` (iter 69), `1*001` (iter 67), `0011` (iter 74), `1*100` (B3),
  `1*101` (iter 68), `0101` (iter 75 ★), `1*110` (B4), `1*111` (B5), `0111`
  (iter 70 ★★ NEW SOTA), `0111-w0.4` (iter 77).
- **columns** (metric): ARI, defect-noise%, Silhouette (cosine), n_clusters.
- **plot type**: matrix heatmap (color = metric value, annotated numeric) + binary
  on/off table 동행.

source iter: ITERATIONS.md iter 67-77 (N7 lattice exploration, 2026-05-12).

핵심 시각 cue:
- iter 69 == iter B1 == iter 74 ARI 0.8514 4자리 동일 — NeCo ≡ DenseCL ≡ NeCo+NEG-noQueue
- iter 70 (no Local, NeCo + Queue + NEG) ARI **0.8797** = peak
- B5 (Local + everything) ARI 0.8564 < B4 0.8605 — NeCo isolated negative
- iter 77 NeCo=0.4 Sil 0.801 highest, ARI back to B4 level

caption 후보 (★ N1 v7 FINAL revised 2026-05-13): "Four-component lattice (Local
DenseCL × MoCo Queue × NEG filter × NeCo) mapped over 12 measured configurations.
Identical aggregate HDBSCAN ARI on NeCo-only (iter 69), Local-only (B1), and
NeCo+NEG-no-Queue (iter 74) demonstrates aggregate-scope **patch-neighbor
equivalence** and NEG-requires-Queue dependency (N7). Per-class purity flips at
single-seed=42 Agglomerative Ward K=42 (RESULTS §16) do NOT propagate to
multi-seed avg (RESULTS §17b): NEW (NeCo only) > B5 (Local + NeCo) on Agglo Ward
K=42 multi-seed (0.9014 ± 0.022 vs 0.8920 ± 0.062, B5 std 2.8× higher).
**v7 single-cfg recipe = iter 70 NEW for both Frontier 1 (unknown-K HDBSCAN) and
Frontier 2 (known-K Agglo Ward K=42).** Per-class complementarity preserved as
single-seed observation only; v6 dual-cfg recipe retracted."

위치: TBD (별도 plot 작성 — `plots/fig_n7_lattice.png`). 예상 size 1200×900.

## Fig. F-N7-multiseed-Sil — ★ DEPRECATED 2026-05-12 (Sil +30% retraction)

> **Deprecated 2026-05-12**: 이전 design 의 "NEW cfg gains +30% Silhouette robustly
> (0.7941 vs 0.6104)" claim 은 HDBSCAN protocol mismatch artefact (paper N8).
> apples-to-apples (eom + mcs=12 + ms=3, defect-only) 재측정 후 B5 Sil = 0.7988,
> NEW Sil = 0.7860 → **equivalent within seed variance (−0.013)**. Figure 자체 폐기,
> 대체 figure 는 N1 v5 Normal-cluster consolidation 시각화로 재설계 예정.

이전 design (historical reference):
- bars: B5 mixed-protocol 0.6104; NEW eom+ms=3 0.7941 — cross-protocol cannot compare
- "+30% robust" 표현 retract

향후 figure 재설계 후보:
1. apples-to-apples 3-seed Sil bar (B5/B4/NEW 모두 eom+mcs=12+ms=3)
2. Normal/defect boundary 시각화 (Normal noise 77.7% → 14.1%)
3. full-set ARI bar (B5 0.69 vs NEW 0.83) — N1 v5 primary headline

caption 후보 (revised): "Apples-to-apples Silhouette comparison (eom + mcs=12 + ms=3,
defect-only) for B5 (iter-37 cfg, 5-component) vs NEW (iter 70, 4-component, no Local).
Single-seed=42: B5 0.7988, NEW 0.7860 — equivalent within seed variance. The genuine
NEW-vs-B5 differentiator is full-set ARI 0.83 vs 0.69 (Normal-cluster consolidation,
paper N1 v5), not defect-cluster geometry."

위치: TBD — 재설계 후 `plots/fig_n1v5_apples_sil.png` 또는 `plots/fig_n1v5_normal_consolidation.png`.

## Fig. F-N7-neco-pareto — NeCo weight × (ARI, Silhouette) Pareto

NEW cfg (Global + Queue + NEG, no Local) 위 NeCo weight sweep — 4 points (seed=42).

- **x-axis**: NeCo weight ∈ {0.0, 0.1, 0.2, 0.4}
- **y-axis (twin)**: ARI (왼쪽), Silhouette (오른쪽)
- **data points (★ corrected 2026-05-12, apples-to-apples Sil)**:
  - w=0.0 (= B4): ARI 0.8605, Sil **0.8012** (apples)
  - w=0.1: ARI 0.860, Sil 0.801 (iter 76 single, mixed-protocol)
  - w=0.2 (= iter 70 NEW): ARI **0.8797**, Sil 0.7860 (apples)
  - w=0.4 (iter 77): ARI 0.8605, Sil 0.8012 (apples)

source iter: ITERATIONS.md iter 70 (w=0.2), iter 76 (w=0.1), iter 77 (w=0.4), B4
baseline (w=0).

핵심 시각 cue (★ revised 2026-05-12):
- ARI inverse-U with peak at w=0.2 (preserved)
- ~~Silhouette monotonic increasing with NeCo weight~~ **RETRACTED** — apples Sil
  pattern is **non-monotonic** (B4 0.8012, w=0.2 NEW 0.7860, w=0.4 0.8012). NeCo at
  ARI-optimal w=0.2 has slightly lower Sil than its neighbors.
- ~~Pareto frontier~~ **retracted** (no monotonic Sil ascent)

caption 후보 (★ revised): "NeCo weight sweep (w ∈ {0.0, 0.1, 0.2, 0.4}) over the NEW
cfg base. ARI peaks (inverse-U) at w=0.2. Under apples-to-apples HDBSCAN protocol
(eom + mcs=12 + ms=3, defect-only) Silhouette does NOT show monotonic ascent — B4
(w=0) Sil 0.8012, NEW (w=0.2) Sil 0.7860, w=0.4 Sil 0.8012. The previously-claimed
geometry-vs-partitioning Pareto frontier is retracted (paper N8 protocol-mismatch
artefact). w=0.2 remains the recommended operating point on ARI grounds."

위치: TBD (별도 plot — `plots/fig_n7_neco_pareto.png`). 예상 size 800×600.

## 변경 history

- 2026-05-05: Fig. 2, 3, 6 자동 생성 (compose_clusters.py, plots/). Fig. 1, 4, 5, 7, 8 TBD.
- 2026-05-12: F-N7-lattice / F-N7-multiseed-Sil / F-N7-neco-pareto 3 figures added
  for N7 (Component Dependency Hierarchy) DISCUSSION §7.10 evidence. all TBD plots.
