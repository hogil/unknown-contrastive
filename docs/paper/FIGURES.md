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

## 변경 history

- 2026-05-05: Fig. 2, 3, 6 자동 생성 (compose_clusters.py, plots/). Fig. 1, 4, 5, 7, 8 TBD.
