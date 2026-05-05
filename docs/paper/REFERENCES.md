# References

## Contrastive learning

- **Oord et al. 2018** — *Representation Learning with Contrastive Predictive Coding* (CPC). [arxiv 1807.03748](https://arxiv.org/abs/1807.03748). InfoNCE loss 정의.
- **Chen et al. 2020** — *A Simple Framework for Contrastive Learning of Visual Representations* (SimCLR). [arxiv 2002.05709](https://arxiv.org/abs/2002.05709). Linear probe benchmark.
- **He et al. 2020** — *Momentum Contrast for Unsupervised Visual Representation Learning* (MoCo). [arxiv 1911.05722](https://arxiv.org/abs/1911.05722). Momentum queue.
- **Caron et al. 2018** — *Deep Clustering for Unsupervised Learning of Visual Features* (DeepCluster). [arxiv 1807.05520](https://arxiv.org/abs/1807.05520). Clustering benchmark.
- **Caron et al. 2020** — *Unsupervised Learning of Visual Features by Contrasting Cluster Assignments* (SwAV). [arxiv 2006.09882](https://arxiv.org/abs/2006.09882). Multi-crop, online clustering.
- **Khosla et al. 2020** — *Supervised Contrastive Learning* (SupCon). [NeurIPS](https://proceedings.neurips.cc/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf). 거부 (D-5).
- **Caron et al. 2021** — *Emerging Properties in Self-Supervised Vision Transformers* (DINO). [arxiv 2104.14294](https://arxiv.org/abs/2104.14294). k-NN benchmark.
- **Robinson et al. 2021** — *Contrastive Learning with Hard Negative Samples*. [ICLR / arxiv 2010.04592](https://arxiv.org/abs/2010.04592). Hard mining baseline (D-6, 후속 NV-Retriever/SCHaNe 로 업데이트 — D-14).
- **NV-Retriever** (NVIDIA 2024) — *Positive-aware Hard-Negative Mining*. [arxiv 2407.15831](https://arxiv.org/abs/2407.15831). False-negative filter, Iter 1 도입 예정 (D-14).
- **SCHaNe** — *When hard negative sampling meets supervised contrastive learning*. [arxiv 2308.14893](https://arxiv.org/abs/2308.14893). SupCon + dissimilarity weight, +3.32% few-shot.
- **ProNC** — *Progressive Neural Collapse / ETF prototype*. [arxiv 2505.24254](https://arxiv.org/abs/2505.24254). ICLR 2026.
- **Wang & Isola 2020** — *Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere*. [ICML / arxiv 2005.10242](https://arxiv.org/abs/2005.10242). Alignment + Uniformity metric.

## Clustering metrics

- **Hubert & Arabie 1985** — *Comparing partitions* (Adjusted Rand Index, ARI).
- **Strehl & Ghosh 2002** — *Cluster Ensembles* (Normalized MI, NMI).
- **Vinh et al. 2010** — *Information Theoretic Measures for Clusterings Comparison: Variants, Properties, Normalization and Correction for Chance* (AMI). [JMLR](https://www.jmlr.org/papers/volume11/vinh10a/vinh10a.pdf).
- **Rosenberg & Hirschberg 2007** — *V-Measure: A Conditional Entropy-Based External Cluster Evaluation Measure* (Homogeneity, Completeness, V-measure). [EMNLP](https://aclanthology.org/D07-1043/).
- **Fowlkes & Mallows 1983** — *A Method for Comparing Two Hierarchical Clusterings* (FMI). 디버그 부록만.
- **Rousseeuw 1987** — *Silhouettes: A graphical aid to the interpretation and validation of cluster analysis*. Journal of Computational and Applied Mathematics.
- **Bagga & Baldwin 1998** — *Algorithms for Scoring Coreference Chains* (B-Cubed). 검토 후 drop (D-2).
- **Davies & Bouldin 1979** — *A Cluster Separation Measure*. Tier 3 (Euclidean 가정 부적합).
- **Calinski & Harabasz 1974** — *A Dendrite Method for Cluster Analysis*. Tier 3.

## Clustering algorithm

- **McInnes et al. 2017** — *hdbscan: Hierarchical density based clustering* (HDBSCAN). [JOSS](https://joss.theoj.org/papers/10.21105/joss.00205).
- **Campello et al. 2013** — *Density-Based Clustering Based on Hierarchical Density Estimates*. PAKDD.

## Backbone

- **Woo et al. 2023** — *ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders*. [CVPR / arxiv 2301.00808](https://arxiv.org/abs/2301.00808). 우리 backbone.
- **He et al. 2022** — *Masked Autoencoders Are Scalable Vision Learners* (MAE). [CVPR](https://arxiv.org/abs/2111.06377). FCMAE 의 직계.

## Domain

- **Wu et al. 2015** — *Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets* (WM-811K). IEEE Transactions on Semiconductor Manufacturing. [데이터셋 페이지](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map).
- Hwang et al. 2020 — *Self-Supervised Wafer Map Defect Pattern Recognition*. IEEE TSM. (비교 reference)

## Sister repo

- `known-cnn` — supervised CNN open-set classifier. https://github.com/hogil/known-cnn

## tooling

- scikit-learn — Pedregosa et al. 2011, JMLR.
- PyTorch — Paszke et al. 2019, NeurIPS.
- timm — Wightman 2019. https://github.com/rwightman/pytorch-image-models
- **pytorch-metric-learning** (Musgrave) — production miner + SupConLoss API. https://github.com/KevinMusgrave/pytorch-metric-learning (6.3k stars, v2.9.0)
- HobbitLong/SupContrast — SupCon reference implementation. https://github.com/HobbitLong/SupContrast (3.4k stars)
