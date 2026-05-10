# Related Work

This section positions our work against four neighboring areas: (1)
self-supervised contrastive frameworks, (2) dense/patch-consistency
losses, (3) wafer-map defect clustering, and (4) open-set / anomaly
detection. We also explicitly contrast against rotation-invariant and
human-in-the-loop methods that are popular in 2024-2026 wafer defect
literature but cannot be adopted in our domain.

## 2.1 Self-supervised contrastive frameworks

The InfoNCE objective (Oord et al. 2018, arXiv:1807.03748) underlies the
modern self-supervised stack. **SimCLR** (Chen et al. 2020,
arXiv:2002.05709) demonstrated that a simple two-view augmentation
pipeline plus an MLP projection head plus InfoNCE can reach competitive
linear-probe accuracy on ImageNet. **MoCo** (He et al. 2020,
arXiv:1911.05722) decoupled the negative pool from the encoder forward
pass via a momentum-updated queue, allowing large effective negative
sets at small batch sizes. We adopt the queue mechanism directly
(`QUEUE_SIZE=4096`, sweep dead at 8192, see ITERATIONS Iter 20).

**SwAV** (Caron et al. 2020, arXiv:2006.09882) introduced multi-crop and
online cluster assignment as an alternative to pairwise InfoNCE. We
considered and rejected multi-crop in our domain: random-position crops
discard wafer-spatial identity (Edge-Top vs. Edge-Bottom or Donut vs.
Center are defined by absolute wafer position). The decision is logged
in `docs/contrastive-eval/DECISIONS.md` D-4.

**Supervised Contrastive (SupCon)** (Khosla et al. 2020) uses class
labels as a positive set. It outperforms cross-entropy classifiers on
in-distribution accuracy, but for our open-set goal — clustering must
generalize to defect classes not in the synthesis label inventory —
SupCon's label dependence is the wrong inductive bias. We use the
supervised model only as a TAPT backbone (initialization), not as a
loss signal during contrastive training. See DECISIONS D-5.

**SCHaNe** (arXiv:2308.14893) and **NV-Retriever** (NVIDIA 2024,
arXiv:2407.15831) both refine the InfoNCE negative pool: SCHaNe via
dissimilarity-weighted hard negative re-weighting on top of SupCon, and
NV-Retriever via a positive-aware false-negative filter that rejects
negatives whose similarity to the anchor exceeds a threshold. We
adopted the NV-Retriever-style threshold (`IGNORE_NEG_SIM`) and tuned
it per base configuration: 0.72 in the P2-King configuration (iter 37
SOTA), 0.65 conditional on the Quality-King base (iter 14). The full
hard-mining alpha sweep (PercPos alpha across 0.95/0.90/0.85/0.80,
iter 7-10) is reported as a dead axis (RESULTS table 10).

**ProNC** (Progressive Neural Collapse, arXiv:2505.24254, ICLR 2026)
proposes ETF prototype geometry for contrastive embeddings. It is on
our future-work list but was not adopted in the iter 0-49 cycle: the
TAPT backbone already provides a domain-aligned starting geometry, and
the iteration budget targeted hyperparameter levers first.

## 2.2 Patch consistency and dense local contrast

**DenseCL** (Wang et al. 2021, arXiv:2011.09157) extends InfoNCE from
global features to dense feature-map cells: pairs of augmented views
contribute a per-cell positive/negative loss in addition to the global
loss. We use DenseCL-style local contrast as our `USE_LOCAL=True`
component, with the cell-level weight `LOCAL_WEIGHT` swept atomically
(0.5, 0.7, 1.0, 1.5). The 0.5-to-1.0 step is one of the five levers
identified in METHOD section 3.2 (-50% defect noise, atomic).

**NeCo** (Pariza et al. 2024, arXiv:2408.11054) builds on DINOv2 and
adds a patch-neighbor consistency loss: for each spatial token in
view 1, the soft-rank of its similarities to k-nearest-neighbor tokens
must match the soft-rank computed in view 2. The original paper
demonstrates +5 to +6 mIoU on Pascal VOC dense prediction in 19 GPU
hours of fine-tuning. We adopt NeCo as our fifth lever
(`NECO_WEIGHT=0.2`, iter 37), and discover a domain-specific mechanism
(Normal-defect boundary repulsion) reported in DISCUSSION section 7.2.

**Semantic Graph Consistency** (arXiv:2406.12944, ECCV 2024) imposes
graph-level consistency between augmented views via spectral graph
Laplacians. It is a stronger constraint than NeCo but requires graph
construction at every step. We considered it as an alternative to NeCo
but did not test it: NeCo's iter-37 SOTA gives Comp 0.991 / noise 0.61%,
and adding graph-level consistency on top would conflate the lever
attribution. Listed as future work in CONCLUSION section 8.3.

## 2.3 Wafer-map defect clustering

**WM-811K** (Wu et al. 2015, IEEE TSM) is the canonical wafer-map
dataset. Subsequent self-supervised wafer work (Hwang et al. 2020, IEEE
TSM) demonstrated that contrastive pre-training improves limited-label
classifier accuracy. Our contribution differs in two ways: we target
clustering (open-set partitioning) rather than classification, and we
operate on synthetic wafers whose chip-internal defect distributions
are conditioned on WM-811K cca-class heatmaps (see METHOD section 1).

**DECOR** (arXiv:2510.03328, AAAI 2026) is a recent
**rotation-invariant** wafer-map self-supervised method: views are
rotated arbitrarily before contrastive matching, so the embedding
becomes invariant to wafer rotation. This is the **wrong inductive
bias for our setting**. The `scratch_rot` class in our 43-class
benchmark is defined by a 21-degree rotation of the `scratch` pattern;
treating these as the same class would collapse identity. We
explicitly preserve rotation: training augmentation uses
RandomRotation plus or minus 15 degrees only (within the rotation
tolerance of inspection-tool stages but well below the 21-degree
sister-class gap), and HFlip / VFlip / 180-degree rotation are forbidden
(see METHOD section 3.3). DECOR's reported gains do not transfer.

**Iterative Cluster Harvesting** (arXiv:2404.15436) iterates between
clustering, human-in-the-loop label confirmation, and re-training. The
human-in-the-loop step is the bottleneck: in a high-throughput fab
setting, an engineer cannot annotate every newly emerged cluster. Our
work runs the full pipeline without human intervention; the
Cluster-Aware Synthesis Loop sketched in CONCLUSION section 8.3
proposes an automated alternative — the cluster-analyzer agent
identifies weak classes (e.g., `Thick-Edge_fork` partial sub-style
entanglement, sister-class rotation pair), and the synthesis pipeline
auto-augments the corresponding chip-internal distribution rather than
calling for human labels.

**WaferDC** (open-source, GitHub) provides a wafer-defect classification
baseline using ResNet-50 and standard cross-entropy. We do not directly
benchmark against it in this paper because the problem framing differs
(closed 9-class classification versus open 43-class clustering) and the
metric vocabulary is incompatible (precision/recall/F1 versus
Completeness/AMI). Our `docs/contrastive-eval/METRICS.md` policy
explicitly forbids reporting classifier-style metrics for clustering
results.

**Mean Teacher + SupCon Wafer** (arXiv:2411.18533, 2024) combines
Mean Teacher self-distillation with SupCon for semi-supervised wafer
defect detection. The Mean Teacher / EMA-teacher idea is orthogonal to
our InfoNCE+NeCo composition, and could plausibly be added on top of
iter 37 (lever stacking) — flagged as future work in
CONCLUSION section 8.3. The SupCon component, as discussed above, is
incompatible with our open-set objective.

## 2.4 Open-set, anomaly, and density-aware contrastive

**Density-Aware Contrastive Learning (DACL)** (arXiv:2412.19871, 2024)
augments InfoNCE with a per-sample density estimate and uses it to
re-weight positives and negatives. It targets long-tail and
imbalanced-class regimes, which match our wafer setting (per-class
sample count ranges 12 to 80 within the avg30 anchor). DACL is a
candidate for the iter 50+ cycle. We did not adopt it inside iter 0-49
because the LR_HEAD lever (lever 2) and the IGNORE_NEG_SIM lever
(lever 3) already addressed the most aggressive imbalance behavior,
and adding a re-weighting term would have collided with the atomic
ablation policy.

**Anomaly clustering** literature (e.g., classical autoencoder
reconstruction error plus DBSCAN) is the closest non-contrastive
neighbor of our work. We chose contrastive + HDBSCAN over
reconstruction + DBSCAN for two reasons: (i) the reconstruction
objective is invariant to the defect-grade palette structure that
carries our discriminative signal (palette indices 0-7 encode defect
intensity), and (ii) HDBSCAN's variable-density assumption matches our
class-imbalance pattern better than DBSCAN's single-eps assumption.

## 2.5 Differentiation summary

| Aspect | Closest work | Our approach | Why we differ |
|---|---|---|---|
| InfoNCE base | SimCLR / MoCo | Same | Build on, do not replace |
| Hard negatives | NV-Retriever, SCHaNe | IGNORE_NEG_SIM threshold (lever 3) | Lighter than full SCHaNe pipeline; SupCon-free |
| Local feature contrast | DenseCL | LOCAL_WEIGHT atomic sweep (lever 1) | Adopted directly with sweep |
| Patch consistency | NeCo (Pariza 2024) | NECO_WEIGHT=0.2 (lever 5) | Adopted; new mechanism (boundary repulsion) reported |
| Rotation handling | DECOR (rotation-invariant) | Rotation-preserving (max 15deg) | scratch vs. scratch_rot is class identity |
| Human-in-the-loop | Iterative Cluster Harvesting | Closed-loop synthesis (future) | Fab-throughput constraint |
| Open-set | DACL (density-aware) | HDBSCAN density on contrastive embedding | Future lever stacking |
| Backbone | DINO / DINOv2 | ConvNeXtV2-base FCMAE + TAPT (frozen) | Domain-specific TAPT |

> Full reference list: `REFERENCES.md`. New citations introduced in this
> section (DECOR, Iterative Cluster Harvesting, Semantic Graph
> Consistency, DACL, Mean Teacher SupCon Wafer) are appended there in
> the same revision.
