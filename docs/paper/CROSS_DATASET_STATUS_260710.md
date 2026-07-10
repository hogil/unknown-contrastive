# Cross-Dataset Contrastive Status (2026-07-10)

All rows below use raw embeddings without UMAP. FINCH is the primary clusterer and Louvain is the sanity check. `capture` is class capture, `recov` is image recovery, and `fragment` is the number of clusters divided by the target class count.

## Primary results

| dataset / protocol | recipe | clusterer | capture | recov | noise% | Comp | Hom | ARI | Sil | fragment |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WM-811K, Normal-only strict novel (7) | DINOv3 frozen | FINCH p2 | 1.000 | 0.609 | 0.0 | 0.267 | 0.435 | 0.149 | 0.057 | 4.00 |
| WM-811K, Normal-only strict novel (7) | SimCLR + queue4096 + local0.3 + ignore0.75, seed4 ep8 | FINCH p2 | 1.000 | 0.665 | 0.0 | 0.448 | 0.520 | 0.333 | -0.010 | 2.57 |
| RESISC45, same-pool self-adaptation | DINOv3 frozen | FINCH p1 | 0.978 | 0.693 | 0.0 | 0.736 | 0.769 | 0.450 | 0.171 | 1.49 |
| RESISC45, same-pool self-adaptation | SimCLR + queue4096 + ignore0.75, ep6 | FINCH p1 | 1.000 | 0.817 | 0.0 | 0.772 | 0.861 | 0.587 | 0.234 | 1.78 |
| DTD, same-pool self-adaptation | DINOv3 frozen | FINCH p1 | 0.936 | 0.578 | 0.0 | 0.641 | 0.679 | 0.324 | 0.189 | 1.60 |
| DTD, same-pool self-adaptation | SimCLR + queue4096 + ignore0.75, ep8 | FINCH p1 | 1.000 | 0.683 | 0.0 | 0.687 | 0.762 | 0.422 | 0.235 | 1.87 |
| unknown synth, controlled 20-class | DINOv3 frozen | FINCH p1 | 0.650 | 0.615 | 0.0 | 0.937 | 0.763 | 0.459 | 0.320 | 0.95 |
| unknown synth, controlled 20-class | DINOv3 + plain SimCLR, ep5 | FINCH p1 | 0.950 | 0.870 | 0.0 | 0.927 | 0.921 | 0.798 | 0.558 | 1.25 |
| unknown synth, controlled 20-class | FCMAE frozen | FINCH p1 | 0.950 | 0.925 | 0.0 | 0.975 | 0.951 | 0.894 | 0.558 | 1.05 |
| unknown synth, controlled 20-class | FCMAE + plain SimCLR, ep8 | FINCH p1 | 1.000 | 0.965 | 0.0 | 0.973 | 0.971 | 0.932 | 0.587 | 1.20 |
| unknown hard, Normal-only open set (42) | DINOv3 frozen, grade-only | FINCH p2 | 0.929 | 0.835 | 0.0 | 0.849 | 0.890 | 0.706 | 0.315 | 1.38 |
| unknown hard, Normal-only open set (42) | SimCLR + queue/local/ignore, best early ep3 | FINCH p2 | 0.833 | 0.637 | 0.0 | 0.754 | 0.765 | 0.469 | 0.239 | 1.31 |
| unknown defect-aware strict novel (32) | DINOv3 frozen, grade-only | FINCH p2 | 0.969 | 0.851 | 0.0 | 0.830 | 0.892 | 0.709 | 0.305 | 1.81 |
| unknown defect-aware strict novel (32) | DINOv3 + plain SimCLR, ep1 interim | FINCH p2 | 0.969 | 0.748 | 0.0 | 0.752 | 0.825 | 0.593 | 0.261 | 1.88 |

WM's trained ARI is `0.313 +/- 0.020` across the two available ep8 seeds (`0.293`, `0.333`). The RESISC45, DTD, and unknown rows above are selected trajectory points and still need the same multi-seed gate before a paper-level claim.

## Louvain sanity check

| dataset | frozen ARI | trained ARI | interpretation |
|---|---:|---:|---|
| WM-811K | 0.115 | 0.150 | positive but much smaller than FINCH; clusterer-sensitive |
| RESISC45 | 0.572 | 0.646 | positive and directionally consistent |
| DTD | 0.413 | 0.486 | positive and directionally consistent |
| unknown hard 42, Normal-only | 0.792 | 0.535 | clear regression |
| unknown defect-aware 32, ep1 interim | 0.785 | 0.614 | below frozen; continue trajectory before deciding |

## Why hard unknown regressed

1. The hard 42-class frozen embedding is already strong (`top-1 retrieval 0.948`, FINCH-p2 ARI `0.706`, Louvain ARI `0.792`), leaving little safe adaptation headroom.
2. Normal-only SimCLR never sees defect families. It optimizes invariance inside Normal while updating the entire backbone, so defect-specific position and shape cues are forgotten.
3. Queue, local loss, and negative filtering cannot create defect positives when the train pool contains only Normal. A Normal-only queue can increase negative diversity but cannot organize unseen defects.
4. The hard synthetic classes differ heavily by spatial position. Small crop and affine invariance can weaken top/bottom and center/ring distinctions as training continues.
5. Projection `z` is not the missing answer in the current run: defect-aware ep1 FINCH-p2 ARI is `0.456` for `z` versus `0.593` for backbone `f`.

## Active improvement protocol

1. Keep `grade_only`: background retention reduced strict-novel FINCH-p2 ARI from `0.709` to `0.646` and Louvain ARI from `0.785` to `0.752`.
2. Complete the defect-aware base for 10 epochs, saving every epoch rather than using only the final epoch.
3. Run the same split with one NV threshold at a time: `0.90`, `0.95`, `0.98`.
4. Run field-mixed all-unlabeled adaptation as a separate production protocol.
5. Run FCMAE initialization as a separate backbone row.
6. Add a multi-option combo only when its individual options beat the same frozen/base references on FINCH and Louvain.

## Artifacts

- Active log: `D:\project\unknown-contrastive\_unknown_mixed_queue_260710.log`
- Active embeddings: `D:\project\unknown-contrastive\result_grouping\_unknown_mixed260710\embeddings`
- Queue definition: `D:\project\unknown-contrastive\_unknown_mixed_queue_260710.sh`
- Queue report: `D:\project\unknown-contrastive\docs\paper\UNKNOWN_MIXED_QUEUE_260710.md`
- Historical full metrics: `D:\project\unknown-contrastive\docs\paper\ROBUSTNESS_SEEDS.md`
- Archived completed checkpoints: `E:\unknown-contrastive-archive\260710_field_robust_ckpts` (57 files, 56.83 GB; moved from D: without deletion)
