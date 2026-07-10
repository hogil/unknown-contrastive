# Unknown Mixed / Defect-Aware Queue 260710

- eval: D:\project\unknown-contrastive\data\images\unknown_eval100
- defect-aware train: D:\project\unknown-contrastive\data\images\unknown_train_defectaware_260710
- all-unlabeled train: D:\project\unknown-contrastive\data\images\unknown_train_all
- embeddings: D:\project\unknown-contrastive\result_grouping\_unknown_mixed260710\embeddings
- log: D:\project\unknown-contrastive\_unknown_mixed_queue_260710.log

Queue order:

1. `unkda_base`: train `D:\project\unknown-contrastive\data\images\unknown_train_defectaware_260710`, score strict novel by excluding the train defect classes.
2. `unkda_nv090`, `unkda_nv095`, `unkda_nv098`: same split, one NV threshold per 10-epoch run.
3. `unkall_base`: train `D:\project\unknown-contrastive\data\images\unknown_train_all`, score field-mixed.
4. `unkda_fcmae`: same defect-aware split with FCMAE initialization; no NV/queue/local combination.

Acceptance rule: do not add queue+NV or another multi-option recipe until its individual options beat `unkda_base` on the same strict-novel split.

Frozen strict-novel baseline (32 unseen classes, `grade_only`, 2026-07-10 rescore):

| clusterer | capture | recov | noise% | Comp | Hom | ARI | Sil | fragment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FINCH p2 | 0.9688 | 0.8506 | 0.0 | 0.8302 | 0.8918 | 0.7090 | 0.3045 | 1.81 |
| Louvain r6 | 0.9688 | 0.8553 | 0.0 | 0.8684 | 0.9075 | 0.7850 | 0.3775 | 1.62 |

Palette check: retaining the original PNG background reduced FINCH-p2 ARI to 0.6456 and Louvain ARI to 0.7519, so `grade_only` remains the primary preprocessing mode.

Defect-aware train split:

- manifest: `D:\project\unknown-contrastive\data\images\unknown_train_defectaware_260710\manifest_260710.json`
- eval: `D:\project\unknown-contrastive\data\images\unknown_eval100`
- embeddings: `D:\project\unknown-contrastive\result_grouping\_unknown_mixed260710\embeddings`
- production review output root: `D:\project\unknown-contrastive\result_grouping\_production_review_260710`

Operational note: field data does not provide a perfect "Normal" oracle. In deployment, Normal should be treated as a high-confidence background set from production metadata and conservative filtering, not as a guaranteed truth label.

### unkda_base (defect-aware strict novel)
