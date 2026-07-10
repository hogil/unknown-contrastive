# Unknown Mixed / Defect-Aware Queue 260710

- eval: D:\project\unknown-contrastive\data\images\unknown_eval100
- defect-aware train: D:\project\unknown-contrastive\data\images\unknown_train_defectaware_260710
- all-unlabeled train: D:\project\unknown-contrastive\data\images\unknown_train_all
- embeddings: D:\project\unknown-contrastive\result_grouping\_unknown_mixed260710\embeddings
- log: D:\project\unknown-contrastive\_unknown_mixed_queue_260710.log

### unkda_base (defect-aware strict novel)

Queue order:

1. `unkda_base`: train `D:\project\unknown-contrastive\data\images\unknown_train_defectaware_260710`, score strict novel by excluding the train defect classes.
2. `unkda_nv095`: same split plus `--nv-filter 0.95`.
3. `unkall_base`: train `D:\project\unknown-contrastive\data\images\unknown_train_all`, score field-mixed.

Defect-aware train split:

- manifest: `D:\project\unknown-contrastive\data\images\unknown_train_defectaware_260710\manifest_260710.json`
- eval: `D:\project\unknown-contrastive\data\images\unknown_eval100`
- embeddings: `D:\project\unknown-contrastive\result_grouping\_unknown_mixed260710\embeddings`
- production review output root: `D:\project\unknown-contrastive\result_grouping\_production_review_260710`

Operational note: field data does not provide a perfect "Normal" oracle. In deployment, Normal should be treated as a high-confidence background set from production metadata and conservative filtering, not as a guaranteed truth label.
