# Unknown Hard NV-Retriever Sweep 260710

- train: D:\project\unknown-contrastive\data\images\unknown_train_normal
- eval: D:\project\unknown-contrastive\data\images\unknown_eval100
- embeddings: D:\project\unknown-contrastive\result_grouping\_unknown_hard_nv260710\embeddings
- log: D:\project\unknown-contrastive\_unknown_hard_nv_sweep_260710.log
- protocol: DINOv3 no-CNN, SimCLR + one NV-Retriever threshold, 10 epochs, score backbone f and projection z.

### unk_nv085 (--method simclr --nv-filter 0.85, DINOv3 no-CNN, unknown hard)

Manual early check before switching GPU to mixed/defect-aware queue:

- embedding: `D:\project\unknown-contrastive\result_grouping\_unknown_hard_nv260710\embeddings\unk_nv085_ep1.npy`
- projection: `D:\project\unknown-contrastive\result_grouping\_unknown_hard_nv260710\embeddings\unk_nv085_ep1_proj.npy`
- grouping output: `D:\project\unknown-contrastive\result_grouping\_production_review_260710\normal_only_nv085_ep1_finch_p2`

Backbone `f`, ep1:

| method | P1 capture | recov | P2 noise% | P3 Comp | P4 Hom | ARI | Sil | k(전체/클래스수/noise) | 파편비 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| finch_p2 | 0.9762 | 0.7376 | 0.0 | 0.7990 | 0.8443 | 0.5832 | 0.3160 | 61/42/0 | 1.45 |
| louvain_res6 | 0.9762 | 0.7900 | 0.0 | 0.8250 | 0.8749 | 0.6807 | 0.3618 | 59/42/0 | 1.40 |

Projection `z`, ep1:

| method | P1 capture | recov | P2 noise% | P3 Comp | P4 Hom | ARI | Sil | k(전체/클래스수/noise) | 파편비 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| finch_p2 | 0.9048 | 0.5514 | 0.0 | 0.6744 | 0.7137 | 0.3873 | 0.1813 | 63/42/0 | 1.50 |
| louvain_res6 | 0.8810 | 0.5840 | 0.0 | 0.6859 | 0.7329 | 0.4293 | 0.2897 | 58/42/0 | 1.38 |

Interpretation: Normal-only NV raises capture relative to frozen but lowers recov/ARI; it is useful as a background-filter probe, not as the main defect-grouping adaptation. The long five-threshold sweep was stopped after `unk_nv085` ep2 to prioritize mixed/defect-aware runs.
