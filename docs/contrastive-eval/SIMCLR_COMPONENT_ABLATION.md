# SimCLR Component Ablation

- train folder: `D:\project\unknown-contrastive\data\images\wm811k_novel_disjoint_v1\cnn_seen_train`
- eval folder: `D:\project\unknown-contrastive\data\images\wm811k_novel_disjoint_v1\novel_eval`
- backbone: `hf_hub:timm/convnext_base.dinov3_lvd1689m`
- eval embedding: backbone feature
- primary metric: k-means(k=3) ARI on held-out novel classes

| Stage | Method | ARI | NMI | AMI | top1 | k5 | k9 | dist ratio | HDB ARI | noise |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | Base SimCLR | 0.4340 | 0.4354 | 0.4347 | 94.13% | 91.97% | 91.11% | 0.4106 | 0.4297 | 63.40% |
| C1 | + MoCo EMA queue | 0.5000 | 0.4671 | 0.4664 | 92.67% | 90.25% | 89.22% | 0.4615 | 0.1670 | 56.13% |
| C2 | + neg-ignore | 0.4693 | 0.4229 | 0.4222 | 93.53% | 91.41% | 89.98% | 0.4615 | 0.4745 | 56.07% |
| C3 | + local grid | 0.5160 | 0.4803 | 0.4797 | 93.53% | 91.79% | 90.93% | 0.5262 | 0.9053 | 90.60% |
| C4 | + NeCo | 0.5583 | 0.4884 | 0.4878 | 94.87% | 91.99% | 91.28% | 0.4992 | 0.1420 | 48.07% |
| C5 | + local grid + NeCo | 0.5360 | 0.4841 | 0.4834 | 93.53% | 91.57% | 90.93% | 0.5192 | 0.5324 | 60.40% |

## Artifacts

- C0 Base SimCLR
  - run: `D:\project\unknown-contrastive\runs\260610_071520_simclr_component_c0_base_simclr`
  - model: `D:\project\unknown-contrastive\runs\260610_071520_simclr_component_c0_base_simclr\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_071520_simclr_component_c0_base_simclr\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\C0_base_simclr.log`
- C1 + MoCo EMA queue
  - run: `D:\project\unknown-contrastive\runs\260610_073027_simclr_component_c1_queue`
  - model: `D:\project\unknown-contrastive\runs\260610_073027_simclr_component_c1_queue\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_073027_simclr_component_c1_queue\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\C1_queue.log`
- C2 + neg-ignore
  - run: `D:\project\unknown-contrastive\runs\260610_074541_simclr_component_c2_queue_neg_ignore`
  - model: `D:\project\unknown-contrastive\runs\260610_074541_simclr_component_c2_queue_neg_ignore\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_074541_simclr_component_c2_queue_neg_ignore\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\C2_queue_neg_ignore.log`
- C3 + local grid
  - run: `D:\project\unknown-contrastive\runs\260610_080110_simclr_component_c3_queue_neg_ignore_local`
  - model: `D:\project\unknown-contrastive\runs\260610_080110_simclr_component_c3_queue_neg_ignore_local\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_080110_simclr_component_c3_queue_neg_ignore_local\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\C3_queue_neg_ignore_local.log`
- C4 + NeCo
  - run: `D:\project\unknown-contrastive\runs\260610_081645_simclr_component_c4_queue_neg_ignore_neco`
  - model: `D:\project\unknown-contrastive\runs\260610_081645_simclr_component_c4_queue_neg_ignore_neco\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_081645_simclr_component_c4_queue_neg_ignore_neco\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\C4_queue_neg_ignore_neco.log`
- C5 + local grid + NeCo
  - run: `D:\project\unknown-contrastive\runs\260610_083650_simclr_component_c5_all_queue_neg_ignore_local_neco`
  - model: `D:\project\unknown-contrastive\runs\260610_083650_simclr_component_c5_all_queue_neg_ignore_local_neco\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_083650_simclr_component_c5_all_queue_neg_ignore_local_neco\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\C5_all_queue_neg_ignore_local_neco.log`
