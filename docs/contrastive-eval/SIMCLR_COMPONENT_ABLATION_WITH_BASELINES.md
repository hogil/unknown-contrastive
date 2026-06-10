# SimCLR Component Ablation With Baselines

- train folder: `D:\project\unknown-contrastive\data\images\wm811k_novel_disjoint_v1\cnn_seen_train`
- eval folder: `D:\project\unknown-contrastive\data\images\wm811k_novel_disjoint_v1\novel_eval`
- eval classes: `Donut, Edge-Loc, Random`
- data: WM-811K class-disjoint v1, not generated synthetic wafer
- eval embedding: full 1024-dim backbone feature, L2-normalized
- primary metric: k-means(k=3) ARI on held-out novel classes
- fixed HDBSCAN: min_cluster_size=12, min_samples=15, leaf, epsilon=0.06

| Stage | Method | ARI | NMI | AMI | P1 capture | P2 noise | P3 comp | P4 homog | top1 | k5 | k9 | dist ratio | HDB ARI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | Raw FCMAE baseline | 0.2047 | 0.2106 | 0.2096 | 0.1180 | 73.33% | 0.4247 | 0.7754 | 94.47% | 91.36% | 90.27% | 0.3082 | 0.3001 |
| B1 | Raw CNN A-supervised baseline | 0.1913 | 0.2114 | 0.2104 | 0.2740 | 65.40% | 0.4773 | 0.3851 | 93.33% | 90.95% | 89.28% | 0.2656 | 0.3068 |
| B2 | Raw DINOv3 baseline | 0.3173 | 0.2957 | 0.2949 | 0.0360 | 95.20% | 0.5833 | 1.0000 | 93.87% | 91.43% | 90.28% | 0.2490 | 0.5792 |
| C0 | Base SimCLR | 0.4340 | 0.4354 | 0.4347 | 0.3160 | 63.40% | 0.5456 | 0.3763 | 94.13% | 91.97% | 91.11% | 0.4106 | 0.4297 |
| C1 | + MoCo EMA queue | 0.5000 | 0.4671 | 0.4664 | 0.4067 | 56.13% | 0.4157 | 0.1381 | 92.67% | 90.25% | 89.22% | 0.4615 | 0.1670 |
| C2 | + neg-ignore | 0.4693 | 0.4229 | 0.4222 | 0.3767 | 56.07% | 0.5739 | 0.3920 | 93.53% | 91.41% | 89.98% | 0.4615 | 0.4745 |
| C3 | + local grid | 0.5160 | 0.4803 | 0.4797 | 0.0860 | 90.60% | 0.7683 | 1.0000 | 93.53% | 91.79% | 90.93% | 0.5262 | 0.9053 |
| C4 | + NeCo | 0.5583 | 0.4884 | 0.4878 | 0.4727 | 48.07% | 0.4443 | 0.1767 | 94.87% | 91.99% | 91.28% | 0.4992 | 0.1420 |
| C5 | + local grid + NeCo | 0.5360 | 0.4841 | 0.4834 | 0.3407 | 60.40% | 0.5697 | 0.4385 | 93.53% | 91.57% | 90.93% | 0.5192 | 0.5324 |

## Artifacts

- B0 Raw FCMAE baseline
  - run: `D:\project\unknown-contrastive\result_grouping\260609_142348_wm811k_novel_v1_raw_vs_Acnn`
  - model: `D:\project\unknown-contrastive\weights\convnextv2_base.fcmae_ft_in22k_in1k_384.pth`
  - embedding: `D:\project\unknown-contrastive\result_grouping\260609_142348_wm811k_novel_v1_raw_vs_Acnn\embeddings\01_Raw_FCMAE_raw.npy`
- B1 Raw CNN A-supervised baseline
  - run: `D:\project\unknown-contrastive\result_grouping\260609_142348_wm811k_novel_v1_raw_vs_Acnn`
  - model: `D:\project\unknown-contrastive\runs\260609_141113_cnn_ddp\cnn\best_model.pth`
  - embedding: `D:\project\unknown-contrastive\result_grouping\260609_142348_wm811k_novel_v1_raw_vs_Acnn\embeddings\02_CNN_backbone_raw.npy`
- B2 Raw DINOv3 baseline
  - run: `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop`
  - model: `hf_hub:timm/convnext_base.dinov3_lvd1689m`
  - embedding: `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\embeddings\dinov3_convnext_base.npy`
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
