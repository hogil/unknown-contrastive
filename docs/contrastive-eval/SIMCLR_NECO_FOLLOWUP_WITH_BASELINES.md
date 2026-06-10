# SimCLR NeCo Follow-up With Baselines

- train folder: `D:\project\unknown-contrastive\data\images\wm811k_novel_disjoint_v1\cnn_seen_train`
- eval folder: `D:\project\unknown-contrastive\data\images\wm811k_novel_disjoint_v1\novel_eval`
- eval classes: `Donut, Edge-Loc, Random`
- data: WM-811K class-disjoint v1, not generated synthetic wafer
- eval embedding: full 1024-dim backbone feature, L2-normalized
- primary metric: k-means(k=3) ARI on held-out novel classes

| Stage | Method | ARI | NMI | AMI | P1 capture | P2 noise | P3 comp | P4 homog | top1 | k5 | k9 | dist ratio | HDB ARI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | Raw FCMAE baseline | 0.2047 | 0.2106 | 0.2096 | 0.1180 | 73.33% | 0.4247 | 0.7754 | 94.47% | 91.36% | 90.27% | 0.3082 | 0.3001 |
| B1 | Raw CNN A-supervised baseline | 0.1913 | 0.2114 | 0.2104 | 0.2740 | 65.40% | 0.4773 | 0.3851 | 93.33% | 90.95% | 89.28% | 0.2656 | 0.3068 |
| B2 | Raw DINOv3 baseline | 0.3173 | 0.2957 | 0.2949 | 0.0360 | 95.20% | 0.5833 | 1.0000 | 93.87% | 91.43% | 90.28% | 0.2490 | 0.5792 |
| N0 | queue+ignore0.70+neco_w0.15 | 0.5040 | 0.4458 | 0.4451 | 0.3433 | 60.07% | 0.6001 | 0.3957 | 93.47% | 91.63% | 90.66% | 0.4888 | 0.4831 |

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
- N0 queue+ignore0.70+neco_w0.15
  - run: `D:\project\unknown-contrastive\runs\260610_114527_simclr_component_n0_neco_w0p15`
  - model: `D:\project\unknown-contrastive\runs\260610_114527_simclr_component_n0_neco_w0p15\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_114527_simclr_component_n0_neco_w0p15\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\N0_neco_w0p15.log`
