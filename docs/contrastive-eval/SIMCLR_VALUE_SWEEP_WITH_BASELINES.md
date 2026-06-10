# SimCLR Value Sweep With Baselines

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
| V0 | queue+ignore0.70+local_w0.1 | 0.5008 | 0.4775 | 0.4769 | 0.4473 | 49.93% | 0.5365 | 0.2277 | 93.27% | 90.87% | 89.73% | 0.5000 | 0.2297 |
| V1 | queue+ignore0.70+local_w0.3 | 0.5502 | 0.5037 | 0.5031 | 0.4500 | 49.80% | 0.5083 | 0.2883 | 92.93% | 90.95% | 89.96% | 0.5426 | 0.2313 |
| V2 | queue+ignore0.60+local_w0.2 | 0.5325 | 0.4966 | 0.4960 | 0.0713 | 89.80% | 0.4880 | 1.0000 | 93.73% | 92.20% | 91.31% | 0.4850 | 0.5527 |
| V3 | queue+ignore0.80+local_w0.2 | 0.5117 | 0.4458 | 0.4451 | 0.4473 | 50.20% | 0.5367 | 0.2050 | 93.60% | 91.77% | 90.61% | 0.5275 | 0.1954 |
| V4 | queue512+ignore0.70+local_w0.2 | 0.5569 | 0.4987 | 0.4981 | 0.4447 | 50.20% | 0.5009 | 0.2330 | 93.80% | 91.88% | 90.96% | 0.5481 | 0.2223 |
| V5 | queue2048+ignore0.70+local_w0.2 | 0.5586 | 0.5139 | 0.5133 | 0.0847 | 89.40% | 0.6755 | 1.0000 | 93.60% | 92.21% | 91.11% | 0.5483 | 0.7285 |
| V6 | temp0.04+queue+ignore0.70+local_w0.2 | 0.4812 | 0.4620 | 0.4613 | 0.3780 | 55.80% | 0.5243 | 0.3647 | 93.27% | 91.36% | 90.54% | 0.5129 | 0.4127 |
| V7 | temp0.06+queue+ignore0.70+local_w0.2 | 0.5579 | 0.4978 | 0.4972 | 0.0840 | 89.53% | 0.6854 | 1.0000 | 93.07% | 91.71% | 90.84% | 0.5305 | 0.7328 |
| V8 | queue+ignore0.70+neco_w0.1 | 0.5031 | 0.4678 | 0.4672 | 0.3013 | 65.00% | 0.6226 | 0.4138 | 94.47% | 91.76% | 90.44% | 0.4912 | 0.5261 |
| V9 | queue+ignore0.70+local_w0.1+neco_w0.1 | 0.5084 | 0.4737 | 0.4730 | 0.3460 | 61.53% | 0.6079 | 0.4630 | 94.07% | 91.88% | 90.70% | 0.5262 | 0.5596 |

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
- V0 queue+ignore0.70+local_w0.1
  - run: `D:\project\unknown-contrastive\runs\260610_085741_simclr_component_v0_local_w0p1`
  - model: `D:\project\unknown-contrastive\runs\260610_085741_simclr_component_v0_local_w0p1\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_085741_simclr_component_v0_local_w0p1\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\V0_local_w0p1.log`
- V1 queue+ignore0.70+local_w0.3
  - run: `D:\project\unknown-contrastive\runs\260610_091315_simclr_component_v1_local_w0p3`
  - model: `D:\project\unknown-contrastive\runs\260610_091315_simclr_component_v1_local_w0p3\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_091315_simclr_component_v1_local_w0p3\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\V1_local_w0p3.log`
- V2 queue+ignore0.60+local_w0.2
  - run: `D:\project\unknown-contrastive\runs\260610_092852_simclr_component_v2_ignore0p60_local`
  - model: `D:\project\unknown-contrastive\runs\260610_092852_simclr_component_v2_ignore0p60_local\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_092852_simclr_component_v2_ignore0p60_local\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\V2_ignore0p60_local.log`
- V3 queue+ignore0.80+local_w0.2
  - run: `D:\project\unknown-contrastive\runs\260610_094431_simclr_component_v3_ignore0p80_local`
  - model: `D:\project\unknown-contrastive\runs\260610_094431_simclr_component_v3_ignore0p80_local\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_094431_simclr_component_v3_ignore0p80_local\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\V3_ignore0p80_local.log`
- V4 queue512+ignore0.70+local_w0.2
  - run: `D:\project\unknown-contrastive\runs\260610_100008_simclr_component_v4_queue512_local`
  - model: `D:\project\unknown-contrastive\runs\260610_100008_simclr_component_v4_queue512_local\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_100008_simclr_component_v4_queue512_local\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\V4_queue512_local.log`
- V5 queue2048+ignore0.70+local_w0.2
  - run: `D:\project\unknown-contrastive\runs\260610_101527_simclr_component_v5_queue2048_local`
  - model: `D:\project\unknown-contrastive\runs\260610_101527_simclr_component_v5_queue2048_local\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_101527_simclr_component_v5_queue2048_local\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\V5_queue2048_local.log`
- V6 temp0.04+queue+ignore0.70+local_w0.2
  - run: `D:\project\unknown-contrastive\runs\260610_103042_simclr_component_v6_temp004_local`
  - model: `D:\project\unknown-contrastive\runs\260610_103042_simclr_component_v6_temp004_local\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_103042_simclr_component_v6_temp004_local\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\V6_temp004_local.log`
- V7 temp0.06+queue+ignore0.70+local_w0.2
  - run: `D:\project\unknown-contrastive\runs\260610_104556_simclr_component_v7_temp006_local`
  - model: `D:\project\unknown-contrastive\runs\260610_104556_simclr_component_v7_temp006_local\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_104556_simclr_component_v7_temp006_local\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\V7_temp006_local.log`
- V8 queue+ignore0.70+neco_w0.1
  - run: `D:\project\unknown-contrastive\runs\260610_110125_simclr_component_v8_neco_w0p1`
  - model: `D:\project\unknown-contrastive\runs\260610_110125_simclr_component_v8_neco_w0p1\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_110125_simclr_component_v8_neco_w0p1\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\V8_neco_w0p1.log`
- V9 queue+ignore0.70+local_w0.1+neco_w0.1
  - run: `D:\project\unknown-contrastive\runs\260610_112525_simclr_component_v9_local_neco_w0p1`
  - model: `D:\project\unknown-contrastive\runs\260610_112525_simclr_component_v9_local_neco_w0p1\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_112525_simclr_component_v9_local_neco_w0p1\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\V9_local_neco_w0p1.log`
