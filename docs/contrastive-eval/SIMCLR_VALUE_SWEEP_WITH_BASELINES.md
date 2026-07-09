# SimCLR Value Sweep With Baselines

- train folder: `D:\project\unknown-contrastive\data\images\wm811k_novel_disjoint_v1\cnn_seen_train`
- eval folder: `D:\project\unknown-contrastive\data\images\wm811k_novel_disjoint_v1\novel_eval`
- eval classes: `Donut, Edge-Loc`
- data: WM-811K class-disjoint v1, not generated synthetic wafer
- eval embedding: full 1024-dim backbone feature, L2-normalized
- primary metric: k-means(k=3) ARI on held-out novel classes
- P1 capture: fraction of eval classes that own at least one dominant-class cluster
- image cap: auxiliary dominant-class image coverage

| Stage | Method | ARI | NMI | AMI | P1 capture | image cap | P2 noise | P3 comp | P4 homog | top1 | k5 | k9 | dist ratio | HDB ARI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | Raw FCMAE baseline | 0.0668 | 0.0569 | 0.0561 | 1.0000 | 0.1390 | 86.10% | 0.4872 | 1.0000 | 96.80% | 94.74% | 93.59% | 0.3571 | 0.4779 |
| B1 | Raw CNN A-supervised baseline | 0.0520 | 0.0427 | 0.0420 | 1.0000 | 0.3640 | 43.90% | 0.2208 | 0.0451 | 96.60% | 93.98% | 92.92% | 0.3050 | 0.0409 |
| B2 | Raw DINOv3 baseline | 0.0090 | 0.0074 | 0.0066 | 0.0000 | 0.0000 | 100.00% | 0.0000 | 0.0000 | 95.50% | 94.14% | 92.68% | 0.2670 | 0.0000 |
| V0 | queue+ignore0.70+local_w0.1 | 0.0056 | 0.0067 | 0.0059 | 1.0000 | 0.2580 | 68.00% | 0.3544 | 0.3729 | 97.10% | 94.64% | 93.59% | 0.5554 | 0.3264 |
| V1 | queue+ignore0.70+local_w0.3 | 0.6364 | 0.5278 | 0.5275 | 1.0000 | 0.3060 | 67.40% | 0.5757 | 0.6757 | 96.10% | 94.84% | 94.08% | 0.6170 | 0.7049 |
| V2 | queue+ignore0.60+local_w0.2 | 0.3813 | 0.3103 | 0.3098 | 1.0000 | 0.4000 | 48.60% | 0.3426 | 0.2604 | 96.70% | 95.00% | 94.12% | 0.5215 | 0.2706 |
| V3 | queue+ignore0.80+local_w0.2 | 0.4619 | 0.3795 | 0.3790 | 1.0000 | 0.2810 | 66.60% | 0.4267 | 0.4163 | 96.00% | 95.10% | 94.18% | 0.5821 | 0.4259 |
| V4 | queue512+ignore0.70+local_w0.2 | 0.5324 | 0.4379 | 0.4375 | 1.0000 | 0.3050 | 62.60% | 0.4729 | 0.3844 | 96.00% | 94.90% | 94.07% | 0.6035 | 0.3937 |
| V5 | queue2048+ignore0.70+local_w0.2 | 0.5266 | 0.4309 | 0.4305 | 1.0000 | 0.1280 | 84.10% | 0.3033 | 0.4739 | 96.00% | 95.26% | 94.10% | 0.6046 | 0.2561 |
| V6 | temp0.04+queue+ignore0.70+local_w0.2 | 0.3690 | 0.2920 | 0.2915 | 1.0000 | 0.1980 | 74.10% | 0.2623 | 0.4475 | 95.70% | 94.50% | 93.86% | 0.5611 | 0.1989 |
| V7 | temp0.06+queue+ignore0.70+local_w0.2 | 0.5354 | 0.4363 | 0.4359 | 1.0000 | 0.1640 | 80.00% | 0.3083 | 0.5779 | 96.30% | 95.04% | 94.26% | 0.5890 | 0.2859 |
| V8 | queue+ignore0.70+neco_w0.1 | 0.4756 | 0.3783 | 0.3779 | 1.0000 | 0.1240 | 84.60% | 0.3575 | 0.5548 | 96.60% | 93.72% | 92.71% | 0.5487 | 0.3001 |
| V9 | queue+ignore0.70+local_w0.1+neco_w0.1 | 0.5711 | 0.4656 | 0.4652 | 1.0000 | 0.1550 | 82.40% | 0.3437 | 0.6324 | 95.80% | 94.52% | 93.71% | 0.5862 | 0.3918 |

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
