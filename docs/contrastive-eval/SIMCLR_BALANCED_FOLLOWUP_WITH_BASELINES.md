# SimCLR Balanced Follow-up With Baselines

- train folder: `D:\project\unknown-contrastive\data\images\wm811k_novel_disjoint_v1\cnn_seen_train`
- eval folder: `D:\project\unknown-contrastive\data\images\wm811k_novel_disjoint_v1\novel_eval`
- eval classes: `Donut, Edge-Loc`
- ignored eval classes: `Normal, Random`
- data: WM-811K class-disjoint v1, not generated synthetic wafer
- eval embedding: full 1024-dim backbone feature, L2-normalized
- primary metric: k-means(k=#eval classes) ARI on held-out shape classes
- P1 capture: strict dominant-class image coverage
- found: auxiliary fraction of eval classes that own at least one dominant-class cluster
- purpose: combine L4 queue3072 ARI strength with L8 lr3e-6 P1/P2 balance

| Stage | Method | ARI | NMI | AMI | P1 capture | found | P2 noise | P3 comp | P4 homog | top1 | k5 | k9 | dist ratio | HDB ARI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | Raw FCMAE baseline | 0.0668 | 0.0569 | 0.0561 | 0.1390 | 1.0000 | 86.10% | 0.4872 | 1.0000 | 96.80% | 94.74% | 93.59% | 0.3571 | 0.4779 |
| B1 | Raw CNN A-supervised baseline | 0.0520 | 0.0427 | 0.0420 | 0.3640 | 1.0000 | 43.90% | 0.2208 | 0.0451 | 96.60% | 93.98% | 92.92% | 0.3050 | 0.0409 |
| B2 | Raw DINOv3 baseline | 0.0090 | 0.0074 | 0.0066 | 0.0000 | 0.0000 | 100.00% | 0.0000 | 0.0000 | 95.50% | 94.14% | 92.68% | 0.2670 | 0.0000 |
| F0 | queue3072+lr3e-6+ignore0.70+local_w0.20 | 0.6953 | 0.5886 | 0.5883 | 0.3290 | 1.0000 | 65.00% | 0.7266 | 0.6528 | 97.30% | 95.16% | 94.10% | 0.6651 | 0.7626 |
| F1 | queue3072+lr3e-6+ignore0.70+local_w0.15 | 0.5621 | 0.4725 | 0.4721 | 0.2920 | 1.0000 | 63.20% | 0.4382 | 0.3311 | 96.80% | 95.50% | 94.52% | 0.6161 | 0.3370 |
| F2 | queue3072+lr3e-6+ignore0.70+local_w0.18 | 0.6953 | 0.6190 | 0.6187 | 0.1360 | 1.0000 | 84.90% | 0.4160 | 0.6828 | 96.90% | 95.12% | 94.14% | 0.6175 | 0.4201 |
| F3 | queue3072+lr3e-6+ignore0.80+local_w0.20 | 0.6986 | 0.5978 | 0.5975 | 0.1630 | 1.0000 | 81.30% | 0.2911 | 0.4996 | 95.70% | 94.70% | 93.97% | 0.6206 | 0.3401 |
| F4 | queue3072+lr3e-6+ignore0.60+local_w0.20 | 0.4923 | 0.4087 | 0.4082 | 0.3090 | 1.0000 | 61.10% | 0.3721 | 0.3448 | 97.00% | 95.58% | 94.62% | 0.6090 | 0.3144 |
| F5 | queue2048+lr3e-6+ignore0.70+local_w0.15 | 0.6886 | 0.5813 | 0.5810 | 0.2470 | 1.0000 | 67.40% | 0.3271 | 0.2917 | 97.10% | 95.40% | 94.60% | 0.6173 | 0.2348 |
| F6 | queue3072+lr3e-6+ignore0.70+local_w0.25 | 0.5956 | 0.4910 | 0.4907 | 0.4070 | 1.0000 | 52.20% | 0.4391 | 0.4429 | 96.90% | 95.58% | 95.00% | 0.6395 | 0.4592 |
| F7 | queue4096+lr3e-6+ignore0.70+local_w0.20 | 0.7154 | 0.6121 | 0.6118 | 0.2640 | 1.0000 | 72.40% | 0.7834 | 0.7342 | 97.60% | 95.44% | 94.44% | 0.6743 | 0.8280 |
| F8 | queue3072+lr3e-6+temp0.04+ignore0.70+local_w0.20 | 0.7154 | 0.6129 | 0.6126 | 0.1790 | 1.0000 | 81.00% | 0.3345 | 0.7848 | 97.30% | 94.80% | 93.92% | 0.5981 | 0.3818 |

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
- F0 queue3072+lr3e-6+ignore0.70+local_w0.20
  - run: `D:\project\unknown-contrastive\runs\260610_152557_simclr_component_f0_queue3072_lrb3e6_local_w0p20`
  - model: `D:\project\unknown-contrastive\runs\260610_152557_simclr_component_f0_queue3072_lrb3e6_local_w0p20\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_152557_simclr_component_f0_queue3072_lrb3e6_local_w0p20\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\F0_queue3072_lrb3e6_local_w0p20.log`
- F1 queue3072+lr3e-6+ignore0.70+local_w0.15
  - run: `D:\project\unknown-contrastive\runs\260610_154356_simclr_component_f1_queue3072_lrb3e6_local_w0p15`
  - model: `D:\project\unknown-contrastive\runs\260610_154356_simclr_component_f1_queue3072_lrb3e6_local_w0p15\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_154356_simclr_component_f1_queue3072_lrb3e6_local_w0p15\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\F1_queue3072_lrb3e6_local_w0p15.log`
- F2 queue3072+lr3e-6+ignore0.70+local_w0.18
  - run: `D:\project\unknown-contrastive\runs\260610_160603_simclr_component_f2_queue3072_lrb3e6_local_w0p18`
  - model: `D:\project\unknown-contrastive\runs\260610_160603_simclr_component_f2_queue3072_lrb3e6_local_w0p18\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_160603_simclr_component_f2_queue3072_lrb3e6_local_w0p18\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\F2_queue3072_lrb3e6_local_w0p18.log`
- F3 queue3072+lr3e-6+ignore0.80+local_w0.20
  - run: `D:\project\unknown-contrastive\runs\260610_162841_simclr_component_f3_queue3072_lrb3e6_ignore0p80_local_w0p20`
  - model: `D:\project\unknown-contrastive\runs\260610_162841_simclr_component_f3_queue3072_lrb3e6_ignore0p80_local_w0p20\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_162841_simclr_component_f3_queue3072_lrb3e6_ignore0p80_local_w0p20\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\F3_queue3072_lrb3e6_ignore0p80_local_w0p20.log`
- F4 queue3072+lr3e-6+ignore0.60+local_w0.20
  - run: `D:\project\unknown-contrastive\runs\260610_164642_simclr_component_f4_queue3072_lrb3e6_ignore0p60_local_w0p20`
  - model: `D:\project\unknown-contrastive\runs\260610_164642_simclr_component_f4_queue3072_lrb3e6_ignore0p60_local_w0p20\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_164642_simclr_component_f4_queue3072_lrb3e6_ignore0p60_local_w0p20\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\F4_queue3072_lrb3e6_ignore0p60_local_w0p20.log`
- F5 queue2048+lr3e-6+ignore0.70+local_w0.15
  - run: `D:\project\unknown-contrastive\runs\260610_170432_simclr_component_f5_queue2048_lrb3e6_local_w0p15`
  - model: `D:\project\unknown-contrastive\runs\260610_170432_simclr_component_f5_queue2048_lrb3e6_local_w0p15\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_170432_simclr_component_f5_queue2048_lrb3e6_local_w0p15\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\F5_queue2048_lrb3e6_local_w0p15.log`
- F6 queue3072+lr3e-6+ignore0.70+local_w0.25
  - run: `D:\project\unknown-contrastive\runs\260610_173026_simclr_component_f6_queue3072_lrb3e6_local_w0p25`
  - model: `D:\project\unknown-contrastive\runs\260610_173026_simclr_component_f6_queue3072_lrb3e6_local_w0p25\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_173026_simclr_component_f6_queue3072_lrb3e6_local_w0p25\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\F6_queue3072_lrb3e6_local_w0p25.log`
- F7 queue4096+lr3e-6+ignore0.70+local_w0.20
  - run: `D:\project\unknown-contrastive\runs\260610_174838_simclr_component_f7_queue4096_lrb3e6_local_w0p20`
  - model: `D:\project\unknown-contrastive\runs\260610_174838_simclr_component_f7_queue4096_lrb3e6_local_w0p20\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_174838_simclr_component_f7_queue4096_lrb3e6_local_w0p20\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\F7_queue4096_lrb3e6_local_w0p20.log`
- F8 queue3072+lr3e-6+temp0.04+ignore0.70+local_w0.20
  - run: `D:\project\unknown-contrastive\runs\260610_180628_simclr_component_f8_queue3072_lrb3e6_temp0p04_local_w0p20`
  - model: `D:\project\unknown-contrastive\runs\260610_180628_simclr_component_f8_queue3072_lrb3e6_temp0p04_local_w0p20\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_180628_simclr_component_f8_queue3072_lrb3e6_temp0p04_local_w0p20\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\F8_queue3072_lrb3e6_temp0p04_local_w0p20.log`
