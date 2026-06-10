# SimCLR Local Follow-up With Baselines

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
| L0 | queue2048+ignore0.70+local_w0.15 | 0.5278 | 0.4714 | 0.4707 | 0.4307 | 54.67% | 0.5061 | 0.1080 | 93.87% | 91.33% | 90.02% | 0.5174 | 0.1223 |
| L1 | queue2048+ignore0.70+local_w0.25 | 0.5179 | 0.4506 | 0.4499 | 0.0927 | 88.20% | 0.6517 | 0.7748 | 93.07% | 91.00% | 90.03% | 0.5299 | 0.6995 |
| L2 | queue2048+ignore0.60+local_w0.20 | 0.5484 | 0.5080 | 0.5074 | 0.1407 | 82.60% | 0.7548 | 0.9756 | 93.27% | 92.11% | 91.30% | 0.4522 | 0.8944 |
| L3 | queue2048+ignore0.80+local_w0.20 | 0.5271 | 0.4575 | 0.4569 | 0.2967 | 67.00% | 0.6579 | 0.5602 | 94.20% | 92.03% | 90.84% | 0.5334 | 0.6973 |
| L4 | queue3072+ignore0.70+local_w0.20 | 0.5644 | 0.5223 | 0.5217 | 0.1033 | 88.20% | 0.8198 | 1.0000 | 93.80% | 92.24% | 91.45% | 0.5475 | 0.8876 |
| L5 | queue4096+ignore0.70+local_w0.20 | 0.5446 | 0.5060 | 0.5054 | 0.3360 | 60.87% | 0.5909 | 0.3858 | 93.60% | 92.03% | 91.25% | 0.5412 | 0.4756 |
| L6 | temp0.06+queue2048+ignore0.70+local_w0.20 | 0.5099 | 0.4555 | 0.4548 | 0.3500 | 56.00% | 0.4973 | 0.4962 | 93.13% | 91.87% | 91.12% | 0.5213 | 0.4818 |
| L7 | lr_backbone1e-6+queue2048+ignore0.70+local_w0.20 | 0.4221 | 0.3802 | 0.3794 | 0.3667 | 54.47% | 0.4191 | 0.3012 | 93.53% | 91.59% | 90.33% | 0.4517 | 0.3790 |
| L8 | lr_backbone3e-6+queue2048+ignore0.70+local_w0.20 | 0.5597 | 0.4985 | 0.4979 | 0.4093 | 55.87% | 0.5575 | 0.2969 | 93.27% | 90.80% | 89.96% | 0.5934 | 0.2444 |
| L9 | epoch8+queue2048+ignore0.70+local_w0.20 | 0.5360 | 0.5014 | 0.5008 | 0.3387 | 61.13% | 0.5552 | 0.5172 | 92.20% | 91.20% | 90.13% | 0.5822 | 0.5437 |

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
- L0 queue2048+ignore0.70+local_w0.15
  - run: `D:\project\unknown-contrastive\runs\260610_122158_simclr_component_l0_queue2048_local_w0p15`
  - model: `D:\project\unknown-contrastive\runs\260610_122158_simclr_component_l0_queue2048_local_w0p15\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_122158_simclr_component_l0_queue2048_local_w0p15\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\L0_queue2048_local_w0p15.log`
- L1 queue2048+ignore0.70+local_w0.25
  - run: `D:\project\unknown-contrastive\runs\260610_123800_simclr_component_l1_queue2048_local_w0p25`
  - model: `D:\project\unknown-contrastive\runs\260610_123800_simclr_component_l1_queue2048_local_w0p25\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_123800_simclr_component_l1_queue2048_local_w0p25\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\L1_queue2048_local_w0p25.log`
- L2 queue2048+ignore0.60+local_w0.20
  - run: `D:\project\unknown-contrastive\runs\260610_125358_simclr_component_l2_queue2048_local_ignore0p60`
  - model: `D:\project\unknown-contrastive\runs\260610_125358_simclr_component_l2_queue2048_local_ignore0p60\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_125358_simclr_component_l2_queue2048_local_ignore0p60\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\L2_queue2048_local_ignore0p60.log`
- L3 queue2048+ignore0.80+local_w0.20
  - run: `D:\project\unknown-contrastive\runs\260610_130953_simclr_component_l3_queue2048_local_ignore0p80`
  - model: `D:\project\unknown-contrastive\runs\260610_130953_simclr_component_l3_queue2048_local_ignore0p80\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_130953_simclr_component_l3_queue2048_local_ignore0p80\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\L3_queue2048_local_ignore0p80.log`
- L4 queue3072+ignore0.70+local_w0.20
  - run: `D:\project\unknown-contrastive\runs\260610_132551_simclr_component_l4_queue3072_local_w0p2`
  - model: `D:\project\unknown-contrastive\runs\260610_132551_simclr_component_l4_queue3072_local_w0p2\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_132551_simclr_component_l4_queue3072_local_w0p2\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\L4_queue3072_local_w0p2.log`
- L5 queue4096+ignore0.70+local_w0.20
  - run: `D:\project\unknown-contrastive\runs\260610_134148_simclr_component_l5_queue4096_local_w0p2`
  - model: `D:\project\unknown-contrastive\runs\260610_134148_simclr_component_l5_queue4096_local_w0p2\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_134148_simclr_component_l5_queue4096_local_w0p2\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\L5_queue4096_local_w0p2.log`
- L6 temp0.06+queue2048+ignore0.70+local_w0.20
  - run: `D:\project\unknown-contrastive\runs\260610_135747_simclr_component_l6_queue2048_local_temp006`
  - model: `D:\project\unknown-contrastive\runs\260610_135747_simclr_component_l6_queue2048_local_temp006\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_135747_simclr_component_l6_queue2048_local_temp006\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\L6_queue2048_local_temp006.log`
- L7 lr_backbone1e-6+queue2048+ignore0.70+local_w0.20
  - run: `D:\project\unknown-contrastive\runs\260610_141347_simclr_component_l7_queue2048_local_lrb1e6`
  - model: `D:\project\unknown-contrastive\runs\260610_141347_simclr_component_l7_queue2048_local_lrb1e6\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_141347_simclr_component_l7_queue2048_local_lrb1e6\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\L7_queue2048_local_lrb1e6.log`
- L8 lr_backbone3e-6+queue2048+ignore0.70+local_w0.20
  - run: `D:\project\unknown-contrastive\runs\260610_142944_simclr_component_l8_queue2048_local_lrb3e6`
  - model: `D:\project\unknown-contrastive\runs\260610_142944_simclr_component_l8_queue2048_local_lrb3e6\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_142944_simclr_component_l8_queue2048_local_lrb3e6\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\L8_queue2048_local_lrb3e6.log`
- L9 epoch8+queue2048+ignore0.70+local_w0.20
  - run: `D:\project\unknown-contrastive\runs\260610_144540_simclr_component_l9_queue2048_local_ep8`
  - model: `D:\project\unknown-contrastive\runs\260610_144540_simclr_component_l9_queue2048_local_ep8\contrastive\best_model.pt`
  - embedding: `D:\project\unknown-contrastive\runs\260610_144540_simclr_component_l9_queue2048_local_ep8\contrastive\embeddings.npy`
  - log: `D:\project\unknown-contrastive\result_grouping\_simclr_component_ablation_logs\L9_queue2048_local_ep8.log`
