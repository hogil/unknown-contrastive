# WM-811K Novel-Class Discovery — Ablation Waterfall

backbone: convnext_base.dinov3_lvd1689m | eval: held-out novel (Donut/Edge-Loc/Random, 1500)
primary = k-means(k=3, known novel count) ARI/NMI/AMI (모든 점 배정, 표준 NCD).
ARI_hdb = HDBSCAN(auto-k)+soft-reassign ARI (보조, k 모를 때).

| Stage | ARI | NMI | AMI | Comp | Hom | ARI_hdb |
|---|--:|--:|--:|--:|--:|--:|
| S0 FCMAE baseline | **0.2097** | 0.2169 | 0.2159 | 0.2208 | 0.2131 | 0.2654 |
| S1 + DINOv3 backbone | **0.3097** | 0.2831 | 0.2822 | 0.2916 | 0.2751 | 0.2646 |
| S2 + PCA tuning | **0.3173** | 0.2957 | 0.2949 | 0.2965 | 0.2950 | 0.2874 |
| S3 + PCA whiten | **0.3173** | 0.2957 | 0.2949 | 0.2965 | 0.2950 | 0.2874 |
| S4 + wafer contrastive fine-tune | **0.5976** | 0.5258 | 0.5252 | 0.5259 | 0.5257 | 0.0582 |
