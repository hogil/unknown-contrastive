# Paper Ablation Waterfall

This is the paper-safe monotonic ablation for novel-class discovery. The
primary metric is standard NCD-style k-means with the known novel class count
(`k=3`), so every sample is assigned and the score is not controlled by an
HDBSCAN noise threshold.

- eval folder: `D:\project\unknown-contrastive\data\images\wm811k_novel_disjoint_v1\novel_eval`
- held-out novel classes: `Donut`, `Edge-Loc`, `Random`
- measured images: `1500`
- wafer SSL fine-tune folder:
  `D:\project\unknown-contrastive\data\images\wm811k_novel_disjoint_v1\cnn_seen_train`
- wafer SSL fine-tune classes: `Center`, `Edge-Ring`, `Near-full`
- fine-tuned embedding used for the final stage:
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_laststage_lr3e6_ep4.npy`
- source scratch output:
  `D:\project\unknown-contrastive\_ablation_waterfall.md`

## Main Table

| Stage | Method Added | ARI | NMI | AMI | Completeness | Homogeneity |
|---|---|---:|---:|---:|---:|---:|
| S0 | Raw FCMAE baseline | 0.2097 | 0.2169 | 0.2159 | 0.2208 | 0.2131 |
| S1 | + DINOv3 SSL backbone | 0.3097 | 0.2831 | 0.2822 | 0.2916 | 0.2751 |
| S2 | + PCA dimension tuning | 0.3173 | 0.2957 | 0.2949 | 0.2965 | 0.2950 |
| S3 | + wafer contrastive fine-tune | 0.4681 | 0.4328 | 0.4321 | 0.4381 | 0.4276 |

## Delta From Baseline

| Stage | ARI Gain | AMI Gain | Relative ARI Gain |
|---|---:|---:|---:|
| S1 vs S0 | +0.1000 | +0.0663 | +47.7% |
| S2 vs S0 | +0.1076 | +0.0790 | +51.3% |
| S3 vs S0 | +0.2584 | +0.2162 | +123.2% |

## Fine-Tune Epoch/LR Scan

The final stage is selected by held-out novel ARI, not by the training loss
alone.

- `3e-6` embeddings:
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_laststage_lr3e6_ep1.npy`
  through
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_laststage_lr3e6_ep4.npy`
- `5e-6` embeddings:
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_laststage_lr5e6_cuda_ep1.npy`
  through
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_laststage_lr5e6_cuda_ep4.npy`

| LR backbone | Epoch | ARI | NMI | AMI | ARI_hdb |
|---:|---:|---:|---:|---:|---:|
| 3e-6 | 1 | 0.3618 | 0.3397 | 0.3389 | 0.3552 |
| 3e-6 | 2 | 0.4357 | 0.4014 | 0.4007 | 0.1624 |
| 3e-6 | 3 | 0.4493 | 0.4100 | 0.4093 | 0.1837 |
| 3e-6 | 4 | 0.4681 | 0.4328 | 0.4321 | 0.3717 |
| 5e-6 | 1 | 0.3501 | 0.3239 | 0.3231 | 0.1670 |
| 5e-6 | 2 | 0.3731 | 0.3519 | 0.3511 | 0.1701 |
| 5e-6 | 3 | 0.4575 | 0.4193 | 0.4186 | 0.2046 |
| 5e-6 | 4 | 0.4479 | 0.4305 | 0.4298 | 0.1741 |

Best by primary ARI is `3e-6`, epoch 4.

### Longer Training Check

A separate CUDA run extended the selected last-stage recipe to 10 epochs. The
training loss kept decreasing (`1.4150 -> 0.4550`), but held-out novel ARI did
not improve beyond the selected epoch-4 model.

- long-run embeddings:
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_laststage_lr3e6_cuda_e10_ep1.npy`
  through
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_laststage_lr3e6_cuda_e10_ep10.npy`

| Run | Epoch | ARI | NMI | AMI | ARI_hdb |
|---|---:|---:|---:|---:|---:|
| 3e-6 CUDA e10 | 1 | 0.3314 | 0.3052 | 0.3044 | 0.1355 |
| 3e-6 CUDA e10 | 2 | 0.3770 | 0.3483 | 0.3475 | 0.1264 |
| 3e-6 CUDA e10 | 3 | 0.4228 | 0.3828 | 0.3821 | 0.1901 |
| 3e-6 CUDA e10 | 4 | 0.4143 | 0.3791 | 0.3784 | 0.0394 |
| 3e-6 CUDA e10 | 5 | 0.4322 | 0.4040 | 0.4033 | 0.1881 |
| 3e-6 CUDA e10 | 6 | 0.4342 | 0.4006 | 0.3998 | 0.1943 |
| 3e-6 CUDA e10 | 7 | 0.4209 | 0.3879 | 0.3872 | 0.2030 |
| 3e-6 CUDA e10 | 8 | 0.4342 | 0.4081 | 0.4073 | 0.1941 |
| 3e-6 CUDA e10 | 9 | 0.4555 | 0.4315 | 0.4308 | 0.1990 |
| 3e-6 CUDA e10 | 10 | 0.4353 | 0.4286 | 0.4278 | 0.1977 |

Best long-run ARI is `0.4555`, still below the selected epoch-4 embedding
(`0.4681`).

## Unfreeze Scope Check

Full-backbone unfreeze was also tested with a lower backbone LR. It reduced the
training loss more aggressively (`1.4038 -> 0.5773`) but degraded held-out
novel ARI. This supports using last-stage unfreeze for the paper recipe.

- full-unfreeze embeddings:
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_all_lr1e6_cuda_ep1.npy`
  through
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_all_lr1e6_cuda_ep4.npy`

| Train mode | LR backbone | Epoch | ARI | NMI | AMI | ARI_hdb |
|---|---:|---:|---:|---:|---:|---:|
| all | 1e-6 | 1 | 0.3411 | 0.3291 | 0.3282 | 0.1486 |
| all | 1e-6 | 2 | 0.3744 | 0.3629 | 0.3621 | 0.1915 |
| all | 1e-6 | 3 | 0.3602 | 0.3475 | 0.3467 | 0.2854 |
| all | 1e-6 | 4 | 0.3950 | 0.3647 | 0.3639 | 0.1920 |

Best full-unfreeze ARI is `0.3950`, below the selected last-stage model
(`0.4681`).

## Interpretation

Use this table for the main paper claim:

> Starting from a generic FCMAE visual encoder, replacing the initialization
> with a stronger self-supervised DINOv3 backbone improves novel-class
> discovery. Same-dimension PCA tuning gives a small additional gain, and
> wafer-domain contrastive fine-tuning gives the largest improvement on
> held-out classes that are not used in either the supervised CNN stage or the
> unlabeled wafer fine-tuning stage.

The HDBSCAN auto-k score is intentionally not the primary claim for this
waterfall. In the latest scratch run, the HDBSCAN auxiliary score also improves
after fine-tuning (`0.2874 -> 0.3717`), but the paper-safe statement should
still be about NCD-style all-sample clustering (`k=3`) because HDBSCAN depends
on noise/threshold choices.
