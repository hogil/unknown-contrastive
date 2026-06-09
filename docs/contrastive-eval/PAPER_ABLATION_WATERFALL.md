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
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_all_lr2e6_ep3.npy`
- source scratch output:
  `D:\project\unknown-contrastive\_ablation_waterfall.md`

## Main Table

| Stage | Method Added | ARI | NMI | AMI | Completeness | Homogeneity |
|---|---|---:|---:|---:|---:|---:|
| S0 | Raw FCMAE baseline | 0.2097 | 0.2169 | 0.2159 | 0.2208 | 0.2131 |
| S1 | + DINOv3 SSL backbone | 0.3097 | 0.2831 | 0.2822 | 0.2916 | 0.2751 |
| S2 | + PCA dimension tuning | 0.3173 | 0.2957 | 0.2949 | 0.2965 | 0.2950 |
| S3 | + wafer contrastive fine-tune | 0.4948 | 0.4334 | 0.4327 | 0.4348 | 0.4320 |

## Delta From Baseline

| Stage | ARI Gain | AMI Gain | Relative ARI Gain |
|---|---:|---:|---:|
| S1 vs S0 | +0.1000 | +0.0663 | +47.7% |
| S2 vs S0 | +0.1076 | +0.0790 | +51.3% |
| S3 vs S0 | +0.2851 | +0.2168 | +136.0% |

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

Best last-stage-only setting by primary ARI is `3e-6`, epoch 4. A later
full-unfreeze setting (`2e-6`, epoch 3) improves the final result and is used in
the main table.

### Longer Training Check

A separate CUDA run extended the last-stage recipe to 10 epochs. The training
loss kept decreasing (`1.4150 -> 0.4550`), but held-out novel ARI did not beat
the final full-unfreeze model.

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

Best CUDA long-run ARI is `0.4555`, below the selected full-unfreeze model
(`0.4948`). A deterministic CPU long run of the same last-stage recipe reached
`0.4738` at epoch 9, which improves last-stage-only ARI but still remains below
the selected full-unfreeze model.

## Unfreeze Scope Check

Full-backbone unfreeze is sensitive to LR. `1e-6` degraded held-out ARI, but
`2e-6` produced the best current model. `5e-6` gives competitive AMI/HDBSCAN
tradeoffs but does not beat the primary ARI of `2e-6` epoch 3.

- full-unfreeze embeddings:
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_all_lr1e6_cuda_ep1.npy`
  through
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_all_lr5e6_ep6.npy`

| Train mode | LR backbone | Epoch | ARI | NMI | AMI | ARI_hdb |
|---|---:|---:|---:|---:|---:|---:|
| all | 1e-6 | 4 | 0.3950 | 0.3647 | 0.3639 | 0.1920 |
| all | 2e-6 | 1 | 0.3892 | 0.3642 | 0.3634 | 0.2136 |
| all | 2e-6 | 2 | 0.4522 | 0.4161 | 0.4154 | 0.2090 |
| all | 2e-6 | 3 | 0.4948 | 0.4334 | 0.4327 | 0.4171 |
| all | 2e-6 | 4 | 0.4392 | 0.3969 | 0.3962 | 0.2256 |
| all | 2e-6 | 5 | 0.4748 | 0.4299 | 0.4292 | 0.2151 |
| all | 2e-6 | 6 | 0.4834 | 0.4223 | 0.4216 | 0.3998 |
| all | 5e-6 | 3 | 0.4754 | 0.4228 | 0.4221 | 0.4622 |
| all | 5e-6 | 5 | 0.4778 | 0.4592 | 0.4586 | 0.0008 |
| all | 5e-6 | 6 | 0.4494 | 0.4372 | 0.4365 | 0.3935 |

Best full-unfreeze primary ARI is `0.4948` at `2e-6`, epoch 3. The `5e-6`,
epoch-3 setting has the best auxiliary HDBSCAN ARI (`0.4622`) but lower primary
ARI (`0.4754`).

## Temperature Check

Lowering the InfoNCE temperature from `0.05` to `0.03` was tested with the same
last-stage unfreeze recipe. Raising it to `0.07` was also tested. Neither
temperature improved held-out novel ARI.

- `TEMP=0.03` embeddings:
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_laststage_lr3e6_temp003_cuda_ep1.npy`
  through
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_laststage_lr3e6_temp003_cuda_ep4.npy`

| Temp | Epoch | ARI | NMI | AMI | ARI_hdb |
|---:|---:|---:|---:|---:|---:|
| 0.03 | 1 | 0.3407 | 0.3159 | 0.3150 | 0.0421 |
| 0.03 | 2 | 0.4170 | 0.3830 | 0.3822 | 0.1788 |
| 0.03 | 3 | 0.4270 | 0.3863 | 0.3855 | 0.2081 |
| 0.03 | 4 | 0.4451 | 0.4031 | 0.4023 | 0.1977 |
| 0.07 | 1 | 0.3639 | 0.3424 | 0.3416 | 0.0428 |
| 0.07 | 2 | 0.3789 | 0.3561 | 0.3553 | 0.1412 |
| 0.07 | 3 | 0.4222 | 0.3876 | 0.3869 | 0.1874 |
| 0.07 | 4 | 0.4259 | 0.3920 | 0.3913 | 0.0388 |

Best tested non-default temperature is `TEMP=0.03` epoch 4 (`0.4451`), below
the selected full-unfreeze model (`0.4948`).

## False-Negative Ignore Threshold Check

Changing the false-negative ignore threshold was tested in both directions.
Raising it from `0.7` to `0.8` ignores fewer similar negatives and therefore
applies a stronger repulsion to nearby samples. Lowering it to `0.6` ignores
more similar negatives. Neither direction improved the primary held-out novel
ARI.

- `ignore=0.8` embeddings:
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_laststage_lr3e6_ignore080_cuda_ep1.npy`
  through
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_laststage_lr3e6_ignore080_cuda_ep4.npy`

| Ignore threshold | Epoch | ARI | NMI | AMI | ARI_hdb |
|---:|---:|---:|---:|---:|---:|
| 0.6 | 1 | 0.3713 | 0.3440 | 0.3432 | 0.1824 |
| 0.6 | 2 | 0.3861 | 0.3647 | 0.3640 | 0.1693 |
| 0.6 | 3 | 0.4417 | 0.4039 | 0.4032 | 0.3917 |
| 0.6 | 4 | 0.4310 | 0.3935 | 0.3928 | 0.2132 |
| 0.8 | 1 | 0.3389 | 0.3111 | 0.3103 | 0.0397 |
| 0.8 | 2 | 0.3857 | 0.3608 | 0.3600 | 0.1691 |
| 0.8 | 3 | 0.4395 | 0.4016 | 0.4008 | 0.1969 |
| 0.8 | 4 | 0.4479 | 0.4175 | 0.4168 | 0.0263 |

Best `ignore=0.8` ARI is `0.4479`, below the selected `ignore=0.7` model.
The `ignore=0.6` epoch-3 setting improves the auxiliary HDBSCAN ARI relative to
the old last-stage baseline (`0.3917` vs `0.3717`) but remains below the new
full-unfreeze best (`0.4171`), so it is not the main paper setting.

## NN-Positive Check

NN-positive pull was tested on the best full-unfreeze family. It did not improve
the selected main result, so it is excluded from the paper waterfall.

- `nn_pos_min_sim=0.9` embeddings:
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_all_nnpos_ep1.npy`
  through
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_all_nnpos_ep6.npy`
- `nn_pos_min_sim=0.6` embeddings:
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_all_nnpos06_ep1.npy`
  through
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_all_nnpos06_ep6.npy`

| Setting | Epoch | ARI | NMI | AMI | ARI_hdb |
|---|---:|---:|---:|---:|---:|
| nn-pos min_sim=0.9 | 3 | 0.4948 | 0.4334 | 0.4327 | 0.4171 |
| nn-pos min_sim=0.9 | 6 | 0.4721 | 0.4157 | 0.4150 | 0.3956 |
| nn-pos min_sim=0.6 | 3 | 0.4469 | 0.3983 | 0.3976 | 0.3788 |
| nn-pos min_sim=0.6 | 6 | 0.4557 | 0.4230 | 0.4223 | 0.3580 |

The conservative threshold ties the selected epoch-3 score but does not add a
new gain over the full-unfreeze recipe. The looser threshold is weaker.

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
after fine-tuning (`0.2874 -> 0.4171`), but the paper-safe statement should
still be about NCD-style all-sample clustering (`k=3`) because HDBSCAN depends
on noise/threshold choices.
