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
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ssl_embeddings\simclr_ep5.npy`
- source scratch output:
  `D:\project\unknown-contrastive\_ablation_waterfall.md`

## Main Table

| Stage | Method Added | ARI | NMI | AMI | Completeness | Homogeneity |
|---|---|---:|---:|---:|---:|---:|
| S0 | Raw FCMAE baseline | 0.2097 | 0.2169 | 0.2159 | 0.2208 | 0.2131 |
| S1 | + DINOv3 SSL backbone | 0.3097 | 0.2831 | 0.2822 | 0.2916 | 0.2751 |
| S2 | + PCA dimension tuning | 0.3173 | 0.2957 | 0.2949 | 0.2965 | 0.2950 |
| S3 | + wafer contrastive fine-tune | 0.5976 | 0.5258 | 0.5252 | 0.5259 | 0.5257 |

Companion embedding retrieval metrics are recorded in
`D:\project\unknown-contrastive\docs\contrastive-eval\PAPER_ABLATION_EMBEDDING_RETRIEVAL.md`.
Those kNN metrics show that the final SimCLR fine-tune mainly improves the
broader neighborhood structure (`k3`/`k5`/`k7`/`k9`) and class-separation
distance ratio, while Raw FCMAE remains slightly higher on the single-nearest
neighbor `top1` metric.

## Delta From Baseline

| Stage | ARI Gain | AMI Gain | Relative ARI Gain |
|---|---:|---:|---:|
| S1 vs S0 | +0.1000 | +0.0663 | +47.7% |
| S2 vs S0 | +0.1076 | +0.0790 | +51.3% |
| S3 vs S0 | +0.3879 | +0.3093 | +185.0% |

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

Best CUDA long-run ARI is `0.4555`, below the selected final model
(`0.5294`). A deterministic CPU long run of the same last-stage recipe reached
`0.4738` at epoch 9, which improves last-stage-only ARI but still remains below
the selected final model.

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

With the default projection-head LR (`1e-3`), best full-unfreeze primary ARI is
`0.4948` at `2e-6`, epoch 3. The `5e-6`, epoch-3 setting has the best auxiliary
HDBSCAN ARI (`0.4622`) but lower primary ARI (`0.4754`).

## Projection Head LR Check

The best backbone LR (`2e-6`) was kept fixed and only the projection-head LR was
changed. Lowering the head LR to `5e-4` improved the paper-primary all-sample
NCD score. A higher nearby value (`7e-4`) and a lower value (`2e-4`) were both
weaker.

- `lr_head=5e-4` embeddings:
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_all_lr2e6_head5e4_ep1.npy`
  through
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_all_lr2e6_head5e4_ep6.npy`
- `lr_head=7e-4` embeddings:
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_all_lr2e6_head7e4_ep1.npy`
  through
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_all_lr2e6_head7e4_ep6.npy`
- `lr_head=2e-4` embeddings:
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_all_lr2e6_head2e4_ep1.npy`
  through
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_all_lr2e6_head2e4_ep6.npy`

| LR head | Epoch | ARI | NMI | AMI | ARI_hdb |
|---:|---:|---:|---:|---:|---:|
| 1e-3 | 3 | 0.4948 | 0.4334 | 0.4327 | 0.4171 |
| 7e-4 | 5 | 0.4860 | 0.4492 | 0.4485 | 0.2218 |
| 7e-4 | 6 | 0.4654 | 0.4072 | 0.4065 | 0.2020 |
| 5e-4 | 4 | 0.4648 | 0.3989 | 0.3982 | 0.2272 |
| 5e-4 | 5 | 0.5143 | 0.4671 | 0.4664 | 0.2039 |
| 5e-4 | 6 | 0.4784 | 0.4155 | 0.4148 | 0.3948 |
| 2e-4 | 3 | 0.4574 | 0.4113 | 0.4106 | 0.2282 |
| 2e-4 | 5 | 0.4506 | 0.4299 | 0.4292 | 0.2125 |

Best paper-primary ARI with `lr_backbone=2e-6` is `0.5143` at `lr_head=5e-4`,
epoch 5. The
auxiliary HDBSCAN score is not monotonic with the paper-primary score, so it is
kept as a separate operational diagnostic.

## Backbone LR Refinement

After selecting `lr_head=5e-4`, the backbone LR was bracketed around `2e-6`.
Raising it slightly to `2.5e-6` improved both the paper-primary ARI and the
auxiliary HDBSCAN ARI. Lowering it to `1.5e-6` was weaker. A tighter right-side
check at `2.75e-6` came close but still stayed below `2.5e-6`, and `3e-6` also
dropped below the selected model.

- `lr_backbone=1.5e-6`, `lr_head=5e-4` embeddings:
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_all_lr1p5e6_head5e4_ep1.npy`
  through
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_all_lr1p5e6_head5e4_ep6.npy`
- `lr_backbone=2.5e-6`, `lr_head=5e-4` embeddings:
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_all_lr2p5e6_head5e4_ep1.npy`
  through
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_all_lr2p5e6_head5e4_ep6.npy`
- `lr_backbone=2.75e-6`, `lr_head=5e-4` embeddings:
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_all_lr2p75e6_head5e4_ep1.npy`
  through
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_all_lr2p75e6_head5e4_ep6.npy`
- `lr_backbone=3e-6`, `lr_head=5e-4` embeddings:
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_all_lr3e6_head5e4_ep1.npy`
  through
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_all_lr3e6_head5e4_ep6.npy`

| LR backbone | LR head | Epoch | ARI | NMI | AMI | ARI_hdb |
|---:|---:|---:|---:|---:|---:|---:|
| 1.5e-6 | 5e-4 | 5 | 0.4878 | 0.4614 | 0.4608 | 0.2230 |
| 1.5e-6 | 5e-4 | 6 | 0.4835 | 0.4355 | 0.4348 | 0.2372 |
| 2.0e-6 | 5e-4 | 5 | 0.5143 | 0.4671 | 0.4664 | 0.2039 |
| 2.5e-6 | 5e-4 | 5 | 0.5294 | 0.4619 | 0.4612 | 0.4477 |
| 2.5e-6 | 5e-4 | 6 | 0.4970 | 0.4298 | 0.4291 | 0.2534 |
| 2.75e-6 | 5e-4 | 5 | 0.5255 | 0.4542 | 0.4535 | 0.3397 |
| 2.75e-6 | 5e-4 | 6 | 0.5028 | 0.4375 | 0.4368 | 0.2309 |
| 3.0e-6 | 5e-4 | 5 | 0.5024 | 0.4618 | 0.4611 | 0.2128 |
| 3.0e-6 | 5e-4 | 6 | 0.4661 | 0.3991 | 0.3984 | 0.3691 |

The selected final model is now `lr_backbone=2.5e-6`, `lr_head=5e-4`, epoch 5.
It improves ARI from `0.5143` to `0.5294` and also gives a stronger auxiliary
HDBSCAN ARI (`0.4477`).

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
the selected final model (`0.5294`).

The selected full-unfreeze recipe was also retested with closer temperatures,
`TEMP=0.04` and `TEMP=0.06`, while keeping `lr_backbone=2.5e-6` and
`lr_head=5e-4`.

- `TEMP=0.04` embeddings:
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_all_lr2p5e6_head5e4_temp004_ep1.npy`
  through
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_all_lr2p5e6_head5e4_temp004_ep6.npy`
- `TEMP=0.06` embeddings:
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_all_lr2p5e6_head5e4_temp006_ep1.npy`
  through
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_all_lr2p5e6_head5e4_temp006_ep6.npy`

| Temp | LR backbone | LR head | Epoch | ARI | NMI | AMI | ARI_hdb |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 2.5e-6 | 5e-4 | 5 | 0.5294 | 0.4619 | 0.4612 | 0.4477 |
| 0.04 | 2.5e-6 | 5e-4 | 5 | 0.4969 | 0.4622 | 0.4616 | 0.2010 |
| 0.04 | 2.5e-6 | 5e-4 | 6 | 0.4851 | 0.4193 | 0.4186 | 0.1948 |
| 0.06 | 2.5e-6 | 5e-4 | 5 | 0.5265 | 0.4739 | 0.4732 | 0.2319 |
| 0.06 | 2.5e-6 | 5e-4 | 6 | 0.5257 | 0.4594 | 0.4588 | 0.4037 |

This keeps `TEMP=0.05` for the selected paper-primary ARI recipe. `TEMP=0.06`
is close on ARI and has higher NMI/AMI at epoch 5, but lower auxiliary HDBSCAN
ARI.

## SSL Method Check

The previous sections tune the MoCo-style queue recipe. A separate SSL-method
comparison was started from the same DINOv3 ConvNeXt-B backbone and the same
wafer split. SimCLR already gives a substantially stronger paper-primary NCD
score, although its HDBSCAN branch is weaker.

- SimCLR embeddings:
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ssl_embeddings\simclr_ep1.npy`
  through
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ssl_embeddings\simclr_ep5.npy`
- Barlow Twins embeddings:
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ssl_embeddings\barlow_ep1.npy`
  through
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ssl_embeddings\barlow_ep5.npy`
- VICReg embeddings:
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ssl_embeddings\vicreg_ep1.npy`
  through
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ssl_embeddings\vicreg_ep5.npy`

| Method | Epoch | ARI | NMI | AMI | ARI_hdb |
|---|---:|---:|---:|---:|---:|
| MoCo-style selected | 5 | 0.5294 | 0.4619 | 0.4612 | 0.4477 |
| SimCLR | 1 | 0.5287 | 0.4784 | 0.4777 | 0.3397 |
| SimCLR | 2 | 0.5699 | 0.5102 | 0.5096 | 0.0000 |
| SimCLR | 3 | 0.5935 | 0.5227 | 0.5222 | 0.0011 |
| SimCLR | 4 | 0.5217 | 0.4730 | 0.4723 | 0.0994 |
| SimCLR | 5 | 0.5976 | 0.5258 | 0.5252 | 0.0582 |
| BYOL | 1 | 0.0671 | 0.1054 | 0.1041 | 0.3587 |
| BYOL | 5 | 0.0190 | 0.0551 | 0.0536 | 0.3407 |
| SimSiam | 1 | 0.4146 | 0.3732 | 0.3725 | 0.4151 |
| SimSiam | 2 | 0.3956 | 0.3656 | 0.3648 | 0.5143 |
| SimSiam | 5 | 0.4307 | 0.4412 | 0.4405 | 0.4022 |
| Barlow Twins | 1 | 0.3517 | 0.3782 | 0.3774 | 0.1529 |
| Barlow Twins | 2 | 0.3536 | 0.3376 | 0.3368 | 0.3993 |
| Barlow Twins | 3 | 0.4099 | 0.3980 | 0.3973 | 0.3947 |
| Barlow Twins | 4 | 0.4332 | 0.4029 | 0.4022 | 0.3819 |
| Barlow Twins | 5 | 0.4031 | 0.4246 | 0.4238 | 0.3773 |
| VICReg | 1 | 0.5043 | 0.5030 | 0.5024 | 0.4510 |
| VICReg | 2 | 0.3858 | 0.3611 | 0.3603 | 0.4492 |
| VICReg | 3 | 0.4952 | 0.4799 | 0.4792 | 0.3137 |
| VICReg | 4 | 0.5295 | 0.5035 | 0.5029 | 0.2523 |
| VICReg | 5 | 0.4904 | 0.4782 | 0.4776 | 0.2219 |

This is the current best paper-primary result. It changes the method family, so
the paper should present it as a method-level improvement rather than as a
minor hyperparameter refinement. HDBSCAN remains weaker for SimCLR, so the
operational grouping branch still needs separate tuning. SimSiam is weaker for
paper-primary NCD, but its epoch-2 HDBSCAN ARI (`0.5143`) is currently the best
operational HDBSCAN result in this SSL-method sweep. Barlow Twins improves over
raw DINOv3 but does not beat the selected SimCLR paper-primary result. VICReg
peaks near the MoCo-style recipe and is also below SimCLR.

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
default-head full-unfreeze HDBSCAN result (`0.4171`), so it is not the main
paper setting.

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

The conservative threshold ties the default-head epoch-3 score but does not add
a gain over the selected `lr_head=5e-4` final model. The looser threshold is
weaker.

## Interpretation

Use this table for the main paper claim:

> Starting from a generic FCMAE visual encoder, replacing the initialization
> with a stronger self-supervised DINOv3 backbone improves novel-class
> discovery. Same-dimension PCA tuning gives a small additional gain, and
> wafer-domain contrastive fine-tuning gives the largest improvement on
> held-out classes that are not used in either the supervised CNN stage or the
> unlabeled wafer fine-tuning stage.

The HDBSCAN auto-k score is intentionally not the primary claim for this
waterfall. HDBSCAN is useful operationally, but it is not strictly monotonic
with the paper-primary all-sample NCD score. The selected final model is also
strong on the auxiliary branch (`0.4477` HDBSCAN ARI), but the paper-safe
statement should remain about NCD-style all-sample clustering (`k=3`) because
HDBSCAN depends on noise/threshold choices.
