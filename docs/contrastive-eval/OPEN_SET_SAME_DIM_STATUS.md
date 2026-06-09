# Open-Set Same-Dim Evaluation Status

Generated: 2026-06-09

## Why

Known-class WM-811K kNN is not the right proof target for contrastive unknown
grouping. The current comparison target is:

- eval on classes not used by the CNN classifier training split;
- enough eval images to reduce n=80 noise;
- include raw FCMAE as a non-TAPT baseline;
- compare all embeddings at the same vector dimension;
- exclude `Normal` from metrics unless explicitly testing noise behavior.

## Script

`D:\project\unknown-contrastive\scripts\eval_open_set_embeddings.py`

The script exports raw and same-dim embeddings, kNN same-class rates, HDBSCAN
metrics, `metrics.json`, `summary.md`, and a t-SNE sheet.

## Synthetic Held-Out Eval

Eval folder:

`D:\project\unknown-contrastive\data\images\synth_clean_contrastive_eval_n50_normal500`

CNN checkpoint:

`D:\project\unknown-contrastive\runs\260609_075855_cnn_ddp\cnn\best_model.pth`

Result:

`D:\project\unknown-contrastive\result_grouping\260609_082252_synth_clean_open_set_raw_fcmae_vs_cnn_same128`

Measured classes: 20  
Measured images: 200  
Ignored: `Normal` 100  
CNN train/eval class overlap: 0

| stage | input dim | same dim | top1 | k3 | k5 | k7 | k9 | HDBSCAN clusters | noise | ARI | AMI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw FCMAE | 1024 | 128 | 0.9750 | 0.9617 | 0.9440 | 0.9229 | 0.8717 | 18 | 0.0450 | 0.8153 | 0.9011 |
| CNN backbone | 1024 | 128 | 1.0000 | 0.9967 | 0.9930 | 0.9900 | 0.9733 | 20 | 0.0050 | 0.9842 | 0.9880 |

Interpretation: the held-out condition is valid, but this synthetic split is
too easy; raw FCMAE already separates it well and synthetic CNN reaches ceiling.

Final contrastive result:

`D:\project\unknown-contrastive\result_grouping\260609_090616_synth_clean_open_set_final_same128`

| stage | input dim | same dim | top1 | k3 | k5 | k7 | k9 | HDBSCAN clusters | noise | ARI | AMI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw FCMAE | 1024 | 128 | 0.9750 | 0.9617 | 0.9440 | 0.9229 | 0.8717 | 18 | 0.0450 | 0.8153 | 0.9011 |
| CNN backbone | 1024 | 128 | 1.0000 | 0.9967 | 0.9930 | 0.9900 | 0.9733 | 20 | 0.0050 | 0.9842 | 0.9880 |
| Synthetic contrastive final | 128 | 128 | 0.9900 | 0.9800 | 0.9710 | 0.9564 | 0.9394 | 17 | 0.0300 | 0.8277 | 0.9210 |

Interpretation: contrastive improves over raw FCMAE on this synthetic held-out
split, but remains below the synthetic supervised CNN backbone ceiling.

## Large Real WM Eval Prepared

Eval folder:

`D:\project\unknown-contrastive\data\images\wm811k_eval500_512\eval`

Source:

`D:\project\unknown-contrastive\data\raw\wm811k\LSWMD.pkl`

Summary:

`D:\project\unknown-contrastive\data\images\wm811k_eval500_512\summary.json`

Class counts:

| class | images |
|---|---:|
| Center | 500 |
| Donut | 500 |
| Edge-Loc | 500 |
| Edge-Ring | 500 |
| Loc | 500 |
| Near-full | 149 |
| Random | 500 |
| Scratch | 500 |

Total images: 3,649

Result with the synthetic-trained contrastive checkpoint:

`D:\project\unknown-contrastive\result_grouping\260609_090842_wm811k_eval500_same128_raw_cnn_contrastive`

| stage | input dim | same dim | top1 | k3 | k5 | k7 | k9 | HDBSCAN clusters | noise | ARI | AMI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw FCMAE | 1024 | 128 | 0.7668 | 0.7364 | 0.7211 | 0.7088 | 0.6998 | 12 | 0.5944 | 0.0637 | 0.3067 |
| CNN backbone | 1024 | 128 | 0.7597 | 0.7328 | 0.7182 | 0.7075 | 0.6996 | 11 | 0.5865 | 0.0644 | 0.2714 |
| Synthetic contrastive final | 128 | 128 | 0.6287 | 0.5967 | 0.5877 | 0.5801 | 0.5717 | 2 | 0.0641 | 0.0131 | 0.0518 |

Interpretation: synthetic contrastive does not transfer to real WM here. It
compresses real WM into two broad clusters, while raw FCMAE/CNN keep more
class-neighbor structure. Next experiment should adapt contrastive directly on
real WM unlabeled data, preferably from raw FCMAE rather than synthetic TAPT.

Reference command for final same-dim eval:

```powershell
python D:\project\unknown-contrastive\scripts\eval_open_set_embeddings.py --eval-dir D:\project\unknown-contrastive\data\images\wm811k_eval500_512\eval --cnn D:\project\unknown-contrastive\runs\260609_075855_cnn_ddp\cnn\best_model.pth --contrastive synthetic_contrastive=D:\project\unknown-contrastive\runs\260609_083231_synth_clean_dcl_fn080_pseudo005_local075_freeze_neco0_b16_logged\contrastive\best_model.pt --tag wm811k_eval500_same128_raw_cnn_contrastive --img-size 384 --batch 32 --pca-dim 128 --min-cluster-size 20 --min-samples 5
```

## Active Run

Completed synthetic contrastive run:

`D:\project\unknown-contrastive\runs\260609_083231_synth_clean_dcl_fn080_pseudo005_local075_freeze_neco0_b16_logged`

Checkpoint:

`D:\project\unknown-contrastive\runs\260609_083231_synth_clean_dcl_fn080_pseudo005_local075_freeze_neco0_b16_logged\contrastive\best_model.pt`

## Real WM Adaptation Run

Real WM split:

`D:\project\unknown-contrastive\data\images\wm811k_500_train_eval_512`

Train:

`D:\project\unknown-contrastive\data\images\wm811k_500_train_eval_512\train`

Eval:

`D:\project\unknown-contrastive\data\images\wm811k_500_train_eval_512\eval`

Counts:

| split | images | notes |
|---|---:|---|
| train | 2,919 | class folders are hidden from contrastive loss; labels only define source buckets |
| eval | 730 | labels used only for metrics |

Run in progress:

`D:\project\unknown-contrastive\runs\260609_091818_wm811k500_rawfcmae_dcl_fn080_pseudo005_local075_freeze_b32_logged`

Important setup:

- backbone source: `D:\project\unknown-contrastive\weights\convnextv2_base.fcmae_ft_in22k_in1k_384.pth`
- no synthetic CNN/TAPT backbone checkpoint
- frozen backbone, projection dim 128, batch 32
- DCL + queue + ignore-neg-sim 0.80 + pseudo-positive 0.05 + local 0.75

Expected checkpoint:

`D:\project\unknown-contrastive\runs\260609_091818_wm811k500_rawfcmae_dcl_fn080_pseudo005_local075_freeze_b32_logged\contrastive\best_model.pt`
