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

Next command after GPU is available:

```powershell
python D:\project\unknown-contrastive\scripts\eval_open_set_embeddings.py --eval-dir D:\project\unknown-contrastive\data\images\wm811k_eval500_512\eval --cnn D:\project\unknown-contrastive\runs\260609_075855_cnn_ddp\cnn\best_model.pth --contrastive clean_contrastive=D:\project\unknown-contrastive\runs\260609_080817_synth_clean_dcl_fn080_pseudo005_local075_freeze_neco0\contrastive\best_model.pt --tag wm811k_eval500_same128_raw_cnn_contrastive --img-size 384 --batch 32 --pca-dim 128 --min-cluster-size 20 --min-samples 5
```

## Active Run

Contrastive run in progress:

`D:\project\unknown-contrastive\runs\260609_080817_synth_clean_dcl_fn080_pseudo005_local075_freeze_neco0`

Expected checkpoint:

`D:\project\unknown-contrastive\runs\260609_080817_synth_clean_dcl_fn080_pseudo005_local075_freeze_neco0\contrastive\best_model.pt`

