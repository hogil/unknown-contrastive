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

Completed result:

`D:\project\unknown-contrastive\result_grouping\260609_100040_wm811k_train_eval_epoch_sweep_same128`

| stage | input dim | same dim | top1 | k3 | k5 | k7 | k9 | HDBSCAN clusters | noise | ARI | AMI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw FCMAE | 1024 | 128 | 0.7068 | 0.6703 | 0.6496 | 0.6313 | 0.6186 | 6 | 0.5466 | 0.0832 | 0.3026 |
| CNN backbone | 1024 | 128 | 0.7027 | 0.6694 | 0.6474 | 0.6301 | 0.6131 | 6 | 0.5548 | 0.0769 | 0.2729 |
| WM contrastive epoch 5 | 128 | 128 | 0.4863 | 0.4644 | 0.4375 | 0.4231 | 0.4119 | 4 | 0.1890 | 0.0230 | 0.1110 |
| WM contrastive epoch 10 | 128 | 128 | 0.5178 | 0.4772 | 0.4542 | 0.4393 | 0.4291 | 2 | 0.2890 | 0.0477 | 0.1297 |
| WM contrastive epoch 15 | 128 | 128 | 0.5151 | 0.4795 | 0.4641 | 0.4511 | 0.4437 | 3 | 0.2589 | 0.0455 | 0.1532 |
| WM contrastive final | 128 | 128 | 0.5329 | 0.4936 | 0.4693 | 0.4511 | 0.4472 | 3 | 0.2616 | 0.0457 | 0.1527 |

Interpretation: this real-WM adaptation run did not improve embedding quality.
Projection embeddings collapsed relative to Raw FCMAE/CNN backbone. The training
log also showed a DCL numerical failure at epoch 20 (`loss=-2.4e8`) caused by
all negatives being masked for some anchors. DCL now skips rows with no valid
negative instead of feeding an all-`-1e9` row to `logsumexp`.

Next direction: run a backbone-preserving adaptation check. Use NCE (not DCL),
small backbone LR, no local/pseudo attraction at first, and evaluate backbone
mode so the raw FCMAE structure is not discarded by a projection-only head.

## Real WM Weighted-Concat And Strong-Pseudo Checks

DCL projection-only weighted concat:

`D:\project\unknown-contrastive\result_grouping\260609_103310_wm811k_train_eval_weighted_concat_same128`

| stage | top1 | k5 | HDBSCAN clusters | noise | ARI | AMI |
|---|---:|---:|---:|---:|---:|---:|
| Raw FCMAE | 0.7068 | 0.6496 | 6 | 0.5466 | 0.0832 | 0.3026 |
| DCL weighted concat epoch 5 | 0.7055 | 0.6496 | 6 | 0.5466 | 0.0832 | 0.3026 |
| DCL weighted concat epoch 10 | 0.7068 | 0.6496 | 6 | 0.5466 | 0.0832 | 0.3026 |
| DCL weighted concat final | 0.7068 | 0.6496 | 6 | 0.5466 | 0.0832 | 0.3026 |

Strong pseudo-positive NCE run:

`D:\project\unknown-contrastive\runs\260609_103619_wm811k500_frozen_nce_pseudo1_topk5_sim082_b48`

Projection-only result:

`D:\project\unknown-contrastive\result_grouping\260609_110221_wm811k_train_eval_pseudo_strong_projection_same128`

Weighted-concat result:

`D:\project\unknown-contrastive\result_grouping\260609_110449_wm811k_train_eval_pseudo_strong_weighted_concat_same128`

| stage | top1 | k5 | HDBSCAN clusters | noise | ARI | AMI |
|---|---:|---:|---:|---:|---:|---:|
| Raw FCMAE | 0.7068 | 0.6496 | 6 | 0.5466 | 0.0832 | 0.3026 |
| strong-pseudo projection epoch 4 | 0.4493 | 0.3652 | 4 | 0.2548 | 0.0826 | 0.1851 |
| strong-pseudo projection final | 0.4000 | 0.3479 | 2 | 0.0740 | 0.0237 | 0.1060 |
| strong-pseudo weighted concat epoch 8 | 0.7055 | 0.6477 | 6 | 0.5301 | 0.0843 | 0.3051 |
| strong-pseudo weighted concat final | 0.7041 | 0.6482 | 6 | 0.5260 | 0.0837 | 0.3048 |

Interpretation: projection-only is still worse than Raw FCMAE. Weighted concat
preserves Raw FCMAE but does not improve top-k metrics; it only slightly lowers
HDBSCAN noise. The next quick ablation should reduce global NCE pressure
(`--no-queue`) and make pseudo-positive alignment dominate, otherwise the
projection head keeps collapsing broad groups.

## 2026-06-09 Contrastive Learning Fix

Problem found after comparing old and new tracks:

- old stable track used batch 8 and queue 4096, and global/queue losses decreased;
- new script track used much larger queue pressure and the NCE/DCL loss either
  rose over epochs or collapsed broad groups;
- current `QueueBank` also used a full random queue from step 1 instead of only
  real queued embeddings;
- false-negative masking could remove every negative for an anchor, making NCE
  effectively zero after early saturation.

Code changes:

- `D:\project\unknown-contrastive\scripts\train_contrastive_ddp.py`
  - default `QUEUE_SIZE` restored to `4096`;
  - queue now tracks `filled` and only exposes actual queued embeddings;
  - progress log prints `q=<filled>/<size>`;
  - false-negative mask keeps at least one lowest-similarity safe negative;
  - `--loss-mode simsiam` added as a negative-free diagnostic alternative;
  - `--lr-backbone 0.0` now stays exactly zero under the cosine scheduler.
- `D:\project\unknown-contrastive\scripts\eval_open_set_embeddings.py`
- `D:\project\unknown-contrastive\scripts\predict_grouping_prod.py`
- `D:\project\unknown-contrastive\scripts\make_tsne_stages.py`
  - inference loaders tolerate SimSiam `pred.*` checkpoint keys.

Verification runs:

`D:\project\unknown-contrastive\runs\260609_132852_smoke_nce_queue_floor_check`

- queue filled from `8/64` to `64/64`, not from random negatives;
- NCE stayed live: epoch 1 `0.2431`, epoch 2 `0.1877`.

`D:\project\unknown-contrastive\runs\260609_132947_wm811k500_nce_queuefix_floor_losscheck`

- real WM train split, frozen raw FCMAE, NCE + NeCo, queue 4096, 25% sample;
- epoch NCE: `0.1232 -> 0.1176 -> 0.1184`;
- this confirms the loop no longer explodes upward and no longer dies at
  `0.0000` after false-negative masking.

Interpretation: this is a learning-loop repair, not yet a final metric win.
The next real experiment should use this fixed queue path for a full run, then
compare projection/backbone/weighted-concat embeddings against Raw FCMAE.

## Real WM No-Queue Pseudo-Dominant Checks

Pseudo-dominant no-queue projection dim 128 run:

`D:\project\unknown-contrastive\runs\260609_110840_wm811k500_frozen_nce_noqueue_pseudo500_topk5_sim082_b64`

Projection-only result:

`D:\project\unknown-contrastive\result_grouping\260609_112536_wm811k_train_eval_pseudo500_noqueue_projection_same128`

Weighted-concat result:

`D:\project\unknown-contrastive\result_grouping\260609_112734_wm811k_train_eval_pseudo500_noqueue_weighted_concat_same128`

Pseudo-dominant no-queue projection dim 512 run:

`D:\project\unknown-contrastive\runs\260609_112942_wm811k500_frozen_nce_noqueue_pseudo500_topk5_sim082_proj512_b64`

Projection-only result:

`D:\project\unknown-contrastive\result_grouping\260609_113906_wm811k_train_eval_pseudo500_noqueue_proj512_projection_same128`

Weighted-concat result:

`D:\project\unknown-contrastive\result_grouping\260609_114043_wm811k_train_eval_pseudo500_noqueue_proj512_weighted_concat_same128`

| stage | top1 | k3 | k5 | k7 | k9 | HDBSCAN clusters | noise | ARI | AMI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw FCMAE | 0.7068 | 0.6703 | 0.6496 | 0.6313 | 0.6186 | 6 | 0.5466 | 0.0832 | 0.3026 |
| CNN backbone | 0.7027 | 0.6694 | 0.6474 | 0.6301 | 0.6131 | 6 | 0.5548 | 0.0769 | 0.2729 |
| p50 no-queue projection 512 | 0.6986 | 0.6420 | 0.6156 | 0.5971 | 0.5776 | 2 | 0.2411 | 0.0380 | 0.0586 |
| p100 no-queue projection 512 | 0.7027 | 0.6429 | 0.6181 | 0.5941 | 0.5761 | 2 | 0.2644 | 0.0406 | 0.0609 |
| p200 no-queue projection 512 | 0.6959 | 0.6425 | 0.6233 | 0.5988 | 0.5793 | 2 | 0.2616 | 0.0410 | 0.0618 |
| p500 no-queue projection 128 | 0.6795 | 0.6242 | 0.5967 | 0.5732 | 0.5524 | 2 | 0.2110 | 0.0223 | 0.0768 |
| p500 no-queue weighted concat 128 | 0.7068 | 0.6703 | 0.6496 | 0.6313 | 0.6186 | 6 | 0.5466 | 0.0833 | 0.3032 |
| p500 no-queue projection 512 | 0.6932 | 0.6470 | 0.6247 | 0.6025 | 0.5857 | 2 | 0.2521 | 0.0403 | 0.0610 |
| p500 no-queue weighted concat 512 | 0.7068 | 0.6703 | 0.6496 | 0.6313 | 0.6186 | 6 | 0.5466 | 0.0833 | 0.3032 |

Interpretation: projection dim 512 is better than projection dim 128, but it
still does not beat Raw FCMAE on WM known-class neighbor metrics. Weighted concat
only preserves Raw FCMAE; it does not add measurable improvement. Current real
WM evidence says projection-only contrastive is the bottleneck, not vector
dimension alone. Among pseudo weights 50/100/200/500, pseudo weight 100 gives
the best projection top1, while pseudo weight 500 gives the best projection k5;
neither beats Raw FCMAE.

## WM-811K Class-Disjoint Near-Novel v1

Split root:

`D:\project\unknown-contrastive\data\images\wm811k_novel_disjoint_v1`

Split summary:

`D:\project\unknown-contrastive\data\images\wm811k_novel_disjoint_v1\summary.json`

The split is class-disjoint:

| split | folder | classes | images |
|---|---|---|---:|
| A supervised CNN seen | `D:\project\unknown-contrastive\data\images\wm811k_novel_disjoint_v1\cnn_seen_train` | Center, Edge-Ring, Near-full | 1149 |
| B contrastive unlabeled train | `D:\project\unknown-contrastive\data\images\wm811k_novel_disjoint_v1\contrastive_unlabeled_train` | Loc, Scratch | 1000 |
| C novel eval | `D:\project\unknown-contrastive\data\images\wm811k_novel_disjoint_v1\novel_eval` | Donut, Edge-Loc, Random | 1500 |

A-CNN checkpoint:

`D:\project\unknown-contrastive\runs\260609_141113_cnn_ddp\cnn\best_model.pth`

Raw-FCMAE init B-contrastive checkpoint:

`D:\project\unknown-contrastive\runs\260609_142557_wm811k_novel_v1_rawfcmae_B_moco_q1024_ep8\contrastive\best_model.pt`

A-CNN init B-contrastive checkpoint:

`D:\project\unknown-contrastive\runs\260609_144848_wm811k_novel_v1_Acnn_B_moco_q1024_ep8\contrastive\best_model.pt`

Training-loop observation:

- Raw-FCMAE init NCE decreased from `1.4210` to `0.4751`.
- A-CNN init NCE decreased from `1.2256` to `0.4231`.
- This confirms the MoCo EMA queue path optimizes normally on B. It does not
  by itself prove novel C metric improvement.

Same C novel eval, same 128-dimensional comparison:

| embedding | top1 | k3 | k5 | HDBSCAN noise | ARI | purity | result |
|---|---:|---:|---:|---:|---:|---:|---|
| Raw FCMAE | 94.00% | 92.62% | 91.43% | 64.13% | 0.0824 | 0.5680 | baseline |
| Raw-FCMAE init B-contrastive projection | 92.73% | 91.89% | 91.16% | 49.87% | 0.2828 | 0.7473 | top-k worse, clustering cleaner |
| Raw-FCMAE init B-contrastive backbone | 94.27% | 92.91% | 92.40% | 61.00% | 0.0949 | 0.6247 | small kNN gain |
| Raw-FCMAE init B-contrastive weighted | 94.13% | 92.96% | 92.41% | 61.07% | 0.0933 | 0.6220 | small kNN gain |
| A-CNN init B-contrastive projection | 93.87% | 92.27% | 91.05% | 59.80% | 0.2080 | 0.6160 | projection still weak |
| A-CNN init B-contrastive backbone | 94.60% | 93.11% | 92.07% | 63.60% | 0.1417 | 0.5780 | best backbone |
| A-CNN init B-contrastive weighted | 94.67% | 93.11% | 92.07% | 63.60% | 0.1407 | 0.5773 | best top1 |

Key artifacts:

- Raw-FCMAE baseline result:
  `D:\project\unknown-contrastive\result_grouping\260609_140755_wm811k_novel_v1_raw_fcmae_baseline`
- Raw vs A-CNN result:
  `D:\project\unknown-contrastive\result_grouping\260609_142348_wm811k_novel_v1_raw_vs_Acnn`
- Raw-FCMAE init weighted t-SNE:
  `D:\project\unknown-contrastive\result_grouping\260609_144429_wm811k_novel_v1_baseline_vs_rawfcmae_B_moco_ep8_weighted\tsne_same_dim_sheet.png`
- A-CNN init weighted t-SNE:
  `D:\project\unknown-contrastive\result_grouping\260609_150844_wm811k_novel_v1_baseline_vs_Acnn_B_moco_ep8_weighted\tsne_same_dim_sheet.png`

Interpretation: projection-only is not the right deployment/eval embedding.
The useful signal is in the adapted backbone or backbone-heavy weighted concat.
On this near-novel split, B-contrastive gives a small but real same-dim kNN gain
over Raw FCMAE, with the best top1 from A-CNN init + B-contrastive weighted
concat. The split is still near-novel and Raw FCMAE is already strong, so the
next target should be far-novel or a harder held-out split.

## DINOv3 ConvNeXt-B Near-Novel v1 Sweep

Backbone source:

`hf_hub:timm/convnext_base.dinov3_lvd1689m`

Code support:

- `D:\project\unknown-contrastive\scripts\eval_open_set_embeddings.py` supports
  frozen timm/HF-Hub backbones through `--timm-model`.
- `D:\project\unknown-contrastive\scripts\train_contrastive_ddp.py` supports
  DINOv3 training init through `--backbone-name`.

Same C novel eval, same 128-dimensional comparison:

| embedding | top1 | k3 | k5 | k7 | k9 | HDBSCAN noise | ARI | AMI | result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Raw FCMAE | 94.00% | 92.62% | 91.43% | 90.79% | 90.49% | 64.13% | 0.0824 | 0.2695 | baseline |
| DINOv3 frozen | 93.80% | 92.33% | 91.48% | 91.26% | 90.48% | 24.27% | 0.2621 | 0.2498 | top-k similar, clustering much denser |
| DINOv3 B-contrastive 3e-6 weighted | 93.87% | 92.84% | 92.43% | 91.88% | 91.41% | 53.60% | 0.2235 | 0.3729 | kNN improves over raw except top1 |
| DINOv3 B-contrastive 3e-6 backbone | 93.73% | 92.98% | 92.45% | 91.85% | 91.36% | 53.67% | 0.2210 | 0.3716 | backbone better than projection |
| DINOv3 B-contrastive 5e-6 weighted | 94.27% | 93.76% | 93.11% | 92.64% | 92.10% | 43.20% | 0.2973 | 0.4252 | strong top-k balance |
| DINOv3 B-contrastive 5e-6 backbone | 94.20% | 93.80% | 93.15% | 92.67% | 92.13% | 42.93% | 0.3004 | 0.4283 | best k3/k5 balance |
| DINOv3 B-contrastive 7e-6 weighted | 95.27% | 93.64% | 92.83% | 92.50% | 92.26% | 50.40% | 0.2815 | 0.4550 | best weighted top1 |
| DINOv3 B-contrastive 7e-6 backbone | 95.33% | 93.53% | 92.83% | 92.53% | 92.32% | 53.87% | 0.2706 | 0.4350 | best top1 |
| DINOv3 B-contrastive 1e-5 weighted | 94.53% | 93.51% | 93.27% | 93.12% | 92.85% | 54.87% | 0.2729 | 0.4346 | best weighted k5/k7/k9 |
| DINOv3 B-contrastive 1e-5 backbone | 94.33% | 93.42% | 93.31% | 93.13% | 92.97% | 57.87% | 0.2390 | 0.4134 | best k5/k7/k9 |

Key artifacts:

- Frozen DINOv3 result:
  `D:\project\unknown-contrastive\result_grouping\260609_161357_wm811k_novel_v1_rawfcmae_vs_dinov3_convnext_base_frozen`
- DINOv3 5e-6 weighted result:
  `D:\project\unknown-contrastive\result_grouping\260609_170031_wm811k_novel_v1_baseline_vs_dinov3_B_moco_lrb5e6_weighted`
- DINOv3 5e-6 backbone result:
  `D:\project\unknown-contrastive\result_grouping\260609_170034_wm811k_novel_v1_dinov3_B_moco_lrb5e6_backbone`
- DINOv3 7e-6 weighted result:
  `D:\project\unknown-contrastive\result_grouping\260609_171504_wm811k_novel_v1_baseline_vs_dinov3_B_moco_lrb7e6_weighted`
- DINOv3 7e-6 backbone result:
  `D:\project\unknown-contrastive\result_grouping\260609_171505_wm811k_novel_v1_dinov3_B_moco_lrb7e6_backbone`
- DINOv3 1e-5 weighted result:
  `D:\project\unknown-contrastive\result_grouping\260609_173010_wm811k_novel_v1_baseline_vs_dinov3_B_moco_lrb1e5_weighted`
- DINOv3 1e-5 backbone result:
  `D:\project\unknown-contrastive\result_grouping\260609_173014_wm811k_novel_v1_dinov3_B_moco_lrb1e5_backbone`

Interpretation: DINOv3 is useful on this split. Frozen DINOv3 does not beat Raw
FCMAE top1, but it gives much denser HDBSCAN clusters. With B-contrastive
fine-tuning, DINOv3 beats Raw FCMAE on same-dim kNN. The best single-neighbor
setting is `lr_backbone=7e-6` with backbone embedding. The best broader top-k
setting is `lr_backbone=1e-5` with backbone embedding. Projection-only remains
the wrong deployment embedding.
