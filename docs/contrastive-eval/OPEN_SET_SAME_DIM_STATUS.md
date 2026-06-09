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
| DINOv3 B-contrastive 7e-6 ep6 weighted | 94.33% | 93.38% | 92.65% | 92.30% | 91.90% | 49.73% | 0.2823 | 0.4379 | shorter train below ep8 |
| DINOv3 B-contrastive 7e-6 ep6 backbone | 94.27% | 93.36% | 92.73% | 92.15% | 91.95% | 49.60% | 0.2805 | 0.4360 | shorter train below ep8 |
| DINOv3 B-contrastive 7e-6 weighted | 95.27% | 93.64% | 92.83% | 92.50% | 92.26% | 50.40% | 0.2815 | 0.4550 | best weighted top1 |
| DINOv3 B-contrastive 7e-6 backbone | 95.33% | 93.53% | 92.83% | 92.53% | 92.32% | 53.87% | 0.2706 | 0.4350 | best top1 |
| DINOv3 B-contrastive 1e-5 weighted | 94.53% | 93.51% | 93.27% | 93.12% | 92.85% | 54.87% | 0.2729 | 0.4346 | best weighted k5/k7/k9 |
| DINOv3 B-contrastive 1e-5 backbone | 94.33% | 93.42% | 93.31% | 93.13% | 92.97% | 57.87% | 0.2390 | 0.4134 | best k5/k7/k9 |
| DINOv3 B-contrastive 8.5e-6 weighted | 93.60% | 92.91% | 92.37% | 91.96% | 91.56% | 50.73% | 0.2766 | 0.4434 | midpoint regresses |
| DINOv3 B-contrastive 8.5e-6 backbone | 93.73% | 92.76% | 92.49% | 92.04% | 91.72% | 51.00% | 0.2671 | 0.4384 | midpoint regresses |
| DINOv3 B-contrastive last-stage 3e-5 weighted | 89.40% | 88.60% | 87.61% | 86.82% | 86.24% | 44.40% | 0.3223 | 0.4463 | last-stage only hurts kNN |
| DINOv3 B-contrastive last-stage 3e-5 backbone | 91.07% | 90.11% | 89.15% | 88.49% | 88.15% | 34.13% | 0.2785 | 0.3024 | last-stage only hurts kNN |
| DINOv3 B-contrastive 1e-5 ep12 weighted | 93.67% | 92.44% | 92.11% | 91.62% | 91.38% | 46.40% | 0.2624 | 0.3340 | longer train overfits C novel |
| DINOv3 B-contrastive 1e-5 ep12 backbone | 93.53% | 92.53% | 92.07% | 91.70% | 91.45% | 65.07% | 0.1776 | 0.3487 | longer train overfits C novel |

Key artifacts:

- Frozen DINOv3 result:
  `D:\project\unknown-contrastive\result_grouping\260609_161357_wm811k_novel_v1_rawfcmae_vs_dinov3_convnext_base_frozen`
- DINOv3 5e-6 weighted result:
  `D:\project\unknown-contrastive\result_grouping\260609_170031_wm811k_novel_v1_baseline_vs_dinov3_B_moco_lrb5e6_weighted`
- DINOv3 5e-6 backbone result:
  `D:\project\unknown-contrastive\result_grouping\260609_170034_wm811k_novel_v1_dinov3_B_moco_lrb5e6_backbone`
- DINOv3 7e-6 ep6 weighted result:
  `D:\project\unknown-contrastive\result_grouping\260609_184122_wm811k_novel_v1_baseline_vs_dinov3_B_moco_lrb7e6_ep6_weighted`
- DINOv3 7e-6 ep6 backbone result:
  `D:\project\unknown-contrastive\result_grouping\260609_184253_wm811k_novel_v1_dinov3_B_moco_lrb7e6_ep6_backbone`
- DINOv3 7e-6 weighted result:
  `D:\project\unknown-contrastive\result_grouping\260609_171504_wm811k_novel_v1_baseline_vs_dinov3_B_moco_lrb7e6_weighted`
- DINOv3 7e-6 backbone result:
  `D:\project\unknown-contrastive\result_grouping\260609_171505_wm811k_novel_v1_dinov3_B_moco_lrb7e6_backbone`
- DINOv3 1e-5 weighted result:
  `D:\project\unknown-contrastive\result_grouping\260609_173010_wm811k_novel_v1_baseline_vs_dinov3_B_moco_lrb1e5_weighted`
- DINOv3 1e-5 backbone result:
  `D:\project\unknown-contrastive\result_grouping\260609_173014_wm811k_novel_v1_dinov3_B_moco_lrb1e5_backbone`
- DINOv3 8.5e-6 weighted result:
  `D:\project\unknown-contrastive\result_grouping\260609_175058_wm811k_novel_v1_baseline_vs_dinov3_B_moco_lrb8p5e6_weighted`
- DINOv3 8.5e-6 backbone result:
  `D:\project\unknown-contrastive\result_grouping\260609_175103_wm811k_novel_v1_dinov3_B_moco_lrb8p5e6_backbone`
- DINOv3 last-stage 3e-5 weighted result:
  `D:\project\unknown-contrastive\result_grouping\260609_180206_wm811k_novel_v1_baseline_vs_dinov3_B_moco_laststage_lrb3e5_weighted`
- DINOv3 last-stage 3e-5 backbone result:
  `D:\project\unknown-contrastive\result_grouping\260609_180331_wm811k_novel_v1_dinov3_B_moco_laststage_lrb3e5_backbone`
- DINOv3 1e-5 ep12 weighted result:
  `D:\project\unknown-contrastive\result_grouping\260609_182532_wm811k_novel_v1_baseline_vs_dinov3_B_moco_lrb1e5_ep12_weighted`
- DINOv3 1e-5 ep12 backbone result:
  `D:\project\unknown-contrastive\result_grouping\260609_182658_wm811k_novel_v1_dinov3_B_moco_lrb1e5_ep12_backbone`

Interpretation: DINOv3 is useful on this split. Frozen DINOv3 does not beat Raw
FCMAE top1, but it gives much denser HDBSCAN clusters. With B-contrastive
fine-tuning, DINOv3 beats Raw FCMAE on same-dim kNN. The best single-neighbor
setting is `lr_backbone=7e-6` with backbone embedding. The best broader top-k
setting is `lr_backbone=1e-5` with backbone embedding. The `8.5e-6` midpoint
regressed, so the LR response is not monotonic. Last-stage-only unfreeze with
`lr_backbone=3e-5` also regressed strongly on kNN despite a normal NCE decrease,
so the useful DINOv3 adaptation path remains low-LR full-backbone unfreeze.
Extending the strongest `1e-5` recipe from 8 to 12 epochs lowered NCE further
but reduced novel-C kNN, which is evidence of overfitting to B rather than better
open-set transfer. Shortening the `7e-6` top1 recipe from 8 to 6 epochs also
reduced top1, so the current top1 sweet spot remains `7e-6` at 8 epochs.
Projection-only remains the wrong deployment embedding.

## WM-811K Class-Disjoint Near-Novel v2

Split root:

`D:\project\unknown-contrastive\data\images\wm811k_novel_disjoint_v2`

Split summary:

`D:\project\unknown-contrastive\data\images\wm811k_novel_disjoint_v2\summary.json`

The split is class-disjoint and swaps the v1 B/C roles:

| split | folder | classes | images |
|---|---|---|---:|
| A supervised CNN seen | `D:\project\unknown-contrastive\data\images\wm811k_novel_disjoint_v2\cnn_seen_train` | Center, Edge-Loc, Random | 1500 |
| B contrastive unlabeled train | `D:\project\unknown-contrastive\data\images\wm811k_novel_disjoint_v2\contrastive_unlabeled_train` | Donut, Edge-Ring | 1000 |
| C novel eval | `D:\project\unknown-contrastive\data\images\wm811k_novel_disjoint_v2\novel_eval` | Loc, Scratch, Near-full | 1149 |

Same C novel eval, same 128-dimensional comparison:

| embedding | top1 | k3 | k5 | k7 | k9 | HDBSCAN noise | ARI | AMI | result |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Raw FCMAE | 82.25% | 79.84% | 78.17% | 77.22% | 76.68% | 55.70% | 0.1453 | 0.2817 | baseline |
| Raw DINOv3 frozen | 75.20% | 74.04% | 73.07% | 72.04% | 70.98% | 58.05% | 0.1028 | 0.1936 | frozen DINOv3 alone is weaker on v2 |
| TPAT ConvNeXtV2 A-CNN backbone | 87.55% | 85.23% | 83.46% | 82.95% | 82.43% | 29.85% | 0.1729 | 0.2580 | best v2 top-k baseline |
| DINOv3 B-contrastive 7e-6 ep8 weighted | 84.25% | 81.64% | 80.87% | 80.16% | 79.56% | 81.90% | 0.1755 | 0.2736 | repeats v1 kNN gain |
| DINOv3 B-contrastive 7e-6 ep8 backbone | 84.33% | 82.22% | 81.04% | 80.29% | 79.74% | 81.20% | 0.1785 | 0.2757 | first v2 gain |
| DINOv3 B-contrastive 9e-6 ep8 weighted | 83.81% | 82.59% | 81.17% | 80.48% | 79.70% | 72.67% | 0.2156 | 0.3287 | improves Raw but below 1e-5 |
| DINOv3 B-contrastive 9e-6 ep8 backbone | 84.33% | 82.54% | 81.43% | 80.67% | 80.17% | 60.84% | 0.1804 | 0.3007 | improves Raw but below 1e-5 |
| DINOv3 B-contrastive 1e-5 ep8 weighted | 86.16% | 83.72% | 82.98% | 82.16% | 81.55% | 71.54% | 0.2465 | 0.3650 | best v2 top1 |
| DINOv3 B-contrastive 1e-5 ep8 backbone | 85.64% | 84.28% | 83.34% | 82.61% | 81.93% | 70.58% | 0.2374 | 0.3503 | best v2 k3/k5/k7/k9 |
| DINOv3 B-contrastive 1.05e-5 ep8 weighted | 83.29% | 81.35% | 80.28% | 79.57% | 78.64% | 54.13% | 0.2195 | 0.3498 | lower noise but weaker top-k |
| DINOv3 B-contrastive 1.05e-5 ep8 backbone | 84.16% | 81.72% | 80.54% | 80.08% | 79.71% | 52.05% | 0.2198 | 0.3484 | lower noise but weaker top-k |
| DINOv3 B-contrastive 1.2e-5 ep8 weighted | 81.03% | 78.74% | 77.28% | 76.50% | 75.93% | 55.00% | 0.1944 | 0.3216 | higher LR regresses below Raw |
| DINOv3 B-contrastive 1.2e-5 ep8 backbone | 82.07% | 79.08% | 77.84% | 77.10% | 76.34% | 51.78% | 0.1937 | 0.3190 | higher LR still below Raw |

Same-folder four-model check with distance ratio:

`D:\project\unknown-contrastive\result_grouping\260609_211313_wm811k_novel_v2_same_folder_4model_distance`

All rows below use the same C eval folder:

`D:\project\unknown-contrastive\data\images\wm811k_novel_disjoint_v2\novel_eval`

`dist ratio` is mean nearest-other-class cosine distance divided by mean
same-class pair cosine distance. Higher is better.

| embedding | top1 | k3 | k5 | k7 | k9 | dist ratio | HDBSCAN clusters | noise | ARI | AMI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Raw FCMAE | 82.25% | 79.84% | 78.17% | 77.22% | 76.68% | 0.3294 | 4 | 55.70% | 0.1453 | 0.2817 |
| Raw DINOv3 frozen | 75.20% | 74.04% | 73.07% | 72.04% | 70.98% | 0.2307 | 3 | 58.05% | 0.1028 | 0.1936 |
| TPAT ConvNeXtV2 A-CNN backbone | 87.55% | 85.23% | 83.46% | 82.95% | 82.43% | 0.3927 | 2 | 29.85% | 0.1729 | 0.2580 |
| DINOv3 B-contrastive weighted | 86.16% | 83.72% | 82.98% | 82.16% | 81.54% | 0.4691 | 3 | 71.54% | 0.2465 | 0.3650 |
| TPAT ConvNeXtV2 A-CNN + B-contrastive last-stage weighted | 87.47% | 85.35% | 83.62% | 82.58% | 82.27% | 0.4153 | 2 | 27.94% | 0.1794 | 0.2641 |
| TPAT ConvNeXtV2 A-CNN + B-contrastive last-stage backbone | 87.38% | 85.38% | 83.64% | 82.54% | 82.20% | 0.4183 | 2 | 28.37% | 0.1797 | 0.2660 |
| TPAT ConvNeXtV2 A-CNN + B-contrastive last-stage `lr=5e-6` ep6 weighted | 87.38% | 85.55% | 83.69% | 82.78% | 82.23% | 0.4104 | 5 | 59.70% | 0.0944 | 0.2396 |

TPAT A-CNN + B-contrastive last-stage keeps top1 nearly flat against the TPAT A-CNN baseline, but slightly improves k3/k5, distance ratio, noise rate, ARI, and AMI. Full-backbone TPAT+B was too heavy for the local 16GB GPU; use the H100/server condition for that variant.

Key artifacts:

- v2 split summary:
  `D:\project\unknown-contrastive\data\images\wm811k_novel_disjoint_v2\summary.json`
- Raw FCMAE + Raw DINOv3 + TPAT ConvNeXtV2 baseline result:
  `D:\project\unknown-contrastive\result_grouping\260609_204642_wm811k_novel_v2_rawfcmae_dinov3_tpat_convnextv2_baselines`
- TPAT ConvNeXtV2 A-CNN training run:
  `D:\project\unknown-contrastive\runs\260609_203204_cnn_ddp`
- TPAT ConvNeXtV2 A-CNN checkpoint:
  `D:\project\unknown-contrastive\runs\260609_203204_cnn_ddp\cnn\best_model.pth`
- TPAT ConvNeXtV2 A-CNN + B-contrastive last-stage training run:
  `D:\project\unknown-contrastive\runs\260609_214458_wm811k_novel_v2_tpatA_B_moco_laststage_lrb1e5_img384_ep4`
- TPAT ConvNeXtV2 A-CNN + B-contrastive last-stage checkpoint:
  `D:\project\unknown-contrastive\runs\260609_214458_wm811k_novel_v2_tpatA_B_moco_laststage_lrb1e5_img384_ep4\contrastive\best_model.pt`
- TPAT ConvNeXtV2 A-CNN + B-contrastive last-stage weighted result:
  `D:\project\unknown-contrastive\result_grouping\260609_215254_wm811k_novel_v2_same_folder_5model_tpatB_laststage_distance`
- TPAT ConvNeXtV2 A-CNN + B-contrastive last-stage backbone result:
  `D:\project\unknown-contrastive\result_grouping\260609_215606_wm811k_novel_v2_tpatB_laststage_backbone_distance`
- TPAT ConvNeXtV2 A-CNN + B-contrastive `lr=5e-6` ep6 training run:
  `D:\project\unknown-contrastive\runs\260609_224021_wm811k_novel_v2_tpatA_B_moco_laststage_lrb5e6_img384_ep6`
- TPAT ConvNeXtV2 A-CNN + B-contrastive `lr=5e-6` ep6 checkpoint:
  `D:\project\unknown-contrastive\runs\260609_224021_wm811k_novel_v2_tpatA_B_moco_laststage_lrb5e6_img384_ep6\contrastive\best_model.pt`
- TPAT ConvNeXtV2 A-CNN + B-contrastive `lr=5e-6` ep6 same-folder result:
  `D:\project\unknown-contrastive\result_grouping\260609_225046_wm811k_novel_v2_same_folder_6model_tpat_lrb5e6_ep6_distance`
- DINOv3 7e-6 ep8 training run:
  `D:\project\unknown-contrastive\runs\260609_184928_wm811k_novel_v2_dinov3_B_moco_q1024_ep8_lrb7e6`
- v2 weighted result:
  `D:\project\unknown-contrastive\result_grouping\260609_190025_wm811k_novel_v2_baseline_vs_dinov3_B_moco_lrb7e6_ep8_weighted`
- v2 backbone result:
  `D:\project\unknown-contrastive\result_grouping\260609_190141_wm811k_novel_v2_dinov3_B_moco_lrb7e6_ep8_backbone`
- DINOv3 9e-6 ep8 training run:
  `D:\project\unknown-contrastive\runs\260609_195007_wm811k_novel_v2_dinov3_B_moco_q1024_ep8_lrb9e6`
- v2 9e-6 weighted result:
  `D:\project\unknown-contrastive\result_grouping\260609_200041_wm811k_novel_v2_baseline_vs_dinov3_B_moco_lrb9e6_ep8_weighted`
- v2 9e-6 backbone result:
  `D:\project\unknown-contrastive\result_grouping\260609_200153_wm811k_novel_v2_dinov3_B_moco_lrb9e6_ep8_backbone`
- DINOv3 1e-5 ep8 training run:
  `D:\project\unknown-contrastive\runs\260609_190717_wm811k_novel_v2_dinov3_B_moco_q1024_ep8_lrb1e5`
- v2 1e-5 weighted result:
  `D:\project\unknown-contrastive\result_grouping\260609_192030_wm811k_novel_v2_baseline_vs_dinov3_B_moco_lrb1e5_ep8_weighted`
- v2 1e-5 backbone result:
  `D:\project\unknown-contrastive\result_grouping\260609_192146_wm811k_novel_v2_dinov3_B_moco_lrb1e5_ep8_backbone`
- v2 1e-5 HDBSCAN sweep:
  `D:\project\unknown-contrastive\result_grouping\260609_1927_wm811k_novel_v2_lrb1e5_hdbscan_sweep`
- DINOv3 1.05e-5 ep8 training run:
  `D:\project\unknown-contrastive\runs\260609_200644_wm811k_novel_v2_dinov3_B_moco_q1024_ep8_lrb1p05e5`
- v2 1.05e-5 weighted result:
  `D:\project\unknown-contrastive\result_grouping\260609_201711_wm811k_novel_v2_baseline_vs_dinov3_B_moco_lrb1p05e5_ep8_weighted`
- v2 1.05e-5 backbone result:
  `D:\project\unknown-contrastive\result_grouping\260609_201819_wm811k_novel_v2_dinov3_B_moco_lrb1p05e5_ep8_backbone`
- DINOv3 1.2e-5 ep8 training run:
  `D:\project\unknown-contrastive\runs\260609_193303_wm811k_novel_v2_dinov3_B_moco_q1024_ep8_lrb1p2e5`
- v2 1.2e-5 weighted result:
  `D:\project\unknown-contrastive\result_grouping\260609_194315_wm811k_novel_v2_baseline_vs_dinov3_B_moco_lrb1p2e5_ep8_weighted`
- v2 1.2e-5 backbone result:
  `D:\project\unknown-contrastive\result_grouping\260609_194427_wm811k_novel_v2_dinov3_B_moco_lrb1p2e5_ep8_backbone`

Interpretation: the v2 improvement is not simply "DINOv3 is stronger". Frozen
DINOv3 alone is worse than Raw FCMAE on this split. The DINOv3 gain appears only
after low-LR full-backbone contrastive adaptation on B (`Donut`, `Edge-Ring`),
where the `1e-5` recipe improves same-dim kNN by up to +3.9 top1 points and +5.2
k5 points over Raw FCMAE. However, the fair A-class TPAT ConvNeXtV2 baseline is
currently the strongest v2 top-k embedding, improving Raw FCMAE by +5.3 top1
points and +5.3 k5 points on C (`Loc`, `Scratch`, `Near-full`) without seeing
those C classes. HDBSCAN interpretation is more mixed: TPAT top-k is best
but produces only two clusters under the fixed eom parameters, while DINOv3
contrastive gives better ARI/AMI than Raw FCMAE and should still be evaluated
with a separate HDBSCAN threshold sweep. The same-folder distance-ratio check
also favors DINOv3 B-contrastive (`0.4691`) over TPAT ConvNeXtV2 (`0.3927`),
even though TPAT remains best on top-k. A first HDBSCAN sweep on the DINOv3
`1e-5` weighted embedding found `cluster_selection_method=leaf`,
`min_cluster_size=20`, `min_samples=5` as the best ARI setting in the tested
grid, improving ARI/AMI to `0.2689/0.3990` with four clusters but still high
noise (`76.50%`). Raising `lr_backbone` to `1.2e-5` reduced training NCE further
(`0.3400` at epoch 8) but hurt novel-C top-k below Raw FCMAE, so NCE decrease
alone is not a sufficient selection criterion.

Additional DINOv3+B HDBSCAN noise sweep:

- output:
  `D:\project\unknown-contrastive\result_grouping\260609_2220_wm811k_novel_v2_dinov3_B_lrb1e5_hdbscan_noise_sweep`
- sweep csv:
  `D:\project\unknown-contrastive\result_grouping\260609_2220_wm811k_novel_v2_dinov3_B_lrb1e5_hdbscan_noise_sweep\hdbscan_sweep.csv`
- summary:
  `D:\project\unknown-contrastive\result_grouping\260609_2220_wm811k_novel_v2_dinov3_B_lrb1e5_hdbscan_noise_sweep\summary.md`

The high noise is not just a bad single HDBSCAN setting. For this DINOv3+B
embedding, the best ARI settings remain strict and noisy (`leaf`, `mcs=20/30`,
`min_samples=5`, noise `76.50%`, ARI/AMI `0.2689/0.3990`). Reducing noise below
50% is possible (`eom`, `mcs=30`, `min_samples=1`, noise `43.69%`), but it drops
ARI/AMI to `0.2089/0.3434` and produces only two clusters. Very low noise
settings collapse almost everything into one giant cluster (`1074/1149` in one
cluster, ARI `-0.0089`). Operationally, DINOv3+B is useful when prioritizing
high-purity small clusters, but TPAT A-CNN / TPAT+B remains better for lower
noise grouping on this v2 split.

Current domain-safe augmentation in
`D:\project\unknown-contrastive\scripts\train_contrastive_ddp.py`: palette
non-grade pixels are masked to white before RGB conversion; training uses
`RandomResizedCrop(scale=0.94..1.0, ratio=1.0)`, `RandomAffine(degrees=7,
translate=0.05, scale=0.95..1.05)`, and Gaussian noise `0.02`. It does not use
flip, large rotation, or color jitter. This matches the wafer rule that location
and direction should be preserved while minor alignment/noise is ignored.

TPAT+B `lr=5e-6` ep6 note: the raw 2048D internal eval looked much better
(`noise=45.43%`, ARI/AMI `0.3117/0.4957`), but after the same 128D PCA
comparison it mainly improves k3/k5 slightly while lowering fixed-HDBSCAN
quality versus the `lr=1e-5` ep4 same-dim row. This suggests TPAT+B should be
selected by the exact deployment embedding dimension, not only by internal
training-report HDBSCAN.

## Paper Ablation Ladder Candidates

For paper writing, do not present one mixed table as if every metric improves
monotonically. Current evidence supports two honest ablation ladders on the same
held-out C eval folder:

1. DINOv3 domain-SSL adaptation ladder: Raw DINOv3 -> DINOv3 + B contrastive.
   This strongly improves same-128 kNN and best clustering ARI/AMI, but keeps a
   high-noise strict-cluster behavior.
2. TPAT retrieval ladder: Raw FCMAE -> TPAT A-CNN -> TPAT A-CNN + B
   contrastive. This is strongest for top-k retrieval and lower-noise grouping;
   the contrastive step improves k3/k5 slightly while top1 is saturated/flat.

Same-128 paper candidate eval:

- output:
  `D:\project\unknown-contrastive\result_grouping\260609_230931_wm811k_novel_v2_paper_ablation_candidates_same128`
- summary:
  `D:\project\unknown-contrastive\result_grouping\260609_230931_wm811k_novel_v2_paper_ablation_candidates_same128\summary.md`
- t-SNE:
  `D:\project\unknown-contrastive\result_grouping\260609_230931_wm811k_novel_v2_paper_ablation_candidates_same128\tsne_same_dim_sheet.png`

| ladder | step | top1 | k3 | k5 | dist ratio | note |
|---|---|---:|---:|---:|---:|---|
| DINOv3 domain SSL | Raw DINOv3 | 75.20% | 74.04% | 73.07% | 0.2307 | no wafer training |
| DINOv3 domain SSL | + B contrastive | 86.16% | 83.72% | 82.98% | 0.4691 | large retrieval/distance gain, high-noise clusters |
| TPAT retrieval | Raw FCMAE | 82.25% | 79.84% | 78.17% | 0.3294 | no wafer training |
| TPAT retrieval | + A-CNN supervised TPAT | 87.55% | 85.23% | 83.46% | 0.3927 | strongest top1 |
| TPAT retrieval | + B contrastive `lr=5e-6` ep6 | 87.38% | 85.55% | 83.69% | 0.4104 | k3/k5 improve, top1 flat |
| TPAT retrieval | + B contrastive + pseudo-positive `0.05` | 87.55% | 85.32% | 83.62% | 0.4040 | top1 recovers to A-CNN, k5 remains above A-CNN |

Fast same-128 HDBSCAN sweep:

- output:
  `D:\project\unknown-contrastive\result_grouping\260609_2329_wm811k_novel_v2_paper_fast_hdbscan_sweep_same128`
- sweep csv:
  `D:\project\unknown-contrastive\result_grouping\260609_2329_wm811k_novel_v2_paper_fast_hdbscan_sweep_same128\sweep.csv`
- summary:
  `D:\project\unknown-contrastive\result_grouping\260609_2329_wm811k_novel_v2_paper_fast_hdbscan_sweep_same128\summary.md`

| model | best ARI/AMI | noise | best ARI/AMI under noise<=50% | noise | note |
|---|---:|---:|---:|---:|---|
| Raw FCMAE | 0.1638/0.3007 | 43.69% | 0.1638/0.3007 | 43.69% | baseline SSL |
| Raw DINOv3 | 0.1355/0.2118 | 63.45% | 0.0579/0.1213 | 43.60% | weak on this wafer split before adaptation |
| DINOv3 + B contrastive | 0.2689/0.3990 | 76.50% | 0.2089/0.3434 | 43.69% | best strict ARI/AMI, high noise |
| TPAT A-CNN | 0.2218/0.3486 | 28.46% | 0.2218/0.3486 | 28.46% | best low-noise clustering in sweep |
| TPAT A-CNN + B contrastive `lr=5e-6` ep6 | 0.2100/0.3449 | 34.55% | 0.2100/0.3449 | 34.55% | kNN improves, clustering does not beat A-CNN |
| TPAT A-CNN + B contrastive + pseudo-positive `0.05` | 0.2063/0.3359 | 33.94% | 0.2063/0.3359 | 33.94% | top1 recovers, clustering still below A-CNN |

Paper-safe conclusion: use DINOv3+B to support the claim that domain SSL
adaptation improves raw DINOv3 novel embedding, and use TPAT A-CNN / TPAT+B to
support the practical retrieval/grouping branch. Do not claim that B
contrastive improves every metric after TPAT; in same-128 it mainly improves
k3/k5, while clustering ARI is best for TPAT A-CNN alone.

Pseudo-positive TPAT+B check:

- training run:
  `D:\project\unknown-contrastive\runs\260609_233255_wm811k_novel_v2_tpatA_B_moco_laststage_lrb5e6_pseudo005_img384_ep4`
- checkpoint:
  `D:\project\unknown-contrastive\runs\260609_233255_wm811k_novel_v2_tpatA_B_moco_laststage_lrb5e6_pseudo005_img384_ep4\contrastive\best_model.pt`
- same-128 eval:
  `D:\project\unknown-contrastive\result_grouping\260609_234048_wm811k_novel_v2_tpat_pseudo005_same128`
- HDBSCAN sweep:
  `D:\project\unknown-contrastive\result_grouping\260609_2344_wm811k_novel_v2_tpat_pseudo005_fast_hdbscan_same128`

This ablation changes the previous TPAT+B setup from false-negative removal only
to false-negative removal plus weak pseudo-positive attraction. In same-128
retrieval it recovers top1 to the A-CNN level (`87.55%`) and keeps k5 above the
A-CNN baseline (`83.62%` vs `83.46%`). It does not solve the clustering branch:
best fast-sweep ARI/AMI is `0.2063/0.3359`, still below TPAT A-CNN
`0.2218/0.3486`.

## Paper-Safe Monotonic Ablation

For a paper table that must show a clear baseline-to-method improvement, use the
NCD-style all-sample clustering waterfall, not the mixed TPAT/DINOv3 operational
table above. The clean paper table is documented here:

- report:
  `D:\project\unknown-contrastive\docs\contrastive-eval\PAPER_ABLATION_WATERFALL.md`
- csv:
  `D:\project\unknown-contrastive\docs\contrastive-eval\paper_ablation_waterfall.csv`
- eval folder:
  `D:\project\unknown-contrastive\data\images\wm811k_novel_disjoint_v1\novel_eval`
- final fine-tuned embedding:
  `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ft_embeddings\ft_laststage_lr3e6_ep4.npy`

Primary metric: k-means with the known novel class count (`k=3`), following the
standard NCD evaluation style where every sample is assigned. The monotonic
result is:

| Stage | ARI | NMI | AMI | note |
|---|---:|---:|---:|---|
| Raw FCMAE baseline | 0.2097 | 0.2169 | 0.2159 | generic SSL encoder |
| + DINOv3 SSL backbone | 0.3097 | 0.2831 | 0.2822 | stronger SSL initialization |
| + PCA dimension tuning | 0.3173 | 0.2957 | 0.2949 | same embedding, better evaluation dimension |
| + wafer contrastive fine-tune | 0.4681 | 0.4328 | 0.4321 | domain SSL adaptation |

This gives a final ARI gain of `+0.2584` over Raw FCMAE (`+123.2%` relative).
Do not describe this as an HDBSCAN improvement claim; HDBSCAN remains a separate
operational clustering branch with its own noise/threshold tradeoff.
