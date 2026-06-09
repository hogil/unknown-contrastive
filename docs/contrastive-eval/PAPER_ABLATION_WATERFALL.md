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
