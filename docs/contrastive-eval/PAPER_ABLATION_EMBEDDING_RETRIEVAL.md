# Paper Ablation Embedding Retrieval Metrics

- eval folder: `D:\project\unknown-contrastive\data\images\wm811k_novel_disjoint_v1\novel_eval`
- metric: same-class kNN purity on held-out novel classes. `top5` is the average same-class fraction among the five nearest neighbors, not hit@5.
- dist ratio: nearest-other-class cosine distance divided by same-class pair cosine distance. Higher is better.

| Stage | input dim | eval dim | top1 | k3 | k5 | k7 | k9 | dist ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 Raw FCMAE baseline | 1024 | 128 | 94.00% | 92.62% | 91.43% | 90.79% | 90.49% | 0.3434 |
| S1 + DINOv3 SSL backbone | 1024 | 128 | 93.80% | 92.33% | 91.49% | 91.26% | 90.48% | 0.2460 |
| S2 + PCA dimension tuning | 1024 | 1024 | 93.87% | 92.31% | 91.43% | 90.62% | 90.28% | 0.2490 |
| S3 + wafer SimCLR SSL fine-tune | 1024 | 1024 | 93.67% | 92.82% | 92.28% | 91.95% | 91.56% | 0.4529 |

## Embedding Paths

- S0 Raw FCMAE baseline: `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\embeddings\FCMAE_baseline.npy`
- S1 + DINOv3 SSL backbone: `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\embeddings\dinov3_convnext_base.npy`
- S2 + PCA dimension tuning: `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\embeddings\dinov3_convnext_base.npy`
- S3 + wafer SimCLR SSL fine-tune: `D:\project\unknown-contrastive\result_grouping\_dinov3_ncd_autoloop\ssl_embeddings\simclr_ep5.npy`
