# FCMAE Residual Adapter Three-Seed Live Result (260725)

## Protocol

- Backbone: no-TAPT FCMAE, frozen
- Train pool: `D:\project\unknown-contrastive\data\images\unknown_train_defectaware_260710`
- Development evaluation: `D:\project\unknown-contrastive\data\images\unknown_eval100`
- Candidate: one-stage zero-init residual adapter, pure SimCLR, fixed epoch 4
- Seeds: 1, 3, 5
- Primary gate: P1 capture, P2 noise, P3 completeness, P4 homogeneity, fragmentation
- Clusterers: FINCH-p2 and Louvain-res6
- ARI/AMI are supporting diagnostics only.

## Frozen Baseline

| Clusterer | P1 | P2 noise | P3 Comp | P4 Hom | k | Frag |
|---|---:|---:|---:|---:|---:|---:|
| FINCH-p2 | 32/32 | 0.0% | 0.8914 | 0.9583 | 62 | 1.9375 |
| Louvain-res6 | 31/32 | 0.0% | 0.9311 | 0.9677 | 53 | 1.6562 |

## Seed 1 Trajectory

| Epoch | Clusterer | P1 | P2 noise | P3 Comp | P4 Hom | k | Frag |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | FINCH-p2 | 29/32 | 0.0% | 0.8994 | 0.9469 | 59 | 1.8438 |
| 2 | FINCH-p2 | 30/32 | 0.0% | 0.9046 | 0.9568 | 60 | 1.8750 |
| 3 | FINCH-p2 | 32/32 | 0.0% | 0.8879 | 0.9523 | 60 | 1.8750 |
| 4 | FINCH-p2 | 29/32 | 0.0% | 0.8903 | 0.9459 | 61 | 1.9062 |
| 1 | Louvain-res6 | 31/32 | 0.0% | 0.9363 | 0.9696 | 52 | 1.6250 |
| 2 | Louvain-res6 | 31/32 | 0.0% | 0.9337 | 0.9700 | 53 | 1.6562 |
| 3 | Louvain-res6 | 31/32 | 0.0% | 0.9336 | 0.9673 | 52 | 1.6250 |
| 4 | Louvain-res6 | 31/32 | 0.0% | 0.9391 | 0.9717 | 52 | 1.6250 |

## Fixed-Epoch Three-Seed Result

| Seed | FINCH P1 | FINCH P3 | FINCH P4 | Louvain P1 | Louvain P3 | Louvain P4 |
|---:|---:|---:|---:|---:|---:|---:|
| frozen | 32/32 | 0.8914 | 0.9583 | 31/32 | 0.9311 | 0.9677 |
| 1 | 29/32 | 0.8903 | 0.9459 | 31/32 | 0.9391 | 0.9717 |
| 3 | 30/32 | 0.8938 | 0.9640 | 31/32 | 0.9324 | 0.9679 |
| 5 | 32/32 | 0.8848 | 0.9621 | 31/32 | 0.9337 | 0.9675 |

The fixed alpha=1.0 candidate is rejected. FINCH loses P1 in seeds 1 and 3,
with mean P3 delta -0.0018 and mean P4 delta -0.0010. Louvain passes, with
mean P3 delta +0.0040 and mean P4 delta +0.0013, but one clusterer cannot
rescue failure in the other. P2 is 0% for every FINCH/Louvain row and is not a
background FAR measurement because the evaluation pool contains target defects
only.

The epoch trajectory is diagnostic only. Ground-truth metrics are not used to
select a deployment epoch.

## Next Controlled Axis

Run the image-disjoint holdout with the completed seeds 1/3/5 checkpoints.
Then screen inference-only residual alpha `0.25`, `0.50`, `0.75`, and `1.00`
on seed 1. If no alpha preserves P1/P2 and improves P3/P4 in both clusterers,
screen temperature `0.05` (existing), `0.07`, `0.10`, and `0.20` at fixed
epoch 4. Only a dual-clusterer winner advances to seeds 1/3/5. Queue size 16k
versus 32k and a real NEG sweep (`off`, `0.70`, `0.75`, `0.80`) follow only
after those preservation axes.
