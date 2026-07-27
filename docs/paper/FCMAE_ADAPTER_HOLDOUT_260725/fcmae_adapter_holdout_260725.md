# FCMAE Adapter Fixed-Epoch Image-Disjoint Holdout Validation

- protocol_id: `4aaba956f57912a625805c459f8303a10bd233e9893ef93c0fc44bef4d43ef42`
- holdout: `D:\project\unknown-contrastive\data\images\unknown_holdout_100_260713`
- holdout manifest: `D:\project\unknown-contrastive\data\images\unknown_holdout_100_260713\manifest_260713.json`
- fixed point: residual adapter, pure SimCLR, epoch 4, seeds 1/3/5
- P1/P2/P3/P4 and fragmentation decide the gate; ARI/AMI are supporting diagnostics only.
- The holdout is image-disjoint from the development evaluation according to its manifest.

## Gate

- accepted: **FALSE**
- next axis: frozen-residual preservation sweep before adding contrastive components

## Frozen Baseline

| recipe | seed | ep | clusterer | P1 | recov | P2 noise% | P3 Comp | P4 Hom | k(target/noise) | frag | Sil | ARI* | AMI* |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen | 0 | 0 | FINCH-p2 | 31/31 | 0.9284 | 0.00 | 0.9043 | 0.9603 | 58/31/14 | 1.87 | 0.4172 | 0.8139 | 0.9266 |
| frozen | 0 | 0 | Louvain-res6 | 31/31 | 0.9539 | 0.00 | 0.9296 | 0.9687 | 51/31/12 | 1.65 | 0.4673 | 0.8803 | 0.9455 |

## Fixed-epoch Candidate Rows

| recipe | seed | ep | clusterer | P1 | recov | P2 noise% | P3 Comp | P4 Hom | k(target/noise) | frag | Sil | ARI* | AMI* |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen | 0 | 0 | FINCH-p2 | 31/31 | 0.9284 | 0.00 | 0.9043 | 0.9603 | 58/31/14 | 1.87 | 0.4172 | 0.8139 | 0.9266 |
| frozen | 0 | 0 | Louvain-res6 | 31/31 | 0.9539 | 0.00 | 0.9296 | 0.9687 | 51/31/12 | 1.65 | 0.4673 | 0.8803 | 0.9455 |
| L0_base_ep4 | 1 | 4 | FINCH-p2 | 29/31 | 0.9003 | 0.00 | 0.9248 | 0.9668 | 54/31/13 | 1.74 | 0.4421 | 0.8524 | 0.9418 |
| L0_base_ep4 | 1 | 4 | Louvain-res6 | 31/31 | 0.9632 | 0.00 | 0.9359 | 0.9743 | 51/31/12 | 1.65 | 0.4768 | 0.8980 | 0.9519 |
| L0_base_ep4 | 3 | 4 | FINCH-p2 | 31/31 | 0.9432 | 0.00 | 0.8969 | 0.9702 | 58/31/12 | 1.87 | 0.3931 | 0.8217 | 0.9270 |
| L0_base_ep4 | 3 | 4 | Louvain-res6 | 31/31 | 0.9565 | 0.00 | 0.9279 | 0.9708 | 51/31/12 | 1.65 | 0.4619 | 0.8825 | 0.9456 |
| L0_base_ep4 | 5 | 4 | FINCH-p2 | 30/31 | 0.8974 | 0.00 | 0.9043 | 0.9578 | 56/31/13 | 1.81 | 0.4350 | 0.8267 | 0.9256 |
| L0_base_ep4 | 5 | 4 | Louvain-res6 | 30/31 | 0.9516 | 0.00 | 0.9477 | 0.9714 | 49/31/12 | 1.58 | 0.4723 | 0.8978 | 0.9570 |

## Gate Details

### FINCH-p2

- accepted: False
- P1_all_seeds: False
- P2_all_seeds: True
- P3_mean_non_worse: True
- P3_direction_2of3: True
- P4_mean_non_worse: True
- P4_direction_2of3: True
- fragment_mean_non_worse: True
- fragment_direction_2of3: True
- recov_mean_within_0.01: False
- mean deltas: recov=-0.0148, P3=+0.0044, P4=+0.0046, fragment improvement=+0.0645, Sil*=+0.0062, ARI*=+0.0197, AMI*=+0.0049

### Louvain-res6

- accepted: False
- P1_all_seeds: False
- P2_all_seeds: True
- P3_mean_non_worse: True
- P3_direction_2of3: True
- P4_mean_non_worse: True
- P4_direction_2of3: True
- fragment_mean_non_worse: True
- fragment_direction_2of3: True
- recov_mean_within_0.01: True
- mean deltas: recov=+0.0032, P3=+0.0076, P4=+0.0035, fragment improvement=+0.0215, Sil*=+0.0030, ARI*=+0.0125, AMI*=+0.0060

## Provenance

- fixed protocol: `D:\project\unknown-contrastive\scripts\fcmae_fixed_protocol.py`
- existing holdout evaluator: `D:\project\unknown-contrastive\scripts\rescore_unknown_strict_novel.py`
- frozen holdout embedding: `D:\project\unknown-contrastive\result_grouping\_hard42_holdout_260713\embeddings\fcmae_frozen_holdout.npy`
- interim evaluator output: `D:\project\unknown-contrastive\runs\fcmae_adapter_holdout_validation_260725\canonical_rescore`

Raw CSV includes ARI/AMI as supporting columns; neither can rescue a failed P1/P2/P3/P4 gate.
