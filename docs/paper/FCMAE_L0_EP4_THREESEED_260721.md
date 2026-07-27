# FCMAE L0 Adapter ep4 Three-Seed Confirmation (260721)

## Decision

- accepted: **FALSE**
- next axis: frozen-residual preservation sweep before adding contrastive components
- fixed point: L0 pure SimCLR residual adapter, adapted f, epoch 4, seeds 1/3/5.
- P1/P2/P3/P4/fragmentation decide the gate. ARI/AMI remain visible but cannot rescue a P1 failure.

## Frozen Baseline

| recipe | seed | ep | clusterer | P1 | recov | P2 noise% | P3 Comp | P4 Hom | k(target/noise) | frag | Sil | ARI* | AMI* |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen | none | 0 | FINCH-p2 | 32/32 | 0.9259 | 0.00 | 0.8914 | 0.9583 | 62/32/16 | 1.94 | 0.3810 | 0.8050 | 0.9178 |
| frozen | none | 0 | Louvain-res6 | 31/32 | 0.9309 | 0.00 | 0.9311 | 0.9677 | 53/32/13 | 1.66 | 0.4718 | 0.8707 | 0.9457 |

## Seed Results

| recipe | seed | ep | clusterer | P1 | recov | P2 noise% | P3 Comp | P4 Hom | k(target/noise) | frag | Sil | ARI* | AMI* |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0_base_ep4 | 1 | 4 | FINCH-p2 | 29/32 | 0.8347 | 0.00 | 0.8903 | 0.9459 | 61/32/17 | 1.91 | 0.3857 | 0.7665 | 0.9111 |
| L0_base_ep4 | 1 | 4 | Louvain-res6 | 31/32 | 0.9366 | 0.00 | 0.9391 | 0.9717 | 52/32/13 | 1.62 | 0.4544 | 0.8841 | 0.9522 |
| L0_base_ep4 | 3 | 4 | FINCH-p2 | 30/32 | 0.8944 | 0.00 | 0.8938 | 0.9640 | 60/32/13 | 1.88 | 0.3727 | 0.8073 | 0.9219 |
| L0_base_ep4 | 3 | 4 | Louvain-res6 | 31/32 | 0.9319 | 0.00 | 0.9324 | 0.9679 | 52/32/13 | 1.62 | 0.4768 | 0.8766 | 0.9466 |
| L0_base_ep4 | 5 | 4 | FINCH-p2 | 32/32 | 0.9269 | 0.00 | 0.8848 | 0.9621 | 64/32/16 | 2.00 | 0.3928 | 0.8023 | 0.9156 |
| L0_base_ep4 | 5 | 4 | Louvain-res6 | 31/32 | 0.9206 | 0.00 | 0.9337 | 0.9675 | 52/32/13 | 1.62 | 0.4629 | 0.8745 | 0.9471 |

## Gate Details

### FINCH-p2

- accepted: False
- P1_all_seeds: False
- P2_all_seeds: True
- P3_mean_non_worse: False
- P3_direction_2of3: False
- P4_mean_non_worse: False
- P4_direction_2of3: True
- fragment_mean_non_worse: True
- fragment_direction_2of3: True
- recov_mean_within_0.01: False
- mean deltas: recov=-0.0406, P3=-0.0018, P4=-0.0010, fragment improvement=+0.0104, Sil*=+0.0027, ARI*=-0.0130, AMI*=-0.0016

### Louvain-res6

- accepted: True
- P1_all_seeds: True
- P2_all_seeds: True
- P3_mean_non_worse: True
- P3_direction_2of3: True
- P4_mean_non_worse: True
- P4_direction_2of3: True
- fragment_mean_non_worse: True
- fragment_direction_2of3: True
- recov_mean_within_0.01: True
- mean deltas: recov=-0.0012, P3=+0.0040, P4=+0.0013, fragment improvement=+0.0312, Sil*=-0.0071, ARI*=+0.0077, AMI*=+0.0029

Raw rows: `D:\project\unknown-contrastive\docs\paper\FCMAE_L0_EP4_THREESEED_260721.csv`
Gate JSON: `D:\project\unknown-contrastive\docs\paper\FCMAE_L0_EP4_THREESEED_260721_gate.json`
