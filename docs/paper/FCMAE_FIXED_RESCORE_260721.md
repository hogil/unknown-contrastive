# FCMAE Fixed-Protocol Re-score (260721)

## Contract

- protocol_id: `810f080c6ff072fbaa9ea3d901dbbfd15beb5194e78393c127340ee7784e6a5d`
- eval: `D:\project\unknown-contrastive\data\images\unknown_eval100` (4149 images, 32 strict-novel targets)
- train manifest: `D:\project\unknown-contrastive\data\images\unknown_train_defectaware_260710\manifest_260710.json`
- eval manifest: `D:\project\unknown-contrastive\docs\paper\FCMAE_FIXED_PROTOCOL_260721_eval_manifest.json`
- scorer bundle sha256: `471dfd5ff69ac57232371e960c5d8402094b21e527fa4947bb9332b0e1001d2e`
- P1 is canonical dominant-main-class capture. Presence coverage is not P1.
- P2 is target-image clusterer noise. This eval pool contains no Normal/Random/R, so this is not FAR.
- ARI/AMI columns are retained only as supporting diagnostics.
- L4/L5 are excluded because the cumulative ladder continued after a failed rung.

## Frozen Baseline

| recipe | seed | ep | clusterer | P1 | recov | P2 noise% | P3 Comp | P4 Hom | k(target/noise) | frag | Sil | ARI* | AMI* |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen | none | 0 | FINCH-p2 | 32/32 | 0.9259 | 0.00 | 0.8914 | 0.9583 | 62/32/16 | 1.94 | 0.3810 | 0.8050 | 0.9178 |
| frozen | none | 0 | Louvain-res6 | 31/32 | 0.9309 | 0.00 | 0.9311 | 0.9677 | 53/32/13 | 1.66 | 0.4718 | 0.8707 | 0.9457 |

## Re-scored Rows

| recipe | seed | ep | clusterer | P1 | recov | P2 noise% | P3 Comp | P4 Hom | k(target/noise) | frag | Sil | ARI* | AMI* |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L0_base | 3 | 0 | FINCH-p2 | 32/32 | 0.9259 | 0.00 | 0.8914 | 0.9583 | 62/32/16 | 1.94 | 0.3810 | 0.8050 | 0.9178 |
| L0_base | 3 | 0 | Louvain-res6 | 31/32 | 0.9312 | 0.00 | 0.9313 | 0.9679 | 53/32/13 | 1.66 | 0.4716 | 0.8710 | 0.9459 |
| L0_base | 3 | 1 | FINCH-p2 | 32/32 | 0.9156 | 0.00 | 0.9029 | 0.9542 | 57/32/14 | 1.78 | 0.3858 | 0.8147 | 0.9227 |
| L0_base | 3 | 1 | Louvain-res6 | 31/32 | 0.9281 | 0.00 | 0.9289 | 0.9670 | 53/32/13 | 1.66 | 0.4688 | 0.8704 | 0.9441 |
| L0_base | 3 | 2 | FINCH-p2 | 30/32 | 0.8709 | 0.00 | 0.8870 | 0.9479 | 61/32/15 | 1.91 | 0.3705 | 0.7737 | 0.9101 |
| L0_base | 3 | 2 | Louvain-res6 | 31/32 | 0.9403 | 0.00 | 0.9365 | 0.9707 | 52/32/13 | 1.62 | 0.4603 | 0.8873 | 0.9503 |
| L0_base | 3 | 3 | FINCH-p2 | 30/32 | 0.8728 | 0.00 | 0.9102 | 0.9555 | 56/32/14 | 1.75 | 0.3952 | 0.8184 | 0.9275 |
| L0_base | 3 | 3 | Louvain-res6 | 31/32 | 0.9247 | 0.00 | 0.9306 | 0.9679 | 53/32/13 | 1.66 | 0.4545 | 0.8621 | 0.9455 |
| L0_base | 3 | 4 | FINCH-p2 | 32/32 | 0.9094 | 0.00 | 0.9056 | 0.9598 | 55/32/12 | 1.72 | 0.3913 | 0.8224 | 0.9270 |
| L0_base | 3 | 4 | Louvain-res6 | 31/32 | 0.9466 | 0.00 | 0.9451 | 0.9745 | 51/32/13 | 1.59 | 0.4757 | 0.9055 | 0.9570 |
| L0_base | 3 | 5 | FINCH-p2 | 30/32 | 0.8484 | 0.00 | 0.9017 | 0.9477 | 55/32/13 | 1.72 | 0.3750 | 0.7896 | 0.9187 |
| L0_base | 3 | 5 | Louvain-res6 | 31/32 | 0.9234 | 0.00 | 0.9299 | 0.9671 | 53/32/13 | 1.66 | 0.4679 | 0.8710 | 0.9447 |
| L1_queue | 3 | 0 | FINCH-p2 | 32/32 | 0.9259 | 0.00 | 0.8914 | 0.9583 | 62/32/16 | 1.94 | 0.3810 | 0.8050 | 0.9178 |
| L1_queue | 3 | 0 | Louvain-res6 | 31/32 | 0.9312 | 0.00 | 0.9313 | 0.9679 | 53/32/13 | 1.66 | 0.4716 | 0.8710 | 0.9459 |
| L1_queue | 3 | 1 | FINCH-p2 | 31/32 | 0.8841 | 0.00 | 0.9145 | 0.9520 | 52/32/14 | 1.62 | 0.4018 | 0.8318 | 0.9285 |
| L1_queue | 3 | 1 | Louvain-res6 | 31/32 | 0.9200 | 0.00 | 0.9286 | 0.9661 | 53/32/13 | 1.66 | 0.4660 | 0.8684 | 0.9435 |
| L1_queue | 3 | 2 | FINCH-p2 | 31/32 | 0.8859 | 0.00 | 0.8965 | 0.9446 | 57/32/14 | 1.78 | 0.4057 | 0.7846 | 0.9142 |
| L1_queue | 3 | 2 | Louvain-res6 | 31/32 | 0.9325 | 0.00 | 0.9315 | 0.9678 | 53/32/13 | 1.66 | 0.4619 | 0.8718 | 0.9460 |
| L1_queue | 3 | 3 | FINCH-p2 | 31/32 | 0.9016 | 0.00 | 0.9061 | 0.9543 | 55/32/14 | 1.72 | 0.4083 | 0.8140 | 0.9247 |
| L1_queue | 3 | 3 | Louvain-res6 | 31/32 | 0.9116 | 0.00 | 0.9178 | 0.9591 | 53/32/13 | 1.66 | 0.4607 | 0.8466 | 0.9339 |
| L1_queue | 3 | 4 | FINCH-p2 | 31/32 | 0.8750 | 0.00 | 0.8759 | 0.9424 | 60/32/14 | 1.88 | 0.3146 | 0.7453 | 0.9008 |
| L1_queue | 3 | 4 | Louvain-res6 | 31/32 | 0.9112 | 0.00 | 0.9175 | 0.9620 | 54/32/13 | 1.69 | 0.4648 | 0.8470 | 0.9351 |
| L1_queue | 3 | 5 | FINCH-p2 | 30/32 | 0.8613 | 0.00 | 0.9068 | 0.9495 | 53/32/12 | 1.66 | 0.3813 | 0.7932 | 0.9226 |
| L1_queue | 3 | 5 | Louvain-res6 | 31/32 | 0.9141 | 0.00 | 0.9210 | 0.9655 | 53/32/12 | 1.66 | 0.4610 | 0.8520 | 0.9388 |
| L2_queue_ignore | 3 | 0 | FINCH-p2 | 32/32 | 0.9259 | 0.00 | 0.8914 | 0.9583 | 62/32/16 | 1.94 | 0.3810 | 0.8050 | 0.9178 |
| L2_queue_ignore | 3 | 0 | Louvain-res6 | 31/32 | 0.9312 | 0.00 | 0.9313 | 0.9679 | 53/32/13 | 1.66 | 0.4716 | 0.8710 | 0.9459 |
| L2_queue_ignore | 3 | 1 | FINCH-p2 | 32/32 | 0.9372 | 0.00 | 0.9276 | 0.9640 | 51/32/12 | 1.59 | 0.3578 | 0.8599 | 0.9419 |
| L2_queue_ignore | 3 | 1 | Louvain-res6 | 31/32 | 0.9194 | 0.00 | 0.9273 | 0.9650 | 53/32/13 | 1.66 | 0.4700 | 0.8648 | 0.9422 |
| L2_queue_ignore | 3 | 2 | FINCH-p2 | 32/32 | 0.8884 | 0.00 | 0.8766 | 0.9437 | 60/32/14 | 1.88 | 0.3650 | 0.7569 | 0.9020 |
| L2_queue_ignore | 3 | 2 | Louvain-res6 | 31/32 | 0.9094 | 0.00 | 0.9168 | 0.9614 | 54/32/13 | 1.69 | 0.4707 | 0.8449 | 0.9344 |
| L2_queue_ignore | 3 | 3 | FINCH-p2 | 31/32 | 0.8744 | 0.00 | 0.8806 | 0.9446 | 62/32/16 | 1.94 | 0.4068 | 0.7538 | 0.9047 |
| L2_queue_ignore | 3 | 3 | Louvain-res6 | 31/32 | 0.9116 | 0.00 | 0.9141 | 0.9623 | 55/32/13 | 1.72 | 0.4665 | 0.8407 | 0.9332 |
| L2_queue_ignore | 3 | 4 | FINCH-p2 | 31/32 | 0.8803 | 0.00 | 0.8790 | 0.9476 | 63/32/16 | 1.97 | 0.4144 | 0.7630 | 0.9052 |
| L2_queue_ignore | 3 | 4 | Louvain-res6 | 31/32 | 0.9141 | 0.00 | 0.9159 | 0.9633 | 55/32/13 | 1.72 | 0.4684 | 0.8441 | 0.9348 |
| L2_queue_ignore | 3 | 5 | FINCH-p2 | 32/32 | 0.9172 | 0.00 | 0.8946 | 0.9554 | 59/32/14 | 1.84 | 0.3876 | 0.8049 | 0.9184 |
| L2_queue_ignore | 3 | 5 | Louvain-res6 | 31/32 | 0.9141 | 0.00 | 0.9151 | 0.9632 | 55/32/13 | 1.72 | 0.4696 | 0.8450 | 0.9343 |
| L3_queue_ignore_nv | 3 | 0 | FINCH-p2 | 32/32 | 0.9259 | 0.00 | 0.8914 | 0.9583 | 62/32/16 | 1.94 | 0.3810 | 0.8050 | 0.9178 |
| L3_queue_ignore_nv | 3 | 0 | Louvain-res6 | 31/32 | 0.9312 | 0.00 | 0.9313 | 0.9679 | 53/32/13 | 1.66 | 0.4716 | 0.8710 | 0.9459 |
| L3_queue_ignore_nv | 3 | 1 | FINCH-p2 | 29/32 | 0.8447 | 0.00 | 0.9136 | 0.9492 | 53/32/14 | 1.66 | 0.3830 | 0.8063 | 0.9264 |
| L3_queue_ignore_nv | 3 | 1 | Louvain-res6 | 30/32 | 0.9125 | 0.00 | 0.9317 | 0.9656 | 53/32/13 | 1.66 | 0.4576 | 0.8572 | 0.9449 |
| L3_queue_ignore_nv | 3 | 2 | FINCH-p2 | 30/32 | 0.8794 | 0.00 | 0.9054 | 0.9531 | 55/32/12 | 1.72 | 0.3444 | 0.8043 | 0.9235 |
| L3_queue_ignore_nv | 3 | 2 | Louvain-res6 | 31/32 | 0.9206 | 0.00 | 0.9221 | 0.9669 | 54/32/13 | 1.69 | 0.4615 | 0.8568 | 0.9402 |
| L3_queue_ignore_nv | 3 | 3 | FINCH-p2 | 32/32 | 0.9125 | 0.00 | 0.9107 | 0.9540 | 55/32/14 | 1.72 | 0.3826 | 0.8242 | 0.9271 |
| L3_queue_ignore_nv | 3 | 3 | Louvain-res6 | 31/32 | 0.9178 | 0.00 | 0.9260 | 0.9648 | 53/32/13 | 1.66 | 0.4890 | 0.8613 | 0.9414 |
| L3_queue_ignore_nv | 3 | 4 | FINCH-p2 | 31/32 | 0.8816 | 0.00 | 0.9121 | 0.9420 | 53/32/13 | 1.66 | 0.4088 | 0.8083 | 0.9221 |
| L3_queue_ignore_nv | 3 | 4 | Louvain-res6 | 31/32 | 0.9000 | 0.00 | 0.9190 | 0.9625 | 54/32/13 | 1.69 | 0.4720 | 0.8481 | 0.9362 |
| L3_queue_ignore_nv | 3 | 5 | FINCH-p2 | 29/32 | 0.8337 | 0.00 | 0.8866 | 0.9362 | 57/32/12 | 1.78 | 0.3767 | 0.7442 | 0.9041 |
| L3_queue_ignore_nv | 3 | 5 | Louvain-res6 | 31/32 | 0.9119 | 0.00 | 0.9236 | 0.9632 | 53/32/13 | 1.66 | 0.4859 | 0.8575 | 0.9392 |

## Fixed Confirmation Target

`L0_base ep4` is the pre-declared seed-confirmation point. It is not accepted from seed 3 alone.
Seeds 1/3/5 must pass the P1-first gate in the separate fixed-epoch report.

Raw rows: `D:\project\unknown-contrastive\docs\paper\FCMAE_FIXED_RESCORE_260721.csv`
Protocol: `D:\project\unknown-contrastive\docs\paper\FCMAE_FIXED_PROTOCOL_260721.json`
