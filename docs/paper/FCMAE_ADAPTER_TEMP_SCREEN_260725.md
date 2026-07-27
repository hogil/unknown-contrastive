# FCMAE Residual Adapter Temperature Screen (260725)

- One-axis screen: temperature only
- Baseline: temperature 0.05, seed 1, fixed epoch 4
- New values: 0.07, 0.10, 0.20
- P1/P2/P3/P4 decide screening; ARI/AMI are supporting only.
- This report proposes a candidate but never launches multi-seed validation.

## Rows

| recipe | seed | ep | clusterer | P1 | recov | P2 noise% | P3 Comp | P4 Hom | k(target/noise) | frag | Sil | ARI* | AMI* |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen | none | 0 | FINCH-p2 | 32/32 | 0.9259 | 0.00 | 0.8914 | 0.9583 | 62/32/16 | 1.94 | 0.3810 | 0.8050 | 0.9178 |
| frozen | none | 0 | Louvain-res6 | 31/32 | 0.9309 | 0.00 | 0.9311 | 0.9677 | 53/32/13 | 1.66 | 0.4718 | 0.8707 | 0.9457 |
| adapter_temp_0.05 | 1 | 4 | FINCH-p2 | 29/32 | 0.8347 | 0.00 | 0.8903 | 0.9459 | 61/32/17 | 1.91 | 0.3857 | 0.7665 | 0.9111 |
| adapter_temp_0.05 | 1 | 4 | Louvain-res6 | 31/32 | 0.9366 | 0.00 | 0.9391 | 0.9717 | 52/32/13 | 1.62 | 0.4544 | 0.8841 | 0.9522 |
| adapter_temp_0.07 | 1 | 4 | FINCH-p2 | 30/32 | 0.8756 | 0.00 | 0.8993 | 0.9586 | 59/32/15 | 1.84 | 0.3584 | 0.8047 | 0.9227 |
| adapter_temp_0.07 | 1 | 4 | Louvain-res6 | 31/32 | 0.9316 | 0.00 | 0.9326 | 0.9692 | 53/32/13 | 1.66 | 0.4684 | 0.8741 | 0.9473 |
| adapter_temp_0.10 | 1 | 4 | FINCH-p2 | 32/32 | 0.9244 | 0.00 | 0.8983 | 0.9592 | 60/32/15 | 1.88 | 0.3670 | 0.8140 | 0.9224 |
| adapter_temp_0.10 | 1 | 4 | Louvain-res6 | 31/32 | 0.9322 | 0.00 | 0.9422 | 0.9704 | 51/32/13 | 1.59 | 0.4864 | 0.8904 | 0.9534 |
| adapter_temp_0.20 | 1 | 4 | FINCH-p2 | 32/32 | 0.9000 | 0.00 | 0.8952 | 0.9515 | 58/32/14 | 1.81 | 0.4053 | 0.8046 | 0.9168 |
| adapter_temp_0.20 | 1 | 4 | Louvain-res6 | 31/32 | 0.9416 | 0.00 | 0.9340 | 0.9716 | 53/32/13 | 1.66 | 0.4708 | 0.8854 | 0.9493 |

## Gate

- temp 0.05: accepted=False, minimum P3/P4 delta=-0.012400
- temp 0.07: accepted=False, minimum P3/P4 delta=0.000300
- temp 0.10: accepted=True, minimum P3/P4 delta=0.000900
- temp 0.20: accepted=False, minimum P3/P4 delta=-0.006800

- proposed temperature: 0.1

## Absolute Outputs

- JSON: `D:\project\unknown-contrastive\docs\paper\FCMAE_ADAPTER_TEMP_SCREEN_260725.json`
- CSV: `D:\project\unknown-contrastive\docs\paper\FCMAE_ADAPTER_TEMP_SCREEN_260725.csv`
- run root: `D:\project\unknown-contrastive\runs\fcmae_adapter_temperature_screen_260725`
