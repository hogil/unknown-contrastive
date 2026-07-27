# 3-Seed Fixed-Epoch Confirmation (260720)

Protocol: all-unfreeze + backbone `f`; labels are used only for scoring. Epochs are locked from the seed-3 discovery trajectory: WM ep8, RESISC45 ep6, DTD ep8. No per-seed best-epoch selection.

Acceptance gate: all three seeds present; every seed preserves P1; at least 2/3 seeds jointly preserve or improve P1/P2/P3/P4; mean P2 does not worsen; mean P3/P4 do not worsen; mean fragmentation is no more than 1.5x frozen; Louvain preserves P1 and P3/P4 on average. ARI and Silhouette remain recorded as supporting metrics.

Raw rows use the canonical metric order: P1 capture, recov, P2 noise, P3 completeness, P4 homogeneity, ARI, Silhouette, k(total/classes/background), fragment ratio.

## wm_l03_ig75 (seeds 3 4 5, fixed ep8 f)
wm_l03_ig75 frozen finch_p2(k28) | 7/7 (1.0000) | 0.5975 | 0.0 | 0.267 | 0.4348 | 0.1492 | 0.0572 | 28/7/4 | 4.0
wm_l03_ig75 frozen louvain_res6 | 7/7 (1.0000) | 0.6835 | 0.0 | 0.2702 | 0.539 | 0.1148 | 0.0767 | 53/7/7 | 7.57
wm_l03_ig75_s3 ep8 f finch_p2(k18) | 7/7 (1.0000) | 0.6229 | 0.0 | 0.3722 | 0.482 | 0.2929 | 0.0692 | 18/7/2 | 2.57
wm_l03_ig75_s3 ep8 f louvain_res6 | 7/7 (1.0000) | 0.6683 | 0.0 | 0.3009 | 0.5993 | 0.1458 | 0.0958 | 54/7/7 | 7.71
wm_l03_ig75_s4 ep8 f finch_p2(k18) | 7/7 (1.0000) | 0.6444 | 0.0 | 0.4482 | 0.5198 | 0.3327 | -0.0101 | 18/7/4 | 2.57
wm_l03_ig75_s4 ep8 f louvain_res6 | 7/7 (1.0000) | 0.7139 | 0.0 | 0.3116 | 0.6141 | 0.1508 | 0.0433 | 52/7/7 | 7.43
