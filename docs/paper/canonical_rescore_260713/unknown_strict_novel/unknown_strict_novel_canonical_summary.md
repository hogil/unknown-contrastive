# Hard-Unknown Strict-Novel Canonical Rescore

Protocol: cluster the complete 42-class pool, then score the 32 defect classes excluded from defect-aware adaptation.
P1 is unique dominant/main-class capture. Normal and train-known classes remain in clustering, so a minority defect inside a Normal-dominant group is not falsely captured.

| Recipe | Diagnostic best epoch by FINCH-p2 ARI | P1 capture | P2 noise% | P3 Comp | P4 Hom | ARI | Sil | k | Fragment | Gate vs frozen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Frozen | 0 | 30/32 (0.938) | 0.0 | 0.830 | 0.892 | 0.709 | 0.304 | 58 | 1.81 | pass |
| SimCLR base | 2 | 26/32 (0.812) | 0.0 | 0.802 | 0.849 | 0.596 | 0.295 | 58 | 1.81 | fail |
| NV 0.50 | 6 | 31/32 (0.969) | 0.0 | 0.831 | 0.889 | 0.674 | 0.312 | 57 | 1.78 | fail |
| NV 0.60 | 1 | 28/32 (0.875) | 0.0 | 0.794 | 0.843 | 0.628 | 0.316 | 53 | 1.66 | fail |
| NV 0.70 | 1 | 27/32 (0.844) | 0.0 | 0.827 | 0.883 | 0.650 | 0.335 | 54 | 1.69 | fail |
| NV 0.75 | 1 | 29/32 (0.906) | 0.0 | 0.801 | 0.868 | 0.638 | 0.249 | 58 | 1.81 | fail |
| NV 0.90 | 1 | 27/32 (0.844) | 0.0 | 0.780 | 0.797 | 0.542 | 0.282 | 50 | 1.56 | fail |

`best epoch` is diagnostic only. A recipe is not accepted unless both FINCH-p2 and Louvain preserve or improve frozen P1/P2/P3/P4/ARI. Silhouette, k, and fragmentation remain required diagnostics rather than hard gates.
