# Hard-Unknown Strict-Novel Canonical Rescore

Protocol: cluster the complete 42-class pool, then score the 32 defect classes excluded from defect-aware adaptation.
P1 is unique dominant/main-class capture. Normal and train-known classes remain in clustering, so a minority defect inside a Normal-dominant group is not falsely captured.

| Recipe | Diagnostic best epoch by FINCH-p2 ARI | P1 capture | P2 noise% | P3 Comp | P4 Hom | ARI | Sil | k | Fragment | Gate vs frozen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FCMAE frozen | 0 | 31/31 (1.000) | 0.0 | 0.904 | 0.960 | 0.814 | 0.417 | 58 | 1.87 | pass |
| Frozen | 0 | 28/31 (0.903) | 0.0 | 0.835 | 0.867 | 0.689 | 0.360 | 55 | 1.77 | pass |

`best epoch` is diagnostic only. A recipe is not accepted unless both FINCH-p2 and Louvain preserve or improve frozen P1/P2/P3/P4/ARI. Silhouette, k, and fragmentation remain required diagnostics rather than hard gates.
