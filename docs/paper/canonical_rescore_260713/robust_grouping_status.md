# Robust Grouping Model Status

All P1 values below use the same contract: full-pool clustering and unique dominant/main-class capture. P2/P3/P4/ARI are scored only on the protocol target classes.

## Cross-Dataset Acceptance

Core gate: P1 does not regress, P2 does not increase, and P3/P4/ARI do not regress. Silhouette, k, and fragment ratio are mandatory diagnostics, not automatic reject rules.

| Dataset | Role | Recipe | P1 capture | P2 noise% | P3 Comp | P4 Hom | ARI | Sil | k | Fragment | Core gate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DTD | frozen | DINOv3 frozen | 43/47 (0.915) | 0.0 | 0.641 | 0.679 | 0.324 | 0.189 | 75 | 1.60 | pass |
| DTD | candidate | SimCLR + queue4096 + ignore0.75, seed3 ep8 | 46/47 (0.979) | 0.0 | 0.687 | 0.762 | 0.422 | 0.235 | 88 | 1.87 | pass |
| RESISC45 | frozen | DINOv3 frozen | 44/45 (0.978) | 0.0 | 0.736 | 0.769 | 0.450 | 0.171 | 67 | 1.49 | pass |
| RESISC45 | candidate | SimCLR + queue4096 + ignore0.75, seed3 ep6 | 44/45 (0.978) | 0.0 | 0.771 | 0.861 | 0.587 | 0.234 | 80 | 1.78 | pass |
| WM-811K | frozen | DINOv3 frozen | 7/7 (1.000) | 0.0 | 0.267 | 0.435 | 0.149 | 0.057 | 28 | 4.00 | pass |
| WM-811K | candidate | SimCLR + queue4096 + local0.3 + ignore0.75, seed4 ep8 | 7/7 (1.000) | 0.0 | 0.448 | 0.520 | 0.333 | -0.010 | 18 | 2.57 | pass |

## Hard-Unknown Strict-Novel

FCMAE frozen ep0 passes both clusterer gates against DINOv3 frozen and needs only the image-disjoint holdout check.
NV 0.50 blend086 ep6 is the strongest learned dual-gate candidate and remains provisional until fixed-seed and holdout validation.

| Method | Row | Recipe / epoch | P1 capture | P2 noise% | P3 Comp | P4 Hom | ARI | Sil | k | Fragment | Core gate vs frozen |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FINCH-p2 | reference | DINOv3 frozen | 30/32 (0.938) | 0.0 | 0.830 | 0.892 | 0.709 | 0.304 | 58 | 1.81 | pass |
| Louvain | reference | DINOv3 frozen | 31/32 (0.969) | 0.0 | 0.868 | 0.907 | 0.785 | 0.378 | 52 | 1.62 | pass |
| FINCH-p2 | backbone candidate | FCMAE frozen ep0 | 32/32 (1.000) | 0.0 | 0.891 | 0.958 | 0.805 | 0.381 | 62 | 1.94 | pass |
| Louvain | backbone candidate | FCMAE frozen ep0 | 31/32 (0.969) | 0.0 | 0.931 | 0.968 | 0.871 | 0.472 | 53 | 1.66 | pass |
| FINCH-p2 | learned candidate | NV 0.50 blend086 ep6 | 30/32 (0.938) | 0.0 | 0.859 | 0.902 | 0.739 | 0.348 | 51 | 1.59 | pass |
| Louvain | learned candidate | NV 0.50 blend086 ep6 | 31/32 (0.969) | 0.0 | 0.901 | 0.924 | 0.831 | 0.429 | 47 | 1.47 | pass |

## Current Deployment Decision

- WM-811K, RESISC45, and DTD: use the accepted learned candidate shown above; retain the frozen embedding as fallback.
- Hard unknown: FCMAE frozen ep0 is the current provisional backbone candidate; retain DINOv3 frozen as the fallback until the image-disjoint holdout check completes.
- Learned hard-unknown candidate: NV 0.50 blend086 ep6; retain it separately until fixed-seed and holdout validation complete.
- Therefore this is a robust model family with an acceptance/fallback policy, not yet one universal learned checkpoint.


## Embedding-Mode Audit

FINCH-p2 rows below compare the trained projection `z` with the matching backbone `f`. The current hard-unknown deployment path remains `f` unless an adapter trial passes the same core gate.

| Recipe / epoch | Space | P1 capture | P2 noise% | P3 Comp | P4 Hom | ARI | Sil | k | Fragment |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base ep2 | backbone f | 26/32 (0.812) | 0.0 | 0.802 | 0.849 | 0.596 | 0.295 | 58 | 1.81 |
| base ep2 | projection z | 26/32 (0.812) | 0.0 | 0.729 | 0.757 | 0.399 | 0.259 | 59 | 1.84 |
| base ep3 | backbone f | 30/32 (0.938) | 0.0 | 0.758 | 0.813 | 0.538 | 0.304 | 54 | 1.69 |
| base ep3 | projection z | 29/32 (0.906) | 0.0 | 0.700 | 0.750 | 0.435 | 0.252 | 57 | 1.78 |
| nv050 ep1 | backbone f | 31/32 (0.969) | 0.0 | 0.795 | 0.884 | 0.640 | 0.375 | 66 | 2.06 |
| nv050 ep1 | projection z | 27/32 (0.844) | 0.0 | 0.696 | 0.769 | 0.460 | 0.225 | 59 | 1.84 |
| nv050 ep2 | backbone f | 30/32 (0.938) | 0.0 | 0.818 | 0.882 | 0.673 | 0.304 | 60 | 1.88 |
| nv050 ep2 | projection z | 27/32 (0.844) | 0.0 | 0.621 | 0.676 | 0.355 | 0.067 | 61 | 1.91 |
| nv050 ep6 | backbone f | 31/32 (0.969) | 0.0 | 0.831 | 0.889 | 0.674 | 0.312 | 57 | 1.78 |
| nv050 ep6 | projection z | 29/32 (0.906) | 0.0 | 0.646 | 0.732 | 0.392 | 0.116 | 60 | 1.88 |
| nv050 ep7 | backbone f | 31/32 (0.969) | 0.0 | 0.822 | 0.882 | 0.672 | 0.325 | 58 | 1.81 |
| nv050 ep7 | projection z | 29/32 (0.906) | 0.0 | 0.660 | 0.758 | 0.412 | 0.221 | 68 | 2.12 |
| nv060 ep7 | backbone f | 32/32 (1.000) | 0.0 | 0.743 | 0.816 | 0.564 | 0.342 | 53 | 1.66 |
| nv060 ep7 | projection z | 29/32 (0.906) | 0.0 | 0.589 | 0.683 | 0.333 | 0.202 | 68 | 2.12 |