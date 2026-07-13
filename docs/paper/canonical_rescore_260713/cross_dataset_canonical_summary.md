# Canonical Cross-Dataset Rescore

P1 is `cluster dominant/main class count / target class count`, computed after clustering the full pool.
`legacy_presence_*` remains in the CSV only for historical audit and is not P1.

| Dataset | Protocol | Recipe | P1 capture | P2 noise% | P3 Comp | P4 Hom | ARI | Sil | k | Fragment |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WM-811K | strict-novel / Normal-only train | DINOv3 frozen | 7/7 (1.000) | 0.0 | 0.267 | 0.4348 | 0.1492 | 0.0572 | 28 | 4.0 |
| WM-811K | strict-novel / Normal-only train | SimCLR + queue4096 + local0.3 + ignore0.75, seed4 ep8 | 7/7 (1.000) | 0.0 | 0.4482 | 0.5198 | 0.3327 | -0.0101 | 18 | 2.57 |
| RESISC45 | transductive self-adaptation | DINOv3 frozen | 44/45 (0.978) | 0.0 | 0.7363 | 0.7694 | 0.4496 | 0.1709 | 67 | 1.49 |
| RESISC45 | transductive self-adaptation | SimCLR + queue4096 + ignore0.75, seed3 ep6 | 44/45 (0.978) | 0.0 | 0.7715 | 0.8605 | 0.587 | 0.2339 | 80 | 1.78 |
| DTD | transductive self-adaptation | DINOv3 frozen | 43/47 (0.915) | 0.0 | 0.6407 | 0.6791 | 0.3242 | 0.1891 | 75 | 1.6 |
| DTD | transductive self-adaptation | SimCLR + queue4096 + ignore0.75, seed3 ep8 | 46/47 (0.979) | 0.0 | 0.687 | 0.7621 | 0.4218 | 0.2352 | 88 | 1.87 |

## Primary Deltas

- WM-811K: ARI 0.1492 -> 0.3327 (delta +0.1835); P1 7/7 -> 7/7.
- RESISC45: ARI 0.4496 -> 0.5870 (delta +0.1374); P1 44/45 -> 44/45.
- DTD: ARI 0.3242 -> 0.4218 (delta +0.0976); P1 43/47 -> 46/47.
