# Hard-Unknown FINCH-p2 Delta Diagnosis

Both rows cluster the full pool and score the same 32 strict-novel defect classes.

| Metric | DINOv3 frozen | nv050_blend086_ep6 | Delta |
|---|---:|---:|---:|
| P1 | 0.9375 | 0.9375 | +0.0000 |
| P2 | 0.0000 | 0.0000 | +0.0000 |
| P3 | 0.8302 | 0.8588 | +0.0286 |
| P4 | 0.8918 | 0.9024 | +0.0106 |
| ARI | 0.7090 | 0.7390 | +0.0300 |
| Sil | 0.3045 | 0.3479 | +0.0434 |
| k | 58.0000 | 51.0000 | -7.0000 |
| fragment | 1.8125 | 1.5938 | -0.2188 |

- Regained unique-dominant captures: Center_fork, Edge-Bottom_scratch_rot.
- Lost unique-dominant captures: Edge-Bottom_scratch, Edge-Top_bank_boundary.
- The detailed CSV reports the target class's largest-cluster coverage and the main class of that cluster; it distinguishes boundary reassignment from simple cluster-count change.
