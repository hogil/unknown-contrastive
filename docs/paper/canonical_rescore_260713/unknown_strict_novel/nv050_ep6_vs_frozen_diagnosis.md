# Hard-Unknown FINCH-p2 Delta Diagnosis

Both rows cluster the full pool and score the same 32 strict-novel defect classes.

| Metric | DINOv3 frozen | NV 0.50 ep6 | Delta |
|---|---:|---:|---:|
| P1 | 0.9375 | 0.9688 | +0.0313 |
| P2 | 0.0000 | 0.0000 | +0.0000 |
| P3 | 0.8302 | 0.8312 | +0.0010 |
| P4 | 0.8918 | 0.8894 | -0.0024 |
| ARI | 0.7090 | 0.6742 | -0.0348 |
| Sil | 0.3045 | 0.3122 | +0.0077 |
| k | 58.0000 | 57.0000 | -1.0000 |
| fragment | 1.8125 | 1.7812 | -0.0312 |

- Regained unique-dominant captures: Center_fork, Edge-Bottom_scratch_rot.
- Lost unique-dominant captures: Edge-Top_scratch.
- The detailed CSV reports the target class's largest-cluster coverage and the main class of that cluster; it distinguishes boundary reassignment from simple cluster-count change.
