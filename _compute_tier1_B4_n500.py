#!/usr/bin/env python3
"""B4 cross-eval (260519_114912) 의 clusters_global_list.txt 로 tier1 metric 계산."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (adjusted_mutual_info_score,
                             adjusted_rand_score,
                             completeness_score,
                             homogeneity_score)

OUT_DIR = Path("D:/project/unknown-contrastive/outputs_contrastive_260519_114912")
LIST_PATH = OUT_DIR / "clusters_global_list.txt"
TIER1_OUT = OUT_DIR / "tier1_B4_n500.json"

# load
lines = LIST_PATH.read_text(encoding="utf-8").strip().split("\n")
header = lines[0].split("\t")
rows = [ln.split("\t") for ln in lines[1:]]
# col 0 = cluster_id, col 1 = class
pred = np.array([int(r[0]) for r in rows])
true_cls = np.array([r[1] for r in rows])
# str class → int
class_names = sorted(set(true_cls))
class_to_idx = {c: i for i, c in enumerate(class_names)}
true = np.array([class_to_idx[c] for c in true_cls])

n_clustered = len(rows)
TOTAL = 19250  # log 의 samples
noise = TOTAL - n_clustered
noise_pct = noise / TOTAL * 100

# class_capture_rate
# = 각 class 별 cluster 매칭 (cluster purity 기반)
cluster_classes = defaultdict(Counter)
for p, c in zip(pred, true_cls):
    cluster_classes[int(p)][c] += 1

# class 가 가장 많이 속한 cluster 의 비율
class_total = Counter(true_cls)
class_capture = {}
for cls, total in class_total.items():
    # cls 가 어디에 가장 많이 있는지
    max_in_cluster = max(
        (cnt for cl, ccnt in cluster_classes.items() for c, cnt in ccnt.items() if c == cls),
        default=0
    )
    class_capture[cls] = max_in_cluster / total
capture_rate = float(np.mean(list(class_capture.values())))

# Tier 1+2 official sklearn metrics
ari = float(adjusted_rand_score(true, pred))
ami = float(adjusted_mutual_info_score(true, pred))
hom = float(homogeneity_score(true, pred))
comp = float(completeness_score(true, pred))

n_clusters = len(set(pred))

result = {
    "cell": "B4_n500_xeval",
    "model_source": "outputs_contrastive_260511_181441 (avg30 anchor B4)",
    "eval_data": "E:/data/images/unknown (per_class=500, normal=2000)",
    "total_samples": TOTAL,
    "clustered_samples": int(n_clustered),
    "noise_count": int(noise),
    "noise_pct": round(noise_pct, 2),
    "n_clusters": int(n_clusters),
    "class_capture_rate": round(capture_rate, 4),
    "completeness": round(comp, 4),
    "homogeneity": round(hom, 4),
    "ari": round(ari, 4),
    "ami": round(ami, 4),
    "silhouette_cosine": None,  # embedding 없어 계산 skip
    "n_classes": len(class_names),
}

TIER1_OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"=== B4 n500 cross-eval tier1 ===")
for k, v in result.items():
    print(f"  {k:25s} {v}")
print(f"\nsaved: {TIER1_OUT}")
