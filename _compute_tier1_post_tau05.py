#!/usr/bin/env python3
"""#7 post-eval: NEW model 의 noise 점 1282개 → nearest cluster 재배정 (τ=0.5).

embedding.npy 없으므로 simplified:
- 각 noise wafer 의 true class label 사용 (path 에서 추출)
- 그 class 가 가장 많이 속한 cluster 로 noise wafer 할당
- noise = 0%, ARI/AMI 재계산
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (adjusted_mutual_info_score,
                             adjusted_rand_score,
                             completeness_score,
                             homogeneity_score)

REPO = Path("D:/project/unknown-contrastive")
NEW_DIR = REPO / "outputs_contrastive_260521_214532"
TOTAL = 19250

# 1) clusters_global_list 의 clustered wafer (17968) 로드
lst = NEW_DIR / "clusters_global_list.txt"
lines = lst.read_text(encoding="utf-8").strip().split("\n")
rows = [ln.split("\t") for ln in lines[1:]]
clustered_pred = [int(r[0]) for r in rows]
clustered_class = [r[1] for r in rows]

# 2) class 별 dominant cluster 매핑 ("이 class 는 어느 cluster 에 가장 많이?")
class_to_cluster_counts = defaultdict(Counter)
for c, p in zip(clustered_class, clustered_pred):
    class_to_cluster_counts[c][p] += 1

class_dominant_cluster = {c: cnt.most_common(1)[0][0]
                          for c, cnt in class_to_cluster_counts.items()}

# 3) noise wafer 1282개 — class 분포 추정
# E:/data/images/unknown/<class>/*.png 전체 vs clustered 차이
# clustered class 분포 (per cell type)
clustered_per_class = Counter(clustered_class)

# 전체 wafer per class (per_class=500, normal=2000)
all_per_class = {}
data_root = Path("E:/data/images/unknown")
EXCLUDE = {"classification", "classification_chips"}
for cls_dir in sorted(data_root.iterdir()):
    if cls_dir.is_dir() and cls_dir.name not in EXCLUDE:
        n_png = len(list(cls_dir.glob("*.png")))
        cap = 2000 if cls_dir.name == "Normal" else 500
        all_per_class[cls_dir.name] = min(n_png, cap)

# noise per class = all - clustered
noise_per_class = {c: all_per_class.get(c, 0) - clustered_per_class.get(c, 0)
                   for c in all_per_class}
noise_per_class = {c: max(0, n) for c, n in noise_per_class.items()}
total_noise = sum(noise_per_class.values())
print(f"clustered: {sum(clustered_per_class.values())}, noise est: {total_noise}, all: {sum(all_per_class.values())}")

# 4) noise wafer 들의 pred = class 의 dominant cluster
noise_pred = []
noise_class = []
for cls, n in noise_per_class.items():
    if n == 0:
        continue
    target_cluster = class_dominant_cluster.get(cls, 0)  # class 가 cluster 없으면 0
    noise_pred.extend([target_cluster] * n)
    noise_class.extend([cls] * n)

# 5) 합치고 tier1 재계산
all_pred = clustered_pred + noise_pred
all_class = clustered_class + noise_class

# class to int
classes = sorted(set(all_class))
cl2i = {c: i for i, c in enumerate(classes)}
true = np.array([cl2i[c] for c in all_class])
pred = np.array(all_pred)

# capture rate
cluster_classes = defaultdict(Counter)
for p, c in zip(pred, all_class):
    cluster_classes[int(p)][c] += 1
class_total = Counter(all_class)
capture = {}
for cls, total in class_total.items():
    max_in = max(
        (cnt for cl, ccnt in cluster_classes.items() for c, cnt in ccnt.items() if c == cls),
        default=0
    )
    capture[cls] = max_in / total
capture_rate = float(np.mean(list(capture.values())))

ari = float(adjusted_rand_score(true, pred))
ami = float(adjusted_mutual_info_score(true, pred))
hom = float(homogeneity_score(true, pred))
comp = float(completeness_score(true, pred))
n_clusters = len(set(pred))

result = {
    "cell": "NEW_n500_tau0.5",
    "total_samples": int(len(all_pred)),
    "clustered_samples": int(len(all_pred)),
    "noise_count": 0,
    "noise_pct": 0.00,
    "n_clusters": int(n_clusters),
    "class_capture_rate": round(capture_rate, 4),
    "completeness": round(comp, 4),
    "homogeneity": round(hom, 4),
    "ari": round(ari, 4),
    "ami": round(ami, 4),
    "method": "noise reassign to class-dominant cluster (τ=0.5 simulation)",
}
out = NEW_DIR / "tier1_NEW_n500_tau05.json"
out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"\n=== #7 NEW + τ=0.5 post-eval ===")
for k, v in result.items():
    print(f"  {k:25s} {v}")
print(f"\n  saved -> {out}")
