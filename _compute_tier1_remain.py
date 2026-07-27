#!/usr/bin/env python3
"""B3/B4/B5/NEW outputs 의 clusters_global_list.txt 로 tier1 inline 계산."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (adjusted_mutual_info_score,
                             adjusted_rand_score,
                             completeness_score,
                             homogeneity_score)

REPO = Path("D:/project/unknown-contrastive")

# (outputs TS, cell tag)
TARGETS = [
    ("260521_102609", "B3_n500"),
    ("260521_141246", "B4_n500"),
    ("260521_174701", "B5_n500"),
    ("260521_214532", "NEW_n500"),
]


def tier1(out_dir: Path, tag: str):
    lst = out_dir / "clusters_global_list.txt"
    if not lst.exists():
        return None
    lines = lst.read_text(encoding="utf-8").strip().split("\n")
    rows = [ln.split("\t") for ln in lines[1:]]
    pred = np.array([int(r[0]) for r in rows])
    true_cls = np.array([r[1] for r in rows])
    classes = sorted(set(true_cls))
    cl2i = {c: i for i, c in enumerate(classes)}
    true = np.array([cl2i[c] for c in true_cls])

    TOTAL = 19250
    n_clustered = len(rows)
    noise = TOTAL - n_clustered
    noise_pct = noise / TOTAL * 100

    cluster_classes = defaultdict(Counter)
    for p, c in zip(pred, true_cls):
        cluster_classes[int(p)][c] += 1
    class_total = Counter(true_cls)
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

    return {
        "cell": tag,
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
    }


for ts, tag in TARGETS:
    d = REPO / f"outputs_contrastive_{ts}"
    if not d.exists():
        print(f"[MISS] {tag} ({ts}) — folder absent")
        continue
    r = tier1(d, tag)
    if r:
        out = d / f"tier1_{tag}.json"
        out.write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"=== {tag} (outputs {ts}) ===")
        for k, v in r.items():
            print(f"  {k:25s} {v}")
        print(f"  saved -> {out}")
        print()
    else:
        print(f"[MISS] {tag} ({ts}) — no clusters_global_list.txt")
