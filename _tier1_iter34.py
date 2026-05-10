"""iter 34 baseline Tier 1+2 official metrics — sklearn ground-truth."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    completeness_score,
    homogeneity_score,
    silhouette_score,
)

run = Path(r"D:/project/unknown-contrastive/outputs_contrastive_260508_123101")
emb = np.load(run / "eval/embeddings/embedding.npy")
cl = np.load(run / "eval/embeddings/cluster_ids.npy")
files = (run / "eval/embeddings/files.txt").read_text(encoding="utf-8").splitlines()
classes = (run / "eval/embeddings/classes.txt").read_text(encoding="utf-8").splitlines()

# label = parent dir name
labels = np.array([Path(f).parent.name for f in files])
class_to_idx = {c: i for i, c in enumerate(classes)}
y = np.array([class_to_idx[l] for l in labels])

mask_def = labels != "Normal"
mask_clean = cl >= 0  # exclude noise

def metrics(y, c, emb, mask_clean):
    cleanmask = mask_clean
    out = {
        "n": int(len(y)),
        "n_noise": int((c == -1).sum()),
        "noise_pct": float((c == -1).mean() * 100),
        "n_clusters": int(len(set(c)) - (1 if -1 in c else 0)),
        "Completeness": float(completeness_score(y, c)),
        "Homogeneity": float(homogeneity_score(y, c)),
        "AMI": float(adjusted_mutual_info_score(y, c)),
        "NMI": float(normalized_mutual_info_score(y, c)),
        "ARI": float(adjusted_rand_score(y, c)),
    }
    # silhouette on non-noise only, cosine
    if cleanmask.sum() > 10 and len(set(c[cleanmask])) > 1:
        from sklearn.preprocessing import normalize
        emb_n = normalize(emb[cleanmask], axis=1)
        out["Silhouette_cos"] = float(silhouette_score(emb_n, c[cleanmask], metric="euclidean"))
    # class_capture_rate: GT class 가 적어도 1 cluster 에 majority? 아니면 적어도 1 wafer 가 cluster 에 속함?
    # contrastive-eval P1 정의: GT class 의 적어도 1 wafer 가 noise 가 아닌 cluster 에 속함 (class 누락 0).
    classes_seen = []
    for cls_idx in set(y):
        in_cluster = ((y == cls_idx) & (c >= 0)).any()
        classes_seen.append(in_cluster)
    out["class_capture_rate"] = float(np.mean(classes_seen))
    out["n_classes_total"] = int(len(set(y)))
    out["n_classes_captured"] = int(sum(classes_seen))
    return out

w_norm = metrics(y, cl, emb, mask_clean)

# without_normal
y_d = y[mask_def]
c_d = cl[mask_def]
e_d = emb[mask_def]
mc_d = c_d >= 0
wo_norm = metrics(y_d, c_d, e_d, mc_d)

print("=== iter 34 newdata baseline (Iter 14 Quality King + new anchor) ===")
print()
print(f"{'metric':<22} {'with_normal':>12} {'without_normal':>15}")
print("-" * 55)
for k in ["n", "n_clusters", "n_noise", "noise_pct",
          "Completeness", "Homogeneity", "AMI", "NMI", "ARI",
          "Silhouette_cos", "class_capture_rate",
          "n_classes_captured", "n_classes_total"]:
    a = w_norm.get(k, "—")
    b = wo_norm.get(k, "—")
    af = f"{a:.4f}" if isinstance(a, float) else str(a)
    bf = f"{b:.4f}" if isinstance(b, float) else str(b)
    print(f"{k:<22} {af:>12} {bf:>15}")

# save
out = {"with_normal": w_norm, "without_normal": wo_norm,
       "config": "iter_34_newdata_quality_king",
       "anchor": "avg30_new_260508_123037"}
(run / "tier1_iter34.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print()
print(f"saved: {run / 'tier1_iter34.json'}")
