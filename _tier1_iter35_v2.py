"""iter 35 Tier 1+2 from file_list.parquet (no cluster_ids.npy in this run)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    completeness_score,
    homogeneity_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize

run = Path(r"D:/project/unknown-contrastive/outputs_contrastive_260508_162812")
emb = np.load(run / "eval/embeddings/embedding.npy")
df = pd.read_parquet(run / "eval/file_list.parquet")
df["label"] = df["path"].apply(lambda p: Path(p).parent.name)

# Same order as embedding (assumed)
labels = df["label"].values
cl = df["cluster_id"].values
classes = sorted(set(labels))
class_to_idx = {c: i for i, c in enumerate(classes)}
y = np.array([class_to_idx[l] for l in labels])
mask_def = labels != "Normal"


def metrics(y, c, emb):
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
    mc = c >= 0
    if mc.sum() > 10 and len(set(c[mc])) > 1:
        emb_n = normalize(emb[mc], axis=1)
        out["Silhouette_cos"] = float(silhouette_score(emb_n, c[mc], metric="euclidean"))
    classes_seen = sum(1 for cls in set(y) if ((y == cls) & (c >= 0)).any())
    out["class_capture_rate"] = float(classes_seen / len(set(y)))
    return out


w = metrics(y, cl, emb)
wo = metrics(y[mask_def], cl[mask_def], emb[mask_def])
print("=== iter 35 Tier 1+2 (Iter 1 P2 King cfg on new anchor) ===")
print(f"{'metric':<22} {'with_normal':>10} {'without_normal':>15}")
print("-" * 50)
for k in ["n", "n_clusters", "noise_pct", "Completeness", "Homogeneity",
          "AMI", "NMI", "ARI", "Silhouette_cos", "class_capture_rate"]:
    a, b = w.get(k, "—"), wo.get(k, "—")
    af = f"{a:.4f}" if isinstance(a, float) else str(a)
    bf = f"{b:.4f}" if isinstance(b, float) else str(b)
    print(f"{k:<22} {af:>10} {bf:>15}")

(run / "tier1_iter35.json").write_text(
    json.dumps({"with_normal": w, "without_normal": wo}, indent=2),
    encoding="utf-8")
print(f"\nsaved: {run / 'tier1_iter35.json'}")
