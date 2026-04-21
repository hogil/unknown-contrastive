import json
import os

data = json.load(open("reports/defect_only_eval_6runs.json"))
labels = {
    "baseline_260420_181745": "smoke",
    "baseline_260420_201121": "10ep",
    "strong_augment_260420_234333": "v2",
    "hard_negatives_260421_055208": "v3",
    "deeper_head_260421_064156": "v4",
    "deeper_head_260421_073523": "v5",
}

print("== Table A: defect-only metrics ==")
print("| run | split | ARI | NMI | purity | silhouette | sil_n_used | noise_ratio | n_clusters |")
print("|---|---|---|---|---|---|---|---|---|")
for e in data:
    name = os.path.basename(e["run_dir"])
    lbl = labels.get(name, name)
    for split in ("val", "test"):
        s = e[split]
        sil = s["silhouette"]
        sil_s = f"{sil:.3f}" if sil is not None else "n/a"
        print(f"| {lbl} | {split} | {s['ari']:.3f} | {s['nmi']:.3f} | "
              f"{s['cluster_purity']:.3f} | {sil_s} | {s['silhouette_n_used']} | "
              f"{s['defect_noise_ratio']:.3f} | {s['n_clusters_found']} |")

print()
print("== Table B: cluster group breakdown (defect-only subset) ==")
for e in data:
    name = os.path.basename(e["run_dir"])
    lbl = labels.get(name, name)
    for split in ("val", "test"):
        b = e[split]["per_cluster_defect_breakdown"]
        print(f"\n--- {lbl} {split} ---")
        items = sorted(b.items(), key=lambda kv: -kv[1]["size"])
        for cid, info in items:
            flag = "DEFECT" if info["is_defect_cluster"] else "none-dom"
            mix = info["class_mix"]
            mix_s = ", ".join(f"{k}:{v}" for k, v in mix.items())
            print(f"  c{cid:>3}  size={info['size']:>2}  "
                  f"dom={info['dominant_class']:<10} purity={info['purity']:.2f}  "
                  f"[{flag}]  mix=({mix_s})")
