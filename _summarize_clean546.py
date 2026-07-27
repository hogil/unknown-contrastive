#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(r"D:\project\unknown-contrastive\runs\clean546")
FILES = [
    ("FCMAE frozen/B4 s42", ROOT / "eval_fcmae.json"),
    ("TAPT-B4 frozen/B4 s42", ROOT / "eval_b4.json"),
    ("FCMAE B0 s42", ROOT / "eval_fcmae_B0_s42.json"),
    ("FCMAE B1 s42", ROOT / "eval_fcmae_B1_s42.json"),
    ("FCMAE B3 s42", ROOT / "eval_fcmae_B3_s42.json"),
    ("FCMAE B3 s1", ROOT / "eval_fcmae_B3_s1.json"),
    ("FCMAE B3 s2", ROOT / "eval_fcmae_B3_s2.json"),
    ("FCMAE B4 s1", ROOT / "eval_fcmae_B4_s1.json"),
    ("FCMAE B4 s2", ROOT / "eval_fcmae_B4_s2.json"),
]


def p1_count(value):
    return int(str(value).split("/", 1)[0])


def row(label, data, key):
    item = data[key]
    lf, off = item["lf"], item["off"]
    return {
        "label": label,
        "epoch": data.get("selected_ep") if key == "selected" else "-",
        "P1": off["P1"],
        "P2": off["P2_noise"],
        "P3": off["P3_comp"],
        "P4": off["P4_hom"],
        "Sil": off["Sil"],
        "k": lf["k"],
        "frag": off["frag"],
        "noise": lf["noise_pct"],
        "stability": lf["stability"],
        "coherence": lf["coherence"],
        "over_merge": lf["over_merge"],
        "ARI": off["ARI"],
        "AMI": off["AMI"],
    }


rows = []
for label, path in FILES:
    if not path.exists():
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    if label == "FCMAE frozen/B4 s42":
        rows.append(row("FCMAE frozen", data, "frozen"))
        rows.append(row("FCMAE B4 s42", data, "selected"))
    elif label == "TAPT-B4 frozen/B4 s42":
        rows.append(row("TAPT-B4 frozen", data, "frozen"))
        rows.append(row("TAPT-B4 adapted s42", data, "selected"))
    elif data.get("selected") is not None:
        rows.append(row(label, data, "selected"))

headers = ["model", "ep", "P1", "P2", "P3", "P4", "Sil", "k", "frag", "noise", "stab", "coh", "over", "ARI", "AMI"]
lines = [
    "# MixedWM38 clean546 contrastive summary",
    "",
    "| " + " | ".join(headers) + " |",
    "|" + "|".join(["---"] * len(headers)) + "|",
]
for r in rows:
    lines.append(
        f"| {r['label']} | {r['epoch']} | {r['P1']} | {r['P2']:.2f} | "
        f"{r['P3']:.4f} | {r['P4']:.4f} | {r['Sil']:.4f} | {r['k']} | "
        f"{r['frag']:.4f} | {r['noise']:.2f} | {r['stability']:.4f} | "
        f"{r['coherence']:.4f} | {r['over_merge']} | {r['ARI']:.4f} | {r['AMI']:.4f} |"
    )

import numpy as np

for recipe in ("B3", "B4"):
    wanted = {f"FCMAE {recipe} s42", f"FCMAE {recipe} s1", f"FCMAE {recipe} s2"}
    seed_rows = [r for r in rows if r["label"] in wanted]
    if len(seed_rows) != 3:
        continue
    lines.extend(["", f"## FCMAE {recipe} three-seed", ""])
    for key in ("P1", "P2", "P3", "P4", "Sil", "noise", "stability", "coherence"):
        values = [p1_count(r[key]) if key == "P1" else float(r[key]) for r in seed_rows]
        lines.append(f"- {key}: {np.mean(values):.4f} +/- {np.std(values, ddof=1):.4f}")

out_md = ROOT / "ablation_summary.md"
out_json = ROOT / "ablation_summary.json"
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
print(out_md.read_text(encoding="utf-8"))
print(f"[OUT] {out_md}")
print(f"[OUT] {out_json}")
