#!/usr/bin/env python3
"""Build paper-style univariate/combination tables from live contrastive scores."""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = [
    REPO / "result_grouping" / "paper_contrastive_grid_260617",
    REPO / "result_grouping" / "paper_contrastive_unknown_grid_260617",
]


def read_rows(root: Path) -> list[dict[str, str]]:
    rows = []
    for p in root.rglob("scores_live.csv"):
        rows.extend(read_csv(p, root))
    for p in root.rglob("scores_all.csv"):
        rows.extend(read_csv(p, root))
    uniq = {}
    for r in rows:
        key = (r.get("condition", ""), r.get("embedding", ""), r.get("method", ""))
        uniq[key] = r
    return list(uniq.values())


def read_csv(path: Path, root: Path) -> list[dict[str, str]]:
    out = []
    cond = path.parent
    rel = cond.relative_to(root)
    parts = rel.parts
    split = parts[0] if parts else cond.name
    palette = parts[1] if len(parts) > 1 else ""
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            row = dict(row)
            row["condition"] = str(cond.resolve())
            row["split"] = split
            row["palette"] = palette
            row["score_file"] = str(path.resolve())
            row["recipe"] = recipe_name(row)
            row["family"] = recipe_family(row["recipe"])
            out.append(row)
    return out


def recipe_name(row: dict[str, str]) -> str:
    name = row.get("embedding_name") or Path(row.get("embedding", "")).stem
    return re.sub(r"_ep\d+$", "", name)


def recipe_family(recipe: str) -> str:
    if recipe in {"DINOv3_frozen", "FCMAE_frozen"} or recipe.endswith("_frozen"):
        return "00_no_train"
    if recipe.startswith("combo_"):
        return "10_combo"
    if recipe.startswith("simclr_t"):
        return "01_temp"
    if recipe.startswith("local") and "_q" not in recipe and "neco" not in recipe and "koleo" not in recipe:
        return "02_local"
    if recipe.startswith("q") and ("ignore" not in recipe and "_nv" not in recipe and "softnce" not in recipe and "sce" not in recipe):
        return "03_queue"
    if "_q" in recipe or "q4096_" in recipe:
        return "10_combo"
    if "neco" in recipe:
        return "11_neco"
    if "koleo" in recipe:
        return "12_koleo"
    if recipe.startswith("moco"):
        return "13_moco"
    if recipe.startswith("barlow"):
        return "14_barlow"
    if recipe.startswith("vicreg"):
        return "15_vicreg"
    return "99_other"


def flt(row: dict[str, str], key: str, default: float = -1e9) -> float:
    try:
        v = row.get(key, "")
        if v == "":
            return default
        return float(v)
    except Exception:
        return default


def best_per_key(rows: list[dict[str, str]], metric: str, lower: bool = False) -> list[dict[str, str]]:
    groups = defaultdict(list)
    for row in rows:
        if flt(row, "P1_capture", 0.0) < 1.0:
            continue
        if not (row.get("method", "").startswith("finch_p2") or row.get("method", "").startswith("louvain")):
            continue
        key = (row["split"], row["palette"], row["family"], row["recipe"], row["method"].split("(")[0])
        groups[key].append(row)
    best = []
    for key, vals in groups.items():
        pick = min(vals, key=lambda r: flt(r, metric, 1e9)) if lower else max(vals, key=lambda r: flt(r, metric))
        best.append(pick)
    return sorted(best, key=lambda r: (r["split"], r["palette"], r["family"], r["recipe"], r["method"]))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    cols = [
        "split", "palette", "family", "recipe", "embedding_name", "method",
        "P1_capture", "P2_noise_pct", "P3_completeness", "P4_homogeneity",
        "ARI", "AMI", "Sil", "k_total", "k_classes", "k_noise", "fragment_ratio",
        "condition", "embedding", "score_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_md(path: Path, title: str, rows: list[dict[str, str]], limit: int = 200) -> None:
    cols = ["split", "palette", "family", "recipe", "method", "P1_capture", "P2_noise_pct",
            "P3_completeness", "P4_homogeneity", "ARI", "Sil", "k_total", "k_noise", "fragment_ratio"]
    lines = [f"# {title}", "", "| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in rows[:limit]:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_gate(path: Path, rows: list[dict[str, str]]) -> None:
    by_cond = defaultdict(list)
    for row in rows:
        by_cond[(row["split"], row["palette"])].append(row)
    lines = ["# Combo Gate", ""]
    for (split, palette), vals in sorted(by_cond.items()):
        temp = [r for r in vals if r["family"] == "01_temp" and r["method"].startswith("finch_p2")]
        local = [r for r in vals if r["family"] == "02_local" and r["method"].startswith("finch_p2")]
        queue = [r for r in vals if r["family"] == "03_queue" and r["method"].startswith("finch_p2")]
        if not temp or not local or not queue:
            lines.append(f"## {split} / {palette}")
            lines.append("- Gate: not ready. Need completed temp, local, and queue univariate rows.")
            lines.append("")
            continue
        temp_top = sorted(temp, key=lambda r: flt(r, "ARI"), reverse=True)[:2]
        local_top = sorted(local, key=lambda r: flt(r, "ARI"), reverse=True)[:2]
        queue_top = sorted(queue, key=lambda r: flt(r, "ARI"), reverse=True)[:2]
        lines.append(f"## {split} / {palette}")
        lines.append("- Gate: ready after this split finishes; candidate combo grid should stay small.")
        lines.append("- Temp: " + ", ".join(f"{r['recipe']} ARI={r['ARI']}" for r in temp_top))
        lines.append("- Local: " + ", ".join(f"{r['recipe']} ARI={r['ARI']}" for r in local_top))
        lines.append("- Queue: " + ", ".join(f"{r['recipe']} ARI={r['ARI']}" for r in queue_top))
        lines.append("- Defer: NeCo/KoLeo/MoCo/ignore/NV/SoftNCE/SCE until local x queue x temp improves.")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=[str(p) for p in DEFAULT_ROOTS])
    ap.add_argument("--out-dir", default=str(REPO / "result_grouping" / "paper_contrastive_supervisor_260617"))
    args = ap.parse_args()

    roots = [Path(x) if Path(x).is_absolute() else REPO / x for x in args.roots]
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for root in roots:
        if root.exists():
            rows.extend(read_rows(root))
    ari = best_per_key(rows, "ARI", lower=False)
    frag = best_per_key(rows, "fragment_ratio", lower=True)
    write_csv(out / "paper_contrastive_univariate_ari.csv", ari)
    write_csv(out / "paper_contrastive_univariate_fragment.csv", frag)
    write_md(out / "paper_contrastive_univariate_ari.md", "Univariate ARI Table", ari)
    write_md(out / "paper_contrastive_univariate_fragment.md", "Univariate Fragment Table", frag)
    write_gate(out / "paper_contrastive_combo_gate.md", ari)
    print(f"[OUT] {out.resolve()}")


if __name__ == "__main__":
    main()
