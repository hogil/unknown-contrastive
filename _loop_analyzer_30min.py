"""
30-min polling loop for tier1_*_n500.json cluster analysis.
Read-only: analyzes eval_summary.json + cluster_report.parquet, writes md only.
"""
from __future__ import annotations

import glob
import json
import time
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path("D:/project/unknown-contrastive")
GLOB_PATTERN = str(ROOT / "outputs_contrastive_*" / "tier1_*_n500.json")
OUT_DIR = ROOT / "docs" / "contrastive-eval"
SUMMARY_PATH = OUT_DIR / "_loop_analyzer_30min_summary.md"
STATE_PATH = OUT_DIR / "_loop_analyzer_30min_state.json"
N_ITERS = 16
SLEEP_S = 30 * 60  # 30 min
MTIME_WINDOW_S = 30 * 60  # 30 min freshness window


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_state():
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"processed": []}


def save_state(state):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def find_new_cells(processed: set[str]) -> list[Path]:
    now = time.time()
    matches = []
    for p in glob.glob(GLOB_PATTERN):
        path = Path(p)
        mtime = path.stat().st_mtime
        if (now - mtime) > MTIME_WINDOW_S:
            continue
        # cell name = filename stem after tier1_ prefix
        stem = path.stem
        if stem.startswith("tier1_"):
            cell = stem[len("tier1_"):]
        else:
            cell = stem
        # cell tag includes _n500 already (e.g. iter78_n500). strip _n500 for tag
        cell_tag = cell.replace("_n500", "")
        key = f"{path.parent.name}::{cell_tag}"
        if key in processed:
            continue
        matches.append((path, cell_tag, key))
    return matches


def analyze_cell(tier1_path: Path, cell_tag: str) -> dict:
    """Analyze a single cell — returns dict with stats. Read-only."""
    run_dir = tier1_path.parent
    out = {
        "cell": cell_tag,
        "run_dir": str(run_dir),
        "tier1_path": str(tier1_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "errors": [],
    }

    # Read tier1 first
    try:
        out["tier1"] = json.loads(tier1_path.read_text(encoding="utf-8"))
    except Exception as e:
        out["errors"].append(f"tier1 read fail: {e}")
        return out

    eval_summary_path = run_dir / "eval" / "eval_summary.json"
    if eval_summary_path.exists():
        try:
            out["eval_summary"] = json.loads(eval_summary_path.read_text(encoding="utf-8"))
        except Exception as e:
            out["errors"].append(f"eval_summary read fail: {e}")

    # cluster_report.parquet
    parquet_path = run_dir / "eval" / "cluster_report.parquet"
    have_parquet = parquet_path.exists()
    if have_parquet:
        try:
            import pandas as pd  # noqa
            df = pd.read_parquet(parquet_path)
            out["cluster_report_cols"] = list(df.columns)
            out["cluster_report_n_rows"] = len(df)
            # Per-cluster size + cosine + silhouette + leakage
            if "cluster_id" in df.columns:
                grp = df.groupby("cluster_id")
                sizes = grp.size().to_dict()
                out["cluster_sizes"] = {int(k): int(v) for k, v in sizes.items()}
                out["n_clusters"] = int(len([c for c in sizes if int(c) >= 0]))
                out["noise_n"] = int(sizes.get(-1, 0))
            if "purity" in df.columns and "cluster_id" in df.columns:
                pur = df.groupby("cluster_id")["purity"].first().to_dict()
                out["cluster_purity"] = {int(k): float(v) for k, v in pur.items()}
            if "dom_class" in df.columns and "cluster_id" in df.columns:
                dom = df.groupby("cluster_id")["dom_class"].first().to_dict()
                out["cluster_dom_class"] = {int(k): str(v) for k, v in dom.items()}
        except Exception as e:
            out["errors"].append(f"cluster_report parse fail: {e}")

    # Embeddings → silhouette + cosine intra/inter
    emb_dir = run_dir / "eval" / "embeddings"
    emb_npy = emb_dir / "embedding.npy"
    files_txt = emb_dir / "files.txt"
    classes_txt = emb_dir / "classes.txt"
    if emb_npy.exists() and files_txt.exists() and classes_txt.exists() and have_parquet:
        try:
            import pandas as pd  # noqa
            X = np.load(emb_npy)
            with open(files_txt, "r", encoding="utf-8") as f:
                files = [ln.strip() for ln in f if ln.strip()]
            with open(classes_txt, "r", encoding="utf-8") as f:
                classes = [ln.strip() for ln in f if ln.strip()]
            df = pd.read_parquet(parquet_path)
            # Build label lookup by basename
            label_col = None
            for cand in ("file", "filename", "path", "basename"):
                if cand in df.columns:
                    label_col = cand
                    break
            cluster_col = "cluster_id" if "cluster_id" in df.columns else None
            if label_col is not None and cluster_col is not None and len(X) == len(files):
                # match by basename
                basename_to_cluster = {}
                for _, row in df.iterrows():
                    bn = os.path.basename(str(row[label_col]))
                    basename_to_cluster[bn] = int(row[cluster_col])
                labels = np.array([basename_to_cluster.get(os.path.basename(f), -1) for f in files], dtype=int)

                # L2-normalize for cosine
                Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

                # Per-cluster intra/inter cosine
                unique_clusters = sorted(set(int(c) for c in labels if c >= 0))
                centroids = {}
                intra_stats = {}
                for c in unique_clusters:
                    mask = labels == c
                    if mask.sum() < 2:
                        continue
                    members = Xn[mask]
                    centroid = members.mean(axis=0)
                    centroid /= (np.linalg.norm(centroid) + 1e-12)
                    centroids[c] = centroid
                    cos_sim = members @ centroid
                    cos_dist = 1.0 - cos_sim
                    intra_stats[c] = {
                        "n": int(mask.sum()),
                        "intra_mean": float(cos_dist.mean()),
                        "intra_median": float(np.median(cos_dist)),
                        "intra_p95": float(np.percentile(cos_dist, 95)),
                    }
                # Inter: nearest other centroid
                cs = sorted(centroids.keys())
                inter_stats = {}
                if len(cs) >= 2:
                    cmat = np.stack([centroids[c] for c in cs], axis=0)
                    sim = cmat @ cmat.T
                    np.fill_diagonal(sim, -np.inf)
                    for i, c in enumerate(cs):
                        # cosine distance = 1 - sim
                        nearest_sim = float(sim[i].max())
                        inter_stats[c] = {"inter_min": float(1.0 - nearest_sim)}
                out["intra_stats"] = {int(k): v for k, v in intra_stats.items()}
                out["inter_stats"] = {int(k): v for k, v in inter_stats.items()}

                # Silhouette (sample if too large)
                try:
                    from sklearn.metrics import silhouette_samples
                    defect_mask = labels >= 0
                    if defect_mask.sum() > 20:
                        Xs = Xn[defect_mask]
                        ls = labels[defect_mask]
                        n_lab = len(set(ls.tolist()))
                        if n_lab >= 2:
                            n_max = 5000
                            if Xs.shape[0] > n_max:
                                rng = np.random.default_rng(42)
                                idx = rng.choice(Xs.shape[0], n_max, replace=False)
                                Xs = Xs[idx]
                                ls = ls[idx]
                            sil = silhouette_samples(Xs, ls, metric="cosine")
                            per_c = {}
                            for c in set(ls.tolist()):
                                m = ls == c
                                if m.sum() > 0:
                                    per_c[int(c)] = float(sil[m].mean())
                            out["silhouette_per_cluster"] = per_c
                            out["silhouette_mean"] = float(sil.mean())
                except Exception as e:
                    out["errors"].append(f"silhouette fail: {e}")

                # Normal leakage: any file in defect cluster but class = Normal*
                try:
                    if len(classes) == len(files):
                        cls_arr = np.array(classes)
                        normal_mask = np.array([c.lower().startswith("normal") for c in cls_arr])
                        leak = int(((labels >= 0) & normal_mask).sum())
                        n_normal = int(normal_mask.sum())
                        normal_noise = int(((labels == -1) & normal_mask).sum())
                        out["normal_total"] = n_normal
                        out["normal_in_defect_cluster"] = leak
                        out["normal_in_noise"] = normal_noise
                        if n_normal > 0:
                            out["normal_noise_pct"] = round(100.0 * normal_noise / n_normal, 2)
                            out["normal_leakage_pct"] = round(100.0 * leak / n_normal, 2)
                except Exception as e:
                    out["errors"].append(f"normal leakage fail: {e}")
        except Exception as e:
            out["errors"].append(f"embeddings analysis fail: {e}")

    return out


def write_md(out_path: Path, analysis: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cell = analysis["cell"]
    tier1 = analysis.get("tier1", {})
    es = analysis.get("eval_summary", {})
    lines = []
    lines.append(f"# Cluster analysis — {cell} (n500)")
    lines.append("")
    lines.append(f"generated: {analysis['generated_at']}")
    lines.append(f"run_dir: `{analysis['run_dir']}`")
    lines.append(f"tier1: `{analysis['tier1_path']}`")
    lines.append("")

    lines.append("## 1. Tier1 metrics (read-only)")
    for k in ("ARI", "NMI", "AMI", "homogeneity", "completeness", "purity",
              "silhouette", "n_clusters", "noise_pct", "normal_noise_pct",
              "class_capture_rate"):
        if k in tier1:
            lines.append(f"- {k}: {tier1[k]}")
    if not any(k in tier1 for k in ("ARI", "NMI")):
        # dump all
        for k, v in tier1.items():
            if isinstance(v, (int, float, str, bool)):
                lines.append(f"- {k}: {v}")
    lines.append("")

    if es:
        lines.append("## 2. eval_summary excerpt")
        for k in ("hdbscan_cfg", "n_clusters", "normal_noise_pct",
                  "per_class_noise"):
            if k in es:
                v = es[k]
                if isinstance(v, dict):
                    lines.append(f"- {k}:")
                    for kk, vv in v.items():
                        lines.append(f"  - {kk}: {vv}")
                else:
                    lines.append(f"- {k}: {v}")
        lines.append("")

    sizes = analysis.get("cluster_sizes", {})
    if sizes:
        defect_sizes = {k: v for k, v in sizes.items() if k >= 0}
        lines.append("## 3. Cluster size distribution")
        n_def = len(defect_sizes)
        vals = list(defect_sizes.values())
        if vals:
            lines.append(f"- n_clusters (defect): {n_def}")
            lines.append(f"- size mean: {np.mean(vals):.1f}")
            lines.append(f"- size median: {np.median(vals):.0f}")
            lines.append(f"- size p10: {np.percentile(vals, 10):.0f}")
            lines.append(f"- size p90: {np.percentile(vals, 90):.0f}")
            lines.append(f"- size min: {min(vals)}, max: {max(vals)}")
        if -1 in sizes:
            lines.append(f"- noise (cluster=-1): {sizes[-1]}")
        lines.append("")

    intra = analysis.get("intra_stats", {})
    inter = analysis.get("inter_stats", {})
    sil = analysis.get("silhouette_per_cluster", {})
    pur = analysis.get("cluster_purity", {})
    dom = analysis.get("cluster_dom_class", {})

    if intra:
        lines.append("## 4. Per-cluster table (top 20 by size + weak)")
        lines.append("")
        lines.append("| id | size | dom_class | purity | sil | intra_p95 | inter_min |")
        lines.append("|---|---|---|---|---|---|---|")
        defect_sizes = {k: v for k, v in sizes.items() if k >= 0}
        sorted_ids = sorted(defect_sizes.keys(), key=lambda k: -defect_sizes[k])
        # take top 20 + any weak
        top_ids = sorted_ids[:20]
        weak_ids = []
        for c in sorted_ids:
            s = sil.get(c, 1.0)
            p = pur.get(c, 1.0)
            i_p95 = intra.get(c, {}).get("intra_p95", 0.0)
            i_min = inter.get(c, {}).get("inter_min", 1.0)
            i_med = intra.get(c, {}).get("intra_median", 0.0)
            is_weak = (s < 0.3) or (i_p95 > 2 * i_med if i_med > 0 else False) or (p < 0.6) or (i_min < i_p95)
            if is_weak and c not in top_ids:
                weak_ids.append(c)
        show_ids = top_ids + weak_ids[:20]
        for c in show_ids:
            sz = defect_sizes.get(c, 0)
            dc = dom.get(c, "?")
            p = pur.get(c, float("nan"))
            s = sil.get(c, float("nan"))
            i_p95 = intra.get(c, {}).get("intra_p95", float("nan"))
            i_min = inter.get(c, {}).get("inter_min", float("nan"))
            mark = ""
            if (s < 0.3 if not np.isnan(s) else False) or \
               (p < 0.6 if not np.isnan(p) else False) or \
               (i_min < i_p95 if not (np.isnan(i_min) or np.isnan(i_p95)) else False):
                mark = " (weak)"
            lines.append(f"| {c} | {sz} | {dc} | {p:.2f} | {s:.2f} | {i_p95:.3f} | {i_min:.3f} |{mark}")
        lines.append("")
        if weak_ids:
            lines.append(f"## 5. Weak clusters ({len(weak_ids)})")
            for c in weak_ids:
                reasons = []
                s = sil.get(c, 1.0)
                p = pur.get(c, 1.0)
                i_p95 = intra.get(c, {}).get("intra_p95", 0.0)
                i_min = inter.get(c, {}).get("inter_min", 1.0)
                i_med = intra.get(c, {}).get("intra_median", 0.0)
                if s < 0.3: reasons.append("low_silhouette")
                if p < 0.6: reasons.append("low_purity")
                if i_min < i_p95: reasons.append("boundary_blur")
                if i_med > 0 and i_p95 > 2 * i_med: reasons.append("intra_skew")
                sz = defect_sizes.get(c, 0)
                lines.append(f"- cluster {c} (size {sz}, dom {dom.get(c, '?')}): {', '.join(reasons)}")
            lines.append("")

    # Normal leakage
    if "normal_total" in analysis:
        lines.append("## 6. Normal leakage")
        lines.append(f"- normal_total: {analysis['normal_total']}")
        lines.append(f"- normal_in_defect_cluster: {analysis['normal_in_defect_cluster']}")
        lines.append(f"- normal_in_noise: {analysis['normal_in_noise']}")
        if "normal_noise_pct" in analysis:
            lines.append(f"- normal_noise_pct: {analysis['normal_noise_pct']}%")
        if "normal_leakage_pct" in analysis:
            lines.append(f"- normal_leakage_pct: {analysis['normal_leakage_pct']}%")
        lines.append("")

    # Class fragmentation
    if dom:
        from collections import Counter
        cls_count = Counter(dom.values())
        frag = {k: v for k, v in cls_count.items() if v >= 2}
        if frag:
            lines.append("## 7. Fragmented classes (dom_class in >=2 clusters)")
            for k, v in sorted(frag.items(), key=lambda x: -x[1]):
                ids = [c for c, d in dom.items() if d == k]
                lines.append(f"- {k}: {v} clusters → ids {sorted(ids)}")
            lines.append("")

    if analysis.get("errors"):
        lines.append("## errors")
        for e in analysis["errors"]:
            lines.append(f"- {e}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    processed = set(state.get("processed", []))

    iter_log = []
    for it in range(1, N_ITERS + 1):
        log(f"=== iter {it}/{N_ITERS} ===")
        try:
            new_cells = find_new_cells(processed)
            log(f"new cells: {len(new_cells)}")
            iter_record = {"iter": it, "ts": datetime.now(timezone.utc).isoformat(),
                           "n_new": len(new_cells), "cells": []}
            for path, cell_tag, key in new_cells:
                try:
                    log(f"  analyzing {cell_tag} ({path.parent.name})")
                    analysis = analyze_cell(path, cell_tag)
                    out_md = OUT_DIR / f"cluster_analysis_{cell_tag}_n500.md"
                    write_md(out_md, analysis)
                    log(f"    wrote {out_md.name}")
                    processed.add(key)
                    iter_record["cells"].append({
                        "cell": cell_tag,
                        "out_md": str(out_md),
                        "n_clusters": analysis.get("n_clusters"),
                        "noise_n": analysis.get("noise_n"),
                        "errors": analysis.get("errors", []),
                    })
                except Exception as e:
                    log(f"    FAIL: {e}")
                    iter_record["cells"].append({"cell": cell_tag, "error": str(e)})
            iter_log.append(iter_record)
            state["processed"] = sorted(processed)
            state["iter_log"] = iter_log
            save_state(state)
        except Exception as e:
            log(f"iter exception: {e}")

        if it < N_ITERS:
            log(f"sleeping {SLEEP_S}s...")
            time.sleep(SLEEP_S)

    # Final summary
    lines = [f"# Loop analyzer 30-min summary",
             f"",
             f"finished: {datetime.now(timezone.utc).isoformat()}",
             f"iters: {N_ITERS}",
             f"total cells processed: {len(processed)}",
             ""]
    lines.append("## Per-iter")
    for rec in iter_log:
        lines.append(f"- iter {rec['iter']} ({rec['ts']}): {rec['n_new']} new cells")
        for c in rec.get("cells", []):
            if "error" in c:
                lines.append(f"  - {c['cell']}: ERROR {c['error']}")
            else:
                lines.append(f"  - {c['cell']}: n_clusters={c.get('n_clusters')}, noise={c.get('noise_n')}")
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    log(f"wrote summary {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
