#!/usr/bin/env python3
"""Canonical strict-novel rescore for retained hard-unknown embeddings.

The score contract is intentionally fixed here:
  * cluster the complete eval pool, including Normal and known defects;
  * score only the 32 defect classes not used for defect-aware adaptation;
  * P1 is a unique dominant/main-class capture, never sample coverage;
  * FINCH-p2 is primary and Louvain is a clusterer sanity check.

The script checkpoints its CSV after each embedding, so a long CPU rescore can
be resumed without recomputing completed rows.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS = ROOT / "result_grouping" / "_unknown_mixed260710" / "embeddings"
FROZEN = ROOT / "result_grouping" / "_field_robust" / "embeddings" / "frozen_unknown_dinov3_grade_only_260709.npy"
POOL = ROOT / "data" / "images" / "unknown_eval100"
OUTPUT = ROOT / "docs" / "paper" / "canonical_rescore_260713" / "unknown_strict_novel"
EXCLUDED = (
    "Normal,Random,R,Center_bank_boundary,Center_scratch,Donut_bank_boundary,"
    "Donut_fork,Edge-Ring_bank_boundary,Edge-Ring_scratch,Edge-Top_fork,"
    "Full_scratch,ParallelScratches,RingDots"
)
NAME_RE = re.compile(r"^(?P<recipe>unkda_(?:base|nv(?P<threshold>\d{3})))_ep(?P<epoch>\d+)$")
METHODS = ("finch_p2", "louvain_res6")


def load_score_module():
    path = ROOT / "_score_umapfree.py"
    spec = importlib.util.spec_from_file_location("canonical_score", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load scorer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, default=EMBEDDINGS)
    parser.add_argument("--frozen", type=Path, default=FROZEN)
    parser.add_argument("--pool", type=Path, default=POOL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--refresh", action="store_true", help="ignore cached rows")
    return parser.parse_args()


def discover(root: Path, frozen: Path) -> list[dict[str, object]]:
    specs = [
        {
            "recipe": "frozen",
            "epoch": 0,
            "threshold": np.nan,
            "embedding": frozen.resolve(),
            "label": "DINOv3 frozen",
        }
    ]
    for path in sorted(root.glob("unkda_*_ep*.npy")):
        if path.stem.endswith("_proj"):
            continue
        match = NAME_RE.fullmatch(path.stem)
        if match is None:
            continue
        threshold = match.group("threshold")
        specs.append(
            {
                "recipe": match.group("recipe"),
                "epoch": int(match.group("epoch")),
                "threshold": float(threshold) / 100.0 if threshold else np.nan,
                "embedding": path.resolve(),
                "label": "SimCLR base" if match.group("recipe") == "unkda_base" else f"NV {int(threshold) / 100:.2f}",
            }
        )
    return specs


def metric_row(scorelib, z: np.ndarray, pred: np.ndarray, labels: list[str], excluded: set[str]) -> dict[str, object]:
    result = scorelib.score(pred, labels, excluded)
    labels_array = np.asarray(labels)
    target = ~np.isin(labels_array, list(excluded))
    target_pred = np.asarray(pred)[target]
    non_noise = target_pred != -1
    try:
        sil = (
            float(silhouette_score(z[target][non_noise], target_pred[non_noise], metric="cosine"))
            if non_noise.sum() > 10 and len(set(target_pred[non_noise])) > 1
            else np.nan
        )
    except ValueError:
        sil = np.nan
    return {
        "P1_capture": result["capture"],
        "P1_capture_count": result["capture_count"],
        "P1_target_class_count": result["target_class_count"],
        "legacy_presence_rate": result["legacy_presence_rate"],
        "P2_noise_pct": result["noise_pct"],
        "P3_completeness": result["completeness"],
        "P4_homogeneity": result["homogeneity"],
        "ARI": result["ari"],
        "AMI": result["ami"],
        "Sil": sil,
        "k_total": result["k_tot"],
        "k_classes": result["n_classes"],
        "k_noise": result["k_noise"],
        "fragment_ratio": result["k_tot"] / max(1, result["n_classes"]),
    }


def score_spec(scorelib, labels: list[str], excluded: set[str], spec: dict[str, object]) -> list[dict[str, object]]:
    embedding = Path(spec["embedding"])
    if not embedding.exists():
        raise FileNotFoundError(f"Missing embedding: {embedding}")
    z = scorelib.FP.l2(np.load(embedding).astype(np.float32))
    if z.shape[0] != len(labels):
        raise ValueError(f"Label/embedding mismatch for {embedding}: {len(labels)} vs {z.shape[0]}")
    finch = scorelib.run_finch(z)
    predictions = {"finch_p2": next(pred for name, pred in finch.items() if name.startswith("finch_p2("))}
    predictions["louvain_res6"] = scorelib.run_louvain(z)
    stamp = embedding.stat().st_mtime_ns
    rows = []
    for method, pred in predictions.items():
        row = metric_row(scorelib, z, pred, labels, excluded)
        row.update(spec)
        row["embedding"] = str(embedding)
        row["embedding_mtime_ns"] = stamp
        row["method"] = method
        rows.append(row)
    return rows


def load_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def cached_methods(frame: pd.DataFrame, spec: dict[str, object]) -> set[str]:
    if frame.empty:
        return set()
    path = str(spec["embedding"])
    stamp = Path(path).stat().st_mtime_ns
    rows = frame[(frame["embedding"] == path) & (frame["embedding_mtime_ns"] == stamp)]
    return set(rows["method"])


def save_cache(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["recipe", "epoch", "method"], inplace=True)
    frame.to_csv(path, index=False)


def display_recipe(recipe: str) -> str:
    if recipe == "frozen":
        return "Frozen"
    if recipe == "unkda_base":
        return "SimCLR base"
    return f"NV {int(recipe.rsplit('nv', 1)[1]) / 100:.2f}"


def add_gates(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for method in METHODS:
        baseline = frame[(frame["recipe"] == "frozen") & (frame["method"] == method)]
        if baseline.empty:
            continue
        base = baseline.iloc[0]
        current = frame["method"] == method
        frame.loc[current, "p1_nonregress"] = frame.loc[current, "P1_capture"] >= base["P1_capture"]
        frame.loc[current, "p2_nonregress"] = frame.loc[current, "P2_noise_pct"] <= base["P2_noise_pct"]
        frame.loc[current, "p3_nonregress"] = frame.loc[current, "P3_completeness"] >= base["P3_completeness"]
        frame.loc[current, "p4_nonregress"] = frame.loc[current, "P4_homogeneity"] >= base["P4_homogeneity"]
        frame.loc[current, "ari_nonregress"] = frame.loc[current, "ARI"] >= base["ARI"]
        frame.loc[current, "sil_nonregress"] = frame.loc[current, "Sil"] >= base["Sil"]
        # Silhouette and fragmentation are mandatory diagnostics, but they are
        # not hard gates: a useful class separation can lower silhouette when
        # cluster geometry becomes less spherical.  P1/P2/P3/P4/ARI define
        # the representation acceptance gate.
        gate_cols = ["p1_nonregress", "p2_nonregress", "p3_nonregress", "p4_nonregress", "ari_nonregress"]
        frame.loc[current, "core_gate"] = frame.loc[current, gate_cols].all(axis=1)
    return frame


def write_summary(frame: pd.DataFrame, output: Path) -> None:
    primary = frame[frame["method"] == "finch_p2"].copy()
    primary["p1_label"] = primary.apply(
        lambda row: f"{int(row.P1_capture_count)}/{int(row.P1_target_class_count)} ({row.P1_capture:.3f})", axis=1
    )
    best = primary.loc[primary.groupby("recipe")["ARI"].idxmax()].sort_values(["recipe", "epoch"])
    lines = [
        "# Hard-Unknown Strict-Novel Canonical Rescore",
        "",
        "Protocol: cluster the complete 42-class pool, then score the 32 defect classes excluded from defect-aware adaptation.",
        "P1 is unique dominant/main-class capture. Normal and train-known classes remain in clustering, so a minority defect inside a Normal-dominant group is not falsely captured.",
        "",
        "| Recipe | Diagnostic best epoch by FINCH-p2 ARI | P1 capture | P2 noise% | P3 Comp | P4 Hom | ARI | Sil | k | Fragment | Gate vs frozen |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in best.itertuples(index=False):
        gate = "pass" if bool(getattr(row, "core_gate", False)) else "fail"
        lines.append(
            f"| {display_recipe(row.recipe)} | {row.epoch} | {row.p1_label} | {row.P2_noise_pct:.1f} | "
            f"{row.P3_completeness:.3f} | {row.P4_homogeneity:.3f} | {row.ARI:.3f} | {row.Sil:.3f} | "
            f"{int(row.k_total)} | {row.fragment_ratio:.2f} | {gate} |"
        )
    lines.extend(
        [
            "",
            "`best epoch` is diagnostic only. A recipe is not accepted unless both FINCH-p2 and Louvain preserve or improve frozen P1/P2/P3/P4/ARI. Silhouette, k, and fragmentation remain required diagnostics rather than hard gates.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def plot_method(frame: pd.DataFrame, method: str, output: Path) -> None:
    current = frame[frame["method"] == method].copy()
    baseline = current[current["recipe"] == "frozen"].iloc[0]
    recipes = [recipe for recipe in current["recipe"].unique() if recipe != "frozen"]
    recipes.sort(key=lambda recipe: -1.0 if recipe == "unkda_base" else float(recipe.rsplit("nv", 1)[1]) / 100)
    metrics = [
        ("P1_capture", "P1 dominant capture", (0.0, 1.05)),
        ("P2_noise_pct", "P2 noise %", (0.0, 100.0)),
        ("P3_completeness", "P3 completeness", (0.0, 1.05)),
        ("P4_homogeneity", "P4 homogeneity", (0.0, 1.05)),
        ("ARI", "ARI", (0.0, 1.05)),
        ("Sil", "Silhouette", (-0.1, 1.0)),
        ("k_total", "k (all non-noise clusters)", (0.0, max(80.0, current["k_total"].max() * 1.1))),
        ("fragment_ratio", "Fragment ratio", (0.0, max(2.5, current["fragment_ratio"].max() * 1.1))),
    ]
    colors = plt.get_cmap("tab10")(np.linspace(0, 1, max(1, len(recipes))))
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.subplots_adjust(left=0.045, right=0.995, bottom=0.16, top=0.86, hspace=0.34, wspace=0.11)
    for axis, (metric, title, y_lim) in zip(axes.ravel(), metrics, strict=True):
        for recipe, color in zip(recipes, colors, strict=True):
            subset = current[current["recipe"] == recipe].sort_values("epoch")
            axis.plot(subset["epoch"], subset[metric], marker="o", linewidth=1.7, markersize=3.5, color=color, label=display_recipe(recipe))
        axis.axhline(baseline[metric], color="#b42318", linestyle="--", linewidth=1.2, label="Frozen")
        axis.set(title=title, xlabel="epoch", ylim=y_lim)
        axis.set_xticks(range(1, 11))
        axis.grid(alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=min(5, len(handles)),
        frameon=False,
    )
    fig.suptitle(
        f"Hard-unknown strict-novel rescore: {method}\n"
        "full-pool clustering; 32 unseen defects; P1 = unique dominant-class capture",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "unknown_strict_novel_canonical_scores.csv"
    scorelib = load_score_module()
    labels = scorelib.labels_pool(args.pool.resolve())
    excluded = {value.strip() for value in EXCLUDED.split(",") if value.strip()}
    specs = discover(args.embeddings.resolve(), args.frozen.resolve())
    if not specs:
        raise RuntimeError("No hard-unknown embeddings discovered")
    cache = pd.DataFrame() if args.refresh else load_cache(cache_path)
    for spec in specs:
        required = set(METHODS)
        if not args.refresh and required <= cached_methods(cache, spec):
            continue
        print(f"[score] {Path(spec['embedding']).name}", flush=True)
        cache = cache[cache["embedding"] != str(spec["embedding"])] if not cache.empty else cache
        cache = pd.concat([cache, pd.DataFrame(score_spec(scorelib, labels, excluded, spec))], ignore_index=True)
        save_cache(cache, cache_path)
    scored = add_gates(cache)
    save_cache(scored, cache_path)
    summary_path = output_dir / "unknown_strict_novel_canonical_summary.md"
    write_summary(scored, summary_path)
    for method in METHODS:
        plot_method(scored, method, output_dir / f"unknown_strict_novel_{method}.png")
    print(f"[scores] {cache_path}")
    print(f"[summary] {summary_path}")
    for method in METHODS:
        print(f"[plot] {output_dir / f'unknown_strict_novel_{method}.png'}")


if __name__ == "__main__":
    main()
