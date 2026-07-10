#!/usr/bin/env python3
"""Score available defect-aware NV embeddings and plot sweep progress."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDINGS = REPO / "result_grouping" / "_unknown_mixed260710" / "embeddings"
DEFAULT_CACHE = REPO / "docs" / "paper" / "UNKNOWN_NV_SWEEP_SCORES_260710.csv"
DEFAULT_SUMMARY = REPO / "docs" / "paper" / "UNKNOWN_NV_SWEEP_SUMMARY_260710.csv"
DEFAULT_PLOT = REPO / "docs" / "paper" / "figs_unknown_nv_sweep_260710.png"
EVAL_POOL = REPO / "data" / "images" / "unknown_eval100"
EXCLUDED = (
    "Normal,Random,R,Center_bank_boundary,Center_scratch,Donut_bank_boundary,"
    "Donut_fork,Edge-Ring_bank_boundary,Edge-Ring_scratch,Edge-Top_fork,"
    "Full_scratch,ParallelScratches,RingDots"
)
FROZEN = {
    "finch_p2": {"ARI": 0.7090, "P1_capture": 0.9688},
    "louvain_res6": {"ARI": 0.7850, "P1_capture": 0.9688},
}
NAME_RE = re.compile(r"^(unkda_(?:base|nv(?P<nv>\d{3})))_ep(?P<epoch>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_PLOT)
    parser.add_argument(
        "--no-score",
        action="store_true",
        help="plot cached rows only; do not score newly discovered embeddings",
    )
    return parser.parse_args()


def discover_embeddings(root: Path) -> list[tuple[Path, str, int, float | None]]:
    found = []
    for path in sorted(root.glob("unkda_*_ep*.npy")):
        if path.stem.endswith("_proj"):
            continue
        match = NAME_RE.fullmatch(path.stem)
        if not match:
            continue
        recipe = match.group(1)
        epoch = int(match.group("epoch"))
        threshold = int(match.group("nv")) / 100 if match.group("nv") else None
        found.append((path.resolve(), recipe, epoch, threshold))
    return found


def read_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "embedding" in frame:
        frame["embedding"] = frame["embedding"].map(lambda value: str(Path(value).resolve()))
    return frame


def score_embedding(path: Path) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
        temp_csv = Path(handle.name)
    try:
        command = [
            sys.executable,
            str(REPO / "_score_umapfree.py"),
            str(path),
            "--skip-umap",
            "--pool",
            str(EVAL_POOL),
            "--exclude-classes",
            EXCLUDED,
            "--out-csv",
            str(temp_csv),
        ]
        subprocess.run(command, cwd=REPO, check=True)
        return pd.read_csv(temp_csv)
    finally:
        temp_csv.unlink(missing_ok=True)


def update_cache(
    cache: pd.DataFrame,
    embeddings: list[tuple[Path, str, int, float | None]],
    score_new: bool,
) -> pd.DataFrame:
    frames = [] if cache.empty else [cache]
    cached = set(cache.get("embedding", pd.Series(dtype=str)).astype(str))
    for path, recipe, epoch, threshold in embeddings:
        key = str(path)
        if key in cached:
            continue
        if not score_new:
            continue
        print(f"[score] {path.name}", flush=True)
        rows = score_embedding(path)
        rows["recipe"] = recipe
        rows["epoch"] = epoch
        rows["nv_threshold"] = threshold
        rows["embedding_space"] = "backbone"
        frames.append(rows)
    if not frames:
        raise RuntimeError("No cached or discoverable score rows were found")
    merged = pd.concat(frames, ignore_index=True)

    metadata = {str(path): (recipe, epoch, threshold) for path, recipe, epoch, threshold in embeddings}
    for index, row in merged.iterrows():
        key = str(Path(row["embedding"]).resolve())
        if key not in metadata:
            continue
        recipe, epoch, threshold = metadata[key]
        merged.at[index, "embedding"] = key
        merged.at[index, "recipe"] = recipe
        merged.at[index, "epoch"] = epoch
        merged.at[index, "nv_threshold"] = threshold
        merged.at[index, "embedding_space"] = "backbone"

    merged = merged.drop_duplicates(subset=["embedding", "method"], keep="last")
    merged["method_family"] = merged["method"].map(
        lambda value: "finch_p2" if str(value).startswith("finch_p2(") else str(value)
    )
    return merged.sort_values(["recipe", "epoch", "method"]).reset_index(drop=True)


def display_label(recipe: str) -> str:
    if recipe == "unkda_base":
        return "SimCLR base"
    threshold = int(recipe.rsplit("nv", 1)[1]) / 100
    return f"NV {threshold:.2f}"


def recipe_order(frame: pd.DataFrame) -> list[str]:
    recipes = frame["recipe"].dropna().unique().tolist()
    return sorted(
        recipes,
        key=lambda recipe: -1.0 if recipe == "unkda_base" else int(recipe.rsplit("nv", 1)[1]) / 100,
    )


def build_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    selected = frame[frame["method_family"].isin(FROZEN)].copy()
    for (recipe, method), group in selected.groupby(["recipe", "method_family"]):
        group = group.sort_values("epoch")
        best = group.loc[group["ARI"].idxmax()]
        latest = group.iloc[-1]
        rows.append(
            {
                "recipe": recipe,
                "label": display_label(recipe),
                "nv_threshold": "off" if recipe == "unkda_base" else float(group["nv_threshold"].iloc[0]),
                "method": method,
                "epochs_available": ",".join(str(int(value)) for value in group["epoch"]),
                "best_epoch": int(best["epoch"]),
                "best_ARI": best["ARI"],
                "best_capture": best["P1_capture"],
                "best_recov": best["recov"],
                "best_completeness": best["P3_completeness"],
                "best_homogeneity": best["P4_homogeneity"],
                "best_silhouette": best["Sil"],
                "best_fragment_ratio": best["fragment_ratio"],
                "latest_epoch": int(latest["epoch"]),
                "latest_ARI": latest["ARI"],
                "latest_capture": latest["P1_capture"],
                "frozen_ARI": FROZEN[method]["ARI"],
            }
        )
    return pd.DataFrame(rows).sort_values(["method", "recipe"]).reset_index(drop=True)


def plot(frame: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    methods = ("finch_p2", "louvain_res6")
    recipes = recipe_order(frame)
    palette = plt.get_cmap("turbo")(np.linspace(0.08, 0.92, max(len(recipes), 2)))
    colors = {recipe: ("#202020" if recipe == "unkda_base" else palette[i]) for i, recipe in enumerate(recipes)}

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.12, top=0.86, hspace=0.42, wspace=0.18)

    for ax, method, title in zip(
        axes[0], methods, ("FINCH-p2 ARI trajectory", "Louvain ARI trajectory"), strict=True
    ):
        method_rows = frame[frame["method_family"] == method]
        for recipe in recipes:
            group = method_rows[method_rows["recipe"] == recipe].sort_values("epoch")
            if group.empty:
                continue
            ax.plot(
                group["epoch"],
                group["ARI"],
                marker="o",
                markersize=4,
                linewidth=1.8,
                color=colors[recipe],
                label=display_label(recipe),
            )
        ax.axhline(
            FROZEN[method]["ARI"],
            color="#c23b22",
            linewidth=1.5,
            linestyle="--",
            label=f"Frozen {FROZEN[method]['ARI']:.3f}",
        )
        ax.set(title=title, xlabel="Epoch", ylabel="ARI", ylim=(0.25, 0.85))
        ax.set_xticks(range(1, 11))
        ax.grid(alpha=0.22)
        ax.legend(fontsize=8, ncol=2)

    finch = summary[summary["method"] == "finch_p2"].copy()
    ordered = [recipe for recipe in recipes if recipe in set(finch["recipe"])]
    positions = np.arange(len(ordered))
    labels = [display_label(recipe).replace("SimCLR ", "") for recipe in ordered]
    selected = finch.set_index("recipe").loc[ordered]

    ax = axes[1, 0]
    ax.plot(positions, selected["best_capture"], marker="o", label="P1 capture")
    ax.plot(positions, selected["best_recov"], marker="s", label="Rediscovery")
    ax.plot(positions, selected["best_completeness"], marker="^", label="P3 completeness")
    ax.plot(positions, selected["best_homogeneity"], marker="D", label="P4 homogeneity")
    ax.axhline(FROZEN["finch_p2"]["P1_capture"], color="#c23b22", linestyle="--", linewidth=1.2)
    ax.set(
        title="FINCH-p2 metrics at each recipe's best-observed ARI epoch",
        xlabel="Recipe / NV threshold",
        ylabel="Score",
        ylim=(0.5, 1.02),
    )
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 1]
    for method, marker, title in (
        ("finch_p2", "o", "FINCH-p2"),
        ("louvain_res6", "s", "Louvain"),
    ):
        method_summary = summary[summary["method"] == method].set_index("recipe").loc[ordered]
        ax.plot(positions, method_summary["best_ARI"], marker=marker, linewidth=1.8, label=f"{title} best observed")
        ax.plot(
            positions,
            method_summary["latest_ARI"],
            marker=marker,
            linewidth=1.2,
            linestyle=":",
            alpha=0.8,
            label=f"{title} latest available",
        )
        ax.axhline(FROZEN[method]["ARI"], color="#c23b22", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set(
        title="Threshold summary: best observed vs latest",
        xlabel="Recipe / NV threshold",
        ylabel="ARI",
        ylim=(0.25, 0.85),
    )
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8)

    fig.suptitle(
        "Unknown hard-42 defect-aware strict-novel NV sweep\n"
        "32 unseen classes | DINOv3 backbone embedding | grade_only | single seed (exploratory)",
        fontsize=15,
        y=0.965,
    )
    fig.text(
        0.5,
        0.025,
        "Dashed red lines are frozen baselines. Best-observed points are not acceptance results; latest epoch is shown separately.",
        ha="center",
        fontsize=9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    embeddings = discover_embeddings(args.embeddings.resolve())
    if not embeddings:
        raise RuntimeError(f"No matching backbone embeddings under {args.embeddings.resolve()}")
    cache = update_cache(read_cache(args.cache.resolve()), embeddings, not args.no_score)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    cache.to_csv(args.cache, index=False)
    summary = build_summary(cache)
    summary.to_csv(args.summary, index=False)
    plot(cache, summary, args.output)
    print(f"[scores] {args.cache.resolve()}")
    print(f"[summary] {args.summary.resolve()}")
    print(f"[plot] {args.output.resolve()}")


if __name__ == "__main__":
    main()
