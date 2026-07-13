#!/usr/bin/env python3
"""Plot the fixed hard-unknown NV0.50 frozen/trained blend sweep."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "docs" / "paper" / "canonical_rescore_260713" / "unknown_strict_novel" / "unknown_strict_novel_canonical_scores.csv"
OUTPUT = ROOT / "docs" / "paper" / "canonical_rescore_260713" / "unknown_strict_novel"
BLEND_RE = re.compile(r"^unkda_nv050_blend(\d{3})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=SCORES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    return parser.parse_args()


def blend_weight(recipe: str) -> float | None:
    match = BLEND_RE.fullmatch(recipe)
    return int(match.group(1)) / 100.0 if match else None


def rows_for_method(frame: pd.DataFrame, method: str) -> pd.DataFrame:
    current = frame[frame["method"] == method].copy()
    frozen = current[current["recipe"] == "frozen"].copy()
    trained = current[(current["recipe"] == "unkda_nv050") & (current["epoch"] == 6)].copy()
    blend = current[current["recipe"].str.match(BLEND_RE, na=False)].copy()
    blend["trained_weight"] = blend["recipe"].map(blend_weight)
    frozen["trained_weight"] = 0.0
    trained["trained_weight"] = 1.0
    return pd.concat([frozen, blend, trained], ignore_index=True).sort_values("trained_weight")


def plot(frame: pd.DataFrame, method: str, output: Path) -> None:
    current = rows_for_method(frame, method)
    baseline = current[current["trained_weight"] == 0.0].iloc[0]
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
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.subplots_adjust(left=0.05, right=0.995, bottom=0.12, top=0.87, hspace=0.35, wspace=0.13)
    for axis, (metric, title, y_lim) in zip(axes.ravel(), metrics, strict=True):
        axis.plot(current["trained_weight"], current[metric], marker="o", color="#007f6d", linewidth=1.8, markersize=4)
        axis.axhline(baseline[metric], color="#b42318", linestyle="--", linewidth=1.2, label="frozen reference")
        axis.axvline(0.86, color="#2e5aac", linestyle=":", linewidth=1.2, label="selected 0.86")
        axis.set(title=title, xlabel="NV0.50 trained embedding weight", ylim=y_lim)
        axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        axis.grid(alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle(
        f"Hard-unknown strict-novel: frozen + NV0.50 ep6 backbone blend ({method})\n"
        "full-pool clustering; P1 = unique dominant-class capture",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.scores.resolve())
    output_dir = args.output_dir.resolve()
    for method in ("finch_p2", "louvain_res6"):
        output = output_dir / f"hard_unknown_nv050_blend_sweep_{method}.png"
        plot(frame, method, output)
        print(f"[OUT] {output}")


if __name__ == "__main__":
    main()
