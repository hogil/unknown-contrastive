#!/usr/bin/env python3
"""Build one canonical status table for cross-dataset grouping quality.

This is a reporting gate, not a tuner.  It keeps the accepted common recipe
separate from hard-unknown candidates that have not yet passed both clusterer
views.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "docs" / "paper" / "canonical_rescore_260713"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def p1_text(row: pd.Series) -> str:
    return f"{int(row.P1_capture_count)}/{int(row.P1_target_class_count)} ({float(row.P1_capture):.3f})"


def core_gate(candidate: pd.Series, frozen: pd.Series) -> bool:
    return bool(
        float(candidate.P1_capture) >= float(frozen.P1_capture)
        and float(candidate.P2_noise_pct) <= float(frozen.P2_noise_pct)
        and float(candidate.P3_completeness) >= float(frozen.P3_completeness)
        and float(candidate.P4_homogeneity) >= float(frozen.P4_homogeneity)
        and float(candidate.ARI) >= float(frozen.ARI)
    )


def main() -> None:
    args = parse_args()
    root = args.input_dir.resolve()
    cross_path = root / "cross_dataset_canonical_scores.csv"
    unknown_path = root / "unknown_strict_novel" / "unknown_strict_novel_canonical_scores.csv"
    if not cross_path.exists() or not unknown_path.exists():
        raise SystemExit(f"Canonical score inputs are missing under {root}")

    cross = pd.read_csv(cross_path)
    cross_primary = cross[
        cross.apply(lambda row: str(row["method"]).startswith(str(row["primary_prefix"])), axis=1)
    ].copy()
    rows: list[dict[str, object]] = []
    for dataset, group in cross_primary.groupby("dataset", sort=True):
        frozen = group[group["label"] == "frozen"].iloc[0]
        candidate = group[group["label"] == "candidate"].iloc[0]
        rows.extend(
            [
                {"dataset": dataset, "role": "frozen", "recipe": frozen.recipe, **frozen.to_dict(), "core_gate": True},
                {
                    "dataset": dataset,
                    "role": "candidate",
                    "recipe": candidate.recipe,
                    **candidate.to_dict(),
                    "core_gate": core_gate(candidate, frozen),
                },
            ]
        )
    combined = pd.DataFrame(rows)
    output_csv = root / "robust_grouping_status.csv"
    combined.to_csv(output_csv, index=False)

    unknown = pd.read_csv(unknown_path)
    frozen = unknown[(unknown["recipe"] == "frozen") & (unknown["method"] == "finch_p2")].iloc[0]
    nv50 = unknown[(unknown["recipe"] == "unkda_nv050") & (unknown["method"] == "finch_p2")]
    exploratory = nv50.loc[nv50["ARI"].idxmax()]
    unknown_louvain = unknown[unknown["method"] == "louvain_res6"]
    unknown_frozen_lv = unknown_louvain[unknown_louvain["recipe"] == "frozen"].iloc[0]
    nv50_lv = unknown_louvain[unknown_louvain["recipe"] == "unkda_nv050"]
    exploratory_lv = nv50_lv.loc[nv50_lv["ARI"].idxmax()]
    unknown_core_finch = core_gate(exploratory, frozen)
    unknown_core_louvain = core_gate(exploratory_lv, unknown_frozen_lv)

    lines = [
        "# Robust Grouping Model Status",
        "",
        "All P1 values below use the same contract: full-pool clustering and unique dominant/main-class capture. P2/P3/P4/ARI are scored only on the protocol target classes.",
        "",
        "## Cross-Dataset Acceptance",
        "",
        "Core gate: P1 does not regress, P2 does not increase, and P3/P4/ARI do not regress. Silhouette, k, and fragment ratio are mandatory diagnostics, not automatic reject rules.",
        "",
        "| Dataset | Role | Recipe | P1 capture | P2 noise% | P3 Comp | P4 Hom | ARI | Sil | k | Fragment | Core gate |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in combined.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['role']} | {row['recipe']} | {p1_text(row)} | "
            f"{float(row['P2_noise_pct']):.1f} | {float(row['P3_completeness']):.3f} | {float(row['P4_homogeneity']):.3f} | "
            f"{float(row['ARI']):.3f} | {float(row['Sil']):.3f} | {int(row['k_total'])} | {float(row['fragment_ratio']):.2f} | "
            f"{'pass' if bool(row['core_gate']) else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "## Hard-Unknown Strict-Novel",
            "",
            "The full hard-unknown 32-class strict-novel gate has not accepted any learned candidate yet. The strongest retained trade-off is listed for diagnosis, not deployment.",
            "",
            "| Method | Row | Recipe / epoch | P1 capture | P2 noise% | P3 Comp | P4 Hom | ARI | Sil | k | Fragment | Core gate vs frozen |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for method, row, label, gate in (
        ("FINCH-p2", frozen, "DINOv3 frozen", True),
        ("FINCH-p2", exploratory, f"NV 0.50 ep{int(exploratory.epoch)}", unknown_core_finch),
        ("Louvain", unknown_frozen_lv, "DINOv3 frozen", True),
        ("Louvain", exploratory_lv, f"NV 0.50 ep{int(exploratory_lv.epoch)}", unknown_core_louvain),
    ):
        lines.append(
            f"| {method} | {'reference' if 'frozen' in label else 'exploratory'} | {label} | {p1_text(row)} | "
            f"{float(row.P2_noise_pct):.1f} | {float(row.P3_completeness):.3f} | {float(row.P4_homogeneity):.3f} | "
            f"{float(row.ARI):.3f} | {float(row.Sil):.3f} | {int(row.k_total)} | {float(row.fragment_ratio):.2f} | "
            f"{'pass' if gate else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "## Current Deployment Decision",
            "",
            "- WM-811K, RESISC45, and DTD: use the accepted learned candidate shown above; retain the frozen embedding as fallback.",
            "- Hard unknown: deploy DINOv3 frozen grouping until a learned candidate passes the core gate under both FINCH-p2 and Louvain.",
            "- Therefore this is a robust model family with a validated acceptance/fallback policy, not yet one universal learned checkpoint.",
            "",
        ]
    )
    projection_path = root / "unknown_strict_novel" / "projection_probe_scores.csv"
    if projection_path.exists():
        probe = pd.read_csv(projection_path)
        probe = probe[probe["method"].str.startswith("finch_p2", na=False)].copy()
        unknown_by_name = unknown.copy()
        unknown_by_name["embedding_name"] = unknown_by_name["embedding"].map(lambda value: Path(value).stem)
        lines.extend(
            [
                "",
                "## Embedding-Mode Audit",
                "",
                "FINCH-p2 rows below compare the trained projection `z` with the matching backbone `f`. The current hard-unknown deployment path remains `f` unless an adapter trial passes the same core gate.",
                "",
                "| Recipe / epoch | Space | P1 capture | P2 noise% | P3 Comp | P4 Hom | ARI | Sil | k | Fragment |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, z_row in probe.sort_values("embedding_name").iterrows():
            f_name = str(z_row["embedding_name"]).removesuffix("_proj")
            f_rows = unknown_by_name[
                (unknown_by_name["embedding_name"] == f_name)
                & (unknown_by_name["method"] == "finch_p2")
            ]
            if f_rows.empty:
                continue
            f_row = f_rows.iloc[0]
            label = f_name.replace("unkda_", "").replace("_", " ")
            for space, row in (("backbone f", f_row), ("projection z", z_row)):
                lines.append(
                    f"| {label} | {space} | {p1_text(row)} | {float(row.P2_noise_pct):.1f} | "
                    f"{float(row.P3_completeness):.3f} | {float(row.P4_homogeneity):.3f} | {float(row.ARI):.3f} | "
                    f"{float(row.Sil):.3f} | {int(row.k_total)} | {float(row.fragment_ratio):.2f} |"
                )
    output_md = root / "robust_grouping_status.md"
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OUT] {output_csv}")
    print(f"[OUT] {output_md}")


if __name__ == "__main__":
    main()
