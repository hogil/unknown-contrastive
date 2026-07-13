#!/usr/bin/env python3
"""Build one canonical status table for cross-dataset grouping quality.

This is a reporting gate, not a tuner.  It keeps the accepted common recipe
separate from hard-unknown candidates that have not yet passed both clusterer
views.
"""
from __future__ import annotations

import argparse
import re
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


def unknown_recipe_label(recipe: str) -> str:
    if recipe == "frozen":
        return "DINOv3 frozen"
    if recipe == "fcmae_frozen":
        return "FCMAE frozen"
    if recipe.startswith("unkda_base"):
        suffix = recipe.removeprefix("unkda_base").strip("_").replace("_", " ")
        return "SimCLR base" if not suffix else f"SimCLR base {suffix}"
    match = re.match(r"^unkda_nv(\d{3})(?:_(.*))?$", recipe, re.IGNORECASE)
    if match is None:
        if recipe.startswith("unkda_"):
            return recipe.removeprefix("unkda_").replace("_", " ")
        return recipe
    suffix = (match.group(2) or "").replace("_", " ")
    label = f"NV {int(match.group(1)) / 100:.2f}"
    return label if not suffix else f"{label} {suffix}"


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
    learned_finch = unknown[
        (~unknown["recipe"].isin(["frozen", "fcmae_frozen"]))
        & (unknown["method"] == "finch_p2")
    ]
    unknown_louvain = unknown[unknown["method"] == "louvain_res6"]
    unknown_frozen_lv = unknown_louvain[unknown_louvain["recipe"] == "frozen"].iloc[0]
    fcmae_finch_rows = unknown[(unknown["recipe"] == "fcmae_frozen") & (unknown["method"] == "finch_p2")]
    fcmae_louvain_rows = unknown[(unknown["recipe"] == "fcmae_frozen") & (unknown["method"] == "louvain_res6")]
    fcmae_finch = fcmae_finch_rows.iloc[0] if not fcmae_finch_rows.empty else None
    fcmae_louvain = fcmae_louvain_rows.iloc[0] if not fcmae_louvain_rows.empty else None
    fcmae_core_finch = fcmae_finch is not None and core_gate(fcmae_finch, frozen)
    fcmae_core_louvain = fcmae_louvain is not None and core_gate(fcmae_louvain, unknown_frozen_lv)
    fcmae_dual_gate = bool(fcmae_core_finch and fcmae_core_louvain)
    dual_gate_rows: list[tuple[pd.Series, pd.Series]] = []
    for _, finch_row in learned_finch.iterrows():
        matching_louvain = unknown_louvain[
            (unknown_louvain["recipe"] == finch_row.recipe)
            & (unknown_louvain["epoch"] == finch_row.epoch)
        ]
        if matching_louvain.empty:
            continue
        louvain_row = matching_louvain.iloc[0]
        if core_gate(finch_row, frozen) and core_gate(louvain_row, unknown_frozen_lv):
            dual_gate_rows.append((finch_row, louvain_row))
    if dual_gate_rows:
        exploratory, exploratory_lv = max(dual_gate_rows, key=lambda pair: float(pair[0].ARI))
    elif not learned_finch.empty:
        exploratory = learned_finch.loc[learned_finch["ARI"].idxmax()]
        exploratory_lv = unknown_louvain[
            (unknown_louvain["recipe"] == exploratory.recipe)
            & (unknown_louvain["epoch"] == exploratory.epoch)
        ].iloc[0]
    else:
        exploratory = None
        exploratory_lv = None
    unknown_core_finch = exploratory is not None and core_gate(exploratory, frozen)
    unknown_core_louvain = exploratory_lv is not None and core_gate(exploratory_lv, unknown_frozen_lv)
    unknown_dual_gate = bool(unknown_core_finch and unknown_core_louvain)
    unknown_label = (
        f"{unknown_recipe_label(str(exploratory.recipe))} ep{int(exploratory.epoch)}"
        if exploratory is not None
        else None
    )

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
            (
                "FCMAE frozen ep0 passes both clusterer gates against DINOv3 frozen and needs only the image-disjoint holdout check."
                if fcmae_dual_gate
                else "No alternate frozen backbone currently passes both hard-unknown clusterer gates."
            ),
            (
                f"{unknown_label} is the strongest learned dual-gate candidate and remains provisional until fixed-seed and holdout validation."
                if unknown_dual_gate
                else "No learned hard-unknown candidate currently passes both clusterer gates."
            ),
            "",
            "| Method | Row | Recipe / epoch | P1 capture | P2 noise% | P3 Comp | P4 Hom | ARI | Sil | k | Fragment | Core gate vs frozen |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    hard_rows = [
        ("FINCH-p2", frozen, "reference", unknown_recipe_label("frozen"), True),
        ("Louvain", unknown_frozen_lv, "reference", unknown_recipe_label("frozen"), True),
    ]
    if fcmae_finch is not None and fcmae_louvain is not None:
        hard_rows.extend(
            [
                ("FINCH-p2", fcmae_finch, "backbone candidate", "FCMAE frozen ep0", fcmae_core_finch),
                ("Louvain", fcmae_louvain, "backbone candidate", "FCMAE frozen ep0", fcmae_core_louvain),
            ]
        )
    if exploratory is not None and exploratory_lv is not None and unknown_label is not None:
        hard_rows.extend(
            [
                ("FINCH-p2", exploratory, "learned candidate", unknown_label, unknown_core_finch),
                ("Louvain", exploratory_lv, "learned candidate", unknown_label, unknown_core_louvain),
            ]
        )
    for method, row, role, label, gate in hard_rows:
        lines.append(
            f"| {method} | {role} | {label} | {p1_text(row)} | "
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
            (
                "- Hard unknown: FCMAE frozen ep0 is the current provisional backbone candidate; retain DINOv3 frozen as the fallback until the image-disjoint holdout check completes."
                if fcmae_dual_gate
                else "- Hard unknown: deploy DINOv3 frozen grouping until an alternate backbone passes both clusterer gates."
            ),
            (
                f"- Learned hard-unknown candidate: {unknown_label}; retain it separately until fixed-seed and holdout validation complete."
                if unknown_dual_gate
                else "- Learned hard-unknown candidate: none accepted yet under both clusterers."
            ),
            "- Therefore this is a robust model family with an acceptance/fallback policy, not yet one universal learned checkpoint.",
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
