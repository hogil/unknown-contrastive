#!/usr/bin/env python3
"""Re-score and validate the hard-42 FCMAE adapter protocol.

The protocol is deliberately narrow:

* strict-novel evaluation on ``unknown_eval100``;
* canonical dominant-class P1 capture;
* FINCH partition 2 and Louvain resolution 6;
* adapted backbone feature ``f`` only;
* fixed epoch 4 for the seed 1/3/5 confirmation.

ARI and AMI are retained as supporting diagnostics. They never decide the
acceptance gate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import silhouette_score


for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import _score_umapfree as fixed_scorer  # noqa: E402
from _common import image_dataset_path, resolve_pool  # noqa: E402
from cluster_metrics import capture_metrics  # noqa: E402


TRAIN_DIR = REPO / "data/pools/unknown_train_defectaware_260710.json"
TRAIN_MANIFEST = TRAIN_DIR
EVAL_DIR = REPO / "data/pools/unknown_eval100.json"
FROZEN_EMBEDDING = REPO / "result_grouping/frozen_fcmae.npy"
ADAPTER_EMBEDDINGS = REPO / "runs/fcmae_adapter_temperature_screen_260725/embeddings"
LADDER_EMBEDDINGS = REPO / "runs/fcmae_adapter_ep4_three_seed_260721/embeddings"
SEED_RUN_ROOT = REPO / "runs/fcmae_adapter_ep4_three_seed_260721"
SEED_EMBEDDINGS = SEED_RUN_ROOT / "embeddings"
SEED_LOG = SEED_RUN_ROOT / "train.log"
SEED3_GENERATED_EMBEDDING = SEED_EMBEDDINGS / "fcmae_ad1_s3_ep4.npy"
SEED3_LEGACY_EMBEDDING = SEED3_GENERATED_EMBEDDING

EXCLUDED_CLASSES = (
    "Normal",
    "Random",
    "R",
    "Center_bank_boundary",
    "Center_scratch",
    "Donut_bank_boundary",
    "Donut_fork",
    "Edge-Ring_bank_boundary",
    "Edge-Ring_scratch",
    "Edge-Top_fork",
    "Full_scratch",
    "ParallelScratches",
    "RingDots",
)

EXISTING_CSV = REPO / "docs/paper/FCMAE_FIXED_RESCORE_260721.csv"
EXISTING_MD = REPO / "docs/paper/FCMAE_FIXED_RESCORE_260721.md"
PROTOCOL_JSON = REPO / "docs/paper/FCMAE_FIXED_PROTOCOL_260721.json"
EVAL_MANIFEST = REPO / "docs/paper/FCMAE_FIXED_PROTOCOL_260721_eval_manifest.json"
SEED_CSV = REPO / "docs/paper/FCMAE_L0_EP4_THREESEED_260721.csv"
SEED_MD = REPO / "docs/paper/FCMAE_L0_EP4_THREESEED_260721.md"
SEED_GATE_JSON = REPO / "docs/paper/FCMAE_L0_EP4_THREESEED_260721_gate.json"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SCORER_FILES = (
    REPO / "_score_umapfree.py",
    REPO / "scripts/cluster_scoring.py",
    REPO / "scripts/cluster_metrics.py",
    Path(__file__).resolve(),
)

CSV_FIELDS = (
    "protocol_id",
    "recipe",
    "recipe_flags",
    "seed",
    "epoch",
    "embedding_space",
    "clusterer",
    "clusterer_detail",
    "P1_capture_count",
    "P1_target_class_count",
    "P1_capture",
    "dominant_cluster_count",
    "recov",
    "P2_noise_pct",
    "P3_completeness",
    "P4_homogeneity",
    "k_total",
    "k_target_classes",
    "k_excluded_dominant",
    "fragment_ratio",
    "Sil",
    "ARI_supporting",
    "AMI_supporting",
    "embedding_path",
    "embedding_sha256",
    "embedding_rows",
    "embedding_dim",
    "eval_dir",
    "eval_manifest_sha256",
    "train_dir",
    "train_manifest_sha256",
    "scorer_bundle_sha256",
    "exclude_classes",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def image_paths(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() == ".json":
        paths, _ = resolve_pool(root, extensions=tuple(IMAGE_EXTENSIONS))
        return [Path(path) for path in paths]
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_eval_manifest() -> tuple[dict[str, Any], str]:
    paths = image_paths(EVAL_DIR)
    canonical_root = image_dataset_path("unknown")
    entries = []
    for index, path in enumerate(paths):
        entries.append(
            {
                "index": index,
                "relative_path": path.relative_to(canonical_root).as_posix(),
                "label": path.parent.name,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "root": str(canonical_root.resolve()),
        "ordering": "sorted recursive image paths",
        "image_count": len(entries),
        "class_counts": dict(sorted(Counter(item["label"] for item in entries).items())),
        "entries": entries,
    }
    digest = sha256_json(manifest)
    return manifest, digest


def scorer_provenance() -> tuple[dict[str, str], str]:
    hashes = {str(path.resolve()): sha256_file(path) for path in SCORER_FILES}
    return hashes, sha256_json(hashes)


def package_versions() -> dict[str, str]:
    versions = {}
    for package in ("numpy", "scikit-learn", "networkx", "finch-clust", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() or "unknown"


def protocol_context(expected_target_class_count: int = 32) -> dict[str, Any]:
    if not TRAIN_MANIFEST.is_file():
        raise FileNotFoundError(f"missing train manifest: {TRAIN_MANIFEST.resolve()}")
    eval_manifest, eval_digest = build_eval_manifest()
    EVAL_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    EVAL_MANIFEST.write_text(
        json.dumps(eval_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    scorer_hashes, scorer_bundle = scorer_provenance()
    labels = fixed_scorer.labels_pool(EVAL_DIR)
    present = set(labels)
    target = sorted(present - set(EXCLUDED_CLASSES))
    excluded_present = sorted(present & set(EXCLUDED_CLASSES))
    background_present = sorted(present & {"Normal", "Random", "R"})
    context = {
        "created_at": datetime.now().astimezone().isoformat(),
        "git_revision": git_revision(),
        "train_dir": str(TRAIN_DIR.resolve()),
        "train_manifest": str(TRAIN_MANIFEST.resolve()),
        "train_manifest_sha256": sha256_file(TRAIN_MANIFEST),
        "eval_dir": str(EVAL_DIR.resolve()),
        "eval_manifest": str(EVAL_MANIFEST.resolve()),
        "eval_manifest_sha256": eval_digest,
        "eval_image_count": len(labels),
        "eval_pool_class_count": len(present),
        "target_classes": target,
        "target_class_count": len(target),
        "excluded_classes": list(EXCLUDED_CLASSES),
        "excluded_classes_present": excluded_present,
        "background_classes_present": background_present,
        "embedding_space": "adapted_f (frozen uses backbone_f)",
        "clusterers": {
            "FINCH-p2": "FINCH hierarchy partition index 2",
            "Louvain-res6": "cosine kNN k=15, resolution=6, seed=42, size<=2 to noise",
        },
        "metric_contract": {
            "P1": "unique target labels that are the unique dominant label of >=1 non-noise cluster / target labels",
            "recov": "mean target-image fraction assigned to clusters dominated by its own class",
            "P2": "target-image clusterer noise percentage",
            "P3": "completeness on non-noise target rows",
            "P4": "homogeneity on non-noise target rows",
            "fragment_ratio": "all non-noise clusters / target class count",
            "ARI_AMI": "supporting only; never used by the acceptance gate",
        },
        "scorer_files": scorer_hashes,
        "scorer_bundle_sha256": scorer_bundle,
        "package_versions": package_versions(),
    }
    identity = dict(context)
    identity.pop("created_at")
    context["protocol_id"] = sha256_json(identity)
    PROTOCOL_JSON.write_text(
        json.dumps(context, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if context["target_class_count"] != expected_target_class_count:
        raise RuntimeError(
            "strict-novel protocol expected "
            f"{expected_target_class_count} targets, found {context['target_class_count']}"
        )
    if background_present:
        raise RuntimeError(
            "unknown_eval100 unexpectedly contains field background; FAR must use a separate scope"
        )
    return context


def existing_specs() -> list[dict[str, Any]]:
    specs = [
        {
            "recipe": "frozen",
            "recipe_flags": "FCMAE frozen; no training",
            "seed": "none",
            "epoch": 0,
            "embedding_space": "backbone_f",
            "path": FROZEN_EMBEDDING,
        }
    ]
    recipes = (
        ("L0_base", "adapter; pure SimCLR B0", ADAPTER_EMBEDDINGS, "ad1_none"),
        ("L1_queue", "adapter + queue4096", LADDER_EMBEDDINGS, "L1_q"),
        ("L2_queue_ignore", "adapter + queue4096 + ignore0.75", LADDER_EMBEDDINGS, "L2_qi"),
        (
            "L3_queue_ignore_nv",
            "adapter + queue4096 + ignore0.75 + NV0.75",
            LADDER_EMBEDDINGS,
            "L3_qiN",
        ),
    )
    for recipe, flags, root, tag in recipes:
        for epoch in range(6):
            specs.append(
                {
                    "recipe": recipe,
                    "recipe_flags": flags,
                    "seed": 3,
                    "epoch": epoch,
                    "embedding_space": "adapted_f",
                    "path": root / f"{tag}_ep{epoch}.npy",
                }
            )
    return specs


def seed_specs() -> list[dict[str, Any]]:
    seed3_path, seed3_source = seed3_embedding_source()
    return [
        {
            "recipe": "L0_base_ep4",
            "recipe_flags": "adapter; pure SimCLR B0; fixed epoch4; embedding_source=generated",
            "seed": 1,
            "epoch": 4,
            "embedding_space": "adapted_f",
            "path": SEED_EMBEDDINGS / "fcmae_ad1_s1_ep4.npy",
        },
        {
            "recipe": "L0_base_ep4",
            "recipe_flags": (
                "adapter; pure SimCLR B0; fixed epoch4; "
                f"embedding_source={seed3_source}"
            ),
            "seed": 3,
            "epoch": 4,
            "embedding_space": "adapted_f",
            "path": seed3_path,
        },
        {
            "recipe": "L0_base_ep4",
            "recipe_flags": "adapter; pure SimCLR B0; fixed epoch4; embedding_source=generated",
            "seed": 5,
            "epoch": 4,
            "embedding_space": "adapted_f",
            "path": SEED_EMBEDDINGS / "fcmae_ad1_s5_ep4.npy",
        },
    ]


def seed3_embedding_source() -> tuple[Path, str]:
    """Prefer the fixed-protocol seed-3 output, with a legacy-only fallback."""
    if SEED3_GENERATED_EMBEDDING.is_file():
        return SEED3_GENERATED_EMBEDDING, "generated"
    if SEED3_LEGACY_EMBEDDING.is_file():
        return SEED3_LEGACY_EMBEDDING, "legacy_fallback"
    return SEED3_GENERATED_EMBEDDING, "generated_missing"


def seed3_source_record() -> dict[str, Any]:
    selected_path, selected_source = seed3_embedding_source()
    return {
        "generated_path": str(SEED3_GENERATED_EMBEDDING.resolve()),
        "legacy_fallback_path": str(SEED3_LEGACY_EMBEDDING.resolve()),
        "selected_path": str(selected_path.resolve()),
        "selected_source": selected_source,
        "fallback_used": selected_source == "legacy_fallback",
        "generated_exists": SEED3_GENERATED_EMBEDDING.is_file(),
        "legacy_fallback_exists": SEED3_LEGACY_EMBEDDING.is_file(),
    }


def silhouette(z: np.ndarray, pred: np.ndarray, labels: list[str]) -> float | None:
    labels_array = np.asarray(labels)
    target = ~np.isin(labels_array, list(EXCLUDED_CLASSES))
    pred_target = np.asarray(pred)[target]
    non_noise = pred_target != -1
    if non_noise.sum() <= 10 or len(set(pred_target[non_noise].tolist())) <= 1:
        return None
    try:
        return round(
            float(
                silhouette_score(
                    z[target][non_noise], pred_target[non_noise], metric="cosine"
                )
            ),
            4,
        )
    except ValueError:
        return None


def score_specs(specs: Iterable[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
    labels = fixed_scorer.labels_pool(EVAL_DIR)
    rows: list[dict[str, Any]] = []
    for spec in specs:
        path = Path(spec["path"])
        if not path.is_file():
            raise FileNotFoundError(f"missing embedding: {path.resolve()}")
        print(f"[score] {path.resolve()}", flush=True)
        z = fixed_scorer.FP.l2(np.load(path).astype(np.float32, copy=False))
        if z.shape[0] != len(labels):
            raise RuntimeError(
                f"label/embedding mismatch for {path.resolve()}: {len(labels)} vs {z.shape[0]}"
            )
        finch = fixed_scorer.run_finch(z)
        finch_keys = [key for key in finch if key.startswith("finch_p2(")]
        if len(finch_keys) != 1:
            raise RuntimeError(f"expected one FINCH p2 partition for {path.resolve()}")
        predictions = (
            ("FINCH-p2", finch_keys[0], finch[finch_keys[0]]),
            ("Louvain-res6", "k15/res6/seed42/size<=2-noise", fixed_scorer.run_louvain(z)),
        )
        embedding_hash = sha256_file(path)
        for clusterer, detail, pred in predictions:
            result = fixed_scorer.score(pred, labels, EXCLUDED_CLASSES)
            canonical = capture_metrics(pred, labels, EXCLUDED_CLASSES)
            if not (
                canonical["capture_count"]
                <= canonical["dominant_cluster_count"]
                <= result["k_tot"]
            ):
                raise RuntimeError(f"canonical P1 invariant failed: {path.resolve()} {clusterer}")
            fragment = round(result["k_tot"] / max(1, result["n_classes"]), 4)
            rows.append(
                {
                    "protocol_id": context["protocol_id"],
                    "recipe": spec["recipe"],
                    "recipe_flags": spec["recipe_flags"],
                    "seed": spec["seed"],
                    "epoch": spec["epoch"],
                    "embedding_space": spec["embedding_space"],
                    "clusterer": clusterer,
                    "clusterer_detail": detail,
                    "P1_capture_count": canonical["capture_count"],
                    "P1_target_class_count": canonical["target_class_count"],
                    "P1_capture": round(canonical["capture_rate"], 4),
                    "dominant_cluster_count": canonical["dominant_cluster_count"],
                    "recov": result["recov"],
                    "P2_noise_pct": result["noise_pct"],
                    "P3_completeness": result["completeness"],
                    "P4_homogeneity": result["homogeneity"],
                    "k_total": result["k_tot"],
                    "k_target_classes": result["n_classes"],
                    "k_excluded_dominant": result["k_noise"],
                    "fragment_ratio": fragment,
                    "Sil": silhouette(z, np.asarray(pred), labels),
                    "ARI_supporting": result["ari"],
                    "AMI_supporting": result["ami"],
                    "embedding_path": str(path.resolve()),
                    "embedding_sha256": embedding_hash,
                    "embedding_rows": int(z.shape[0]),
                    "embedding_dim": int(z.shape[1]),
                    "eval_dir": context["eval_dir"],
                    "eval_manifest_sha256": context["eval_manifest_sha256"],
                    "train_dir": context["train_dir"],
                    "train_manifest_sha256": context["train_manifest_sha256"],
                    "scorer_bundle_sha256": context["scorer_bundle_sha256"],
                    "exclude_classes": ",".join(EXCLUDED_CLASSES),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def metric_text(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def row_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| recipe | seed | ep | clusterer | P1 | recov | P2 noise% | P3 Comp | P4 Hom | k(target/noise) | frag | Sil | ARI* | AMI* |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {recipe} | {seed} | {epoch} | {clusterer} | {p1}/{total} | {recov} | {p2} | "
            "{p3} | {p4} | {k}/{classes}/{noise} | {frag} | {sil} | {ari} | {ami} |".format(
                recipe=row["recipe"],
                seed=row["seed"],
                epoch=row["epoch"],
                clusterer=row["clusterer"],
                p1=row["P1_capture_count"],
                total=row["P1_target_class_count"],
                recov=metric_text(row["recov"]),
                p2=metric_text(row["P2_noise_pct"], 2),
                p3=metric_text(row["P3_completeness"]),
                p4=metric_text(row["P4_homogeneity"]),
                k=row["k_total"],
                classes=row["k_target_classes"],
                noise=row["k_excluded_dominant"],
                frag=metric_text(row["fragment_ratio"], 2),
                sil=metric_text(row["Sil"]),
                ari=metric_text(row["ARI_supporting"]),
                ami=metric_text(row["AMI_supporting"]),
            )
        )
    return lines


def frozen_by_clusterer(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    frozen = {row["clusterer"]: row for row in rows if row["recipe"] == "frozen"}
    if set(frozen) != {"FINCH-p2", "Louvain-res6"}:
        raise RuntimeError("both frozen clusterer baselines are required")
    return frozen


def evaluate_three_seed_gate(
    candidate_rows: list[dict[str, Any]], frozen_rows: list[dict[str, Any]], seeds=(1, 3, 5)
) -> dict[str, Any]:
    """Apply the fixed P1-first gate; ARI/AMI are deliberately absent."""
    frozen = frozen_by_clusterer(frozen_rows)
    seed_set = set(seeds)
    result: dict[str, Any] = {
        "gate_contract": {
            "P1": "every seed must preserve the clusterer-specific frozen capture count",
            "P2": "every seed must not increase target noise",
            "P3_P4_fragment": "mean non-worse and >=2/3 seeds improve in each clusterer",
            "recov": "clusterer mean may decline by at most 0.01",
            "ARI_AMI": "supporting only; excluded from gate",
        },
        "clusterers": {},
    }
    all_pass = True
    for clusterer in ("FINCH-p2", "Louvain-res6"):
        baseline = frozen[clusterer]
        selected = sorted(
            (row for row in candidate_rows if row["clusterer"] == clusterer),
            key=lambda row: int(row["seed"]),
        )
        present = {int(row["seed"]) for row in selected}
        if present != seed_set:
            raise RuntimeError(f"{clusterer}: expected seeds {sorted(seed_set)}, found {sorted(present)}")

        def deltas(metric: str, lower_is_better: bool = False) -> list[float]:
            values = [float(row[metric]) - float(baseline[metric]) for row in selected]
            return [-value for value in values] if lower_is_better else values

        p3_delta = deltas("P3_completeness")
        p4_delta = deltas("P4_homogeneity")
        recov_delta = deltas("recov")
        fragment_gain = deltas("fragment_ratio", lower_is_better=True)
        p1_pass = all(
            int(row["P1_capture_count"]) >= int(baseline["P1_capture_count"])
            for row in selected
        )
        p2_pass = all(
            float(row["P2_noise_pct"]) <= float(baseline["P2_noise_pct"]) + 1e-9
            for row in selected
        )
        checks = {
            "P1_all_seeds": p1_pass,
            "P2_all_seeds": p2_pass,
            "P3_mean_non_worse": float(np.mean(p3_delta)) >= -1e-9,
            "P3_direction_2of3": sum(value >= 0 for value in p3_delta) >= 2,
            "P4_mean_non_worse": float(np.mean(p4_delta)) >= -1e-9,
            "P4_direction_2of3": sum(value >= 0 for value in p4_delta) >= 2,
            "fragment_mean_non_worse": float(np.mean(fragment_gain)) >= -1e-9,
            "fragment_direction_2of3": sum(value >= 0 for value in fragment_gain) >= 2,
            "recov_mean_within_0.01": float(np.mean(recov_delta)) >= -0.01,
        }
        cluster_pass = all(checks.values())
        all_pass = all_pass and cluster_pass
        result["clusterers"][clusterer] = {
            "accepted": cluster_pass,
            "checks": checks,
            "mean_delta": {
                "recov": round(float(np.mean(recov_delta)), 6),
                "P3_completeness": round(float(np.mean(p3_delta)), 6),
                "P4_homogeneity": round(float(np.mean(p4_delta)), 6),
                "fragment_improvement": round(float(np.mean(fragment_gain)), 6),
                "Sil_supporting": round(
                    float(
                        np.mean([float(row["Sil"]) for row in selected])
                        - float(baseline["Sil"])
                    ),
                    6,
                ),
                "ARI_supporting": round(
                    float(
                        np.mean([float(row["ARI_supporting"]) for row in selected])
                        - float(baseline["ARI_supporting"])
                    ),
                    6,
                ),
                "AMI_supporting": round(
                    float(
                        np.mean([float(row["AMI_supporting"]) for row in selected])
                        - float(baseline["AMI_supporting"])
                    ),
                    6,
                ),
            },
        }
    result["accepted"] = all_pass
    result["next_axis"] = (
        "isolated ignore sweep {0.70,0.75,0.80} on L0"
        if all_pass
        else "frozen-residual preservation sweep before adding contrastive components"
    )
    return result


def write_existing_report(rows: list[dict[str, Any]], context: dict[str, Any]) -> None:
    frozen = frozen_by_clusterer(rows)
    lines = [
        "# FCMAE Fixed-Protocol Re-score (260721)",
        "",
        "## Contract",
        "",
        f"- protocol_id: `{context['protocol_id']}`",
        f"- eval: `{context['eval_dir']}` ({context['eval_image_count']} images, 32 strict-novel targets)",
        f"- train manifest: `{context['train_manifest']}`",
        f"- eval manifest: `{context['eval_manifest']}`",
        f"- scorer bundle sha256: `{context['scorer_bundle_sha256']}`",
        "- P1 is canonical dominant-main-class capture. Presence coverage is not P1.",
        "- P2 is target-image clusterer noise. This eval pool contains no Normal/Random/R, so this is not FAR.",
        "- ARI/AMI columns are retained only as supporting diagnostics.",
        "- L4/L5 are excluded because the cumulative ladder continued after a failed rung.",
        "",
        "## Frozen Baseline",
        "",
    ]
    lines.extend(row_table(list(frozen.values())))
    lines.extend(["", "## Re-scored Rows", ""])
    lines.extend(row_table([row for row in rows if row["recipe"] != "frozen"]))
    lines.extend(
        [
            "",
            "## Fixed Confirmation Target",
            "",
            "`L0_base ep4` is the pre-declared seed-confirmation point. It is not accepted from seed 3 alone.",
            "Seeds 1/3/5 must pass the P1-first gate in the separate fixed-epoch report.",
            "",
            f"Raw rows: `{EXISTING_CSV.resolve()}`",
            f"Protocol: `{PROTOCOL_JSON.resolve()}`",
        ]
    )
    EXISTING_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_seed_report(
    rows: list[dict[str, Any]], frozen_rows: list[dict[str, Any]], gate: dict[str, Any]
) -> None:
    lines = [
        "# FCMAE L0 Adapter ep4 Three-Seed Confirmation (260721)",
        "",
        "## Decision",
        "",
        f"- accepted: **{str(gate['accepted']).upper()}**",
        f"- next axis: {gate['next_axis']}",
        "- fixed point: L0 pure SimCLR residual adapter, adapted f, epoch 4, seeds 1/3/5.",
        "- P1/P2/P3/P4/fragmentation decide the gate. ARI/AMI remain visible but cannot rescue a P1 failure.",
        "",
        "## Frozen Baseline",
        "",
    ]
    lines.extend(row_table(frozen_rows))
    lines.extend(["", "## Seed Results", ""])
    lines.extend(row_table(rows))
    lines.extend(["", "## Gate Details", ""])
    for clusterer, detail in gate["clusterers"].items():
        lines.append(f"### {clusterer}")
        lines.append("")
        lines.append(f"- accepted: {detail['accepted']}")
        for name, passed in detail["checks"].items():
            lines.append(f"- {name}: {passed}")
        means = detail["mean_delta"]
        lines.append(
            "- mean deltas: recov={recov:+.4f}, P3={p3:+.4f}, P4={p4:+.4f}, "
            "fragment improvement={frag:+.4f}, Sil*={sil:+.4f}, ARI*={ari:+.4f}, AMI*={ami:+.4f}".format(
                recov=means["recov"],
                p3=means["P3_completeness"],
                p4=means["P4_homogeneity"],
                frag=means["fragment_improvement"],
                sil=means["Sil_supporting"],
                ari=means["ARI_supporting"],
                ami=means["AMI_supporting"],
            )
        )
        lines.append("")
    lines.extend(
        [
            f"Raw rows: `{SEED_CSV.resolve()}`",
            f"Gate JSON: `{SEED_GATE_JSON.resolve()}`",
        ]
    )
    SEED_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rescore_existing() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    context = protocol_context()
    rows = score_specs(existing_specs(), context)
    write_csv(EXISTING_CSV, rows)
    write_existing_report(rows, context)
    print(f"[out] {EXISTING_CSV.resolve()}")
    print(f"[out] {EXISTING_MD.resolve()}")
    return rows, context


def load_frozen_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    return score_specs([existing_specs()[0]], context)


def rescore_seeds() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    context = protocol_context()
    rows = score_specs(seed_specs(), context)
    frozen_rows = load_frozen_rows(context)
    gate = evaluate_three_seed_gate(rows, frozen_rows)
    write_csv(SEED_CSV, rows)
    SEED_GATE_JSON.write_text(
        json.dumps(gate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_seed_report(rows, frozen_rows, gate)
    print(json.dumps(gate, indent=2, ensure_ascii=False))
    print(f"[out] {SEED_CSV.resolve()}")
    print(f"[out] {SEED_MD.resolve()}")
    return rows, gate


def training_command(seed: int) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(REPO / "_ssl_methods.py"),
        "--method",
        "simclr",
        "--timm",
        "convnextv2_base.fcmae_ft_in22k_in1k_384",
        "--freeze-backbone",
        "--head",
        "adapter",
        "--pdim",
        "128",
        "--seed",
        str(seed),
        "--epochs",
        "4",
        "--batch",
        "8",
        "--ckpt-every",
        "100",
        "--train-dir",
        str(TRAIN_DIR.resolve()),
        "--eval-dir",
        str(EVAL_DIR.resolve()),
        "--wafer-rot-deg",
        "0",
        "--wafer-translate",
        "0",
        "--wafer-scale-min",
        "1.0",
        "--wafer-crop-min",
        "1.0",
        "--out-dir",
        str(SEED_EMBEDDINGS.resolve()),
        "--tag",
        f"fcmae_ad1_s{seed}",
    ]


def run_seed_queue() -> None:
    SEED_EMBEDDINGS.mkdir(parents=True, exist_ok=True)
    config = {
        "fixed_epoch": 4,
        "seeds": [1, 3, 5],
        "commands": {str(seed): training_command(seed) for seed in (1, 3, 5)},
        "seed3_source": seed3_source_record(),
        "note": "Seeds 1/3/5 are all trained or resumed at fixed epoch 4; seed_specs prefers generated seed3 output and only then uses the legacy fallback.",
    }
    (SEED_RUN_ROOT / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    for seed in (1, 3, 5):
        expected = SEED_EMBEDDINGS / f"fcmae_ad1_s{seed}_ep4.npy"
        if expected.is_file():
            print(f"[skip] seed {seed}: {expected.resolve()}", flush=True)
            continue
        command = training_command(seed)
        print(f"[train] seed {seed}: {' '.join(command)}", flush=True)
        with SEED_LOG.open("a", encoding="utf-8") as log:
            log.write(f"\n[{datetime.now().astimezone().isoformat()}] seed={seed}\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=REPO,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        if code != 0:
            raise RuntimeError(f"seed {seed} training failed with exit code {code}")
        if not expected.is_file():
            raise RuntimeError(f"seed {seed} finished without {expected.resolve()}")
        config["seed3_source"] = seed3_source_record()
        (SEED_RUN_ROOT / "config.json").write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    config["seed3_source"] = seed3_source_record()
    (SEED_RUN_ROOT / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rescore_seeds()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("rescore-existing", "run-seeds", "rescore-seeds"),
        help="Fixed protocol operation to execute.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "rescore-existing":
        rescore_existing()
    elif args.action == "run-seeds":
        run_seed_queue()
    else:
        rescore_seeds()


if __name__ == "__main__":
    main()
