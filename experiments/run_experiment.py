#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Runner for a SINGLE experiment preset.

Usage
-----
    python -m experiments.run_experiment \
        --preset baseline \
        --dataset-root /.../data/unknown \
        --train-root   /.../data/self \
        --output-base  /.../outputs_exp \
        [--use-subset] \
        [--extra-cfg TEMP=0.08 BATCH=128]

Why per-invocation (not sweep)?
    contrastive.py computes RUN_TS = datetime.now()... at module load time,
    which means a second preset in the same process would reuse the old
    timestamp. Run each preset in its own process (bash loop on the server).

After contrastive.main() returns we:
    1. locate the run directory created by contrastive (OUTPUT_DIR_<TS>)
    2. parse clusters_global_list.txt for (filename, cluster id) pairs
    3. derive ground-truth labels from the parent folder name of each image
    4. compute ARI/NMI/purity via experiments.eval_metrics
    5. dump everything to <run_dir>/eval_summary.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# =========================================================
# Helpers
# =========================================================
def _parse_extra_cfg(pairs: List[str]) -> Dict[str, Any]:
    """Parse KEY=VALUE strings; values are JSON-decoded if possible."""
    out: Dict[str, Any] = {}
    for raw in pairs:
        if "=" not in raw:
            raise ValueError(f"--extra-cfg entry must be KEY=VALUE, got: {raw}")
        k, v = raw.split("=", 1)
        k = k.strip()
        v = v.strip()
        try:
            out[k] = json.loads(v)
        except Exception:
            if v.lower() in ("true", "false"):
                out[k] = (v.lower() == "true")
            elif v.lower() in ("none", "null"):
                out[k] = None
            else:
                # numeric fallback
                try:
                    if "." in v or "e" in v.lower():
                        out[k] = float(v)
                    else:
                        out[k] = int(v)
                except ValueError:
                    out[k] = v
    return out


def _build_subset_tree(
    items,
    tmp_root: Path,
) -> Path:
    """Symlink (fallback copy) the selected items into an ImageFolder tree."""
    tmp_root = Path(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)
    for src, cls in items:
        dst_dir = tmp_root / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / Path(src).name
        if dst.exists() or dst.is_symlink():
            continue
        try:
            os.symlink(os.fspath(src), os.fspath(dst))
        except Exception:
            shutil.copy2(os.fspath(src), os.fspath(dst))
    return tmp_root


def _find_latest_run_dir(output_dir: str) -> Path:
    """
    contrastive.main() creates run_dir = f"{OUTPUT_DIR}_{RUN_TS}".
    Find the newest one that starts with OUTPUT_DIR's basename.
    """
    parent = Path(output_dir).parent
    stem = Path(output_dir).name
    candidates = sorted(
        [p for p in parent.glob(f"{stem}_*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No run directory found for base prefix {output_dir!r}"
        )
    return candidates[0]


def _parse_clusters_global_list(txt_path: Path) -> List[Tuple[int, str, str, str]]:
    """
    Parse clusters_global_list.txt.
    Columns: clusterno  status  class  filename  relpath  root  step  wafer  yyyymmdd  hhmmss
    Returns list of (cluster_no, status, class_name, relpath).
    """
    out: List[Tuple[int, str, str, str]] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        header = f.readline()  # skip
        if not header.lower().startswith("cluster"):
            # edge case: no header — rewind and read all as data
            f.seek(0)
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            try:
                cid = int(parts[0])
            except ValueError:
                continue
            status = parts[1]
            cls = parts[2]
            relpath = parts[4]
            out.append((cid, status, cls, relpath))
    return out


def _compute_and_save_eval(
    run_dir: Path,
    preset_name: str,
    cfg_snapshot: Dict[str, Any],
    wall_time_sec: float,
    unknown_dir: str,
) -> Dict[str, Any]:
    from experiments.eval_metrics import compute_cluster_metrics

    glist = run_dir / "clusters_global_list.txt"
    if not glist.exists():
        raise FileNotFoundError(f"Expected {glist} not found.")

    records = _parse_clusters_global_list(glist)
    if not records:
        raise RuntimeError(f"No records parsed from {glist}")

    # NOTE: clusters_global_list.txt only contains assigned clusters — HDBSCAN
    # noise (-1) is NOT written there. Re-derive noise by diffing against the
    # full unknown dir. We only count samples actually present in the list for
    # clustering metrics; noise is reported separately.
    # For ARI/NMI we use all records in the list (these are the non-noise
    # samples), their GT labels being the class column.
    class_names = sorted({r[2] for r in records})
    cls_to_idx = {c: i for i, c in enumerate(class_names)}

    cluster_ids = np.asarray([r[0] for r in records], dtype=np.int64)
    gt_ids = np.asarray([cls_to_idx[r[2]] for r in records], dtype=np.int64)

    metrics = compute_cluster_metrics(cluster_ids, gt_ids)

    # Approximate n_noise by counting total files under unknown_dir and
    # subtracting how many made it into the list.
    total_files = 0
    try:
        from experiments.eval_metrics import IMG_EXTS
        for p in Path(unknown_dir).rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                total_files += 1
    except Exception:
        total_files = metrics["n_samples"]
    derived_noise = max(0, total_files - metrics["n_samples"])

    summary = {
        "preset": preset_name,
        "run_dir": str(run_dir),
        "unknown_dir": unknown_dir,
        "wall_time_sec": wall_time_sec,
        "total_unknown_files": total_files,
        "assigned_samples": metrics["n_samples"],
        "derived_noise": derived_noise,
        "derived_noise_ratio": (
            float(derived_noise) / float(total_files) if total_files else 0.0
        ),
        "metrics": metrics,
        "cfg_snapshot": cfg_snapshot,
        "class_to_idx": cls_to_idx,
    }

    out_path = run_dir / "eval_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


# =========================================================
# Preset application
# =========================================================
def apply_preset(
    contrastive_module,
    preset,
    dataset_root: str,
    train_root: str,
    output_base: str,
    use_subset: bool,
    extra_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Mutates contrastive_module.CFG in place. Returns a snapshot (deep copy)
    of the final CFG after all overrides.
    """
    CFG = contrastive_module.CFG

    CFG["TRAIN_DIR"] = train_root
    CFG["UNKNOWN_DIR"] = dataset_root
    # Keep OUTPUT_DIR's basename rooted at preset — final run_dir will be
    # <output_base>/<preset>_<RUN_TS> as built by contrastive.main().
    CFG["OUTPUT_DIR"] = str(Path(output_base) / preset.name)

    # Preset CFG overrides
    CFG.update(preset.cfg_overrides)

    # ad-hoc extra overrides (highest precedence)
    CFG.update(extra_cfg)

    # Structural markers — contrastive.py (Task 4) will read these.
    CFG["__INPUT_MODE__"] = preset.input_mode
    CFG["__UNFREEZE_STAGES__"] = list(preset.unfreeze_stages)
    CFG["__HARD_NEGATIVES__"] = bool(preset.hard_negatives)
    CFG["__STRONG_AUGMENT__"] = bool(preset.strong_augment)
    CFG["__MULTICROP__"] = bool(preset.multicrop)
    CFG["__DEEPER_HEAD__"] = bool(preset.deeper_head)
    CFG["__PRESET_NAME__"] = preset.name
    CFG["__PRESET_DESCRIPTION__"] = preset.description

    # If --use-subset, build a symlink tree of the top-N x N_PER_CLASS images
    # and point BOTH TRAIN_DIR and UNKNOWN_DIR at it so sweeps are apples-to-
    # apples. The subset tree lives at <output_base>/_subsets/<preset>/ so it
    # persists across reruns and is safe to reuse.
    if use_subset:
        from experiments.eval_metrics import compute_subset_selection
        from experiments.presets import EVAL_SUBSET

        subset_root = Path(output_base) / "_subsets" / preset.name
        # Only build if missing/empty
        needs_build = True
        if subset_root.exists():
            has_any = any(subset_root.rglob("*"))
            if has_any:
                needs_build = False
        if needs_build:
            items = compute_subset_selection(
                Path(dataset_root),
                n_classes=EVAL_SUBSET["N_CLASSES"],
                n_per_class=EVAL_SUBSET["N_PER_CLASS"],
                selection=EVAL_SUBSET["SELECTION"],
                explicit_classes=EVAL_SUBSET["EXPLICIT_CLASSES"],
                seed=EVAL_SUBSET["SEED"],
            )
            _build_subset_tree(items, subset_root)

        CFG["UNKNOWN_DIR"] = str(subset_root)
        CFG["TRAIN_DIR"] = str(subset_root)

    return copy.deepcopy(CFG)


# =========================================================
# CLI
# =========================================================
def build_arg_parser() -> argparse.ArgumentParser:
    from experiments.presets import PRESETS

    p = argparse.ArgumentParser(
        prog="experiments.run_experiment",
        description=(
            "Run a single contrastive.py preset with CFG overrides, then "
            "compute ARI/NMI/purity and write eval_summary.json."
        ),
    )
    p.add_argument(
        "--preset",
        required=True,
        choices=sorted(PRESETS.keys()),
        help="Preset name from experiments.presets.PRESETS.",
    )
    p.add_argument(
        "--dataset-root",
        required=True,
        help="Path to UNKNOWN_DIR (ImageFolder-style, class sub-dirs).",
    )
    p.add_argument(
        "--train-root",
        required=True,
        help="Path to TRAIN_DIR (self-supervised training images).",
    )
    p.add_argument(
        "--output-base",
        required=True,
        help="Base dir for all preset runs (e.g. /.../outputs_exp).",
    )
    p.add_argument(
        "--use-subset",
        action="store_true",
        help=(
            "Restrict training & eval to the 10 largest classes x 200 images "
            "subset defined in experiments.presets.EVAL_SUBSET."
        ),
    )
    p.add_argument(
        "--extra-cfg",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Extra ad-hoc CFG overrides applied after preset. "
            "Example: --extra-cfg TEMP=0.08 BATCH=128"
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Apply preset + print the resulting CFG then exit (no training).",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    from experiments.presets import PRESETS  # noqa: F401
    from experiments.presets import get_preset

    # Import contrastive lazily so --help works without torch installed.
    import contrastive  # type: ignore

    preset = get_preset(args.preset)
    extra = _parse_extra_cfg(args.extra_cfg)

    cfg_snapshot = apply_preset(
        contrastive_module=contrastive,
        preset=preset,
        dataset_root=args.dataset_root,
        train_root=args.train_root,
        output_base=args.output_base,
        use_subset=args.use_subset,
        extra_cfg=extra,
    )

    print(f"[run_experiment] preset = {preset.name}")
    print(f"[run_experiment] description = {preset.description}")
    print(f"[run_experiment] OUTPUT_DIR = {contrastive.CFG['OUTPUT_DIR']}")
    print(f"[run_experiment] TRAIN_DIR  = {contrastive.CFG['TRAIN_DIR']}")
    print(f"[run_experiment] UNKNOWN_DIR= {contrastive.CFG['UNKNOWN_DIR']}")

    if args.dry_run:
        print(json.dumps(cfg_snapshot, indent=2, default=str, ensure_ascii=False))
        return 0

    t0 = time.time()
    contrastive.main()
    wall = time.time() - t0

    # Locate the run directory created by contrastive.main()
    run_dir = _find_latest_run_dir(contrastive.CFG["OUTPUT_DIR"])
    print(f"[run_experiment] run_dir = {run_dir}")

    summary = _compute_and_save_eval(
        run_dir=run_dir,
        preset_name=preset.name,
        cfg_snapshot=cfg_snapshot,
        wall_time_sec=wall,
        unknown_dir=contrastive.CFG["UNKNOWN_DIR"],
    )
    m = summary["metrics"]
    print(
        "[run_experiment] metrics: "
        f"ARI={m['ari']:.4f} NMI={m['nmi']:.4f} "
        f"purity={m['cluster_purity']:.4f} "
        f"clusters={m['n_clusters_found']} "
        f"noise_ratio={summary['derived_noise_ratio']:.3f} "
        f"wall={wall:.1f}s"
    )
    print(f"[run_experiment] wrote {run_dir / 'eval_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
