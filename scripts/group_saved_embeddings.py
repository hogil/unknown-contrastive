#!/usr/bin/env python3
"""Create review grouping artifacts from a saved embedding matrix.

Outputs:
- clusters.csv: path, group_id, label
- groups/<group_id>/*.png when --copy-groups is used
- representatives/cluster_xxx/rep-*.png: centroid-nearest references
- representatives/composite/*.png: composite maps from representatives
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import predict_grouping_prod as pg  # noqa: E402
import _score_umapfree as scorelib  # noqa: E402


EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_imgs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)


def l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def cluster(z: np.ndarray, method: str) -> np.ndarray:
    method = method.lower()
    if method.startswith("finch_p"):
        part = int(method.removeprefix("finch_p"))
        from finch import FINCH
        c, _, _ = FINCH(z, verbose=False)
        if part >= c.shape[1]:
            raise SystemExit(f"{method} unavailable; FINCH returned {c.shape[1]} partitions")
        return c[:, part].astype(int)
    if method == "louvain":
        return scorelib.run_louvain(z).astype(int)
    if method == "hdbscan":
        return scorelib.run_hdbscan_raw(z).astype(int)
    raise SystemExit(f"unknown method: {method}")


def parse_csv_set(value: str) -> set[str]:
    return {x.strip().lower() for x in str(value).split(",") if x.strip()}


def background_group_stats(
    pred: np.ndarray,
    labels: list[str],
    background_classes: set[str],
    min_count: int,
    min_ratio: float,
) -> tuple[set[int], dict[int, dict[str, int | float]]]:
    stats: dict[int, dict[str, int | float]] = {}
    group_counts = Counter(int(x) for x in pred.tolist())
    bg_counts: Counter[int] = Counter()
    for gid, label in zip(pred.tolist(), labels):
        if label.lower() in background_classes:
            bg_counts[int(gid)] += 1
    excluded: set[int] = set()
    for gid, total in group_counts.items():
        bg = int(bg_counts.get(gid, 0))
        ratio = float(bg / total) if total else 0.0
        stats[int(gid)] = {
            "group_size": int(total),
            "background_count": bg,
            "background_ratio": ratio,
        }
        if gid >= 0 and bg >= min_count and ratio >= min_ratio:
            excluded.add(int(gid))
    return excluded, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--pool", required=True, help="eval image root used to create the embedding")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--method", default="finch_p2",
                    choices=["finch_p0", "finch_p1", "finch_p2", "finch_p3", "louvain", "hdbscan"])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--copy-groups", action="store_true")
    ap.add_argument("--background-classes", default="Normal,Random,R",
                    help="comma-separated labels treated as background for review-group exclusion")
    ap.add_argument("--exclude-background-min-count", type=int, default=20,
                    help="exclude non-noise groups from review artifacts when background count is at least this value")
    ap.add_argument("--exclude-background-min-ratio", type=float, default=0.5,
                    help="exclude non-noise groups from review artifacts when background ratio is at least this value")
    args = ap.parse_args()

    emb_path = Path(args.embedding)
    pool = Path(args.pool)
    out_dir = Path(args.out_dir)
    if not emb_path.is_absolute():
        emb_path = REPO / emb_path
    if not pool.is_absolute():
        pool = REPO / pool
    if not out_dir.is_absolute():
        out_dir = REPO / out_dir

    paths = list_imgs(pool)
    z = l2(np.load(emb_path).astype(np.float32))
    if len(paths) != z.shape[0]:
        raise SystemExit(f"path/embedding mismatch: {len(paths)} paths vs {z.shape[0]} embeddings")

    pred = cluster(z, args.method)
    labels = [p.parent.name for p in paths]
    background_classes = parse_csv_set(args.background_classes)
    excluded_groups, group_stats = background_group_stats(
        pred, labels, background_classes,
        int(args.exclude_background_min_count),
        float(args.exclude_background_min_ratio),
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path, label, gid in zip(paths, labels, pred):
        gid_i = int(gid)
        st = group_stats[gid_i]
        if gid_i == -1:
            status = "noise"
        elif gid_i in excluded_groups:
            status = "background_excluded"
        else:
            status = "candidate"
        rows.append({
            "path": str(path.resolve()),
            "label": label,
            "group_id": gid_i,
            "review_status": status,
            "group_size": int(st["group_size"]),
            "background_count": int(st["background_count"]),
            "background_ratio": f"{float(st['background_ratio']):.6f}",
        })
    with (out_dir / "clusters.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "path", "label", "group_id", "review_status",
            "group_size", "background_count", "background_ratio",
        ])
        w.writeheader()
        w.writerows(rows)

    excluded_rows = [
        {
            "group_id": gid,
            "group_size": int(group_stats[gid]["group_size"]),
            "background_count": int(group_stats[gid]["background_count"]),
            "background_ratio": f"{float(group_stats[gid]['background_ratio']):.6f}",
        }
        for gid in sorted(excluded_groups)
    ]
    with (out_dir / "background_excluded_groups.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "group_id", "group_size", "background_count", "background_ratio",
        ])
        w.writeheader()
        w.writerows(excluded_rows)

    np.save(out_dir / "embeddings.npy", z.astype(np.float32))

    if args.copy_groups:
        groups = out_dir / "groups"
        if groups.exists():
            shutil.rmtree(groups)
        for path, gid in zip(paths, pred):
            if int(gid) in excluded_groups:
                continue
            sub = groups / f"{int(gid):03d}"
            sub.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, sub / path.name)

    pg.SAVE_REFERENCE_COMPOSITES = True
    review_pred = pred.copy()
    if excluded_groups:
        review_pred[np.isin(review_pred, list(excluded_groups))] = -1
    saved = pg.save_grouping_representatives(
        out_dir, z, review_pred, [str(p) for p in paths], int(args.reps))

    cnt = Counter(int(x) for x in pred.tolist())
    n_clusters = len([k for k in cnt if k >= 0])
    excluded_image_count = sum(int(group_stats[g]["group_size"]) for g in excluded_groups)
    summary = {
        "embedding": str(emb_path.resolve()),
        "pool": str(pool.resolve()),
        "method": args.method,
        "n_images": len(paths),
        "n_clusters": n_clusters,
        "n_noise": int(cnt.get(-1, 0)),
        "background_filter": {
            "classes": sorted(background_classes),
            "min_count": int(args.exclude_background_min_count),
            "min_ratio": float(args.exclude_background_min_ratio),
            "excluded_group_count": len(excluded_groups),
            "excluded_image_count": int(excluded_image_count),
            "note": "Excluded groups are removed from review artifacts only; original group_id is preserved and not converted to noise.",
        },
        "background_excluded_groups": {
            str(gid): {
                "group_size": int(group_stats[gid]["group_size"]),
                "background_count": int(group_stats[gid]["background_count"]),
                "background_ratio": float(group_stats[gid]["background_ratio"]),
            }
            for gid in sorted(excluded_groups)
        },
        "groups": {str(k): int(v) for k, v in sorted(cnt.items())},
        "representatives_saved": saved,
        "reps_per_cluster": int(args.reps),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OUT] {out_dir.resolve()}")
    print(f"[REP] {out_dir.resolve() / 'representatives'}")


if __name__ == "__main__":
    main()
