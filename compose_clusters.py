#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""compose_clusters.py — cluster composite map (binary + grademean).

Per cluster:
  1. centroid = mean(member embeddings) L2-normalized
  2. K=20 closest members by cosine distance (or all if cluster_size < K)
  3. stream-load palette PNGs (idx 0-31), accumulate:
       binary_count[y,x] = sum(1 if 1<=grade<=7 else 0)
       grade_sum[y,x]    = sum(grade if 1<=grade<=7 else 0)
       valid_count[y,x]  = sum(1 if grade<=7 else 0)
  4. derive:
       Method A (binary):    val = binary_count / N            range [0,1]
       Method B (grademean): val = grade_sum / N / 7           range [0,1]
       invalid mask: valid_count == 0  →  val = 0  →  rendered white
  5. render RGB: 0=white, 1=red linear (R=255, G=255*(1-v), B=255*(1-v))
  6. save:
       <run>/cluster_summary/composite/binary/cluster_XXX.png
       <run>/cluster_summary/composite/grademean/cluster_XXX.png

Usage:
    python compose_clusters.py --run outputs/logs_contrastive/overall
    python compose_clusters.py --run <run> --only-cluster 0
    python compose_clusters.py --run <run> --skip-existing
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_K = 20
EXPECTED_DIM = 6400
PIL_DECOMPRESSION_LIMIT = 6400 * 6400 * 4
Image.MAX_IMAGE_PIXELS = PIL_DECOMPRESSION_LIMIT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path,
                   help="run dir, e.g. outputs/logs_contrastive/overall")
    p.add_argument("--k", type=int, default=DEFAULT_K, help="medoids per cluster")
    p.add_argument("--methods", default="binary,grademean",
                   help="comma list: binary,grademean")
    p.add_argument("--only-cluster", type=int, default=None,
                   help="run on single cluster id (smoke)")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip cluster if output PNG already present")
    return p.parse_args()


def load_inputs(run: Path):
    embedding = np.load(run / "eval/embeddings/embedding.npy")
    files = (run / "eval/embeddings/files.txt").read_text(encoding="utf-8").splitlines()
    file_list = pd.read_parquet(run / "eval/file_list.parquet")
    if len(embedding) != len(files):
        raise ValueError(f"embedding ({len(embedding)}) vs files.txt ({len(files)}) mismatch")
    return embedding, files, file_list


def normalize_rows(arr: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr / n


def pick_top_k(emb_n: np.ndarray, member_idx: np.ndarray, k: int) -> np.ndarray:
    members = emb_n[member_idx]
    centroid = members.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-12
    cos_dist = 1.0 - members @ centroid
    if len(member_idx) <= k:
        order = np.argsort(cos_dist)
    else:
        order = np.argpartition(cos_dist, k)[:k]
        order = order[np.argsort(cos_dist[order])]
    return member_idx[order]


_SOURCE_INDEX: dict[str, Path] | None = None


def _build_source_index() -> dict[str, Path]:
    """One-shot rglob: basename → source 6400 palette PNG path.
    file_list.parquet 의 path 는 384 RGB cache 라 무시. 항상 unknown/ 원본 사용.
    """
    global _SOURCE_INDEX
    if _SOURCE_INDEX is not None:
        return _SOURCE_INDEX
    idx: dict[str, Path] = {}
    for root in (Path("D:/project/data/wm-811k/unknown"),
                 Path("D:/project/data/wm-811k/unknown_archive")):
        if not root.exists():
            continue
        for cand in root.rglob("*.png"):
            idx.setdefault(cand.name, cand)
    _SOURCE_INDEX = idx
    return idx


def resolve_path(row: pd.Series) -> Path:
    base = str(row["wafer_basename"])
    if not base.endswith(".png"):
        base = base + ".png"
    idx = _build_source_index()
    p = idx.get(base)
    if p is None:
        raise FileNotFoundError(f"source PNG not found for {base}")
    return p


def accumulate(paths: list[Path], H: int, W: int):
    binary_count = np.zeros((H, W), dtype=np.uint8)
    grade_sum = np.zeros((H, W), dtype=np.uint8)
    valid_count = np.zeros((H, W), dtype=np.uint8)
    used = 0
    for p in paths:
        try:
            with Image.open(p) as im:
                if im.mode != "P":
                    im = im.convert("P")
                arr = np.asarray(im, dtype=np.uint8)
        except Exception as e:
            print(f"  [skip] {p.name}: {e}")
            continue
        if arr.shape != (H, W):
            print(f"  [skip] {p.name}: shape {arr.shape} != ({H},{W})")
            continue
        valid = arr <= 7
        defect = (arr >= 1) & valid
        valid_count += valid
        binary_count += defect
        grade_sum += np.where(valid, arr, np.uint8(0))
        used += 1
    return binary_count, grade_sum, valid_count, used


def render_white_to_red(value: np.ndarray) -> np.ndarray:
    """0 → white (255,255,255), 1 → red (255,0,0). value clipped to [0,1]."""
    v = np.clip(value, 0.0, 1.0)
    inv = (255.0 * (1.0 - v)).round().astype(np.uint8)
    H, W = v.shape
    rgb = np.empty((H, W, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    rgb[..., 1] = inv
    rgb[..., 2] = inv
    return rgb


def save_png(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG", compress_level=6)


def main() -> int:
    args = parse_args()
    run = args.run.resolve()
    if not run.exists():
        print(f"run not found: {run}", file=sys.stderr)
        return 2
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    valid_methods = {"binary", "grademean"}
    bad = [m for m in methods if m not in valid_methods]
    if bad:
        print(f"unknown methods: {bad}", file=sys.stderr)
        return 2

    print(f"[load] {run}")
    embedding, files, file_list = load_inputs(run)
    emb_n = normalize_rows(embedding)

    base2idx = {Path(f).name: i for i, f in enumerate(files)}

    out_root = run / "cluster_summary" / "composite"
    print(f"[output] {out_root}")

    clusters = sorted([c for c in file_list["cluster_id"].unique().tolist() if c >= 0])
    if args.only_cluster is not None:
        if args.only_cluster not in clusters:
            print(f"cluster {args.only_cluster} not found", file=sys.stderr)
            return 2
        clusters = [args.only_cluster]

    print(f"[clusters] {len(clusters)} (skipping noise -1)")

    H = W = EXPECTED_DIM
    t0 = time.time()
    saved = 0
    skipped = 0
    for cid in clusters:
        cluster_rows = file_list[file_list["cluster_id"] == cid].reset_index(drop=True)
        n_members = len(cluster_rows)
        if n_members < 2:
            print(f"[cluster {cid}] size={n_members} < 2 — skip")
            continue

        # outputs
        out_paths = {m: out_root / m / f"cluster_{cid:03d}.png" for m in methods}
        if args.skip_existing and all(p.exists() for p in out_paths.values()):
            print(f"[cluster {cid}] all methods exist — skip-existing")
            skipped += 1
            continue

        # member indices in embedding
        try:
            member_idx = np.array([base2idx[Path(p).name]
                                   for p in cluster_rows["path"]], dtype=np.int64)
        except KeyError as e:
            print(f"[cluster {cid}] basename mismatch: {e}")
            continue

        top_idx = pick_top_k(emb_n, member_idx, args.k)
        cluster_by_base = {Path(r["path"]).name: r for _, r in cluster_rows.iterrows()}
        top_paths: list[Path] = []
        for i in top_idx:
            row = cluster_by_base.get(Path(files[i]).name)
            if row is None:
                continue
            try:
                top_paths.append(resolve_path(row))
            except FileNotFoundError as e:
                print(f"  [warn] {e}")
        if len(top_paths) < 2:
            print(f"[cluster {cid}] only {len(top_paths)} resolvable paths — skip")
            continue

        print(f"[cluster {cid}] size={n_members} K={len(top_paths)} accumulating...", end=" ", flush=True)
        ts = time.time()
        binary_count, grade_sum, valid_count, used = accumulate(top_paths, H, W)
        print(f"acc {time.time()-ts:.1f}s used={used}", end=" ", flush=True)

        N = used
        invalid_mask = valid_count == 0

        if "binary" in methods:
            ts = time.time()
            val = binary_count.astype(np.float32) / float(N)
            val[invalid_mask] = 0.0
            rgb = render_white_to_red(val)
            save_png(rgb, out_paths["binary"])
            print(f"binary {time.time()-ts:.1f}s", end=" ", flush=True)
            del val, rgb
            saved += 1

        if "grademean" in methods:
            ts = time.time()
            val = grade_sum.astype(np.float32) / float(N) / 7.0
            val[invalid_mask] = 0.0
            rgb = render_white_to_red(val)
            save_png(rgb, out_paths["grademean"])
            print(f"grademean {time.time()-ts:.1f}s", end=" ", flush=True)
            del val, rgb
            saved += 1

        print()
        del binary_count, grade_sum, valid_count

    print(f"\n[done] saved={saved} skipped_existing={skipped} elapsed={time.time()-t0:.1f}s")
    print(f"[output] {out_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
