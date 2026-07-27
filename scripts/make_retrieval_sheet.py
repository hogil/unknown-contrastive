#!/usr/bin/env python3
"""Build nearest-neighbor retrieval sheets from saved embeddings."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _abs_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def _safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(text))
    return text.strip("_") or "item"


def _topk_neighbors(emb: np.ndarray, topk: int, block: int = 512) -> tuple[np.ndarray, np.ndarray]:
    n = emb.shape[0]
    top_idx = np.empty((n, topk), dtype=np.int32)
    top_sim = np.empty((n, topk), dtype=np.float32)
    all_t = emb.T
    for start in range(0, n, block):
        end = min(start + block, n)
        sim = emb[start:end] @ all_t
        rows = np.arange(end - start)
        sim[rows, np.arange(start, end)] = -np.inf
        part = np.argpartition(-sim, kth=np.arange(topk), axis=1)[:, :topk]
        part_sim = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(-part_sim, axis=1)
        top_idx[start:end] = np.take_along_axis(part, order, axis=1)
        top_sim[start:end] = np.take_along_axis(part_sim, order, axis=1)
    return top_idx, top_sim


def _choose_queries(
    emb: np.ndarray,
    labels: list[str] | None,
    queries_per_class: int,
    max_queries: int,
) -> list[int]:
    if labels:
        chosen: list[int] = []
        label_arr = np.asarray(labels)
        for label in sorted(set(labels)):
            idxs = np.where(label_arr == label)[0]
            if idxs.size == 0:
                continue
            centroid = emb[idxs].mean(axis=0)
            centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
            scores = emb[idxs] @ centroid
            order = np.argsort(-scores)[:queries_per_class]
            chosen.extend(int(idxs[i]) for i in order)
        return chosen

    centroid = emb.mean(axis=0)
    centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-12)
    scores = emb @ centroid
    return [int(i) for i in np.argsort(-scores)[:max_queries]]


def _thumb(path: Path, size: int) -> Image.Image:
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im = ImageOps.contain(im, (size, size), method=Image.Resampling.BILINEAR)
            canvas = Image.new("RGB", (size, size), (245, 245, 245))
            x = (size - im.width) // 2
            y = (size - im.height) // 2
            canvas.paste(im, (x, y))
            return canvas
    except Exception:
        canvas = Image.new("RGB", (size, size), (235, 235, 235))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), "LOAD FAIL", fill=(180, 0, 0))
        return canvas


def _draw_fit(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int]) -> None:
    text = str(text)
    if len(text) > 24:
        text = text[:21] + "..."
    draw.text(xy, text, fill=fill)


def _make_sheet(
    query_idx: int,
    neigh_idx: np.ndarray,
    neigh_sim: np.ndarray,
    paths: list[str],
    labels: list[str] | None,
    out_file: Path,
    thumb_size: int,
) -> None:
    font = ImageFont.load_default()
    gap = 8
    text_h = 44
    title_h = 34
    cols = 1 + len(neigh_idx)
    w = cols * thumb_size + (cols + 1) * gap
    h = title_h + thumb_size + text_h + 2 * gap
    sheet = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    q_label = labels[query_idx] if labels else "query"
    title = f"query={query_idx} label={q_label} topk={len(neigh_idx)}"
    draw.text((gap, 8), title, fill=(0, 0, 0), font=font)

    items = [(query_idx, 1.0, True)] + [
        (int(i), float(s), bool(labels and labels[int(i)] == q_label))
        for i, s in zip(neigh_idx, neigh_sim)
    ]
    for col, (idx, sim, same) in enumerate(items):
        x = gap + col * (thumb_size + gap)
        y = title_h
        border = (30, 90, 220) if col == 0 else ((20, 150, 60) if same else (190, 40, 40))
        tile = _thumb(Path(paths[idx]), thumb_size)
        sheet.paste(tile, (x, y))
        draw.rectangle([x, y, x + thumb_size - 1, y + thumb_size - 1], outline=border, width=3)
        lab = labels[idx] if labels else Path(paths[idx]).parent.name
        rank = "Q" if col == 0 else f"#{col}"
        _draw_fit(draw, (x, y + thumb_size + 4), f"{rank} {lab}", border)
        _draw_fit(draw, (x, y + thumb_size + 20), f"sim={sim:.4f}", (60, 60, 60))

    out_file.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_file)


def _metrics(top_idx: np.ndarray, labels: list[str] | None, ks: list[int]) -> dict[str, Any]:
    if not labels:
        return {}
    label_arr = np.asarray(labels)
    same = label_arr[top_idx] == label_arr[:, None]
    out: dict[str, Any] = {"overall": {}, "per_class": {}}
    for k in ks:
        kk = min(k, top_idx.shape[1])
        out["overall"][f"precision_at_{k}"] = float(same[:, :kk].mean())
        out["overall"][f"any_same_at_{k}"] = float(same[:, :kk].any(axis=1).mean())
    for label in sorted(set(labels)):
        mask = label_arr == label
        row: dict[str, float | int] = {"n": int(mask.sum())}
        for k in ks:
            kk = min(k, top_idx.shape[1])
            row[f"precision_at_{k}"] = float(same[mask, :kk].mean())
            row[f"any_same_at_{k}"] = float(same[mask, :kk].any(axis=1).mean())
        out["per_class"][label] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--paths", required=True)
    ap.add_argument("--labels", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--queries-per-class", type=int, default=8)
    ap.add_argument("--max-queries", type=int, default=64)
    ap.add_argument("--thumb-size", type=int, default=160)
    args = ap.parse_args()

    embedding_path = _abs_path(args.embedding)
    paths_path = _abs_path(args.paths)
    labels_path = _abs_path(args.labels) if args.labels else None
    out_dir = _abs_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb = _normalize(np.load(embedding_path))
    paths = [str(_abs_path(p)) for p in _load_json(paths_path)]
    labels = [str(x) for x in _load_json(labels_path)] if labels_path else None
    if emb.shape[0] != len(paths):
        raise ValueError(f"embedding rows {emb.shape[0]} != paths {len(paths)}")
    if labels and len(labels) != len(paths):
        raise ValueError(f"labels {len(labels)} != paths {len(paths)}")

    top_idx, top_sim = _topk_neighbors(emb, int(args.topk))
    ks = sorted(set([1, 3, 5, 10, int(args.topk)]))
    metrics = _metrics(top_idx, labels, ks)

    queries = _choose_queries(emb, labels, int(args.queries_per_class), int(args.max_queries))
    sheets_dir = out_dir / "sheets"
    rows: list[dict[str, Any]] = []
    for q in queries:
        q_label = labels[q] if labels else Path(paths[q]).parent.name
        out_file = sheets_dir / _safe_name(q_label) / f"{_safe_name(q_label)}__query_{q:05d}.png"
        _make_sheet(q, top_idx[q], top_sim[q], paths, labels, out_file, int(args.thumb_size))
        row: dict[str, Any] = {
            "query_index": q,
            "query_label": q_label,
            "query_path": paths[q],
            "sheet_path": str(out_file.resolve()),
        }
        for rank, (ni, sim) in enumerate(zip(top_idx[q], top_sim[q]), 1):
            ni = int(ni)
            row[f"n{rank}_index"] = ni
            row[f"n{rank}_sim"] = float(sim)
            row[f"n{rank}_label"] = labels[ni] if labels else Path(paths[ni]).parent.name
            row[f"n{rank}_path"] = paths[ni]
        rows.append(row)

    report_csv = out_dir / "retrieval_report.csv"
    if rows:
        with open(report_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "embedding": str(embedding_path),
        "paths": str(paths_path),
        "labels": str(labels_path) if labels_path else None,
        "out_dir": str(out_dir),
        "n": int(emb.shape[0]),
        "dim": int(emb.shape[1]),
        "topk": int(args.topk),
        "n_query_sheets": len(rows),
        "metrics": metrics,
        "retrieval_report_csv": str(report_csv.resolve()),
        "sheets_dir": str(sheets_dir.resolve()),
    }
    summary_path = out_dir / "retrieval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
