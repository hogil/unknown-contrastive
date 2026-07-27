#!/usr/bin/env python3
"""Run small local x queue x temp combo grids selected from univariate tables."""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = REPO / "result_grouping" / "paper_contrastive_supervisor_260617" / "paper_contrastive_univariate_ari.csv"
DEFAULT_OUT = REPO / "result_grouping" / "paper_contrastive_grid_260617"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def flt(row: dict[str, str], key: str, default: float = -1e9) -> float:
    try:
        v = row.get(key, "")
        if v == "":
            return default
        return float(v)
    except Exception:
        return default


def top(rows: list[dict[str, str]], split: str, palette: str, family: str, n: int) -> list[dict[str, str]]:
    vals = [
        r for r in rows
        if r.get("split") == split
        and r.get("palette") == palette
        and r.get("family") == family
        and r.get("method", "").startswith("finch_p2")
        and flt(r, "P1_capture", 0.0) >= 1.0
    ]
    vals.sort(key=lambda r: flt(r, "ARI"), reverse=True)
    seen = set()
    out = []
    for row in vals:
        recipe = row["recipe"]
        if recipe in seen:
            continue
        out.append(row)
        seen.add(recipe)
        if len(out) >= n:
            break
    return out


def temp_from_recipe(recipe: str) -> float:
    m = re.search(r"t(\d+)$", recipe)
    if not m:
        raise ValueError(recipe)
    return int(m.group(1)) / 100.0


def local_from_recipe(recipe: str) -> float:
    m = re.search(r"local(\d+)$", recipe)
    if not m:
        raise ValueError(recipe)
    return int(m.group(1)) / 1000.0


def queue_from_recipe(recipe: str) -> int:
    m = re.search(r"q(\d+)$", recipe)
    if not m:
        raise ValueError(recipe)
    return int(m.group(1))


def run(cmd: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8", errors="replace") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.Popen(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, text=True)
        rc = p.wait()
    if rc != 0:
        raise SystemExit(f"command failed rc={rc}: {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(DEFAULT_TABLE))
    ap.add_argument("--condition-root", required=True,
                    help="Condition root containing condition.json and embeddings.")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--top-temp", type=int, default=1)
    ap.add_argument("--top-local", type=int, default=2)
    ap.add_argument("--top-queue", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    table = Path(args.table)
    if not table.is_absolute():
        table = REPO / table
    cond = Path(args.condition_root)
    if not cond.is_absolute():
        cond = REPO / cond
    meta = json.loads((cond / "condition.json").read_text(encoding="utf-8"))
    split = meta["split"]["name"]
    palette = meta["palette_mode"]
    rows = read_rows(table)

    temps = top(rows, split, palette, "01_temp", args.top_temp)
    locals_ = top(rows, split, palette, "02_local", args.top_local)
    queues = top(rows, split, palette, "03_queue", args.top_queue)
    if not temps or not locals_ or not queues:
        print(f"[NOT_READY] split={split} palette={palette} temps={len(temps)} locals={len(locals_)} queues={len(queues)}")
        return

    emb_dir = cond / "embeddings"
    log = cond / "combo_driver.log"
    recipes = []
    for tr in temps:
        temp = temp_from_recipe(tr["recipe"])
        for lr in locals_:
            local = local_from_recipe(lr["recipe"])
            for qr in queues:
                q = queue_from_recipe(qr["recipe"])
                tag = f"combo_l{str(local).replace('.', '')}_q{q}_t{str(temp).replace('.', '')}"
                recipes.append((tag, local, q, temp))

    for tag, local, q, temp in recipes:
        final = emb_dir / f"{tag}_ep{args.epochs}.npy"
        print(f"[COMBO] {tag}: temp={temp} local={local} queue={q} final={final}")
        if args.dry_run or final.exists():
            continue
        cmd = [
            sys.executable, str(REPO / "_ssl_methods.py"),
            "--method", "simclr",
            "--temp", str(temp),
            "--local", str(local),
            "--use-queue",
            "--queue-size", str(q),
            "--epochs", str(args.epochs),
            "--batch", "4",
            "--train-dir", meta["split"]["train"],
            "--eval-dir", meta["split"]["eval"],
            "--out-dir", str(emb_dir),
            "--tag", tag,
            "--palette-mode", palette,
            "--ckpt-every", "50",
        ]
        run(cmd, log)

    score_csv = cond / "scores_combo.csv"
    embs = sorted(str(p) for p in emb_dir.glob("combo_*_ep*.npy"))
    if embs and not args.dry_run:
        run([
            sys.executable, str(REPO / "_score_umapfree.py"),
            *embs,
            "--skip-umap",
            "--pool", meta["split"]["eval"],
            "--exclude-classes", ",".join(meta["split"]["exclude_classes"]),
            "--out-csv", str(score_csv),
        ], log)
        print(f"[CSV] {score_csv.resolve()}")


if __name__ == "__main__":
    main()
