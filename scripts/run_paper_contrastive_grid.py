#!/usr/bin/env python3
"""Run paper-style contrastive grids over prepared open-set splits.

This driver intentionally runs many small controlled recipes instead of a single
hand-picked trial. It is resumable: existing final epoch embeddings are skipped.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_GUARD = REPO / "result_grouping" / "paper_contrastive_supervisor_260617" / "STOP_RESOURCE_GUARD.txt"


def recipes(level: str, epochs: int) -> list[dict]:
    base = [
        ("simclr_t004", "--method simclr --temp 0.04", 4),
        ("simclr_t005", "--method simclr --temp 0.05", 4),
        ("simclr_t006", "--method simclr --temp 0.06", 4),
        ("local005", "--method simclr --temp 0.05 --local 0.05", 4),
        ("local015", "--method simclr --temp 0.05 --local 0.15", 4),
        ("local030", "--method simclr --temp 0.05 --local 0.30", 4),
        ("local050", "--method simclr --temp 0.05 --local 0.50", 4),
        ("q1024", "--method simclr --temp 0.05 --use-queue --queue-size 1024", 4),
        ("q2048", "--method simclr --temp 0.05 --use-queue --queue-size 2048", 4),
        ("q4096", "--method simclr --temp 0.05 --use-queue --queue-size 4096", 4),
        ("q8192", "--method simclr --temp 0.05 --use-queue --queue-size 8192", 4),
    ]
    full = list(base)
    for local in (0.15, 0.30, 0.50):
        for q in (2048, 4096):
            for temp in (0.04, 0.05, 0.06):
                tag = f"l{str(local).replace('.', '')}_q{q}_t{str(temp).replace('.', '')}"
                full.append((tag, f"--method simclr --temp {temp} --local {local} --use-queue --queue-size {q}", 4))
    for neco in (0.05, 0.10, 0.20, 0.50):
        full.append((f"local015_neco{str(neco).replace('.', '')}",
                     f"--method simclr --temp 0.05 --local 0.15 --neco {neco}", 4))
    for koleo in (0.05, 0.10, 0.20):
        full.append((f"local015_koleo{str(koleo).replace('.', '')}",
                     f"--method simclr --temp 0.05 --local 0.15 --koleo {koleo}", 4))
    for m in (0.90, 0.95, 0.99):
        full.append((f"moco_m{str(m).replace('.', '')}",
                     f"--method moco --temp 0.05 --local 0.15 --use-queue --queue-size 4096 --m {m}", 4))
    for thr in (0.70, 0.72, 0.80):
        full.append((f"q4096_ignore{str(thr).replace('.', '')}",
                     f"--method simclr --temp 0.05 --local 0.15 --use-queue --queue-size 4096 --ignore {thr}", 4))
    for nv in (0.90, 0.95):
        full.append((f"q4096_nv{str(nv).replace('.', '')}",
                     f"--method simclr --temp 0.05 --local 0.15 --use-queue --queue-size 4096 --nv-filter {nv}", 4))
    full.extend([
        ("q4096_softnce_ls002_top32",
         "--method simclr --temp 0.05 --local 0.15 --use-queue --queue-size 4096 --ls 0.02 --ls-topk 32", 4),
        ("q4096_sce03",
         "--method simclr --temp 0.05 --local 0.15 --use-queue --queue-size 4096 --sce 0.3", 4),
        ("barlow_local015", "--method barlow --local 0.15", 4),
        ("vicreg_local015", "--method vicreg --local 0.15", 4),
    ])
    rows = base if level == "quick" else full
    return [{"tag": tag, "flags": flags, "batch": batch, "epochs": epochs} for tag, flags, batch in rows]


def run(cmd: list[str], log: Path, env: dict[str, str]) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8", errors="replace") as f:
        f.write("\n$ " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT, text=True)
        return p.wait()


def check_guard(guard_file: Path | None) -> None:
    if guard_file and guard_file.exists():
        msg = guard_file.read_text(encoding="utf-8", errors="replace").strip()
        raise SystemExit(f"resource guard active; refusing to start next recipe: {guard_file} {msg}")


def score_condition(cond_root: Path, eval_dir: str, exclude: list[str], log: Path, env: dict[str, str]) -> None:
    emb_dir = cond_root / "embeddings"
    embs = sorted(str(p) for p in emb_dir.glob("*.npy"))
    frozen = sorted(str(p) for p in (cond_root / "embeddings").glob("*frozen*.npy"))
    # _field_pipeline also writes to the same embeddings dir when root=cond_root.
    all_embs = sorted(set(embs + frozen))
    if not all_embs:
        return
    out_csv = cond_root / "scores_all.csv"
    cmd = [
        sys.executable, str(REPO / "_score_umapfree.py"), *all_embs,
        "--skip-umap",
        "--pool", eval_dir,
        "--exclude-classes", ",".join(exclude),
        "--out-csv", str(out_csv),
    ]
    run(cmd, log, env)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="E:/data/images/paper_contrastive_splits_260617/manifest.json")
    ap.add_argument("--out-root", default="result_grouping/paper_contrastive_grid_260617")
    ap.add_argument("--split-glob", default="wm_train_normal_random_full_cap2149")
    ap.add_argument("--palette-modes", default="grade_only")
    ap.add_argument("--level", choices=["quick", "full"], default="quick")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--max-runs", type=int, default=0,
                    help="0 means no cap. Counts SSL recipes, not frozen baselines.")
    ap.add_argument("--skip-frozen", action="store_true")
    ap.add_argument("--score-every", action="store_true",
                    help="Score all saved embeddings after every recipe. Default scores once at the end.")
    ap.add_argument("--guard-file", default=str(DEFAULT_GUARD),
                    help="If this file exists, stop before launching the next long operation. Use empty string to disable.")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO / manifest_path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = REPO / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    guard_file = Path(args.guard_file) if args.guard_file else None
    if guard_file and not guard_file.is_absolute():
        guard_file = REPO / guard_file

    selected_splits = [
        s for s in manifest["splits"]
        if fnmatch.fnmatch(s["name"], args.split_glob)
    ]
    if not selected_splits:
        raise SystemExit(f"no split matched: {args.split_glob}")

    palette_modes = [x.strip() for x in args.palette_modes.split(",") if x.strip()]
    grid = recipes(args.level, args.epochs)
    started = 0
    for split in selected_splits:
        for palette in palette_modes:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["HF_HUB_OFFLINE"] = env.get("HF_HUB_OFFLINE", "1")
            env["UC_PALETTE_MODE"] = palette
            cond_root = out_root / split["name"] / palette
            emb_dir = cond_root / "embeddings"
            log = cond_root / "driver.log"
            emb_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "split": split,
                "palette_mode": palette,
                "level": args.level,
                "epochs": args.epochs,
                "recipes": grid,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            (cond_root / "condition.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

            if not args.skip_frozen and not (emb_dir / "DINOv3_frozen.npy").exists():
                check_guard(guard_file)
                cmd = [
                    sys.executable, str(REPO / "_field_pipeline.py"),
                    "--stage", "frozen",
                    "--pool", split["eval"],
                    "--root", str(cond_root),
                ]
                run(cmd, log, env)

            for rec in grid:
                check_guard(guard_file)
                if args.max_runs and started >= args.max_runs:
                    score_condition(cond_root, split["eval"], split["exclude_classes"], log, env)
                    print(f"[STOP] max-runs reached. partial condition: {cond_root.resolve()}")
                    return
                final_emb = emb_dir / f"{rec['tag']}_ep{args.epochs}.npy"
                if final_emb.exists():
                    continue
                cmd = [
                    sys.executable, str(REPO / "_ssl_methods.py"),
                    *rec["flags"].split(),
                    "--epochs", str(rec["epochs"]),
                    "--batch", str(rec["batch"]),
                    "--train-dir", split["train"],
                    "--eval-dir", split["eval"],
                    "--out-dir", str(emb_dir),
                    "--tag", rec["tag"],
                    "--palette-mode", palette,
                    "--ckpt-every", "50",
                ]
                rc = run(cmd, log, env)
                started += 1
                if rc != 0:
                    print(f"[WARN] recipe failed rc={rc}: {rec['tag']} ({cond_root})")
                if args.score_every:
                    score_condition(cond_root, split["eval"], split["exclude_classes"], log, env)

            score_condition(cond_root, split["eval"], split["exclude_classes"], log, env)
            print(f"[DONE] {cond_root.resolve()}")


if __name__ == "__main__":
    main()
