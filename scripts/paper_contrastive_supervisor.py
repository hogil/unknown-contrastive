#!/usr/bin/env python3
"""Continuous monitor/analyzer/planner loop for paper contrastive grids."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = [
    REPO / "result_grouping" / "paper_contrastive_grid_260617",
    REPO / "result_grouping" / "paper_contrastive_unknown_grid_260617",
]
LIVE_CSV = "scores_live.csv"
SEEN_JSON = ".scores_live_seen.json"


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_text(cmd: list[str], timeout: int = 30) -> str:
    try:
        p = subprocess.run(cmd, cwd=REPO, text=True, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=timeout)
        return p.stdout.strip()
    except Exception as exc:
        return f"[ERR] {exc}"


def log_line(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", errors="replace") as f:
        f.write(f"[{now()}] {text}\n")


def find_conditions(roots: list[Path]) -> list[Path]:
    conds = []
    for root in roots:
        if not root.exists():
            continue
        for meta in root.rglob("condition.json"):
            conds.append(meta.parent)
    return sorted(set(conds))


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def append_csv(dst: Path, src: Path) -> int:
    if not src.exists():
        return 0
    with src.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        fields = f.readline()
    if not rows:
        return 0
    write_header = not dst.exists()
    with dst.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def score_new_embeddings(cond: Path, max_new: int, quiet_age_sec: int, log: Path) -> int:
    meta_path = cond / "condition.json"
    if not meta_path.exists():
        return 0
    meta = load_json(meta_path, {})
    split = meta.get("split", {})
    eval_dir = split.get("eval")
    exclude = split.get("exclude_classes", [])
    if not eval_dir:
        return 0
    emb_dir = cond / "embeddings"
    if not emb_dir.exists():
        return 0

    seen_path = cond / SEEN_JSON
    seen = set(load_json(seen_path, []))
    candidates = []
    cutoff = time.time() - quiet_age_sec
    for p in sorted(emb_dir.glob("*.npy"), key=lambda x: x.stat().st_mtime):
        sp = str(p.resolve())
        if sp in seen:
            continue
        try:
            if p.stat().st_mtime > cutoff:
                continue
        except OSError:
            continue
        candidates.append(p)
    if not candidates:
        return 0
    selected = candidates[:max_new]
    tmp = cond / f".scores_live_tmp_{int(time.time())}.csv"
    cmd = [
        sys.executable, str(REPO / "_score_umapfree.py"),
        *[str(p) for p in selected],
        "--skip-umap",
        "--pool", str(eval_dir),
        "--exclude-classes", ",".join(exclude),
        "--out-csv", str(tmp),
    ]
    log_line(log, f"score {len(selected)} embeddings: {cond}")
    rc = subprocess.call(cmd, cwd=REPO)
    if rc != 0:
        log_line(log, f"score failed rc={rc}: {cond}")
        return 0
    rows = append_csv(cond / LIVE_CSV, tmp)
    try:
        tmp.unlink()
    except OSError:
        pass
    seen.update(str(p.resolve()) for p in selected)
    seen_path.write_text(json.dumps(sorted(seen), indent=2), encoding="utf-8")
    return rows


def read_rows(cond: Path) -> list[dict[str, str]]:
    rows = []
    for name in (LIVE_CSV, "scores_all.csv"):
        p = cond / name
        if p.exists():
            with p.open("r", encoding="utf-8", newline="") as f:
                rows.extend(csv.DictReader(f))
    uniq = {}
    for r in rows:
        key = (r.get("embedding", ""), r.get("method", ""))
        uniq[key] = r
    return list(uniq.values())


def f(row: dict[str, str], key: str, default: float = -1e9) -> float:
    try:
        value = row.get(key, "")
        if value == "":
            return default
        return float(value)
    except Exception:
        return default


def best_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str] | None]:
    def filt(prefix: str):
        return [r for r in rows if r.get("method", "").startswith(prefix)]

    def cap_ok(r):
        return f(r, "P1_capture", 0.0) >= 1.0

    out: dict[str, dict[str, str] | None] = {}
    for method in ("finch_p2", "louvain"):
        pool = filt(method)
        pool_cap = [r for r in pool if cap_ok(r)] or pool
        out[f"{method}_ari"] = max(pool_cap, key=lambda r: f(r, "ARI"), default=None)
        out[f"{method}_frag"] = min(pool_cap, key=lambda r: f(r, "fragment_ratio", 1e9), default=None)
    hdb = [r for r in filt("hdbscan") if f(r, "P2_noise_pct", 100.0) <= 40.0]
    out["hdbscan_ari_low_noise"] = max(hdb, key=lambda r: f(r, "ARI"), default=None)
    return out


def row_summary(row: dict[str, str] | None) -> str:
    if not row:
        return "-"
    return (
        f"{row.get('embedding_name', Path(row.get('embedding', '')).stem)} / {row.get('method', '')} "
        f"| P1={row.get('P1_capture', '')} P2={row.get('P2_noise_pct', '')} "
        f"P3={row.get('P3_completeness', '')} P4={row.get('P4_homogeneity', '')} "
        f"ARI={row.get('ARI', '')} Sil={row.get('Sil', '')} "
        f"k={row.get('k_total', '')}/{row.get('k_classes', '')}/{row.get('k_noise', '')} "
        f"frag={row.get('fragment_ratio', '')}"
    )


def resource_snapshot() -> dict[str, str]:
    disk = shutil.disk_usage(REPO.anchor or "D:\\")
    gpu = run_text([
        "nvidia-smi",
        "--query-gpu=memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    procs = run_text([
        "powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.CommandLine -match '_paper_contrastive|run_paper_contrastive_grid|_ssl_methods|_score_umapfree|export_best_groupings' } | "
        "Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"
    ], timeout=20)
    return {
        "disk_free_gb": f"{disk.free / (1024 ** 3):.1f}",
        "disk_total_gb": f"{disk.total / (1024 ** 3):.1f}",
        "gpu": gpu,
        "processes_json": procs,
    }


def is_wm_done(root: Path) -> bool:
    log = root / "wm_quick_driver_260617.log"
    return log.exists() and "DONE WM quick grid" in log.read_text(encoding="utf-8", errors="replace")


def is_unknown_running_or_done(root: Path) -> bool:
    log = root / "unknown_quick_driver_260617.log"
    running = "_paper_contrastive_unknown_quick" in run_text([
        "powershell", "-NoProfile", "-Command",
        "(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '_paper_contrastive_unknown_quick' }).Count"
    ], timeout=10)
    done = log.exists() and "DONE unknown quick grid" in log.read_text(encoding="utf-8", errors="replace")
    return running or done


def maybe_launch_unknown(auto_launch: bool, log: Path) -> None:
    wm_root = DEFAULT_ROOTS[0]
    unk_root = DEFAULT_ROOTS[1]
    if not auto_launch or not is_wm_done(wm_root):
        return
    proc_count = run_text([
        "powershell", "-NoProfile", "-Command",
        "(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match '_paper_contrastive_unknown_quick|run_paper_contrastive_grid.py' }).Count"
    ], timeout=10).strip()
    if proc_count and proc_count != "0":
        return
    log_path = unk_root / "unknown_quick_driver_260617.log"
    if log_path.exists() and "DONE unknown quick grid" in log_path.read_text(encoding="utf-8", errors="replace"):
        return
    ps1 = REPO / "_paper_contrastive_unknown_quick_260617.ps1"
    if not ps1.exists():
        return
    cmd = [
        "powershell", "-NoProfile", "-Command",
        f"Start-Process -WindowStyle Hidden -FilePath pwsh -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File','{ps1}')"
    ]
    subprocess.call(cmd, cwd=REPO)
    log_line(log, f"auto-launched unknown quick grid: {ps1}")


def write_summary(root: Path, conds: list[Path], resources: dict[str, str]) -> None:
    lines = [
        "# Paper Contrastive Supervisor",
        "",
        f"- updated: {now()}",
        f"- disk: {resources['disk_free_gb']} GB free / {resources['disk_total_gb']} GB total",
        f"- gpu: `{resources['gpu']}`",
        "",
        "## Conditions",
        "",
    ]
    for cond in conds:
        rows = read_rows(cond)
        if not rows:
            continue
        best = best_rows(rows)
        lines.extend([
            f"### {cond}",
            f"- rows: {len(rows)}",
            f"- finch_p2 ARI: {row_summary(best['finch_p2_ari'])}",
            f"- finch_p2 fragment: {row_summary(best['finch_p2_frag'])}",
            f"- louvain ARI: {row_summary(best['louvain_ari'])}",
            f"- louvain fragment: {row_summary(best['louvain_frag'])}",
            f"- hdbscan low-noise ARI: {row_summary(best['hdbscan_ari_low_noise'])}",
            "",
        ])
    lines.extend([
        "## Decision Rule",
        "",
        "- Continue current WM quick grid while GPU use is healthy and disk free is above 30 GB.",
        "- After a split finishes, export best groupings by finch_p2 ARI, finch_p2 fragment ratio, and louvain ARI.",
        "- After WM quick completes, run unknown quick grid with grade_only vs grade_bg only if no long job is active.",
        "- Do not launch full grid until quick grid identifies the best split and stable technique family.",
        "",
    ])
    out = root / "paper_contrastive_supervisor_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=[str(p) for p in DEFAULT_ROOTS])
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--max-new-per-condition", type=int, default=4)
    ap.add_argument("--quiet-age-sec", type=int, default=60)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--auto-launch", action="store_true")
    args = ap.parse_args()

    roots = [Path(x) if Path(x).is_absolute() else REPO / x for x in args.roots]
    supervisor_root = REPO / "result_grouping" / "paper_contrastive_supervisor_260617"
    supervisor_root.mkdir(parents=True, exist_ok=True)
    log = supervisor_root / "supervisor.log"
    log_line(log, f"start interval={args.interval}s roots={roots}")

    while True:
        resources = resource_snapshot()
        free_gb = float(resources["disk_free_gb"])
        if free_gb < 30.0:
            log_line(log, f"BLOCK low disk: {free_gb:.1f} GB")
        else:
            conds = find_conditions(roots)
            total_rows = 0
            for cond in conds:
                total_rows += score_new_embeddings(cond, args.max_new_per_condition, args.quiet_age_sec, log)
            maybe_launch_unknown(args.auto_launch, log)
            write_summary(supervisor_root, conds, resources)
            subprocess.call([
                sys.executable,
                str(REPO / "scripts" / "make_paper_contrastive_tables.py"),
                "--out-dir",
                str(supervisor_root),
            ], cwd=REPO)
            log_line(log, f"tick scored_rows={total_rows} conds={len(conds)} disk_free_gb={resources['disk_free_gb']} gpu={resources['gpu']}")
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
