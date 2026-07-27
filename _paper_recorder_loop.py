"""paper-recorder 60-min polling loop. 8회 반복.

매 iter:
1. glob tier1_*_n500.json — 신규 (mtime 60min 이내)
2. 신규 발견 → PERFORMANCE_HISTORY.md Step 2 표 TBD replace + ITERATIONS.md append
3. 0 새거 → polling 진행 1줄 ITERATIONS.md append
4. 60min sleep

8회 후 _loop_recorder_summary.md 작성.

Tier 1+2 official metric only. P1 capture > P2 noise > P3 Comp > P4 Hom. append-only.
"""

import json
import glob
import time
import os
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(r"D:/project/unknown-contrastive")
PERF_MD = ROOT / "docs" / "paper" / "PERFORMANCE_HISTORY.md"
ITER_MD = ROOT / "docs" / "paper" / "ITERATIONS.md"
SUMMARY_MD = ROOT / "_loop_recorder_summary.md"
STATE = ROOT / "_paper_recorder_state.json"

INTERVAL_SEC = 60 * 60  # 60 min
N_ITERS = 8

CELL_PARAM_LOOKUP = {
    # cell_name -> (use_local, local_w, use_queue, neg_sim, neco_w)
    "B0_n500": ("F", "0",   "F", "1.0",  "0"),
    "B1_n500": ("T", "0.5", "F", "1.0",  "0"),
    "B3_n500": ("T", "0.5", "T", "1.0",  "0"),
    "B4_n500": ("T", "0.5", "T", "0.72", "0"),
    "B5_n500": ("T", "0.5", "T", "0.72", "0.2"),
    "NEW_n500":("F", "0",   "T", "0.72", "0.2"),
}


def ts():
    return datetime.now().strftime("%y%m%d %H:%M:%S")


def find_new_tier1(since_sec: float):
    """tier1_*_n500.json 중 mtime 60min 이내 + 미처리 분."""
    pat1 = str(ROOT / "outputs_contrastive_*" / "tier1_*_n500.json")
    pat2 = str(ROOT / "outputs_contrastive_*" / "tier1_*.json")  # broader
    cands = glob.glob(pat1) + glob.glob(pat2)
    cutoff = time.time() - since_sec
    new = []
    for p in cands:
        try:
            mt = os.path.getmtime(p)
            if mt >= cutoff:
                new.append((p, mt))
        except OSError:
            pass
    new.sort(key=lambda x: x[1])
    return new


def load_state():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"processed": [], "iter_results": []}


def save_state(st):
    STATE.write_text(json.dumps(st, indent=2), encoding="utf-8")


def parse_tier1(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:
        return None
    # robust extract Tier 1+2 official metrics
    def g(*keys):
        for k in keys:
            if k in d:
                return d[k]
        # nested search
        for v in d.values():
            if isinstance(v, dict):
                for k in keys:
                    if k in v:
                        return v[k]
        return None
    out = {
        "path": path,
        "AMI": g("AMI", "ami", "adjusted_mutual_info"),
        "ARI": g("ARI", "ari", "adjusted_rand_index"),
        "Completeness": g("Completeness", "completeness"),
        "Homogeneity": g("Homogeneity", "homogeneity"),
        "noise_pct": g("noise_pct", "noise_percentage", "noise"),
        "n_cluster": g("n_cluster", "n_clusters", "num_clusters"),
        "capture_rate": g("class_capture_rate", "capture_rate", "capture"),
    }
    return out


def detect_cell_name(path):
    """run.log or path 에서 cell name 추출 시도."""
    base = os.path.basename(path)  # e.g., tier1_B0.json or tier1_B0_n500.json
    m = re.search(r"tier1_([A-Z0-9]+(?:_n500)?(?:_seed\d+)?)\.json", base)
    if m:
        cand = m.group(1)
        if not cand.endswith("_n500") and cand in {"B0","B1","B3","B4","B5","NEW"}:
            return cand + "_n500"
        return cand
    # try run_info.json or run.log in same dir
    rd = Path(path).parent
    log = rd / "run.log"
    if log.exists():
        try:
            txt = log.read_text(encoding="utf-8", errors="ignore")[:5000]
            for cell in CELL_PARAM_LOOKUP:
                if cell in txt or cell.replace("_n500","") in txt:
                    return cell
        except Exception:
            pass
    return None


def update_perf_md(cell, parsed):
    """Step 2 표의 cell row 의 TBD → 값 replace."""
    if not PERF_MD.exists():
        return False
    txt = PERF_MD.read_text(encoding="utf-8")
    # 정확한 row pattern: "| B0_n500 | F | 0 | F | 1.0 | 0 | TBD | TBD | TBD | TBD | 🔄 학습 중 |"
    if cell not in CELL_PARAM_LOOKUP:
        return False
    ul, lw, uq, ns, nc = CELL_PARAM_LOOKUP[cell]
    # find line starting with "| <cell> |"
    lines = txt.splitlines()
    new_lines = []
    replaced = False
    ami = parsed.get("AMI")
    ari = parsed.get("ARI")
    noise = parsed.get("noise_pct")
    ncl = parsed.get("n_cluster")

    def fmt(v, prec=3):
        if v is None:
            return "TBD"
        try:
            if isinstance(v, str):
                return v
            return f"{float(v):.{prec}f}"
        except Exception:
            return str(v)

    def fmt_pct(v):
        if v is None:
            return "TBD"
        try:
            f = float(v)
            # if already in percent (>1), keep; else *100
            if f <= 1.0:
                f *= 100
            return f"{f:.2f}%"
        except Exception:
            return str(v)

    for line in lines:
        if line.startswith(f"| {cell} |") and "TBD" in line:
            new_line = f"| {cell} | {ul} | {lw} | {uq} | {ns} | {nc} | {fmt(ami)} | {fmt(ari)} | {fmt_pct(noise)} | {ncl if ncl is not None else 'TBD'} | done {ts()} |"
            new_lines.append(new_line)
            replaced = True
        else:
            new_lines.append(line)
    if replaced:
        PERF_MD.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    return replaced


def append_iter_log(text):
    """ITERATIONS.md 끝에 append."""
    with open(ITER_MD, "a", encoding="utf-8") as f:
        f.write(text)


def run_one_iter(iter_idx, state):
    new_files = find_new_tier1(INTERVAL_SEC + 60)  # 60 min window + small slack
    processed_set = set(state["processed"])
    fresh = [(p, m) for p, m in new_files if p not in processed_set]
    iter_result = {"iter": iter_idx, "ts": ts(), "new_cells": []}
    if fresh:
        for path, mt in fresh:
            parsed = parse_tier1(path)
            if parsed is None:
                continue
            cell = detect_cell_name(path)
            entry = {
                "cell": cell, "path": path,
                "AMI": parsed.get("AMI"), "ARI": parsed.get("ARI"),
                "noise_pct": parsed.get("noise_pct"), "n_cluster": parsed.get("n_cluster"),
                "Comp": parsed.get("Completeness"), "Hom": parsed.get("Homogeneity"),
                "capture": parsed.get("capture_rate"),
            }
            iter_result["new_cells"].append(entry)
            updated = False
            if cell and cell in CELL_PARAM_LOOKUP:
                updated = update_perf_md(cell, parsed)
            # append ITERATIONS.md
            msg = (
                f"\n### {ts()} paper-recorder polling loop iter {iter_idx}/{N_ITERS} — new tier1 detected\n"
                f"- file: `{os.path.relpath(path, ROOT).replace(os.sep, '/')}`\n"
                f"- cell: `{cell}` | "
                f"AMI={entry['AMI']} ARI={entry['ARI']} noise={entry['noise_pct']} "
                f"n_cluster={entry['n_cluster']} Comp={entry['Comp']} Hom={entry['Hom']} capture={entry['capture']}\n"
                f"- PERFORMANCE_HISTORY.md Step 2 row updated: {updated}\n"
            )
            append_iter_log(msg)
            state["processed"].append(path)
    else:
        # 0 새거 → polling 진행 1줄 append
        # quick chain status
        try:
            outs = sorted(glob.glob(str(ROOT / "outputs_contrastive_260517_*")))
            latest = outs[-1] if outs else None
        except Exception:
            latest = None
        msg = (
            f"\n### {ts()} paper-recorder polling loop iter {iter_idx}/{N_ITERS}\n"
            f"- 신규 tier1_*_n500.json 0 files (60min window) — Step 2 표 변경 없음\n"
            f"- latest output dir: `{os.path.basename(latest) if latest else 'none'}`\n"
        )
        append_iter_log(msg)
    state["iter_results"].append(iter_result)
    save_state(state)
    return iter_result


def write_summary(state):
    lines = ["# paper-recorder polling loop summary", ""]
    lines.append(f"- spawned: {state.get('spawned_at','?')}")
    lines.append(f"- finished: {ts()}")
    lines.append(f"- total iters: {len(state['iter_results'])}")
    new_total = sum(len(r["new_cells"]) for r in state["iter_results"])
    lines.append(f"- new cells recorded: {new_total}")
    lines.append("")
    lines.append("## Per-iter")
    for r in state["iter_results"]:
        lines.append(f"- iter {r['iter']}/{N_ITERS} @ {r['ts']} — new={len(r['new_cells'])}")
        for c in r["new_cells"]:
            lines.append(
                f"  - {c.get('cell')} AMI={c.get('AMI')} ARI={c.get('ARI')} "
                f"noise={c.get('noise_pct')} n_cluster={c.get('n_cluster')}"
            )
    SUMMARY_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    st = load_state()
    if "spawned_at" not in st:
        st["spawned_at"] = ts()
        save_state(st)
    # respawn-aware: continue indefinitely (사용자 명시 학습 끝까지 polling)
    done = len(st.get("iter_results", []))
    start_i = done + 1
    if start_i > N_ITERS:
        # iters_done >= N_ITERS — reset to keep polling for new tier1 results
        st["iter_results"] = []
        save_state(st)
        start_i = 1
    st["respawned_at"] = ts()
    save_state(st)
    print(f"[paper-recorder loop] spawned at {st['spawned_at']}, respawned at {st['respawned_at']}, "
          f"resuming iter {start_i}/{N_ITERS}, interval {INTERVAL_SEC}s")
    for i in range(start_i, N_ITERS + 1):
        print(f"\n[paper-recorder loop] iter {i}/{N_ITERS} @ {ts()}")
        res = run_one_iter(i, st)
        print(f"  new cells: {len(res['new_cells'])}")
        for c in res["new_cells"]:
            print(f"    {c.get('cell')} AMI={c.get('AMI')} ARI={c.get('ARI')} noise={c.get('noise_pct')}")
        if i < N_ITERS:
            print(f"  sleeping {INTERVAL_SEC}s (60min) ...")
            time.sleep(INTERVAL_SEC)
    write_summary(st)
    print(f"\n[paper-recorder loop] DONE — summary: {SUMMARY_MD}")


if __name__ == "__main__":
    main()
