#!/usr/bin/env python3
"""P1 threshold stability check (team-lead 260726 request, before adopting P1 as the operating point).

질문 3개:
  1. P1 계산에 들어가는 calibration 유사도 표본이 몇 개인가 (t=1..3 클러스터 수 합)?
  2. 그 표본 수에서 P1 과 P2 임계값이 얼마나 다른가 (거의 같으면 P1 이 P2 보다 나을 게 없음)?
  3. calibration window 를 옮겨도(REF@4+calib{1,2,3} → REF@5+calib{2,3,4}) 임계가 안정적인가?

frozen f0 캐시 위에서만 동작 — GPU/재추출 불필요, 순수 CPU HDBSCAN 재클러스터링(t=1..5, 5회)만.

★ 260726: --emb-dir/--sim-name 으로 v1(원본 CrossScratch, 배경 10 class)과 v2(21 class) 양쪽
다 돌릴 수 있게 일반화 — task #27(v1 champion-vs-frozen 재검증)이 v1 쪽 안정성도 요구한다.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from run_temporal_novelty_analysis import (  # noqa: E402
    l2, batch_indices, static_metrics, best_sim_to_set, TEMPORAL_ROOT, EMB_DIR as DEFAULT_EMB_DIR,
)


def calib_distribution(per_t: dict, calib_ts: list[int], ref_t: int) -> list[float]:
    ref = per_t[ref_t]["centroids"]
    sims = []
    for t in calib_ts:
        for c in per_t[t]["centroids"].values():
            sims.append(best_sim_to_set(c, ref))
    return sims


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", default=None, help="embedding cache dir (default: v1 cache)")
    ap.add_argument("--sim-name", default="unknown_novelty_sim",
                     help="dir name under data/pools/temporal/ (default: original v1 CrossScratch sim)")
    ap.add_argument("--arm", default="frozen", choices=["frozen", "champion"],
                     help="which embedding array to stability-check (f0_frozen.npy or f_champion.npy)")
    a = ap.parse_args()
    emb_dir = Path(a.emb_dir) if a.emb_dir else DEFAULT_EMB_DIR
    sim_root = TEMPORAL_ROOT / a.sim_name
    emb_file = "f0_frozen.npy" if a.arm == "frozen" else "f_champion.npy"

    emb = l2(np.load(emb_dir / emb_file))
    meta = json.loads((emb_dir / "paths_index.json").read_text(encoding="utf-8"))
    paths, labels = meta["paths"], meta["labels"]
    path_to_row = {p: i for i, p in enumerate(paths)}
    print(f"[stability-check] emb_dir={emb_dir} sim={a.sim_name} arm={a.arm}", flush=True)

    # cumulative clustering t=1..5 (background only, shared across all novel-class sim variants)
    per_t: dict[int, dict] = {}
    cum_idx: list[int] = []
    for t in range(1, 6):
        idx, _ = batch_indices(sim_root / f"batch_{t:02d}.json", path_to_row)
        cum_idx = cum_idx + idx
        sub_idx = np.array(cum_idx)
        res = static_metrics(emb[sub_idx], [labels[i] for i in sub_idx])
        per_t[t] = res
        print(f"t={t:02d} n={len(sub_idx)} k={res['n_clusters']}", flush=True)

    print("\n=== design A (original): REF@t=4, calib={1,2,3} ===", flush=True)
    simsA = calib_distribution(per_t, [1, 2, 3], 4)
    print(f"n_samples={len(simsA)}", flush=True)
    for p in (1, 2, 5, 10, 20):
        print(f"  P{p} threshold = {np.percentile(simsA, p):.5f}", flush=True)
    print(f"  min={min(simsA):.5f} 2nd-min={sorted(simsA)[1]:.5f} max={max(simsA):.5f}", flush=True)

    print("\n=== design B (shifted +1): REF@t=5, calib={2,3,4} ===", flush=True)
    simsB = calib_distribution(per_t, [2, 3, 4], 5)
    print(f"n_samples={len(simsB)}", flush=True)
    for p in (1, 2, 5, 10, 20):
        print(f"  P{p} threshold = {np.percentile(simsB, p):.5f}", flush=True)
    print(f"  min={min(simsB):.5f} 2nd-min={sorted(simsB)[1]:.5f} max={max(simsB):.5f}", flush=True)

    print("\n=== comparison: |A - B| threshold delta ===", flush=True)
    for p in (1, 2, 5, 10, 20):
        av = np.percentile(simsA, p)
        bv = np.percentile(simsB, p)
        print(f"  P{p}: A={av:.5f} B={bv:.5f} |delta|={abs(av-bv):.5f}", flush=True)

    out = {
        "emb_dir": str(emb_dir), "sim_name": a.sim_name, "arm": a.arm,
        "design_A": {"ref_t": 4, "calib_ts": [1, 2, 3], "n_samples": len(simsA),
                     "values_sorted": sorted(simsA),
                     "percentiles": {p: float(np.percentile(simsA, p)) for p in (1, 2, 5, 10, 20)}},
        "design_B": {"ref_t": 5, "calib_ts": [2, 3, 4], "n_samples": len(simsB),
                     "values_sorted": sorted(simsB),
                     "percentiles": {p: float(np.percentile(simsB, p)) for p in (1, 2, 5, 10, 20)}},
    }
    out_path = emb_dir / f"p1_stability_check_{a.sim_name}_{a.arm}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OUT] {out_path}", flush=True)


if __name__ == "__main__":
    main()
