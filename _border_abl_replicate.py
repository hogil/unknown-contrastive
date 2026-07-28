#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""palette 전처리 ablation 을 **서로 겹치지 않는 subset 여러 개**로 재현한다.

  python _border_abl_replicate.py [--subsets 5] [--per-class 20]

pool 1개 결과는 우연일 수 있다. unknown 43 class 에서 disjoint slice 를 잘라
같은 비교를 반복하고, **subset 마다 승자가 유지되는지**를 본다.
  subset s 는 class 마다 [s*per_class : (s+1)*per_class] 를 쓴다 -> 겹침 0.

조건 (한 번에 하나씩만 벗김):
  ① raw       MASK=0                  전부
  ② unify     MASK=1 grade_only       0~7 + 경계10
  ③ noborder  MASK=1 grade_noborder   0~7 만
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
ROOT = "E:/data/images/unknown"
BB = "weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth"
OUT = REPO / "runs" / "_border_repl"
CONDS = [("raw", "0", "grade_only"), ("keepbg", "1", "grade_bg"),
         ("unify", "1", "grade_only"), ("noborder", "1", "grade_noborder")]
DIALS = [6, 10]
# ★ Thick-Edge_fork 는 50장뿐이라 disjoint subset 5개(=100장)를 못 채운다.
#   빼면 남은 42 class 가 전부 200장 이상이라 subset 마다 class 구성이 **동일**해진다.
#   빼지 않으면 subset 3,4 에서만 그 class 가 사라져 P1 분모가 달라지고 비교가 깨진다.
EXCLUDE = {"Thick-Edge_fork"}


def make_pool(s: int, per: int) -> Path:
    files = []
    for d in sorted(Path(ROOT).iterdir()):
        if not d.is_dir() or d.name in EXCLUDE:
            continue
        pngs = sorted(d.glob("*.png"))[s * per:(s + 1) * per]
        files += [{"path": f"{d.name}/{p.name}", "label": d.name} for p in pngs]
    p = REPO / "data" / "pools" / f"_border_repl_s{s}.json"
    p.write_text(json.dumps({"root": ROOT, "files": files}), encoding="utf-8")
    return p


def run(cmd, env):
    e = dict(os.environ); e.update(env); e["PYTHONIOENCODING"] = "utf-8"
    return subprocess.call(cmd, cwd=str(REPO), env=e,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", type=int, default=5)
    ap.add_argument("--per-class", type=int, default=20)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    for s in range(a.subsets):
        pool = make_pool(s, a.per_class)
        n = len(json.loads(pool.read_text(encoding="utf-8"))["files"])
        print(f"\n===== subset {s}  ({n}장, class 당 {a.per_class}, "
              f"슬라이스 [{s*a.per_class}:{(s+1)*a.per_class}]) =====", flush=True)
        for name, mask, mode in CONDS:
            env = {"UC_PALETTE_MASK": mask, "UC_PALETTE_MODE": mode}
            cdir = OUT / f"_cache_s{s}_{name}"
            run([sys.executable, "deploy/build_cache.py", "--pool", str(pool),
                 "--cache-dir", str(cdir)], env)
            for m in DIALS:
                o = OUT / f"s{s}_m{m}_{name}"
                if (o / "offline_summary.json").exists():
                    print(f"  s{s} mcs{m:<3} {name:<9} skip (이미 있음)", flush=True)
                    continue
                rc = run([sys.executable, "grouping_deploy.py", "--pool", str(pool),
                          "--backbone", BB, "--cache", str(cdir), "--out", str(o),
                          "--mcs", str(m), "--ms", "3", "--reassign", "nearest_q90",
                          "--offline-eval", "--no-composites", "--batch", "32",
                          "--reps", "1"], env)
                ok = (o / "offline_summary.json").exists()
                print(f"  s{s} mcs{m:<3} {name:<9} rc={rc} {'OK' if ok else '★산출없음'}",
                      flush=True)
    print("\n[REPLICATE_DONE]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
