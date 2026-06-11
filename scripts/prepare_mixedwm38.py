#!/usr/bin/env python3
"""MixedWM38 (38,015장, 52x52, 38 class: Normal + 8 single + 29 mixed) → palette PNG 렌더.

입력: data/raw/mixedwm38/Wafer_Map_Datasets.npz  (arr_0 maps, arr_1 8-dim multi-hot)
라벨 비트 순서 (Description.pdf 검증): [C, D, EL, ER, L, NF, S, R]
렌더: prepare_wm811k_subset 과 동일 정책 — defect(2,3)→palette idx6(red), 나머지 white,
      NEAREST 정수 확대 + 512 canvas 중앙 배치. (값3 은 Random 맵 미세 아티팩트 → defect 취급)

출력: data/images/mixedwm38/all/<class>/<idx>.png + manifest.csv
class 명: Normal / 단일 "C","D","EL","ER","L","NF","S","R" / 혼합 "C+EL" 등 (비트 순 join)
"""
from __future__ import annotations
import argparse, csv, importlib.util, sys
from collections import Counter
from pathlib import Path
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
REPO = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("pws", REPO / "scripts/prepare_wm811k_subset.py")
pws = importlib.util.module_from_spec(spec); sys.modules["pws"] = pws
spec.loader.exec_module(pws)

BITS = ["C", "D", "EL", "ER", "L", "NF", "S", "R"]
NPZ = REPO / "data/raw/mixedwm38/Wafer_Map_Datasets.npz"
OUT = REPO / "data/images/mixedwm38"


def cls_name(row) -> str:
    on = [BITS[i] for i, v in enumerate(row) if v]
    return "+".join(on) if on else "Normal"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--limit-per-class", type=int, default=0, help="0=전부")
    args = ap.parse_args()
    d = np.load(NPZ)
    X, Y = d["arr_0"], d["arr_1"]
    names = [cls_name(r) for r in Y]
    print(f"[mixedwm38] {len(X)} maps, {len(set(names))} classes", flush=True)
    done = Counter()
    rows = []
    for i, (wm, name) in enumerate(zip(X, names)):
        if args.limit_per_class and done[name] >= args.limit_per_class:
            continue
        wm = np.asarray(wm).copy()
        wm[wm == 3] = 2  # Random 맵 아티팩트 → defect 취급
        img = pws._render_wafer_map(wm, args.image_size, show_normal_dies=False)
        cdir = OUT / "all" / name
        cdir.mkdir(parents=True, exist_ok=True)
        fp = cdir / f"mwm38_{i:05d}.png"
        img.save(fp)
        done[name] += 1
        rows.append((i, name, str(fp.relative_to(REPO))))
        if (len(rows)) % 5000 == 0:
            print(f"  rendered {len(rows)}", flush=True)
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["row_index", "class", "path"])
        w.writerows(rows)
    print(f"[done] {len(rows)}장, class 분포:", flush=True)
    for c, n in sorted(done.items()):
        print(f"  {c}: {n}", flush=True)
    print(f"[OUT] {OUT.resolve()}", flush=True)


if __name__ == "__main__":
    main()
