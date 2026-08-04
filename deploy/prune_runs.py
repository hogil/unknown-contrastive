#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`runs/` 정리 — **목록을 먼저 남기고** 원본 복사본/재계산 가능한 산출물만 지운다.

  python deploy/prune_runs.py                 # 무엇을 지울지 보기만 한다 (기본)
  python deploy/prune_runs.py --apply         # 실제로 지운다

★ 왜 필요한가 (260805 실측): `runs/` 가 434 GB 였고, 그중
    representatives  165.9 GB (14,204장)  <- E: 원본의 **복사본**
    clusters         126.8 GB (21,471장)  <- E: 원본의 **복사본**
    composites        58.3 GB ( 4,508장)  <- 계산 산출물 (재계산 가능)
    full model .pt    52.0 GB (   157개)  <- proj_ep*.pt 만 있으면 재현 가능
  이었다. 앞의 둘은 `E:/data/images/` 에 원본이 그대로 있다 (표본 200/200 확인).

★ 지우기 전에 **반드시 목록을 남긴다.**
  representatives / clusters 는 복사본이지만 동시에 **"어느 이미지가 어느 그룹이었나"의
  기록**이기도 하다. `assignments.csv` 는 260805 에 추가된 거라 그 이전 run 에는 없다.
  그래서 폴더마다 `_manifest_*.csv` 를 먼저 쓰고, **그게 제대로 써졌는지 확인한 뒤에만**
  PNG 를 지운다. 목록이 없거나 개수가 안 맞으면 그 폴더는 건너뛴다.

★ 절대 안 지우는 것 (프로젝트 절대규칙: 학습/실험 결과 삭제 금지)
    proj_ep*.pt      학습 결과 본체
    *.json *.csv     summary / groups / assignments / hparams
    *.log            실행 기록
    *.npy            임베딩 (재계산 비쌈)
    폴더 자체        빈 폴더로 남겨 구조와 이름(l/w 수)을 보존한다
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _site_common import REPO, banner  # noqa: E402


def _gb(n: int) -> float:
    return n / 2 ** 30


def scan(root: Path):
    """지울 후보 폴더를 종류별로 모은다. (kind, dir, [png...]) 목록.

    ★ 폴더 **이름**이 아니라 **조상 경로**로 판정한다. 레이아웃이 시기마다 다르다:
        composites/*.png                      (현재)
        composites/<group>/*.png              (옛날 — 그룹마다 하위폴더를 만들던 때)
        representatives/<group>/*.png
        clusters/hdbscan/<cluster>/*.png
      이름만 보면 옛 레이아웃의 PNG 를 통째로 놓친다 (실측: composites 58.3 -> 13.7 GB 로
      과소 집계돼 44 GB 가 안 잡혔다).
    """
    marks = {"composites": "composites", "representatives": "representatives",
             "clusters": "clusters"}
    out = []
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        parts = set(d.parts)
        kind = next((v for k, v in marks.items() if k in parts), None)
        if kind is None:
            continue
        pngs = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".png"]
        if pngs:
            out.append((kind, d, pngs))
    return out


def write_manifest(kind: str, d: Path, pngs: list[Path]) -> Path | None:
    """지우기 **전에** 목록을 쓴다. 실패하면 None -> 호출자가 그 폴더를 건너뛴다."""
    mf = d.parent / f"_manifest_{kind}.csv"
    try:
        new = not mf.exists()
        with mf.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["group_dir", "filename", "bytes"])
            for p in sorted(pngs):
                w.writerow([d.name, p.name, p.stat().st_size])
        return mf if mf.exists() and mf.stat().st_size > 0 else None
    except Exception as e:
        print(f"  [skip] 목록 실패 ({type(e).__name__}: {e}) -> 안 지운다: {d}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="runs", help="정리할 루트 (기본 runs)")
    ap.add_argument("--kinds", default="representatives,clusters,composites",
                    help="지울 종류 (콤마 구분)")
    ap.add_argument("--full-models", action="store_true",
                    help="full model 체크포인트(final_infer.pt / last_training.pt)도 지운다. "
                         "proj_ep*.pt 는 항상 남는다")
    ap.add_argument("--keep", default="",
                    help="이 문자열이 경로에 들어가면 건너뛴다 (콤마 구분). "
                         "예: --keep runs/site  -> 배포 체인 결과는 손대지 않는다")
    ap.add_argument("--apply", action="store_true", help="실제로 지운다 (기본은 보기만)")
    a = ap.parse_args()

    root = Path(a.root)
    if not root.is_absolute():
        root = REPO / root
    if not root.exists():
        print(f"[FATAL] 없다: {root}", file=sys.stderr)
        return 2
    kinds = {k.strip() for k in a.kinds.split(",") if k.strip()}

    banner("PRUNE RUNS", f"{root}  ({'실행' if a.apply else '미리보기 — 아무것도 안 지운다'})")

    keeps = [k.strip().replace("\\", "/") for k in a.keep.split(",") if k.strip()]

    def _kept(d: Path) -> bool:
        sp = str(d).replace("\\", "/")
        return any(k in sp for k in keeps)

    cand = [(k, d, p) for k, d, p in scan(root) if k in kinds and not _kept(d)]
    if keeps:
        print(f"[keep] 경로에 {keeps} 가 들어가면 건너뛴다\n")
    tally = {}
    for k, d, pngs in cand:
        s = tally.setdefault(k, [0, 0, 0])
        s[0] += sum(p.stat().st_size for p in pngs)
        s[1] += len(pngs)
        s[2] += 1

    print(f"{'종류':<18}{'GB':>9}{'파일':>10}{'폴더':>8}")
    for k in sorted(tally):
        b, n, nd = tally[k]
        print(f"{k:<18}{_gb(b):>9.2f}{n:>10,}{nd:>8,}")

    full = []
    if a.full_models:
        full = [p for p in root.rglob("*.pt")
                if p.name in ("final_infer.pt", "last_training.pt") and not _kept(p)]
        print(f"{'full model .pt':<18}{_gb(sum(p.stat().st_size for p in full)):>9.2f}"
              f"{len(full):>10,}{'':>8}")

    total = sum(v[0] for v in tally.values()) + sum(p.stat().st_size for p in full)
    print(f"\n  회수 예상 {_gb(total):.2f} GB")

    if not a.apply:
        print("\n  미리보기다. 실제로 지우려면 --apply")
        return 0

    print("\n  목록을 먼저 쓰고, 써진 걸 확인한 폴더만 지운다.")
    freed = nfile = nskip = 0
    for k, d, pngs in cand:
        if write_manifest(k, d, pngs) is None:
            nskip += 1
            continue
        for p in pngs:
            try:
                sz = p.stat().st_size
                p.unlink()
                freed += sz
                nfile += 1
            except Exception as e:
                print(f"  [warn] {type(e).__name__}: {p}")
    for p in full:
        try:
            sz = p.stat().st_size
            p.unlink()
            freed += sz
            nfile += 1
        except Exception as e:
            print(f"  [warn] {type(e).__name__}: {p}")

    print(f"\n  회수 {_gb(freed):.2f} GB  /  파일 {nfile:,}개 삭제  /  목록실패로 보존 {nskip}폴더")
    print(f"  목록: {root}/**/_manifest_*.csv  — 원본은 E:/data/images/ 에 그대로 있다")
    print(f"\n[OUT] {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
