#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 0 — 사내 이미지 폴더 -> manifest + pool 기하 리포트 + 권장 다이얼.

  python deploy/step0_prepare.py

여기서 정한 다이얼(mcs/ms)을 이후 step1~5 가 전부 물려 쓴다.

★ 불량 종수(k)는 입력하지 않는다. 모르는 게 전제이고, HDBSCAN 을 쓰는 이유가 k-free 라서다.
  다이얼은 **"몇 장 이상 뭉쳐야 하나의 그룹으로 볼 것인가"**(Cluster.MCS)로 정한다 —
  이건 클래스 수가 아니라 **보고 가치가 있는 최소 그룹 크기**라 k 를 몰라도 답할 수 있다.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _site_common import (add_path, REPO, banner,  # noqa: E402
                          die, env, rel, save_result,
                          run_root, show_config, show_images, cache_dir_for)


from config import Cluster, Paths, Runtime  # noqa: E402


class Config:
    """★ 설정은 deploy/config.py 한 곳에서 관리한다. 여기는 참조만."""
    IMAGE_ROOT = Paths.IMAGE_ROOT
    POOL_MANIFEST = Paths.POOL_MANIFEST
    EXTS = Paths.EXTS
    OUT_BASE = Paths.OUT_ROOT          # runs/site (캐시·latest.txt 가 여기)
    OUT_ROOT = Paths.OUT_ROOT          # main() 에서 타임스탬프 run 으로 교체된다
    BACKBONE = Paths.BACKBONE
    DEVICE = Runtime.DEVICE
    CACHE = Runtime.CACHE
    CACHE_DIR = Runtime.CACHE_DIR
    DECODE_WORKERS = Runtime.DECODE_WORKERS
    BATCH = Runtime.BATCH
    MCS = Cluster.MCS
    MS = Cluster.MS


def resolve_dial(n: int, pool_path: str):
    """다이얼은 config 값 그대로다. ★ k 를 쓰지 않는다.

    `Cluster.MCS` / `Cluster.MS` 를 직접 읽는다 — 중간 유도 규칙 없음.
    (260728 이전에는 MIN_GROUP_SIZE 에서 mcs 를 유도하고 AUTO_DIAL 로 안정성 스캔을
     하는 경로가 있었으나, "자동"이 두 뜻으로 쓰여 혼란만 커서 걷어냈다.)
    """
    mcs, ms = int(Config.MCS), int(Config.MS)
    if mcs < 2 or ms < 1:
        die(f"다이얼이 잘못됐다: mcs={mcs} ms={ms}. "
            f"SITE_MCS>=2, SITE_MS>=1 이어야 한다.")
    if ms > mcs:
        print(f"[dial] ★ 경고: ms({ms}) > mcs({mcs}). 보통 ms 는 mcs 의 1/4 쯤이다.", flush=True)
    print(f"[dial] mcs={mcs} ms={ms} (config 값 그대로)", flush=True)
    return mcs, ms, []



def build_decode_cache(pool_path, out_root) -> str:
    """★ 디코드 캐시를 여기서 만들어 둔다 — step1~5 가 전부 재사용한다.

    안 만들면 step1 의 arm 6개가 **같은 이미지를 6번 다시 디코드**한다.
    실측(4060 Ti): 캐시 없음 7.6 img/s(GPU 21~38%) -> 캐시 19.7 img/s(GPU 99.5%).
    GPU 가 빠를수록 격차가 커진다. 결과는 비트 단위로 동일하다.
    """
    if not Config.CACHE:
        print("[cache] SITE_CACHE=0 -> 캐시 없이 매번 디코드한다 (느리다)\n")
        return ""
    from build_cache import build
    from scripts._common import load_pool_manifest
    # ★ 캐시는 타임스탬프 run 이 아니라 **base** 에 둔다 — run 마다 8.7GiB 를
    #   다시 만들 이유가 없다. 유효성은 (파일목록, palette_mask) 해시로 검사하므로
    #   run 이 달라도 안전하게 공유된다.
    cdir = cache_dir_for(Config.OUT_BASE, Config.CACHE_DIR)
    paths = [str(p) for p, _ in sorted(load_pool_manifest(Path(str(rel(pool_path)))),
                                       key=lambda e: e[0])]
    pm = str(os.environ.get("UC_PALETTE_MASK", "0")).lower() in ("1", "true", "yes", "on")
    try:
        build(paths, cdir, pm, int(Config.DECODE_WORKERS))
    except Exception as e:
        print(f"[cache] 생성 실패({type(e).__name__}: {e}) -> 캐시 없이 진행한다", flush=True)
        return ""
    return cdir



def main() -> int:
    banner("STEP 0", "manifest 생성 + pool 기하 -> 권장 다이얼")
    Config.OUT_ROOT = str(run_root(Config.OUT_BASE, create=True))
    cfg = show_config(Config)
    show_images(Config.IMAGE_ROOT, Config.POOL_MANIFEST, Config.EXTS)

    # ── 기존 manifest 사용 경로 ──────────────────────────────────────────
    if str(Config.POOL_MANIFEST).strip():
        mp = rel(Config.POOL_MANIFEST)
        if not mp.exists():
            die(f"SITE_POOL_MANIFEST 가 없다: {mp}")
        man = json.loads(mp.read_text(encoding="utf-8"))
        n = len(man.get("files", []))
        if n == 0:
            die(f"manifest 에 files 가 없다: {mp}")
        root = Path(man["root"])
        out_root = rel(Config.OUT_ROOT)
        # 캐시를 먼저 만든다 (이후 step 들이 전부 재사용).
        cdir = build_decode_cache(mp, out_root)
        mcs, ms, _ = resolve_dial(n, str(mp))
        print(f"\n[pool] 기존 manifest 사용: {mp}")
        print(f"[pool] n = {n:,} 장   root = {root}")
        save_result(out_root, "step0", {
            "cache_dir": cdir,
            "n_images": n,
            "manifest": str(mp.relative_to(REPO)) if mp.is_relative_to(REPO) else str(mp),
            "image_root": root.as_posix(), "subdirs": [],
            "dial": {"mcs": mcs, "ms": ms, "method": str(Cluster.METHOD), "eps": Cluster.EPS},
            "config": cfg})
        print("\n다음:  python deploy/step1_zeroshot.py")
        print(f"\n[OUT] {out_root}")
        return 0

    root = rel(Config.IMAGE_ROOT)
    if not root.exists():
        die(f"image_root 가 없다: {root}\n"
            f"  SITE_IMAGE_ROOT 환경변수로 지정하거나 Config.IMAGE_ROOT default 를 고쳐라.\n"
            f"  (프로젝트 루트 기준 상대경로 권장 — 예: data/site_images)")

    exts = {("." + e.strip().lstrip(".")).lower() for e in str(Config.EXTS).split(",") if e.strip()}
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)
    n = len(files)
    if n == 0:
        die(f"이미지가 하나도 없다: {root} (확장자 {sorted(exts)})")

    # 하위 폴더명이 있으면 참고용으로만 기록. 사내 무라벨 pool 이면 label=null 로 둔다.
    subdirs = sorted({p.parent.relative_to(root).as_posix() for p in files
                      if p.parent != root})
    entries = []
    for p in files:
        relp = p.relative_to(root).as_posix()
        entries.append({"path": relp, "label": None})

    manifest = {
        "root": root.as_posix(),
        "note": "site pool (unlabeled). label=null 은 의도된 것 — 부모폴더명으로 오염시키지 마라.",
        "files": entries,
    }
    out_root = rel(Config.OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)
    mpath = out_root / "site_pool.json"
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    cdir = build_decode_cache(mpath, out_root)      # ★ 다이얼 결정 전에 (위 주석 참조)
    mcs, ms, _ = resolve_dial(n, str(mpath))
    print(f"\n[pool] n = {n:,} 장   |   하위폴더 {len(subdirs)}개"
          + (f"  (예: {', '.join(subdirs[:4])}{' ...' if len(subdirs) > 4 else ''})" if subdirs else ""))

    payload = {
        "n_images": n,
        "manifest": str(mpath.relative_to(REPO)) if mpath.is_relative_to(REPO) else str(mpath),
        "image_root": root.as_posix(), "subdirs": subdirs[:50],
        "dial": {"mcs": mcs, "ms": ms, "method": str(Cluster.METHOD), "eps": Cluster.EPS},
        "config": cfg,
    }
    payload["cache_dir"] = cdir
    save_result(out_root, "step0", payload)

    print("\n다음:  python deploy/step1_zeroshot.py")
    print("       (step0 결과의 다이얼을 자동으로 물려 쓴다)")
    print(f"\n[OUT] {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
