#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""이미지 디코드 캐시 — **한 번 디코드해서 모든 arm 이 재사용한다.**

  python deploy/build_cache.py                      # config 의 pool 로
  python deploy/build_cache.py --pool <manifest>    # 직접 지정

왜 필요한가
-----------
병목은 GPU 가 아니라 6400x6400 palette PNG 디코드다 (배치 시간의 60~80%).
그런데 step1 은 arm 이 6개(frozen / z0 / champion / frozen_masked / b4 / b4_backbone)라
**같은 이미지를 6번 다시 디코드한다.** 21,143장이면 126,858번이다. 그동안 H200 은 논다.
step3 스윕은 셀 수만큼 또 반복한다.

그래서 384 로 줄인 uint8 을 한 번만 만들어 두고 전부 거기서 읽는다.
  - **픽셀 단위로 동일하다** (검증: 최대 절대차 0.000e+00).
    임베딩 TF 는 `Resize(384) -> ToTensor -> Normalize` 로 고정이라 Resize 결과를
    uint8 로 저장해도 뒤 두 단계가 결정적이다.
    ⚠ 학습은 RandomResizedCrop 이라 원본이 필요하다 — **학습에는 쓰지 마라.**
      (384 캐시로 학습을 시도했다가 실패한 기록이 있다: crop 이 upsample 이 되어버린다.)
  - 크기: 384*384*3 = 432 KiB/장. 21,143장 = 8.7 GiB.

병렬
----
스레드가 아니라 **프로세스**로 디코드한다. PIL 이 GIL 을 일부 놓긴 하지만 palette->RGB
변환과 텐서 스택은 GIL 을 잡는다. 코어 수만큼 프로세스를 쓰면 그대로 배가 된다.

무결성
------
`cache_384.json` 에 (파일 목록, IMG, palette_mask) 를 적어둔다. 하나라도 다르면
캐시를 무시하고 원본을 읽는다 — 잘못된 캐시로 조용히 틀린 결과를 내는 게 최악이다.
중간에 죽어도 `done` 비트맵이 있어 다시 돌리면 남은 것만 채운다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))

IMG = 384
CACHE_NAME = "cache_384"


def _key(paths, palette_mask: bool) -> str:
    h = hashlib.sha1()
    h.update(f"{IMG}|{int(palette_mask)}|{len(paths)}|".encode())
    for p in paths:
        h.update(str(p).encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


# ── 워커 (top-level 이어야 pickle 된다) ───────────────────────────────────
_W: dict = {}


def _init(npy_path, n, palette_mask):
    from PIL import Image
    from torchvision import transforms as T
    os.environ["UC_PALETTE_MASK"] = "1" if palette_mask else "0"
    mask_fn = None
    if palette_mask:
        try:
            from scripts._common import mask_palette_non_grade_to_white as mask_fn
        except Exception:
            mask_fn = None
    _W["arr"] = np.lib.format.open_memmap(npy_path, mode="r+")
    _W["rs"] = T.Resize((IMG, IMG), interpolation=T.InterpolationMode.BILINEAR)
    _W["Image"] = Image
    _W["mask"] = mask_fn


def _one(job):
    i, path = job
    try:
        im = _W["Image"].open(path)
        if _W["mask"] is not None:
            im = _W["mask"](im)
        _W["arr"][i] = np.asarray(_W["rs"](im.convert("RGB")), dtype=np.uint8)
        return i, ""
    except Exception as e:                       # 한 장 깨져도 전체를 멈추지 않는다
        return i, f"{type(e).__name__}: {e}"


# ── 공개 API ──────────────────────────────────────────────────────────────
def cache_paths(cache_dir) -> tuple[Path, Path, Path]:
    d = Path(cache_dir)
    return d / f"{CACHE_NAME}.npy", d / f"{CACHE_NAME}.json", d / f"{CACHE_NAME}.done"


def load_cache(cache_dir, paths, palette_mask: bool):
    """유효하면 (N,384,384,3) uint8 memmap, 아니면 None.

    파일 목록·IMG·palette_mask 가 하나라도 다르면 **무시한다**. 조용히 틀린 캐시를
    쓰는 것보다 느린 게 낫다.
    """
    npy, meta, done = cache_paths(cache_dir)
    if not (npy.exists() and meta.exists()):
        return None
    try:
        m = json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return None
    if m.get("key") != _key(paths, palette_mask) or m.get("img") != IMG:
        return None
    if not m.get("complete"):
        return None
    arr = np.load(npy, mmap_mode="r")
    if arr.shape != (len(paths), IMG, IMG, 3):
        return None
    return arr


def build(paths, cache_dir, palette_mask: bool, workers: int = 0,
          force: bool = False) -> Path:
    from concurrent.futures import ProcessPoolExecutor
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    npy, meta, done_p = cache_paths(d)
    n = len(paths)
    key = _key(paths, palette_mask)
    workers = workers or max(4, min(64, (os.cpu_count() or 4)))

    prev = {}
    if meta.exists():
        try:
            prev = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
    reusable = (not force and prev.get("key") == key and prev.get("img") == IMG
                and npy.exists())
    if reusable and prev.get("complete"):
        print(f"[cache] 이미 완성돼 있다: {npy}  ({npy.stat().st_size / 2**30:.1f} GiB)")
        return npy

    if not reusable:
        if npy.exists():
            npy.unlink()
        if done_p.exists():
            done_p.unlink()
        np.lib.format.open_memmap(npy, mode="w+", dtype=np.uint8,
                                  shape=(n, IMG, IMG, 3)).flush()

    done = (np.fromfile(done_p, dtype=np.uint8) if done_p.exists()
            else np.zeros(n, np.uint8))
    if len(done) != n:
        done = np.zeros(n, np.uint8)
    todo = [(i, paths[i]) for i in range(n) if not done[i]]
    print(f"[cache] {npy}")
    print(f"[cache] {n:,} 장 중 {len(todo):,} 장 남음   워커 {workers} (cpu {os.cpu_count()})   "
          f"예상 {n * IMG * IMG * 3 / 2**30:.1f} GiB", flush=True)
    if not todo:
        meta.write_text(json.dumps({"key": key, "img": IMG, "n": n,
                                    "palette_mask": palette_mask, "complete": True},
                                   ensure_ascii=False), encoding="utf-8")
        return npy

    t0, ok, bad = time.time(), 0, []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(str(npy), n, palette_mask)) as ex:
        for k, (i, err) in enumerate(ex.map(_one, todo, chunksize=8), 1):
            if err:
                bad.append((paths[i], err))
            else:
                done[i] = 1
                ok += 1
            if k % max(1, len(todo) // 40) == 0 or k == len(todo):
                el = time.time() - t0
                r = k / el if el else 0
                print(f"  [cache] {k:,}/{len(todo):,} ({100*k/len(todo):5.1f}%)  "
                      f"{r:6.1f} img/s  경과 {el/60:.1f}분  "
                      f"남은 {(len(todo)-k)/r/60 if r else 0:.1f}분", flush=True)
                done.tofile(done_p)
    done.tofile(done_p)

    complete = bool(done.all())
    meta.write_text(json.dumps({"key": key, "img": IMG, "n": n,
                                "palette_mask": palette_mask, "complete": complete,
                                "n_failed": len(bad)}, ensure_ascii=False), encoding="utf-8")
    el = time.time() - t0
    print(f"[cache] 완료 {ok:,} 장 / 실패 {len(bad)} 장   {el/60:.1f}분   "
          f"{ok/el if el else 0:.1f} img/s", flush=True)
    for p, e in bad[:10]:
        print(f"    [실패] {p} -> {e}")
    if not complete:
        print("[cache] ★ 미완성이다 — 다시 돌리면 남은 것만 채운다. "
              "미완성 캐시는 사용되지 않는다(원본을 읽는다).", flush=True)
    return npy


def main() -> int:
    from config import Cluster, Paths, Runtime  # noqa: F401
    from _site_common import rel

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="", help="manifest json. 없으면 step0 결과를 쓴다")
    ap.add_argument("--cache-dir", default="", help="기본 <OUT_ROOT>/_cache")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="기존 캐시를 버리고 다시 만든다")
    a = ap.parse_args()

    pool = a.pool
    if not pool:
        s0 = rel(Paths.OUT_ROOT) / "step0_result.json"
        if not s0.exists():
            print("[FATAL] step0 결과가 없다. 먼저:  python deploy/step0_prepare.py",
                  file=sys.stderr)
            return 2
        pool = json.loads(s0.read_text(encoding="utf-8"))["manifest"]

    from scripts._common import load_pool_manifest
    entries = sorted(load_pool_manifest(Path(str(rel(pool)))), key=lambda e: e[0])
    paths = [str(p) for p, _ in entries]

    cache_dir = a.cache_dir or (rel(Paths.OUT_ROOT) / "_cache")
    pm = str(os.environ.get("UC_PALETTE_MASK", "0")).lower() in ("1", "true", "yes", "on")
    print(f"[cache] pool={rel(pool)}  n={len(paths):,}  palette_mask={pm}")
    npy = build(paths, cache_dir, pm, a.workers, a.force)
    print(f"\n[OUT] {Path(npy).parent}")
    return 0


if __name__ == "__main__":          # ★ Windows spawn 에 필수
    raise SystemExit(main())
