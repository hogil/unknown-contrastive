#!/usr/bin/env python3
"""data/images/ -> data/images_384/ 384x384 사전 리사이즈 캐시 (DataLoader 병목 제거).

배경: WM-811K 기반 합성 wafer PNG는 6400x6400 palette(mode="P")인데 학습 파이프라인은
어차피 384x384 로 Resize 한다 — 매 epoch 마다 278배(6400^2/384^2) 픽셀을 디코딩하고 버림.
실측: DataLoader warmup 58.2s/epoch (10 img/s), GPU util 10%.

★ 순서 재현 (변경 절대 금지 — _grouping_eval.py / _may_repro_src.py 실제 로딩과 동일):
  Image.open() -> .convert("RGB") -> torchvision.transforms.functional.resize(384x384, BILINEAR)
  (ToTensor/Normalize 는 텐서 단계라 캐시 대상 아님 — 런타임에 그대로 수행)
  두 스크립트 모두 mask_palette_non_grade_to_white() 를 호출하지 않음(grep 확인, 260726) —
  이 캐시도 마스킹 없이 순수 convert+resize만 수행한다. palette masking 을 쓰는 다른
  스크립트(_ssl_methods.py, scripts/train_contrastive.py 등 27개)는 이 캐시를 그대로 쓰면
  안 된다 — RGB로 변환+저장하는 순간 palette index 정보가 사라져 마스킹을 다시 적용할 수 없다.

캐시 사용법: _grouping_eval.py --pool data/images_384/<pool> 처럼 캐시 경로를 그대로 넘기면 됨.
캐시된 이미지가 이미 384x384라 파이프라인의 Resize((384,384), BILINEAR) 는 항등 연산이 되어
결과가 원본 경로와 bit-exact (실측 maxdiff=0.0, 아래 참고). 기존 스크립트 무수정 — 후방호환.

원본(data/images/)은 읽기 전용, 어떤 경우에도 쓰거나 지우지 않음.
재개 가능(skip-existing) — 중단돼도 다시 실행하면 이미 만든 파일은 건너뜀.
기본 workers=2 — GPU 학습이 동시에 돌고 있을 수 있으므로 CPU 를 과점유하지 않기 위함.

사용:
  python _cache_images_384.py                          # 기본 우선순위 3 pool
  python _cache_images_384.py --pools mwm38_clean546    # 특정 pool만
  python _cache_images_384.py --workers 1 --limit 20    # smoke test
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent
IMG = 384
EXTENSIONS = (".png", ".jpg", ".jpeg")

# 우선순위: 검증용(작음) -> GPU 학습이 읽는 중인 anchor(읽기만) -> 큰 eval pool -> 나머지는 --pools 로.
DEFAULT_POOLS = "mwm38_clean546,anchor_avg30_repro,unknown_eval100"


def _resize_one(job: tuple[str, str]) -> str:
    """1장 리사이즈. 실패해도 전체를 죽이지 않고 에러 문자열을 반환."""
    src_str, dst_str = job
    from PIL import Image
    import torchvision.transforms.functional as TF
    from torchvision.transforms import InterpolationMode

    dst = Path(dst_str)
    if dst.exists():
        return "skip"
    try:
        with Image.open(src_str) as im:
            img = im.convert("RGB")
            img = TF.resize(img, [IMG, IMG], interpolation=InterpolationMode.BILINEAR)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp")  # 중단 시 잘린 PNG 방지 — 원자적 rename
        img.save(tmp, "PNG")
        os.replace(tmp, dst)
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"err:{src_str}:{e}"


def collect_jobs(src_root: Path, dst_root: Path, pools: list[str], limit: int) -> list[tuple[str, str]]:
    jobs: list[tuple[str, str]] = []
    for pool in pools:
        pool_src = src_root / pool
        if not pool_src.is_dir():
            print(f"[skip pool] {pool} (not found under {src_root})", flush=True)
            continue
        n_pool = 0
        for f in sorted(pool_src.rglob("*")):
            if f.is_file() and f.suffix.lower() in EXTENSIONS:
                rel = f.relative_to(src_root)
                jobs.append((str(f), str(dst_root / rel)))
                n_pool += 1
        print(f"[pool] {pool}: {n_pool} images", flush=True)
    if limit:
        jobs = jobs[:limit]
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", default=str(REPO / "data" / "images"))
    ap.add_argument("--dst-root", default=str(REPO / "data" / "images_384"))
    ap.add_argument("--pools", default=DEFAULT_POOLS,
                     help="comma-separated pool dir names under --src-root, processed in order given")
    ap.add_argument("--workers", type=int, default=2, help="CPU worker 수 (GPU 학습 굶기지 않게 <=2 권장)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test 용 — 전체 job 중 앞 N개만")
    a = ap.parse_args()

    src_root = Path(a.src_root)
    dst_root = Path(a.dst_root)
    pools = [p.strip() for p in a.pools.split(",") if p.strip()]

    jobs = collect_jobs(src_root, dst_root, pools, a.limit)
    print(f"[plan] {len(jobs)} files total, workers={a.workers} -> {dst_root}", flush=True)

    t0 = time.time()
    done = skip = err = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, res in enumerate(ex.map(_resize_one, jobs, chunksize=8), 1):
            if res == "ok":
                done += 1
            elif res == "skip":
                skip += 1
            else:
                err += 1
                print(f"[ERR] {res}", flush=True)
            if i % 200 == 0 or i == len(jobs):
                el = time.time() - t0
                rate = done / max(1e-6, el)
                eta = (len(jobs) - i) / max(1e-6, rate) if rate > 0 else 0.0
                print(f"[progress] {i}/{len(jobs)} done={done} skip={skip} err={err} "
                      f"| {rate:.1f} img/s | elapsed={el:.0f}s | eta={eta:.0f}s", flush=True)

    print(f"[DONE] done={done} skip={skip} err={err} -> {dst_root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
