"""6400×6400 wafer fail-bit PNG → {384, 512, 1024} BICUBIC RGB cache.

Cache 한 번 만들고 cnn_train.py에 --data-dir 로 지정해 학습 시 GPU starvation 해소.
- 출력 컨벤션: D:/project/data/wm-811k/unknown_{size}_bicubic/<class>/*.png
- 다운샘플: PIL.Image.BICUBIC (사용자 결정, 시각 비교 결과 BILINEAR/LANCZOS/BOX와 차이 미미)
- multiprocessing pool로 워커당 1장씩 처리, --workers 12 default

사용:
    python _make_cache.py                                         # 1024 default
    python _make_cache.py --sizes 384,512,1024 --workers 12       # 다중 size 한 번에
    python _make_cache.py --sizes 1024 --interp bicubic           # 1024만

이미 존재하는 출력 PNG는 skip (overwrite 안 함). 강제 재생성은 폴더 삭제 후 실행.
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

from PIL import Image

# ===== Defaults =====
DATA_ROOT = Path("D:/project/data/wm-811k/unknown")
CACHE_PARENT = Path("D:/project/data/wm-811k")
SIZES_DEFAULT = (1024,)

INTERP_MAP = {
    "bicubic": Image.BICUBIC,
    "bilinear": Image.BILINEAR,
    "lanczos": Image.LANCZOS,
    "box": Image.BOX,
    "nearest": Image.NEAREST,
}


def cache_dir_for(size: int, interp: str) -> Path:
    return CACHE_PARENT / f"unknown_{size}_{interp}"


def _process_one(args):
    """1 image 처리. (src_path, dst_path, size, interp_const) tuple."""
    src_path, dst_path, size, interp_const = args
    try:
        if dst_path.exists():
            return ("skip", str(src_path))
        img = Image.open(src_path).convert("RGB")
        out = img.resize((size, size), interp_const)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        out.save(dst_path, optimize=True)
        return ("ok", str(src_path))
    except Exception as e:
        return ("err", f"{src_path}: {e}")


def build_one_size(size: int, interp_name: str, workers: int):
    interp_const = INTERP_MAP[interp_name]
    cache_dir = cache_dir_for(size, interp_name)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # task list
    tasks = []
    n_total = 0
    for cls_dir in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir()):
        for src in sorted(cls_dir.glob("*.png")):
            dst = cache_dir / cls_dir.name / src.name
            tasks.append((src, dst, size, interp_const))
            n_total += 1

    print(f"[{size}/{interp_name}] tasks: {n_total}, output: {cache_dir}", flush=True)
    if n_total == 0:
        print(f"[{size}/{interp_name}] no input PNG found at {DATA_ROOT}", flush=True)
        return

    t0 = time.time()
    n_ok = n_skip = n_err = 0
    err_msgs = []
    with Pool(processes=workers) as pool:
        for i, (status, msg) in enumerate(pool.imap_unordered(_process_one, tasks, chunksize=4), 1):
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_err += 1
                err_msgs.append(msg)
            if i % 200 == 0 or i == n_total:
                rate = i / max(0.1, time.time() - t0)
                eta = (n_total - i) / max(0.1, rate)
                print(f"  [{size}] {i}/{n_total}  ok={n_ok} skip={n_skip} err={n_err}  "
                      f"rate={rate:.1f}/s  ETA={eta:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"[{size}/{interp_name}] done in {elapsed:.1f}s — ok={n_ok} skip={n_skip} err={n_err}",
          flush=True)
    if err_msgs:
        print(f"[{size}/{interp_name}] errors (first 5):", flush=True)
        for m in err_msgs[:5]:
            print(f"  {m}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default=",".join(str(s) for s in SIZES_DEFAULT),
                   help="comma-separated sizes, e.g. 384,512,1024 (default 1024)")
    p.add_argument("--interp", default="bicubic", choices=list(INTERP_MAP),
                   help="interpolation (default bicubic)")
    p.add_argument("--workers", type=int, default=min(12, max(1, cpu_count() - 2)))
    args = p.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    print(f"sizes={sizes} interp={args.interp} workers={args.workers}", flush=True)
    print(f"input root: {DATA_ROOT}", flush=True)

    if not DATA_ROOT.exists():
        print(f"[!] input root not found: {DATA_ROOT}", file=sys.stderr)
        sys.exit(2)

    for sz in sizes:
        build_one_size(sz, args.interp, args.workers)
    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
