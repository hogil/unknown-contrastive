#!/usr/bin/env python3
"""data/pools/*.json manifest -> <root>_<size>/ 사전 리사이즈 캐시 (DataLoader 병목 제거).

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

★ 260726 갱신: 캐시는 data/pools/*.json manifest(root + files[].path) 를 그대로 읽는다.
  구 버전은 `data/images/<pool>/` 디렉토리가 존재한다고 가정했는데, 260726 cleanup 사고로
  그 디렉토리들이 삭제되고 실제 pool은 전부 manifest 기반 subset(마스터 E:/data/images/unknown
  등의 상대경로 목록)으로 재구성됐다 — 디렉토리 가정이 stale. manifest 모드가 기본이다.
  캐시 대상 = manifest 에 **등장하는 파일만**(마스터 전체 21k장이 아님).
  캐시 경로 = <manifest root>_<size> (예: --size 384 -> E:/data/images/unknown_384).
  같은 root 를 공유하는 여러 manifest 는 dst 기준으로 dedup(중복 리사이즈 없음).
  각 pool마다 data/pools/<name>_<size>.json 을 파생(소스 manifest 와 같은 위치) — root만
  캐시 경로로 바뀌고 files는 원본과 동일 — 기존 --pool 소비 코드(scripts/_common.py::resolve_pool)는
  무수정으로 캐시를 선택함. 구 디렉토리 글롭 모드(--src-root/--pools)는 후방호환으로 유지.

★★ 260726 paired 학습 검증 결과 (384px 는 FAIL — 학습에 쓰지 말 것):
  eval 경로(Resize 항등연산)는 bit-exact 지만, 학습 경로(RandomResizedCrop(scale=0.94-1.0))는
  384px 캐시 위에서 "다운샘플"이 아니라 "업샘플"이 돼(0.94~1.0 crop of 384 -> 다시 384로 확대)
  원본(6400px 위 다운샘플)과 다른 blur artifact 를 만든다. 5-epoch paired 검증에서 P1/ARI/Hom
  이 잡음폭 밖으로 벌어짐(orig/cache Rule-C 선택 epoch 도 불일치) — 상세 runs/clean546/eval_cache_equiv_*.json.
  안전 조건(team-lead 260726 계산): RandomResizedCrop min crop scale=0.94 -> 선형 배율 sqrt(0.94)=0.9695.
  cache_size * 0.9695 >= 384 인 학습해상도 384 를 다시 만족하려면 cache_size >= 396. --size 448 로 재검증 중
  (docs 는 team-lead 판단 참고, 이 스크립트는 크기와 무관하게 진행 — 최종 판단은 paired 재검증으로).
  --size < 396 이면 derived manifest 에 eval_only=True 강제 표시. >=396 은 아직 "학습 검증됨"이 아니라
  "학습 미검증"으로 표시 — paired 검증 통과 전까지 학습에 쓰지 말 것.

캐시 사용법: --pool data/pools/<name>_<size>.json 처럼 파생 manifest 를 그대로 넘기면 됨.
캐시된 이미지가 이미 <size>x<size>라 파이프라인의 Resize((384,384), BILINEAR) 는(caches size==384일 때)
항등 연산이 되어 eval 경로 결과가 원본 경로와 bit-exact (실측 maxdiff=0.0). 기존 스크립트 무수정 — 후방호환.

원본(E:/data/images/...)은 읽기 전용, 어떤 경우에도 쓰거나 지우지 않음. 하드링크/심볼릭링크
사용 안 함 — 매 파일 실제로 디코딩 후 리사이즈해 새 PNG로 저장.
재개 가능(skip-existing) — 중단돼도 다시 실행하면 이미 만든 파일은 건너뜀.
기본 workers=2 — GPU 학습이 동시에 돌고 있을 수 있으므로 CPU 를 과점유하지 않기 위함.

사용:
  python _cache_images_384.py                                   # 기본 우선순위 4 manifest, size=384(eval 전용)
  python _cache_images_384.py --manifests anchor_avg30_repro     # 특정 manifest만 (이름 or 경로)
  python _cache_images_384.py --size 448 --manifests anchor_avg30_repro  # 학습 재검증용 448 캐시
  python _cache_images_384.py --workers 4 --limit 20             # smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent
EXTENSIONS = (".png", ".jpg", ".jpeg")
POOLS_DIR = REPO / "data" / "pools"
MIN_TRAIN_SAFE_SIZE = 396  # sqrt(0.94)*size >= 384 근사 (RandomResizedCrop scale=(0.94,1.0) 안전선, team-lead 260726)

# 우선순위 (team-lead 260726): 작은 검증용 anchor 먼저(GPU 학습이 읽는 중, 읽기만) ->
# 큰 eval pool -> train/holdout. 나머지는 --manifests 로.
DEFAULT_MANIFESTS = (
    "anchor_avg30_repro,unknown_eval100,"
    "unknown_train_defectaware_260710,unknown_holdout_100_260713"
)


def _resize_one(job: tuple[str, str, int]) -> str:
    """1장 리사이즈. 실패해도 전체를 죽이지 않고 에러 문자열을 반환."""
    src_str, dst_str, size = job
    from PIL import Image
    import torchvision.transforms.functional as TF
    from torchvision.transforms import InterpolationMode

    dst = Path(dst_str)
    if dst.exists():
        return "skip"
    try:
        with Image.open(src_str) as im:
            img = im.convert("RGB")
            img = TF.resize(img, [size, size], interpolation=InterpolationMode.BILINEAR)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp")  # 중단 시 잘린 PNG 방지 — 원자적 rename
        img.save(tmp, "PNG")
        os.replace(tmp, dst)
        return "ok"
    except Exception as e:  # noqa: BLE001
        return f"err:{src_str}:{e}"


def _resolve_root(root: str) -> Path:
    """manifest 의 "root" 문자열을 절대경로로. 절대경로는 그대로, relative 는 REPO 기준
    (scripts/_common.py::resolve_path 와 동일 규칙)."""
    p = Path(str(root).replace("\\", "/"))
    return p if p.is_absolute() else (REPO / p)


def _cache_root_for(src_root: Path, size: int) -> Path:
    """<root> -> <root>_<size> (sibling, 같은 부모 디렉토리 아래)."""
    return src_root.parent / f"{src_root.name}_{size}"


def _manifest_path(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.suffix.lower() == ".json" and p.is_file():
        return p
    cand = POOLS_DIR / f"{name_or_path}.json"
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"pool manifest not found: {name_or_path} (tried {p}, {cand})")


def collect_jobs_from_manifests(names: list[str], size: int, limit: int):
    """각 manifest 를 읽어 (src, dst, size) job 을 모으고 dst 기준으로 dedup.

    같은 root 를 공유하는 여러 pool(예: anchor_avg30_repro 와 unknown_eval100 모두
    E:/data/images/unknown 하위 subset)은 겹치는 파일을 한 번만 리사이즈한다.
    """
    seen_dst: set[str] = set()
    jobs: list[tuple[str, str, int]] = []
    manifest_info: list[dict] = []  # 이후 <name>_<size>.json 파생용
    for name in names:
        mpath = _manifest_path(name)
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        src_root = _resolve_root(manifest["root"])
        dst_root = _cache_root_for(src_root, size)
        n_pool = n_new = 0
        for item in manifest["files"]:
            rel = Path(item["path"])
            if rel.suffix.lower() not in EXTENSIONS:
                continue
            n_pool += 1
            dst = dst_root / rel
            key = str(dst)
            if key in seen_dst:
                continue
            seen_dst.add(key)
            jobs.append((str(src_root / rel), key, size))
            n_new += 1
        print(f"[manifest] {mpath.name}: {n_pool} files (root={src_root}) -> {n_new} new job(s), "
              f"cache_root={dst_root}", flush=True)
        manifest_info.append({"name": mpath.stem, "manifest_path": mpath,
                               "manifest": manifest, "dst_root": dst_root})
    if limit:
        jobs = jobs[:limit]
    return jobs, manifest_info


def write_derived_manifests(manifest_info: list[dict], size: int) -> None:
    """각 pool 에 대해 root 만 캐시 경로로 바꾼 <name>_<size>.json 을 씀 (files/labels 는 원본과 동일)."""
    train_safe = size >= MIN_TRAIN_SAFE_SIZE
    for info in manifest_info:
        derived = dict(info["manifest"])
        derived["root"] = info["dst_root"].as_posix()
        derived["cache_source_manifest"] = str(info["manifest_path"])
        if not train_safe:
            # ★ 260726 실측(paired 검증 FAIL, size=384): RandomResizedCrop(scale 0.94-1.0)이 이 해상도의
            # 캐시 위에서는 다운샘플이 아니라 업샘플이 돼 학습 표현이 갈린다(P1/ARI/Hom 잡음폭 밖).
            # eval(Resize 항등연산)에서만 bit-exact 검증됨 — 이 manifest 로 학습 절대 금지.
            derived["eval_only"] = True
            note = (f"{size}x{size} 사전 리사이즈 캐시 (_cache_images_384.py). "
                    "files/labels는 cache_source_manifest와 동일. "
                    "★ EVAL ONLY — 학습에 쓰지 말 것 (260726 paired 검증 FAIL @384px, eval_only 필드 참조).")
        else:
            # size >= MIN_TRAIN_SAFE_SIZE 는 RandomResizedCrop 방향(다운샘플)은 만족하지만,
            # 그 자체로 "학습 등가성 검증 통과"를 의미하지 않는다 — paired 재검증 전까지는 미검증.
            derived["eval_only"] = False
            derived["train_verified"] = False
            note = (f"{size}x{size} 사전 리사이즈 캐시 (_cache_images_384.py). "
                    "files/labels는 cache_source_manifest와 동일. "
                    f"RandomResizedCrop 다운샘플 방향은 만족(size>={MIN_TRAIN_SAFE_SIZE}) 하지만 "
                    "paired 학습 등가성 검증 전까지는 학습 사용 금지 — train_verified 필드 확인.")
        derived["cache_note"] = note
        out_path = info["manifest_path"].parent / f"{info['name']}_{size}.json"  # source manifest 와 같은 위치(sibling) — data/pools/ 로 flatten 안 함
        out_path.write_text(json.dumps(derived, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[derived manifest] {out_path}", flush=True)


def collect_jobs(src_root: Path, dst_root: Path, pools: list[str], size: int, limit: int) -> list[tuple[str, str, int]]:
    """구 디렉토리 글롭 모드 (후방호환). data/images/<pool>/ 형태가 실제 존재할 때만 동작."""
    jobs: list[tuple[str, str, int]] = []
    for pool in pools:
        pool_src = src_root / pool
        if not pool_src.is_dir():
            print(f"[skip pool] {pool} (not found under {src_root})", flush=True)
            continue
        n_pool = 0
        for f in sorted(pool_src.rglob("*")):
            if f.is_file() and f.suffix.lower() in EXTENSIONS:
                rel = f.relative_to(src_root)
                jobs.append((str(f), str(dst_root / rel), size))
                n_pool += 1
        print(f"[pool] {pool}: {n_pool} images", flush=True)
    if limit:
        jobs = jobs[:limit]
    return jobs


def _run(jobs: list[tuple[str, str, int]], workers: int) -> tuple[int, int, int]:
    print(f"[plan] {len(jobs)} files total, workers={workers}", flush=True)
    t0 = time.time()
    done = skip = err = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
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
    print(f"[DONE] done={done} skip={skip} err={err}", flush=True)
    return done, skip, err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", default=DEFAULT_MANIFESTS,
                     help="comma-separated pool manifest names (data/pools/ 아래, .json 생략 가능) "
                          "or explicit .json paths, 주어진 순서대로 처리. 기본 모드.")
    ap.add_argument("--size", type=int, default=384,
                     help=f"캐시 정사각 해상도. 384 는 eval 전용(paired 학습검증 FAIL). "
                          f">= {MIN_TRAIN_SAFE_SIZE} 는 RandomResizedCrop 다운샘플 방향은 만족하지만 "
                          f"학습 사용 전 paired 재검증 필수(기본 384, 후방호환).")
    ap.add_argument("--src-root", default=None,
                     help="주면 구 디렉토리 글롭 모드로 전환 (--manifests 무시, 후방호환)")
    ap.add_argument("--dst-root", default=str(REPO / "data" / "images_384"))
    ap.add_argument("--pools", default="",
                     help="구 디렉토리 글롭 모드에서 --src-root 아래 pool 디렉토리 이름들")
    ap.add_argument("--workers", type=int, default=2, help="CPU worker 수 (GPU 학습 굶기지 않게 <=4 권장)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test 용 — 전체 job 중 앞 N개만")
    a = ap.parse_args()

    if a.src_root:
        src_root = Path(a.src_root)
        dst_root = Path(a.dst_root)
        pools = [p.strip() for p in a.pools.split(",") if p.strip()]
        jobs = collect_jobs(src_root, dst_root, pools, a.size, a.limit)
        _run(jobs, a.workers)
        print(f"[DONE] -> {dst_root.resolve()}", flush=True)
        return

    names = [n.strip() for n in a.manifests.split(",") if n.strip()]
    jobs, manifest_info = collect_jobs_from_manifests(names, a.size, a.limit)
    _done, _skip, err = _run(jobs, a.workers)
    write_derived_manifests(manifest_info, a.size)
    if err:
        print(f"[WARN] {err} file(s) failed to cache — derived manifest still written, "
              f"but consuming those images from cache will fail until re-run fixes them.", flush=True)
    for info in manifest_info:
        print(f"[OUT] {info['dst_root']}", flush=True)


if __name__ == "__main__":
    main()
