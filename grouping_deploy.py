#!/usr/bin/env python3
"""★ 재현 가능한 단일 진입점 — 무라벨 wafer grouping 배포.

배경: `_grouping_deliverable.py` 는 이긴 체크포인트 아키텍처(Linear->BN->ReLU->Linear, "net."
prefix 없음)를 로드하는 유일한 스크립트지만 (1) 앙상블 미지원 (2) soft-reassign 미이식
(3) 최소 1단계 하위폴더를 강제해 진짜 flat 폴더에서 n=0 크래시 (4) HDBSCAN 다이얼 기본값이
May-dial(mcs12/ms15) 로 stale (5) 라벨 없는 사내 데이터에서도 offline_eval 을 무조건 생성해
가짜 점수를 만들 위험이 있었다. 이 스크립트는 그 5개를 고친 배포용 진입점이다.

원본(`_grouping_deliverable.py`, `scripts/predict_grouping_prod.py`)은 수정하지 않는다 —
기존 결과 재현성 보존을 위해 이 파일을 새로 만들었다. soft-reassign 은
`scripts/predict_grouping_prod.py::reassign_noise_to_nearest_cluster` 를 그대로 import 해서
재사용한다 (새로 안 짬).

산출 스키마는 `_grouping_deliverable.py` 와 동일 (하위호환):
    groups.csv (label-free), representatives/group_XXX/, composites/group_XXX.png,
    summary.json (label-free, 항상 생성 — n/k/noise%/over_merge/stability/coherence),
    offline_eval.csv + offline_summary.json (★--offline-eval 명시 시에만 생성, 기본 OFF).

★ 라벨-free 지표 정의 (사내 배포 시 유일한 판단 창 — 정확히 이 정의로 고정, 임의 변경 금지):
    - group_coherence  = 최종(재배정 후) 그룹 멤버 pairwise cosine 유사도 평균. 재배정 포함 —
      "지금 이 그룹이 실제로 얼마나 뭉쳐 보이는가"를 재는 거라 최종 멤버십을 써야 맞음.
    - group_stability  = HDBSCAN이 낸 **seed(재배정 전) 클러스터**의 bootstrap co-assignment
      robustness (n_boot=5, frac=0.8, seed=42 고정 — `_grouping_deliverable.py` 원본과 동일).
      ★ 반드시 seed_pred(재배정 전) 로 계산 — 재배정 후 pred 로 계산하면 "구조 안정성"이 아니라
      "nearest-centroid 재배정 휴리스틱의 안정성"으로 의미가 바뀌어 stability >= 0.75 무라벨
      선택 게이트 판정이 실제로 뒤집힐 수 있다 (실측: 0.4547 재배정후 vs 0.8441 재배정전,
      champion 0.8825 와 후자가 일치하는 쪽 — 260726 team-lead 리뷰로 확정, 코드에도 주석).
    - noise_pct (summary.json)  = 재배정 후 최종 noise / n (전체, 배경 포함).

현재 보존된 E: canonical pool의 manifest 입력 예시:
    python grouping_deploy.py \\
        --backbone weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth \\
        --proj runs/sweep/abl_sw_t20_B4_260724_102757/checkpoints/proj_ep20.pt \\
               runs/sweep/abl_best_s1_B4_260724_111053/checkpoints/proj_ep18.pt \\
        --pool data/pools/unknown_eval100.json \\
        --out result_grouping/unknown_eval_grouping \\
        --reassign nearest_q90 --device auto

flat 폴더(사내 실데이터, 하위폴더 없음) 배포:
    python grouping_deploy.py --backbone <pth> --proj <ckpt> \\
        --pool E:/data/images/prod/AA/K1AA/20260726 --out result_grouping/prod_run
    (하위폴더가 없으면 라벨 없이 동작 — offline-eval 은 자동으로 건너뜀)
"""
import argparse
import os
import csv
import importlib.util
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent
# ★ append: insert(0) 은 repo 안 폴더가 site-packages 를 가리게 만든다
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
from scripts._common import load_pool_manifest  # noqa: E402  (manifest-based --pool selection, backward-compat)
IMG = 384
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
TF = T.Compose([T.Resize((IMG, IMG), interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# ★ 이긴 운영점 다이얼 (real-domain clean546 champion) — 사용자 승인 없이 값 변경 금지, CLI 로 override 만.
DEFAULT_MCS = 6
DEFAULT_MS = 3
DEFAULT_METHOD = "leaf"
DEFAULT_EPS = 0.06


def _load_reassign_fn():
    """scripts/predict_grouping_prod.py 의 reassign_noise_to_nearest_cluster 를 그대로 import (재구현 금지)."""
    path = REPO_ROOT / "scripts" / "predict_grouping_prod.py"
    spec = importlib.util.spec_from_file_location("_pgp_reused", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.reassign_noise_to_nearest_cluster


def build_proj():
    return nn.Sequential(nn.Linear(1024, 1024, bias=False), nn.BatchNorm1d(1024),
                          nn.ReLU(inplace=True), nn.Linear(1024, 128))


def load_backbone(path, device):
    bb = timm.create_model("convnextv2_base.fcmae_ft_in22k_in1k_384", pretrained=False, num_classes=0, global_pool="")
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    for p in ("model.", "backbone.", "module."):
        if any(k.startswith(p) for k in sd):
            sd = {(k[len(p):] if k.startswith(p) else k): v for k, v in sd.items()}
    bb.load_state_dict(sd, strict=False)
    return bb.eval().to(device)


class _Head(nn.Module):
    """proj (+ 있으면 residual adapter). 학습 시 forward 와 동일한 순서:
         pool -> pool + gamma*adapt(pool) -> proj
    adapter 는 GAP 직후 proj **앞**에 붙는다. 체크포인트에 "adapt" 가 있는데 이걸
    빼고 proj 만 쓰면 학습 때와 다른 입력을 proj 에 먹이게 된다."""

    def __init__(self, proj, adapt=None, gamma=None):
        super().__init__()
        self.proj, self.adapt = proj, adapt
        self.gamma = gamma if gamma is not None else 0.0

    def forward(self, x):
        if self.adapt is not None:
            g = self.gamma
            g = g.to(x.device) if torch.is_tensor(g) else g
            x = x + g * self.adapt(x)
        return self.proj(x)


def build_adapter(d=1024, h=None):
    h = h or d // 8
    return nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))


def load_proj(path, device):
    d = torch.load(path, map_location="cpu")
    pj = d["proj"] if isinstance(d, dict) and "proj" in d else d
    pj = {k[len("net."):] if k.startswith("net.") else k: v for k, v in pj.items()}
    proj = build_proj()
    proj.load_state_dict(pj)
    ad = None
    if isinstance(d, dict) and "adapt" in d:
        hid = d["adapt"]["0.weight"].shape[0] if "0.weight" in d["adapt"] else None
        ad = build_adapter(1024, hid)
        ad.load_state_dict(d["adapt"])
        print(f"[model] residual adapter 적용 (hidden={hid}, gamma={float(d.get('gamma', 0)):.4f})",
              flush=True)
    return _Head(proj, ad, d.get("gamma") if isinstance(d, dict) else None).eval().to(device)


def resolve_device(requested: str) -> str:
    """어느 장치로 도는지 **모호함 없이** 찍는다.

    서버에서 "GPU 를 안 쓴다"가 반복돼서 조용한 fallback 을 없앴다. 원인은 셋뿐이다:
      (A) --device 를 안 줬다        -> 예전 기본값이 cpu 였다. 지금은 auto.
      (B) CPU-only torch wheel      -> torch.version.cuda 가 None. `pip install torch` 만
                                       하면 이게 깔린다. CUDA 휠로 다시 깔아야 한다.
      (C) 드라이버/컨테이너에서 GPU 미노출 -> nvidia-smi 는 되는데 is_available()=False.
    """
    built = torch.version.cuda
    avail = torch.cuda.is_available()
    print(f"[gpu] torch {torch.__version__} | CUDA 빌드 {built or '없음 (CPU-only wheel)'} "
          f"| is_available {avail}", flush=True)
    if avail:
        i = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(i)
        print(f"[gpu] cuda:{i} {p.name}  {p.total_memory / 2**30:.0f}GiB  "
              f"cc{p.major}.{p.minor}", flush=True)

    dev = "cuda" if (requested in ("auto", "cuda") and avail) else "cpu"

    if requested in ("auto", "cuda") and not avail:
        print("[gpu] ★ GPU 를 못 쓴다 — CPU 로 돈다. 둘 중 하나다:", flush=True)
        if not built:
            print("[gpu]   (B) torch 가 CPU-only wheel 이다. 이게 원인이다. 다시 깔아라:", flush=True)
            print("[gpu]       pip install --force-reinstall torch torchvision "
                  "--index-url https://download.pytorch.org/whl/cu124", flush=True)
        else:
            print(f"[gpu]   (C) torch 는 CUDA {built} 빌드인데 GPU 가 안 보인다. "
                  f"nvidia-smi / 컨테이너 --gpus all / CUDA_VISIBLE_DEVICES 를 확인하라.", flush=True)
    elif dev == "cuda":
        # 실제로 GPU 에서 연산이 되는지 한 번 돌려본다 (is_available 만으로는 부족).
        t = torch.randn(2048, 2048, device="cuda")
        float((t @ t).sum())
        torch.cuda.synchronize()
        print(f"[gpu] 자체검사 통과 — VRAM 할당 "
              f"{torch.cuda.memory_allocated() / 2**20:.0f}MiB", flush=True)
        del t
        torch.cuda.empty_cache()
    return dev


def embed_backbone(paths, backbone, device, batch=32, cache=None, what="feat"):
    """raw GAP feature (proj 없이). ★ deploy step 들이 공유하는 **단일 경로**.

    step0·step2(채점)·step5(시간축)가 각자 임베딩 루프를 갖고 있었는데
    셋 다 이것들이 빠져 있었다:
      (a) 디코드 캐시 미사용   -> 6400x6400 을 매번 다시 디코드 (GPU 가 굶는다)
      (b) 단일스레드 디코드    -> 코어를 하나만 쓴다
      (c) UC_PALETTE_MASK 무시 -> ★ 마스킹을 켜고 학습해도 채점은 안 켜진 채 돈다.
          `Image.open(p).convert("RGB")` 를 직접 부르면 `_open_rgb` 를 우회한다.
          학습/채점 입력이 달라지는 **정합성 버그**라 결과가 조용히 무의미해진다.
    같은 코드를 세 벌 두면 또 갈라지므로 여기 하나로 모은다.
    """
    import time
    embs, t0, n = [], time.time(), len(paths)
    every = max(1, (n // batch) // 20)
    _reset_timers()
    with torch.no_grad():
        for bi, done, x in _iter_batches(paths, batch, cache=cache):
            g = time.time()
            with amp_ctx():
                f = backbone.forward_features(
                    x.to(device, non_blocking=True)).mean(dim=(2, 3))
            e = f.float().cpu()
            _GPU[0] += time.time() - g
            embs.append(e)
            if bi % every == 0 or done >= n:
                _progress(done, n, t0, what)
    return torch.cat(embs)


def load_cache_for(cache_dir, paths):
    """deploy/build_cache.py 캐시를 현재 palette_mask 설정으로 연다. 없으면 None."""
    if not cache_dir:
        return None
    try:
        sys.path.append(str(REPO_ROOT / "deploy"))
        from build_cache import load_cache
        c = load_cache(cache_dir, paths, _PALETTE_MASK)
    except Exception as e:
        print(f"[cache] 로드 실패({type(e).__name__}: {e}) -> 원본을 읽는다", flush=True)
        return None
    if c is None:
        print(f"[cache] {cache_dir} 를 쓸 수 없다 (없거나 pool/palette_mask 불일치) "
              f"-> 원본을 읽는다", flush=True)
    else:
        print(f"[cache] 사용 {cache_dir}  shape={tuple(c.shape)}  디코드 건너뜀", flush=True)
    return c


def proj_tag(proj_path: str) -> str:
    """체크포인트 경로에서 라벨-free 태그 derive (seed는 run_info.json 에서, epoch은 파일명에서)."""
    p = Path(proj_path)
    m = re.search(r"ep(\d+)", p.stem)
    epoch = m.group(1) if m else p.stem
    run_dir = p.parent.parent if p.parent.name == "checkpoints" else p.parent
    seed = None
    ri = run_dir / "run_info.json"
    if ri.exists():
        try:
            cfg = json.loads(ri.read_text(encoding="utf-8")).get("cfg", {})
            seed = cfg.get("SEED")
        except Exception:
            pass
    return f"s{seed}_ep{epoch}" if seed is not None else f"{run_dir.name}_ep{epoch}"


def collect_pool(pool_dir: str):
    """flat + nested 둘 다 지원 (rglob). 하위폴더 있으면 그 이름을 label 로 선택적 수집, flat 이면 label=None.
    .json manifest(make_pool_manifest.py 생성)도 지원 — 기존 flat/nested 로직은 무변경, manifest 분기만 추가.
    manifest label=null 은 그대로 label=None 으로 보존한다(사내 실데이터 무라벨 pool 을 manifest 로 지정할 때
    부모폴더명으로 오염되지 않게 — resolve_pool() 의 fallback 과 달리 이 스크립트는 label=None 을 의미 있는
    상태로 취급하므로)."""
    root = Path(pool_dir)
    if root.is_file() and root.suffix.lower() == ".json":
        entries = sorted(load_pool_manifest(root), key=lambda e: e[0])  # flat path 정렬, collect_pool 의 rglob 정렬과 동일 의미
        return [str(p) for p, _ in entries], [lab for _, lab in entries]
    paths, labels = [], []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in EXTS:
            paths.append(str(p))
            labels.append(p.parent.name if p.parent != root else None)
    return paths, labels


def embed_one_checkpoint(paths, backbone, proj, device, batch=32, cache=None):
    """raw GAP f -> proj -> L2 (학습과 동일 순서; normalize 후 projection 은 버그)."""
    import time
    embs, t0, n = [], time.time(), len(paths)
    every = max(1, (n // batch) // 20)
    _reset_timers()
    with torch.no_grad():
        for bi, done, x in _iter_batches(paths, batch, cache=cache):
            _g = time.time()
            with amp_ctx():
                pool = backbone.forward_features(
                    x.to(device, non_blocking=True)).mean(dim=(2, 3))
            e = F.normalize(proj(pool.float()), dim=1).cpu()
            _GPU[0] += time.time() - _g
            embs.append(e)
            if bi % every == 0 or done >= n:
                _progress(done, n, t0)
    return torch.cat(embs)


# ── palette 전처리 ────────────────────────────────────────────────────────
# wafer PNG 는 mode="P" 이고 index 0~7 이 결함 grade, 그 위 index 는 border/background/
# invalid 계열이다. 그 색들이 clustering shortcut 이 되므로 흰색으로 지우는 전처리가 있다
# (scripts/_common.mask_palette_non_grade_to_white).
#
# ★ 단 **학습 때와 반드시 같아야 한다.** champion 과 May b4 는 마스킹 **없이** 학습됐으므로
#   그 head 에 마스킹을 켜서 넣으면 학습 때와 다른 입력이 되어 결과가 무의미해진다.
#   그래서 기본값은 off 이고, 켜려면 UC_PALETTE_MASK=1 로 명시해야 한다.
# composite 는 **원본 해상도 그대로** 쓴다. 리사이즈 금지 —
# chip 경계는 1px 선이라 축소하면 통째로 사라진다 (1024/1600/800 에서 idx10 소멸 실측).
# 그룹당 30~45MB 를 먹지만 화질을 바꾸지 않는 게 우선이다.
_PALETTE_MASK = os.environ.get("UC_PALETTE_MASK", "0").strip().lower() in ("1", "true", "yes", "on")
try:
    from scripts._common import mask_palette_non_grade_to_white as _mask_palette
except Exception:
    _mask_palette = None


def _open_rgb(path):
    im = Image.open(path)
    if _PALETTE_MASK and _mask_palette is not None:
        im = _mask_palette(im)
    return im.convert("RGB")


def amp_ctx():
    """SITE_AMP=fp16|bf16 이면 autocast, 아니면 아무것도 안 함.

    ★ 기본 off. 실측 1.5배지만 임베딩이 8.5e-4(fp16)/6.4e-3(bf16) 어긋난다.
      5.5e-4 만으로도 HDBSCAN bootstrap 이 뒤집혀 mean_stability 가
      0.9682 -> 0.9624 로 갈린 적이 있다. 켤 거면 **모든 arm 에 동일하게** 켜라.
    """
    import contextlib
    m = os.environ.get("SITE_AMP", "off").strip().lower()
    if m in ("fp16", "float16"):
        return torch.autocast("cuda", dtype=torch.float16)
    if m in ("bf16", "bfloat16"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def maybe_channels_last(m):
    if os.environ.get("SITE_CHANNELS_LAST", "0").strip().lower() in ("1", "true", "yes", "on"):
        return m.to(memory_format=torch.channels_last)
    return m


def decode_workers() -> int:
    """디코드 스레드 수.

    ★ 병목은 GPU 가 아니라 6400x6400 palette PNG 디코드다 (실측 배치 시간의 62~79%).
      H200 처럼 GPU 가 빠를수록 nvidia-smi util 은 오히려 낮게(한 자리) 보인다 —
      GPU 를 안 쓰는 게 아니라 굶는 것이다. 코어를 다 쓰는 게 유일한 해법이다.
      예전 상한 16 은 코어 많은 서버에서 그대로 낭비였다. 지금은 상한 없음.
    """
    import os
    n = int(os.environ.get("SITE_DECODE_WORKERS", "0"))
    return n if n > 0 else max(4, os.cpu_count() or 4)


def _iter_batches(paths, batch, workers=None, cache=None):
    """이미지 디코드를 스레드로 병렬화하고 다음 배치를 미리 준비한다.

    병목은 GPU 가 아니라 **단일 스레드 PIL 디코드**다 (실측: 배치당 8.25s 중 GPU 0.58s).
    PIL 은 디코드/리사이즈 중 GIL 을 놓으므로 스레드가 실제로 병렬로 돈다.
    픽셀은 그대로다 — 순서도 보존한다.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor
    if workers is None:
        workers = decode_workers()
    prefetch = max(2, int(os.environ.get("SITE_PREFETCH", "3")))
    idx = list(range(0, len(paths), batch))

    if cache is not None:
        # ★ 이미 384 로 줄여 놓은 uint8 을 읽는다. 디코드 0 — GPU 가 굶지 않는다.
        #   픽셀은 원본 경로와 동일하다 (deploy/build_cache.py 참조, 최대 절대차 0).
        def build(i):
            a = torch.from_numpy(np.array(cache[i:i + batch]))   # memmap -> 쓰기가능 복사
            x = a.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
            # ★ contiguous() 필수. 값이 같아도 레이아웃이 다르면 cuDNN 이 다른 커널을
            #   골라 임베딩이 5.5e-4 어긋나고, HDBSCAN bootstrap 이 뒤집혀
            #   stability 가 0.9682 -> 0.9624 로 갈렸다 (실측). 붙이면 비트 단위로 같다.
            return (x - _MEAN) / _STD
    else:
        def build(i):
            return torch.stack([TF(_open_rgb(p)) for p in paths[i:i + batch]])

    import time
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(build, i) for i in idx[:prefetch]]
        nxt = prefetch
        for k in range(len(idx)):
            _t = time.time()
            x = futs[k].result()
            _WAIT[0] += time.time() - _t      # ★ 디코드 대기 = GPU 가 노는 시간
            if nxt < len(idx):
                futs.append(ex.submit(build, idx[nxt])); nxt += 1
            yield k, min(idx[k] + batch, len(paths)), x


# 디코드 대기 / GPU 연산 누적 시간. "GPU util 이 왜 6% 인가"를 추측이 아니라 숫자로 답한다.
_WAIT = [0.0]
_GPU = [0.0]


def _reset_timers():
    _WAIT[0] = _GPU[0] = 0.0


def _progress(done, total, t0, what="embed"):
    """긴 임베딩 구간이 멈춘 것처럼 보이지 않게 진행률을 찍는다.

    ★ 디코드 대기와 GPU 연산 비율을 같이 찍는다. 6400x6400 palette PNG 디코드가
      배치 시간의 대부분이라 nvidia-smi util 이 한 자리로 보이는 게 정상이다 —
      그게 GPU 를 안 쓰는 것과 다르다는 걸 로그에서 바로 구분할 수 있어야 한다.
    """
    import time
    el = time.time() - t0
    rate = done / el if el > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    split = ""
    if el > 0 and (_WAIT[0] or _GPU[0]):
        split = (f"  [디코드대기 {100*_WAIT[0]/el:4.1f}% / GPU {100*_GPU[0]/el:4.1f}%]")
    print(f"  [{what}] {done}/{total} ({100*done/max(1,total):5.1f}%)  "
          f"{rate:5.1f} img/s  경과 {el/60:.1f}분  남은 {eta/60:.1f}분{split}", flush=True)


def partition_keys(paths, spec: str):
    """파일명에서 분리 그룹핑 키를 뽑는다. spec 이 비면 None (= 분리 안 함).

    사내 파일명 예: `AAQ729_00C_20_20260501_010000_83.0_17_PE_ENGINEER.png`
      `_` 로 나눈 필드 -> [AAQ729, 00C, 20, 20260501, 010000, 83.0, 17, PE, ENGINEER]
      제품/라인 코드는 **필드 1** (`00C` 1,259장 / `00P` 1,001장).

    spec 형식:
      "field:1"        -> `_` 로 나눈 1번 필드 (권장. 위치가 고정이라 안전하다)
      "<regex>"        -> 캡처그룹 1을 키로. 예: `_(\\d{2}[A-Z])_`
      ""               -> 분리 안 함

    ⚠ "언더바 사이 3글자" 를 길이로만 잡으면 안 된다 — 같은 파일명에 `PWQ` 도 3글자다
      (실측 2,260장 중 742장이 3글자 필드를 2개 갖는다). 그래서 위치/정규식으로 고정한다.
    """
    spec = (spec or "").strip()
    if not spec:
        return None
    keys = []
    if spec.startswith("field:"):
        i = int(spec.split(":", 1)[1])
        for p in paths:
            f = Path(p).stem.split("_")
            keys.append(f[i] if 0 <= i < len(f) else "_unknown")
    else:
        rx = re.compile(spec)
        for p in paths:
            m = rx.search(Path(p).name)
            keys.append((m.group(1) if m.groups() else m.group(0)) if m else "_unknown")
    return keys


def cluster_by_partition(z, keys, a, reassign_fn):
    """키마다 **독립으로** HDBSCAN 을 돌리고 cluster id 를 겹치지 않게 이어 붙인다.

    다이얼(mcs/ms)은 그대로 쓴다 — "몇 장 이상 뭉쳐야 그룹인가"는 운영 판단이라
    파티션이 나뉘어도 같은 값이 맞다. ★ k 는 여기서도 쓰지 않는다.
    """
    keys = list(keys)
    uniq = sorted(set(keys))
    seed = np.full(len(keys), -1, dtype=int)
    final = np.full(len(keys), -1, dtype=int)
    offset = 0
    rows, n_re = [], 0
    print(f"[partition] {a.partition_by} -> {len(uniq)}개로 분리 클러스터링: "
          + ", ".join(f"{k}({keys.count(k)})" for k in uniq), flush=True)
    # ★ 다이얼은 **파티션 크기 대비**로 봐야 한다. 전체 n 으로 정한 mcs 를 작은
    #   파티션에 그대로 쓰면 그룹이 하나도 안 생겨 조용히 k=0 / noise 100% 가 된다
    #   (실측: n=36 파티션에 mcs=20 -> k=0). 미리 경고한다.
    small = [(k, keys.count(k)) for k in uniq if keys.count(k) < 3 * a.mcs]
    if small:
        print(f"[partition] ★ 경고: mcs={a.mcs} 에 비해 너무 작은 파티션이 있다 "
              f"-> {', '.join(f'{k}(n={v})' for k, v in small)}", flush=True)
        print(f"[partition]   그룹이 안 만들어져 k=0/noise 100% 가 될 수 있다. "
              f"SITE_MCS 를 낮추거나 --partition-by 를 끄라.", flush=True)
    for k in uniq:
        idx = np.array([i for i, v in enumerate(keys) if v == k])
        zk = z[idx]
        sp = hdbscan_predict(zk, a.mcs, a.ms, a.method, a.eps)
        sn = int((sp == -1).sum())
        fp = sp
        if reassign_fn is not None:
            fp, info = reassign_fn(zk, sp, a.reassign)
            n_re += int(info.get("n_reassigned", 0))
        kk = sorted(int(c) for c in set(sp.tolist()) if c != -1)
        remap = {c: offset + j for j, c in enumerate(kk)}
        seed[idx] = [remap.get(int(c), -1) for c in sp]
        final[idx] = [remap.get(int(c), -1) for c in fp]
        offset += len(kk)
        rows.append({"key": k, "n": int(len(idx)), "k": len(kk),
                     "seed_noise_pct": round(100.0 * sn / max(1, len(idx)), 2),
                     "noise_pct": round(100.0 * int((fp == -1).sum()) / max(1, len(idx)), 2)})
        print(f"  [partition] {k:<8} n={len(idx):>6,}  k={len(kk):>4}  "
              f"seed_noise={rows[-1]['seed_noise_pct']:>6.2f}%  "
              f"noise={rows[-1]['noise_pct']:>6.2f}%", flush=True)
    info = {"mode": a.reassign, "seed_n_noise": int((seed == -1).sum()),
            "n_reassigned": n_re, "assign_quantile": None,
            "partition_by": a.partition_by}
    return seed, final, rows, info


def hdbscan_predict(z, mcs, ms, method, eps):
    import hdbscan
    return hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                            metric=os.environ.get("SITE_METRIC", "euclidean"),
                            cluster_selection_method=method,
                            cluster_selection_epsilon=eps).fit_predict(z.astype(np.float64)).astype(int)


def per_group_stability(z, base_pred, mcs, ms, method, eps,
                        n_boot=None, frac=None, seed=None):
    """★ 기본값 n_boot=5 / frac=0.8 / seed=42 는 `_grouping_deliverable.py` 원본과 같다.
    바꾸면 stability 숫자가 통째로 달라져 과거 기록과 비교가 깨진다 —
    `deploy/config.py::Cluster.STABILITY_*` 로만 조정하고 기본값은 건드리지 마라."""
    n_boot = int(os.environ.get("SITE_STAB_NBOOT", 5)) if n_boot is None else n_boot
    frac = float(os.environ.get("SITE_STAB_FRAC", 0.8)) if frac is None else frac
    seed = int(os.environ.get("SITE_STAB_SEED", 42)) if seed is None else seed
    """group 별 co-assignment stability: 부분표본에서 그 그룹 멤버쌍이 같은 cluster 유지 평균 비율."""
    rng = np.random.RandomState(seed)
    n = len(z)
    keep = defaultdict(list)
    for _ in range(n_boot):
        idx = np.sort(rng.choice(n, int(frac * n), replace=False))
        pos = {g: i for i, g in enumerate(idx)}
        p2 = hdbscan_predict(z[idx], mcs, ms, method, eps)
        for c in set(base_pred.tolist()):
            if c == -1:
                continue
            mem = [g for g in np.where(base_pred == c)[0] if g in pos]
            if len(mem) < 2:
                continue
            lb = np.array([p2[pos[g]] for g in mem])
            same = 0
            tot = 0
            for i in range(len(mem)):
                for j in range(i + 1, len(mem)):
                    tot += 1
                    if lb[i] == lb[j] and lb[i] != -1:
                        same += 1
            if tot:
                keep[c].append(same / tot)
    return {c: float(np.mean(v)) if v else 0.0 for c, v in keep.items()}


def reassign_display(mode: str) -> str:
    return {"none": "", "nearest_q90": "_reassign0.90", "nearest_q80": "_reassign0.80",
            "assign_all": "_reassignall"}.get(mode, f"_reassign{mode}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", required=True, help="frozen backbone .pth (FCMAE 등)")
    ap.add_argument("--proj", nargs="*", default=[],
                     help="projection head .pt 1개 이상 (2개+ 주면 앙상블: 각각 raw-GAP->proj->L2 후 concat+L2)")
    ap.add_argument("--pool", required=True, help="이미지 루트 (flat 또는 중첩 폴더 모두 지원, rglob)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=12, help="cluster당 대표 이미지 수")
    ap.add_argument("--mcs", type=int, default=DEFAULT_MCS, help=f"HDBSCAN min_cluster_size (기본 {DEFAULT_MCS}=champion)")
    ap.add_argument("--ms", type=int, default=DEFAULT_MS, help=f"HDBSCAN min_samples (기본 {DEFAULT_MS}=champion)")
    ap.add_argument("--method", default=DEFAULT_METHOD, choices=["leaf", "eom"],
                     help=f"HDBSCAN cluster_selection_method (기본 {DEFAULT_METHOD}=champion)")
    ap.add_argument("--eps", type=float, default=DEFAULT_EPS, help=f"HDBSCAN cluster_selection_epsilon (기본 {DEFAULT_EPS}=champion)")
    ap.add_argument("--reassign", default="none", choices=["none", "nearest_q90", "nearest_q80", "assign_all"],
                     help="HDBSCAN noise(-1) 를 nearest cluster centroid 로 재배정 (라벨無, scripts/predict_grouping_prod.py 재사용). "
                          "champion 운영점 = nearest_q90. 기본 none.")
    ap.add_argument("--offline-eval", action="store_true",
                     help="★ 명시할 때만 라벨 기반 P1-P4/ARI/AMI 채점 (offline_eval.csv+offline_summary.json). "
                          "기본 OFF — 폴더명이 진짜 라벨이 아니면(사내 실데이터) 가짜 점수 방지.")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                     help="기본 auto = GPU 있으면 GPU. 강제로 CPU 를 쓰려면 --device cpu.")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--partition-by", default="",
                     help="파일명 코드마다 **따로** 클러스터링. 예 field:1 이면 `_` 로 나눈 "
                          "1번 필드(_00P_ / _00C_)별로 분리. 정규식도 됨(캡처그룹 1). "
                          "비우면 분리 안 함. ★ k 는 여기서도 입력하지 않는다.")
    ap.add_argument("--no-composites", action="store_true",
                     help="composite 생략. ★ arm/셀 비교 단계에서는 켜라 — composite 는 "
                          "원본 6400x6400 을 다시 읽어 arm 당 ~70초를 먹는데, arm 을 "
                          "고르는 데는 쓰이지 않는다(판정은 seed_noise/k/coherence).")
    ap.add_argument("--cache", default="",
                     help="deploy/build_cache.py 로 만든 384 캐시 폴더. 있으면 디코드를 "
                          "건너뛴다(픽셀 동일). arm 마다 다시 디코드하는 낭비를 없앤다.")
    a = ap.parse_args()

    device = resolve_device(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    paths, labels = collect_pool(a.pool)
    if not paths:
        raise SystemExit(f"[grouping_deploy] no images found under {a.pool} (checked recursively, exts={sorted(EXTS)})")
    n_labeled = sum(1 for l in labels if l is not None)
    import os as _os
    _w = decode_workers()
    print(f"[run] palette_mask={_PALETTE_MASK} (학습 때와 같아야 한다 — champion/b4 는 off 로 학습됨)", flush=True)
    print(f"[run] device={device}  decode_workers={_w} (cpu {_os.cpu_count()})  "
          f"batch={a.batch}  prefetch={_os.environ.get('SITE_PREFETCH', '3')}", flush=True)
    print(f"[data] {len(paths)} imgs ({n_labeled} with a parent-folder label, {len(paths) - n_labeled} flat/root)", flush=True)

    # ★ 디코드 캐시. 병목이 디코드라 arm 마다 다시 읽으면 GPU 가 그만큼 논다.
    _cache = None
    _cdir = a.cache or _os.environ.get("SITE_CACHE_DIR", "")
    if _cdir:
        try:
            sys.path.append(str(REPO_ROOT / "deploy"))
            from build_cache import load_cache
            _cache = load_cache(_cdir, paths, _PALETTE_MASK)
        except Exception as e:
            print(f"[cache] 로드 실패({type(e).__name__}: {e}) -> 원본을 읽는다", flush=True)
        if _cache is None:
            print(f"[cache] {_cdir} 를 쓸 수 없다 (없거나 pool/palette_mask 불일치) "
                  f"-> 원본을 읽는다. 만들려면: python deploy/build_cache.py", flush=True)
        else:
            print(f"[cache] 사용 {_cdir}  shape={tuple(_cache.shape)}  "
                  f"디코드 건너뜀", flush=True)

    print(f"[model] backbone={a.backbone}", flush=True)
    bb = maybe_channels_last(load_backbone(a.backbone, device))
    tags = []
    per_ckpt_embs = []
    if not a.proj:
        # proj 없음 = frozen backbone 기준선. GAP -> L2 만 한다.
        # (사다리 (1) 에서 frozen / TAPT-backbone-only arm 을 재려면 필수)
        print("[model] proj 없음 -> frozen backbone (GAP -> L2)", flush=True)
        tags.append("frozen")
        import time
        _e, _t0, _n = [], time.time(), len(paths)
        _every = max(1, (_n // a.batch) // 20)
        _reset_timers()
        with torch.no_grad():
            for _bi, _done, _x in _iter_batches(paths, a.batch, cache=_cache):
                _g = time.time()
                with amp_ctx():
                    _f = bb.forward_features(
                        _x.to(device, non_blocking=True)).mean(dim=(2, 3))
                _q = F.normalize(_f.float(), dim=1).cpu()
                _GPU[0] += time.time() - _g
                _e.append(_q)
                if _bi % _every == 0 or _done >= _n:
                    _progress(_done, _n, _t0)
        per_ckpt_embs.append(torch.cat(_e))
    for proj_path in a.proj:
        tag = proj_tag(proj_path)
        tags.append(tag)
        print(f"[model] proj={proj_path} (tag={tag})", flush=True)
        proj = load_proj(proj_path, device)
        per_ckpt_embs.append(embed_one_checkpoint(paths, bb, proj, device, a.batch, _cache))

    if len(per_ckpt_embs) > 1:
        z = F.normalize(torch.cat(per_ckpt_embs, dim=1), dim=1).numpy().astype("float32")
        print(f"[ensemble] concat {len(per_ckpt_embs)} checkpoints -> dim {z.shape[1]}, L2 renormalized", flush=True)
    else:
        z = per_ckpt_embs[0].numpy().astype("float32")

    print(f"[cluster] HDBSCAN mcs={a.mcs} ms={a.ms} method={a.method} eps={a.eps} "
          f"— n={len(z)} 시작 (큰 pool 이면 수 분 걸린다)", flush=True)

    # ★ 분리 그룹핑 — 파일명의 코드(예 _00P_ / _00C_)마다 **따로** 클러스터링한다.
    #   서로 다른 제품/라인은 애초에 같은 그룹이 될 수 없으므로 한 덩어리로 묶어
    #   돌리면 다이얼 하나가 두 모집단을 동시에 맞춰야 해서 양쪽 다 나빠진다.
    parts = partition_keys(paths, a.partition_by)
    reassign_fn = _load_reassign_fn() if a.reassign != "none" else None
    if parts is None:
        seed_pred = hdbscan_predict(z, a.mcs, a.ms, a.method, a.eps)
        pred = seed_pred
        seed_noise = int((seed_pred == -1).sum())
        reassign_info = {"mode": a.reassign, "seed_n_noise": seed_noise,
                         "n_reassigned": 0, "assign_quantile": None}
        if reassign_fn is not None:
            pred, reassign_info = reassign_fn(z, seed_pred, a.reassign)
            print(f"[cluster] reassign={a.reassign} seed_noise={seed_noise} "
                  f"reassigned={reassign_info['n_reassigned']} "
                  f"final_noise={int((pred == -1).sum())}", flush=True)
        part_summary = None
    else:
        seed_pred, pred, part_summary, reassign_info = cluster_by_partition(
            z, parts, a, reassign_fn)
        seed_noise = int((seed_pred == -1).sum())
    n = len(pred)

    clusters = sorted(int(c) for c in set(pred.tolist()) if c != -1)
    # stability = seed(pre-reassign) 클러스터의 bootstrap co-assignment robustness.
    # reassign 된 noise 는 HDBSCAN 이 낸 구조가 아니라 후처리 휴리스틱 소속이라 여기 섞으면
    # "구조 안정성"이 아니라 "재배정 휴리스틱 안정성"이 되어 의미가 달라진다
    # (base _grouping_deliverable.py 는 애초 reassign 이 없어 pred==seed_pred 였음 -- 그 의미를 유지).
    stab = per_group_stability(z, seed_pred, a.mcs, a.ms, a.method, a.eps)
    lab = np.array([l if l is not None else "" for l in labels])

    _kind = "frozen" if not a.proj else ("adapted_ens" if len(tags) > 1 else "adapted")
    model_mode = f"{_kind}[{'+'.join(tags)}]_mcs{a.mcs}ms{a.ms}{a.method}{reassign_display(a.reassign)}"

    # --- representatives + composites + groups.csv (label-free) ---
    rep_root = out / "representatives"
    comp_root = out / "composites"
    rep_root.mkdir(exist_ok=True)
    comp_root.mkdir(exist_ok=True)
    rows, off_rows = [], []
    for c in clusters:
        idx = np.where(pred == c)[0]
        center = z[idx].mean(0)
        order = idx[np.argsort(np.linalg.norm(z[idx] - center, axis=1))]
        sim = z[idx] @ z[idx].T
        iu = np.triu_indices(len(idx), 1)
        coh = float(sim[iu].mean()) if len(idx) > 1 else 1.0
        # ★ 과병합 판정은 **자기 파티션 크기** 기준. 전체 n 으로 재면 분리 그룹핑을 켠
        #   순간 모든 그룹이 임계 밑으로 내려가 과병합이 조용히 안 잡힌다
        #   (실측: 분리 전 3건 -> 분리 후 0건).
        _base = n if parts is None else sum(1 for v in parts if v == parts[idx[0]])
        over = len(idx) >= float(os.environ.get("SITE_OVER_MERGE_FRAC", 0.20)) * _base
        gdir = rep_root / f"group_{c:03d}_n{len(idx)}"
        gdir.mkdir(exist_ok=True)
        for rank, i in enumerate(order[:a.reps], 1):
            src = Path(paths[i])
            if src.exists():
                shutil.copy2(src, gdir / f"rep{rank:02d}_{src.name}")
        # composite — mapviewer(api/composite_map.py) 공식 규격.
        # RGB 평균이 아니라 **palette index 의 grade 빈도**를 쌓아 스칼라 맵으로 만들고
        # colormap 으로 그린다. grade 는 순서형 범주라 색 평균은 의미가 없다.
        # 원본 해상도 유지 / chip 경계는 전부 idx10 회색 / chip 안 bin 번호는 제거된다.
        # ★ --no-composites 면 건너뛴다. composite 는 임베딩 캐시(384)를 못 쓰고 원본
        #   6400x6400 을 다시 읽어야 해서 arm 당 ~70초를 먹는데(실측: 임베딩은 5초),
        #   arm 을 고르는 판정에는 쓰이지 않는다. 고른 뒤 그 arm 만 만들면 된다.
        gdir_c = comp_root / f"group_{c:03d}_n{len(idx)}"
        if not a.no_composites:
            try:
                from deploy.composite import write_group_composites
            except Exception:
                sys.path.append(str(REPO_ROOT / "deploy"))
                from composite import write_group_composites
            try:
                write_group_composites([paths[i] for i in idx], gdir_c)
            except Exception as e:
                print(f"  [warn] group {c} composite 실패: {e}", flush=True)
        # medoid = 그룹 중심에 가장 가까운 **실제 원본** (heatmap 과 대조용)
        gdir_c.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths[order[0]], gdir_c / f"medoid_{Path(paths[order[0]]).name}")
        rows.append({"group_id": c, "group_size": len(idx), "group_stability": round(stab.get(c, 0.0), 4),
                      "group_coherence": round(coh, 4),
                      "review_status": "over_merged_review" if over else "candidate",
                      "model_mode": model_mode})
        if n_labeled:
            vals, cnts = np.unique(lab[idx], return_counts=True)
            off_rows.append({"group_id": c, "group_size": len(idx),
                              "majority_label": str(vals[cnts.argmax()]),
                              "purity": round(float(cnts.max() / cnts.sum()), 4)})

    with (out / "groups.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group_id", "group_size", "group_stability", "group_coherence",
                                           "review_status", "model_mode"])
        w.writeheader()
        w.writerows(rows)

    noise = 100.0 * (pred == -1).sum() / n
    summary = {"n": n, "k": len(clusters), "noise_pct": round(noise, 2),
               "seed_noise_pct": round(100.0 * seed_noise / n, 2),
               "over_merge_groups": [r["group_id"] for r in rows if r["review_status"].startswith("over")],
               "mean_coherence": round(float(np.mean([r["group_coherence"] for r in rows])) if rows else 0, 4),
               "mean_stability": round(float(np.mean([r["group_stability"] for r in rows])) if rows else 0, 4),
               "model_mode": model_mode,
               "dial": {"mcs": a.mcs, "ms": a.ms, "method": a.method, "eps": a.eps},
               "reassign": reassign_info}
    if part_summary is not None:
        # ★ 분리 그룹핑을 했으면 **키별 표를 따로 남긴다.** 전체 합산값만 보면
        #   한쪽 코드가 나빠도 다른 쪽에 묻힌다.
        summary["partition_by"] = a.partition_by
        summary["partitions"] = part_summary
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[summary] {json.dumps(summary, ensure_ascii=False)}", flush=True)

    # --- offline_eval (★opt-in only, 라벨 없으면 자동 skip) ---
    if a.offline_eval:
        if not n_labeled:
            print("[offline-eval] skipped: no parent-folder labels found under pool (flat folder) "
                  "-> nothing to score against", flush=True)
        else:
            _sp = str(REPO_ROOT / "scripts")
            if _sp not in sys.path:
                sys.path.append(_sp)
            from eval_may37_checkpoints import _summarize_predictions, _expand_ignored, _drop_megaclusters
            BG = {"Normal", "R", "Random"}
            ignored = _expand_ignored(lab, set(BG))
            measured = ~np.isin(lab, list(ignored))
            fp = _drop_megaclusters(pred.copy(), 0.20)
            r = _summarize_predictions(z[measured], lab[measured], fp[measured], fp, lab, ignored, "deliverable")
            with (out / "offline_eval.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["group_id", "group_size", "majority_label", "purity"])
                w.writeheader()
                w.writerows(off_rows)
            (out / "offline_summary.json").write_text(json.dumps(
                {"P1_capture": r["P1_capture"], "P2_noise_pct": r["P2_noise_pct"],
                 "P3_completeness": r["P3_completeness"], "P4_homogeneity": r["P4_homogeneity"],
                 "Sil_cos": r["Sil_cos"], "fragment_ratio": r["fragment_ratio"],
                 "ARI": r["ARI"], "AMI": r["AMI"], "k": r["k"]}, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[offline] P1={r['P1_capture']} P2noise={r['P2_noise_pct']} "
                  f"P4hom={r['P4_homogeneity']} ARI={r['ARI']}", flush=True)
    else:
        print("[offline-eval] not requested (default OFF) -- pass --offline-eval to score against folder labels", flush=True)

    print(f"[OUT] {out.resolve()}", flush=True)
    print("[DONE_GROUPING_DEPLOY]", flush=True)


if __name__ == "__main__":
    main()
