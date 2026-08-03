#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wafer Defect Auto-Clustering (Contrastive + HDBSCAN, Auto-K)
- Train: TRAIN_DIR만 사용(자기지도), 매 에폭 랜덤 샘플링(비율)
- Embed/Cluster: UNKNOWN 전수
- Aug: small rotate/translate/scale, noise, NO flip
- Loss: Global InfoNCE (+옵션 Queue) + Local InfoNCE
- Embedding L2 normalize
- Saves:
  - clusters/hdbscan/cluster_XXX/... + cluster_XXX.txt
  - clusters_summary.txt    ← run_dir 최상위에 저장
  - clusters_global_list.txt← run_dir 최상위에 저장
  - run.log, run_info.json
  - 대표 이미지: run_dir/cluster_summary/ (medoid)
- Overlay export: 저장 시 overlay가 있으면 overlay 이미지로 저장(없으면 raw)
"""

import os, sys, json, time, random, shutil, warnings, platform, logging
# Windows 콘솔(cp949) 이 em-dash 등 유틸 문자를 못 그려 UnicodeEncodeError로 죽는 걸 방지 —
# 무인 3일 캠페인에서 exit code 오염(정상 완료를 실패로 오탐) 막는 근본 수정 (260726, team-lead directive).
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from datetime import datetime
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
from torchvision.datasets import ImageFolder

import timm
import hdbscan


# ===================== CONFIG =====================
CFG: Dict[str, Any] = {
    # =========================
    # Paths
    # =========================
    "TRAIN_DIR":  "/home/sr5/ho.choi/project/wafer-defect-clustering/data/self",        # 학습 전용(작은 셋)
    "UNKNOWN_DIR": "/home/sr5/ho.choi/project/wafer-defect-clustering/data/unknown",    # 클러스터링 대상(미지)
    "OVERLAY_DIR": "/home/sr5/ho.choi/project/wafer-defect-clustering/data/overlay",    # UNKNOWN_DIR와 동일 구조(오버레이용)
    "OUTPUT_DIR": "/home/sr5/ho.choi/project/wafer-defect-clustering/outputs_hdbscan",  # 결과 저장 루트
    "IMAGE_SIZE": 384,                                                                   # 입력 리사이즈(정사각)

    # =========================
    # Model / Head
    # =========================
    "BACKBONE_NAME": "convnextv2_base.fcmae_ft_in22k_in1k_384",                         # 백본 아키텍처
    "LOCAL_BACKBONE_WEIGHTS": "/home/sr5/ho.choi/project/wafer-defect-clustering/weights/convnextv2_base.fcmae_ft_in22k_in1k_384_91.8_91.7_250724_1505.pth",  # 로컬 가중치 경로(선택)
    "PROJ_DIM": 128,                                                                     # projection head 차원
    "FREEZE_BACKBONE": True,                                                             # 백본 고정(헤드만 학습)

    # =========================
    # Train
    # =========================
    "EPOCHS": 20,                 # 총 에폭
    "WARMUP_EPOCHS": 2,           # 워밍업 에폭(학습 안정화)
    "BATCH": 256,                 # 배치 크기 (GPU 메모리 고려)
    "LR_HEAD": 1e-3,              # 헤드 학습률
    # ★ backbone 부분 해동 (260729). 기본 0 = 기존 동작(완전 freeze) 그대로.
    #   근거: b4 배포본이 이긴 이유가 backbone 을 **아주 약하게 방향성 있게** 미세조정한
    #   것이었다 — FCMAE 대비 중앙값 0.135% 변화. 같은 크기 랜덤섭동 8개는 전부 b4 에
    #   못 미쳤다(`_b4_perturbation_control.py`, runs/b4_perturb/clean546_n8.json).
    #   그런데 우리 레시피는 backbone 을 얼리고 head 만 학습해서 그 축을 아예 못 건드렸다.
    #   N = 뒤에서부터 해동할 stage 수 (ConvNeXt 는 stages.0~3). 1 이면 stages.3 만.
    "UNFREEZE_LAST_N": 0,
    #   해동한 backbone 의 학습률. 0 이면 LR_HEAD/100 (b4 수준의 미세 변화를 노린 값).
    #   ⚠ head LR 을 그대로 쓰면 사전학습 표현이 부서진다 — 반드시 훨씬 작게.
    "LR_BACKBONE": 0.0,
    "WD": 1e-6,                   # weight decay
    "TEMP": 0.07,                 # InfoNCE temperature
    "SEED": 42,                   # 시드(가능한 결정성↑)
    "TRAIN_SAMPLING_RATIO": 0.25, # 0<r<1: 매 에폭 r비율 랜덤 샘플링, 1.0: 전수

    # =========================
    # DataLoader
    # =========================
    "NUM_WORKERS": 32,            # 데이터 로딩 워커 수
    "PIN_MEMORY": True,           # 고정 메모리 pin
    "PREFETCH_FACTOR": 4,         # 프리페치 배수
    "PERSISTENT": True,           # 워커 유지
    "DROP_LAST": False,           # 마지막 미만배치 드롭 여부(재현성↑ 원하면 True 권장)

    # =========================
    # Global InfoNCE + Queue
    # =========================
    "USE_QUEUE": True,            # 큐 사용 (클래스 다양성↑)
    "QUEUE_SIZE": 16384,          # 큐 길이
    "QUEUE_WEIGHT": 1.0,          # 큐 로짓 가중치
    "IGNORE_NEG_SIM": 0.72,       # 이 값 이상 유사한 음성은 무시(거짓-음성 완화)

    # =========================
    # Local InfoNCE (multi-positive)
    # =========================
    "USE_LOCAL": True,            # 로컬(패치) 대비 사용
    "LOCAL_WEIGHT": 0.5,          # 로컬 손실 가중치
    "LOCAL_ANCHORS": "grid36_full", # 앵커 샘플링 패턴
    "LOCAL_SEARCH": "window",     # 매칭 방식('window' 또는 'global')
    "LOCAL_WINDOW": 4,            # window 반경 r (r=4 → 9x9 근방)
    "LOCAL_POS_TOPK": 12,         # 양성으로 채택할 top-k(값 클수록 느슨)
    "LOCAL_POS_MIN_SIM": 0.70,    # 양성 인정 최소 유사도(낮출수록 느슨)
    "LOCAL_SUBBATCH": 32,         # 로컬 계산 서브배치(메모리 절약)
    "LOCAL_EVERY_N": 1,           # 매 스텝마다(2로 늘리면 속도↑)

    # =========================
    # HDBSCAN (클러스터링)
    # =========================
    "HDBSCAN_METRIC": "euclidean",    # 거리(metric). 임베딩 L2 정규화면 'cosine'도 후보
    "MIN_CLUSTER_SIZE": 12,           # 최소 군집 크기(너무 작으면 노이즈로)
    "MIN_SAMPLES": 4,                 # 코어 포인트 기준(차원/밀도 민감)
    "CLUSTER_SELECTION_METHOD": "leaf", # 'leaf'는 세분화, 'eom'은 굵게
    "CLUSTER_SELECTION_EPSILON": 0.06,  # 클러스터 선택 스무딩(크면 합치기 경향)
    "ALLOW_SINGLE_CLUSTER": False,      # 전체가 1클러스터 허용 여부

    # =========================
    # Embedding progress logging
    # =========================
    "EMBED_LOG_EVERY_BATCH": True,  # 배치마다 임베딩 로그(빈도 높으면 I/O↑)
    "EMBED_LOG_TICKS": 20,          # 로그 스텝 간격(클수록 드뭄)
}

# ★ 260722 재현 override — 원본 recipe 유지, 환경만 적응 (Windows + anchor 데이터 + 복구 backbone)
CFG.update({
    "TRAIN_DIR":   r"D:/project/unknown-contrastive/data/images/anchor_avg30_repro",  # 학습 = anchor 2260 (plan: 현 manifest 고정)
    "UNKNOWN_DIR": r"D:/project/unknown-contrastive/data/images/anchor_avg30_repro",  # 클러스터링 대상 = 동일 (transductive)
    "OVERLAY_DIR": "",                                                                 # overlay 우회
    "OUTPUT_DIR":  r"D:/project/unknown-contrastive/runs/may_repro/run",
    "LOCAL_BACKBONE_WEIGHTS": r"D:/project/unknown-contrastive/_b4_backbone.pth",       # ★ b4 진짜 backbone (contrastive_b4.pt 에서 추출; backbone.pth 와 다름)
    "BATCH": 128,        # 16GB 안전 (발산 batch8 대비 16×; feedback_batch_same_condition = same-condition)
    # ★ 플랫폼별로 갈린다 (260730). Windows = 0 (spawn 으로 학습이 죽은 전례, task #25).
    #   Linux(사내 Ubuntu24) = 워커 사용 — 학습이 **6400x6400 단일스레드 디코드**에 GPU 를
    #   굶기고 있었다 (실측 anchor 2.65 img/s = epoch 213s / 565장, GPU 연산 한계 아님).
    #   persistent=False 라 워커 leak 은 없다. REPRO_WORKERS 로 언제든 덮어쓸 수 있다.
    "NUM_WORKERS": 0 if __import__("sys").platform.startswith("win")
                   else min(16, (__import__("os").cpu_count() or 8)),
    "PERSISTENT": False,
})

# ★ env override — 다른 데이터셋(MixedWM38 등) 학습·클러스터용. REPRO_DATA=<dir>, REPRO_BACKBONE=<pth>, REPRO_OUT=<dir>
import os as _os
_rd = _os.environ.get("REPRO_DATA")
if _rd:
    CFG["TRAIN_DIR"] = _rd; CFG["UNKNOWN_DIR"] = _rd
_rb = _os.environ.get("REPRO_BACKBONE")
if _rb:
    CFG["LOCAL_BACKBONE_WEIGHTS"] = _rb
_ro = _os.environ.get("REPRO_OUT")
if _ro:
    CFG["OUTPUT_DIR"] = _ro
if _os.environ.get("REPRO_BATCH"):
    CFG["BATCH"] = int(_os.environ["REPRO_BATCH"])
# ★ recipe knob sweep (coordinate-descent 최적화용). 값 없으면 baseline 유지.
if _os.environ.get("REPRO_TEMP"):    CFG["TEMP"] = float(_os.environ["REPRO_TEMP"])
if _os.environ.get("REPRO_LR"):      CFG["LR_HEAD"] = float(_os.environ["REPRO_LR"])
if _os.environ.get("REPRO_UNFREEZE"):    CFG["UNFREEZE_LAST_N"] = int(_os.environ["REPRO_UNFREEZE"])
# ★ PROJ_DIM 은 그동안 하드코딩(128)이라 한 번도 스윕된 적이 없다. anchor(측정 37클래스)
#   에서 학습한 head 가 P1 을 28/37 -> 17/37 로 **떨어뜨리는** 게 관측돼(260730),
#   1024 -> 128 압축이 클래스 많을 때 병목이라는 가설을 시험하려고 노브를 뚫는다.
if _os.environ.get("REPRO_PROJ_DIM"):    CFG["PROJ_DIM"] = int(_os.environ["REPRO_PROJ_DIM"])
# ★ 학습기 **자체 클러스터링** 다이얼 (cluster_summary/·clusters/ 를 만드는 그것).
#   deploy step 들의 다이얼(config.Cluster)과는 **다른 값**이었고, 여기엔 override 가
#   아예 없어서 config 로도 env 로도 못 바꿨다 — 36장 pool 에서 MIN_CLUSTER_SIZE=12 가
#   너무 커 클러스터가 0개로 나와도 손댈 방법이 없었다 (260731).
#   deploy 경로에서는 `_site_common.train_env` 가 config.Cluster 값을 넣어준다.
if _os.environ.get("REPRO_MCS"):    CFG["MIN_CLUSTER_SIZE"] = int(_os.environ["REPRO_MCS"])
if _os.environ.get("REPRO_MS"):     CFG["MIN_SAMPLES"] = int(_os.environ["REPRO_MS"])
if _os.environ.get("REPRO_METHOD"): CFG["CLUSTER_SELECTION_METHOD"] = _os.environ["REPRO_METHOD"]
if _os.environ.get("REPRO_EPS"):    CFG["CLUSTER_SELECTION_EPSILON"] = float(_os.environ["REPRO_EPS"])
if _os.environ.get("REPRO_METRIC"): CFG["HDBSCAN_METRIC"] = _os.environ["REPRO_METRIC"]
if _os.environ.get("REPRO_LR_BACKBONE"): CFG["LR_BACKBONE"] = float(_os.environ["REPRO_LR_BACKBONE"])
if _os.environ.get("REPRO_QUEUE"):   CFG["QUEUE_SIZE"] = int(_os.environ["REPRO_QUEUE"])
if _os.environ.get("REPRO_NEG"):     CFG["IGNORE_NEG_SIM"] = float(_os.environ["REPRO_NEG"])
if _os.environ.get("REPRO_EPOCHS"):  CFG["EPOCHS"] = int(_os.environ["REPRO_EPOCHS"])
if _os.environ.get("REPRO_SAMPLING"): CFG["TRAIN_SAMPLING_RATIO"] = float(_os.environ["REPRO_SAMPLING"])
if _os.environ.get("REPRO_WORKERS"): CFG["NUM_WORKERS"] = int(_os.environ["REPRO_WORKERS"])  # 큰 이미지 decode 가속 (persistent=False 라 leak 안전)
if _os.environ.get("REPRO_SEED"):    CFG["SEED"] = int(_os.environ["REPRO_SEED"])  # 이미 기본 42 -- paired 비교에서 명시적으로 고정하고 싶을 때만

RUN_TS = datetime.now().strftime("%y%m%d_%H%M%S")


# ===================== Logging =====================
def setup_logger(run_dir: Path) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("wafer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); sh.setLevel(logging.INFO)
    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8"); fh.setFormatter(fmt); fh.setLevel(logging.INFO)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger

def _fmt_hms(secs: float) -> str:
    secs = int(secs)
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    return f"{h:d}:{m:02d}:{s:02d}"


# ===================== Utils / Aug =====================
def seed_all(x=42):
    random.seed(x); np.random.seed(x); torch.manual_seed(x); torch.cuda.manual_seed_all(x)

# module-level (picklable) replacement for a T.Lambda closure — spawned Windows
# DataLoader workers pickle the transform pipeline, and lambdas can't be pickled.
class AddGaussianNoise:
    def __init__(self, std=0.02): self.std = std
    def __call__(self, x): return (x + torch.randn_like(x) * self.std).clamp(0, 1)

def tfm(train=True):
    norm = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    if train:
        return T.Compose([
            T.RandomResizedCrop((CFG["IMAGE_SIZE"], CFG["IMAGE_SIZE"]), scale=(0.94,1.0), ratio=(1.0,1.0), interpolation=T.InterpolationMode.BILINEAR),
            T.RandomAffine(degrees=7, translate=(0.05,0.05), scale=(0.95,1.05)),
            T.ToTensor(),
            AddGaussianNoise(0.02),
            norm
        ])
    return T.Compose([
        T.Resize((CFG["IMAGE_SIZE"], CFG["IMAGE_SIZE"]), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(), norm
    ])

def link_or_copy(src: Path, dst: Path, logger: logging.Logger = None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst); return
    except Exception as e:
        if logger: logger.debug(f"Hardlink fail: {e}")
    try:
        shutil.copy2(src, dst)
    except shutil.SameFileError:
        return

def parse_fields_from_filename(path_str: str):
    base = Path(path_str).stem
    toks = base.split('_')
    if len(toks) >= 5:
        root, step, wafer, ymd, hms = toks[-5:]
    else:
        need = 5 - len(toks); toks = toks + (['']*need)
        root, step, wafer, ymd, hms = toks[:5]
    return root, step, wafer, ymd, hms


# ===================== Data =====================
# ★ palette 전처리 (UC_PALETTE_MASK / UC_PALETTE_MODE).
#   wafer PNG 는 mode="P" 이고 index 0~7 이 결함 grade, 그 위는 칩경계·배경·invalid 다.
#   그 색이 clustering shortcut 이 되므로 경계는 기본 경계색(idx10)으로 **통일**하고
#   나머지는 흰색으로 지운다 (`scripts._common.mask_palette_non_grade_to_white`).
#   ⚠ 260731 이전에는 이 학습기에 마스킹 코드가 **아예 없어서** 항상 raw 로 학습했다.
#     채점(grouping_deploy)만 마스킹하면 학습 때와 다른 입력이 되어 결과가 무의미해진다 —
#     그래서 여기와 채점이 **같은 env** 를 봐야 한다. deploy 는 `train_env()` 로 넘긴다.
_PALETTE_MASK = _os.environ.get("UC_PALETTE_MASK", "0").strip().lower() in ("1", "true", "yes", "on")
try:
    from scripts._common import mask_palette_non_grade_to_white as _mask_palette
except Exception:
    _mask_palette = None


def _open_rgb(p):
    """Image.open -> (선택) palette 마스킹 -> RGB. 학습·임베딩이 같은 경로를 쓴다."""
    im = Image.open(p)
    if _PALETTE_MASK and _mask_palette is not None:
        im = _mask_palette(im)
    return im.convert("RGB")


class PairDS(Dataset):
    def __init__(self, items, t): self.items=items; self.t=t
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p,y,c = self.items[i]
        im = _open_rgb(p)
        try:
            x1 = self.t(im.copy()); x2 = self.t(im.copy())
        finally:
            im.close()
        return x1, x2, y, str(p), c

def build_loader(ds, device):
    kw = {}
    if CFG["NUM_WORKERS"]>0:
        kw = dict(num_workers=CFG["NUM_WORKERS"], prefetch_factor=CFG["PREFETCH_FACTOR"], persistent_workers=CFG["PERSISTENT"])
    return DataLoader(
        ds, batch_size=CFG["BATCH"], shuffle=True, drop_last=CFG["DROP_LAST"],
        pin_memory=(CFG["PIN_MEMORY"] and device.type=="cuda"), **kw
    )


# ===================== Data loading adapter (260726 severstal rehearsal) =====================
# Restored from docs/archive/legacy_code_deleted_paths_260726.zip (_may_repro_src.py).
# ONLY functional change vs the archived original: ImageFolder(CFG[...]) call sites now
# route through _open_dataset(), which additionally accepts a pool-manifest .json
# (scripts/_common.py::load_pool_manifest schema, same schema scripts/make_pool_manifest.py
# emits). A real directory still goes through plain ImageFolder, unchanged -- this keeps the
# original mwm38_clean546 / anchor_avg30_repro champion-recipe runs bit-identical. No physical
# folder copy or symlink is created; the manifest branch only reads an existing file list.
def _open_dataset(root):
    p = Path(root)
    if p.is_file() and p.suffix.lower() == ".json":
        _repo = Path(__file__).resolve().parent
        if str(_repo) not in sys.path:
            sys.path.insert(0, str(_repo))
        from scripts._common import load_pool_manifest
        entries = load_pool_manifest(p)  # [(abs_path, label_or_None), ...]
        labels = [lab if lab is not None else "unlabeled" for _, lab in entries]
        classes = sorted(set(labels))
        c2i = {c: i for i, c in enumerate(classes)}
        samples = [(str(path), c2i[lab]) for (path, _), lab in zip(entries, labels)]
        return SimpleNamespace(classes=classes, samples=samples)
    return ImageFolder(root)


# ===================== Model =====================
class Proj(nn.Module):
    def __init__(self, d, m):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d, bias=False),
            nn.BatchNorm1d(d),
            nn.ReLU(True),
            nn.Linear(d, m)
        )
    def forward(self,x): return self.net(x)

class CL(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(CFG["BACKBONE_NAME"], pretrained=False, num_classes=0, global_pool="")
        # local weights (prefix strip: model./backbone./module.)
        sd = torch.load(CFG["LOCAL_BACKBONE_WEIGHTS"], map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
        for pref in ("model.","backbone.","module."):
            if any(k.startswith(pref) for k in sd.keys()):
                sd = { (k[len(pref):] if k.startswith(pref) else k): v for k,v in sd.items() }
        self.backbone.load_state_dict(sd, strict=False)

        with torch.no_grad():
            dummy = torch.randn(1,3,CFG["IMAGE_SIZE"],CFG["IMAGE_SIZE"])
            fm = self._forward_features(dummy)
            d = fm.shape[1]
        self.proj = Proj(d, CFG["PROJ_DIM"])
        if CFG["FREEZE_BACKBONE"]:
            for p in self.backbone.parameters(): p.requires_grad=False
            # ★ 부분 해동 — 뒤쪽 stage N 개만 다시 켠다 (260729).
            #   앞쪽 stage 는 저수준 특징이라 건드릴 이유가 적고, 해동 범위가 넓을수록
            #   사전학습 표현이 망가질 위험이 커진다. 그래서 뒤에서부터만 연다.
            n_uf = int(CFG.get("UNFREEZE_LAST_N", 0) or 0)
            if n_uf > 0:
                stages = getattr(self.backbone, "stages", None)
                if stages is None:
                    raise RuntimeError("UNFREEZE_LAST_N 을 썼는데 backbone 에 .stages 가 없다 "
                                       "— 이 backbone 구조에선 해동 범위를 정의할 수 없다.")
                n_uf = min(n_uf, len(stages))
                opened = []
                for s in range(len(stages) - n_uf, len(stages)):
                    for p in stages[s].parameters(): p.requires_grad = True
                    opened.append(f"stages.{s}")
                # norm_pre 는 stages 직후라 같이 연다 (있을 때만)
                if hasattr(self.backbone, "norm_pre"):
                    for p in self.backbone.norm_pre.parameters(): p.requires_grad = True
                    opened.append("norm_pre")
                n_tr = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
                n_all = sum(p.numel() for p in self.backbone.parameters())
                print(f"[unfreeze] backbone 부분 해동: {opened}  "
                      f"({n_tr/1e6:.1f}M / {n_all/1e6:.1f}M = {100*n_tr/n_all:.1f}%)", flush=True)
        # ★ residual adapter (REPRO_ADAPTER=1): f + γ·adapt(f), γ=0 init → frozen 하한 보장. backbone 은 여전히 frozen.
        self.use_adapter = _os.environ.get("REPRO_ADAPTER") == "1"
        if self.use_adapter:
            self.adapt = nn.Sequential(nn.Linear(d, d // 8), nn.GELU(), nn.Linear(d // 8, d))
            self.gamma = nn.Parameter(torch.zeros(1))

    def _forward_features(self, x):
        if hasattr(self.backbone, "forward_features"):
            return self.backbone.forward_features(x)
        y = self.backbone(x)
        return y if y.ndim==4 else y[:, :, None, None]

    def forward(self, x, return_local=False):
        x = x.to(memory_format=torch.channels_last)
        fm = self._forward_features(x)          # [B,C,H,W]
        pool = fm.mean(dim=(2,3))               # GAP
        if getattr(self, "use_adapter", False):
            pool = pool + self.gamma * self.adapt(pool)   # residual adapter (γ=0 init)
        z = F.normalize(self.proj(pool), dim=1) # [B,D], L2
        if return_local:
            fm = F.normalize(fm, dim=1)
            return z, fm
        return z


# ===================== Losses =====================
def info_nce_global(z1,z2,t=0.2):
    z1=F.normalize(z1,dim=1); z2=F.normalize(z2,dim=1)
    z=torch.cat([z1,z2],0); sim=z@z.T
    n=sim.size(0); mask=torch.eye(n,device=sim.device,dtype=torch.bool)
    pos = torch.cat([torch.diagonal(sim, offset=z1.size(0)),
                     torch.diagonal(sim, offset=-z1.size(0))], 0).unsqueeze(1)
    neg = sim[~mask].view(n,n-1)
    logit=torch.cat([pos,neg],1)/t
    return F.cross_entropy(logit, torch.zeros(n,dtype=torch.long,device=sim.device))

def info_nce_with_queue(z1, z2, bank, t=0.2, ignore_sim=0.8):
    B, D = z1.size()
    pos = (z1 * z2).sum(1, keepdim=True)                 # [B,1]
    neg_ib = z1 @ z2.T                                   # [B,B]
    neg_ib.fill_diagonal_(-1e9)
    if ignore_sim is not None:
        neg_ib = torch.where(neg_ib >= ignore_sim, torch.full_like(neg_ib, -1e9), neg_ib)

    if bank is not None:
        neg_bank = z1 @ bank.T                           # [B,N]
        if ignore_sim is not None:
            neg_bank = torch.where(neg_bank >= ignore_sim, torch.full_like(neg_bank, -1e9), neg_bank)
        logits = torch.cat([pos, neg_ib, neg_bank], dim=1) / t
    else:
        logits = torch.cat([pos, neg_ib], dim=1) / t

    labels = torch.zeros(B, dtype=torch.long, device=z1.device)
    return F.cross_entropy(logits, labels)


# ---- Local InfoNCE (multi-positive) ----
def info_nce_local_multi(
    f1: torch.Tensor, f2: torch.Tensor,
    anchors=32,
    search: str = 'global',
    r: int = 1,
    pos_topk: int = -1,
    pos_min_sim: float = 0.0,
    t: float = 0.2,
    subbatch: int = 32
) -> torch.Tensor:

    B, C, H, W = f1.shape
    device = f1.device
    search = (search or 'global').lower()

    # ---------- helpers ----------
    def _to_idx(norm, size):
        return max(0, min(size - 1, int(round(norm * (size - 1)))))

    def _grid3x3_rect(i, j):
        y0, y1 = i / 3.0, (i + 1) / 3.0
        x0, x1 = j / 3.0, (j + 1) / 3.0
        return y0, y1, x0, x1

    def _indices_in_rect(H, W, rect):
        y0, y1, x0, x1 = rect
        ys = [y for y in range(H) if y0 <= (y + 0.5) / H < y1]
        xs = [x for x in range(W) if x0 <= (x + 0.5) / W < x1]
        return ys, xs

    def _even_inside_rect(H, W, rect, inner=2):
        y0, y1, x0, x1 = rect
        ys, xs = [], []
        for ii in range(inner):
            for jj in range(inner):
                yn = y0 + (ii + 0.5) / inner * (y1 - y0)
                xn = x0 + (jj + 0.5) / inner * (x1 - x0)
                ys.append(_to_idx(yn, H))
                xs.append(_to_idx(xn, W))
        coord = [(ys[k], xs[k]) for k in range(len(ys))]
        seen, out = set(), []
        for c in coord:
            if c not in seen:
                out.append(c); seen.add(c)
        return out

    def _uniq(seq):
        seen, out = set(), []
        for v in seq:
            if v not in seen:
                out.append(v); seen.add(v)
        return out

    def _uniq_pad_random(coords, target, H, W, device=None):
        uniq = _uniq(coords)
        if len(uniq) >= target:
            return uniq[:target]
        existing = set(uniq)
        rest = [(y, x) for y in range(H) for x in range(W) if (y, x) not in existing]
        if rest:
            idx = torch.randperm(len(rest), device=device)[:(target - len(uniq))].tolist()
            uniq += [rest[i] for i in idx]
        return uniq[:target]

    def _grid_coords(H, W, g=6, shift=(0.0, 0.0)):
        ys = [min(H - 1, max(0, int(round(((i + 0.5 + shift[1]) * H) / g)))) for i in range(g)]
        xs = [min(W - 1, max(0, int(round(((j + 0.5 + shift[0]) * W) / g)))) for j in range(g)]
        return [(y, x) for y in ys for x in xs]

    # ---------- build anchor indices ----------
    anchor_list_per_b = []

    if isinstance(anchors, str):
        if anchors == "grid36_full":
            coords = _grid_coords(H, W, g=6, shift=(0.0, 0.0))
            j_idx = torch.tensor([y * W + x for (y, x) in coords], device=device, dtype=torch.long)
            anchor_list_per_b = [j_idx] * B
        elif anchors == "grid3x3_full":
            j_idx = torch.arange(H * W, device=device, dtype=torch.long)
            anchor_list_per_b = [j_idx] * B
        elif anchors == "grid16_shift4":
            shifts = [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)]
            coords = []
            for sh in shifts:
                coords += _grid_coords(H, W, g=4, shift=sh)
            coords_64 = _uniq_pad_random(coords, target=64, H=H, W=W, device=device)
            j_idx = torch.tensor([y * W + x for (y, x) in coords_64], device=device, dtype=torch.long)
            anchor_list_per_b = [j_idx] * B
        else:
            # 단순화: 위 주요 패턴만 지원
            j_idx = torch.arange(H * W, device=device, dtype=torch.long)
            anchor_list_per_b = [j_idx] * B
    elif anchors is None or (isinstance(anchors, int) and anchors >= H * W):
        j_idx = torch.arange(H * W, device=device, dtype=torch.long)
        anchor_list_per_b = [j_idx] * B
    elif isinstance(anchors, int):
        for _ in range(B):
            idx = torch.randperm(H * W, device=device)[:int(anchors)]
            anchor_list_per_b.append(idx)
    else:
        j_idx = torch.arange(H * W, device=device, dtype=torch.long)
        anchor_list_per_b = [j_idx] * B

    # ---------- compute loss (vectorized over anchors) ----------
    losses = []

    for b in range(B):
        fm1 = f1[b]         # [C,H,W]
        fm2 = f2[b]         # [C,H,W]
        idx = anchor_list_per_b[b]   # [A]
        A = idx.numel()
        if A == 0:
            continue

        v1 = fm1.reshape(C, H * W).T.index_select(0, idx)  # [A,C]

        if search == 'global':
            f2_flat = fm2.reshape(C, H * W).T              # [HW,C]
            sims = v1 @ f2_flat.T                          # [A, HW]

            if pos_topk is None or pos_topk < 0:
                P = sims >= pos_min_sim if pos_min_sim > -1.0 else torch.ones_like(sims, dtype=torch.bool, device=device)
                best = sims.argmax(dim=1, keepdim=True)
                P.scatter_(1, best, True)
            else:
                topk = min(pos_topk, sims.size(1))
                top_idx = sims.topk(k=topk, dim=1, largest=True).indices
                P = torch.zeros_like(sims, dtype=torch.bool, device=device)
                P.scatter_(1, top_idx, True)
                best = sims.argmax(dim=1, keepdim=True)
                P.scatter_(1, best, True)

            S = sims / t
            S = S - S.max(dim=1, keepdim=True).values
            expS = torch.exp(S)
            pos_sum = (expS * P).sum(dim=1) + 1e-12
            all_sum = expS.sum(dim=1) + 1e-12
            loss = -torch.log(pos_sum / all_sum).mean()
            losses.append(loss)

        elif search == 'window':
            k = 2 * r + 1
            U = F.unfold(fm2.unsqueeze(0), kernel_size=(k, k), padding=r, stride=1)[0]
            U = U.view(C, k * k, H * W).index_select(2, idx)     # [C, Nc, A]
            sims = torch.einsum('ac,cna->an', v1, U)             # [A, Nc]

            if pos_topk is None or pos_topk < 0:
                P = sims >= pos_min_sim if pos_min_sim > -1.0 else torch.ones_like(sims, dtype=torch.bool, device=device)
                best = sims.argmax(dim=1, keepdim=True)
                P.scatter_(1, best, True)
            else:
                topk = min(pos_topk, sims.size(1))
                top_idx = sims.topk(k=topk, dim=1, largest=True).indices
                P = torch.zeros_like(sims, dtype=torch.bool, device=device)
                P.scatter_(1, top_idx, True)
                best = sims.argmax(dim=1, keepdim=True)
                P.scatter_(1, best, True)

            S = sims / t
            S = S - S.max(dim=1, keepdim=True).values
            expS = torch.exp(S)
            pos_sum = (expS * P).sum(dim=1) + 1e-12
            all_sum = expS.sum(dim=1) + 1e-12
            loss = -torch.log(pos_sum / all_sum).mean()
            losses.append(loss)
        else:
            raise ValueError(f"search must be 'global' or 'window', got: {search}")

    if len(losses) == 0:
        return torch.tensor(0.0, device=device)

    return torch.stack(losses, dim=0).mean()


# ===================== Queue =====================
class QueueBank(nn.Module):
    def __init__(self, dim, K):
        super().__init__()
        self.K = int(K)
        self.register_buffer("queue", F.normalize(torch.randn(self.K, dim), dim=1))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("filled", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, x):
        if x is None or x.numel() == 0: return
        x = F.normalize(x.detach(), dim=1)
        b = x.size(0)
        k = min(b, self.K)
        p = int(self.ptr.item()); end = p + k
        if end <= self.K:
            self.queue[p:end] = x[:k]
        else:
            n1 = self.K - p
            self.queue[p:] = x[:n1]
            self.queue[:end - self.K] = x[n1:k]
        self.ptr[0] = (p + k) % self.K
        self.filled[0] = min(self.K, int(self.filled.item()) + k)

    @torch.no_grad()
    def get(self):
        n = int(self.filled.item())
        if n <= 0: return None
        return self.queue[:n]


# ===================== Embedding =====================
@torch.no_grad()
def extract(model, items, device, logger: logging.Logger, log_progress: bool = True, log_name: str = "Embed"):
    class SingleView(Dataset):
        def __init__(self,items,t): self.items=items; self.t=t
        def __len__(self): return len(self.items)
        def __getitem__(self,i):
            p,y,c = self.items[i]
            im = _open_rgb(p)          # ★ 학습(PairDS)과 **같은 전처리**여야 한다
            try:
                x = self.t(im)
            finally:
                im.close()
            return x, y, str(p), c

    ds=SingleView(items, tfm(False))
    kw = {}
    if CFG["NUM_WORKERS"]>0:
        kw = dict(num_workers=CFG["NUM_WORKERS"], prefetch_factor=CFG["PREFETCH_FACTOR"], persistent_workers=CFG["PERSISTENT"])
    ld=DataLoader(ds, batch_size=CFG["BATCH"], shuffle=False,
                  pin_memory=(CFG["PIN_MEMORY"] and device.type=="cuda"), **kw)

    total_imgs = len(items)
    total_batches = max(1, len(ld))
    tick = max(1, total_batches // max(1, CFG["EMBED_LOG_TICKS"]))
    t0=time.time()
    _ = next(iter(ld)) if total_batches>0 else None
    logger.info(f"[{log_name} DataLoader warmup] {time.time()-t0:.1f}s (batches={total_batches}, imgs={total_imgs})")

    model.eval(); embs=[]; labs=[]; files=[]; clss=[]
    processed = 0
    start = time.time()

    for it, (xb,yb,fb,cb) in enumerate(ld, 1):
        xb=xb.to(device,non_blocking=True).to(memory_format=torch.channels_last)
        zb = model(xb)
        embs.append(zb.float().cpu().numpy())
        labs += [int(y) for y in yb]; files += list(fb); clss += list(cb)

        # progress
        if log_progress:
            processed += xb.size(0)
            elapsed = time.time() - start
            ips = processed / max(1e-6, elapsed)
            remaining = max(0, total_imgs - processed)
            eta = remaining / max(1e-6, ips)
            pct = 100.0 * processed / max(1, total_imgs)
            do_log = CFG["EMBED_LOG_EVERY_BATCH"] or ((it % tick) == 0) or (it == total_batches)

            if do_log:
                if device.type == "cuda":
                    mem_gb = torch.cuda.memory_allocated() / (1024**3)
                    logger.info(f"[{log_name}] {processed:7d}/{total_imgs:<7d} imgs ({pct:5.1f}%) | {ips:6.1f} img/s | elapsed={_fmt_hms(elapsed)} | eta={_fmt_hms(eta)} | mem={mem_gb:.1f}GB")
                else:
                    logger.info(f"[{log_name}] {processed:7d}/{total_imgs:<7d} imgs ({pct:5.1f}%) | {ips:6.1f} img/s | elapsed={_fmt_hms(elapsed)} | eta={_fmt_hms(eta)}")

    embs=np.concatenate(embs,0).astype(np.float32,copy=False) if embs else np.zeros((0, CFG["PROJ_DIM"]), dtype=np.float32)
    return embs,labs,files,clss


# ===================== Cluster Metrics =====================
def _cluster_neighbor_cohesion(emb: np.ndarray, idxs: np.ndarray, k: int = 5) -> float:
    if len(idxs) <= 1: return 0.0
    E = emb[idxs]
    S = (E @ E.T).astype(np.float32)   # cosine (emb L2-normalized)
    np.fill_diagonal(S, -1e9)
    n = S.shape[0]
    topk = min(k, max(1, n-1))
    part = np.partition(S, -topk, axis=1)[:, -topk:]
    m = part.mean(axis=1)
    return float(np.median(m))

def _cluster_margin_ratio(emb: np.ndarray, idxs: np.ndarray, centers: np.ndarray, my_idx: int) -> float:
    n = len(idxs)
    if n <= 1: return 0.0
    E = emb[idxs]
    ctr = E.mean(0, keepdims=True)
    intra = np.median(np.linalg.norm(E - ctr, axis=1))
    if centers.shape[0] <= 1: return 0.0
    dists = np.linalg.norm(ctr - centers, axis=1)
    dists[my_idx] = np.inf
    nearest_other = float(np.min(dists))
    if intra <= 1e-6: return float('inf')
    return nearest_other / max(intra, 1e-6)

def _label_purity(idxs: np.ndarray, classes: List[str]) -> float:
    if len(idxs)==0: return 0.0
    hist: Dict[str,int] = {}
    for i in idxs:
        c = classes[i]; hist[c] = hist.get(c,0)+1
    m = max(hist.values())
    return m / len(idxs)


# ===================== HDBSCAN + SAVE =====================
def cluster_and_save_hdbscan(
    emb: np.ndarray,
    files: List[str],
    labels: List[int],
    classes: List[str],
    run_dir: Path,
    logger: logging.Logger
):
    """
    - HDBSCAN 수행
    - 클러스터별 이미지 저장 + per-cluster txt
    - 요약: run_dir/clusters_summary.txt (size/확률/지표 + class 분포)
    - 대표 이미지: run_dir/cluster_summary/ 에 메도이드 1장씩
    - 글로벌 리스트: run_dir/clusters_global_list.txt (class 컬럼 포함)
    """

    def _pick_export_src(raw_path_str: str) -> Path:
        raw_p = Path(raw_path_str)
        # overlay 우선
        if CFG.get("OVERLAY_DIR"):
            try:
                rel = raw_p.relative_to(CFG["UNKNOWN_DIR"])
                cand = Path(CFG["OVERLAY_DIR"]) / rel
                if cand.exists():
                    return cand
            except Exception:
                pass
        # 폴백: raw
        return raw_p

    out_clusters_root = Path(run_dir) / "clusters" / "hdbscan"
    out_clusters_root.mkdir(parents=True, exist_ok=True)

    # 대표 이미지 폴더
    summary_dir = Path(run_dir) / "cluster_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # HDBSCAN
    params = dict(
        min_cluster_size=CFG["MIN_CLUSTER_SIZE"],
        min_samples=CFG["MIN_SAMPLES"],
        metric=CFG["HDBSCAN_METRIC"],
        cluster_selection_method=CFG["CLUSTER_SELECTION_METHOD"],
        cluster_selection_epsilon=CFG["CLUSTER_SELECTION_EPSILON"],
        allow_single_cluster=CFG["ALLOW_SINGLE_CLUSTER"]
    )
    logger.info(f"[HDBSCAN] {params}")
    clusterer = hdbscan.HDBSCAN(**params)

    t0 = time.time()
    cids = clusterer.fit_predict(emb)         # -1 = noise
    probs = clusterer.probabilities_          # [N]
    persist = clusterer.cluster_persistence_  # [n_clusters]
    fit_sec = time.time() - t0

    uniq = sorted([int(u) for u in np.unique(cids) if u >= 0])   # 클러스터 라벨 목록
    noise_idx = np.where(cids == -1)[0].tolist()
    pers_map = {lab: float(persist[i]) for i, lab in enumerate(uniq)}

    # ----- stats 수집 -----
    stats = []
    for lab in uniq:
        idxs = np.where(cids == lab)[0]
        size = int(len(idxs))
        if size > 0:
            medp = float(np.median(probs[idxs]))
            meanp = float(np.mean(probs[idxs]))
            stdp = float(np.std(probs[idxs]))
        else:
            medp = meanp = stdp = 0.0
        stats.append({
            "lab": lab,
            "idxs": idxs,
            "size": size,
            "median_prob": medp,
            "mean_prob": meanp,
            "std_prob": stdp
        })

    # 클러스터 중심(centers)
    centers = []
    for s in stats:
        if s["size"] > 0:
            centers.append(emb[s["idxs"]].mean(0))
        else:
            centers.append(np.zeros((emb.shape[1],), dtype=emb.dtype))
    centers = np.stack(centers, 0) if len(centers) > 0 else np.zeros((0, emb.shape[1]), dtype=emb.dtype)

    # 고급 지표
    for ci, s in enumerate(stats):
        s["persistence"] = pers_map.get(s["lab"], 0.0)
        s["cohesion"] = _cluster_neighbor_cohesion(emb, s["idxs"], k=5)
        s["margin"] = _cluster_margin_ratio(emb, s["idxs"], centers, my_idx=ci)
        s["purity"] = _label_purity(s["idxs"], classes)

    # 요약 로그
    def _pct_local(a, ps):
        if len(a) == 0: return {p:0.0 for p in ps}
        return {p: float(np.percentile(a, p)) for p in ps}

    sizes = [s["size"] for s in stats]
    medps = [s["median_prob"] for s in stats]
    perss = [s["persistence"] for s in stats]
    p_sizes = _pct_local(sizes, [10,25,50,75,90]) if sizes else {}
    p_medp  = _pct_local(medps, [10,25,50,75,90]) if medps else {}
    p_pers  = _pct_local(perss, [10,25,50,75,90]) if perss else {}

    logger.info("\n[HDBSCAN Summary]")
    logger.info(f"  samples={len(files)} | fit_time={fit_sec:.1f}s")
    logger.info(f"  clusters_found(pre-filter)={len(uniq)} | noise={len(noise_idx)}")
    if sizes:
        logger.info(f"  size percentiles  : P10={p_sizes[10]:.0f} P25={p_sizes[25]:.0f} P50={p_sizes[50]:.0f} P75={p_sizes[75]:.0f} P90={p_sizes[90]:.0f}")
    if medps:
        logger.info(f"  prob percentiles  : P10={p_medp[10]:.2f} P25={p_medp[25]:.2f} P50={p_medp[50]:.2f} P75={p_medp[75]:.2f} P90={p_medp[90]:.2f}")
    if perss:
        logger.info(f"  pers percentiles  : P10={p_pers[10]:.2f} P25={p_pers[25]:.2f} P50={p_pers[50]:.2f} P75={p_pers[75]:.2f} P90={p_pers[90]:.2f}")

    # ----- 저장 준비 -----
    per_cluster_class_counts: Dict[int, Dict[str,int]] = {}
    global_records: List[Tuple[str,str,str,str,str,str,str]] = []  # (clusterno, class, root, step, wafer, ymd, hms)

    # 개별 클러스터 저장
    def write_cluster(srow):
        lab, idxs = srow["lab"], srow["idxs"]
        # ← 추가: 폴더/파일 이름 공통 태그
        _lots = {Path(files[j]).stem.split("_", 1)[0] for j in idxs}
        tag = f"cluster_{lab:03d}_l{len(_lots)}_w{srow['size']}"

        # 기존: cdir = out_clusters_root / f"cluster_{lab:03d}"
        cdir = out_clusters_root / tag
        cdir.mkdir(parents=True, exist_ok=True)

        hist: Dict[str,int] = {}
        recs = []
        for i in idxs:
            cname = classes[i]
            hist[cname] = hist.get(cname, 0) + 1
            src = _pick_export_src(files[i])
            dst = cdir / f"{cname}_{src.name}"
            link_or_copy(src, dst, logger)

            root_f, step_f, wafer_f, ymd, hms = parse_fields_from_filename(files[i])
            recs.append((cname, root_f, step_f, wafer_f, ymd, hms))
            global_records.append((f"{lab:03d}", cname, root_f, step_f, wafer_f, ymd, hms))

        per_cluster_class_counts[lab] = dict(hist)

        # 대표 이미지(메도이드)
        if len(idxs) > 0:
            E = emb[idxs]
            ctr = E.mean(0, keepdims=True)
            d = np.linalg.norm(E - ctr, axis=1)
            local = int(np.argmin(d))
            rep_idx = int(idxs[local])
            src = _pick_export_src(files[rep_idx])
            # 기존: rep_name = f"cluster_{lab:03d}__{classes[rep_idx]}__..."
            rep_name = f"{tag}__{classes[rep_idx]}__medoid_dist{d[local]:.4f}{src.suffix}"
            link_or_copy(src, summary_dir / rep_name, logger)

            # ★ 클러스터별 composite map — medoid 옆에 같이 둔다 (260731).
            #   예전엔 `cluster_summary/` 에 medoid 1장만 있어서, 이 폴더로 클러스터를
            #   검토할 때 "그 그룹이 전체적으로 어떤 모양인가"를 볼 수 없었다.
            #   composite 는 원본 6400x6400 을 다시 읽어야 해 비싸므로 **중심에 가까운
            #   순으로 상한(Composite.MAX_MEMBERS, 기본 10)** 만 쓴다.
            #   실패해도 학습 결과를 버리지 않는다 — 경고만 남기고 계속한다.
            try:
                import os as _o
                _mx = int(_o.environ.get("SITE_COMPOSITE_MAX_MEMBERS", 10))
                _order = [int(idxs[j]) for j in np.argsort(d)]
                _use = _order if _mx <= 0 else _order[:_mx]
                _srcs = [str(_pick_export_src(files[j])) for j in _use]
                import sys as _s
                _dep = str(Path(__file__).resolve().parent / "deploy")
                if _dep not in _s.path:
                    _s.path.append(_dep)
                from composite import write_group_composites as _wgc
                _wgc(_srcs, summary_dir, prefix=f"{tag}__")
                logger.info(f"  [composite] {tag} 멤버 {len(idxs)} 중 {len(_use)}장 -> cluster_summary/")
            except Exception as _e:
                import traceback as _tb
                logger.warning(f"  [composite] {tag} 실패: {type(_e).__name__}: {_e}")
                logger.warning(_tb.format_exc())

        # per-cluster txt (이름도 태그로 맞춤)
        # 기존: with open(cdir / f"cluster_{lab:03d}.txt", ...
        with open(cdir / f"{tag}.txt", "w", encoding="utf-8") as f:
            f.write(
                f"[Cluster {lab:03d}] status=raw size={srow['size']}, "
                f"median_prob={srow['median_prob']:.3f}, mean_prob={srow['mean_prob']:.3f}, std_prob={srow['std_prob']:.3f}, "
                f"persistence={srow['persistence']:.3f}, cohesion={srow['cohesion']:.2f}, margin={srow['margin']:.2f}, purity={srow['purity']:.2f}\n"
            )
            f.write("Class counts:\n")
            for k, v in sorted(hist.items(), key=lambda x: (-x[1], x[0])):
                f.write(f"  - {k}: {v}\n")
            f.write("\nclass\troot\tstep\twafer\tyyyymmdd\thhmmss\n")
            for (cname, root_f, step_f, wafer_f, ymd, hms) in recs:
                f.write(f"{cname}\t{root_f}\t{step_f}\t{wafer_f}\t{ymd}\t{hms}\n")


    for s in stats:
        write_cluster(s)

    # clusters_summary.txt → run_dir 최상위에 저장
    with open(Path(run_dir) / "clusters_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Total: {len(files)}\n")
        f.write(f"HDBSCAN fit_time: {fit_sec:.1f}s\n")
        f.write(f"Clusters found (pre-filter): {len(uniq)}\n")
        f.write(f"Noise: {len(noise_idx)}\n\n")

        f.write("===== Per-cluster counts by class =====\n")
        for s in stats:
            lab = s["lab"]
            f.write(
                f"[Cluster {lab:03d}] size={s['size']}, "
                f"median_prob={s['median_prob']:.2f}, mean_prob={s['mean_prob']:.2f}, std_prob={s['std_prob']:.2f}, "
                f"persistence={s['persistence']:.2f}, cohesion={s['cohesion']:.2f}, margin={s['margin']:.2f}, purity={s['purity']:.2f}\n"
            )
            cnts = per_cluster_class_counts.get(lab, {})
            for cname, v in sorted(cnts.items(), key=lambda x: (-x[1], x[0])):
                f.write(f"  - {cname}: {v}\n")
        f.write("\n")

        f.write("===== Global list (tab-separated) =====\n")
        f.write("clusterno\tclass\troot\tstep\twafer\tyyyymmdd\thhmmss\n")
        for (clno, cname, root_f, step_f, wafer_f, ymd, hms) in global_records:
            f.write(f"{clno}\t{cname}\t{root_f}\t{step_f}\t{wafer_f}\t{ymd}\t{hms}\n")

    # clusters_global_list.txt → run_dir 최상위
    with open(Path(run_dir) / "clusters_global_list.txt", "w", encoding="utf-8") as f:
        f.write("clusterno\tclass\troot\tstep\twafer\tyyyymmdd\thhmmss\n")
        for (clno, cname, root_f, step_f, wafer_f, ymd, hms) in global_records:
            f.write(f"{clno}\t{cname}\t{root_f}\t{step_f}\t{wafer_f}\t{ymd}\t{hms}\n")


# ===================== Save run info =====================
def save_run_info(run_dir: Path, device: torch.device, class_to_idx: Dict[str,int]):
    info = {
        "timestamp": RUN_TS,
        "device": str(device),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "timm": getattr(timm, "__version__", "unknown"),
        "hdbscan": getattr(hdbscan, "__version__", "unknown"),
        "cfg": CFG,
        "class_to_idx": class_to_idx,
    }
    with open(run_dir/"run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)


# ===================== Main =====================
def main():
    _gpu_fraction_raw = _os.environ.get("REPRO_GPU_MEMORY_FRACTION")
    if _gpu_fraction_raw is not None:
        try:
            _gpu_fraction = float(_gpu_fraction_raw)
        except ValueError as exc:
            raise ValueError("REPRO_GPU_MEMORY_FRACTION must be a float in (0, 1]") from exc
        if not 0.0 < _gpu_fraction <= 1.0:
            raise ValueError("REPRO_GPU_MEMORY_FRACTION must be in (0, 1]")
        if not torch.cuda.is_available():
            raise RuntimeError("REPRO_GPU_MEMORY_FRACTION was requested but CUDA is unavailable")
        # Must precede seed_all(), whose manual_seed_all can initialize CUDA.
        torch.cuda.set_per_process_memory_fraction(_gpu_fraction, device=0)
    seed_all(CFG["SEED"])
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # ★ device 는 **요청받은 대로** 쓴다. 예전엔 CUDA 가 없으면 조용히 CPU 로 떨어졌다 —
    #   6400x6400 이미지로 CPU 학습을 하면 정상처럼 보이면서 수십 배 느려지고,
    #   사내 서버에서 "step2 가 GPU 를 전혀 안 쓴다(nvidia-smi 0)"로 관측됐다 (260730).
    #   REPRO_DEVICE=cuda 를 명시했는데 CUDA 가 없으면 **진단을 찍고 죽는다.**
    _dev_req = (_os.environ.get("REPRO_DEVICE") or "").strip().lower()
    _cuda_ok = torch.cuda.is_available()
    if _dev_req.startswith("cuda") and not _cuda_ok:
        raise RuntimeError(
            "REPRO_DEVICE=cuda 인데 torch.cuda.is_available()==False — CPU 로 조용히 "
            "떨어지지 않고 여기서 멈춘다.\n"
            f"  torch            : {torch.__version__}\n"
            f"  torch CUDA build : {torch.version.cuda}\n"
            f"  device_count     : {torch.cuda.device_count()}\n"
            f"  CUDA_VISIBLE_DEVICES = {_os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}\n"
            "  흔한 원인: (1) CPU-only torch wheel 설치 — torch.version.cuda 가 None 이면 이것.\n"
            "             (2) 드라이버/CUDA 불일치  (3) CUDA_VISIBLE_DEVICES='' 로 가려짐\n"
            "  CPU 로 진짜 돌리려면 REPRO_DEVICE=cpu 를 명시해라.")
    if _dev_req == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if _cuda_ok else "cpu")
        if not _cuda_ok:
            print("[warn] ★ CUDA 를 못 찾아 **CPU 로 학습한다** — 6400x6400 이미지에선 "
                  f"수십 배 느리다. torch={torch.__version__} cuda_build={torch.version.cuda} "
                  f"CUDA_VISIBLE_DEVICES={_os.environ.get('CUDA_VISIBLE_DEVICES','(unset)')}",
                  flush=True)
    run_dir = Path(f"{CFG['OUTPUT_DIR']}_{RUN_TS}")
    logger = setup_logger(run_dir)
    logger.info("===== RUN START =====")
    logger.info(f"Device: {device}")
    logger.info("CFG:")
    logger.info(json.dumps(CFG, indent=2, ensure_ascii=False))

    # Data
    # ★ 절대경로 + 장수를 먼저 찍는다 — 상대경로가 어느 폴더로 풀렸는지,
    #   실제 몇 장이 잡히는지가 학습 사고의 대부분이다 (deploy/step 들의 [images] 와 동일 취지).
    logger.info(f"[images] TRAIN_DIR   = {Path(CFG['TRAIN_DIR']).resolve()}")
    logger.info(f"[images] UNKNOWN_DIR = {Path(CFG['UNKNOWN_DIR']).resolve()}")
    ds_u = _open_dataset(CFG["UNKNOWN_DIR"])               # 전수 임베딩/클러스터링 대상
    ds_t = _open_dataset(CFG["TRAIN_DIR"])                 # 학습 대상(작은 셋)
    logger.info(f"[images] TRAIN_DIR   이미지 수 = {len(ds_t.samples):,} 장  "
                f"({len(ds_t.classes)} class)")
    logger.info(f"[images] UNKNOWN_DIR 이미지 수 = {len(ds_u.samples):,} 장  "
                f"({len(ds_u.classes)} class)")

    # class_to_idx는 UNKNOWN 기준으로 기록
    class_to_idx = {c: i for i, c in enumerate(ds_u.classes)}
    logger.info(f"[TRAIN_DIR classes] {ds_t.classes}")
    logger.info(f"[UNKNOWN_DIR classes] {ds_u.classes}")

    # === 학습 셋 풀리스트 (에폭별 샘플링용) ===
    items_train_full = [(Path(p), -1, ds_t.classes[int(y)]) for (p, y) in ds_t.samples]
    logger.info(f"[Train from TRAIN_DIR] full_items={len(items_train_full)}")

    # === 임베딩/클러스터링 셋: UNKNOWN 전수 ===
    unknown_full = [(Path(p), -1, ds_u.classes[int(y)]) for (p, y) in ds_u.samples]
    items_eval = unknown_full
    if CFG.get("OVERLAY_DIR"):
        miss = sum(1 for (p,_,_) in items_eval if not (Path(CFG["OVERLAY_DIR"]) / Path(p).relative_to(CFG["UNKNOWN_DIR"])).exists())
        logger.info(f"[Overlay coverage] missing={miss} / total={len(items_eval)}")
    logger.info(f"[Eval set] unknown_only={len(unknown_full)}")

    # * 260726 post-mortem: previous run was killed when a concurrent repo
    # cleanup permanently deleted its D: local data folder mid-training
    # (docs/CLEANUP_MANIFEST_260726.md). Write a lock file recording the
    # active run + data source so cleanup tooling can check this instead of
    # relying on a process snapshot alone.
    _lock_path = Path(run_dir) / ".RUNNING"
    _lock_path.write_text(json.dumps({
        "pid": os.getpid(),
        "started_at": RUN_TS,
        "train_dir": CFG["TRAIN_DIR"],
        "unknown_dir": CFG["UNKNOWN_DIR"],
        "n_train_files": len(items_train_full),
        "n_eval_files": len(unknown_full),
    }, indent=2), encoding="utf-8")

    # Model / Optim
    model = CL().to(device).to(memory_format=torch.channels_last)
    # 옵티마 — backbone 을 부분 해동했으면 **별도 param group 에 훨씬 작은 LR** 을 준다.
    # ★ head LR(보통 4e-3)을 사전학습 backbone 에 그대로 먹이면 표현이 부서진다.
    #   목표는 b4 수준(중앙값 0.135%)의 미세한 방향성 변화라 기본은 LR_HEAD/100.
    _bb_ids = {id(p) for p in model.backbone.parameters()}
    _bb = [p for p in model.parameters() if p.requires_grad and id(p) in _bb_ids]
    _hd = [p for p in model.parameters() if p.requires_grad and id(p) not in _bb_ids]
    if _bb:
        lr_bb = float(CFG.get("LR_BACKBONE") or 0.0) or CFG["LR_HEAD"] / 100.0
        groups = [{"params": _hd, "lr": CFG["LR_HEAD"]},
                  {"params": _bb, "lr": lr_bb}]
        print(f"[optim] head {sum(p.numel() for p in _hd)/1e6:.2f}M @ lr={CFG['LR_HEAD']:.2e}  |  "
              f"backbone {sum(p.numel() for p in _bb)/1e6:.1f}M @ lr={lr_bb:.2e}", flush=True)
    else:
        groups = [{"params": _hd, "lr": CFG["LR_HEAD"]}]
    opt = torch.optim.AdamW(groups, lr=CFG["LR_HEAD"], weight_decay=CFG["WD"])
    # 워밍업이 group 별 LR 을 덮어쓰지 않도록 각자의 기준 LR 을 보관해 둔다.
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type=="cuda"))
    neg_bank = QueueBank(CFG["PROJ_DIM"], CFG["QUEUE_SIZE"]).to(device) if CFG["USE_QUEUE"] else None

    # Train (에폭마다 랜덤 샘플링 + 로더 생성)
    logger.info("[학습 시작]")
    for ep in range(1, CFG["EPOCHS"]+1):
        # LR warmup (옵션)
        if CFG["WARMUP_EPOCHS"] > 0 and ep <= CFG["WARMUP_EPOCHS"]:
            # ★ group 마다 **자기 기준 LR** 에 비율을 곱한다. 예전처럼 한 값으로 덮으면
            #   backbone group 이 head LR 을 받아 사전학습 표현이 부서진다 (260729).
            _r = ep / max(1, CFG["WARMUP_EPOCHS"])
            for g in opt.param_groups:
                g["lr"] = g.get("base_lr", CFG["LR_HEAD"]) * _r
            _lrs = [round(g["lr"], 10) for g in opt.param_groups]
            logger.info(f"[LR warmup] epoch {ep} ratio={_r:.2f} lrs={_lrs}")

        # 에폭별 샘플링
        r = float(CFG.get("TRAIN_SAMPLING_RATIO", 1.0))
        if 0.0 < r < 1.0:
            rng = random.Random(CFG["SEED"] + ep)
            k = max(1, int(len(items_train_full) * r))
            items_train = rng.sample(items_train_full, k)
            logger.info(f"[Train sampling@epoch{ep}] ratio={r:.2f} -> {len(items_train)} items")
        else:
            items_train = items_train_full
            logger.info(f"[Train sampling@epoch{ep}] ratio=1.00 (full) -> {len(items_train)} items")

        # 에폭별 DataLoader 생성 + 워밍업
        train_ds = PairDS(items_train, tfm(True))
        ld = build_loader(train_ds, device)
        t0dl = time.time(); _ = next(iter(ld)); logger.info(f"[Train DataLoader warmup@epoch{ep}] {time.time()-t0dl:.1f}s (batches={len(ld)}, imgs={len(items_train)})")

        # 학습 루프
        t0=time.time(); model.train()
        run_g=run_q=run_l=0.0; n=0
        total=len(ld); tick=max(1,total//20)

        for it, (x1,x2,_,_,_) in enumerate(ld, 1):
            x1=x1.to(device,non_blocking=True); x2=x2.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            dtype=(torch.bfloat16 if (device.type=="cuda" and torch.cuda.is_bf16_supported()) else torch.float16)
            with torch.amp.autocast("cuda", enabled=(device.type=="cuda"), dtype=dtype):
                z1,fm1 = model(x1, return_local=True); z2,fm2 = model(x2, return_local=True)

                # Global InfoNCE (+옵션 Queue)
                loss_g = info_nce_global(z1,z2,CFG["TEMP"])
                if CFG["USE_QUEUE"] and neg_bank is not None:
                    bank = neg_bank.get()
                    lq1 = info_nce_with_queue(z1, z2, bank, CFG["TEMP"], CFG["IGNORE_NEG_SIM"])
                    lq2 = info_nce_with_queue(z2, z1, bank, CFG["TEMP"], CFG["IGNORE_NEG_SIM"])
                    loss_q = 0.5*(lq1 + lq2) * CFG["QUEUE_WEIGHT"]
                else:
                    loss_q = torch.tensor(0.0, device=z1.device)

                # Local InfoNCE
                if CFG["USE_LOCAL"] and ((it % CFG["LOCAL_EVERY_N"]) == 0):
                    loss_l = info_nce_local_multi(
                        fm1, fm2,
                        anchors=CFG["LOCAL_ANCHORS"],
                        search=CFG["LOCAL_SEARCH"],
                        r=CFG["LOCAL_WINDOW"],
                        pos_topk=CFG["LOCAL_POS_TOPK"],
                        pos_min_sim=CFG["LOCAL_POS_MIN_SIM"],
                        t=CFG["TEMP"],
                        subbatch=CFG["LOCAL_SUBBATCH"]
                    ) * CFG["LOCAL_WEIGHT"]
                else:
                    loss_l = torch.tensor(0.0, device=z1.device)

                loss = loss_g + loss_q + loss_l

            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            with torch.no_grad():
                if CFG["USE_QUEUE"] and neg_bank is not None:
                    neg_bank.enqueue(z2)

            run_g += float(loss_g); run_q += float(loss_q); run_l += float(loss_l); n+=1
            if (it%tick)==0 or it==total:
                logger.info(f"  Epoch {ep}/{CFG['EPOCHS']} | {100.0*it/total:5.1f}% ({it}/{total}) | G={run_g/n:.4f} Q={run_q/n:.4f} L={run_l/n:.4f}")

        logger.info(f"Epoch {ep} done | G={run_g/max(1,n):.4f} Q={run_q/max(1,n):.4f} L={run_l/max(1,n):.4f} | time={time.time()-t0:.1f}s")

        # ★ per-epoch proj(+adapter) 저장 (canonical HDBSCAN 채점용, epoch 1~20 전부)
        _ck = Path(run_dir) / "checkpoints"; _ck.mkdir(parents=True, exist_ok=True)
        _sv = {"proj": model.proj.state_dict(), "epoch": ep,
               "G": run_g/max(1,n), "Q": run_q/max(1,n), "L": run_l/max(1,n)}
        if getattr(model, "use_adapter", False):
            _sv["adapt"] = model.adapt.state_dict(); _sv["gamma"] = model.gamma.detach().cpu()
        torch.save(_sv, _ck / f"proj_ep{ep}.pt")
        logger.info(f"[save] proj_ep{ep}.pt")

    # save checkpoints + run info
    ckpt_dir = Path(run_dir) / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    if _os.environ.get("REPRO_LEAN") == "1":
        # ★ 스윕 lean: full-model(~700MB/run) 저장 skip. proj_ep(작음)만 유지 → 디스크 절약.
        logger.info("[lean] full-model ckpt(final_infer/last_training) skip — proj_ep 만 유지")
    else:
        torch.save({"state_dict": model.state_dict()}, ckpt_dir / "final_infer.pt")
        torch.save({
            "epoch": CFG["EPOCHS"],
            "model": model.state_dict(),
            "cfg": CFG,
            "class_to_idx": class_to_idx,
        }, ckpt_dir / "last_training.pt")
        logger.info(f"[저장] {(ckpt_dir/'final_infer.pt').resolve()}")
    save_run_info(run_dir, device, class_to_idx)
    if device.type == "cuda":
        logger.info(
            "[gpu-memory] configured_fraction=%s max_allocated_mb=%.1f max_reserved_mb=%.1f",
            _gpu_fraction_raw,
            torch.cuda.max_memory_allocated(0) / (1024 ** 2),
            torch.cuda.max_memory_reserved(0) / (1024 ** 2),
        )

    if _os.environ.get("REPRO_SKIP_FINAL_EMBED") == "1":
        logger.info("[skip] final embedding/HDBSCAN export (epoch checkpoints preserved)")
        _lock_path.unlink(missing_ok=True)
        logger.info(f"[완료] 결과: {run_dir.resolve()}")
        logger.info("===== RUN END =====")
        return

    # Embedding + HDBSCAN (최종 embedding 저장 — canonical 채점은 별도 _score_may_repro.py)
    logger.info("[임베딩 추출]")
    emb,labs,files,clss = extract(model, items_eval, device, logger, log_progress=True, log_name="Embed")
    import numpy as _np
    _np.save(Path(run_dir) / "final_emb.npy", emb.astype("float32"))
    with open(Path(run_dir) / "final_labels.txt", "w", encoding="utf-8") as _f:
        _f.write("\n".join(clss))
    logger.info(f"[save] final_emb.npy {emb.shape}")
    try:
        logger.info("[HDBSCAN + 저장]"); cluster_and_save_hdbscan(emb, files, labs, clss, run_dir, logger)
    except Exception as _e:
        logger.warning(f"[HDBSCAN export 우회] {_e}")

    logger.info(f"[완료] 결과: {run_dir.resolve()}")
    logger.info("===== RUN END =====")
    _lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
