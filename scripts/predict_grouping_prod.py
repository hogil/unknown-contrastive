#!/usr/bin/env python3
"""현업 grouping predict — contrastive best_model.pt 로 폴더 내 wafer cluster.

폴더 구조 (제품/라인/날짜):
    IMAGE_ROOT/AA/K1AA/20260502/*.png
    IMAGE_ROOT/AA/K1AB/20260502/*.png
    IMAGE_ROOT/BB/K1BA/20260502/*.png
    ...

또는 단일 폴더:
    IMAGE_ROOT/*.png 또는 IMAGE_ROOT 의 모든 하위 .png

산출:
    OUTPUT_DIR/{product}/{line}/{date}/clusters.parquet  (DB ingestion 용)
    OUTPUT_DIR/{product}/{line}/{date}/summary.json
    OUTPUT_DIR/{product}/{line}/{date}/groups/<group_id>/*.png  (옵션 — 시각 review)
"""
from __future__ import annotations

# ===================================================================
# === CONFIG (실행 시 이 부분만 수정) ===
# ===================================================================
# 입력 — 옵션 A: 단일 폴더 지정 (모든 .png 하위 walk)
IMAGE_ROOT          = "E:/data/images/prod/AA/K1AA/20260502"  # ★ 절대규: 모든 이미지 E:/data/images/

# 옵션 B: 여러 (product, line, date) 자동 walk
# IMAGE_BASE 만 주면 IMAGE_BASE/<product>/<line>/<date>/*.png 자동 enum
IMAGE_BASE          = None         # 예: "data/prod" — 비우면 IMAGE_ROOT 만 사용
PRODUCT_FILTER      = None         # 예: ["AA", "BB"] — None=all
LINE_FILTER         = None
DATE_FILTER         = None

# 옵션 C: 실전 폴더 여러 개 직접 선택 (--image-roots a,b,c). IMAGE_BASE/IMAGE_ROOT 보다 우선.
#   POOL=False(기본): 각 폴더 따로 grouping (폴더별 출력). POOL=True: 전부 합쳐 1 grouping.
IMAGE_ROOTS         = None         # list[str] 또는 None
POOL                = False
POOL_NAME           = "pooled"

# 모델 — contrastive best
MODEL_PATH          = "runs/<TS>_pipeline/contrastive/best_model.pt"

# 출력
OUTPUT_DIR          = "result_grouping"
COPY_PNG_TO_GROUPS  = False        # True 면 group 폴더에 PNG 복사 (디스크 증가)

# Inference
IMG_SIZE            = 384
BATCH               = 32
NUM_WORKERS         = 4
PROGRESS_EVERY      = 20          # embedding loop 진행률 출력 batch 간격

# HDBSCAN
MIN_CLUSTER_SIZE    = 12
MIN_SAMPLES         = 3
CLUSTER_SELECTION_METHOD = "eom"
CLUSTER_SELECTION_EPSILON = 0.0

# Backbone (같은 architecture 가정)
BACKBONE            = "convnextv2_base.fcmae_ft_in22k_in1k_384"
PROJ_DIM            = 128

SEED                = 42
# ===================================================================

import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from _common import (
    log_stage_metric,
    make_run_dir,
    resolve_path,
    snapshot_config,
    system_info,
)


class FolderImageDataset(Dataset):
    """폴더 하위 모든 .png (recursive). class 없음 — grouping 전용."""
    EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    def __init__(self, root, transform=None):
        roots = root if isinstance(root, (list, tuple)) else [root]
        self.roots = [Path(r) for r in roots]
        self.root = self.roots[0]
        self.transform = transform
        paths: list[Path] = []
        for r in self.roots:
            for ext in self.EXTS:
                paths.extend(Path(r).rglob(f"*{ext}"))
        self.paths = sorted(set(paths))

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"[CORRUPT-SKIP] {p}: {type(e).__name__}", flush=True)
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(0, 0, 0))
        if self.transform is not None:
            img = self.transform(img)
        return img, str(p)


def build_eval_tf():
    norm = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        norm,
    ])


class ContrastiveInferModel(nn.Module):
    """backbone + projection head — same architecture as train_contrastive.ContrastiveModel."""
    def __init__(self, backbone_name: str, proj_dim: int):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone_name, pretrained=False,
                                          num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim), nn.GELU(),
            nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x):
        f = self.backbone(x)
        z = self.proj(f)
        return F.normalize(z, dim=1)


def enumerate_targets():
    """반환: (product, line, date, folder_or_list, out_name) 리스트.
    우선순위: IMAGE_ROOTS(직접 선택) > IMAGE_BASE(walk) > IMAGE_ROOT(단일)."""
    # 옵션 C — 실전 폴더 직접 선택
    if IMAGE_ROOTS:
        roots = [resolve_path(r) for r in IMAGE_ROOTS]
        if POOL:
            return [(None, None, None, roots, POOL_NAME)]          # 전부 합쳐 1 grouping
        out = []
        seen = {}
        for r in roots:
            name = r.name
            if name in seen:                                        # 폴더명 중복 → 부모 붙여 구분
                seen[name] += 1; name = f"{r.parent.name}_{r.name}"
            else:
                seen[name] = 1
            out.append((None, None, None, r, name))
        return out
    # 옵션 A — 단일
    if not IMAGE_BASE:
        f = resolve_path(IMAGE_ROOT)
        return [(None, None, None, f, f.name)]
    # 옵션 B — product/line/date walk
    base = resolve_path(IMAGE_BASE)
    targets = []
    for prod in sorted(base.iterdir()):
        if not prod.is_dir(): continue
        if PRODUCT_FILTER and prod.name not in PRODUCT_FILTER: continue
        for line in sorted(prod.iterdir()):
            if not line.is_dir(): continue
            if LINE_FILTER and line.name not in LINE_FILTER: continue
            for date in sorted(line.iterdir()):
                if not date.is_dir(): continue
                if DATE_FILTER and date.name not in DATE_FILTER: continue
                targets.append((prod.name, line.name, date.name, date, None))
    return targets


def cluster_folder(folder_path, model: nn.Module, device, output_subdir: Path,
                   product=None, line=None, date=None):
    """한 폴더(또는 폴더 list)의 이미지 cluster 후 산출."""
    tf = build_eval_tf()
    ds = FolderImageDataset(folder_path, transform=tf)
    disp = (", ".join(str(Path(p)) for p in folder_path)
            if isinstance(folder_path, (list, tuple)) else str(folder_path))
    if len(ds) == 0:
        print(f"[skip] {disp}: no images")
        return None
    print(f"[{disp}] {len(ds)} images", flush=True)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    # embed
    all_z, all_path = [], []
    model.eval()
    t0 = time.time()
    total_batches = len(loader)
    print(f"[embed] start batch={BATCH} workers={NUM_WORKERS} batches={total_batches}", flush=True)
    with torch.no_grad():
        for bi, (imgs, paths) in enumerate(loader, start=1):
            imgs = imgs.to(device, non_blocking=True)
            z = model(imgs).cpu().numpy()
            all_z.append(z); all_path.extend(paths)
            if bi == 1 or bi % PROGRESS_EVERY == 0 or bi == total_batches:
                print(f"[embed] {len(all_path)}/{len(ds)} images "
                      f"({bi}/{total_batches} batches, {time.time()-t0:.0f}s)", flush=True)
    embeddings = np.concatenate(all_z, axis=0)
    print(f"[embed] done {len(all_path)} images in {time.time()-t0:.0f}s", flush=True)

    # cluster
    import hdbscan
    print(f"[cluster] HDBSCAN start n={len(embeddings)} dim={embeddings.shape[1]}", flush=True)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        cluster_selection_method=CLUSTER_SELECTION_METHOD,
        cluster_selection_epsilon=CLUSTER_SELECTION_EPSILON,
        metric="euclidean",
    )
    pred = clusterer.fit_predict(embeddings)
    n_clusters = len(set(p for p in pred if p >= 0))
    n_noise = int((pred == -1).sum())
    print(f"[cluster] {n_clusters} groups + {n_noise} noise ({n_noise/len(pred)*100:.1f}%)", flush=True)

    # save
    output_subdir.mkdir(parents=True, exist_ok=True)
    # parquet (DB ingestion)
    try:
        import pandas as pd
        df = pd.DataFrame({
            "path": all_path,
            "group_id": pred.astype(int),
            "product": [product] * len(all_path),
            "line": [line] * len(all_path),
            "date": [date] * len(all_path),
            "embed_dim": [embeddings.shape[1]] * len(all_path),
        })
        df.to_parquet(output_subdir / "clusters.parquet", index=False)
        df.to_csv(output_subdir / "clusters.csv", index=False)
    except Exception as e:
        print(f"[warn] parquet save fail: {e}, fallback to JSON")
        (output_subdir / "clusters.json").write_text(
            json.dumps([{"path": p, "group_id": int(g)} for p, g in zip(all_path, pred)],
                       indent=2), encoding="utf-8")

    np.save(output_subdir / "embeddings.npy", embeddings)

    summary = {
        "folder": disp,
        "product": product, "line": line, "date": date,
        "n_images": len(all_path),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_pct": round(n_noise / len(pred) * 100, 2),
        "embed_dim": int(embeddings.shape[1]),
        "groups": {},
    }
    from collections import Counter
    cnts = Counter(pred.tolist())
    for g, c in sorted(cnts.items()):
        summary["groups"][str(int(g))] = c
    (output_subdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # optional: copy images into group/<id>/ for review
    if COPY_PNG_TO_GROUPS:
        groups_dir = output_subdir / "groups"
        for gid in set(int(g) for g in pred):
            (groups_dir / str(gid)).mkdir(parents=True, exist_ok=True)
        for p, g in zip(all_path, pred):
            src = Path(p)
            dst = groups_dir / str(int(g)) / src.name
            try:
                shutil.copy2(src, dst)
            except Exception:
                pass

    return summary


def _apply_args():
    """CLI 로 CONFIG global override (실전 폴더/모델 선택)."""
    import argparse
    global MODEL_PATH, IMAGE_ROOT, IMAGE_BASE, IMAGE_ROOTS, POOL, POOL_NAME
    global OUTPUT_DIR, COPY_PNG_TO_GROUPS, MIN_CLUSTER_SIZE, MIN_SAMPLES
    global BATCH, NUM_WORKERS, PROGRESS_EVERY
    global PRODUCT_FILTER, LINE_FILTER, DATE_FILTER
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=None, help="contrastive best_model.pt 경로")
    ap.add_argument("--image-roots", type=str, default=None,
                    help="실전 폴더 콤마구분 다중 선택. 예: E:/data/images/prodA,E:/data/images/prodB")
    ap.add_argument("--pool", action="store_true", help="여러 폴더를 합쳐 1 grouping (기본: 폴더별 따로)")
    ap.add_argument("--pool-name", type=str, default=None, help="pool 시 출력 폴더명")
    ap.add_argument("--image-root", type=str, default=None, help="단일 폴더")
    ap.add_argument("--image-base", type=str, default=None, help="product/line/date walk base")
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--copy-png", action="store_true", help="group 폴더에 PNG 복사 (시각 review)")
    ap.add_argument("--batch", type=int, default=None, help="inference batch size")
    ap.add_argument("--workers", type=int, default=None, help="DataLoader num_workers")
    ap.add_argument("--progress-every", type=int, default=None, help="embedding progress 출력 batch 간격")
    ap.add_argument("--min-cluster-size", type=int, default=None)
    ap.add_argument("--min-samples", type=int, default=None)
    a = ap.parse_args()
    if a.model:            MODEL_PATH = a.model
    if a.image_roots:      IMAGE_ROOTS = [x.strip() for x in a.image_roots.split(",") if x.strip()]
    if a.pool:             POOL = True
    if a.pool_name:        POOL_NAME = a.pool_name
    if a.image_root:       IMAGE_ROOT = a.image_root
    if a.image_base:       IMAGE_BASE = a.image_base
    if a.output_dir:       OUTPUT_DIR = a.output_dir
    if a.copy_png:         COPY_PNG_TO_GROUPS = True
    if a.batch is not None: BATCH = a.batch
    if a.workers is not None: NUM_WORKERS = a.workers
    if a.progress_every is not None: PROGRESS_EVERY = max(1, a.progress_every)
    if a.min_cluster_size is not None: MIN_CLUSTER_SIZE = a.min_cluster_size
    if a.min_samples is not None:      MIN_SAMPLES = a.min_samples


def _pipeline_summary_model(run_dir: Path) -> Path | None:
    summary = run_dir / "summary.json"
    if not summary.exists():
        return None
    try:
        info = json.loads(summary.read_text(encoding="utf-8"))
    except Exception:
        return None
    best = info.get("contrastive_best")
    if not best:
        return None
    p = resolve_path(best)
    return p if p.exists() else None


def _resolve_model_path(model_arg: str) -> Path:
    """사용자가 .pt, contrastive run dir, pipeline run dir 중 무엇을 줘도 실제 .pt 로 해석."""
    p = resolve_path(model_arg)
    if p.is_file():
        if p.name == "summary.json":
            from_summary = _pipeline_summary_model(p.parent)
            if from_summary is not None:
                return from_summary
        return p

    if p.is_dir():
        direct = p / "contrastive" / "best_model.pt"
        if direct.exists():
            return direct
        from_summary = _pipeline_summary_model(p)
        if from_summary is not None:
            return from_summary

    # 흔한 실수: runs/<pipeline>/contrastive/best_model.pt 를 줌.
    # pipeline run 안에는 모델이 없고 summary.json 이 실제 contrastive run 을 가리킨다.
    for parent in [p.parent, p.parent.parent, p.parent.parent.parent]:
        if parent and parent != parent.parent:
            from_summary = _pipeline_summary_model(parent)
            if from_summary is not None:
                return from_summary

    return p


def _model_not_found_message(model_arg: str, model_path: Path) -> str:
    repo = Path(__file__).resolve().parent.parent
    found_models = sorted(repo.glob("runs/*/contrastive/best_model.pt"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
    found_pipelines = []
    for summary in sorted(repo.glob("runs/*/summary.json"),
                          key=lambda p: p.stat().st_mtime, reverse=True):
        p = _pipeline_summary_model(summary.parent)
        if p is not None:
            found_pipelines.append((summary.parent, p))

    msg = [f"MODEL_PATH not found: {model_path}",
           f"  받은 --model 값: {model_arg}",
           "  pipeline 재학습이 필요한 뜻이 아님. 실제 contrastive best_model.pt 경로만 필요함."]
    if found_pipelines:
        msg.append("pipeline run 폴더를 바로 줄 수도 있음:")
        for run, p in found_pipelines[:5]:
            msg.append(f"  --model {run}    # → {p}")
    if found_models:
        msg.append("디스크에서 찾은 contrastive 모델 (.pt):")
        for p in found_models[:10]:
            msg.append(f"  --model {p}")
    if not found_models and not found_pipelines:
        msg.append("contrastive best_model.pt 가 없음 → stage2 만 학습:")
        msg.append("  python scripts/train_contrastive_ddp.py --backbone runs/<CNN run>/cnn/best_model.pth")
    return "\n".join(msg)


def main():
    _apply_args()
    model_path = _resolve_model_path(MODEL_PATH)
    if not model_path.exists():
        raise SystemExit(_model_not_found_message(MODEL_PATH, model_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContrastiveInferModel(BACKBONE, PROJ_DIM).to(device)
    ck = torch.load(model_path, map_location=device, weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    if isinstance(sd, dict) and any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
    load = model.load_state_dict(sd, strict=False)
    if load.missing_keys or load.unexpected_keys:
        raise SystemExit(
            f"MODEL_PATH is not a compatible contrastive best_model.pt: {model_path}\n"
            f"  missing_keys={len(load.missing_keys)} unexpected_keys={len(load.unexpected_keys)}\n"
            f"  grouping에는 CNN .pth 가 아니라 contrastive best_model.pt 를 넣어야 함")
    print(f"[model] loaded from {model_path}")

    run_dir = make_run_dir(OUTPUT_DIR, "grouping")
    print(f"[run_dir] {run_dir.resolve()}")

    cfg = {k: v for k, v in globals().items()
           if k.isupper() and not k.startswith("_")
           and isinstance(v, (str, int, float, bool, tuple, list, type(None), set))}
    cfg = {k: (list(v) if isinstance(v, set) else v) for k, v in cfg.items()}
    cfg["RESOLVED_MODEL_PATH"] = str(model_path)
    snapshot_config(run_dir, cfg)
    system_info(run_dir)

    log_stage_metric(run_dir, "grouping_setup", {
        "model": str(model_path),
        "image_base": IMAGE_BASE,
        "image_root": IMAGE_ROOT,
    })

    targets = enumerate_targets()
    print(f"[targets] {len(targets)} folder(s)")

    all_summaries = []
    for product, line, date, folder, out_name in targets:
        folders = folder if isinstance(folder, (list, tuple)) else [folder]
        if not any(Path(f).exists() for f in folders):
            print(f"[miss] {folder}")
            continue
        if product:                                    # IMAGE_BASE walk
            out = run_dir / product / line / date
        else:
            out = run_dir / (out_name or Path(folders[0]).name)
        s = cluster_folder(folder, model, device, out,
                           product=product, line=line, date=date)
        if s is not None:
            all_summaries.append(s)
            log_stage_metric(run_dir, f"grouping_{product}_{line}_{date}" if product else "grouping",
                             {"n_images": s["n_images"], "n_clusters": s["n_clusters"],
                              "n_noise": s["n_noise"], "noise_pct": s["noise_pct"]})

    # global summary
    (run_dir / "all_summaries.json").write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OUT] {run_dir.resolve()}")


if __name__ == "__main__":
    main()
