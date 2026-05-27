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
# ===================================================================
# === CONFIG (실행 시 이 부분만 수정) ===
# ===================================================================
# 입력 — 옵션 A: 단일 폴더 지정 (모든 .png 하위 walk)
IMAGE_ROOT          = "data/prod/AA/K1AA/20260502"   # 프로젝트 상대 (또는 절대)

# 옵션 B: 여러 (product, line, date) 자동 walk
# IMAGE_BASE 만 주면 IMAGE_BASE/<product>/<line>/<date>/*.png 자동 enum
IMAGE_BASE          = None         # 예: "data/prod" — 비우면 IMAGE_ROOT 만 사용
PRODUCT_FILTER      = None         # 예: ["AA", "BB"] — None=all
LINE_FILTER         = None
DATE_FILTER         = None

# 모델 — contrastive best
MODEL_PATH          = "runs/<TS>_pipeline/contrastive/best_model.pt"

# 출력
OUTPUT_DIR          = "result_grouping"
COPY_PNG_TO_GROUPS  = False        # True 면 group 폴더에 PNG 복사 (디스크 증가)

# Inference
IMG_SIZE            = 384
BATCH               = 32
NUM_WORKERS         = 4

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

    def __init__(self, root: Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.paths: list[Path] = []
        for ext in self.EXTS:
            self.paths.extend(self.root.rglob(f"*{ext}"))
        self.paths = sorted(self.paths)

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
    """IMAGE_BASE 가 있으면 (product, line, date) tuple 들 walk, 아니면 IMAGE_ROOT 하나."""
    if not IMAGE_BASE:
        return [(None, None, None, resolve_path(IMAGE_ROOT))]
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
                targets.append((prod.name, line.name, date.name, date))
    return targets


def cluster_folder(folder_path: Path, model: nn.Module, device, output_subdir: Path,
                   product=None, line=None, date=None):
    """한 폴더의 이미지 cluster 후 산출."""
    tf = build_eval_tf()
    ds = FolderImageDataset(folder_path, transform=tf)
    if len(ds) == 0:
        print(f"[skip] {folder_path}: no images")
        return None
    print(f"[{folder_path}] {len(ds)} images")
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    # embed
    all_z, all_path = [], []
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        for imgs, paths in loader:
            imgs = imgs.to(device, non_blocking=True)
            z = model(imgs).cpu().numpy()
            all_z.append(z); all_path.extend(paths)
    embeddings = np.concatenate(all_z, axis=0)
    print(f"[embed] {len(all_path)} images in {time.time()-t0:.0f}s")

    # cluster
    import hdbscan
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
    print(f"[cluster] {n_clusters} groups + {n_noise} noise ({n_noise/len(pred)*100:.1f}%)")

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
        "folder": str(folder_path.resolve()),
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


def main():
    run_dir = make_run_dir(OUTPUT_DIR, "grouping")
    print(f"[run_dir] {run_dir.resolve()}")

    cfg = {k: v for k, v in globals().items()
           if k.isupper() and not k.startswith("_")
           and isinstance(v, (str, int, float, bool, tuple, list, type(None), set))}
    cfg = {k: (list(v) if isinstance(v, set) else v) for k, v in cfg.items()}
    snapshot_config(run_dir, cfg)
    system_info(run_dir)

    model_path = resolve_path(MODEL_PATH)
    if not model_path.exists():
        raise SystemExit(f"MODEL_PATH not found: {model_path}\n"
                         f"먼저 학습: python scripts/train_pipeline.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContrastiveInferModel(BACKBONE, PROJ_DIM).to(device)
    ck = torch.load(model_path, map_location=device, weights_only=False)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    missing = model.load_state_dict(sd, strict=False)
    print(f"[model] loaded from {MODEL_PATH}, missing={len(missing.missing_keys)}, "
          f"unexpected={len(missing.unexpected_keys)}")

    log_stage_metric(run_dir, "grouping_setup", {
        "model": MODEL_PATH,
        "image_base": IMAGE_BASE,
        "image_root": IMAGE_ROOT,
    })

    targets = enumerate_targets()
    print(f"[targets] {len(targets)} folder(s)")

    all_summaries = []
    for product, line, date, folder in targets:
        if not folder.exists():
            print(f"[miss] {folder}")
            continue
        if IMAGE_BASE:
            out = run_dir / product / line / date
        else:
            out = run_dir / folder.name
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
