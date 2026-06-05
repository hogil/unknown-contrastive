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
SAVE_REPRESENTATIVES = True        # cluster별 중심에 가까운 대표 이미지만 저장
REPS_PER_CLUSTER    = 30
SAVE_REFERENCE_COMPOSITES = True   # representatives/composite/ 에 cluster별 linear2_weighted_average 저장
REFERENCE_COMPOSITE_MAX_PX = 1536  # composite map 최대 한 변 크기. 0 이면 원본 크기 유지
REFERENCE_COMPOSITE_MIN_SUPPORT = 2
REFERENCE_COMPOSITE_MIN_SUPPORT_RATIO = 0.10
REPRESENTATIVES_ONLY = None        # 기존 grouping 결과 폴더에 representatives 만 후처리

# Inference
IMG_SIZE            = 512
BATCH               = 128         # H100 inference 기본. OOM 나면 --batch 64/32 로 낮춤.
NUM_WORKERS         = 16          # 6400 PNG decode 병목 완화. RAM 부족하면 --workers 8.
PREFETCH_FACTOR     = 2
PROGRESS_EVERY      = 20          # embedding loop 진행률 출력 batch 간격

# HDBSCAN
MIN_CLUSTER_SIZE    = 12
MIN_SAMPLES         = 15
CLUSTER_SELECTION_METHOD = "leaf"
CLUSTER_SELECTION_EPSILON = 0.06

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
    mask_palette_non_grade_to_white,
    resolve_path,
    snapshot_config,
    system_info,
)


class FolderImageDataset(Dataset):
    """폴더 하위 모든 .png (recursive). class 없음 — grouping 전용."""
    EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    def __init__(self, root, transform=None):
        roots = root if isinstance(root, (list, tuple)) else [root]
        self.roots = [Path(r) for r in roots]
        self.root = self.roots[0]
        self.transform = transform
        paths: list[Path] = []
        for r in self.roots:
            paths.extend(p for p in Path(r).rglob("*")
                         if p.is_file() and p.suffix.lower() in self.EXTS)
        self.paths = sorted(set(paths))

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        try:
            with Image.open(p) as im:
                img = mask_palette_non_grade_to_white(im).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
        except Exception as e:
            return None, str(p), f"{type(e).__name__}: {e}"
        return img, str(p), None


def collate_skip_corrupt(batch):
    good = [x for x in batch if x[0] is not None]
    bad = [(path, err) for _img, path, err in batch if err is not None]
    if not good:
        return None, [], bad
    imgs = torch.stack([x[0] for x in good], dim=0)
    paths = [x[1] for x in good]
    return imgs, paths, bad


def build_eval_tf():
    norm = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        norm,
    ])


def make_pbar(*args, **kwargs):
    """tqdm 있으면 진행바, 없으면 None."""
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm(*args, **kwargs)


def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-+" else "_" for ch in str(s))


def _reference_target_size(size: tuple[int, int]) -> tuple[int, int]:
    max_px = int(REFERENCE_COMPOSITE_MAX_PX or 0)
    if max_px <= 0:
        return size
    w, h = size
    scale = min(1.0, max_px / max(1, max(w, h)))
    return max(1, int(round(w * scale))), max(1, int(round(h * scale)))


def _reference_palette(source_palette: list[int] | None) -> list[int]:
    palette = list(source_palette or [])
    if len(palette) < 768:
        palette.extend([0] * (768 - len(palette)))

    # Composite map에는 chip/background/border 색이 필요 없다. defect gradient만 남긴다.
    for idx in range(8, 24):
        palette[idx * 3:idx * 3 + 3] = [255, 255, 255]

    # mapviewer style composite gradient, slightly darker than pure white->red
    # so low/mid weighted signals remain visible without over-tinting.
    start, end, count = 24, 255, 255 - 24 + 1
    for i in range(count):
        ratio = i / max(count - 1, 1)
        ratio = ratio ** 0.72
        gb = int(round(248 * (1.0 - ratio)))
        idx = (start + i) * 3
        palette[idx:idx + 3] = [255, gb, gb]
    return palette[:768]


def _reference_positions_path(src: Path) -> Path | None:
    stem = src.stem + ".json"
    candidates = [src.with_suffix(".json")]
    parts = list(src.parts)
    lowered = [p.lower() for p in parts]
    if "images" in lowered:
        i = lowered.index("images")
        candidates.append(Path(*parts[:i]) / "positions" / Path(*parts[i + 1:]).with_suffix(".json"))
    candidates.append(Path("data") / "positions" / stem)
    for p in candidates:
        if p.exists():
            return p
    return None


def _build_reference_base_indices(src: Path, target_size: tuple[int, int],
                                  chip_area: np.ndarray) -> np.ndarray:
    width, height = target_size
    base = np.full((height, width), 8, dtype=np.uint8)
    pos_path = _reference_positions_path(src)
    if pos_path is None:
        base[chip_area] = 0
        return base
    try:
        data = json.loads(pos_path.read_text(encoding="utf-8"))
    except Exception:
        base[chip_area] = 0
        return base

    chips = data.get("chips")
    if not isinstance(chips, list) or not chips:
        base[chip_area] = 0
        return base

    coord = data.get("coord", {})
    canvas = coord.get("canvas", {}) if isinstance(coord, dict) else {}
    canvas_w = int(canvas.get("width", width)) if isinstance(canvas, dict) else width
    canvas_h = int(canvas.get("height", height)) if isinstance(canvas, dict) else height
    scale_x = width / float(canvas_w or width)
    scale_y = height / float(canvas_h or height)

    for chip in chips:
        if not isinstance(chip, dict):
            continue
        rect = chip.get("rect", {})
        if not isinstance(rect, dict):
            continue
        try:
            x0 = float(rect["x0"]); y0 = float(rect["y0"])
            x1 = float(rect["x1"]); y1 = float(rect["y1"])
        except Exception:
            continue
        sx0 = max(0, min(width, int(x0 * scale_x)))
        sy0 = max(0, min(height, int(y0 * scale_y)))
        sx1 = max(0, min(width, int(x1 * scale_x + 0.9999)))
        sy1 = max(0, min(height, int(y1 * scale_y + 0.9999)))
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        base[sy0:sy1, sx0:sx1] = 0
    return base


def _load_palette_index_array(src: Path, target_size: tuple[int, int]) -> np.ndarray | None:
    """palette wafer PNG를 index array로 로드."""
    try:
        with Image.open(src) as img:
            if img.size != target_size:
                img = img.resize(target_size, Image.NEAREST)
            if img.mode == "P":
                return np.asarray(img, dtype=np.uint8)
            gray = np.asarray(img.convert("L"), dtype=np.float32)
            return np.clip(np.rint(gray / 255.0 * 7.0), 0, 7).astype(np.uint8)
    except Exception:
        return None


def _render_reference_sum_map(base_indices: np.ndarray, value_map: np.ndarray,
                              mask: np.ndarray) -> np.ndarray:
    """mapviewer _render_sum_map_palette와 같은 palette-index 렌더."""
    result = base_indices.copy()
    if not np.any(mask):
        return result
    vals = value_map[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return result
    v_min, v_max = 0.0, float(vals.max())
    if v_max <= v_min:
        grad_idx = np.full(value_map[mask].shape, 255 if v_max > 0 else 24, dtype=np.uint8)
    else:
        scaled = (value_map[mask].astype(np.float32) - v_min) / (v_max - v_min)
        grad_idx = np.clip(np.rint(scaled * (255 - 24) + 24), 24, 255).astype(np.uint8)
    result[mask] = grad_idx
    return result


def save_reference_composites(reference_dir: Path, label: str, cluster_size: int,
                              image_paths: list[Path]) -> list[dict]:
    """대표 이미지들로 mapviewer식 linear2_weighted_average 저장."""
    valid = [Path(p) for p in image_paths if Path(p).exists()]
    if not valid:
        return []
    reference_dir.mkdir(parents=True, exist_ok=True)

    first_src = valid[0]
    try:
        with Image.open(first_src) as first:
            target_size = _reference_target_size(first.size)
            palette = _reference_palette(first.getpalette())
    except Exception:
        return []

    width, height = target_size
    grade_counts = np.zeros((8, height, width), dtype=np.uint16)
    has_0_7 = np.zeros((height, width), dtype=bool)
    has_8_13 = np.zeros((height, width), dtype=bool)
    all_invalid = np.ones((height, width), dtype=bool)
    defect_support = np.zeros((height, width), dtype=np.uint16)
    used = 0
    for src in valid:
        arr = _load_palette_index_array(src, target_size)
        if arr is None:
            continue
        if arr.shape != (height, width):
            continue
        ge14 = arr >= 14
        ge8 = arr >= 8
        mid = ge8 & ~ge14
        has_0_7 |= (~ge8) | ge14
        has_8_13 |= mid
        all_invalid &= ge14
        defect_support += ((arr >= 1) & (arr <= 7))
        grade_counts[0] += ge14
        for grade in range(8):
            grade_counts[grade] += (arr == grade)
        used += 1
    if used == 0:
        return []

    idx_8_13_only = has_8_13 & ~has_0_7
    invalid_mask = all_invalid
    chip_area = (has_0_7 | idx_8_13_only) & ~invalid_mask
    base_indices = _build_reference_base_indices(first_src, target_size, chip_area)
    chip_inner_mask = base_indices == 0

    grade_counts_float = grade_counts.astype(np.float32, copy=False)
    # 기존 square(g^2)는 고 grade가 과하게 강해질 수 있어, grade 2 이상은 g*2로 완화.
    # g: 0 1 2 3 4 5 6 7  →  w: 0 1 4 6 8 10 12 14
    linear2_weights = np.asarray([0, 1, 4, 6, 8, 10, 12, 14],
                                 dtype=np.float32).reshape(8, 1, 1)
    linear2_sums = np.sum(grade_counts_float * linear2_weights, axis=0, dtype=np.float32)
    weight_factors = np.asarray([1, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32).reshape(8, 1, 1)
    weight_sum = np.sum(grade_counts_float * weight_factors, axis=0, dtype=np.float32)
    min_support = max(
        REFERENCE_COMPOSITE_MIN_SUPPORT,
        int(np.ceil(float(used) * REFERENCE_COMPOSITE_MIN_SUPPORT_RATIO)),
    )
    weighted_mask = (
        chip_inner_mask & ~idx_8_13_only & ~invalid_mask &
        (weight_sum > 0) & (defect_support >= min_support)
    )
    linear2_weighted_average = np.zeros_like(linear2_sums, dtype=np.float32)
    linear2_weighted_average[weighted_mask] = linear2_sums[weighted_mask] / weight_sum[weighted_mask]

    rows = []
    safe = _safe_name(label)
    variants = [
        ("linear2_weighted_average", "weighted_linear2_mean", linear2_weighted_average, weighted_mask),
    ]
    for suffix, variant_type, value_map, mask in variants:
        out = reference_dir / f"{safe}_n-{cluster_size}_reps-{used}_{suffix}.png"
        idx_arr = _render_reference_sum_map(base_indices, value_map, mask)
        idx_arr = idx_arr.astype(np.uint8, copy=False)
        img = Image.frombytes("P", (idx_arr.shape[1], idx_arr.shape[0]), idx_arr.tobytes())
        img.putpalette(palette)
        img.save(out)
        rows.append({
            "label": label,
            "type": variant_type,
            "filename": out.name,
            "cluster_size": cluster_size,
            "representatives_used": used,
            "min_support": min_support,
            "support_rule": "defect_support>=max(2,ceil(reps*0.10))",
            "path": str(out),
            "width": target_size[0],
            "height": target_size[1],
        })
    return rows


def save_grouping_representatives(output_subdir: Path, embeddings: np.ndarray,
                                  pred: np.ndarray, paths: list[str],
                                  per_cluster: int) -> int:
    """cluster별 centroid에 가까운 이미지 몇 장만 복사. -1(HDBSCAN noise)은 제외."""
    rep_dir = output_subdir / "representatives"
    if rep_dir.exists():
        shutil.rmtree(rep_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    composite_dir = rep_dir / "composite"
    composite_rows = []
    if SAVE_REFERENCE_COMPOSITES:
        composite_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    saved = 0
    pred = np.asarray(pred, dtype=int)
    cluster_ids = sorted(int(c) for c in set(pred.tolist()) if int(c) != -1)
    for cl_id in cluster_ids:
        idx = np.where(pred == cl_id)[0]
        if len(idx) == 0:
            continue
        center = embeddings[idx].mean(axis=0)
        dist = np.linalg.norm(embeddings[idx] - center[None, :], axis=1)
        order = idx[np.argsort(dist)[:per_cluster]]
        sub = rep_dir / f"cluster_{cl_id:03d}_n-{len(idx)}"
        sub.mkdir(parents=True, exist_ok=True)
        selected_srcs = []
        for rank_i, i in enumerate(order, 1):
            src = Path(paths[i])
            if not src.exists():
                continue
            selected_srcs.append(src)
            dst = sub / f"rep-{rank_i:02d}_{src.name}"
            try:
                shutil.copy2(src, dst)
                saved += 1
                rows.append({
                    "cluster_id": cl_id,
                    "rank": rank_i,
                    "cluster_size": len(idx),
                    "distance_to_centroid": f"{float(np.linalg.norm(embeddings[i] - center)):.6f}",
                    "src": str(src),
                    "dst": str(dst),
                })
            except Exception:
                pass
        if SAVE_REFERENCE_COMPOSITES:
            refs = save_reference_composites(
                composite_dir, f"cluster_{cl_id:03d}", len(idx), selected_srcs)
            for ref in refs:
                ref["cluster_id"] = cl_id
                composite_rows.append(ref)

    import csv
    with (rep_dir / "representatives.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "cluster_id", "rank", "cluster_size",
            "distance_to_centroid", "src", "dst",
        ])
        w.writeheader()
        w.writerows(rows)
    if SAVE_REFERENCE_COMPOSITES:
        with (composite_dir / "composite_maps.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "cluster_id", "label", "type", "filename", "cluster_size",
                "representatives_used", "min_support", "support_rule",
                "width", "height", "path",
            ])
            w.writeheader()
            w.writerows(composite_rows)
    return saved


def add_representatives_to_existing_result(result_dir: Path, per_cluster: int) -> int:
    """기존 result_grouping run 에 representatives/ 만 추가."""
    if not result_dir.exists():
        raise SystemExit(f"result dir not found: {result_dir}")
    targets = []
    for emb in result_dir.rglob("embeddings.npy"):
        out_dir = emb.parent
        csv_path = out_dir / "clusters.csv"
        if csv_path.exists():
            targets.append((out_dir, emb, csv_path))
    if not targets:
        raise SystemExit(f"embeddings.npy + clusters.csv pair not found under: {result_dir}")

    total = 0
    print(f"[representatives-only] {len(targets)} target(s) under {result_dir}", flush=True)
    for out_dir, emb_path, csv_path in targets:
        import pandas as pd
        print(f"[representatives-only] {out_dir}", flush=True)
        embeddings = np.load(emb_path)
        df = pd.read_csv(csv_path)
        if "path" not in df.columns or "group_id" not in df.columns:
            raise SystemExit(f"clusters.csv missing path/group_id columns: {csv_path}")
        n = min(len(df), len(embeddings))
        saved = save_grouping_representatives(
            out_dir,
            embeddings[:n],
            df["group_id"].to_numpy(dtype=int)[:n],
            df["path"].astype(str).tolist()[:n],
            per_cluster,
        )
        print(f"  saved={saved} → {out_dir / 'representatives'}", flush=True)
        total += saved
    return total


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
    ld_kw = {"batch_size": BATCH, "shuffle": False,
             "num_workers": NUM_WORKERS, "pin_memory": True}
    if NUM_WORKERS > 0:
        ld_kw["prefetch_factor"] = PREFETCH_FACTOR
    print(f"[loader] batch={BATCH} workers={NUM_WORKERS} "
          f"prefetch={PREFETCH_FACTOR if NUM_WORKERS > 0 else 'n/a'}", flush=True)
    loader = DataLoader(ds, collate_fn=collate_skip_corrupt, **ld_kw)

    # embed
    all_z, all_path = [], []
    model.eval()
    t0 = time.time()
    total_batches = len(loader)
    print(f"[embed] start batch={BATCH} workers={NUM_WORKERS} batches={total_batches}", flush=True)
    pbar = make_pbar(total=len(ds), desc="embed", unit="img", dynamic_ncols=True,
                     mininterval=1.0)
    corrupt = []
    processed = 0
    with torch.no_grad():
        for bi, (imgs, paths, bad) in enumerate(loader, start=1):
            if bad:
                corrupt.extend(bad)
                for path, err in bad[:3]:
                    print(f"[CORRUPT-SKIP] {path}: {err}", flush=True)
                if len(bad) > 3:
                    print(f"[CORRUPT-SKIP] batch {bi}: +{len(bad)-3} more", flush=True)
            processed += len(paths) + len(bad)
            if imgs is None:
                if pbar is not None:
                    pbar.update(len(bad))
                    pbar.set_postfix(batch=f"{bi}/{total_batches}",
                                     good=len(all_path), skip=len(corrupt))
                continue
            imgs = imgs.to(device, non_blocking=True)
            z = model(imgs).cpu().numpy()
            all_z.append(z); all_path.extend(paths)
            if pbar is not None:
                pbar.update(len(paths) + len(bad))
                pbar.set_postfix(batch=f"{bi}/{total_batches}",
                                 good=len(all_path), skip=len(corrupt))
            elif bi == 1 or bi % PROGRESS_EVERY == 0 or bi == total_batches:
                print(f"[embed] processed={processed}/{len(ds)} good={len(all_path)} "
                      f"skip={len(corrupt)} ({bi}/{total_batches} batches, {time.time()-t0:.0f}s)",
                      flush=True)
    if pbar is not None:
        pbar.close()
    if not all_z:
        output_subdir.mkdir(parents=True, exist_ok=True)
        if corrupt:
            (output_subdir / "corrupt_files.txt").write_text(
                "\n".join(f"{p}\t{e}" for p, e in corrupt), encoding="utf-8")
        print(f"[skip] no valid images after corrupt skip: {disp}", flush=True)
        return None
    embeddings = np.concatenate(all_z, axis=0)
    print(f"[embed] done good={len(all_path)} skipped_corrupt={len(corrupt)} "
          f"total={len(ds)} in {time.time()-t0:.0f}s", flush=True)

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
    from collections import Counter
    cnts = Counter(pred.tolist())
    non_noise_sizes = [c for g, c in cnts.items() if int(g) != -1]
    largest_group = max(non_noise_sizes) if non_noise_sizes else 0
    largest_group_pct = largest_group / max(1, len(pred)) * 100.0
    top_groups = sorted(
        ((int(g), int(c), int(c) / max(1, len(pred)) * 100.0) for g, c in cnts.items()),
        key=lambda x: x[1], reverse=True,
    )[:5]
    top_groups_text = ", ".join(f"{g}:{c}({pct:.1f}%)" for g, c, pct in top_groups)
    print(f"[cluster] {n_clusters} groups + {n_noise} noise ({n_noise/len(pred)*100:.1f}%) "
          f"| largest={largest_group_pct:.1f}% | top={top_groups_text}", flush=True)

    # save
    output_subdir.mkdir(parents=True, exist_ok=True)
    # parquet (DB ingestion)
    print(f"[save] writing outputs → {output_subdir}", flush=True)
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
        print("[save] clusters.parquet + clusters.csv done", flush=True)
    except Exception as e:
        print(f"[warn] parquet save fail: {e}, fallback to JSON")
        (output_subdir / "clusters.json").write_text(
            json.dumps([{"path": p, "group_id": int(g)} for p, g in zip(all_path, pred)],
                       indent=2), encoding="utf-8")

    np.save(output_subdir / "embeddings.npy", embeddings)
    print("[save] embeddings.npy done", flush=True)

    summary = {
        "folder": disp,
        "product": product, "line": line, "date": date,
        "n_images": len(all_path),
        "n_input_images": len(ds),
        "n_corrupt_skipped": len(corrupt),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_pct": round(n_noise / len(pred) * 100, 2),
        "largest_group_size": int(largest_group),
        "largest_group_pct": round(largest_group_pct, 2),
        "top_groups": [
            {"group_id": int(g), "size": int(c), "pct": round(float(pct), 2)}
            for g, c, pct in top_groups
        ],
        "embed_dim": int(embeddings.shape[1]),
        "groups": {},
    }
    for g, c in sorted(cnts.items()):
        summary["groups"][str(int(g))] = c

    n_representatives = 0
    n_reference_composites = 0
    if SAVE_REPRESENTATIVES:
        print(f"[representatives] saving {REPS_PER_CLUSTER}/cluster", flush=True)
        n_representatives = save_grouping_representatives(
            output_subdir, embeddings, pred, all_path, REPS_PER_CLUSTER)
        composite_dir = output_subdir / "representatives" / "composite"
        if composite_dir.exists():
            n_reference_composites = len(list(composite_dir.glob("*.png")))
        print(f"[representatives] saved={n_representatives} → "
              f"{output_subdir / 'representatives'}", flush=True)
        if SAVE_REFERENCE_COMPOSITES:
            print(f"[composite maps] saved={n_reference_composites} → {composite_dir}",
                  flush=True)
    summary["n_representatives_saved"] = n_representatives
    summary["n_reference_composites_saved"] = n_reference_composites
    summary["n_composite_maps_saved"] = n_reference_composites
    (output_subdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[save] summary.json done", flush=True)
    if corrupt:
        (output_subdir / "corrupt_files.txt").write_text(
            "\n".join(f"{p}\t{e}" for p, e in corrupt), encoding="utf-8")
        print(f"[save] corrupt_files.txt done ({len(corrupt)} skipped)", flush=True)

    # optional: copy images into group/<id>/ for review
    if COPY_PNG_TO_GROUPS:
        print("[copy] PNG → groups/ start", flush=True)
        groups_dir = output_subdir / "groups"
        for gid in set(int(g) for g in pred):
            (groups_dir / str(gid)).mkdir(parents=True, exist_ok=True)
        pairs = list(zip(all_path, pred))
        pbar = make_pbar(pairs, desc="copy", unit="img", dynamic_ncols=True,
                         mininterval=1.0)
        it = pbar if pbar is not None else pairs
        copied = 0
        for p, g in it:
            src = Path(p)
            dst = groups_dir / str(int(g)) / src.name
            try:
                shutil.copy2(src, dst)
                copied += 1
            except Exception:
                pass
        if pbar is not None:
            pbar.close()
        print(f"[copy] done copied={copied}/{len(all_path)}", flush=True)

    return summary


def _apply_args():
    """CLI 로 CONFIG global override (실전 폴더/모델 선택)."""
    import argparse
    global MODEL_PATH, IMAGE_ROOT, IMAGE_BASE, IMAGE_ROOTS, POOL, POOL_NAME
    global OUTPUT_DIR, COPY_PNG_TO_GROUPS, SAVE_REPRESENTATIVES, REPS_PER_CLUSTER
    global SAVE_REFERENCE_COMPOSITES, REFERENCE_COMPOSITE_MAX_PX
    global REPRESENTATIVES_ONLY, MIN_CLUSTER_SIZE, MIN_SAMPLES
    global CLUSTER_SELECTION_METHOD, CLUSTER_SELECTION_EPSILON
    global IMG_SIZE, BATCH, NUM_WORKERS, PREFETCH_FACTOR, PROGRESS_EVERY
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
    ap.add_argument("--no-representatives", action="store_true",
                    help="cluster 대표 이미지 저장 끄기")
    ap.add_argument("--reps-per-cluster", type=int, default=None,
                    help="cluster별 대표 이미지 저장 개수")
    ap.add_argument("--no-reference-composites", action="store_true",
                    help="representatives/composite cluster composite map 저장 끄기")
    ap.add_argument("--reference-composite-size", type=int, default=None,
                    help="composite map 최대 한 변 크기. 0이면 원본 크기")
    ap.add_argument("--representatives-only", type=str, default=None,
                    help="기존 result_grouping run 폴더에 representatives/ 만 후처리 생성")
    ap.add_argument("--batch", type=int, default=None, help="inference batch size")
    ap.add_argument("--img-size", type=int, default=None, help="inference input size. 기본 512")
    ap.add_argument("--workers", type=int, default=None, help="DataLoader num_workers")
    ap.add_argument("--prefetch-factor", type=int, default=None, help="DataLoader prefetch_factor (workers>0)")
    ap.add_argument("--progress-every", type=int, default=None, help="embedding progress 출력 batch 간격")
    ap.add_argument("--min-cluster-size", type=int, default=None)
    ap.add_argument("--min-samples", type=int, default=None)
    ap.add_argument("--cluster-selection-method", type=str, default=None,
                    choices=["eom", "leaf"])
    ap.add_argument("--cluster-selection-epsilon", type=float, default=None)
    a = ap.parse_args()
    if a.model:            MODEL_PATH = a.model
    if a.image_roots:      IMAGE_ROOTS = [x.strip() for x in a.image_roots.split(",") if x.strip()]
    if a.pool:             POOL = True
    if a.pool_name:        POOL_NAME = a.pool_name
    if a.image_root:       IMAGE_ROOT = a.image_root
    if a.image_base:       IMAGE_BASE = a.image_base
    if a.output_dir:       OUTPUT_DIR = a.output_dir
    if a.copy_png:         COPY_PNG_TO_GROUPS = True
    if a.no_representatives: SAVE_REPRESENTATIVES = False
    if a.reps_per_cluster is not None: REPS_PER_CLUSTER = max(1, a.reps_per_cluster)
    if a.no_reference_composites: SAVE_REFERENCE_COMPOSITES = False
    if a.reference_composite_size is not None:
        REFERENCE_COMPOSITE_MAX_PX = max(0, a.reference_composite_size)
    if a.representatives_only: REPRESENTATIVES_ONLY = a.representatives_only
    if a.batch is not None: BATCH = a.batch
    if a.img_size is not None: IMG_SIZE = a.img_size
    if a.workers is not None: NUM_WORKERS = a.workers
    if a.prefetch_factor is not None: PREFETCH_FACTOR = max(1, a.prefetch_factor)
    if a.progress_every is not None: PROGRESS_EVERY = max(1, a.progress_every)
    if a.min_cluster_size is not None: MIN_CLUSTER_SIZE = a.min_cluster_size
    if a.min_samples is not None:      MIN_SAMPLES = a.min_samples
    if a.cluster_selection_method is not None:
        CLUSTER_SELECTION_METHOD = a.cluster_selection_method
    if a.cluster_selection_epsilon is not None:
        CLUSTER_SELECTION_EPSILON = a.cluster_selection_epsilon


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
        direct_self = p / "best_model.pt"
        if direct_self.exists():
            return direct_self
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
    if REPRESENTATIVES_ONLY:
        n = add_representatives_to_existing_result(
            resolve_path(REPRESENTATIVES_ONLY), REPS_PER_CLUSTER)
        print(f"[OUT] representatives saved={n}")
        return

    model_path = _resolve_model_path(MODEL_PATH)
    if not model_path.exists():
        raise SystemExit(_model_not_found_message(MODEL_PATH, model_path))
    if model_path.is_dir():
        raise SystemExit(
            f"MODEL_PATH resolved to a directory, not a .pt file: {model_path}\n"
            f"  use --model {model_path / 'best_model.pt'} "
            f"or --model {model_path / 'contrastive' / 'best_model.pt'}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[model] loading {model_path} on {device}", flush=True)
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
    print(f"[targets] {len(targets)} folder(s)", flush=True)
    for i, (_product, _line, _date, folder, out_name) in enumerate(targets[:20], start=1):
        disp = ", ".join(str(Path(p)) for p in folder) if isinstance(folder, (list, tuple)) else str(folder)
        print(f"  [target {i}] {disp} -> {out_name or 'product/line/date'}", flush=True)
    if len(targets) > 20:
        print(f"  ... {len(targets)-20} more targets", flush=True)

    all_summaries = []
    for ti, (product, line, date, folder, out_name) in enumerate(targets, start=1):
        folders = folder if isinstance(folder, (list, tuple)) else [folder]
        print(f"[target {ti}/{len(targets)}] start", flush=True)
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
                              "n_noise": s["n_noise"], "noise_pct": s["noise_pct"],
                              "largest_group_pct": s["largest_group_pct"]})
            print(f"[target {ti}/{len(targets)}] done clusters={s['n_clusters']} "
                  f"noise={s['n_noise']} ({s['noise_pct']}%) "
                  f"largest={s['largest_group_pct']}%", flush=True)

    # global summary
    (run_dir / "all_summaries.json").write_text(
        json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OUT] {run_dir.resolve()}")


if __name__ == "__main__":
    main()
