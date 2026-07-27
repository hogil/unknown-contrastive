#!/usr/bin/env python3
"""Embedding extraction for the leakage-fixed v2 temporal novelty-detection sims.

★ 260726 (leakage-fix v2): champion_v1(`fcmae_ad1_t010_s1_ep4`)의 학습 pool
(`unknown_train_defectaware_260710`)이 SHA-256 감사로 unknown_eval100/holdout/anchor 와
겹침이 확정 → v2 leakage-free split(`data/pools/v2/unknown/strict_novel_train.json`)으로
champion 을 재학습(perf-anchor 담당, 이 스크립트는 그 산출을 소비만 함)해 시간축 시뮬을
재실행한다.

★ 팀리드 지시(260726): 이번 실험은 "누수" 단 하나만 격리해야 한다 — backbone 을
`_b4_backbone.pth`(TAPT) 로 바꾸면 누수+backbone 두 변수가 동시에 바뀐다. champion_v1 이
plain FCMAE(`weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth`) 위의 residual adapter였으므로,
이 v2 실험도 **동일 plain FCMAE backbone** 을 쓴다 (`_grouping_eval.py::load_backbone` 과
동일한 raw-GAP 방식 — global_pool="" + forward_features().mean(dim=(2,3))).

두 단계로 나뉜다 (proj 체크포인트가 아직 없어도 1단계는 먼저 실행 가능 — GPU 학습과 병행):
  1. raw GAP feature f0 캐시 (--stage raw)  — backbone forward만, proj 없음 = "frozen" arm 그 자체.
  2. champion 임베딩 파생 (--stage champion --proj-ckpt <path/to/proj_epNN.pt>) — f0 위에 저비용
     proj 적용, 재추출 없음.

출력: result_grouping/temporal_novelty_v2_260726/{f0_frozen.npy, f_champion.npy, paths_index.json}
(원본 result_grouping/temporal_novelty_260726/ 과 별도 디렉토리 — 기존 v1 산출 불변).
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from _common import resolve_pool  # noqa: E402

TEMPORAL_ROOT = REPO / "data/pools/temporal"
OUT_DIR = REPO / "result_grouping/temporal_novelty_v2_260726"
BACKBONE_PATH = REPO / "weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth"  # plain FCMAE — 팀리드 지시, 1-축 유지
IMG = 384
SIM_DIRS = ["unknown_novelty_sim_v2_diagonalsmear", "unknown_novelty_sim_v2_ringdots"]


def collect_manifests() -> list[Path]:
    files: list[Path] = []
    for name in SIM_DIRS:
        sim_dir = TEMPORAL_ROOT / name
        if not sim_dir.is_dir():
            continue
        files += sorted(sim_dir.glob("batch_*.json"))
        for sub in sorted(sim_dir.iterdir()):
            if sub.is_dir():
                files += sorted(sub.glob("batch_*.json"))
    return files


class ImgDS:
    """Module-level (picklable) dataset — Windows spawn multiprocessing requires this."""

    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        return self.transform(img)


def build_proj() -> nn.Module:
    # _grouping_eval.py::build_proj 와 동일 구조 — champion_v2 proj 체크포인트 로드 호환.
    return nn.Sequential(nn.Linear(1024, 1024, bias=False), nn.BatchNorm1d(1024),
                          nn.ReLU(inplace=True), nn.Linear(1024, 128))


def pick_device(min_free_gb: float = 3.0) -> torch.device:
    if torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024 ** 3)
        if free_gb >= min_free_gb:
            return torch.device("cuda")
        print(f"[device] GPU free={free_gb:.1f}GB < {min_free_gb}GB headroom — falling back to CPU", flush=True)
    return torch.device("cpu")


def stage_raw() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifests = collect_manifests()
    print(f"[collect] {len(manifests)} batch manifests across v2 sim variants", flush=True)

    wanted: dict[str, str] = {}
    order: list[str] = []
    for m in manifests:
        paths, labels = resolve_pool(m)
        for p, lab in zip(paths, labels):
            if p not in wanted:
                wanted[p] = lab
                order.append(p)
    print(f"[collect] {len(order)} unique images required", flush=True)

    idx_path, f0_path = OUT_DIR / "paths_index.json", OUT_DIR / "f0_frozen.npy"
    cached_paths: list[str] = []
    cached_f0 = None
    if idx_path.exists() and f0_path.exists():
        meta = json.loads(idx_path.read_text(encoding="utf-8"))
        cached_paths = meta["paths"]
        cached_f0 = np.load(f0_path)
        assert cached_f0.shape[0] == len(cached_paths)
        print(f"[cache] found existing raw cache with {len(cached_paths)} images — reusing", flush=True)

    cached_set = set(cached_paths)
    new_paths = [p for p in order if p not in cached_set]
    print(f"[cache] {len(new_paths)} new images to extract, {len(order) - len(new_paths)} reused", flush=True)

    if not new_paths:
        print("[done] nothing new — raw cache already complete", flush=True)
        return

    import timm
    import torchvision.transforms as T
    from PIL import Image
    from torch.utils.data import DataLoader

    dev = pick_device()
    print(f"[device] using {dev}", flush=True)

    bb = timm.create_model("convnextv2_base.fcmae_ft_in22k_in1k_384", pretrained=False,
                            num_classes=0, global_pool="")
    sd = torch.load(BACKBONE_PATH, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    for prefix in ("model.", "backbone.", "module."):
        if any(k.startswith(prefix) for k in sd):
            sd = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}
    bb.load_state_dict(sd, strict=False)
    bb = bb.eval().to(dev)

    tf = T.Compose([T.Resize((IMG, IMG)), T.ToTensor(),
                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    dl = DataLoader(ImgDS(new_paths, tf), batch_size=32, shuffle=False, num_workers=4,
                     pin_memory=(dev.type == "cuda"))

    t0 = time.time()
    chunks = []
    n_done = 0
    with torch.no_grad():
        for x in dl:
            fm = bb.forward_features(x.to(dev, non_blocking=True))
            pool = fm.mean(dim=(2, 3))
            chunks.append(pool.cpu().numpy().astype(np.float32))
            n_done += x.size(0)
            if n_done % (32 * 10) == 0 or n_done == len(new_paths):
                el = time.time() - t0
                eta = (len(new_paths) - n_done) * el / max(1, n_done)
                print(f"  {n_done}/{len(new_paths)} ({el:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)
    f0_new = np.concatenate(chunks, axis=0)
    assert f0_new.shape[0] == len(new_paths)

    if cached_f0 is not None:
        f0_all = np.concatenate([cached_f0, f0_new], axis=0)
        order_all = cached_paths + new_paths
        old_meta = json.loads(idx_path.read_text(encoding="utf-8"))
        old_label_map = dict(zip(old_meta["paths"], old_meta["labels"]))
        labels_all = [old_label_map.get(p, wanted.get(p)) for p in cached_paths] + [wanted[p] for p in new_paths]
    else:
        f0_all, order_all = f0_new, new_paths
        labels_all = [wanted[p] for p in new_paths]

    np.save(f0_path, f0_all)
    idx_path.write_text(json.dumps({"paths": order_all, "labels": labels_all}, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"[OUT] {f0_path} (total cached: {len(order_all)} images, {len(new_paths)} newly added)", flush=True)


def stage_champion(proj_ckpt: str) -> None:
    f0 = np.load(OUT_DIR / "f0_frozen.npy")
    ckpt = torch.load(proj_ckpt, map_location="cpu")
    pj = ckpt["proj"] if "proj" in ckpt else ckpt
    pj = {k[len("net."):] if k.startswith("net.") else k: v for k, v in pj.items()}
    proj = build_proj()
    proj.load_state_dict(pj)
    proj.eval()
    with torch.no_grad():
        f0_t = torch.from_numpy(f0).float()
        z = proj(f0_t)
        z = F.normalize(z, dim=1).numpy().astype(np.float32)
    np.save(OUT_DIR / "f_champion.npy", z)
    print(f"[champion] proj={proj_ckpt} -> {OUT_DIR / 'f_champion.npy'} shape={z.shape}", flush=True)
    print(f"[OUT] {OUT_DIR}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["raw", "champion"], required=True)
    ap.add_argument("--proj-ckpt", default=None, help="proj_epNN.pt path (required for --stage champion)")
    a = ap.parse_args()
    if a.stage == "raw":
        stage_raw()
    else:
        if not a.proj_ckpt:
            raise SystemExit("--proj-ckpt required for --stage champion")
        stage_champion(a.proj_ckpt)


if __name__ == "__main__":
    main()
