#!/usr/bin/env python3
"""Single-pass (incremental) embedding extraction for the temporal novelty-detection sims.

data/pools/temporal/unknown_novelty_sim*/**/batch_*.json 전체의 합집합(중복 없음 — 배치
설계상 이미지 재사용이 없음. 단 배경 스트림은 여러 sim 변형 간 의도적으로 공유되므로 경로
기준 dedup 함)에 대해 frozen FCMAE backbone feature f0 (1024-d, pre-L2) 를 캐시한다.

★ 260726: 여러 novel-class 변형(CrossScratch/DiagonalSmear/BrokenRing)이 배경 스트림을
공유하므로, **이미 캐시된 경로는 재추출하지 않는다** — 기존 result_grouping/temporal_novelty_260726/
f0_frozen.npy·f_champion.npy·paths_index.json 를 읽어 안 겹치는 새 이미지(주로 새 novel
클래스)만 추가 추출 후 캐시를 확장 저장한다 (팀리드 지시: GPU/CPU 자원 절약, 배경 재추출 금지).

champion(adapter) 임베딩은 f0 위에서 저비용 후처리로 파생한다 — 학습된 residual adapter
(`runs/fcmae_adapter_temperature_screen_260725/embeddings/fcmae_ad1_t010_s1_ckpt.pt`, 이미
unknown_train_defectaware_260710 pool 로 학습 완료된 champion, --freeze-backbone 이라 bb 가중치는
frozen FCMAE 와 동일함이 검증됨)의 residual만 f0 에 적용한다:
    f_champion = f0 + ad_gamma * ad(f0)          (ad = Linear(1024,128)->GELU->Linear(128,1024))
→ 두 arm(frozen/champion)이 backbone forward 를 공유해 GPU 비용을 절반으로 줄인다.

체크포인트는 read-only 로만 열람 (torch.load) — 어떤 기존 산출물도 덮어쓰지 않는다.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from _common import mask_palette_non_grade_to_white, resolve_pool  # noqa: E402

TEMPORAL_ROOT = REPO / "data/pools/temporal"
OUT_DIR = REPO / "result_grouping/temporal_novelty_260726"
CHAMPION_CKPT = REPO / "runs/fcmae_adapter_temperature_screen_260725/embeddings/fcmae_ad1_t010_s1_ckpt.pt"
TIMM_ID = "convnextv2_base.fcmae_ft_in22k_in1k_384"
IMG = 384


class ImgDS:
    """Module-level (picklable) dataset — Windows spawn multiprocessing requires this,
    a local class inside main() cannot be pickled to worker processes."""

    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        from _common import mask_palette_non_grade_to_white
        with Image.open(self.paths[i]) as im:
            img = mask_palette_non_grade_to_white(im).convert("RGB")
        return self.transform(img)


def collect_manifests() -> list[Path]:
    """모든 sim 변형(unknown_novelty_sim, unknown_novelty_sim_<novelclass>, ...) 을 합친다."""
    files: list[Path] = []
    for sim_dir in sorted(TEMPORAL_ROOT.glob("unknown_novelty_sim*")):
        if not sim_dir.is_dir():
            continue
        files += sorted(sim_dir.glob("batch_*.json"))
        for sub in sorted(sim_dir.iterdir()):
            if sub.is_dir():
                files += sorted(sub.glob("batch_*.json"))
    return files


def apply_champion_residual(f0: np.ndarray) -> tuple[np.ndarray, float]:
    ckpt = torch.load(CHAMPION_CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["model"]
    w0, b0 = sd["ad.0.weight"], sd["ad.0.bias"]
    w2, b2 = sd["ad.2.weight"], sd["ad.2.bias"]
    gamma = sd["ad_gamma"]
    with torch.no_grad():
        f0_t = torch.from_numpy(f0)
        h = F.gelu(f0_t @ w0.t() + b0)
        delta = h @ w2.t() + b2
        f_champ = (f0_t + gamma * delta).numpy().astype(np.float32)
    return f_champ, float(gamma)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifests = collect_manifests()
    print(f"[collect] {len(manifests)} batch manifests across all sim variants", flush=True)

    wanted: dict[str, str] = {}  # abs-path-str -> label
    wanted_order: list[str] = []
    for m in manifests:
        paths, labels = resolve_pool(m)
        for p, lab in zip(paths, labels):
            if p not in wanted:
                wanted[p] = lab
                wanted_order.append(p)
    print(f"[collect] {len(wanted_order)} unique images required across all sim variants", flush=True)

    # ── 기존 캐시 로드 (있으면 재사용 — 배경 재추출 금지) ──
    f0_path, fc_path, idx_path = OUT_DIR / "f0_frozen.npy", OUT_DIR / "f_champion.npy", OUT_DIR / "paths_index.json"
    cached_paths: list[str] = []
    cached_f0 = cached_fc = None
    if idx_path.exists() and f0_path.exists() and fc_path.exists():
        meta = json.loads(idx_path.read_text(encoding="utf-8"))
        cached_paths = meta["paths"]
        cached_f0 = np.load(f0_path)
        cached_fc = np.load(fc_path)
        assert cached_f0.shape[0] == cached_fc.shape[0] == len(cached_paths)
        print(f"[cache] found existing cache with {len(cached_paths)} images — will reuse, only extract new ones", flush=True)

    cached_set = set(cached_paths)
    new_paths = [p for p in wanted_order if p not in cached_set]
    print(f"[cache] {len(new_paths)} new images to extract, {len(wanted_order) - len(new_paths)} reused from cache", flush=True)

    if not new_paths:
        print("[done] nothing new to extract — cache already covers all requested images", flush=True)
        print(f"[OUT] {OUT_DIR}", flush=True)
        return

    import timm
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    # ★ 260726 팀리드 지시: device 는 코드에서 직접 지정 (이 torch 빌드는
    # CUDA_VISIBLE_DEVICES="" 를 무시함 — 다른 에이전트 실측). GPU 가 실제로
    # 비어있을 때만 사용 (다른 트랙 학습과 OOM 경합 방지 — 최소 3GB 여유 요구).
    dev = torch.device("cpu")
    if torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024 ** 3)
        if free_gb >= 3.0:
            dev = torch.device("cuda")
        else:
            print(f"[device] GPU free={free_gb:.1f}GB < 3GB headroom — falling back to CPU", flush=True)
    print(f"[device] using {dev}", flush=True)
    bb = timm.create_model(TIMM_ID, pretrained=True, num_classes=0, global_pool="avg").to(dev).eval()
    tf = T.Compose([T.Resize((IMG, IMG)), T.ToTensor(),
                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    ds = ImgDS(new_paths, tf)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=(dev.type == "cuda"))  # 260726: CPU 공유 자원 — 4 로 자제

    t0 = time.time()
    chunks = []
    n_done = 0
    with torch.no_grad():
        for x in dl:
            f = bb(x.to(dev, non_blocking=True))
            chunks.append(f.cpu().numpy().astype(np.float32))
            n_done += x.size(0)
            if n_done % (32 * 10) == 0 or n_done == len(new_paths):
                el = time.time() - t0
                eta = (len(new_paths) - n_done) * el / max(1, n_done)
                print(f"  {n_done}/{len(new_paths)} ({el:.0f}s elapsed, ETA {eta:.0f}s)", flush=True)
    f0_new = np.concatenate(chunks, axis=0)  # [N_new, 1024], pre-L2
    assert f0_new.shape[0] == len(new_paths)

    f_champ_new, gamma = apply_champion_residual(f0_new)
    print(f"[champion] ad_gamma={gamma:.5f} (checkpoint: {CHAMPION_CKPT.name}, read-only)", flush=True)

    if cached_f0 is not None:
        f0_all = np.concatenate([cached_f0, f0_new], axis=0)
        fc_all = np.concatenate([cached_fc, f_champ_new], axis=0)
        order_all = cached_paths + new_paths
        labels_all = [wanted.get(p) for p in cached_paths] + [wanted[p] for p in new_paths]
        # cached_paths 의 라벨은 새 wanted dict 에 없을 수 있음(다른 sim 변형에서 안 쓰였을 뿐,
        # 캐시 자체 meta 에서 복구)
        old_meta = json.loads(idx_path.read_text(encoding="utf-8"))
        old_label_map = dict(zip(old_meta["paths"], old_meta["labels"]))
        labels_all = [old_label_map.get(p, wanted.get(p)) for p in cached_paths] + [wanted[p] for p in new_paths]
    else:
        f0_all, fc_all, order_all = f0_new, f_champ_new, new_paths
        labels_all = [wanted[p] for p in new_paths]

    np.save(f0_path, f0_all)
    np.save(fc_path, fc_all)
    idx_path.write_text(
        json.dumps({"paths": order_all, "labels": labels_all}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[OUT] {OUT_DIR} (total cached: {len(order_all)} images, {len(new_paths)} newly added)", flush=True)


if __name__ == "__main__":
    main()
