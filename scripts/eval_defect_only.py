"""Defect-only eval: recompute ARI/NMI/purity/silhouette with `none` class excluded.

The full-set eval masks defect-discrimination quality because the dataset is
~80% `none` (non-defect) samples — those dominate the ARI/NMI/purity signal
even when defect-cluster assignments are arbitrary.

This script, for each run_dir:
  1. Loads trained model → embeds val + test images (L2-normalized).
  2. Runs hdbscan.approximate_predict via the pickled clusterer.
  3. Drops every sample whose GT class is `none` before computing metrics.
  4. Writes ``eval_summary_val_defect_only.json`` and
     ``eval_summary_test_defect_only.json`` into the run_dir.

Silhouette is computed on the filtered subset (cosine metric), with noise
excluded as usual.

Usage::

    python scripts/eval_defect_only.py \\
        --run-dir outputs_gpu_v4/deeper_head_260421_064156

Pass multiple ``--run-dir`` flags or use ``--run-dirs`` with a comma list.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiments.eval_metrics import (  # noqa: E402
    compute_cluster_metrics,
    compute_silhouette_cosine,
)

EXCLUDE_CLASS = "none"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _list_items(root: Path) -> tuple[list[Path], list[str]]:
    paths: list[Path] = []
    labels: list[str] = []
    for cdir in sorted(root.iterdir()):
        if not cdir.is_dir():
            continue
        for p in sorted(cdir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                paths.append(p)
                labels.append(cdir.name)
    return paths, labels


def _embed(run_dir: Path, paths: list[Path], device: str) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    import contrastive

    sib = run_dir / "checkpoints" / "last_training.pt"
    if sib.exists():
        side = torch.load(sib, map_location="cpu", weights_only=False)
        side_cfg = side.get("cfg") if isinstance(side, dict) else None
        if isinstance(side_cfg, dict):
            contrastive.CFG.update(side_cfg)
    contrastive.CFG["LOCAL_BACKBONE_WEIGHTS"] = ""
    contrastive.CFG["FREEZE_BACKBONE"] = True

    model = contrastive.CL(logger=None)
    ckpt = torch.load(run_dir / "checkpoints" / "final_infer.pt",
                      map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    sd = contrastive.strip_prefixes(sd, prefixes=("module.", "model."))
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    tfm_val = contrastive.tfm(train=False)

    class _DS(Dataset):
        def __init__(self, ps, t):
            self.ps = [str(p) for p in ps]
            self.t = t

        def __len__(self):
            return len(self.ps)

        def __getitem__(self, i):
            with Image.open(self.ps[i]) as im:
                im = im.convert("RGB")
                x = self.t(im)
            return x, i

    loader = DataLoader(_DS(paths, tfm_val), batch_size=64, num_workers=0, shuffle=False)
    D = int(contrastive.CFG.get("PROJ_DIM", 128))
    embs = np.zeros((len(paths), D), dtype=np.float32)
    with torch.no_grad():
        for x, idx in loader:
            x = x.to(device, non_blocking=True)
            f = model(x)
            f = torch.nn.functional.normalize(f, dim=1)
            embs[idx.numpy()] = f.cpu().numpy()
    return embs


def _assign(clusterer_path: Path, emb: np.ndarray) -> np.ndarray:
    import hdbscan
    with clusterer_path.open("rb") as f:
        clusterer = pickle.load(f)
    try:
        _ = clusterer.prediction_data_
    except AttributeError:
        clusterer.generate_prediction_data()
    labels, _ = hdbscan.approximate_predict(clusterer, emb)
    return labels


def _breakdown(labels: np.ndarray, gt_names: list[str]) -> dict:
    """Per-cluster: {size, dominant_class, purity, class_mix, is_defect_cluster}.
    Input already defect-only filtered."""
    out: dict[str, dict] = {}
    for cid in sorted(int(c) for c in np.unique(labels) if int(c) != -1):
        idxs = np.where(labels == cid)[0]
        classes = [gt_names[i] for i in idxs]
        counts: dict[str, int] = {}
        for c in classes:
            counts[c] = counts.get(c, 0) + 1
        sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
        dom, dom_n = sorted_counts[0]
        out[str(cid)] = {
            "size": int(idxs.size),
            "dominant_class": dom,
            "purity": float(dom_n) / float(idxs.size),
            "class_mix": dict(sorted_counts),
            "is_defect_cluster": dom != EXCLUDE_CLASS,
        }
    return out


def _run_one(run_dir: Path, split: str, split_dir: Path, device: str) -> dict:
    paths, gt_names = _list_items(split_dir)
    print(f"  [{split}] listed {len(paths)} images, {len(set(gt_names))} classes")

    embs = _embed(run_dir, paths, device)
    print(f"  [{split}] embedded {embs.shape}")

    labels = _assign(run_dir / "centroids" / "clusterer.pkl", embs)
    print(f"  [{split}] assigned; n_noise={(labels == -1).sum()}")

    keep = np.array([n != EXCLUDE_CLASS for n in gt_names])
    emb_d = embs[keep]
    lab_d = labels[keep]
    names_d = [gt_names[i] for i, k in enumerate(keep) if k]

    defect_classes = sorted(set(names_d))
    cls2idx = {c: i for i, c in enumerate(defect_classes)}
    gt = np.array([cls2idx[n] for n in names_d], dtype=np.int64)

    m = compute_cluster_metrics(lab_d, gt)
    sil = compute_silhouette_cosine(emb_d, lab_d, sample_size=None)

    return {
        "split": split,
        "n_total": int(len(paths)),
        "n_defect": int(emb_d.shape[0]),
        "n_dropped_none": int((~keep).sum()),
        "n_defect_noise": int((lab_d == -1).sum()),
        "defect_noise_ratio": float((lab_d == -1).mean()),
        "ari": m["ari"],
        "nmi": m["nmi"],
        "cluster_purity": m["cluster_purity"],
        "n_clusters_found": m["n_clusters_found"],
        "silhouette": sil["silhouette"],
        "silhouette_n_used": sil["n_used"],
        "silhouette_n_clusters": sil["n_clusters"],
        "per_cluster_defect_breakdown": _breakdown(lab_d, names_d),
        "defect_classes": defect_classes,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", action="append", default=[])
    ap.add_argument("--run-dirs", default=None, help="Comma-separated alt")
    ap.add_argument("--val-dir", default="data/wm811k_val")
    ap.add_argument("--test-dir", default="data/wm811k_test")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    run_dirs: list[Path] = [Path(p) for p in args.run_dir]
    if args.run_dirs:
        run_dirs += [Path(p.strip()) for p in args.run_dirs.split(",") if p.strip()]
    if not run_dirs:
        print("[err] no --run-dir", file=sys.stderr)
        return 1

    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    val_dir = Path(args.val_dir)
    test_dir = Path(args.test_dir)

    all_results: list[dict] = []
    for rd in run_dirs:
        if not (rd / "centroids" / "clusterer.pkl").exists():
            print(f"[skip] {rd} missing centroids/clusterer.pkl")
            continue
        print(f"\n=== {rd} ===")
        entry = {"run_dir": str(rd)}
        for split, d in [("val", val_dir), ("test", test_dir)]:
            s = _run_one(rd, split, d, device)
            entry[split] = s
            out = rd / f"eval_summary_{split}_defect_only.json"
            out.write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"  [{split}] wrote {out.name}")
        all_results.append(entry)

    combined = Path("reports/defect_only_eval_6runs.json")
    combined.parent.mkdir(parents=True, exist_ok=True)
    combined.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[ok] combined report at {combined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
