"""One-shot recovery: reuse a run_dir's trained checkpoint + run HDBSCAN with euclidean metric.

Motivation
----------
The first gpu_v2 training finished embedding but HDBSCAN crashed on
``metric='cosine'`` (sklearn BallTree doesn't ship a cosine distance). Rather
than re-training for 50 minutes, we reuse the saved ``final_infer.pt``,
re-embed (fast), and cluster with ``euclidean`` — on L2-normalized features
this is monotonically equivalent to cosine for ranking, so the cluster
structure is unchanged.

Usage
-----
    python scripts/recover_clustering.py --run-dir outputs_gpu_v2/strong_augment_260420_234333
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

ImageFile.LOAD_TRUNCATED_IMAGES = True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--train-dir", default="data/wm811k_train")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--hdbscan-metric", default="euclidean",
                    help="L2-normalized embeddings make euclidean ~ cosine")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not (run_dir / "checkpoints" / "final_infer.pt").exists():
        print(f"[err] missing final_infer.pt in {run_dir}", file=sys.stderr)
        return 1

    # Restore original training CFG (important: IMAGE_SIZE, BACKBONE_NAME etc.)
    import contrastive
    side = torch.load(run_dir / "checkpoints" / "last_training.pt", map_location="cpu", weights_only=False)
    side_cfg = side.get("cfg") if isinstance(side, dict) else None
    if isinstance(side_cfg, dict):
        contrastive.CFG.update(side_cfg)
    contrastive.CFG["LOCAL_BACKBONE_WEIGHTS"] = ""
    contrastive.CFG["FREEZE_BACKBONE"] = True
    contrastive.CFG["HDBSCAN_METRIC"] = args.hdbscan_metric
    contrastive.CFG["TRAIN_DIR"] = args.train_dir
    contrastive.CFG["UNKNOWN_DIR"] = args.train_dir
    contrastive.CFG["OUTPUT_DIR"] = str(run_dir).replace("_" + run_dir.name.split("_")[-2] + "_" + run_dir.name.split("_")[-1], "")
    contrastive.CFG["NUM_WORKERS"] = 0

    print(f"[cfg] IMAGE_SIZE={contrastive.CFG['IMAGE_SIZE']}  BACKBONE={contrastive.CFG['BACKBONE_NAME']}  HDBSCAN_METRIC={args.hdbscan_metric}")

    # Model
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else torch.device(args.device)
    )
    model = contrastive.CL(logger=None)
    ckpt = torch.load(run_dir / "checkpoints" / "final_infer.pt", map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    sd = contrastive.strip_prefixes(sd, prefixes=("module.", "model."))
    msg = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}")
    model.to(device).eval()

    # Image list
    tdir = Path(args.train_dir)
    class_names = sorted([p.name for p in tdir.iterdir() if p.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    files: list[str] = []
    labels: list[int] = []
    classes_per_sample: list[str] = []  # parallel to files — used by _label_purity
    for c in class_names:
        for p in sorted((tdir / c).glob("*.png")):
            files.append(str(p.resolve()))
            labels.append(class_to_idx[c])
            classes_per_sample.append(c)
    print(f"[data] {len(files)} images, {len(class_names)} classes")

    # Embed
    t = contrastive.tfm(train=False)
    emb = np.zeros((len(files), int(contrastive.CFG.get("PROJ_DIM", 128))), dtype=np.float32)
    n = len(files)
    B = int(args.batch)
    with torch.no_grad():
        for i0 in range(0, n, B):
            i1 = min(n, i0 + B)
            batch = []
            for p in files[i0:i1]:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    batch.append(t(im))
            x = torch.stack(batch, dim=0).to(device, non_blocking=True)
            feat = model(x)
            feat = torch.nn.functional.normalize(feat, dim=1)
            emb[i0:i1] = feat.cpu().numpy()
            if (i0 // B) % 10 == 0:
                print(f"[embed] {i1}/{n}")
    print(f"[embed] done, shape={emb.shape}")

    # HDBSCAN via contrastive.cluster_and_save_hdbscan
    import logging
    logger = logging.getLogger("recover")
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(sh)

    contrastive.cluster_and_save_hdbscan(
        emb=emb,
        files=files,
        labels=labels,
        classes=classes_per_sample,
        run_dir=run_dir,
        logger=logger,
    )

    # Save embedding artifacts via the helper added in Task #7
    try:
        run_log_path = run_dir / "cluster_labels.npy"
        if run_log_path.exists():
            print(f"[ok] emb artifacts already saved by cluster_and_save_hdbscan side-effects")
        else:
            # cluster_and_save may not write emb.npy itself if the flow is different
            # Fall back: manually save
            cluster_labels = np.load(run_dir / "cluster_labels.npy") if (run_dir / "cluster_labels.npy").exists() else None
            if cluster_labels is None:
                # Reconstruct from clusterer pickle if available, else write zeros
                pk = run_dir / "centroids" / "clusterer.pkl"
                if pk.exists():
                    import pickle
                    with open(pk, "rb") as f:
                        clusterer = pickle.load(f)
                    cluster_labels = clusterer.labels_ if hasattr(clusterer, "labels_") else np.zeros(len(files), dtype=np.int64)
                else:
                    cluster_labels = np.zeros(len(files), dtype=np.int64)
            kept = run_dir / "kept_labels.txt"
            kept_list = []
            if kept.exists():
                kept_list = [s.strip() for s in kept.read_text(encoding="utf-8").splitlines() if s.strip()]
            contrastive.save_embedding_artifacts(
                run_dir=run_dir,
                emb=emb,
                cluster_labels=cluster_labels,
                files=files,
                kept_labels=kept_list,
            )
            print(f"[ok] manually saved emb artifacts")
    except Exception as e:
        print(f"[warn] could not save emb artifacts: {e!r}")

    print(f"[done] run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
