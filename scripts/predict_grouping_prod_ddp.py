#!/usr/bin/env python3
"""DDP production grouping predict.

Embedding is sharded across ranks. HDBSCAN/output writing runs on rank 0 only.
CLI options are the same as predict_grouping_prod.py.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))
import predict_grouping_prod as pg
from _ddp_utils import all_gather_concat, cleanup_ddp, is_main, launch_ddp, setup_ddp


def _load_model(model_path: Path, device):
    print(f"[rank {dist.get_rank()}] loading {model_path} on {device}", flush=True)
    model = pg.ContrastiveInferModel(pg.BACKBONE, pg.PROJ_DIM).to(device)
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
    model.eval()
    return model


def _cluster_and_save(embeddings: np.ndarray, all_path: list[str], corrupt: list[tuple[str, str]],
                      input_count: int, disp: str, output_subdir: Path,
                      product=None, line=None, date=None):
    import hdbscan

    print(f"[cluster] HDBSCAN start n={len(embeddings)} dim={embeddings.shape[1]}", flush=True)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=pg.MIN_CLUSTER_SIZE,
        min_samples=pg.MIN_SAMPLES,
        cluster_selection_method=pg.CLUSTER_SELECTION_METHOD,
        cluster_selection_epsilon=pg.CLUSTER_SELECTION_EPSILON,
        metric="euclidean",
    )
    pred = clusterer.fit_predict(embeddings)
    n_clusters = len(set(p for p in pred if p >= 0))
    n_noise = int((pred == -1).sum())
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

    output_subdir.mkdir(parents=True, exist_ok=True)
    print(f"[save] writing outputs -> {output_subdir}", flush=True)
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
        print(f"[warn] parquet save fail: {e}, fallback to JSON", flush=True)
        (output_subdir / "clusters.json").write_text(
            json.dumps([{"path": p, "group_id": int(g)} for p, g in zip(all_path, pred)],
                       indent=2), encoding="utf-8")

    np.save(output_subdir / "embeddings.npy", embeddings)
    print("[save] embeddings.npy done", flush=True)

    summary = {
        "folder": disp,
        "product": product, "line": line, "date": date,
        "n_images": len(all_path),
        "n_input_images": input_count,
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
        "ddp_world_size": dist.get_world_size() if dist.is_initialized() else 1,
    }
    for g, c in sorted(cnts.items()):
        summary["groups"][str(int(g))] = c

    n_representatives = 0
    n_reference_composites = 0
    if pg.SAVE_REPRESENTATIVES:
        print(f"[representatives] saving {pg.REPS_PER_CLUSTER}/cluster", flush=True)
        n_representatives = pg.save_grouping_representatives(
            output_subdir, embeddings, pred, all_path, pg.REPS_PER_CLUSTER)
        composite_dir = output_subdir / "representatives" / "composite"
        if composite_dir.exists():
            n_reference_composites = len(list(composite_dir.glob("*.png")))
        print(f"[representatives] saved={n_representatives} -> "
              f"{output_subdir / 'representatives'}", flush=True)
        if pg.SAVE_REFERENCE_COMPOSITES:
            print(f"[composite maps] saved={n_reference_composites} -> {composite_dir}",
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

    if pg.COPY_PNG_TO_GROUPS:
        print("[copy] PNG -> groups/ start", flush=True)
        groups_dir = output_subdir / "groups"
        for gid in set(int(g) for g in pred):
            (groups_dir / str(gid)).mkdir(parents=True, exist_ok=True)
        copied = 0
        for p, g in zip(all_path, pred):
            src = Path(p)
            dst = groups_dir / str(int(g)) / src.name
            try:
                shutil.copy2(src, dst)
                copied += 1
            except Exception:
                pass
        print(f"[copy] done copied={copied}/{len(all_path)}", flush=True)

    return summary


def _embed_target_ddp(folder_path, model, device, rank: int, world_size: int):
    tf = pg.build_eval_tf()
    ds = pg.FolderImageDataset(folder_path, transform=tf)
    disp = (", ".join(str(Path(p)) for p in folder_path)
            if isinstance(folder_path, (list, tuple)) else str(folder_path))
    if is_main(rank):
        print(f"[{disp}] {len(ds)} images", flush=True)
        print(f"[loader-ddp] world_size={world_size} batch/rank={pg.BATCH} "
              f"workers_total={pg.NUM_WORKERS}", flush=True)
    if len(ds) == 0:
        return disp, len(ds), np.zeros((0, pg.PROJ_DIM), dtype=np.float32), [], []

    local_indices = list(range(rank, len(ds), world_size))
    local_ds = Subset(ds, local_indices)
    workers_per_rank = max(0, pg.NUM_WORKERS // max(1, world_size))
    ld_kw = {"batch_size": pg.BATCH, "shuffle": False,
             "num_workers": workers_per_rank, "pin_memory": True}
    if workers_per_rank > 0:
        ld_kw["prefetch_factor"] = pg.PREFETCH_FACTOR
    if is_main(rank):
        print(f"[loader-ddp] workers/rank={workers_per_rank} "
              f"total~={workers_per_rank * world_size}", flush=True)
    loader = DataLoader(local_ds, collate_fn=pg.collate_skip_corrupt, **ld_kw)

    local_z = []
    local_paths = []
    local_bad = []
    t0 = time.time()
    total_batches = len(loader)
    processed = 0
    with torch.no_grad():
        for bi, (imgs, paths, bad) in enumerate(loader, start=1):
            if bad:
                local_bad.extend(bad)
                for path, err in bad[:3]:
                    print(f"[rank {rank} CORRUPT-SKIP] {path}: {err}", flush=True)
            processed += len(paths) + len(bad)
            if imgs is not None:
                imgs = imgs.to(device, non_blocking=True)
                z = model(imgs).detach()
                local_z.append(z)
                local_paths.extend(paths)
            if bi == 1 or bi % pg.PROGRESS_EVERY == 0 or bi == total_batches:
                print(f"[embed-ddp rank {rank}] {processed}/{len(local_ds)} "
                      f"good={len(local_paths)} skip={len(local_bad)} "
                      f"batch={bi}/{total_batches} {time.time()-t0:.0f}s", flush=True)

    if local_z:
        z_local = torch.cat(local_z, dim=0)
    else:
        z_local = torch.zeros((0, pg.PROJ_DIM), dtype=torch.float32, device=device)
    z_all = all_gather_concat(z_local, device).cpu().numpy()

    objs = [None for _ in range(world_size)]
    dist.all_gather_object(objs, {"paths": local_paths, "bad": local_bad})
    if is_main(rank):
        all_paths = []
        all_bad = []
        for obj in objs:
            all_paths.extend(obj["paths"])
            all_bad.extend(obj["bad"])
        print(f"[embed-ddp] done good={len(all_paths)} skipped_corrupt={len(all_bad)} "
              f"total={len(ds)} in {time.time()-t0:.0f}s", flush=True)
        return disp, len(ds), z_all, all_paths, all_bad
    return disp, len(ds), np.zeros((0, pg.PROJ_DIM), dtype=np.float32), [], []


def worker(rank: int, world_size: int):
    setup_ddp(rank, world_size, backend="nccl" if torch.cuda.is_available() else "gloo")
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    pg._apply_args()
    if pg.REPRESENTATIVES_ONLY:
        if is_main(rank):
            n = pg.add_representatives_to_existing_result(
                pg.resolve_path(pg.REPRESENTATIVES_ONLY), pg.REPS_PER_CLUSTER)
            print(f"[OUT] representatives saved={n}", flush=True)
        cleanup_ddp()
        return

    model_path = pg._resolve_model_path(pg.MODEL_PATH)
    if not model_path.exists():
        raise SystemExit(pg._model_not_found_message(pg.MODEL_PATH, model_path))
    if model_path.is_dir():
        raise SystemExit(
            f"MODEL_PATH resolved to a directory, not a .pt file: {model_path}\n"
            f"  use --model {model_path / 'best_model.pt'} "
            f"or --model {model_path / 'contrastive' / 'best_model.pt'}")

    model = _load_model(model_path, device)

    if is_main(rank):
        run_dir = pg.make_run_dir(pg.OUTPUT_DIR, "grouping_ddp")
        print(f"[run_dir] {run_dir.resolve()}", flush=True)
        cfg = {k: v for k, v in vars(pg).items()
               if k.isupper() and not k.startswith("_")
               and isinstance(v, (str, int, float, bool, tuple, list, type(None), set))}
        cfg = {k: (list(v) if isinstance(v, set) else v) for k, v in cfg.items()}
        cfg["RESOLVED_MODEL_PATH"] = str(model_path)
        cfg["DDP_WORLD_SIZE"] = world_size
        pg.snapshot_config(run_dir, cfg)
        pg.system_info(run_dir)
        pg.log_stage_metric(run_dir, "grouping_ddp_setup", {
            "model": str(model_path),
            "world_size": world_size,
            "image_base": pg.IMAGE_BASE,
            "image_root": pg.IMAGE_ROOT,
        })
    else:
        run_dir = None
    obj = [str(run_dir) if run_dir is not None else None]
    dist.broadcast_object_list(obj, src=0)
    run_dir = Path(obj[0])

    targets = pg.enumerate_targets()
    if is_main(rank):
        print(f"[targets] {len(targets)} folder(s)", flush=True)
        for i, (_product, _line, _date, folder, out_name) in enumerate(targets[:20], start=1):
            disp = ", ".join(str(Path(p)) for p in folder) if isinstance(folder, (list, tuple)) else str(folder)
            print(f"  [target {i}] {disp} -> {out_name or 'product/line/date'}", flush=True)
        if len(targets) > 20:
            print(f"  ... {len(targets)-20} more targets", flush=True)

    all_summaries = []
    for ti, (product, line, date, folder, out_name) in enumerate(targets, start=1):
        folders = folder if isinstance(folder, (list, tuple)) else [folder]
        exists = any(Path(f).exists() for f in folders)
        if is_main(rank):
            print(f"[target {ti}/{len(targets)}] start", flush=True)
            if not exists:
                print(f"[miss] {folder}", flush=True)
        if not exists:
            dist.barrier()
            continue

        if product:
            out = run_dir / product / line / date
        else:
            out = run_dir / (out_name or Path(folders[0]).name)

        disp, input_count, embeddings, all_path, corrupt = _embed_target_ddp(
            folder, model, device, rank, world_size)

        if is_main(rank):
            if len(all_path) == 0:
                out.mkdir(parents=True, exist_ok=True)
                if corrupt:
                    (out / "corrupt_files.txt").write_text(
                        "\n".join(f"{p}\t{e}" for p, e in corrupt), encoding="utf-8")
                print(f"[skip] no valid images after corrupt skip: {disp}", flush=True)
            else:
                s = _cluster_and_save(embeddings, all_path, corrupt, input_count, disp, out,
                                      product=product, line=line, date=date)
                all_summaries.append(s)
                pg.log_stage_metric(run_dir, f"grouping_{product}_{line}_{date}" if product else "grouping",
                                    {"n_images": s["n_images"], "n_clusters": s["n_clusters"],
                                     "n_noise": s["n_noise"], "noise_pct": s["noise_pct"],
                                     "largest_group_pct": s["largest_group_pct"]})
                print(f"[target {ti}/{len(targets)}] done clusters={s['n_clusters']} "
                      f"noise={s['n_noise']} ({s['noise_pct']}%) "
                      f"largest={s['largest_group_pct']}%", flush=True)
        dist.barrier()

    if is_main(rank):
        (run_dir / "all_summaries.json").write_text(
            json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[OUT] {run_dir.resolve()}", flush=True)
    cleanup_ddp()


def main():
    if "-h" in sys.argv or "--help" in sys.argv:
        pg._apply_args()
        return
    launch_ddp(worker)


if __name__ == "__main__":
    main()
