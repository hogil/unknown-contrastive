"""DDP 공통 utility — CUDA_VISIBLE_DEVICES 자동 감지 + mp.spawn.

사용 (train_*_ddp.py 안):
    from _ddp_utils import launch_ddp, is_main, setup_ddp, cleanup_ddp, all_reduce_avg

    def train_worker(rank, world_size, cfg):
        setup_ddp(rank, world_size)
        ...
        if is_main(rank):
            print("hello from rank 0")
        ...
        cleanup_ddp()

    if __name__ == "__main__":
        launch_ddp(train_worker, cfg)

CLI:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/train_cnn_ddp.py
    → world_size=4 자동 (mp.spawn 으로 4 worker spawn)
"""
from __future__ import annotations
import os
import socket
import sys
import traceback
from typing import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def world_size_from_env() -> int:
    """CUDA_VISIBLE_DEVICES env → world_size."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if vis:
        ws = len([x for x in vis.split(",") if x.strip()])
        return max(1, ws)
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1


def find_free_port() -> int:
    """OS 가 비어있는 TCP port 자동 할당 (DDP MASTER_PORT 충돌 방지)."""
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def setup_ddp(rank: int, world_size: int, port: int | None = None,
              backend: str = "nccl") -> None:
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    if port is None:
        port = int(os.environ.get("MASTER_PORT", 29500))
    os.environ["MASTER_PORT"] = str(port)
    # Windows nccl 미지원 → gloo fallback
    if os.name == "nt" and backend == "nccl":
        backend = "gloo"
    if torch.cuda.is_available():
        n_dev = torch.cuda.device_count()
        if rank >= n_dev:
            raise RuntimeError(f"rank {rank} requested but torch sees only {n_dev} CUDA devices")
        torch.cuda.set_device(rank)
    # timeout 넉넉히 (rank0 단독 eval/save 동안 다른 rank barrier 대기 견디게)
    from datetime import timedelta
    dist.init_process_group(backend, rank=rank, world_size=world_size,
                            timeout=timedelta(minutes=60))


def cleanup_ddp() -> None:
    if not dist.is_initialized():
        return
    rank = "?"
    backend = "?"
    try:
        rank = dist.get_rank()
    except Exception:
        pass
    try:
        backend = dist.get_backend()
    except Exception:
        pass
    strict = os.environ.get("DDP_STRICT_CLEANUP", "0") == "1"
    destroy = os.environ.get("DDP_DESTROY_PROCESS_GROUP", "0") == "1"
    if backend == "nccl" and not (strict or destroy):
        # These scripts run in short-lived mp.spawn child processes. On some
        # CUDA/NCCL stacks, destroy_process_group can fail during teardown after
        # all outputs are already saved; process exit will release NCCL/CUDA.
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception as e:
        print(f"[DDP cleanup warning][rank {rank}] cuda synchronize failed during teardown: "
              f"{type(e).__name__}: {e}", flush=True)
        if strict:
            raise
    try:
        dist.destroy_process_group()
    except Exception as e:
        # If this happens after all work is saved, treating NCCL teardown as fatal only
        # prevents the parent mp.spawn from moving to the next pipeline stage.
        print(f"[DDP cleanup warning][rank {rank}] destroy_process_group failed during teardown: "
              f"{type(e).__name__}: {e}", flush=True)
        if strict:
            raise


def is_main(rank: int) -> bool:
    return rank == 0


def all_reduce_avg(metric: dict, device) -> dict:
    """모든 rank 의 scalar metric 평균 (rank 0 가 reported)."""
    if not dist.is_initialized():
        return metric
    out = {}
    ws = dist.get_world_size()
    for k, v in metric.items():
        if isinstance(v, (int, float)):
            t = torch.tensor(float(v), dtype=torch.float32, device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            out[k] = t.item() / ws
        else:
            out[k] = v
    return out


def all_gather_concat(local: torch.Tensor, device) -> torch.Tensor:
    """모든 rank 의 (variable shape[0]) tensor concat → rank 0 만 사용."""
    if not dist.is_initialized():
        return local
    ws = dist.get_world_size()
    sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(ws)]
    my_size = torch.tensor([local.size(0)], dtype=torch.long, device=device)
    dist.all_gather(sizes, my_size)
    max_sz = int(max(s.item() for s in sizes))
    # pad
    if local.size(0) < max_sz:
        pad_shape = (max_sz - local.size(0),) + tuple(local.shape[1:])
        pad = torch.zeros(*pad_shape, dtype=local.dtype, device=device)
        local_p = torch.cat([local, pad], dim=0)
    else:
        local_p = local
    bufs = [torch.zeros_like(local_p) for _ in range(ws)]
    dist.all_gather(bufs, local_p)
    parts = [b[:int(s.item())] for b, s in zip(bufs, sizes)]
    return torch.cat(parts, dim=0)


def _spawn_entry(rank: int, worker_fn: Callable, world_size: int, args: tuple) -> None:
    try:
        worker_fn(rank, world_size, *args)
    except SystemExit as e:
        print(f"[DDP rank {rank}] SystemExit: {e}", file=sys.stderr, flush=True)
        raise
    except BaseException as e:
        print(f"[DDP rank {rank}] {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise


def launch_ddp(worker_fn: Callable, *args, world_size: int | None = None) -> None:
    """spawn world_size workers. single-GPU 면 직접 호출."""
    if world_size is None:
        world_size = world_size_from_env()
    print(f"[DDP] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','(unset)')}  → world_size={world_size}")
    if world_size <= 1:
        worker_fn(0, 1, *args)
    else:
        # MASTER_PORT 자동 할당 (충돌 방지)
        os.environ["MASTER_PORT"] = str(find_free_port())
        mp.spawn(_spawn_entry, args=(worker_fn, world_size, args), nprocs=world_size, join=True)
