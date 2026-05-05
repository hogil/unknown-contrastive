#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_align_uniform.py — alignment + uniformity post-hoc 계산.

Wang & Isola 2020 (ICML) "Understanding Contrastive Representation Learning
through Alignment and Uniformity on the Hypersphere".

저장된 embedding 으로 학습 후 평가 (학습 도중 호출용 X — contrastive.py 수정 금지 정책).
새 학습 dispatch 후 이 helper 로 alignment + uniformity 추출.

Usage:
    python eval_align_uniform.py --run outputs/logs_contrastive/overall

산출:
    <run>/eval/align_uniform.json
    {
      "alignment": float,           # mean ||f(x)-f(x')||² for positive pairs
      "uniformity": float,          # log mean exp(-2 ||f(xi)-f(xj)||²)
      "n_positive_pairs_used": int, # 같은 wafer 의 두 view 못 찾으면 augment 가정 X
      "method": "intra_class_proxy"|"raw"
    }

주의:
    saved embedding 이 augmentation 두 view 없으면 (contrastive.py 가 final inference
    에 한 view 만 저장) positive pair 직접 계산 X. 대안:
    - "intra_class_proxy": 같은 GT class 끼리 평균 distance 를 alignment proxy 로
      (실제 augment positive 와 다르지만 학술 상 자주 쓰는 proxy)

표준 reference: https://arxiv.org/abs/2005.10242
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def alignment_loss(x: np.ndarray, y: np.ndarray, alpha: float = 2.0) -> float:
    """L_align = mean ||f(x) - f(x')||^alpha for positive pairs (x, x').

    Args:
        x, y: (N, D) — paired embeddings, L2-normalized
        alpha: exponent, default 2 (Wang & Isola 2020 default)
    """
    diff = x - y
    return float(np.power(np.linalg.norm(diff, axis=1), alpha).mean())


def uniformity_loss(x: np.ndarray, t: float = 2.0,
                    max_pairs: int = 1_000_000,
                    rng: np.random.Generator | None = None) -> float:
    """L_unif = log( mean exp(-t * ||f(xi) - f(xj)||^2) ) over random pairs.

    Args:
        x: (N, D) — L2-normalized embeddings
        t: temperature, default 2 (Wang & Isola 2020 default)
        max_pairs: large N 시 random sub-sampling (정확 N(N-1)/2 vs O(N) 시간)
        rng: numpy Generator
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(x)
    n_full = n * (n - 1) // 2
    if n_full <= max_pairs:
        # full pair (small N)
        sq = np.sum(x ** 2, axis=1)
        d2 = sq[:, None] + sq[None, :] - 2 * x @ x.T
        d2 = np.clip(d2, 0, None)
        iu = np.triu_indices(n, k=1)
        vals = d2[iu]
    else:
        # random sub-sample
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n, size=max_pairs)
        mask = i != j
        i, j = i[mask], j[mask]
        diff = x[i] - x[j]
        vals = np.sum(diff ** 2, axis=1)
    return float(np.log(np.mean(np.exp(-t * vals))))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--method", default="intra_class_proxy",
                   choices=["intra_class_proxy", "raw"])
    p.add_argument("--max-uniform-pairs", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    run = args.run.resolve()
    emb_path = run / "eval/embeddings/embedding.npy"
    files_path = run / "eval/embeddings/files.txt"
    classes_path = run / "eval/embeddings/classes.txt"
    if not emb_path.exists():
        print(f"missing: {emb_path}", file=sys.stderr)
        return 2

    emb = np.load(emb_path)
    # L2 normalize (이미 정규화 되어있을 수 있지만 보장)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    n = len(emb)
    print(f"[load] {emb.shape} from {emb_path}")

    # Alignment
    align = None
    if args.method == "intra_class_proxy":
        if not classes_path.exists():
            print("classes.txt missing — switching to raw", file=sys.stderr)
            args.method = "raw"

    if args.method == "intra_class_proxy":
        classes = classes_path.read_text(encoding="utf-8").splitlines()
        cls_arr = np.array(classes)
        rng = np.random.default_rng(args.seed)
        # 각 class 안에서 random pair 형성 → alignment proxy
        diffs = []
        for cls in sorted(set(classes)):
            mask = cls_arr == cls
            n_c = mask.sum()
            if n_c < 2:
                continue
            idx = np.where(mask)[0]
            rng.shuffle(idx)
            half = n_c // 2
            a = idx[:half]
            b = idx[half:2 * half]
            diff = emb[a] - emb[b]
            diffs.append(np.power(np.linalg.norm(diff, axis=1), 2.0))
        if diffs:
            align = float(np.concatenate(diffs).mean())
    else:
        # raw: 인접한 두 row 를 positive pair 로 (의미 없음, debug only)
        a = emb[::2]
        b = emb[1::2][:len(a)]
        align = alignment_loss(a, b)

    # Uniformity
    rng = np.random.default_rng(args.seed)
    unif = uniformity_loss(emb, t=2.0, max_pairs=args.max_uniform_pairs, rng=rng)

    out = {
        "run_dir": str(run),
        "ts": datetime.now().isoformat(timespec="seconds"),
        "method": args.method,
        "n_total": n,
        "alignment": align,
        "uniformity": unif,
    }
    out_path = run / "eval/align_uniform.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"alignment  : {align:.4f}   (positive pair 평균 거리², 낮을수록 좋음)")
    print(f"uniformity : {unif:.4f}   (random pair, log scale, 음수 클수록 좋음)")
    print(f"method     : {args.method}")
    print(f"n          : {n}")
    print(f"saved      : {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
