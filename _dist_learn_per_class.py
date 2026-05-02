#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 1 — Per-class chip 좌표 분포 학습 ablation.

5 method × 33 wafer class × 5 data-amount (10/30/50/100/full) sweep.

Methods:
  - heatmap            : np.histogram2d 빈도 + normalize
  - heatmap_smooth     : heatmap + gaussian_filter(σ=1.0)
  - gmm                : GaussianMixture(n_components=BIC 최적), surface evaluation
  - kde                : KernelDensity(bandwidth=Silverman's rule), surface
  - hybrid             : 0.5 × heatmap_smooth + 0.5 × gmm

절대 규칙:
  wafer class 명에 obj 들어가면 그 obj chip 만 사용.
  안 들어가면 모든 b≥200 chip 사용 (Starburst, CommaCluster, Normal 등).

출력:
  _dist_heatmaps_per_class/<class>__<method>__n=<N>[__nc=<bic_n>].npy
  plots/dist_compare/<class>__compare.png   (5 method × 5 N grid)
  plots/dist_compare/_summary_grid.png       (33 class × method=hybrid 한 페이지)
  plots/dist_compare/_bic_curves.png         (per-class GMM BIC curve)
  results/stage1_distribution.csv            (class, method, n, log_likelihood, ...)
"""
# ===================== CONFIG =====================
PNG_ROOT       = "D:/project/data/wm-811k/unknown"
JSON_ROOT      = "D:/project/data/positions/unknown"
OUT_NPY_ROOT   = "_dist_heatmaps_per_class"
OUT_PLOT_ROOT  = "plots/dist_compare"
OUT_CSV        = "results/stage1_distribution.csv"

OBJECTS        = ("bank_boundary", "invalid_main", "particle_blast",
                  "scratch", "scratch_21deg")
GRID           = 32
MIN_DEFECT_BIN = 200
DATA_AMOUNTS   = (10, 30, 50, 100, None)              # None = full
METHODS        = ("heatmap", "heatmap_smooth", "gmm", "kde", "hybrid")
GMM_COMPONENTS_RANGE = range(2, 16)                    # BIC sweep candidates
KDE_BANDWIDTH  = 1.5                                   # 2D, grid 32×32 적정
SMOOTH_SIGMA   = 1.0                                   # gaussian_filter
# ==================================================

import argparse, csv, json, os, sys, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# --------- chip 좌표 collect (절대 규칙) ----------
def collect_chip_coords(class_name: str, png_root: Path, json_root: Path) -> np.ndarray:
    """wafer class 명에 obj 들어가면 그 obj chip 만, 아니면 모든 b≥200 chip."""
    cls_png_dir = png_root / class_name
    cls_json_dir = json_root / class_name
    if not cls_json_dir.is_dir():
        return np.zeros((0, 2), dtype=np.int32)

    # 절대 규칙
    wafer_obj = None
    for obj in OBJECTS:
        if obj in class_name:
            wafer_obj = obj
            break

    coords = []
    for json_path in sorted(cls_json_dir.glob("*.json")):
        try:
            j = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for chip in (j.get("chips") or []):
            try:
                b = int(str(chip.get("b", "0")).strip())
            except Exception:
                continue
            if b < MIN_DEFECT_BIN:
                continue
            # 절대규칙: chip 에 obj 정보 있으면 (rare — positions JSON 에는 보통 없음) wafer_obj 매칭 chip 만
            # obj 정보 없으면 그냥 사용 (현재 데이터셋 default — wafer class 기반 분포 학습)
            chip_obj = chip.get("obj")
            if wafer_obj is not None and chip_obj is not None and chip_obj != wafer_obj:
                continue
            gx = int(chip.get("x_abs", -1))
            gy = int(chip.get("y_abs", -1))
            if gx < 0 or gy < 0 or gx >= GRID or gy >= GRID:
                continue
            coords.append((gx, gy))
    return np.array(coords, dtype=np.int32)


# --------- 5 method 분포 학습 ----------
def fit_heatmap(coords: np.ndarray) -> Tuple[np.ndarray, dict]:
    h = np.zeros((GRID, GRID), dtype=np.float64)
    for x, y in coords:
        h[y, x] += 1
    s = h.sum()
    if s > 0:
        h = h / s
    return h, {"method": "heatmap"}


def fit_heatmap_smooth(coords: np.ndarray, sigma: float = SMOOTH_SIGMA) -> Tuple[np.ndarray, dict]:
    from scipy.ndimage import gaussian_filter
    h, _ = fit_heatmap(coords)
    h = gaussian_filter(h, sigma=sigma)
    s = h.sum()
    if s > 0:
        h = h / s
    return h, {"method": "heatmap_smooth", "sigma": sigma}


def fit_gmm(coords: np.ndarray) -> Tuple[Optional[np.ndarray], dict]:
    from sklearn.mixture import GaussianMixture
    if len(coords) < 4:
        return None, {"method": "gmm", "skipped": "insufficient data"}
    # BIC sweep
    best_bic = float('inf')
    best_n = 2
    bic_log = {}
    for n in GMM_COMPONENTS_RANGE:
        if n > len(coords):
            break
        try:
            gmm = GaussianMixture(n_components=n, covariance_type='full',
                                  random_state=42, max_iter=200, reg_covar=1e-3)
            gmm.fit(coords.astype(np.float64))
            bic = gmm.bic(coords.astype(np.float64))
            bic_log[n] = float(bic)
            if bic < best_bic:
                best_bic = bic
                best_n = n
        except Exception:
            continue
    if not bic_log:
        return None, {"method": "gmm", "skipped": "fit failed"}
    # final fit
    gmm = GaussianMixture(n_components=best_n, covariance_type='full',
                          random_state=42, max_iter=200, reg_covar=1e-3)
    gmm.fit(coords.astype(np.float64))
    # eval on 32×32 grid
    xx, yy = np.meshgrid(np.arange(GRID), np.arange(GRID))
    grid_pts = np.c_[xx.ravel(), yy.ravel()].astype(np.float64)
    log_dens = gmm.score_samples(grid_pts).reshape(GRID, GRID)
    surface = np.exp(log_dens)
    s = surface.sum()
    if s > 0:
        surface = surface / s
    return surface, {"method": "gmm", "best_n": best_n, "bic_log": bic_log}


def fit_kde(coords: np.ndarray, bandwidth: float = KDE_BANDWIDTH) -> Tuple[Optional[np.ndarray], dict]:
    from sklearn.neighbors import KernelDensity
    if len(coords) < 2:
        return None, {"method": "kde", "skipped": "insufficient data"}
    try:
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(coords.astype(np.float64))
        xx, yy = np.meshgrid(np.arange(GRID), np.arange(GRID))
        grid_pts = np.c_[xx.ravel(), yy.ravel()].astype(np.float64)
        log_dens = kde.score_samples(grid_pts).reshape(GRID, GRID)
        surface = np.exp(log_dens)
        s = surface.sum()
        if s > 0:
            surface = surface / s
        return surface, {"method": "kde", "bandwidth": bandwidth}
    except Exception as e:
        return None, {"method": "kde", "skipped": str(e)}


def fit_hybrid(coords: np.ndarray) -> Tuple[Optional[np.ndarray], dict]:
    smooth, _ = fit_heatmap_smooth(coords)
    gmm_surface, gmm_meta = fit_gmm(coords)
    if gmm_surface is None:
        return smooth, {"method": "hybrid", "fallback": "smooth_only", "gmm_meta": gmm_meta}
    hybrid = 0.5 * smooth + 0.5 * gmm_surface
    s = hybrid.sum()
    if s > 0:
        hybrid = hybrid / s
    return hybrid, {"method": "hybrid", "gmm_meta": gmm_meta}


def compute_log_likelihood(surface: np.ndarray, coords: np.ndarray) -> float:
    """coords 좌표들의 surface 값 합 의 log = total log-likelihood (un-normalized)."""
    if surface is None or len(coords) == 0:
        return float('nan')
    eps = 1e-12
    ll = 0.0
    for x, y in coords:
        ll += np.log(max(surface[y, x], eps))
    return float(ll)


# --------- 시각화 ----------
def plot_class_compare(class_name: str, surfaces: Dict, out_path: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    methods_used = METHODS
    n_amounts = len(DATA_AMOUNTS)
    n_methods = len(methods_used)

    fig, axes = plt.subplots(n_amounts, n_methods, figsize=(3*n_methods, 3*n_amounts))
    if n_amounts == 1: axes = axes[None, :]
    if n_methods == 1: axes = axes[:, None]

    for i, n in enumerate(DATA_AMOUNTS):
        n_label = "full" if n is None else str(n)
        for j, method in enumerate(methods_used):
            ax = axes[i, j]
            key = (n, method)
            surface = surfaces.get(key)
            if surface is None:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center',
                        transform=ax.transAxes)
            else:
                im = ax.imshow(surface, cmap='hot', aspect='equal')
            ax.set_title(f'{method} n={n_label}', fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(class_name, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=80)
    plt.close()


def plot_summary_grid(all_surfaces: Dict, classes: List[str], method: str, out_path: Path):
    """class 33 × method=hybrid (full data) 한 페이지."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_cls = len(classes)
    cols = 6
    rows = (n_cls + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.5*cols, 2.5*rows))
    axes = axes.flatten()
    for i, cls in enumerate(classes):
        ax = axes[i]
        surface = all_surfaces.get(cls, {}).get((None, method))
        if surface is None:
            ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
        else:
            ax.imshow(surface, cmap='hot', aspect='equal')
        ax.set_title(cls, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(n_cls, len(axes)):
        axes[j].axis('off')
    plt.suptitle(f'All classes — {method} (full data)', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=80)
    plt.close()


def plot_bic_curves(bic_logs: Dict, classes: List[str], out_path: Path):
    """per-class GMM BIC curve."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_cls = len(classes)
    cols = 6
    rows = (n_cls + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 2.5*rows))
    axes = axes.flatten()
    for i, cls in enumerate(classes):
        ax = axes[i]
        log = bic_logs.get(cls, {})
        if log:
            ns, bics = zip(*sorted(log.items()))
            ax.plot(ns, bics, marker='o')
            best_n = ns[np.argmin(bics)]
            ax.axvline(x=best_n, color='red', linestyle='--', alpha=0.5)
            ax.set_title(f'{cls}\nbest n={best_n}', fontsize=8)
        else:
            ax.text(0.5, 0.5, "N/A", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(cls, fontsize=8)
        ax.set_xlabel('n_components', fontsize=7)
        ax.set_ylabel('BIC', fontsize=7)
    for j in range(n_cls, len(axes)):
        axes[j].axis('off')
    plt.suptitle('GMM BIC curves (lower is better)', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=80)
    plt.close()


# --------- 메인 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png-root", default=PNG_ROOT)
    ap.add_argument("--json-root", default=JSON_ROOT)
    ap.add_argument("--out-npy-root", default=OUT_NPY_ROOT)
    ap.add_argument("--out-plot-root", default=OUT_PLOT_ROOT)
    ap.add_argument("--out-csv", default=OUT_CSV)
    ap.add_argument("--classes", default=None,
                    help="comma-separated subset of class names (default: all under PNG_ROOT)")
    ap.add_argument("--methods", default=",".join(METHODS),
                    help=f"comma-separated. default: {','.join(METHODS)}")
    ap.add_argument("--n-list", default=None,
                    help="comma-separated. default: 10,30,50,100,full")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    png_root = Path(args.png_root)
    json_root = Path(args.json_root)
    out_npy_root = Path(args.out_npy_root); out_npy_root.mkdir(parents=True, exist_ok=True)
    out_plot_root = Path(args.out_plot_root); out_plot_root.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.classes:
        classes = args.classes.split(",")
    else:
        classes = sorted(p.name for p in png_root.iterdir()
                         if p.is_dir() and p.name not in {"Normal", "classification", "classification_chips"})

    methods = args.methods.split(",")
    if args.n_list:
        n_list_raw = args.n_list.split(",")
        n_list = [None if x == "full" else int(x) for x in n_list_raw]
    else:
        n_list = list(DATA_AMOUNTS)

    print(f"[*] classes: {len(classes)}", file=sys.stderr)
    print(f"[*] methods: {methods}", file=sys.stderr)
    print(f"[*] n_list: {n_list}", file=sys.stderr)

    fit_fn = {
        "heatmap": fit_heatmap,
        "heatmap_smooth": fit_heatmap_smooth,
        "gmm": fit_gmm,
        "kde": fit_kde,
        "hybrid": fit_hybrid,
    }

    rng = np.random.default_rng(args.seed)
    csv_rows: List[Dict] = []
    all_surfaces: Dict[str, Dict] = {}        # class → {(n, method): surface}
    all_bic_logs: Dict[str, Dict] = {}        # class → {n: bic}

    t_total = time.time()
    for ci, cls in enumerate(classes):
        t0 = time.time()
        coords_full = collect_chip_coords(cls, png_root, json_root)
        n_full = len(coords_full)
        if n_full == 0:
            print(f"[{ci+1}/{len(classes)}] {cls}: 0 chips, skip", file=sys.stderr)
            continue

        all_surfaces[cls] = {}
        for n_target in n_list:
            if n_target is None:
                coords_subset = coords_full
            elif n_target >= n_full:
                coords_subset = coords_full
            else:
                idxs = rng.choice(n_full, size=n_target, replace=False)
                coords_subset = coords_full[sorted(idxs.tolist())]

            for method in methods:
                if method not in fit_fn:
                    continue
                surface, meta = fit_fn[method](coords_subset)
                if surface is not None:
                    n_label = "full" if n_target is None else str(n_target)
                    bic_n = meta.get("best_n") if method in ("gmm", "hybrid") else ""
                    if method in ("gmm", "hybrid") and bic_n:
                        npy_name = f"{cls}__{method}__n={n_label}__nc={bic_n}.npy"
                    else:
                        npy_name = f"{cls}__{method}__n={n_label}.npy"
                    np.save(out_npy_root / npy_name, surface)
                    # log_likelihood on full coords (held-out evaluation)
                    ll_full = compute_log_likelihood(surface, coords_full)
                    csv_rows.append({
                        "class": cls,
                        "method": method,
                        "n_used": n_full if n_target is None else min(n_target, n_full),
                        "n_label": n_label,
                        "log_likelihood_full": f"{ll_full:.4f}",
                        "best_n_components": meta.get("best_n", ""),
                        "n_chips_full": n_full,
                    })
                    all_surfaces[cls][(n_target, method)] = surface
                    if method == "gmm" and "bic_log" in meta:
                        all_bic_logs[cls] = meta["bic_log"]
                else:
                    csv_rows.append({
                        "class": cls,
                        "method": method,
                        "n_used": 0 if n_target is None else min(n_target, n_full),
                        "n_label": "full" if n_target is None else str(n_target),
                        "log_likelihood_full": "",
                        "best_n_components": "",
                        "n_chips_full": n_full,
                    })

        # per-class compare plot
        plot_class_compare(cls, all_surfaces[cls], out_plot_root / f"{cls}__compare.png")
        elapsed = time.time() - t0
        print(f"[{ci+1}/{len(classes)}] {cls}: n={n_full} chips, {elapsed:.1f}s", flush=True)

    # CSV 작성
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class", "method", "n_used", "n_label",
                                           "log_likelihood_full", "best_n_components",
                                           "n_chips_full"])
        w.writeheader()
        w.writerows(csv_rows)
    print(f"[*] CSV: {out_csv}", file=sys.stderr)

    # summary grid
    if "hybrid" in methods:
        plot_summary_grid(all_surfaces, classes, "hybrid", out_plot_root / "_summary_grid_hybrid.png")
    if "heatmap_smooth" in methods:
        plot_summary_grid(all_surfaces, classes, "heatmap_smooth", out_plot_root / "_summary_grid_heatmap_smooth.png")

    # BIC curves
    if "gmm" in methods or "hybrid" in methods:
        plot_bic_curves(all_bic_logs, classes, out_plot_root / "_bic_curves.png")

    print(f"[*] DONE — {len(classes)} classes, {(time.time() - t_total):.1f}s total", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
