"""Composite map computation (pure numpy + PIL port of mapviewer logic).

Given a stack of palette-index wafer maps ``(N, H, W)`` with values in
``0..13`` (produced by :func:`common.palette_io.stack_and_normalize`), the
main entry point :func:`render_composite_png` produces a PNG where the base
layer is a palette-colored median of the stack, overlaid in defect regions
with a blue -> cyan -> yellow -> red heatmap of the per-pixel "square mean"
(``Σ_g count_g * g^2 / N``).

This is a lightweight re-implementation that intentionally drops the
``numba`` / ``vips`` / ``turbojpeg`` acceleration path and the personal
color-scheme plumbing found in ``D:/project/mapviewer/api/composite_map.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image, ImageDraw

from .palette_io import get_default_palette

PathLike = Union[str, Path]

NUM_GRADES = 14  # 0..13


# ---------------------------------------------------------------------------
# Core numeric helpers
# ---------------------------------------------------------------------------

def compute_grade_counts(stacked: np.ndarray) -> np.ndarray:
    """Count the number of images exhibiting each grade at each pixel.

    Parameters
    ----------
    stacked : np.ndarray
        ``uint8`` array of shape ``(N, H, W)`` with values in ``0..13``.

    Returns
    -------
    np.ndarray
        ``uint32`` array of shape ``(14, H, W)`` where
        ``out[g, y, x]`` is the number of input images where pixel
        ``(y, x)`` had grade ``g``.
    """
    if stacked.ndim != 3:
        raise ValueError(f"stacked must be 3D (N,H,W); got shape {stacked.shape}")

    _, h, w = stacked.shape
    grade_counts = np.zeros((NUM_GRADES, h, w), dtype=np.uint32)
    for g in range(NUM_GRADES):
        grade_counts[g] = (stacked == g).sum(axis=0).astype(np.uint32)
    return grade_counts


def compute_square_mean(
    grade_counts: np.ndarray,
    image_count: int,
    invalid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute ``Σ_g (count_g * g^2) / image_count`` per pixel.

    The invalid-mask region is forced to ``0``.
    """
    if grade_counts.shape[0] != NUM_GRADES:
        raise ValueError(
            f"grade_counts first dim must be {NUM_GRADES}; got {grade_counts.shape[0]}"
        )
    if image_count <= 0:
        raise ValueError("image_count must be > 0")

    grades_squared = (np.arange(NUM_GRADES, dtype=np.float32) ** 2).reshape(
        NUM_GRADES, 1, 1
    )
    weighted = grade_counts.astype(np.float32) * grades_squared
    square_mean = weighted.sum(axis=0) / float(image_count)

    if invalid_mask is not None:
        square_mean[invalid_mask.astype(bool)] = 0.0

    return square_mean.astype(np.float32)


def compute_base_indices(
    stacked: np.ndarray,
    invalid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a base-layer palette-index map of shape ``(H, W)``.

    Follows mapviewer's ``_FAST_MEDIAN=True`` shortcut: use the mean of the
    stack as an approximation of the median (cheap, fine for visualization).
    ``invalid_mask == True`` pixels are forced to grade ``8`` (background-ish).
    """
    if stacked.ndim != 3:
        raise ValueError(f"stacked must be 3D (N,H,W); got shape {stacked.shape}")

    mean = stacked.astype(np.float32).mean(axis=0)
    base = np.clip(np.round(mean), 0, NUM_GRADES - 1).astype(np.uint8)

    if invalid_mask is not None:
        base[invalid_mask.astype(bool)] = 8

    return base


# ---------------------------------------------------------------------------
# Colormap LUT
# ---------------------------------------------------------------------------

_GRADIENT_STOPS = np.array(
    [
        [0, 0, 255],     # blue
        [0, 255, 255],   # cyan
        [255, 255, 0],   # yellow
        [255, 0, 0],     # red
    ],
    dtype=np.float32,
)


def _build_gradient_lut(n: int = 256) -> np.ndarray:
    """Return an ``(n, 3) uint8`` LUT linearly interpolating the stops."""
    stops = _GRADIENT_STOPS
    n_stops = stops.shape[0]
    xs = np.linspace(0.0, n_stops - 1, n, dtype=np.float32)
    lo = np.floor(xs).astype(np.int32)
    hi = np.clip(lo + 1, 0, n_stops - 1)
    frac = (xs - lo).reshape(-1, 1)
    lut = stops[lo] * (1.0 - frac) + stops[hi] * frac
    return np.clip(lut, 0, 255).astype(np.uint8)


def _palette_to_rgb_lut(palette: List[int]) -> np.ndarray:
    """Convert a flat 768-int palette into a ``(256, 3) uint8`` LUT."""
    if len(palette) < 256 * 3:
        # pad with black
        palette = list(palette) + [0] * (256 * 3 - len(palette))
    arr = np.asarray(palette[: 256 * 3], dtype=np.uint8).reshape(256, 3)
    return arr


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_composite_png(
    stacked: np.ndarray,
    output_path: PathLike,
    *,
    palette: Optional[List[int]] = None,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """Render a composite PNG from a normalized stack of grade indices.

    Parameters
    ----------
    stacked : np.ndarray
        ``uint8`` array of shape ``(N, H, W)`` with values in ``0..13``
        (the output of :func:`common.palette_io.stack_and_normalize`).
    output_path : str or Path
        Destination PNG.
    palette : list[int], optional
        Flat 768-int RGB palette. Defaults to :func:`get_default_palette`.
    title : str, optional
        If given, drawn in the top-left corner of the image.

    Returns
    -------
    dict
        ``{'path', 'width', 'height', 'image_count', 'square_mean_stats': {...}}``
    """
    if stacked.ndim != 3:
        raise ValueError(f"stacked must be 3D (N,H,W); got shape {stacked.shape}")
    if stacked.dtype != np.uint8:
        stacked = stacked.astype(np.uint8)

    n, h, w = stacked.shape
    if palette is None:
        palette = get_default_palette()

    # 1) invalid_mask reconstruction: mapviewer treats "all zeros" as "nothing
    #    to show", but stacked may already have values forced by
    #    stack_and_normalize. We approximate the invalid region as pixels
    #    whose entire stack is zero AND that were never touched by any defect.
    #    That's a fine fallback: everything == 0 means background.
    all_zero = (stacked == 0).all(axis=0)

    # 2) grade counts & value map
    grade_counts = compute_grade_counts(stacked)
    square_mean = compute_square_mean(grade_counts, image_count=n, invalid_mask=None)

    # 3) base layer
    base_indices = compute_base_indices(stacked, invalid_mask=all_zero)

    # 4) RGB canvas from palette
    palette_lut = _palette_to_rgb_lut(palette)
    rgb = palette_lut[base_indices].copy()  # (H, W, 3) uint8

    # 5) overlay mask: pixels with any defect contribution
    defect_mask = ~all_zero & (square_mean > 0)
    square_mean_stats = {
        "min": float(square_mean.min()),
        "max": float(square_mean.max()),
        "p95": float(np.percentile(square_mean, 95)) if square_mean.size else 0.0,
    }

    if defect_mask.any():
        vals = square_mean[defect_mask]
        vmin = float(vals.min())
        vmax = float(vals.max())
        if vmax > vmin:
            norm = (vals - vmin) / (vmax - vmin)
        else:
            norm = np.zeros_like(vals, dtype=np.float32)

        lut = _build_gradient_lut(256)
        idx = np.clip(np.round(norm * 255), 0, 255).astype(np.int32)
        colors = lut[idx]  # (K, 3) uint8
        rgb[defect_mask] = colors

    # 6) PIL image + optional title
    img = Image.fromarray(rgb, mode="RGB")
    if title:
        draw = ImageDraw.Draw(img)
        # built-in default font — available everywhere, small but always works
        draw.text((4, 4), str(title), fill=(0, 0, 0))

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")

    return {
        "path": str(out_path),
        "width": int(w),
        "height": int(h),
        "image_count": int(n),
        "square_mean_stats": square_mean_stats,
    }


__all__ = [
    "compute_grade_counts",
    "compute_square_mean",
    "compute_base_indices",
    "render_composite_png",
]
