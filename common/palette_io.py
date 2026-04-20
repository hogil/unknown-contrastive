"""Palette PNG I/O helpers for wafer grade-index images.

Mirrors the loading/normalization semantics used by
``D:/project/mapviewer/api/composite_map.py`` (``create_sum_map``) but depends
only on ``numpy`` and ``PIL`` (no numba / vips / turbojpeg).

Grade convention (palette index value):

* ``0``        : normal chip
* ``1..7``     : chip defect severity (higher = worse)
* ``8..13``    : low-grade chip (minor defects, still chip area)
* ``14..30``   : invalid (should be treated as background / 0)
* ``31``       : transparent / outside-wafer background
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Default palette
# ---------------------------------------------------------------------------

def _lerp(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return (
        int(round(a[0] + (b[0] - a[0]) * t)),
        int(round(a[1] + (b[1] - a[1]) * t)),
        int(round(a[2] + (b[2] - a[2]) * t)),
    )


def get_default_palette() -> List[int]:
    """Return a flat 768-int RGB palette (256 entries) for wafer grade display.

    * 0         : green (normal chip)
    * 1..7      : green -> yellow -> orange -> red gradient (defect severity)
    * 8..13     : pale yellow-ish (low grade, still chip area)
    * 14..30    : mid grey (invalid)
    * 31        : white (transparent / background)
    * 32..255   : black padding
    """
    palette: List[Tuple[int, int, int]] = [(0, 0, 0)] * 256

    # Grade 0: strong green for normal chips
    palette[0] = (0, 180, 0)

    # Grades 1..7: green -> yellow -> orange -> red gradient
    stops = [
        (0, 180, 0),     # green
        (180, 220, 0),   # yellow-green
        (255, 210, 0),   # yellow
        (255, 140, 0),   # orange
        (220, 0, 0),     # red
    ]
    n_defect = 7  # grades 1..7
    for i in range(1, n_defect + 1):
        # map i in [1..7] -> t in [0..1]
        t = (i - 1) / float(n_defect - 1) if n_defect > 1 else 0.0
        seg_pos = t * (len(stops) - 1)
        lo = int(np.floor(seg_pos))
        hi = min(lo + 1, len(stops) - 1)
        frac = seg_pos - lo
        palette[i] = _lerp(stops[lo], stops[hi], frac)

    # Grades 8..13: pale yellow / cream (low-grade chip area)
    low_grade_color = (245, 235, 170)
    for i in range(8, 14):
        palette[i] = low_grade_color

    # Grades 14..30: neutral grey (invalid/unused)
    for i in range(14, 31):
        palette[i] = (160, 160, 160)

    # 31: white (transparent / outside wafer background)
    palette[31] = (255, 255, 255)

    # 32..255 left as (0,0,0)

    flat: List[int] = []
    for r, g, b in palette:
        flat.extend((r, g, b))
    return flat


def _hex_to_rgb(h: str) -> Tuple[int, int, int]:
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def get_failmap_palette() -> List[int]:
    """Return fail-map's official 32-entry palette (padded to 256).

    Mirrors ``D:/project/fail-map/utils.py::build_palette`` with the default
    ``color-legends.json`` values. Grade 0..7 follow fail-map's Grade0..Grade7
    colors exactly; indices 8..30 are the bottom-border palette entries used
    by fail-map (background, text, borders); index 31 is invalid_fill (white).
    """
    grade_colors = [
        "#FFFFFF",  # 0 Grade0 (pass/white)
        "#9B9B9B",  # 1 Grade1 (grey)
        "#009619",  # 2 Grade2 (green)
        "#0000FF",  # 3 Grade3 (blue)
        "#D91DFF",  # 4 Grade4 (violet)
        "#FFFF00",  # 5 Grade5 (yellow)
        "#FF0000",  # 6 Grade6 (red)
        "#000000",  # 7 Grade7 (black)
    ]
    # fail-map bottom palette entries (indices 8..30 mapped in order)
    bottom_colors = [
        "#FEFEFE",  # 8  background (bg)
        "#000001",  # 9  text
        "#BEBEBE",  # 10 border Normal
        "#FF9900",  # 11 border Invalid
        "#0099FF",  # 12 border B285
        "#FF714F",  # 13 border B286
        "#66FFCC",  # 14 border B287
        "#DA26CD",  # 15 border B288
        "#FFD700",  # 16 border B290
        "#32CD32",  # 17 border B291
        "#AAAAAA",  # 18 border B300
        "#00C8FF",  # 19 border B385
        "#FF00C8",  # 20 border B386
        "#00FF66",  # 21 border B388
        "#FF6666",  # 22 border B389
        "#6666FF",  # 23 border B390
        "#999999",  # 24 border ETC
    ]
    palette: List[Tuple[int, int, int]] = [(0, 0, 0)] * 256
    for i, h in enumerate(grade_colors):
        palette[i] = _hex_to_rgb(h)
    for j, h in enumerate(bottom_colors):
        palette[8 + j] = _hex_to_rgb(h)
    palette[31] = (255, 255, 255)  # invalid_fill (fail-map overrides to white)

    flat: List[int] = []
    for r, g, b in palette:
        flat.extend((r, g, b))
    return flat


# ---------------------------------------------------------------------------
# Single-image loading
# ---------------------------------------------------------------------------

def load_palette_indices(path: PathLike) -> np.ndarray:
    """Load a palette-PNG and return the raw grade-index array.

    * If mode is ``'P'`` the palette index is used directly.
    * Otherwise the image is converted to greyscale and divided by 32 (same
      fallback as mapviewer).
    * Fully transparent pixels (alpha == 0) are marked as ``31``.

    Returns ``np.ndarray`` of shape ``(H, W)`` with ``dtype=int16``. Values are
    in ``0..31`` but ``14+`` values are left unchanged — normalization happens
    in :func:`stack_and_normalize`.
    """
    p = Path(path)
    with Image.open(p) as img:
        is_transparent = None
        if "A" in img.getbands():
            alpha = np.array(img.getchannel("A"))
            is_transparent = alpha == 0

        if img.mode == "P":
            raw = np.array(img, dtype=np.int16)
        else:
            raw = np.array(img.convert("L"), dtype=np.int16) // 32

        if is_transparent is not None:
            raw[is_transparent] = 31

    return raw


# ---------------------------------------------------------------------------
# Stacking + normalization
# ---------------------------------------------------------------------------

def stack_and_normalize(
    paths: Sequence[PathLike],
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a batch of palette PNGs, stack them, and normalize per mapviewer rules.

    Rules (mirroring ``create_sum_map``):

    1. Resize mismatched images with ``Image.NEAREST`` to either ``target_size``
       (``(width, height)``) or the size of the first image.
    2. Fully transparent pixels are tagged as ``31`` before stacking.
    3. ``raw >= 14`` (invalid) is converted to ``0`` so remaining valid images
       still contribute.
    4. Pixels where *every* image was invalid are returned as ``invalid_mask``.
    5. Pixels whose values across all images are only in ``[8, 13]`` (no
       ``0..7`` / invalid contribution) are forced to ``8``.
    6. Output is clipped to ``[0, 13]`` as ``uint8``.

    Returns
    -------
    stacked : np.ndarray
        ``uint8`` array of shape ``(N, H, W)`` with values in ``0..13``.
    invalid_mask : np.ndarray
        Boolean array of shape ``(H, W)`` — ``True`` where *all* input images
        were invalid at that pixel.
    """
    if not paths:
        raise ValueError("paths is empty")

    raw_indices_list: List[np.ndarray] = []

    # Decide canonical (width, height)
    if target_size is not None:
        width, height = target_size
    else:
        with Image.open(Path(paths[0])) as first_img:
            width, height = first_img.size

    for img_path in paths:
        p = Path(img_path)
        with Image.open(p) as img:
            if img.size != (width, height):
                img = img.resize((width, height), Image.NEAREST)

            is_transparent = None
            if "A" in img.getbands():
                alpha = np.array(img.getchannel("A"))
                is_transparent = alpha == 0

            if img.mode == "P":
                raw = np.array(img, dtype=np.int16)
            else:
                raw = np.array(img.convert("L"), dtype=np.int16) // 32

            if is_transparent is not None:
                raw[is_transparent] = 31

        raw_indices_list.append(raw)

    stacked_raw = np.stack(raw_indices_list, axis=0)  # (N, H, W) int16

    idx_8_13_mask = (stacked_raw >= 8) & (stacked_raw <= 13)
    idx_0_7_mask = (stacked_raw >= 0) & (stacked_raw <= 7)
    idx_14_plus_mask = stacked_raw >= 14

    # Flip invalid -> 0 so valid images still contribute
    stacked_raw[idx_14_plus_mask] = 0

    has_8_13 = idx_8_13_mask.any(axis=0)
    has_0_7 = idx_0_7_mask.any(axis=0) | idx_14_plus_mask.any(axis=0)

    all_invalid = idx_14_plus_mask.all(axis=0)

    # Pixels where the only contribution (across images) is 8..13 -> force 8
    idx_8_13_only = has_8_13 & ~has_0_7
    if idx_8_13_only.any():
        stacked_raw[:, idx_8_13_only] = 8

    stacked = np.clip(stacked_raw, 0, 13).astype(np.uint8)
    invalid_mask = all_invalid.astype(bool)

    return stacked, invalid_mask


__all__ = [
    "get_default_palette",
    "load_palette_indices",
    "stack_and_normalize",
]
