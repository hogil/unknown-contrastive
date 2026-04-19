"""LSWMD.pkl → palette PNG 변환.

Output: ``data/wm811k/<class>/<idx>.png`` (9 classes, 각 N_PER_CLASS장)

WM-811K value mapping:

    0 (outside wafer)  -> palette index 31 (transparent via tRNS chunk)
    1 (normal die)     -> palette index 0  (grade 0 = normal chip)
    2 (defect die)     -> palette index 7  (grade 7 = severe defect)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from PIL import Image

# Make the project root importable when running this file directly.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.palette_io import get_default_palette  # noqa: E402


CLASSES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none",
]


def wm_to_palette(wm: np.ndarray) -> np.ndarray:
    """WM-811K 2D array → palette index 2D array (uint8)."""
    # Some pkl rows store the waferMap as list-of-list → coerce via np.asarray.
    wm = np.asarray(wm)
    out = np.full(wm.shape, 31, dtype=np.uint8)  # default: outside-wafer
    out[wm == 1] = 0
    out[wm == 2] = 7
    return out


def save_palette_png(idx_array: np.ndarray, path: Path, palette: List[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if idx_array.dtype != np.uint8:
        idx_array = idx_array.astype(np.uint8)
    img = Image.fromarray(idx_array, mode="P")
    img.putpalette(palette)
    img.save(path, transparency=31, optimize=True)


def _norm_failure_type(x) -> str:
    """Normalize a WM-811K ``failureType`` entry to a plain string.

    Raw cells are frequently 2D numpy arrays like ``array([['none']], dtype=object)``.
    """
    if isinstance(x, (list, np.ndarray)):
        arr = np.asarray(x).ravel()
        if arr.size == 0:
            return ""
        return str(arr[0])
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data_raw/LSWMD.pkl")
    ap.add_argument("--out", default="data/wm811k")
    ap.add_argument("--n-per-class", type=int, default=222)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pkl_path = Path(args.pkl)
    assert pkl_path.exists(), (
        f"PKL not found: {pkl_path} — run data_prep/download_wm811k.py first"
    )

    df = pd.read_pickle(pkl_path)

    # WM-811K has a well-known typo: the column is sometimes ``failureType`` and
    # sometimes ``faliureType``. Accept either.
    if "failureType" in df.columns:
        ft_col = "failureType"
    elif "faliureType" in df.columns:
        ft_col = "faliureType"
    else:
        raise KeyError(
            f"Could not find failureType column in {pkl_path}; got {list(df.columns)}"
        )

    df = df.copy()
    df["_ft"] = df[ft_col].apply(_norm_failure_type)

    # Drop unlabeled / unknown rows.
    df = df[df["_ft"].isin(CLASSES)]

    rng = np.random.default_rng(args.seed)
    out_root = Path(args.out)
    palette = get_default_palette()

    summary = {}
    for cls in CLASSES:
        sub = df[df["_ft"] == cls]
        n_have = len(sub)
        if n_have == 0:
            print(f"[WARN] no samples for class: {cls}")
            summary[cls] = 0
            continue

        take = min(args.n_per_class, n_have)
        picks = rng.choice(n_have, size=take, replace=False)
        subset = sub.iloc[picks]

        saved = 0
        for i, (_, row) in enumerate(subset.iterrows()):
            try:
                idx_arr = wm_to_palette(row["waferMap"])
                save_palette_png(
                    idx_arr,
                    out_root / cls / f"wafer_{i:05d}.png",
                    palette,
                )
                saved += 1
            except Exception as e:
                print(f"[skip] {cls}/{i}: {e}")

        summary[cls] = saved
        print(f"[{cls}] {saved}/{take} saved (total available: {n_have})")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    total = sum(summary.values())
    n_nonzero = sum(1 for v in summary.values() if v > 0)
    print(f"\n[Done] total saved: {total} images across {n_nonzero} classes")
    print(f"Output: {out_root.resolve()}")


if __name__ == "__main__":
    main()
