"""Stage 2: WM-811K → 4000×4000 palette PNG (train/val disjoint).

``configs/pixel_rules.yaml`` (Stage 1 산출) 을 입력으로 받아 WM-811K 원본
wafer 에 grade 배정 규칙을 적용하고 fail-map 포맷 palette PNG 를 생성한다.

Grade 매핑 (pixel-design skill 기반)
-----------------------------------
* ``0`` (outside) → palette **31** (transparent)
* ``1`` (normal)  → palette **0** (Grade 0)
* ``2`` (defect)  → palette **random choice in {1..7}** with YAML weights
  (exponential decay ``1/2^(k-1)`` 가 기본).

Upscaling
---------
원본 해상도에서 grade 배정 → ``Image.resize((W, H), Image.NEAREST)``. Palette
index 가 discrete 이므로 NEAREST 만 허용 (BILINEAR/BICUBIC 은 무효 index 생성).

Train / Val 분할
----------------
Class 별로 ``rng.permutation(n_have)`` 후 앞 ``N_train`` → train, 다음 ``N_val``
→ val. 두 슬라이스는 disjoint. 복원추출 금지.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from common.palette_io import get_failmap_palette  # noqa: E402


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
DEFECT_CLASSES = [c for c in CLASSES if c != "none"]


# ---------------------------------------------------------------------------
# Grade assignment
# ---------------------------------------------------------------------------

def _normalize_probs(
    grades: list[int] | tuple[int, ...],
    weights: list[float] | tuple[float, ...] | None,
) -> tuple[np.ndarray, np.ndarray]:
    g_arr = np.asarray(list(grades), dtype=np.uint8)
    if weights is None:
        w_arr = np.array([1.0 / (2 ** k) for k in range(len(g_arr))], dtype=np.float64)
    else:
        w_arr = np.asarray(list(weights), dtype=np.float64)
    if w_arr.size != g_arr.size:
        raise ValueError(
            f"weights length {w_arr.size} != grades length {g_arr.size}"
        )
    if (w_arr < 0).any():
        raise ValueError("weights must be non-negative")
    total = float(w_arr.sum())
    if total <= 0:
        raise ValueError("weights sum must be > 0")
    return g_arr, w_arr / total


def assign_grades_textured(
    wm: np.ndarray,
    rng: np.random.Generator,
    target_size: tuple[int, int],
    grades: list[int] | tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
    weights: list[float] | tuple[float, ...] | None = None,
    defect_perturb_p: float = 0.30,
    normal_scatter_p: float = 0.03,
    normal_scatter_grades: tuple[int, ...] = (1,),
    global_scatter_n: int = 0,
    global_scatter_radius: tuple[int, int] = (2, 6),
) -> np.ndarray:
    """Upsample WM-811K map to ``target_size`` with synthesized sub-die texture.

    Steps
    -----
    1. NEAREST upscale ``wm (h0, w0)`` to ``target_size (H, W)``.
    2. Outside (value 0) -> palette 31 (transparent).
    3. Normal die (value 1) -> base 0, with ``normal_scatter_p`` fraction of
       pixels randomly flipped to a grade drawn from ``normal_scatter_grades``
       (micro-defect background).
    4. Defect die (value 2) -> per-die base grade drawn from ``grades`` with
       weighted probability, then ``defect_perturb_p`` fraction of pixels
       shifted by +-1 / +-2 grade (triangular weight) to create intra-die
       texture.
    5. Optional: ``global_scatter_n`` scatter spots across the wafer
       (contamination particles) of grade 1-3 on otherwise-normal pixels.

    Result: 4000x4000 palette array where each defect die is visibly richer
    than uniform blocks while preserving its base grade identity.
    """
    wm = np.asarray(wm)
    if wm.ndim != 2:
        raise ValueError(f"waferMap must be 2D, got shape {wm.shape}")
    if target_size[0] <= 0 or target_size[1] <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")

    H, W = int(target_size[0]), int(target_size[1])
    h0, w0 = wm.shape

    grades_arr, probs = _normalize_probs(grades, weights)

    # Step 1: NEAREST upscale
    im = Image.fromarray(wm.astype(np.uint8), mode="L").resize((W, H), Image.NEAREST)
    wm_up = np.array(im, dtype=np.int32)

    outside_mask = wm_up == 0
    normal_mask = wm_up == 1
    defect_mask = wm_up == 2

    out = np.full((H, W), 31, dtype=np.uint8)
    out[normal_mask] = 0

    # Step 4: per-die base grade for defect dies
    if defect_mask.any():
        ys = (np.arange(H) * h0 // H).astype(np.int64)
        xs = (np.arange(W) * w0 // W).astype(np.int64)
        die_id = ys[:, None] * w0 + xs[None, :]

        defect_die_ids = np.unique(die_id[defect_mask])
        die_to_grade = np.zeros(h0 * w0, dtype=np.uint8)
        chosen = rng.choice(grades_arr, size=defect_die_ids.size, p=probs)
        die_to_grade[defect_die_ids] = chosen.astype(np.uint8)
        base_grade = die_to_grade[die_id]
        out[defect_mask] = base_grade[defect_mask]

        # Intra-die perturbation
        if defect_perturb_p > 0:
            noise = rng.random((H, W))
            pert_mask = defect_mask & (noise < defect_perturb_p)
            n_pert = int(pert_mask.sum())
            if n_pert:
                deltas = rng.choice(
                    np.array([-2, -1, 1, 2], dtype=np.int16),
                    size=n_pert,
                    p=[0.15, 0.35, 0.35, 0.15],
                )
                current = out[pert_mask].astype(np.int16)
                out[pert_mask] = np.clip(current + deltas, 1, 7).astype(np.uint8)

    # Step 3: normal die micro-scatter
    if normal_scatter_p > 0 and normal_mask.any():
        noise = rng.random((H, W))
        scatter_mask = normal_mask & (noise < normal_scatter_p)
        n_scat = int(scatter_mask.sum())
        if n_scat:
            ns_arr = np.asarray(normal_scatter_grades, dtype=np.uint8)
            picks = rng.choice(ns_arr, size=n_scat)
            out[scatter_mask] = picks

    # Step 5: optional global scatter contamination
    if global_scatter_n > 0:
        inside_mask = normal_mask | defect_mask
        if inside_mask.any():
            coords = np.argwhere(inside_mask)
            n_pick = min(int(global_scatter_n), len(coords))
            sel = rng.choice(len(coords), size=n_pick, replace=False)
            rmin, rmax = int(global_scatter_radius[0]), int(global_scatter_radius[1])
            rs = rng.integers(rmin, rmax + 1, size=n_pick)
            gs = rng.choice(
                np.array([1, 2, 3], dtype=np.uint8),
                size=n_pick,
                p=[0.6, 0.3, 0.1],
            )
            for (y, x), r, g in zip(coords[sel], rs, gs):
                y0, y1 = max(0, int(y) - int(r)), min(H, int(y) + int(r) + 1)
                x0, x1 = max(0, int(x) - int(r)), min(W, int(x) + int(r) + 1)
                sub = out[y0:y1, x0:x1]
                local_inside = inside_mask[y0:y1, x0:x1]
                local_zero = sub == 0
                sub[local_inside & local_zero] = int(g)

    return out


def assign_grades_random_skewed(
    wm: np.ndarray,
    rng: np.random.Generator,
    grades: list[int] | tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7),
    weights: list[float] | tuple[float, ...] | None = None,
) -> np.ndarray:
    """WM-811K 2D map (0/1/2) → palette index 2D map (uint8).

    Rules
    -----
    * ``wm == 0`` → 31 (transparent / outside wafer)
    * ``wm == 1`` → 0  (Grade 0, normal die)
    * ``wm == 2`` → ``rng.choice(grades, p=weights/sum(weights))`` per-die.

    Parameters
    ----------
    wm : np.ndarray
        2D array of WM-811K die values (0 / 1 / 2).
    rng : np.random.Generator
        Seeded RNG for reproducibility.
    grades : sequence of int
        Candidate palette indices for defect dies (default 1..7).
    weights : sequence of float, optional
        Per-grade weights (any non-negative scale). Normalized to sum=1.
        Default = exponential ``1 / 2**(k-1)``.
    """
    wm = np.asarray(wm)
    if wm.ndim != 2:
        raise ValueError(f"waferMap must be 2D, got shape {wm.shape}")

    grades_arr = np.asarray(list(grades), dtype=np.uint8)
    if weights is None:
        weights_arr = np.array(
            [1.0 / (2 ** (k)) for k in range(len(grades_arr))], dtype=np.float64
        )
    else:
        weights_arr = np.asarray(list(weights), dtype=np.float64)

    if weights_arr.size != grades_arr.size:
        raise ValueError(
            f"weights length {weights_arr.size} != grades length {grades_arr.size}"
        )
    if (weights_arr < 0).any():
        raise ValueError("weights must be non-negative")
    total = float(weights_arr.sum())
    if total <= 0:
        raise ValueError("weights sum must be > 0")
    p = weights_arr / total

    out = np.full(wm.shape, 31, dtype=np.uint8)
    out[wm == 1] = 0

    mask_def = wm == 2
    n_def = int(mask_def.sum())
    if n_def:
        draws = rng.choice(grades_arr, size=n_def, p=p)
        out[mask_def] = draws
    return out


# ---------------------------------------------------------------------------
# Palette PNG save
# ---------------------------------------------------------------------------

def save_palette_png(
    idx_array: np.ndarray,
    path: Path,
    palette: list[int],
    target_size: tuple[int, int] | None = None,
) -> None:
    """Save ``idx_array`` as palette PNG, optionally NEAREST-resized."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(idx_array, mode="P")
    img.putpalette(palette)
    if target_size is not None:
        img = img.resize(target_size, Image.NEAREST)
    img.save(path, transparency=31, optimize=True)


# ---------------------------------------------------------------------------
# File naming (fail-map 5-field convention)
# ---------------------------------------------------------------------------

def _class_token(cls: str) -> str:
    return cls.replace("-", "").replace("_", "").upper()


def _filename_for(cls: str, wafer_idx: int, base_dt: datetime) -> str:
    tok = _class_token(cls)
    t = base_dt + timedelta(seconds=wafer_idx + 1)
    return f"{tok}_00P_W{wafer_idx:03d}_{t.strftime('%Y%m%d')}_{t.strftime('%H%M%S')}.png"


# ---------------------------------------------------------------------------
# pkl helpers
# ---------------------------------------------------------------------------

def _normalize_failure_type(x: Any) -> str:
    if isinstance(x, (list, np.ndarray)):
        arr = np.asarray(x).ravel()
        return "" if arr.size == 0 else str(arr[0])
    return str(x)


# ---------------------------------------------------------------------------
# YAML config
# ---------------------------------------------------------------------------

_ALLOWED_TEXTURE_MODES = {"none", "synthetic_scatter"}


def _validate_config(cfg: dict[str, Any]) -> None:
    required = ["version", "seed", "size", "upscale", "mapping", "split", "paths"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"pixel_rules.yaml missing keys: {missing}")
    if cfg["upscale"] != "nearest":
        raise ValueError(f"only 'nearest' upscale supported, got {cfg['upscale']!r}")
    mapping = cfg["mapping"]
    defect = mapping.get("defect", {})
    if defect.get("mode") != "random_skewed":
        raise ValueError(
            f"only 'random_skewed' defect mode supported, got {defect.get('mode')!r}"
        )
    grades = defect.get("grades", [])
    weights = defect.get("weights", [])
    if len(grades) != len(weights) or not grades:
        raise ValueError("grades and weights must be non-empty equal-length lists")
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")
    if sum(weights) <= 0:
        raise ValueError("weights sum must be > 0")
    texture = cfg.get("texture")
    if texture is not None:
        mode = texture.get("mode", "none")
        if mode not in _ALLOWED_TEXTURE_MODES:
            raise ValueError(
                f"unknown texture.mode {mode!r}; expected one of {_ALLOWED_TEXTURE_MODES}"
            )
        for k in ("defect_perturb_p", "normal_scatter_p"):
            v = texture.get(k, 0.0)
            if not (0.0 <= float(v) <= 1.0):
                raise ValueError(f"texture.{k} must be in [0, 1], got {v}")
        n = texture.get("global_scatter_n", 0)
        if int(n) < 0:
            raise ValueError("texture.global_scatter_n must be >= 0")
    # 3-way split support: train + val + test. Each split entry must contain
    # ``defect_n_per_class`` and ``normal_n``; ``paths`` must contain matching
    # ``<split>_out`` keys.
    splits = cfg.get("split", {})
    if "train" not in splits:
        raise ValueError("split.train is required")
    for name, s in splits.items():
        if "defect_n_per_class" not in s or "normal_n" not in s:
            raise ValueError(
                f"split.{name} missing defect_n_per_class or normal_n"
            )
        if f"{name}_out" not in cfg.get("paths", {}):
            raise ValueError(f"paths.{name}_out is required for split {name!r}")


def _load_config(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: YAML must parse to a dict")
    _validate_config(cfg)
    return cfg


# ---------------------------------------------------------------------------
# Split sampler
# ---------------------------------------------------------------------------

def split_indices_disjoint(
    rng: np.random.Generator,
    n_have: int,
    n_train: int,
    n_val: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx) — disjoint, no replacement.

    If ``n_train + n_val > n_have`` raises ValueError.
    """
    if n_train + n_val > n_have:
        raise ValueError(
            f"requested {n_train}+{n_val}={n_train + n_val} > pool {n_have}"
        )
    perm = rng.permutation(n_have)
    return perm[:n_train], perm[n_train : n_train + n_val]


def split_indices_multi(
    rng: np.random.Generator,
    n_have: int,
    splits: dict[str, int],
) -> dict[str, np.ndarray]:
    """Return ``{split_name: index_array}`` — all disjoint, no replacement.

    Iteration order of ``splits`` is preserved; earlier splits get earlier
    positions of the random permutation, which makes the 3-way cut reproducible
    given the same RNG state.
    """
    total = sum(int(v) for v in splits.values())
    if total > n_have:
        raise ValueError(
            f"requested splits sum to {total} > pool {n_have} "
            f"(splits={splits})"
        )
    perm = rng.permutation(n_have)
    out: dict[str, np.ndarray] = {}
    offset = 0
    for name, n in splits.items():
        n_i = int(n)
        out[name] = perm[offset : offset + n_i]
        offset += n_i
    return out


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def _generate_for_rows(
    subset: pd.DataFrame,
    out_dir: Path,
    cls: str,
    base_dt: datetime,
    rng: np.random.Generator,
    grades: list[int],
    weights: list[float],
    target_size: tuple[int, int],
    palette: list[int],
    texture: dict[str, Any] | None = None,
) -> int:
    saved = 0
    use_texture = (
        texture is not None and texture.get("mode", "none") == "synthetic_scatter"
    )
    for i, (_, row) in enumerate(subset.iterrows()):
        try:
            if use_texture:
                idx_arr = assign_grades_textured(
                    row["waferMap"],
                    rng,
                    target_size=(target_size[1], target_size[0]),  # (H, W)
                    grades=grades,
                    weights=weights,
                    defect_perturb_p=float(texture.get("defect_perturb_p", 0.30)),
                    normal_scatter_p=float(texture.get("normal_scatter_p", 0.03)),
                    normal_scatter_grades=tuple(
                        texture.get("normal_scatter_grades", (1,))
                    ),
                    global_scatter_n=int(texture.get("global_scatter_n", 0)),
                )
                # Already at target_size; save without extra resize.
                fname = _filename_for(cls, i, base_dt)
                save_palette_png(
                    idx_arr,
                    out_dir / cls / fname,
                    palette,
                    target_size=None,
                )
            else:
                idx_arr = assign_grades_random_skewed(
                    row["waferMap"], rng, grades=grades, weights=weights
                )
                fname = _filename_for(cls, i, base_dt)
                save_palette_png(
                    idx_arr,
                    out_dir / cls / fname,
                    palette,
                    target_size=target_size,
                )
            saved += 1
        except Exception as e:
            print(f"[skip] {cls}/{i}: {e}", file=sys.stderr)
    return saved


def run(cfg: dict[str, Any]) -> int:
    pkl_path = Path(cfg["paths"]["pkl"])
    size = tuple(cfg["size"])  # (H, W)
    target_size = (size[1], size[0])  # PIL resize wants (W, H)
    seed = int(cfg["seed"])

    grades = list(cfg["mapping"]["defect"]["grades"])
    weights = list(cfg["mapping"]["defect"]["weights"])
    texture = cfg.get("texture")

    # 3-way split (train/val/test) — any split names defined in YAML are
    # honoured. Each split must have matching ``<name>_out`` in paths.
    splits_cfg = cfg["split"]
    split_names = list(splits_cfg.keys())
    out_dirs = {name: Path(cfg["paths"][f"{name}_out"]) for name in split_names}

    if not pkl_path.exists():
        print(f"[err] PKL not found: {pkl_path}", file=sys.stderr)
        return 1

    print(f"[load] {pkl_path} ({pkl_path.stat().st_size / 1e9:.2f} GB)")
    df = pd.read_pickle(pkl_path)
    ft_col = "failureType" if "failureType" in df.columns else "faliureType"
    df["_ft"] = df[ft_col].apply(_normalize_failure_type)
    df = df[df["_ft"].isin(CLASSES)].reset_index(drop=True)
    print(f"[filter] kept {len(df)} wafers")

    palette = get_failmap_palette()
    base_dt = datetime.strptime("20260420_000000", "%Y%m%d_%H%M%S")

    master_rng = np.random.default_rng(seed)
    summaries: dict[str, dict[str, int]] = {name: {} for name in split_names}

    for cls in CLASSES:
        sub = df[df["_ft"] == cls].reset_index(drop=True)
        n_have = len(sub)
        if n_have == 0:
            print(f"[WARN] {cls}: 0 samples")
            for name in split_names:
                summaries[name][cls] = 0
            continue

        needs: dict[str, int] = {}
        for name in split_names:
            s = splits_cfg[name]
            if cls == "none":
                needs[name] = int(s["normal_n"])
            else:
                needs[name] = int(s["defect_n_per_class"])

        class_rng = np.random.default_rng(master_rng.integers(0, 2**63 - 1))
        split_idx = split_indices_multi(class_rng, n_have, needs)

        progress = "  ".join(f"{name}={len(split_idx[name])}" for name in split_names)
        print(f"[{cls:10s}] pool={n_have}  {progress}")

        for name in split_names:
            indices = split_idx[name]
            grade_rng = np.random.default_rng(master_rng.integers(0, 2**63 - 1))
            n_saved = _generate_for_rows(
                sub.iloc[indices],
                out_dirs[name],
                cls,
                base_dt,
                grade_rng,
                grades,
                weights,
                target_size,
                palette,
                texture=texture,
            )
            summaries[name][cls] = n_saved

    for name in split_names:
        out_dir = out_dirs[name]
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_obj = {
            "split": name,
            "seed": seed,
            "size": list(size),
            "class_counts": summaries[name],
            "total": sum(summaries[name].values()),
        }
        (out_dir / "summary.json").write_text(
            json.dumps(summary_obj, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[write] {out_dir}/summary.json  total={summary_obj['total']}")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/pixel_rules.yaml",
        help="stage 1 산출 YAML 경로",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(
            f"[err] config not found: {cfg_path}\n"
            f"       → run `python data_prep/analyze_wm811k.py` first",
            file=sys.stderr,
        )
        return 1

    cfg = _load_config(cfg_path)
    cfg_hash = hashlib.sha256(cfg_path.read_bytes()).hexdigest()[:16]
    print(f"[config] {cfg_path} (sha256 {cfg_hash})")

    return run(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
