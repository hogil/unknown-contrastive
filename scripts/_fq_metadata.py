"""Synthetic FTN/QTN metadata for generated positions JSON.

The generated fail-bit PNG already carries the spatial anomaly signal. These
helpers add matching per-chip FTN/QTN vectors so a few deterministic test items
are high on the same wafer/chip locations as the abnormal BIN map.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

DEFAULT_FQ_ITEM_COUNT = 128
FTN_KEY_START = 1000
QTN_KEY_START = 5000
HOT_ITEM_COUNT = 8
SPREAD_RADIUS = 3


def stable_seed(*parts: object) -> int:
    text = "|".join(str(p) for p in parts)
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def ftn_keys(item_count: int = DEFAULT_FQ_ITEM_COUNT) -> List[int]:
    return list(range(FTN_KEY_START, FTN_KEY_START + int(item_count)))


def qtn_keys(item_count: int = DEFAULT_FQ_ITEM_COUNT) -> List[int]:
    return list(range(QTN_KEY_START, QTN_KEY_START + int(item_count)))


def _clean_token(value: str, fallback: str = "UNKNOWN") -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper()
    return token or fallback


def synthetic_partid(class_label: str, root: str) -> str:
    cls = _clean_token(class_label)[:28]
    lot = _clean_token(root, "LOT")[:12]
    return f"PART_{cls}_{lot}"


def synthetic_pgm(step: str, item_count: int = DEFAULT_FQ_ITEM_COUNT) -> str:
    step_token = _clean_token(step, "STEP")[:8]
    return f"PGM_SYN_FQ{int(item_count)}_{step_token}"


def _hot_indices(class_label: str, item_count: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(stable_seed("hot-items", class_label, item_count))
    n_hot = min(HOT_ITEM_COUNT, max(1, item_count // 4))
    f_hot = np.sort(rng.choice(item_count, size=n_hot, replace=False))
    q_hot = np.sort(rng.choice(item_count, size=n_hot, replace=False))
    return f_hot.astype(np.int64), q_hot.astype(np.int64)


def _chip_bin(chip: Dict[str, Any]) -> int:
    try:
        return int(str(chip.get("b", "0")).strip())
    except Exception:
        return 0


def _defect_score_grid(chips: Sequence[Dict[str, Any]]) -> np.ndarray:
    max_x = max((int(c.get("x_abs", 0)) for c in chips), default=31)
    max_y = max((int(c.get("y_abs", 0)) for c in chips), default=31)
    w = max(32, max_x + 1)
    h = max(32, max_y + 1)
    score = np.zeros((h, w), dtype=np.float32)
    defect_xy: List[Tuple[int, int]] = []

    for chip in chips:
        b = _chip_bin(chip)
        if b >= 200:
            y = int(chip.get("y_abs", 0))
            x = int(chip.get("x_abs", 0))
            if 0 <= y < h and 0 <= x < w:
                defect_xy.append((y, x))

    if not defect_xy:
        return score

    offsets: List[Tuple[int, int, float]] = []
    for dy in range(-SPREAD_RADIUS, SPREAD_RADIUS + 1):
        for dx in range(-SPREAD_RADIUS, SPREAD_RADIUS + 1):
            d2 = dx * dx + dy * dy
            if d2 <= SPREAD_RADIUS * SPREAD_RADIUS:
                offsets.append((dy, dx, float(np.exp(-d2 / 4.0))))

    for y, x in defect_xy:
        for dy, dx, weight in offsets:
            yy = y + dy
            xx = x + dx
            if 0 <= yy < h and 0 <= xx < w and weight > score[yy, xx]:
                score[yy, xx] = weight
    return score


def _chip_scores(chips: Sequence[Dict[str, Any]]) -> np.ndarray:
    grid = _defect_score_grid(chips)
    values = np.zeros(len(chips), dtype=np.float32)
    for i, chip in enumerate(chips):
        y = int(chip.get("y_abs", 0))
        x = int(chip.get("x_abs", 0))
        if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
            values[i] = grid[y, x]
    return values


def synthetic_fq_arrays(
    chips: Sequence[Dict[str, Any]],
    class_label: str,
    seed_text: str,
    item_count: int = DEFAULT_FQ_ITEM_COUNT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return dense int arrays shaped (n_chips, item_count) for FTN and QTN."""
    item_count = int(item_count)
    rng = np.random.default_rng(stable_seed("fq-values", class_label, seed_text, item_count))
    n = len(chips)
    scores = _chip_scores(chips)
    f_hot, q_hot = _hot_indices(class_label, item_count)

    # Low baseline with a long tail, similar to reference bucket-B vectors.
    f = rng.lognormal(mean=4.85, sigma=1.25, size=(n, item_count)).astype(np.int32)
    q = rng.lognormal(mean=4.65, sigma=1.30, size=(n, item_count)).astype(np.int32)

    if n and len(f_hot):
        score_col = scores[:, None] ** 1.25
        f_boost = rng.normal(loc=28000.0, scale=4500.0, size=(n, len(f_hot)))
        q_boost = rng.normal(loc=22000.0, scale=4000.0, size=(n, len(q_hot)))
        f[:, f_hot] += np.maximum(0, f_boost * score_col).astype(np.int32)
        q[:, q_hot] += np.maximum(0, q_boost * score_col).astype(np.int32)

        # A smaller shoulder on neighboring item ids makes the signal less toy-like.
        for hot in f_hot:
            for nb in (hot - 1, hot + 1):
                if 0 <= nb < item_count:
                    f[:, nb] += (scores * rng.integers(2500, 5500)).astype(np.int32)
        for hot in q_hot:
            for nb in (hot - 1, hot + 1):
                if 0 <= nb < item_count:
                    q[:, nb] += (scores * rng.integers(1800, 4800)).astype(np.int32)

    np.clip(f, 0, 99999, out=f)
    np.clip(q, 0, 99999, out=q)
    return f, q


def add_synthetic_fq_to_json(
    obj: Dict[str, Any],
    class_label: str,
    seed_text: str,
    item_count: int = DEFAULT_FQ_ITEM_COUNT,
    overwrite_fq: bool = False,
) -> bool:
    """Mutate a positions JSON object in place.

    Returns True when chip FTN/QTN arrays were added or replaced. Top-level
    keys (partid/part_id/pgm/ftn_keys/qtn_keys) are always populated/mirrored.
    """
    item_count = int(item_count)
    obj["ftn_keys"] = ftn_keys(item_count)
    obj["qtn_keys"] = qtn_keys(item_count)
    if not obj.get("partid"):
        obj["partid"] = synthetic_partid(class_label, str(obj.get("root", "")))
    if not obj.get("pgm"):
        obj["pgm"] = synthetic_pgm(str(obj.get("step", "")), item_count)
    # part_id (snake_case) — partid mirror로 두 컨벤션 호환 유지
    obj["part_id"] = obj["partid"]

    chips = obj.get("chips") or []
    needs_fq = overwrite_fq or any(("f" not in chip or "q" not in chip) for chip in chips)
    if not needs_fq:
        return False

    f_values, q_values = synthetic_fq_arrays(chips, class_label, seed_text, item_count)
    for chip, f_row, q_row in zip(chips, f_values, q_values):
        chip["f"] = f_row.tolist()
        chip["q"] = q_row.tolist()
    return True
