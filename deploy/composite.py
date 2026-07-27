#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""composite map — `D:/project/mapviewer` 의 공식 규격을 그대로 따른다.

★ RGB 평균이 아니다. 원본(`api/composite_map.py`)은 **palette index 단위로 grade
  빈도를 쌓고, 그걸 스칼라 맵으로 환산해 colormap 으로 그린다.**
  grade 는 순서형 범주라서 색을 평균 내면 의미 없는 색이 나온다
  (빨강 grade6 + 초록 grade2 의 평균색은 어떤 grade 도 아니다).

원본 규칙 (api/composite_map.py:614-691, 1860-1885):

  누적 (`_numba_accumulate_image`)
    idx 0~7   -> grade_counts[idx] += 1,  has_0_7 = True
    idx 8~13  -> has_8_13 = True          (grade 카운트에 넣지 않음)
    idx >=14  -> **grade 0 으로 카운트**   (invalid 를 정상으로 취급)

  가중치
    SQ_WEIGHTS = [0, 1, 4, 9, 16, 25, 36, 49]   (= grade^2)
    WT_FACTORS = [1, 1, 2,  3,  4,  5,  6,  7]

  두 가지 맵
    square_average          = Σ(count[g]·g²) / 이미지수
    square_weighted_average = Σ(count[g]·g²) / Σ(count[g]·WT[g])

  마스크
    calc_mask     = grade 카운트가 있는 픽셀 & idx8 아님 & all-invalid 아님
    weighted_mask = calc_mask & 가중합 > 0

grade 부분집합(`grades=[3,5]` 등)을 주면 그 grade 만 카운트에 넣는다
(원본 파일명 규칙 `square_weighted_average_35.png` 와 동일한 의미).

사용:
    from composite import composite_maps, render_heatmap
    sq, wt, m_sq, m_wt = composite_maps(paths, size=1024)
    render_heatmap(sq, m_sq).save("square_average.png")
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 원본과 동일한 상수 (api/composite_map.py:662-663)
SQ_WEIGHTS = np.array([0, 1, 4, 9, 16, 25, 36, 49], dtype=np.float32)
WT_FACTORS = np.array([1, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32)

GRADE_MAX = 8        # grade 0~7
BAND_LO, BAND_HI = 8, 13     # 8~13 = border/background 대역 (grade 카운트 제외)
INVALID_FROM = 14            # 14 이상 = invalid -> grade 0 으로 카운트


def _indices(path, size: int | None) -> np.ndarray:
    """palette index 배열. RGB 로 바꾸지 않는다 — index 자체가 grade 다.

    리사이즈는 **NEAREST 만** 쓴다. index 는 범주값이라 보간하면
    없는 grade 가 만들어진다 (grade 2 와 6 사이를 보간하면 4 가 생긴다).
    """
    im = Image.open(path)
    if im.mode != "P":
        im = im.convert("P", palette=Image.Palette.ADAPTIVE)
    if size:
        im = im.resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(im, dtype=np.uint8)


def accumulate(paths, size: int | None = 1024, grades: list[int] | None = None):
    """이미지들을 훑어 grade 빈도와 마스크를 쌓는다 (원본 누적 규칙 그대로)."""
    keep = set(range(GRADE_MAX) if grades is None else grades)
    gc = None
    has_0_7 = has_8_13 = all_invalid = None
    n = 0
    for p in paths:
        a = _indices(p, size)
        if gc is None:
            H, W = a.shape
            gc = np.zeros((GRADE_MAX, H, W), dtype=np.int32)
            has_0_7 = np.zeros((H, W), dtype=bool)
            has_8_13 = np.zeros((H, W), dtype=bool)
            all_invalid = np.ones((H, W), dtype=bool)

        low = a < GRADE_MAX                       # 0~7
        band = (a >= BAND_LO) & (a <= BAND_HI)    # 8~13
        inv = a >= INVALID_FROM                   # 14+

        for g in range(GRADE_MAX):
            if g in keep:
                gc[g] += (low & (a == g))
        if 0 in keep:
            gc[0] += inv                          # ★ invalid -> grade 0
        has_0_7 |= low | inv
        has_8_13 |= band
        all_invalid &= ~(low | band)
        n += 1
    return gc, has_0_7, has_8_13, all_invalid, n


def composite_maps(paths, size: int | None = 1024, grades: list[int] | None = None):
    """(square_average, square_weighted_average, calc_mask, weighted_mask)."""
    gc, has_0_7, has_8_13, all_invalid, n = accumulate(paths, size, grades)
    if gc is None or n == 0:
        raise ValueError("이미지가 없다")
    gcf = gc.astype(np.float32)
    sq_sum = np.tensordot(SQ_WEIGHTS, gcf, axes=(0, 0))
    wt_sum = np.tensordot(WT_FACTORS, gcf, axes=(0, 0))

    calc = (gcf.sum(axis=0) > 0) & ~has_8_13 & ~all_invalid
    sq = np.zeros_like(sq_sum)
    sq[calc] = sq_sum[calc] / float(n)

    wmask = calc & (wt_sum > 0)
    wt = np.zeros_like(sq_sum)
    wt[wmask] = sq_sum[wmask] / wt_sum[wmask]
    return sq, wt, calc, wmask


def render_heatmap(value_map, mask, vmin=None, vmax=None, cmap="turbo"):
    """스칼라 맵 -> colormap PNG. 마스크 밖은 흰색 (원본 v_min/v_max 정규화와 동일)."""
    v = value_map[mask]
    if v.size == 0:
        return Image.fromarray(np.full(value_map.shape + (3,), 255, np.uint8))
    lo = float(v.min()) if vmin is None else vmin
    hi = float(v.max()) if vmax is None else vmax
    den = (hi - lo) or 1.0
    norm = np.clip((value_map - lo) / den, 0, 1)
    try:
        import matplotlib.cm as cm
        rgb = (cm.get_cmap(cmap)(norm)[..., :3] * 255).astype(np.uint8)
    except Exception:                    # matplotlib 없으면 회색조
        rgb = np.repeat((norm * 255).astype(np.uint8)[..., None], 3, axis=2)
    rgb[~mask] = 255
    return Image.fromarray(rgb)


def write_group_composites(paths, out_dir, size: int = 1024,
                           grades: list[int] | None = None, prefix: str = "") -> dict:
    """한 그룹에 대해 원본과 같은 두 장을 쓴다."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sq, wt, m_sq, m_wt = composite_maps(paths, size, grades)
    tag = "" if grades is None else "_" + "".join(str(g) for g in sorted(grades))
    render_heatmap(sq, m_sq).save(out / f"{prefix}square_average{tag}.png")
    render_heatmap(wt, m_wt).save(out / f"{prefix}square_weighted_average{tag}.png")
    return {"n": len(list(paths)), "size": size, "grades": grades,
            "sq_range": [float(sq[m_sq].min()), float(sq[m_sq].max())] if m_sq.any() else None,
            "wt_range": [float(wt[m_wt].min()), float(wt[m_wt].max())] if m_wt.any() else None}
