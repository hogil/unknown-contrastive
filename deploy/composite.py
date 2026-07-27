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

색 (api/composite_colors.py + logs/color-legends.json 실측):
  quantile0~100 = #FFFFFF -> #FFE6E6 -> ... -> #FF0000  (흰색→빨강 11단계)
  저장된 스킴(change/engqa01)이 전부 이 기본값을 쓴다.
  ★ 마스크 밖은 회색 배경이 아니라 **base wafer 의 palette 색 그대로** 깔린다
    (`_numba_render_composite` 의 else 분기 = palette[base_indices]).
  정규화는 (val - v_min)/(v_max - v_min) 선형 — quantile 은 색상표 위치일 뿐이다.

사용:
    from composite import composite_maps, render_composite
    sq, wt, m_sq, m_wt = composite_maps(paths, size=1024)
    render_composite(wt, m_wt, base_path=paths[0], size=1024).save("out.png")
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

GRADE_MAX = 8        # grade 0~7
BAND_LO, BAND_HI = 8, 13     # 8~13 = border/background 대역 (grade 카운트 제외)
INVALID_FROM = 14            # 14 이상 = invalid -> grade 0 으로 카운트

# 원본과 동일한 상수 (api/composite_map.py:662-663)
SQ_WEIGHTS = np.array([0, 1, 4, 9, 16, 25, 36, 49], dtype=np.float32)
WT_FACTORS = np.array([1, 1, 2, 3, 4, 5, 6, 7], dtype=np.float32)

# ── 변형 (uc): 고등급 과증폭 완화 ──────────────────────────────────────────
# 분자: grade 0,1 은 제곱, 2 부터는 2g  (grade2 는 4 로 양쪽 동일)
#   [0,1,4,9,16,25,36,49] -> [0,1,4,6,8,10,12,14].  grade7/grade2 = 12.25x -> 3.5x
# 분모: 전부 1 (= 등장 픽셀 수).  ★ 분모에 g 를 두면 2g/g = 2 로 등급이 전부 같아진다.
#   WT0 만 따로 둔다 — 정상(grade0) 픽셀이 분모를 얼마나 채울지 정하는 손잡이.
#   1.0 = 정상도 한 표 (드문 결함이 희석됨) / 0.0 = 결함 픽셀만 (드물어도 그대로 드러남)
UC_NUM = np.array([0, 1, 4, 6, 8, 10, 12, 14], dtype=np.float32)


def uc_den(wt0: float = 1.0) -> np.ndarray:
    d = np.ones(GRADE_MAX, dtype=np.float32)
    d[0] = np.float32(wt0)
    return d



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


def composite_maps(paths, size: int | None = 1024, grades: list[int] | None = None,
                   num=None, den=None):
    """(average, weighted_average, calc_mask, weighted_mask).

    num/den 을 주면 그 가중치로, 안 주면 mapviewer 원본 상수로 계산한다.
    """
    num = SQ_WEIGHTS if num is None else np.asarray(num, dtype=np.float32)
    den = WT_FACTORS if den is None else np.asarray(den, dtype=np.float32)
    gc, has_0_7, has_8_13, all_invalid, n = accumulate(paths, size, grades)
    if gc is None or n == 0:
        raise ValueError("이미지가 없다")
    gcf = gc.astype(np.float32)
    sq_sum = np.tensordot(num, gcf, axes=(0, 0))
    wt_sum = np.tensordot(den, gcf, axes=(0, 0))

    calc = (gcf.sum(axis=0) > 0) & ~has_8_13 & ~all_invalid
    sq = np.zeros_like(sq_sum)
    sq[calc] = sq_sum[calc] / float(n)

    wmask = calc & (wt_sum > 0)
    wt = np.zeros_like(sq_sum)
    wt[wmask] = sq_sum[wmask] / wt_sum[wmask]
    return sq, wt, calc, wmask


# ── 색 (mapviewer api/composite_colors.py + composite_map.py 규격) ─────────
# 실제 저장값 확인: mapviewer/logs/color-legends.json 의 composite.* 스킴이
# 전부 아래 흰색→빨강 11단계를 쓴다 (admin 만 quantile0 을 초록으로 바꿔둠).
COLOR_STOPS_HEX = ["#FFFFFF", "#FFE6E6", "#FFCCCC", "#FFB2B2", "#FF9999", "#FF8080",
                   "#FF6666", "#FF4D4D", "#FF3333", "#FF1919", "#FF0000"]
QUANTILE_POS = np.array([q for q in range(0, 101, 10)], dtype=np.float32)


def _hex(c: str) -> tuple[int, int, int]:
    c = c.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def build_lut(stops_hex=None, n: int = 256) -> np.ndarray:
    """색상표 -> 256 단계 LUT (원본 `_interpolate_percentile_colors` 와 동일)."""
    stops = np.array([_hex(c) for c in (stops_hex or COLOR_STOPS_HEX)], dtype=np.float32)
    pos = np.linspace(0.0, 100.0, len(stops), dtype=np.float32)
    q = np.linspace(0.0, 100.0, n, dtype=np.float32)
    lut = np.empty((n, 3), dtype=np.uint8)
    for ch in range(3):
        lut[:, ch] = np.clip(np.interp(q, pos, stops[:, ch]), 0, 255).astype(np.uint8)
    return lut


def render_composite(value_map, mask, base_path, size, vmin=None, vmax=None,
                     stops_hex=None):
    """원본 `_numba_render_composite` 그대로.

      mask 안  -> (val - v_min)/(v_max - v_min) 선형 정규화 후 LUT 색
      mask 밖  -> **base wafer 의 palette 색 그대로** (회색 배경이 아니다)

    base_path 는 배경으로 깔 원본 이미지(보통 그룹 medoid).
    """
    base = Image.open(base_path)
    if base.mode != "P":
        base = base.convert("P", palette=Image.Palette.ADAPTIVE)
    if size:
        base = base.resize((size, size), Image.Resampling.NEAREST)
    pal = base.getpalette() or []
    pal = (pal + [0] * (768 - len(pal)))[:768]
    pal = np.array(pal, dtype=np.uint8).reshape(256, 3)
    rgb = pal[np.asarray(base, dtype=np.uint8)]          # 마스크 밖 = 원본 wafer

    v = value_map[mask]
    if v.size:
        lo = float(v.min()) if vmin is None else float(vmin)
        hi = float(v.max()) if vmax is None else float(vmax)
        den = hi - lo
        if den > 0:
            lut = build_lut(stops_hex)
            scaled = np.clip((value_map[mask] - lo) / den, 0.0, 1.0)
            rgb[mask] = lut[np.rint(scaled * 255).astype(np.int32)]
    return Image.fromarray(rgb)


def write_group_composites(paths, out_dir, size: int = 1024,
                           grades: list[int] | None = None, prefix: str = "",
                           num=None, den=None, base_path=None) -> dict:
    """한 그룹에 대해 원본과 같은 두 장을 쓴다."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = list(paths)
    sq, wt, m_sq, m_wt = composite_maps(paths, size, grades, num=num, den=den)
    tag = "" if grades is None else "_" + "".join(str(g) for g in sorted(grades))
    base = base_path or paths[0]
    render_composite(sq, m_sq, base, size).save(out / f"{prefix}square_average{tag}.png")
    render_composite(wt, m_wt, base, size).save(out / f"{prefix}square_weighted_average{tag}.png")
    return {"n": len(paths), "size": size, "grades": grades,
            "sq_range": [float(sq[m_sq].min()), float(sq[m_sq].max())] if m_sq.any() else None,
            "wt_range": [float(wt[m_wt].min()), float(wt[m_wt].max())] if m_wt.any() else None}
