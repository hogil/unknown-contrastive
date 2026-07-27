#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""composite map — `D:/project/mapviewer` 의 공식 규격을 우리 palette 에 맞춰 적용한다.

★ RGB 평균이 아니다. 원본(`api/composite_map.py`)은 **palette index 단위로 grade
  빈도를 쌓고, 그걸 스칼라 맵으로 환산해 colormap 으로 그린다.**
  grade 는 순서형 범주라서 색을 평균 내면 의미 없는 색이 나온다
  (빨강 grade6 + 초록 grade2 의 평균색은 어떤 grade 도 아니다).

── 우리 palette 실측 (32 entry) ───────────────────────────────────────────
    idx 0~7    grade 0~7                      -> **계산 대상**
    idx 8      RGB(220,238,255)  배경          -> 원본색 유지
    idx 9      RGB(0,0,1)        bin 번호 text ┐
    idx 10     RGB(190,190,190)  경계 회색     │
    idx 11     RGB(255,153,0)    주황          │ 전부 chip 경계.
    idx 12~23  파랑/연주황/청록/자주/노랑/초록… │ **한 색(idx 10 회색)으로 통일**
    idx 24~30  (예비)                          ┘
    idx 31     투명                            -> 원본색 유지

  ★ 경계 index 는 **이미지마다 다르게 등장한다** (24장 중 20장이 서로 다른 집합).
    base 한 장만 보고 그리면 그 장에 없는 경계선이 통째로 빠진다.
    -> 경계 마스크는 **전체 이미지의 union** 으로 잡는다.

  ★ mapviewer 의 `idx >= 14 -> grade 0 으로 카운트` 규칙은 그쪽 데이터 기준이다.
    우리 palette 에서 14~23 은 경계 마커라, 그 규칙을 그대로 쓰면
    **경계선이 grade 0 픽셀로 집계돼 히트맵에 덮여 사라진다.**

가중치 (api/composite_map.py:662-663)
    SQ_WEIGHTS = [0, 1, 4, 9, 16, 25, 36, 49]   (= grade^2)
    WT_FACTORS = [1, 1, 2,  3,  4,  5,  6,  7]

두 가지 맵
    square_average          = Σ(count[g]·g²) / 이미지수
    square_weighted_average = Σ(count[g]·g²) / Σ(count[g]·WT[g])

grade 부분집합(`grades=[3,5]` 등)을 주면 그 grade 만 카운트에 넣는다
(원본 파일명 규칙 `square_weighted_average_35.png` 와 동일한 의미).

색 (api/composite_colors.py + logs/color-legends.json 실측):
  quantile0~100 = #FFFFFF -> #FFE6E6 -> ... -> #FF0000  (흰색→빨강 11단계)
  정규화는 (val - v_min)/(v_max - v_min) 선형 — quantile 은 색상표 위치일 뿐이다.

해상도는 **원본 그대로**. 리사이즈하지 않는다 (경계 1px 이 사라진다).

사용:
    from composite import composite_full, render_composite
    avg, wt, m, mw, border, text, n = composite_full(paths)
    render_composite(wt, mw, border, base_path=paths[0], text=text).save("out.png")
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

GRADE_MAX = 8          # grade 0~7
BG_IDX = 8             # 배경 — 원본색 유지
TEXT_IDX = 9           # chip 안 bin 번호. ★ composite 에 절대 남으면 안 된다
BORDER_LO = 10         # 10~30 = chip 경계 대역 (회색격자 + 컬러마커 전부)
BORDER_HI = 30
BORDER_IDX = 10        # 경계 대표색. 모든 경계는 이 색 하나로 통일한다
TRANSPARENT_IDX = 31   # 투명 — 원본색 유지

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


# ── 색 (mapviewer api/composite_colors.py + logs/color-legends.json 실측) ──
COLOR_STOPS_HEX = ["#FFFFFF", "#FFE6E6", "#FFCCCC", "#FFB2B2", "#FF9999", "#FF8080",
                   "#FF6666", "#FF4D4D", "#FF3333", "#FF1919", "#FF0000"]


def _hex(c: str):
    c = c.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def build_lut(stops_hex=None, n: int = 256) -> np.ndarray:
    stops = np.array([_hex(c) for c in (stops_hex or COLOR_STOPS_HEX)], np.float32)
    pos = np.linspace(0.0, 100.0, len(stops), dtype=np.float32)
    q = np.linspace(0.0, 100.0, n, dtype=np.float32)
    lut = np.empty((n, 3), np.uint8)
    for ch in range(3):
        lut[:, ch] = np.clip(np.interp(q, pos, stops[:, ch]), 0, 255).astype(np.uint8)
    return lut


def accumulate_sums(paths, num, den, grades=None):
    """원본 해상도 그대로 누적. 리사이즈 없음.

    8채널 카운트 배열(6400^2 x 8)은 메모리를 너무 먹으므로
    분자합·분모합만 float32 로 바로 쌓는다.

    반환 border 는 **전체 이미지 union** — 어느 한 장에라도 경계면 경계다.
    반환 cnt 는 **픽셀별 유효 장수**. 고정 n 으로 나누면 안 된다:
      어떤 픽셀이 24장 중 3장에서 bin 번호 text 에 가려지면 그 3장 몫이 빠진 채
      24 로 나뉘어 **더 옅은 유령 글자**가 남는다. 경계 위치가 장마다 다른 것도 같다.
    """
    keep = set(range(GRADE_MAX) if grades is None else grades)
    keep_arr = np.zeros(GRADE_MAX, bool)
    for g in keep:
        keep_arr[g] = True

    num_sum = den_sum = cnt = has_grade = border = text = None
    n = 0
    for p in paths:
        im = Image.open(p)
        if im.mode != "P":
            im = im.convert("P", palette=Image.Palette.ADAPTIVE)
        a = np.asarray(im, dtype=np.uint8)
        if num_sum is None:
            num_sum = np.zeros(a.shape, np.float32)
            den_sum = np.zeros(a.shape, np.float32)
            cnt = np.zeros(a.shape, np.float32)
            has_grade = np.zeros(a.shape, bool)
            border = np.zeros(a.shape, bool)
            text = np.zeros(a.shape, bool)
        low = a < GRADE_MAX
        g = np.where(low, a, 0)
        valid = low & keep_arr[g]
        num_sum += np.where(valid, num[g], 0).astype(np.float32)
        den_sum += np.where(valid, den[g], 0).astype(np.float32)
        cnt += valid
        has_grade |= low
        border |= (a >= BORDER_LO) & (a <= BORDER_HI)
        text |= a == TEXT_IDX
        n += 1
    return num_sum, den_sum, cnt, has_grade, border, text, n


def composite_full(paths, num=None, den=None, grades=None):
    """원본 해상도 composite. (average, weighted, mask, weighted_mask, border, text, n)"""
    num = SQ_WEIGHTS if num is None else np.asarray(num, np.float32)
    den = WT_FACTORS if den is None else np.asarray(den, np.float32)
    ns, ds, cnt, has_grade, border, text, n = accumulate_sums(paths, num, den, grades)
    if ns is None:
        raise ValueError("이미지가 없다")
    # grade 픽셀만. 경계는 어느 장에서든 경계면 제외.
    # ★ text 는 제외하지 않는다 — 가려진 장만 빼고 나머지 장으로 평균 내면 글자가 사라진다.
    mask = has_grade & ~border
    avg = np.zeros_like(ns)
    amask = mask & (cnt > 0)
    avg[amask] = ns[amask] / cnt[amask]      # ★ 고정 n 이 아니라 픽셀별 유효 장수
    wmask = mask & (ds > 0)
    wt = np.zeros_like(ns)
    wt[wmask] = ns[wmask] / ds[wmask]
    return avg, wt, amask, wmask, border, text, n


def render_composite(value_map, mask, border, base_path, text=None,
                     vmin=None, vmax=None, stops_hex=None):
    """원본 해상도 그대로 렌더. 리사이즈 없음 — 화질이 원본과 동일하다.

      chip 안 bin 번호(text) -> **완전히 지운다.** 다른 장의 grade 로 덮이고,
                                전 장이 text 인 픽셀은 grade0 색으로 메운다.
      grade 픽셀(mask)  -> (val-v_min)/(v_max-v_min) 선형 -> 흰색→빨강 LUT
      경계(border)      -> **전부 idx 10 회색 한 색**
                           (원본의 파랑·초록·노랑·자주·주황 마커도 회색으로 통일)
      그 외             -> 원본 palette 색 그대로 (배경 idx8, 투명 idx31)
    """
    base = Image.open(base_path)
    if base.mode != "P":
        base = base.convert("P", palette=Image.Palette.ADAPTIVE)
    raw = base.getpalette() or []
    pal = np.array((raw + [0] * (768 - len(raw)))[:768], np.uint8).reshape(256, 3)
    idx = np.asarray(base, np.uint8)
    rgb = pal[idx]

    # ★ text 선삭제. base 에 남은 글자를 grade0 색으로 메운 뒤 heat 로 덮는다.
    #   전 장이 text 인 픽셀(mask 밖)도 여기서 흰색이 되어 글자가 남지 않는다.
    t = (idx == TEXT_IDX) if text is None else (text | (idx == TEXT_IDX))
    rgb[t] = pal[0]

    # 경계: 전체 union 을 한 색으로. base 에 없던 줄도 여기서 채워진다.
    rgb[border] = pal[BORDER_IDX]

    v = value_map[mask]
    if v.size:
        lo = float(v.min()) if vmin is None else float(vmin)
        hi = float(v.max()) if vmax is None else float(vmax)
        if hi > lo:
            lut = build_lut(stops_hex)
            sc = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
            rgb[mask] = lut[np.rint(sc * 255).astype(np.int32)]
    return Image.fromarray(rgb)


def write_group_composites(paths, out_dir, grades: list[int] | None = None,
                           prefix: str = "", num=None, den=None,
                           base_path=None) -> dict:
    """한 그룹에 대해 원본과 같은 두 장을 원본 해상도로 쓴다."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = list(paths)
    sq, wt, m_sq, m_wt, border, text, n = composite_full(
        paths, num=num, den=den, grades=grades)
    tag = "" if grades is None else "_" + "".join(str(g) for g in sorted(grades))
    base = base_path or paths[0]
    render_composite(sq, m_sq, border, base, text).save(
        out / f"{prefix}square_average{tag}.png")
    render_composite(wt, m_wt, border, base, text).save(
        out / f"{prefix}square_weighted_average{tag}.png")
    return {"n": n, "grades": grades,
            "sq_range": [float(sq[m_sq].min()), float(sq[m_sq].max())] if m_sq.any() else None,
            "wt_range": [float(wt[m_wt].min()), float(wt[m_wt].max())] if m_wt.any() else None}
