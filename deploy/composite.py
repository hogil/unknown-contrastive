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
TRANSPARENT_IDX = 31   # invalid 칩 **내부** (PNG 투명 인덱스로 그려진다)
INVALID_IDX = 11       # invalid 칩 **테두리** (주황)
# 실측 근거 (Center_invalid_main 한 장):
#   idx31  733,112px / 43 덩어리, 최대 36,946px ≈ 칩 하나(200x200)  -> 칩 내부
#   idx11   31,680px / 9 덩어리, 둘레의 49.7% 가 idx31 과 접함        -> 그 테두리
#   위치 y[1202,4597] x[1802,4197] = 중앙  -> 클래스명(Center_invalid_main)과 일치
#
# invalid 처리 방식 (deploy/config.py::Composite.INVALID):
#   "grade0"  ★ 채택 — invalid 를 **grade 0 으로 카운트**한다. mapviewer 원본의
#             `idx >= 14 -> grade 0` 과 같은 취지다. invalid 는 "측정 안 된 정상"이므로
#             그 자리 평균을 낮춰 **가장 옅은 색**으로 나타난다.
#   "border"  테두리는 경계 회색, 내부는 다른 웨이퍼 grade 로 칠해진다.
#             ← 예전 동작. invalid 클래스에서 결함 정보가 **하나도 안 남는다**(실측):
#               idx11 100% 가 border 로, idx31 100% 가 남의 heat 로 흡수됐다.
#   "exclude" 아예 카운트에서 뺀다(기권). 그 자리는 나머지 장으로만 평균낸다.

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


# ── 색 (mapviewer api/composite_colors.py) ────────────────────────────────
# UI 와 동일한 구조: **quantile 0/10/20/…/100 지점에 색을 하나씩 박고 그 사이를 선형 보간.**
#   QUANTILE_KEYS   = [quantile0 … quantile100]        (11개)
#   기본색           = gb = round(255·(1-step/100)) -> "#FF{gb}{gb}"   (공식, 하드코딩 아님)
#   보간             = _interpolate_percentile_colors -> np.interp
#   값→위치          = min-max 선형 (`_percentile_ranks` 는 이름과 달리 순위가 아니다.
#                      docstring: "기존 Percentile(순위) 방식 대신 값의 크기를 그대로 반영")
# 따라서 "아래쪽을 연하게" 하려면 감마 같은 걸 끼우는 게 아니라
# **낮은 quantile 의 색을 흰쪽으로 옮기면 된다** — UI 에서 하는 것과 같은 조작.
QUANTILE_STEPS = list(range(0, 101, 10))


def default_stops() -> list[str]:
    """mapviewer `_default_color_for_step` 과 같은 식. 부동소수 오차까지 동일하게 재현된다
    (step90 -> 255*(1-0.9)=25.499… -> 25 -> #FF1919, logs/color-legends.json 과 일치)."""
    out = []
    for s in QUANTILE_STEPS:
        gb = int(round(255 * (1 - s / 100)))
        out.append(f"#FF{gb:02X}{gb:02X}")
    return out


def light_low_stops(strength: float = 0.5) -> list[str]:
    """아래쪽(낮은 quantile)만 흰쪽으로 당긴 색상표. 위쪽은 건드리지 않는다.

    가중치 w = (1 - step/100)^2 로 낮은 step 일수록 크게 흰색과 섞는다.
    strength 0 = 기본색 그대로, 1 = 아래쪽이 거의 흰색.
    quantile100(#FF0000) 은 w=0 이라 항상 그대로다.
    """
    out = []
    for s in QUANTILE_STEPS:
        gb = 255 * (1 - s / 100)
        w = strength * (1 - s / 100) ** 2
        gb = gb + (255 - gb) * w          # 흰색(255) 쪽으로 w 만큼
        v = int(round(min(255.0, gb)))
        out.append(f"#FF{v:02X}{v:02X}")
    return out


def _read_stops(path, scheme: str, strict: bool):
    """한 파일에서 한 스킴을 읽는다. 못 읽으면 None."""
    import json
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / p
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        if strict:
            raise
        return None
    entry = (doc.get("composite") or {}).get(scheme)
    if not isinstance(entry, dict):
        if strict:
            raise KeyError(f"{p} 에 composite.{scheme} 없음")
        return None
    out, missing = [], []
    for s in QUANTILE_STEPS:
        c = entry.get(f"quantile{s}")
        if not c:
            missing.append(s)
            c = default_stops()[QUANTILE_STEPS.index(s)]
        out.append(str(c).upper())
    if missing and strict:
        raise KeyError(f"{p} scheme={scheme}: quantile{missing} 없음")
    return out


def load_stops(path=None, scheme: str | None = None, strict: bool = False):
    """`color-legends.json` 에서 색상표를 읽는다 (mapviewer 와 같은 스키마).

        {"composite": {"<scheme>": {"quantile0": "#FFFFFF", ... "quantile100": "#FF0000"}}}

    mapviewer 의 `logs/color-legends.json` 을 절대경로로 그대로 지정해도 읽힌다.

    찾는 순서 — **앞에서 못 찾으면 다음으로 넘어간다**:
      1. path (config 의 COLOR_LEGENDS. 절대경로 가능)
      2. 동봉본 `deploy/color-legends.json`
         ★ 1번이 사내 서버에 없을 때를 위한 안전망. 이게 없으면 절대경로를
           박아둔 순간 서버에서 색이 계산 기본값으로 조용히 바뀐다.
      3. 계산 기본색 (mapviewer 공식과 동일)
    strict=True 면 1번에서 실패 시 예외.
    """
    scheme = scheme or DEFAULT_SCHEME
    if path:
        got = _read_stops(path, scheme, strict)
        if got:
            return got
    if not path or str(path).replace("\\", "/") != DEFAULT_LEGENDS:
        got = _read_stops(DEFAULT_LEGENDS, scheme, False)
        if got:
            return got
    return default_stops()


COLOR_STOPS_HEX = default_stops()


def _hex(c: str):
    c = c.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def build_lut(stops_hex=None, n: int = 256, positions=None) -> np.ndarray:
    """색상표 -> 256 단계 LUT. mapviewer `_interpolate_percentile_colors` 와 동일.

    positions 를 주면 각 색이 놓이는 백분위 지점을 바꿀 수 있다 (기본 0,10,…,100).
    """
    stops = np.array([_hex(c) for c in (stops_hex or COLOR_STOPS_HEX)], np.float32)
    if positions is None:
        pos = np.linspace(0.0, 100.0, len(stops), dtype=np.float32)
    else:
        pos = np.clip(np.asarray(positions, np.float32), 0.0, 100.0)
    q = np.linspace(0.0, 100.0, n, dtype=np.float32)
    lut = np.empty((n, 3), np.uint8)
    for ch in range(3):
        lut[:, ch] = np.clip(np.interp(q, pos, stops[:, ch]), 0, 255).astype(np.uint8)
    return lut


# ── numba 가속 (mapviewer 방식) ─────────────────────────────────────────
# mapviewer(api/composite_map.py) 는 픽셀별 누적을 numba `njit(parallel=True)` 로
# 컴파일해서 돈다. 우리는 순수 numpy 로 이미지당 10개 넘는 전체배열 연산
# (where/fancy-index 마다 6400x6400 새 배열 할당)을 하고 있었다 — 실측 이미지당 630ms.
# 단일 컴파일 패스로 합치면 247ms (2.5배). numba 없으면 기존 numpy 경로로 자동 폴백한다.
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:
    njit = prange = None
    _HAS_NUMBA = False


if _HAS_NUMBA:
    @njit(cache=True)
    def _nb_fill_holes_sub(inv):
        """scipy.ndimage.binary_fill_holes 와 동치인 4-connectivity flood fill.
        bbox 로 잘라 들어온 작은 배열에 쓴다 (inv 전체가 아니라)."""
        H, W = inv.shape
        outside = np.zeros((H, W), dtype=np.bool_)
        stack = np.empty(H * W, dtype=np.int64)
        sp = 0
        for x in range(W):
            if not inv[0, x] and not outside[0, x]:
                outside[0, x] = True; stack[sp] = x; sp += 1
            if not inv[H - 1, x] and not outside[H - 1, x]:
                outside[H - 1, x] = True; stack[sp] = (H - 1) * W + x; sp += 1
        for y in range(H):
            if not inv[y, 0] and not outside[y, 0]:
                outside[y, 0] = True; stack[sp] = y * W; sp += 1
            if not inv[y, W - 1] and not outside[y, W - 1]:
                outside[y, W - 1] = True; stack[sp] = y * W + (W - 1); sp += 1
        while sp > 0:
            sp -= 1
            idx = stack[sp]
            y = idx // W; x = idx % W
            if y > 0 and not inv[y - 1, x] and not outside[y - 1, x]:
                outside[y - 1, x] = True; stack[sp] = (y - 1) * W + x; sp += 1
            if y < H - 1 and not inv[y + 1, x] and not outside[y + 1, x]:
                outside[y + 1, x] = True; stack[sp] = (y + 1) * W + x; sp += 1
            if x > 0 and not inv[y, x - 1] and not outside[y, x - 1]:
                outside[y, x - 1] = True; stack[sp] = y * W + (x - 1); sp += 1
            if x < W - 1 and not inv[y, x + 1] and not outside[y, x + 1]:
                outside[y, x + 1] = True; stack[sp] = y * W + (x + 1); sp += 1
        return ~outside

    @njit(parallel=True, cache=True)
    def _nb_accumulate_one(a, inv, num_lut, den_lut, keep_lut,
                           num_sum, den_sum, cnt, has_grade, border, text):
        """한 이미지의 누적을 **단일 컴파일 패스**로. numpy 판(accumulate_sums 의
        grade0 분기)과 값이 완전히 같아야 한다 — 검증: 랜덤/실데이터 byte-exact 일치."""
        H, W = a.shape
        for y in prange(H):
            for x in range(W):
                v = a[y, x]
                is_inv = inv[y, x]
                if is_inv:
                    g = 0
                    take = True
                elif v < GRADE_MAX:
                    g = v
                    take = True
                else:
                    g = 0
                    take = False
                if take and keep_lut[g]:
                    num_sum[y, x] += num_lut[g]
                    den_sum[y, x] += den_lut[g]
                    cnt[y, x] += np.float32(1.0)
                    has_grade[y, x] = True
                if (v >= BORDER_LO) and (v <= BORDER_HI) and not is_inv:
                    border[y, x] = True
                if v == TEXT_IDX:
                    text[y, x] = True


def _nb_fill_invalid_chips(inv):
    """`_fill_invalid_chips` 의 numba 가속판. bbox 를 numpy 로 찾고(cheap),
    그 안만 `_nb_fill_holes_sub` 로 채운다. scipy 대비 실측 2~5배."""
    if not inv.any():
        return inv
    ys = np.where(inv.any(1))[0]
    xs = np.where(inv.any(0))[0]
    y0, y1, x0, x1 = ys[0], ys[-1] + 1, xs[0], xs[-1] + 1
    out = inv.copy()
    out[y0:y1, x0:x1] = _nb_fill_holes_sub(np.ascontiguousarray(inv[y0:y1, x0:x1]))
    return out


def _decode_indices(paths, workers: int = 0, prefetch: int = 0):
    """palette index 배열을 **스레드로 미리 디코드**해서 순서대로 흘려준다.

    composite 는 원본 6400x6400 을 그대로 읽어야 해서 (임베딩 캐시는 384 라 못 쓴다)
    디코드가 비용의 대부분이다 — 실측 24장 12.84s 중 대부분. PIL 디코드는 GIL 을
    놓으므로 스레드가 실제로 병렬로 돈다 (`grouping_deploy._iter_batches` 와 동일 근거).
    ★ prefetch 가 worker 수보다 작으면 스레드를 열어놓고도 못 쓴다 — 예전엔
      workers=8 인데 prefetch=4 로 고정돼 있어서 최대 4장만 동시에 돌았다.
      기본을 workers 와 맞춰서 실제로 다 쓰게 한다.
    ★ worker 수는 코어의 **절반**만 쓴다 — 이 머신은 embedding 디코드(전체 코어)나
      다른 작업과 동시에 돈다. 전체 코어를 다 잡으면 그것들을 굶긴다.
    메모리는 6400^2 uint8 = 41MB * prefetch 만 쓴다.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    def one(p):
        im = Image.open(p)
        if im.mode != "P":
            im = im.convert("P", palette=Image.Palette.ADAPTIVE)
        return np.asarray(im, dtype=np.uint8)

    w = workers or int(os.environ.get("SITE_COMPOSITE_WORKERS", "0")) \
        or max(2, (os.cpu_count() or 4) // 2)
    prefetch = prefetch or max(w, 4)
    paths = list(paths)
    with ThreadPoolExecutor(max_workers=w) as ex:
        futs = [ex.submit(one, p) for p in paths[:prefetch]]
        nxt = prefetch
        for k in range(len(paths)):
            a = futs[k].result()
            if nxt < len(paths):
                futs.append(ex.submit(one, paths[nxt])); nxt += 1
            yield a


def _fill_invalid_chips(inv):
    """invalid 칩 영역을 확정한다 — 테두리·내부의 구멍(bin 번호, 남은 grade 픽셀)을 메운다.

    6400x6400 전체에 fill_holes 를 돌리면 비싸므로 invalid 의 bounding box 안에서만 한다.
    scipy 가 없으면 원본을 그대로 돌려준다(글자가 조금 드러날 뿐 결과가 깨지진 않는다).
    """
    if not inv.any():
        return inv
    try:
        from scipy import ndimage
    except Exception:
        return inv
    ys, xs = np.where(inv.any(1))[0], np.where(inv.any(0))[0]
    y0, y1, x0, x1 = ys[0], ys[-1] + 1, xs[0], xs[-1] + 1
    out = inv.copy()
    out[y0:y1, x0:x1] = ndimage.binary_fill_holes(inv[y0:y1, x0:x1])
    return out


def invalid_mode() -> str:
    """config 의 Composite.INVALID. 못 읽으면 grade0."""
    c = _cfg()
    return str(getattr(c, "INVALID", "grade0") if c else "grade0").lower()


def accumulate_sums(paths, num, den, grades=None, invalid=None):
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

    # ★ numba 컴파일 패스 — grade0(기본, 유일하게 쓰이는 설정)만 가속한다.
    #   border/exclude 모드는 드물게 쓰이므로 검증된 numpy 경로를 그대로 둔다.
    #   실측(10장, 6400x6400): numpy 630ms/장 -> numba 247ms/장 (2.5배).
    use_nb = _HAS_NUMBA and invalid == "grade0"
    num32 = num.astype(np.float32)
    den32 = den.astype(np.float32)

    num_sum = den_sum = cnt = has_grade = border = text = None
    n = 0
    for a in _decode_indices(paths):
        if num_sum is None:
            num_sum = np.zeros(a.shape, np.float32)
            den_sum = np.zeros(a.shape, np.float32)
            cnt = np.zeros(a.shape, np.float32)
            has_grade = np.zeros(a.shape, bool)
            border = np.zeros(a.shape, bool)
            text = np.zeros(a.shape, bool)

        if use_nb:
            inv = _nb_fill_invalid_chips((a == INVALID_IDX) | (a == TRANSPARENT_IDX))
            _nb_accumulate_one(a, inv, num32, den32, keep_arr,
                               num_sum, den_sum, cnt, has_grade, border, text)
            n += 1
            continue

        low = a < GRADE_MAX
        inv = (a == INVALID_IDX) | (a == TRANSPARENT_IDX)
        if invalid == "grade0":
            # ★ **invalid 칩 안의 모든 픽셀을 grade 0 으로** 센다.
            #   테두리(11)·내부(31) 만으로는 부족하다. 구멍을 메운 칩 영역 안에는
            #   실측으로 bin 번호 text(9), 경계회색(10), 그리고 **grade 1·2 픽셀까지**
            #   들어 있다 (6장에 grade 196,020px 중 1·2 가 71,215px).
            #   - text 를 빼면 그 픽셀만 이 웨이퍼가 기권해 남의 grade 로 평균이 나서
            #     invalid 자리보다 20% 높아지고 **글자가 다시 드러난다**(0.354 vs 0.294).
            #   - grade 1·2 를 제 값으로 세면 "invalid = 측정 안 됨" 이라는 전제가 깨진다.
            #   그래서 칩 영역을 hole-fill 로 확정하고 그 안은 전부 g=0 으로 강제한다.
            inv = _fill_invalid_chips(inv)
            take = low | inv
            g = np.where(low & ~inv, a, 0)      # 칩 안은 무조건 grade 0
        else:
            take = low
            g = np.where(low, a, 0)
        valid = take & keep_arr[g]
        num_sum += np.where(valid, num[g], 0).astype(np.float32)
        den_sum += np.where(valid, den[g], 0).astype(np.float32)
        cnt += valid
        has_grade |= take
        b = (a >= BORDER_LO) & (a <= BORDER_HI)
        if invalid == "grade0":
            b &= ~inv          # invalid 테두리를 경계로 잡으면 grade0 처리가 무의미해진다
        border |= b
        text |= a == TEXT_IDX
        n += 1
    return num_sum, den_sum, cnt, has_grade, border, text, n


def composite_full(paths, num=None, den=None, grades=None, invalid=None):
    """원본 해상도 composite. (average, weighted, mask, weighted_mask, border, text, n)"""
    num = SQ_WEIGHTS if num is None else np.asarray(num, np.float32)
    den = WT_FACTORS if den is None else np.asarray(den, np.float32)
    invalid = invalid_mode() if invalid is None else invalid
    ns, ds, cnt, has_grade, border, text, n = accumulate_sums(
        paths, num, den, grades, invalid)
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
                     vmin=None, vmax=None, stops_hex=None, positions=None):
    """원본 해상도 그대로 렌더. 리사이즈 없음 — 화질이 원본과 동일하다.

      chip 안 bin 번호(text) -> **완전히 지운다.** 다른 장의 grade 로 덮이고,
                                전 장이 text 인 픽셀은 grade0 색으로 메운다.
      grade 픽셀(mask)  -> (val-v_min)/(v_max-v_min) 선형 -> 흰색→빨강 LUT
      경계(border)      -> **전부 idx 10 회색 한 색**
                           (원본의 파랑·초록·노랑·자주·주황 마커도 회색으로 통일)
      그 외             -> 원본 palette 색 그대로 (배경 idx8, 투명 idx31)

    stops_hex / positions: UI 의 quantile 색상표. 11색을 0/10/…/100 지점에 놓고
      선형 보간한다. **아래쪽을 연하게 하려면 낮은 quantile 색을 바꾼다**
      (`light_low_stops()`). 값 정규화는 손대지 않으므로 그룹 간 비교가 깨지지 않는다.
    """
    base = Image.open(base_path)
    if base.mode != "P":
        base = base.convert("P", palette=Image.Palette.ADAPTIVE)
    raw = base.getpalette() or []
    pal = np.array((raw + [0] * (768 - len(raw)))[:768], np.uint8).reshape(256, 3)
    idx = np.asarray(base, np.uint8)
    rgb = pal[idx]

    # ★ text 선삭제. base 에 남은 글자/텍스트 마스킹 픽셀은 흰색 처리.
    #   전 장이 text 인 픽셀(mask 밖)도 여기서 흰색이 되어 글자가 남지 않는다.
    t = (idx == TEXT_IDX) if text is None else (text | (idx == TEXT_IDX))
    rgb[t] = 255

    # 요청 기준: 배경도 사라져야 하므로 background/transparent는 흰색으로 통일.
    rgb[idx == BG_IDX] = 255
    rgb[idx == TRANSPARENT_IDX] = 255

    # 경계: 전체 union 을 한 색으로. base 에 없던 줄도 여기서 채워진다.
    rgb[border] = pal[BORDER_IDX]

    v = value_map[mask]
    if v.size:
        lo = float(v.min()) if vmin is None else float(vmin)
        hi = float(v.max()) if vmax is None else float(vmax)
        if hi > lo:
            lut = build_lut(stops_hex, positions=positions)
            sc = np.clip((v - lo) / (hi - lo), 0.0, 1.0)   # min-max 선형 (원본과 동일)
            rgb[mask] = lut[np.rint(sc * 255).astype(np.int32)]
    return Image.fromarray(rgb)


# ── 채택 설정 (260727) — 값은 `deploy/config.py::Composite` 가 정본이다 ──────
#   config 를 못 읽는 상황(단독 실행 등)에서만 아래 값이 쓰인다.
DEFAULT_NUM = UC_NUM
DEFAULT_DEN = uc_den(1.0)
DEFAULT_LIGHT_LOW = 0.5
DEFAULT_LEGENDS = "deploy/color-legends.json"
DEFAULT_SCHEME = "default"


def _cfg():
    """deploy/config.py::Composite. 없으면 None."""
    try:
        from config import Composite as C           # deploy/ 가 sys.path 에 있을 때
        return C
    except Exception:
        try:
            from deploy.config import Composite as C
            return C
        except Exception:
            return None


def default_config_stops():
    """config.Composite 를 우선 쓰고, 없으면 이 파일의 기본값으로 되돌아간다."""
    c = _cfg()
    if c is None:
        return load_stops(DEFAULT_LEGENDS, DEFAULT_SCHEME)
    return load_stops(c.COLOR_LEGENDS, c.COLOR_SCHEME)


def default_config_weights():
    """config 의 METHOD/WT0 -> (분자, 분모)."""
    c = _cfg()
    if c is None:
        return DEFAULT_NUM, DEFAULT_DEN
    if str(getattr(c, "METHOD", "uc")).lower() == "sq":
        return SQ_WEIGHTS, WT_FACTORS
    return UC_NUM, uc_den(float(getattr(c, "WT0", 1.0)))


# ── mapviewer 규격 palette-indexed 출력 (api/composite_map.py:1198-1290) ──
# ★ RGB 배열을 만들지 않는다. **palette index 배열 (H,W) uint8 로 바로 렌더**한다.
#     index 0~23    원본 palette (grade / 경계 / 배경)
#     index 24~255  composite gradient 232 단계
#   이렇게 하면
#     - 6400x6400 RGB(123MB) 중간 배열이 안 생긴다 (index 배열 41MB 만)
#     - 파일이 3.5배 작다 (실측 44.6MB -> 12.8MB)
#     - index 의 의미가 **결정적**이라 파일끼리 비교 가능하고,
#       색만 바꾸려면 PLTE 청크만 갈아끼우면 된다 (재렌더 불필요) — mapviewer UI 가 그렇게 한다
COMPOSITE_GRADIENT_START = 24
COMPOSITE_GRADIENT_END = 255
COMPOSITE_GRADIENT_COUNT = COMPOSITE_GRADIENT_END - COMPOSITE_GRADIENT_START + 1   # 232


def build_gradient_entries(stops_hex=None) -> list[int]:
    """gradient 색을 palette 바이트로 (232*3=696). mapviewer `_build_composite_gradient_entries`."""
    stops = [_hex(c) for c in (stops_hex or COLOR_STOPS_HEX)]
    out: list[int] = []
    n = len(stops) - 1
    for i in range(COMPOSITE_GRADIENT_COUNT):
        f = i / max(COMPOSITE_GRADIENT_COUNT - 1, 1) * n
        lo = max(0, min(n, int(f)))
        hi = min(n, lo + 1)
        t = f - lo
        out.extend(int(a + (b - a) * t) for a, b in zip(stops[lo], stops[hi]))
    return out


def build_sum_map_palette(base_palette: list[int], stops_hex=None) -> list[int]:
    """256*3=768. index 0~23 = 원본 palette, 24~255 = gradient.
    mapviewer `_build_sum_map_palette`."""
    pal = list(base_palette[:768])
    if len(pal) < 768:
        pal.extend([0] * (768 - len(pal)))
    pal[COMPOSITE_GRADIENT_START * 3:] = build_gradient_entries(stops_hex)
    return pal


def render_composite_indexed(value_map, mask, border, base_path, text=None,
                             vmin=None, vmax=None, stops_hex=None):
    """palette index 배열 + palette 를 돌려준다 (RGB 를 만들지 않는다).

    렌더 규칙은 `render_composite` 와 같다:
      chip 안 bin 번호 -> 지운다 / 경계 -> 전부 idx10 / grade -> gradient 24~255
    ⚠ 원본 palette 의 index 24~30 은 gradient 에 덮인다. 우리 데이터는 0~23 + 31 만
      쓰므로 안전하지만, 24~30 을 쓰는 palette 가 오면 경계로 보내야 한다.
    """
    base = Image.open(base_path)
    if base.mode != "P":
        base = base.convert("P", palette=Image.Palette.ADAPTIVE)
    raw = base.getpalette() or []
    idx = np.asarray(base, np.uint8).copy()

    t = (idx == TEXT_IDX) if text is None else (text | (idx == TEXT_IDX))
    idx[t] = 0                                   # bin 번호 -> grade0
    idx[idx == BG_IDX] = 0                       # 배경 -> grade0 색(흰색)
    idx[idx == TRANSPARENT_IDX] = 0
    idx[border] = BORDER_IDX

    v = value_map[mask]
    if v.size:
        lo = float(v.min()) if vmin is None else float(vmin)
        hi = float(v.max()) if vmax is None else float(vmax)
        if hi > lo:
            sc = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
            g = np.rint(sc * (COMPOSITE_GRADIENT_COUNT - 1) + COMPOSITE_GRADIENT_START)
            idx[mask] = np.clip(g, COMPOSITE_GRADIENT_START,
                                COMPOSITE_GRADIENT_END).astype(np.uint8)
        else:
            idx[mask] = COMPOSITE_GRADIENT_START
    return idx, build_sum_map_palette(raw, stops_hex)


def save_palette_png(index_array, palette_list, path):
    """mapviewer `_save_palette_png` 와 동일. compress_level 은 환경변수로."""
    import os
    im = Image.fromarray(np.asarray(index_array, np.uint8), mode="P")
    im.putpalette(list(palette_list)[:768])
    im.save(str(path), format="PNG", optimize=False,
            compress_level=int(os.environ.get("SITE_PNG_COMPRESS", "6")))
    return path


def write_group_composites(paths, out_dir, grades: list[int] | None = None,
                           prefix: str = "", num=None, den=None,
                           base_path=None, stops_hex=None, positions=None,
                           also_square_average: bool = False) -> dict:
    """한 그룹의 square_weighted_average 를 원본 해상도로 쓴다.

    기본은 채택안(method 3 + `color-legends.json` 의 색상표).
    `also_square_average=True` 면 비교용 square_average 도 같이 쓴다.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = list(paths)
    cnum, cden = default_config_weights()
    num = cnum if num is None else num
    den = cden if den is None else den
    stops_hex = default_config_stops() if stops_hex is None else stops_hex
    c = _cfg()
    if c is not None and getattr(c, "ALSO_SQUARE_AVERAGE", False):
        also_square_average = True
    sq, wt, m_sq, m_wt, border, text, n = composite_full(
        paths, num=num, den=den, grades=grades)
    tag = "" if grades is None else "_" + "".join(str(g) for g in sorted(grades))
    base = base_path or paths[0]
    ix, pal = render_composite_indexed(wt, m_wt, border, base, text, stops_hex=stops_hex)
    save_palette_png(ix, pal, out / f"{prefix}square_weighted_average{tag}.png")
    if also_square_average:
        ix2, pal2 = render_composite_indexed(sq, m_sq, border, base, text, stops_hex=stops_hex)
        save_palette_png(ix2, pal2, out / f"{prefix}square_average{tag}.png")
    return {"n": n, "grades": grades, "stops": list(stops_hex),
            "wt_range": [float(wt[m_wt].min()), float(wt[m_wt].max())] if m_wt.any() else None,
            "sq_range": [float(sq[m_sq].min()), float(sq[m_sq].max())] if m_sq.any() else None}
