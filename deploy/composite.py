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


def _decode_indices(paths, workers: int = 0, prefetch: int = 4):
    """palette index 배열을 **스레드로 미리 디코드**해서 순서대로 흘려준다.

    composite 는 원본 6400x6400 을 그대로 읽어야 해서 (임베딩 캐시는 384 라 못 쓴다)
    디코드가 비용의 대부분이다 — 실측 24장 12.84s 중 대부분. 누적 자체는 numpy 라
    GIL 을 놓으므로 디코드만 앞질러 돌려도 그만큼 줄어든다.
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
        or max(2, min(8, (os.cpu_count() or 4)))
    paths = list(paths)
    with ThreadPoolExecutor(max_workers=w) as ex:
        futs = [ex.submit(one, p) for p in paths[:prefetch]]
        nxt = prefetch
        for k in range(len(paths)):
            a = futs[k].result()
            if nxt < len(paths):
                futs.append(ex.submit(one, paths[nxt])); nxt += 1
            yield a


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
    for a in _decode_indices(paths):
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
    # PNG 압축 레벨. 6400x6400 에서 level6 2.32s/36.5MB vs level1 0.91s/54.4MB —
    # arm 6개 x 그룹 여러 개면 저장만으로 수십 초라 기본을 1 로 둔다.
    import os
    _cl = int(os.environ.get("SITE_PNG_COMPRESS", "1"))
    render_composite(wt, m_wt, border, base, text,
                     stops_hex=stops_hex, positions=positions).save(
        out / f"{prefix}square_weighted_average{tag}.png", compress_level=_cl)
    if also_square_average:
        render_composite(sq, m_sq, border, base, text,
                         stops_hex=stops_hex, positions=positions).save(
            out / f"{prefix}square_average{tag}.png", compress_level=_cl)
    return {"n": n, "grades": grades, "stops": list(stops_hex),
            "wt_range": [float(wt[m_wt].min()), float(wt[m_wt].max())] if m_wt.any() else None,
            "sq_range": [float(sq[m_sq].min()), float(sq[m_sq].max())] if m_sq.any() else None}
