"""WM-811K (LSWMD.pkl) → fail-map 정식 포맷 palette PNG 변환.

fail-map (``D:/project/fail-map``) 의 파이프라인 산출물과 형식적으로 동일한
이미지를 생성한다:

* **Palette**: fail-map ``color-legends.json`` 공식 32색
  (``common.palette_io.get_failmap_palette``). Grade0=#FFFFFF, Grade7=#000000,
  invalid=31=white 등 fail-map 과 바이트 단위로 일치.
* **Grade 0..7 분포**: WM-811K die 값 ``{0=바깥, 1=정상, 2=불량}`` 를
  fail-map 처럼 8단계로 다양화 — **defect-neighbor density** 기반 결정적 규칙으로
  grade 3..7 배정. 조밀한 결함 → grade 7, 고립 결함 → grade 3.
* **파일명**: ``{CLASS}_00P_W{idx:03d}_{YYYYMMDD}_{HHMMSS}.png`` —
  ``contrastive.py::parse_fields_from_filename`` 가 요구하는
  ``root_step_wafer_ymd_hms`` 5-field 규약을 만족.
* **Transparency**: 바깥(index 31) 은 ``transparency=31`` tRNS 마킹.
* **Mode**: ``'P'`` palette-indexed PNG.

WM-811K 원본은 die 당 0/1/2 이진 정보만 제공하므로 grade 1..6 의 세밀한
심각도 계층은 합성적이다. 다만 "정상=grade 0, 결함=grade 3..7 (조밀도 순)"
규칙은 결정적 + 이웃 결함 수만으로 계산되므로 재현 가능.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Make the project root importable when running this file directly.
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


def _class_token(cls: str) -> str:
    """fail-map 파일명의 root field 는 대문자 알파벳 토큰. ``_``/``-`` 제거."""
    return cls.replace("-", "").replace("_", "").upper()


def assign_grades_by_neighbor_density(wm: np.ndarray) -> np.ndarray:
    """WM-811K 2D map (values 0/1/2) → fail-map palette index 2D map.

    Mapping
    -------
    * ``0`` (outside wafer) → 31 (transparent)
    * ``1`` (normal die)    → 0  (Grade0, 흰색)
    * ``2`` (defect die)    → 3..7 (이웃 결함 개수 기반)

    3x3 kernel 로 센 이웃 결함 수 :math:`n \\in [0..8]` (center 제외) → grade:
      - 0      → 3  (isolated defect, e.g. Random)
      - 1-2    → 4
      - 3-4    → 5
      - 5-6    → 6
      - 7-8    → 7  (dense cluster, e.g. Center/Edge-Ring 내부)

    Deterministic (no randomness). 같은 waferMap 입력 → 같은 출력.
    """
    wm = np.asarray(wm)
    if wm.ndim != 2:
        raise ValueError(f"waferMap must be 2D, got shape {wm.shape}")

    defect = (wm == 2).astype(np.int32)

    # 3x3 neighbor sum via zero-padded shift-accumulate (center 제외).
    padded = np.pad(defect, 1, mode="constant", constant_values=0)
    nc = np.zeros_like(defect, dtype=np.int32)
    h, w = defect.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            nc += padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]

    # density → grade 3..7.  (nc+1)//2 → 0→0, 1→1, 2→1, 3→2, ..., 8→4
    grade_def = 3 + np.clip((nc + 1) // 2, 0, 4).astype(np.uint8)

    out = np.full(wm.shape, 31, dtype=np.uint8)
    out[wm == 1] = 0
    mask_def = wm == 2
    out[mask_def] = grade_def[mask_def]
    return out


# Back-compat name so existing tests/callers that import ``wm_to_palette``
# continue to work. 새 pipeline 은 density-based grade 할당을 사용.
def wm_to_palette(wm: np.ndarray) -> np.ndarray:
    """Alias of :func:`assign_grades_by_neighbor_density` (fail-map format)."""
    return assign_grades_by_neighbor_density(wm)


def save_palette_png(
    idx_array: np.ndarray,
    path: Path,
    palette: list[int],
    target_size: tuple[int, int] | None = None,
) -> None:
    """Save ``idx_array`` as palette PNG. If ``target_size=(W, H)`` is given,
    NEAREST-resize so all wafers land on a uniform canvas (index values
    preserved exactly — no interpolation of grades)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(idx_array, mode="P")
    img.putpalette(palette)
    if target_size is not None:
        img = img.resize(target_size, Image.NEAREST)
    img.save(path, transparency=31, optimize=True)


def _normalize_failure_type(x) -> str:
    if isinstance(x, (list, np.ndarray)):
        x = np.asarray(x).ravel()
        if x.size == 0:
            return ""
        return str(x[0])
    return str(x)


def _filename_for(cls: str, wafer_idx: int, base_dt: datetime) -> str:
    """fail-map 5-field 규약 ``{ROOT}_{STEP}_{WAFER}_{YYYYMMDD}_{HHMMSS}.png``.

    예: ``CENTER_00P_W001_20260420_000001.png``
    """
    tok = _class_token(cls)
    t = base_dt + timedelta(seconds=wafer_idx + 1)
    ymd = t.strftime("%Y%m%d")
    hms = t.strftime("%H%M%S")
    return f"{tok}_00P_W{wafer_idx:03d}_{ymd}_{hms}.png"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data_raw/LSWMD.pkl")
    ap.add_argument("--out", default="data/wm811k")
    ap.add_argument("--n-per-class", type=int, default=222,
                    help="각 실패 유형에서 뽑을 장수 (부족하면 있는 만큼)")
    ap.add_argument("--seed", type=int, default=42,
                    help="wafer 선택 시드. grade 할당 자체는 deterministic.")
    ap.add_argument("--base-date", default="20260420",
                    help="합성 파일명 timestamp 의 YYYYMMDD (fail-map 규약용)")
    ap.add_argument("--target-size", nargs=2, type=int, default=None,
                    metavar=("W", "H"),
                    help="모든 wafer 를 이 크기로 NEAREST resize. "
                         "미지정시 WM-811K 원본 크기 유지 (가변).")
    args = ap.parse_args()

    pkl_path = Path(args.pkl)
    if not pkl_path.exists():
        print(f"[err] PKL not found: {pkl_path}\n"
              f"       → run `python data_prep/download_wm811k.py` first",
              file=sys.stderr)
        return 1

    print(f"[load] {pkl_path} ({pkl_path.stat().st_size / 1e6:.1f} MB)")
    df = pd.read_pickle(pkl_path)

    ft_col = "failureType" if "failureType" in df.columns else "faliureType"
    df["_ft"] = df[ft_col].apply(_normalize_failure_type)
    df = df[df["_ft"].isin(CLASSES)]

    rng = np.random.default_rng(args.seed)
    out_root = Path(args.out)
    base_dt = datetime.strptime(args.base_date + "_000000", "%Y%m%d_%H%M%S")

    palette = get_failmap_palette()
    target_size = tuple(args.target_size) if args.target_size else None
    if target_size is not None:
        print(f"[target-size] uniform canvas {target_size[0]}x{target_size[1]} (NEAREST resize)")
    summary: dict[str, int] = {}

    for cls in CLASSES:
        sub = df[df["_ft"] == cls]
        n_have = len(sub)
        if n_have == 0:
            print(f"[WARN] {cls}: 0 samples available")
            summary[cls] = 0
            continue
        take = min(args.n_per_class, n_have)
        sel = rng.choice(n_have, size=take, replace=False)
        subset = sub.iloc[sel]

        saved = 0
        for i, (_, row) in enumerate(subset.iterrows()):
            try:
                idx_arr = assign_grades_by_neighbor_density(row["waferMap"])
                fname = _filename_for(cls, i, base_dt)
                save_palette_png(
                    idx_arr,
                    out_root / cls / fname,
                    palette,
                    target_size=target_size,
                )
                saved += 1
            except Exception as e:
                print(f"[skip] {cls}/{i}: {e}")
        summary[cls] = saved
        print(f"[{cls:10s}] saved {saved}/{take} (available {n_have})")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    total = sum(summary.values())
    classes_nonempty = sum(1 for v in summary.values() if v > 0)
    print(f"\n[done] total={total} images across {classes_nonempty} classes")
    print(f"[out]  {out_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
