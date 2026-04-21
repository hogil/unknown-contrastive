"""Stage 1: WM-811K 분석 + pixel_rules.yaml 설계.

LSWMD.pkl 을 로드해 class 별 wafer 수 / 크기 / defect 비율 / 공간 분포를
요약하고, contrastive learning 용 palette grade 배정 규칙을 설계한다.

산출물
------
* ``configs/wm811k_analysis.md`` — 분석 보고서 (human-readable).
* ``configs/pixel_rules.yaml``   — stage 2 가 소비하는 기계 파싱 규칙.

Weight 설계
-----------
불량 die (WM-811K 값 2) 를 palette grade 1..7 로 분산시키는 weight:

* 채택: exponential decay ``1 / 2**(k-1)`` — grade 1 이 지배적, grade 7 <1%.
* 근거: 실제 반도체 wafer 에서 grade 7 급 심각 다결함 cluster 는 희귀함.
* 대안: power-law (``1/k^1.5``), linear (``8-k``) — 각각 decay 가 완만/평탄해
  grade 5-7 비중이 과다.

실행
----
    python data_prep/analyze_wm811k.py --pkl data_raw/LSWMD.pkl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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


def _normalize_failure_type(x: Any) -> str:
    if isinstance(x, (list, np.ndarray)):
        arr = np.asarray(x).ravel()
        return "" if arr.size == 0 else str(arr[0])
    return str(x)


def _safe_wafer_stats(wm: Any) -> dict[str, float] | None:
    """Return per-wafer summary stats or None if malformed."""
    try:
        arr = np.asarray(wm)
    except Exception:
        return None
    if arr.ndim != 2 or arr.size == 0:
        return None
    h, w = arr.shape
    normal = int((arr == 1).sum())
    defect = int((arr == 2).sum())
    denom = normal + defect
    if denom == 0:
        defect_ratio = 0.0
    else:
        defect_ratio = defect / denom
    return {
        "h": h,
        "w": w,
        "n_normal": normal,
        "n_defect": defect,
        "defect_ratio": defect_ratio,
    }


def analyze(df: pd.DataFrame, ft_col: str) -> dict[str, Any]:
    """Run per-class analysis on filtered WM-811K DataFrame."""
    per_class: dict[str, dict[str, Any]] = {}
    for cls in CLASSES:
        sub = df[df[ft_col] == cls]
        n_available = len(sub)
        if n_available == 0:
            per_class[cls] = {
                "n_available": 0,
                "sizes": None,
                "defect_ratio": None,
            }
            continue

        heights, widths, ratios = [], [], []
        for _, row in sub.iterrows():
            stats = _safe_wafer_stats(row["waferMap"])
            if stats is None:
                continue
            heights.append(stats["h"])
            widths.append(stats["w"])
            ratios.append(stats["defect_ratio"])

        if heights:
            size_summary = {
                "h_min": int(np.min(heights)),
                "h_max": int(np.max(heights)),
                "h_median": float(np.median(heights)),
                "w_min": int(np.min(widths)),
                "w_max": int(np.max(widths)),
                "w_median": float(np.median(widths)),
                "upscale_factor_median": round(4000 / max(1.0, float(np.median(heights))), 1),
            }
            defect_summary = {
                "mean": float(np.mean(ratios)),
                "median": float(np.median(ratios)),
                "min": float(np.min(ratios)),
                "max": float(np.max(ratios)),
            }
        else:
            size_summary = None
            defect_summary = None

        per_class[cls] = {
            "n_available": n_available,
            "n_valid_2d": len(heights),
            "sizes": size_summary,
            "defect_ratio": defect_summary,
        }
    return per_class


def exponential_weights(n: int = 7) -> list[float]:
    """``1 / 2**(k-1)`` for k in 1..n, rounded to 6 decimal places."""
    return [round(1.0 / (2 ** (k - 1)), 6) for k in range(1, n + 1)]


def power_law_weights(n: int = 7, alpha: float = 1.5) -> list[float]:
    return [round(1.0 / (k ** alpha), 6) for k in range(1, n + 1)]


def linear_weights(n: int = 7) -> list[float]:
    return [float(n + 1 - k) for k in range(1, n + 1)]


def normalize(w: list[float]) -> list[float]:
    s = sum(w)
    return [round(x / s, 6) for x in w]


def pool_check(
    per_class: dict[str, Any],
    splits: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    """Verify each class has enough unique wafers for all splits combined."""
    defect_total = sum(int(s["defect_n_per_class"]) for s in splits.values())
    normal_total = sum(int(s["normal_n"]) for s in splits.values())
    problems = []
    for cls in DEFECT_CLASSES:
        n = per_class[cls]["n_available"]
        problems.append({
            "class": cls,
            "available": n,
            "need": defect_total,
            "ok": n >= defect_total,
        })
    n_none = per_class["none"]["n_available"]
    problems.append({
        "class": "none",
        "available": n_none,
        "need": normal_total,
        "ok": n_none >= normal_total,
    })
    return problems


def build_yaml(
    seed: int = 42,
    size: tuple[int, int] = (4000, 4000),
) -> dict[str, Any]:
    return {
        "version": 1,
        "seed": seed,
        "size": list(size),
        "upscale": "nearest",
        "mapping": {
            "outside": 31,
            "normal": 0,
            "defect": {
                "mode": "random_skewed",
                "grades": [1, 2, 3, 4, 5, 6, 7],
                "weights": exponential_weights(7),
            },
        },
        # Sub-die texture synthesis. NEAREST upscale alone leaves die blocks
        # ~148 px uniform; texture injects pixel-level variation so the
        # 4000x4000 canvas carries meaningful fine-grained content.
        "texture": {
            "mode": "synthetic_scatter",
            "defect_perturb_p": 0.30,
            "normal_scatter_p": 0.03,
            "normal_scatter_grades": [1],
            "global_scatter_n": 0,
            "global_scatter_radius": [2, 6],
        },
        "split": {
            "train": {
                "defect_n_per_class": 50,
                "normal_n": 1600,
                "total": 2000,
            },
            "val": {
                "defect_n_per_class": 20,
                "normal_n": 640,
                "total": 800,
            },
            "test": {
                "defect_n_per_class": 20,
                "normal_n": 640,
                "total": 800,
            },
        },
        "paths": {
            "pkl": "data_raw/LSWMD.pkl",
            "train_out": "data/wm811k_train",
            "val_out": "data/wm811k_val",
            "test_out": "data/wm811k_test",
        },
    }


def render_report(
    per_class: dict[str, Any],
    problems: list[dict[str, Any]],
    yaml_cfg: dict[str, Any],
    pkl_path: Path,
) -> str:
    exp_w = yaml_cfg["mapping"]["defect"]["weights"]
    exp_p = normalize(exp_w)
    pow_w = power_law_weights()
    pow_p = normalize(pow_w)
    lin_w = linear_weights()
    lin_p = normalize(lin_w)

    lines: list[str] = []
    lines.append("# WM-811K 분석 및 pixel 규칙 설계\n")
    lines.append(f"- 원본 pickle: `{pkl_path}` ({pkl_path.stat().st_size / 1e9:.2f} GB)")
    lines.append(f"- 분석 일시: 자동 생성, seed={yaml_cfg['seed']}")
    lines.append(f"- 목표 이미지 크기: {yaml_cfg['size'][0]}×{yaml_cfg['size'][1]} (NEAREST upscale)\n")

    lines.append("## 1. Class 분포 통계\n")
    lines.append("| Class | 사용 가능 | 유효 2D | H median | W median | Upscale factor | Defect 비율 (mean / median / max) |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for cls in CLASSES:
        info = per_class[cls]
        if info["sizes"] is None:
            lines.append(f"| {cls} | {info['n_available']} | 0 | - | - | - | - |")
            continue
        s = info["sizes"]
        r = info["defect_ratio"]
        lines.append(
            f"| {cls} | {info['n_available']} | {info['n_valid_2d']} | "
            f"{s['h_median']:.0f} | {s['w_median']:.0f} | {s['upscale_factor_median']:.1f}× | "
            f"{r['mean']:.3f} / {r['median']:.3f} / {r['max']:.3f} |"
        )
    lines.append("")

    lines.append("## 2. Upscaling 방법: NEAREST\n")
    lines.append(
        "원본 wafer 의 중앙값 크기는 class 별로 다르며(표 참고) 4000×4000 으로 upscale 할 "
        "경우 upscale factor 가 일반적으로 **75× ~ 154×** 범위. Palette PNG 는 각 픽셀이 "
        "`{0..7, 8..13, 31}` 중 하나의 discrete index 를 가지므로 BILINEAR/BICUBIC 등 "
        "interpolation 은 palette 에 없는 중간 값을 생성해 계약을 깨뜨림. 따라서 "
        "`Image.resize(..., Image.NEAREST)` 를 **유일 허용** 방식으로 사용한다.\n"
    )
    lines.append(
        "부산물: die 블록이 정사각형으로 단순 확대되므로 defect 시각화 크기가 class 별로 "
        "다름. fail-map composite pipeline 과 호환.\n"
    )

    lines.append("## 3. Grade 분포 설계\n")
    lines.append(
        "WM-811K 원본 die 값 `{0=outside, 1=normal, 2=defect}` 를 fail-map palette "
        "`{31=transparent, 0=Grade0, 1..7=Grade1..7}` 로 매핑한다.\n"
    )
    lines.append("### 3.1 고정 매핑 (재협상 불가)\n")
    lines.append("- `outside (0) → palette 31`: fail-map transparency 규약 (`common/palette_io.py`).")
    lines.append("- `normal (1) → palette 0`: Grade0 이 정상 die 표준.\n")
    lines.append("### 3.2 불량 → grade 1..7 분포 후보 비교\n")
    lines.append("| Grade | Exponential ``1/2^(k-1)`` | Power-law ``1/k^1.5`` | Linear ``8-k`` |")
    lines.append("|---:|---:|---:|---:|")
    for k in range(7):
        lines.append(
            f"| {k + 1} | {exp_p[k] * 100:.2f}% | {pow_p[k] * 100:.2f}% | {lin_p[k] * 100:.2f}% |"
        )
    lines.append("")
    lines.append("**선택: Exponential decay.** 근거 3 가지:")
    lines.append(
        "1. **현실성** — 실제 wafer 에서 grade 7 (조밀 결함 cluster 내부 die) 는 전체 결함 "
        "의 1% 미만이다. Exponential 은 이 희귀성을 직접 반영한다. Linear 는 grade 7 을 "
        "3.6% 로 부풀려 과다 생성."
    )
    lines.append(
        "2. **학습 안정성** — Contrastive learning 에서 rare grade 를 과다 생성하면 "
        "해당 texture 가 실제 분포와 괴리되어 역효과. Exponential 의 급속 decay 는 grade "
        "5-7 을 소수 guaranteed minority 로 유지."
    )
    lines.append(
        "3. **재현성** — `1/2^(k-1)` 은 명시적 공식, 소수점 6자리 고정. 서로 다른 실행 환경 "
        "에서 동일 weight 재생산 가능."
    )
    lines.append("")
    lines.append(f"최종 weights (소수점 6자리): `{exp_w}`")
    lines.append(f"재정규화 확률 (합=1): `{exp_p}`\n")

    lines.append("## 4. Split 구성 및 풀 충분성\n")
    for split_name, s in yaml_cfg["split"].items():
        lines.append(
            f"- **{split_name:5s}**: defect {s['defect_n_per_class']} × 8 + normal "
            f"{s['normal_n']} = {s['total']} (defect:normal = "
            f"{8 * s['defect_n_per_class']}:{s['normal_n']} "
            f"≈ 1:{s['normal_n'] / max(1, 8 * s['defect_n_per_class']):.1f})"
        )
    lines.append("")
    lines.append(f"| Class | 사용 가능 | 필요 ({'+'.join(yaml_cfg['split'].keys())}) | OK? |")
    lines.append("|---|---:|---:|:---:|")
    for p in problems:
        lines.append(f"| {p['class']} | {p['available']} | {p['need']} | {'OK' if p['ok'] else 'FAIL'} |")
    lines.append("")

    all_ok = all(p["ok"] for p in problems)
    if not all_ok:
        lines.append(
            "> **경고**: 일부 class 의 pool 이 요청 샘플 수보다 적다. 복원추출 우회는 "
            "금지이므로 해당 class 의 `defect_n_per_class` / `normal_n` 을 줄이거나, "
            "다른 데이터 소스를 추가해야 한다.\n"
        )
    else:
        lines.append("> 모든 class 의 pool 이 요청 샘플 수를 충족한다. Stage 2 실행 가능.\n")

    lines.append("## 5. 산출 YAML 요약\n")
    lines.append("```yaml")
    lines.append(yaml.safe_dump(yaml_cfg, sort_keys=False, allow_unicode=True).rstrip())
    lines.append("```\n")

    lines.append("## 6. 금기 재강조\n")
    lines.append("- BILINEAR / BICUBIC upscale 금지 (palette 깨짐).")
    lines.append("- Weight 를 데이터 분석 없이 임의 조정 금지 — 본 문서 근거에 입각.")
    lines.append("- Train / Val source wafer 복원추출 금지.")
    lines.append("- 기존 `configs/pixel_rules.yaml` 덮어쓰기 전 diff 확인 권장.\n")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="data_raw/LSWMD.pkl")
    ap.add_argument("--out-yaml", default="configs/pixel_rules.yaml")
    ap.add_argument("--out-report", default="configs/wm811k_analysis.md")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, default=4000)
    args = ap.parse_args()

    pkl_path = Path(args.pkl)
    if not pkl_path.exists():
        print(f"[err] PKL not found: {pkl_path}", file=sys.stderr)
        return 1

    print(f"[load] {pkl_path} ({pkl_path.stat().st_size / 1e9:.2f} GB) - may take a while")
    df = pd.read_pickle(pkl_path)
    ft_col = "failureType" if "failureType" in df.columns else "faliureType"
    df["_ft"] = df[ft_col].apply(_normalize_failure_type)
    df = df[df["_ft"].isin(CLASSES)]
    print(f"[filter] kept {len(df)} wafers with recognized class labels")

    print("[stats] per-class analysis...")
    per_class = analyze(df, "_ft")

    yaml_cfg = build_yaml(seed=args.seed, size=(args.size, args.size))

    problems = pool_check(per_class, yaml_cfg["split"])

    report = render_report(per_class, problems, yaml_cfg, pkl_path)

    out_yaml = Path(args.out_yaml)
    out_report = Path(args.out_report)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    out_yaml.write_text(
        yaml.safe_dump(yaml_cfg, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    out_report.write_text(report, encoding="utf-8")

    yaml_hash = hashlib.sha256(out_yaml.read_bytes()).hexdigest()[:16]
    print(f"[write] {out_yaml} (sha256 {yaml_hash})")
    print(f"[write] {out_report}")

    all_ok = all(p["ok"] for p in problems)
    if not all_ok:
        bad = [p["class"] for p in problems if not p["ok"]]
        print(f"[warn] pool 부족 class: {bad}", file=sys.stderr)
        return 2
    print("[ok] pool check passed; Stage 2 실행 가능")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
