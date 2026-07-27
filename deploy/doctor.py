#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""환경 진단 — import 오류가 날 때 먼저 이걸 돌려라.

  python deploy/doctor.py

`ImportError: cannot import name 'Image' from 'PIL' (unknown location)` 같은
오류는 대부분 **sys.path 앞쪽에 가짜 패키지 폴더**가 있어서 생긴다.
`unknown location` = 그 패키지가 `__init__.py` 없는 빈 디렉토리(namespace package)로
잡혔다는 뜻이다. 이 스크립트가 어느 경로가 범인인지 짚어준다.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

OK, BAD, WARN = "  [OK]  ", "  [!!]  ", "  [??]  "


def line(t=""):
    print(t, flush=True)


def main() -> int:
    line("=" * 74)
    line("환경 진단")
    line("=" * 74)
    line(f"  python      : {sys.executable}")
    line(f"  version     : {sys.version.split()[0]}")
    line(f"  cwd         : {os.getcwd()}")
    line(f"  PYTHONPATH  : {os.environ.get('PYTHONPATH', '(없음)')}")
    line()

    # ── stdlib site 가 가려졌나 ──────────────────────────────────────────
    import site as _site
    loc = getattr(_site, "__file__", None)
    if loc:
        line(f"{OK}stdlib site : {loc}")
    else:
        line(f"{BAD}stdlib site 가 namespace 로 잡혔다: {list(getattr(_site, '__path__', []))}")
        line("       -> site-packages 가 등록되지 않아 모든 외부 패키지가 깨진다.")
        line("       -> sys.path 앞쪽의 'site' 디렉토리를 치워라.")

    n_sp = sum(("site-packages" in p) or ("dist-packages" in p) for p in sys.path)
    line((OK if n_sp else BAD) + f"site-packages 경로 {n_sp} 개")
    if not n_sp:
        line("       -> venv/conda 가 활성화되지 않았거나 site 모듈이 가려졌다.")
    line()

    # ── 문제 패키지들 ────────────────────────────────────────────────────
    bad = []
    for name in ("PIL", "PIL.Image", "numpy", "torch", "torchvision", "timm",
                 "hdbscan", "sklearn"):
        try:
            spec = importlib.util.find_spec(name)
        except Exception as e:
            line(f"{BAD}{name:12s} find_spec 실패: {e}")
            bad.append(name)
            continue
        if spec is None:
            line(f"{BAD}{name:12s} 설치 안 됨")
            bad.append(name)
            continue
        if spec.origin in (None, "namespace"):
            line(f"{BAD}{name:12s} namespace package (빈 폴더) -> {list(spec.submodule_search_locations or [])}")
            bad.append(name)
            continue
        line(f"{OK}{name:12s} {spec.origin}")
    line()

    # ── sys.path 안에 가짜 패키지 폴더가 있나 ────────────────────────────
    line("sys.path 검사 (가짜 패키지 폴더 탐지):")
    suspects = ("PIL", "numpy", "torch", "torchvision", "site", "timm")
    found = False
    for i, entry in enumerate(sys.path):
        if not entry:
            entry = os.getcwd()
        d = Path(entry)
        if not d.is_dir():
            continue
        for s in suspects:
            cand = d / s
            if cand.is_dir() and not (cand / "__init__.py").exists():
                if "site-packages" in str(cand) or "dist-packages" in str(cand):
                    continue          # 정상 namespace 배포도 있으니 제외
                line(f"{BAD}sys.path[{i}] 에 가짜 '{s}' 폴더: {cand}")
                line(f"       -> 이걸 치우거나 이름을 바꿔라. 이게 import 를 가로챈다.")
                found = True
    if not found:
        line(f"{OK}sys.path 에 가짜 패키지 폴더 없음")
    line()

    # ── 실제 import ─────────────────────────────────────────────────────
    line("실제 import:")
    try:
        from PIL import Image  # noqa: F401
        import PIL
        line(f"{OK}from PIL import Image  (Pillow {getattr(PIL, '__version__', '?')})")
    except Exception as e:
        line(f"{BAD}from PIL import Image -> {type(e).__name__}: {e}")
        line("       조치: pip install -U --force-reinstall Pillow")
        bad.append("PIL")
    try:
        import torch
        line(f"{OK}torch {torch.__version__}  cuda={torch.cuda.is_available()}"
             + (f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    except Exception as e:
        line(f"{BAD}torch -> {type(e).__name__}: {e}")
        bad.append("torch")
    try:
        import torchvision
        line(f"{OK}torchvision {torchvision.__version__}")
    except Exception as e:
        line(f"{BAD}torchvision -> {type(e).__name__}: {e}")
        bad.append("torchvision")
    line()

    line("=" * 74)
    if bad:
        line(f"  문제 {len(bad)} 건: {', '.join(sorted(set(bad)))}")
        line("  위 [!!] 줄의 조치를 먼저 하고 다시 돌려라.")
        return 1
    line("  이상 없음 — deploy/step0_prepare.py 부터 진행해도 된다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
