#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""아무 step 이나 **다른 backbone 으로** 격리해서 돌린다 (기존 결과를 안 건드린다).

  python deploy/run_with_backbone.py --step 2 --backbone weights/tapt/backbone_tapt.pth

★ 왜 필요한가
  step2/3/5 는 `Paths.BACKBONE`(기본 FCMAE)을 쓴다. 환경변수로 바꿔서 그냥 돌리면
  같은 run 폴더의 `stepN_result.json` 을 **덮어써서** 원래 backbone 결과가 사라진다.
  그러면 "FCMAE 로 한 것"과 "TAPT 로 한 것"을 나란히 못 본다.
  step4 가 step3 를 TAPT 로 돌릴 때 쓰는 격리 방식(작업 폴더를 따로 만들고 step0 결과를
  복사해 물려주는 것)을 그대로 일반화한 것이다.

★ 무엇을 격리하나
  <run>/backbone_<이름>_<시각>/  아래에 step0_result.json 과 latest.txt 를 넣어서
  서브프로세스의 run_root() 가 그 폴더를 run 으로 보게 한다. pool·다이얼·캐시는
  step0 것을 그대로 물려받으므로 **1축(backbone)만 다른 비교**가 된다.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _site_common import REPO, banner, die, rel, run_root, step_dir  # noqa: E402

from config import Paths  # noqa: E402

STEPS = {"1": "step1_zeroshot", "2": "step2_recipe", "3": "step3_sweep", "5": "step5_temporal"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # ★ 기본값을 둔다 — 가장 흔한 용도가 "step2 를 TAPT 로" 라서, 인자 없이 그냥
    #   돌아가는 게 맞다. 무엇으로 정해졌는지는 아래에서 반드시 찍는다.
    ap.add_argument("--step", default="2", choices=sorted(STEPS),
                    help="돌릴 step 번호 (기본 2. 4 는 이미 TAPT 전용이라 제외)")
    ap.add_argument("--backbone", default=Paths.TAPT_BACKBONE,
                    help=f"쓸 backbone .pth (기본 {Paths.TAPT_BACKBONE})")
    ap.add_argument("--tag", default="", help="작업 폴더 이름에 쓸 꼬리표 (기본: backbone 파일명)")
    a = ap.parse_args()

    script = STEPS[a.step]
    bb = rel(a.backbone)
    print(f"[args] --step {a.step} ({STEPS[a.step]})   --backbone {a.backbone}"
          + ("   (둘 다 기본값)" if a.step == "2" and a.backbone == Paths.TAPT_BACKBONE else ""))
    if not bb.exists():
        die(f"backbone 이 없다: {bb}\n"
            "  TAPT 를 쓰려면 먼저 만들어라:\n"
            "    python deploy/extract_tapt.py --src <known-cnn 의 best_model.pth>\n"
            "  다른 backbone 을 쓰려면:  --backbone <경로>")

    banner(f"STEP {a.step} @ 다른 backbone", f"{bb.name} 으로 격리 실행")

    # ★ FCMAE 와 같은 파일이면 격리해서 돌릴 이유가 없다. step4 가 가짜 TAPT 로
    #   sweep 을 통째로 헛돌린 전례(260730)와 같은 사고를 여기서도 막는다.
    try:
        import torch

        def _sd(p):
            d = torch.load(str(p), map_location="cpu", weights_only=False)
            for k in ("state_dict", "model"):
                if isinstance(d, dict) and k in d:
                    d = d[k]
            return d

        _t, _f = _sd(bb), _sd(rel(Paths.BACKBONE))
        _c = [k for k in _t if k in _f and hasattr(_t[k], "shape") and _t[k].shape == _f[k].shape]
        _n = sum(1 for k in _c if (_t[k].float() - _f[k].float()).abs().mean().item() > 1e-9)
        print(f"[check] 기본 backbone 대비: 매칭 {len(_c)}개 중 다른 텐서 {_n}개")
        if _c and _n == 0:
            die(f"이 backbone 은 기본값({Paths.BACKBONE})과 **완전히 동일**하다.\n"
                f"  격리 실행할 이유가 없다 — 같은 것을 같은 것과 비교하게 된다.")
    except SystemExit:
        raise
    except Exception as e:
        print(f"[warn] 동일성 검사를 못 했다 ({type(e).__name__}: {e}) — 계속한다")

    run = run_root(Paths.OUT_ROOT, create=False)
    tag = a.tag or bb.stem
    work = step_dir(run, f"backbone_{tag}")

    src0 = rel(run) / "step0_result.json"
    if not src0.exists():
        die(f"step0 결과가 없다: {src0}\n  먼저:  python deploy/step0_prepare.py")
    (work / "step0_result.json").write_text(src0.read_text(encoding="utf-8"), encoding="utf-8")
    # 서브프로세스의 run_root() 가 work 자체를 run 으로 보게 한다
    (work / "latest.txt").write_text(str(work), encoding="utf-8")
    # step2 는 step1 기준선과 비교한다 — 있으면 같이 물려준다 (없으면 비교만 생략된다)
    for extra in ("step1_result.json", "step2_result.json"):
        p = rel(run) / extra
        if p.exists():
            (work / extra).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

    e = dict(os.environ)
    e.update({"PYTHONIOENCODING": "utf-8",
              "SITE_ENV_OVERRIDE": "1",          # ★ 없으면 아래 SITE_* 가 전부 무시된다
              "SITE_BACKBONE": str(bb),
              "SITE_OUT_ROOT": str(work)})
    print(f"[run] {script}.py\n      backbone={bb}\n      work={work}")
    rc = subprocess.call([sys.executable, "-u", f"deploy/{script}.py"], cwd=str(REPO), env=e)
    if rc != 0:
        print(f"[warn] 종료코드 {rc}")

    print(f"\n원래 backbone 결과와 나란히 보려면:")
    print(f"  {rel(run) / f'step{a.step}_result.json'}          (기본 backbone)")
    print(f"  {work / f'step{a.step}_result.json'}   ({bb.name})")
    print(f"\n[OUT] {work}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
