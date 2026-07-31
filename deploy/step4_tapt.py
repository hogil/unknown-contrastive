#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 4 — 배포 사다리 ④: **CNN TAPT** backbone 으로 바꾼 뒤 sweep -> predict.

  python deploy/step4_tapt.py

사다리에서 **가장 비싸고 마지막**이다. ①~③ 이 충분하면 여기까지 올 필요 없다.

★ 전제조건: TAPT backbone 이 필요하다. 그건 **라벨이 있어야 만든다**
  (자매 repo `known-cnn` 의 supervised CNN 학습 결과에서 backbone 추출).
  사내엔 라벨이 없으므로 다음 중 하나가 선행돼야 한다:
    (a) 소량이라도 라벨링된 subset 으로 supervised CNN 을 학습 -> backbone 추출
    (b) 우리가 만든 TAPT backbone 을 그대로 이식 (도메인이 다르면 불리)

★ 주의 — 우리 실측에서 TAPT 는 **new-domain 에 불리**했다(260724).
  same-domain(anchor)에서는 TAPT 가 필수였지만, 처음 보는 도메인에서는
  no-TAPT FCMAE 가 TAPT backbone 을 앞섰다. 사내 데이터는 new-domain 에 해당하므로
  **기대치를 낮게 잡고, ③ 결과를 반드시 이겨야만 채택해라.**

이 스크립트는 backbone 만 바꿔서 step3 와 동일한 sweep 을 돌리고, ③ 과 비교한다.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _site_common import (REPO, band, banner, die, env, fmt_row, read_summary, rel,  # noqa: E402
                          save_result, run_root, show_config, show_images, step_dir)


from config import Paths, Sweep  # noqa: E402


class Config:
    """★ 설정은 deploy/config.py 한 곳에서 관리한다. 여기는 참조만."""
    OUT_BASE = Paths.OUT_ROOT          # runs/site (캐시·latest.txt 가 여기)
    OUT_ROOT = Paths.OUT_ROOT          # main() 에서 타임스탬프 run 으로 교체된다
    TAPT_BACKBONE = Paths.TAPT_BACKBONE
    CELLS = "lr008,lr002,base"
    ROUND1_SEED = Sweep.ROUND1_SEED
    TOP_N = 0


GUIDE = """
TAPT backbone 만드는 법 (라벨 필요):

  1. 사내 데이터 일부를 결함 종류별로 라벨링한다 (클래스당 수십 장이면 시작 가능).
  2. 자매 repo `known-cnn` 의 supervised 분류기를 그 라벨로 학습한다.
       python cnn_train_wafer.py --epochs 30 --batch 16 --model-tag site_tapt
  3. best_model.pth 에서 backbone state_dict 만 추출해 아래 경로에 둔다.
       {tapt}
  4. 다시 이 스크립트를 실행한다.

우리 실측 경고 (260723 / 260724):
  - same-domain 에서는 TAPT 가 필수였다 (f26 -> contrastive 35, noise 2%).
  - 그러나 **new-domain 에서는 no-TAPT FCMAE 가 TAPT backbone 을 앞섰다.**
    사내 데이터는 new-domain 이므로 TAPT 가 오히려 불리할 수 있다.
  - 그래서 사다리 ④는 **마지막**이고, ③ 결과를 잡음폭 밖으로 이겨야만 채택한다.
"""


def main() -> int:
    banner("STEP 4", "CNN TAPT backbone + sweep", "④ TAPT 후 학습 sweep 후 predict (마지막 수단)")
    Config.OUT_ROOT = str(run_root(Config.OUT_BASE, create=False))
    cfg = show_config(Config)
    s0p = rel(Config.OUT_ROOT) / "step0_result.json"
    s0 = json.loads(s0p.read_text(encoding="utf-8")) if s0p.exists() else {}
    show_images(s0.get("image_root", ""), s0.get("manifest", ""),
                getattr(Config, "EXTS", ""))

    tapt = rel(Config.TAPT_BACKBONE)
    if not tapt.exists():
        print(f"[skip] TAPT backbone 이 없다: {tapt}")
        print(GUIDE.format(tapt=tapt))
        save_result(rel(Config.OUT_ROOT), "step4",
                    {"status": "skipped", "reason": "no TAPT backbone",
                     "expected_path": str(tapt), "config": cfg})
        print("\n사다리 ①~③ 만으로 운영하려면:  python deploy/step5_temporal.py")
        return 0

    # ★ TAPT 파일이 **FCMAE 와 같은 가중치면** 시험할 게 없다. 실측(260730):
    #   `weights/tapt/backbone_tapt.pth` 가 FCMAE 와 378개 텐서 전부 동일(상대차 0.0000%)
    #   이었는데, step4 는 그걸 모르고 sweep 을 3.8분 다시 돌려 "③과 ④ 동률" 이라는
    #   결론을 냈다. FCMAE 를 FCMAE 와 비교한 것이라 사다리 ④ 판정이 통째로 무의미했다.
    #   여기서 미리 잡아 **헛된 학습과 잘못된 결론을 막는다.**
    try:
        import torch
        import numpy as _np

        def _sd(p):
            d = torch.load(str(p), map_location="cpu", weights_only=False)
            for k in ("state_dict", "model"):
                if isinstance(d, dict) and k in d:
                    d = d[k]
            return d

        _t, _f = _sd(tapt), _sd(rel(Paths.BACKBONE))
        _common = [k for k in _t if k in _f and hasattr(_t[k], "shape")
                   and _t[k].shape == _f[k].shape]
        _n_diff = sum(1 for k in _common
                      if (_t[k].float() - _f[k].float()).abs().mean().item() > 1e-9)
        print(f"[check] TAPT vs FCMAE: 매칭 {len(_common)}개 중 **다른 텐서 {_n_diff}개**")
        if _common and _n_diff == 0:
            print("\n[FATAL] TAPT backbone 이 FCMAE 와 **완전히 동일하다** — TAPT 가 아니다.")
            print(f"        {tapt}")
            print("  이대로 돌리면 FCMAE 를 FCMAE 와 비교해 '③과 ④ 동률' 이라는")
            print("  **가짜 결론**이 나온다 (실측으로 이미 그렇게 나온 적 있다, 260730).")
            print("  진짜 TAPT backbone 으로 교체하고 다시 돌려라 — 자매 repo known-cnn 의")
            print("  supervised CNN 학습 결과에서 backbone 을 추출해야 한다(라벨 필요).")
            save_result(rel(Config.OUT_ROOT), "step4",
                        {"status": "aborted", "reason": "TAPT backbone == FCMAE (not TAPT)",
                         "tapt_backbone": str(tapt), "n_tensors_compared": len(_common),
                         "n_tensors_differing": _n_diff, "config": cfg})
            return 2
    except Exception as _e:
        print(f"[warn] TAPT/FCMAE 동일성 검사를 못 했다 ({type(_e).__name__}: {_e}) — 계속한다")

    s3p = rel(Config.OUT_ROOT) / "step3_result.json"
    if not s3p.exists():
        die("step3 결과가 없다. ④는 ③을 이겨야 의미가 있으므로 먼저 돌려라:\n"
            "  python deploy/step3_sweep.py")
    s3 = json.loads(s3p.read_text(encoding="utf-8"))

    # step3 를 그대로 재사용하되 backbone/출력만 바꾼다 (코드 중복 방지)
    e = dict(os.environ)
    e.update({
        "PYTHONIOENCODING": "utf-8",
        # ★ 필수 — 이게 없으면 아래 SITE_* 가 **전부 무시된다**. config.py 가 권위라서
        #   env 는 SITE_ENV_OVERRIDE=1 일 때만 이긴다(_site_common.env). 예전엔 이게
        #   빠져 있어서 step4 가 TAPT 가 아니라 **FCMAE 로 step3 를 통째로 재실행**하고,
        #   OUT_ROOT 도 안 바뀌어 **진짜 step3_result.json 을 덮어썼다** (260729 감사).
        "SITE_ENV_OVERRIDE": "1",
        "SITE_BACKBONE": str(tapt),
        # ★ 재실행하면 새 폴더. 고정 이름이면 앞 TAPT sweep 결과를 덮어썼다 (260730).
        "SITE_OUT_ROOT": str(step_dir(Config.OUT_ROOT, "step4_tapt_work")),
        "SITE_CELLS": str(Config.CELLS),
        "SITE_ROUND1_SEED": str(Config.ROUND1_SEED),
        "SITE_TOP_N": str(Config.TOP_N),
    })
    # step0 결과를 TAPT 작업 폴더로 복사 (pool/다이얼 동일하게 유지)
    work = Path(e["SITE_OUT_ROOT"])
    work.mkdir(parents=True, exist_ok=True)
    src0 = rel(Config.OUT_ROOT) / "step0_result.json"
    (work / "step0_result.json").write_text(src0.read_text(encoding="utf-8"), encoding="utf-8")
    # 서브프로세스의 run_root() 가 work 자체를 run 폴더로 보게 한다
    # (없으면 work 아래 latest.txt 를 찾다가 죽는다)
    (work / "latest.txt").write_text(str(work), encoding="utf-8")

    print(f"[run] step3_sweep 를 TAPT backbone 으로 재실행\n      backbone={tapt}\n      work={work}")
    rc = subprocess.call([sys.executable, "deploy/step3_sweep.py"], cwd=str(REPO), env=e)
    if rc != 0:
        print(f"[warn] 종료코드 {rc}")

    s4p = work / "step3_result.json"
    s4 = json.loads(s4p.read_text(encoding="utf-8")) if s4p.exists() else {}
    a, b = s3.get("final_summary"), s4.get("final_summary")

    print("\n" + "=" * 78)
    print("[비교] ③ FCMAE sweep  vs  ④ TAPT sweep")
    print("=" * 78)
    print(fmt_row(f"[③] {s3.get('final_winner')}", a))
    print(fmt_row(f"[④] {s4.get('final_winner')}", b))

    _B = band("noise")          # ★ config.Judge.BAND 에서 live 로 읽는다
    sn = lambda s: (s or {}).get("seed_noise_pct", (s or {}).get("seed_noise"))
    notes = []
    if sn(a) is None or sn(b) is None:
        notes.append("비교 불가 (한쪽 결과 없음)")
    else:
        d = sn(a) - sn(b)
        if d > _B:
            notes.append(f"TAPT 가 {d:.2f}pp 우세 (잡음폭 밖) -> ④ 채택 검토")
        elif d < -_B:
            notes.append(f"TAPT 가 {-d:.2f}pp 열세 -> ④ 기각. ③을 쓴다. "
                         f"(new-domain 에서 TAPT 가 불리하다는 260724 관측과 일치)")
        else:
            notes.append(f"Δ{d:+.2f}pp — 잡음폭 안. **더 싼 ③을 쓴다.** "
                         f"사다리는 낮은 순위가 우선이다.")
    for n in notes:
        print(f"\n  {n}")

    save_result(rel(Config.OUT_ROOT), "step4",
                {"status": "done", "tapt_backbone": str(tapt),
                 "tier3": {"winner": s3.get("final_winner"), "summary": a},
                 "tier4": {"winner": s4.get("final_winner"), "summary": b},
                 "notes": notes, "config": cfg})
    print("\n다음:  python deploy/step5_temporal.py")
    print(f"\n[OUT] {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
