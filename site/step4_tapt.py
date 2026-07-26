#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 4 — 배포 사다리 ④: **CNN TAPT** backbone 으로 바꾼 뒤 sweep -> predict.

  python site/step4_tapt.py

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
from _site_common import (REPO, banner, die, env, fmt_row, read_summary, rel,  # noqa: E402
                          save_result, show_config)


# ═══════════════════════════════════════════════════════════════════════════
class Config:
    """환경변수 우선, 없으면 default. 경로는 전부 프로젝트 루트 기준 상대경로."""

    OUT_ROOT = env("SITE_OUT_ROOT", "runs/site")

    # ★ TAPT backbone. 이게 없으면 이 step 은 아무것도 안 하고 안내만 출력한다.
    TAPT_BACKBONE = env("SITE_TAPT_BACKBONE", "weights/tapt/backbone_tapt.pth")

    # step3 와 동일한 셀 목록을 쓰되, 비용이 크므로 기본은 상위 축만.
    CELLS = env("SITE_TAPT_CELLS", "lr008,lr002,base")
    ROUND1_SEED = env("SITE_ROUND1_SEED", 42)
    TOP_N = env("SITE_TAPT_TOP_N", 0)     # 기본 0 = round-2 생략 (비용)
# ═══════════════════════════════════════════════════════════════════════════


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
    cfg = show_config(Config)

    tapt = rel(Config.TAPT_BACKBONE)
    if not tapt.exists():
        print(f"[skip] TAPT backbone 이 없다: {tapt}")
        print(GUIDE.format(tapt=tapt))
        save_result(rel(Config.OUT_ROOT), "step4",
                    {"status": "skipped", "reason": "no TAPT backbone",
                     "expected_path": str(tapt), "config": cfg})
        print("\n사다리 ①~③ 만으로 운영하려면:  python site/step5_temporal.py")
        return 0

    s3p = rel(Config.OUT_ROOT) / "step3_result.json"
    if not s3p.exists():
        die("step3 결과가 없다. ④는 ③을 이겨야 의미가 있으므로 먼저 돌려라:\n"
            "  python site/step3_sweep.py")
    s3 = json.loads(s3p.read_text(encoding="utf-8"))

    # step3 를 그대로 재사용하되 backbone/출력만 바꾼다 (코드 중복 방지)
    e = dict(os.environ)
    e.update({
        "PYTHONIOENCODING": "utf-8",
        "SITE_BACKBONE": str(tapt),
        "SITE_OUT_ROOT": str(rel(Config.OUT_ROOT) / "step4_tapt_work"),
        "SITE_CELLS": str(Config.CELLS),
        "SITE_ROUND1_SEED": str(Config.ROUND1_SEED),
        "SITE_TOP_N": str(Config.TOP_N),
    })
    # step0 결과를 TAPT 작업 폴더로 복사 (pool/다이얼 동일하게 유지)
    work = Path(e["SITE_OUT_ROOT"])
    work.mkdir(parents=True, exist_ok=True)
    src0 = rel(Config.OUT_ROOT) / "step0_result.json"
    (work / "step0_result.json").write_text(src0.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"[run] step3_sweep 를 TAPT backbone 으로 재실행\n      backbone={tapt}\n      work={work}")
    rc = subprocess.call([sys.executable, "site/step3_sweep.py"], cwd=str(REPO), env=e)
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

    sn = lambda s: (s or {}).get("seed_noise_pct", (s or {}).get("seed_noise"))
    notes = []
    if sn(a) is None or sn(b) is None:
        notes.append("비교 불가 (한쪽 결과 없음)")
    else:
        d = sn(a) - sn(b)
        if d > 2.28:
            notes.append(f"TAPT 가 {d:.2f}pp 우세 (잡음폭 밖) -> ④ 채택 검토")
        elif d < -2.28:
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
    print("\n다음:  python site/step5_temporal.py")
    print(f"\n[OUT] {work}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
