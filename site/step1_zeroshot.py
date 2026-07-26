#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 1 — 배포 사다리 ①: 여기서 만든 모델로 **학습 없이** 바로 predict.

  python site/step1_zeroshot.py

3-arm 을 같은 다이얼로 비교한다:
    frozen    : backbone 만 (projection head 없음)  <- 기준선
    z0        : 랜덤 head (champion 과 용량 동일)     <- ★ 진짜 대조군
    champion  : 여기서 만든 head 앙상블               <- 사다리 ①

★ 우리 실험에서 사다리 ①은 실패했다(cca 14 source 중 frozen 을 이기는 게 0개).
  그래도 사내에서 반드시 돌려야 하는 이유:
    - 공짜다(학습 0). 혹시 되면 가장 싼 배포 경로다.
    - 안 되더라도 **frozen 기준선과 headroom** 을 그 pool 에서 실측하게 된다.
      그 값이 step2/step3 의 판정 기준이 된다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _site_common import (REPO, banner, check_inputs, deploy_cmd, die, env,  # noqa: E402
                          fmt_row, make_z0_set, read_summary, rel, run,
                          save_result, show_config)


# ═══════════════════════════════════════════════════════════════════════════
class Config:
    """환경변수 우선, 없으면 default. 경로는 전부 프로젝트 루트 기준 상대경로."""

    OUT_ROOT = env("SITE_OUT_ROOT", "runs/site")

    # frozen backbone (필수). 보내드릴 체크포인트 1번.
    BACKBONE = env("SITE_BACKBONE", "weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth")

    # champion projection head — ★ 2개(concat+L2 앙상블)가 배포 기본값이다.
    # 콤마로 구분. 보내드릴 체크포인트 2번/3번.
    CHAMPION_PROJ = env("SITE_CHAMPION_PROJ",
                        "weights/champion/proj_s42_ep20.pt,weights/champion/proj_s1_ep18.pt")

    DEVICE = env("SITE_DEVICE", "cuda")          # cuda | cpu
    BATCH = env("SITE_BATCH", 32)
    REASSIGN = env("SITE_REASSIGN", "nearest_q90")   # none | nearest_q90 | nearest_q80 | assign_all
    Z0_SEED = env("SITE_Z0_SEED", 42)

    # champion 체크포인트가 아직 없으면 True 로 두면 frozen/z0 만 비교한다.
    SKIP_CHAMPION = env("SITE_SKIP_CHAMPION", False)
# ═══════════════════════════════════════════════════════════════════════════


def load_step0(out_root: Path) -> dict:
    p = rel(out_root) / "step0_result.json"
    if not p.exists():
        die("step0 결과가 없다. 먼저 실행해라:  python site/step0_prepare.py")
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    banner("STEP 1", "zero-shot predict (학습 0)", "① 여기서 만든 모델로 바로 predict")
    cfg = show_config(Config)

    s0 = load_step0(Config.OUT_ROOT)
    pool = s0["manifest"]
    mcs, ms = s0["dial"]["mcs"], s0["dial"]["ms"]
    print(f"[step0] pool={pool}  n={s0['n_images']:,}  dial mcs={mcs} ms={ms}\n")

    champ = [p.strip() for p in str(Config.CHAMPION_PROJ).split(",") if p.strip()]
    use_champ = (not Config.SKIP_CHAMPION) and all(rel(p).exists() for p in champ)
    if not use_champ and not Config.SKIP_CHAMPION:
        missing = [p for p in champ if not rel(p).exists()]
        print(f"[warn] champion head 없음 -> frozen/z0 만 비교한다. 없는 파일: {missing}")
        print("       체크포인트를 받으면 다시 돌려라. (site/README.md 참조)\n")

    check_inputs(Config.BACKBONE, s0["image_root"], s0["k_hat"],
                 champ if use_champ else None)

    out_root = rel(Config.OUT_ROOT) / "step1_zeroshot"
    arms: list[tuple[str, list[str] | None]] = [("frozen", None)]
    # ★ z0 는 champion 과 head 수를 맞춘다. 용량이 다르면 비교가 무효다.
    n_heads = len(champ) if use_champ else 1
    arms.append((f"z0_x{n_heads}", make_z0_set(out_root / "_z0", n_heads, int(Config.Z0_SEED))))
    if use_champ:
        arms.append(("champion", champ))

    results = {}
    for name, projs in arms:
        out = out_root / name
        rc = run(deploy_cmd(Config.BACKBONE, pool, out, mcs, ms, projs,
                            Config.DEVICE, int(Config.BATCH), Config.REASSIGN),
                 log_path=out_root / f"{name}.log")
        if rc != 0:
            print(f"[warn] {name} 종료코드 {rc} — summary.json 존재로 판정한다")
        results[name] = read_summary(out)

    print("\n" + "=" * 78)
    print("[결과] 같은 다이얼, 같은 pool, 같은 reassign — 1축 비교")
    print("=" * 78)
    for name, _ in arms:
        print(fmt_row(name, results.get(name)))

    verdict, note = judge(results, arms)
    print("\n" + "-" * 78)
    print(f"[판정] {verdict}")
    for line in note:
        print(f"  {line}")
    print("-" * 78)

    save_result(rel(Config.OUT_ROOT), "step1",
                {"dial": {"mcs": mcs, "ms": ms}, "arms": {k: results.get(k) for k, _ in arms},
                 "verdict": verdict, "notes": note, "config": cfg})
    print("\n다음:  python site/step2_recipe.py")
    print(f"\n[OUT] {out_root}")
    return 0


def judge(results: dict, arms: list) -> tuple[str, list[str]]:
    """label-free 판정. 라벨이 없으므로 seed_noise / k / coherence / stability 로만 본다."""
    fz, notes = results.get("frozen"), []
    if not fz:
        return "판정 불가 (frozen 결과 없음)", ["frozen arm 이 실패했다. 로그를 봐라."]

    def sn(s):
        return s.get("seed_noise_pct", s.get("seed_noise")) if s else None

    fz_sn = sn(fz)
    notes.append(f"frozen 기준선: seed_noise={fz_sn}, k={fz.get('k')} "
                 f"— 이 값이 step2/step3 의 비교 대상이다.")

    z0key = next((k for k, _ in arms if k.startswith("z0")), None)
    z0 = results.get(z0key) if z0key else None
    if z0 and fz_sn is not None and sn(z0) is not None:
        d = sn(z0) - fz_sn
        if d < -2.28:
            notes.append(f"★ 랜덤 head 만으로 frozen 보다 seed_noise 가 {-d:.2f}pp 낮다. "
                         f"이 pool 에서는 '학습 없이 head 만 붙이는 것'도 후보다.")
        elif d > 2.28:
            notes.append(f"랜덤 head 는 frozen 보다 {d:.2f}pp 나쁘다 — 이 pool 에서 랜덤 투영은 해롭다.")
        else:
            notes.append(f"랜덤 head 는 frozen 과 잡음폭 안(Δ{d:+.2f}pp) — 대조군으로 중립.")

    ch = results.get("champion")
    if not ch:
        return "사다리 ① 미평가 (champion head 없음)", notes + [
            "champion 체크포인트를 받아서 다시 돌려라. frozen/z0 기준선은 위에 확보됐다."]

    ch_sn, z_sn = sn(ch), sn(z0) if z0 else None
    beats_frozen = ch_sn is not None and fz_sn is not None and ch_sn < fz_sn - 2.28
    beats_z0 = ch_sn is not None and z_sn is not None and ch_sn < z_sn - 2.28

    if beats_frozen and beats_z0:
        notes.append("champion 이 frozen 과 z0 를 모두 잡음폭 밖으로 이겼다.")
        notes.append("→ 사다리 ①이 이 pool 에서 통한다. 가장 싼 경로다. "
                     "그래도 step2 를 돌려 비교해라(우리 pool 에서는 ①이 실패했었다).")
        return "사다리 ① 성공 후보", notes
    if beats_frozen and not beats_z0:
        notes.append("champion 이 frozen 은 이기지만 랜덤 head(z0)는 못 이긴다.")
        notes.append("→ 개선이 '학습' 때문이 아니라 projection head 의 기하 효과일 수 있다. "
                     "학습 근거로 쓰지 마라.")
        return "사다리 ① 애매 (z0 미돌파)", notes
    notes.append("champion 이 frozen 을 못 이긴다 — 우리 실험(cca 14 source)과 같은 결과다.")
    notes.append("→ 사다리 ①은 쓰지 마라. **frozen 을 기준선으로 삼고 step2(레시피 학습)로 가라.**")
    return "사다리 ① 실패 (예상된 결과)", notes


if __name__ == "__main__":
    raise SystemExit(main())
