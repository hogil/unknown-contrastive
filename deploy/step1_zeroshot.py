#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 1 — 배포 사다리 ①: 여기서 만든 모델로 **학습 없이** 바로 predict.

  python deploy/step1_zeroshot.py

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
                          save_result, run_root, live_dial, step_dir, show_config, show_images)


from config import Cluster, Composite, Paths, Runtime  # noqa: E402


class Config:
    """★ 설정은 deploy/config.py 한 곳에서 관리한다. 여기는 참조만."""
    OUT_BASE = Paths.OUT_ROOT          # runs/site (캐시·latest.txt 가 여기)
    OUT_ROOT = Paths.OUT_ROOT          # main() 에서 타임스탬프 run 으로 교체된다
    BACKBONE = Paths.BACKBONE
    DEVICE = Runtime.DEVICE
    CACHE = Runtime.CACHE
    BATCH = Runtime.BATCH
    CHAMPION_PROJ = Paths.CHAMPION_PROJ
    B4_BACKBONE = Paths.B4_BACKBONE
    B4_PROJ = Paths.B4_PROJ
    REASSIGN = Cluster.REASSIGN
    PARTITION_BY = Cluster.PARTITION_BY
    Z0_SEED = 42
    SKIP_CHAMPION = False
    PALETTE_PROBE = Cluster.PALETTE_PROBE
    COMPOSITE_ALL_ARMS = Composite.ALL_ARMS


def load_step0(out_root: Path) -> dict:
    p = rel(out_root) / "step0_result.json"
    if not p.exists():
        die("step0 결과가 없다. 먼저 실행해라:  python deploy/step0_prepare.py")
    return json.loads(p.read_text(encoding="utf-8"))


def pick_best_arm(results: dict, arms: list):
    """composite 를 만들 arm 하나를 고른다 — **라벨 없이** seed_noise 최소.

    k=0(전부 noise)인 arm 은 제외한다. 그런 arm 의 seed_noise 는 0% 가 아니라
    100% 지만, 혹시 축퇴 해가 낮게 나와도 뽑히지 않게 명시적으로 거른다.
    전부 실패면 None -> composite 를 만들지 않는다.
    """
    def sn(s):
        return s.get("seed_noise_pct", s.get("seed_noise")) if s else None
    cand = [(sn(results.get(n)), n, bb, pj) for n, bb, pj in arms
            if results.get(n) and sn(results.get(n)) is not None
            and (results[n].get("k") or 0) > 0]
    if not cand:
        print("[composite] 쓸만한 arm 이 없다 (전부 k=0) -> 생략")
        return None
    cand.sort(key=lambda t: t[0])
    _, n, bb, pj = cand[0]
    return n, bb, pj



def main() -> int:
    banner("STEP 1", "zero-shot predict (학습 0)", "① 여기서 만든 모델로 바로 predict")
    Config.OUT_ROOT = str(run_root(Config.OUT_BASE, create=False))
    cfg = show_config(Config)

    s0 = load_step0(Config.OUT_ROOT)
    pool = s0["manifest"]
    mcs, ms, method, eps = live_dial()             # ★ config 에서 매번 읽는다 (얼리지 않음)
    show_images(s0.get("image_root", ""), pool, getattr(Config, "EXTS", ""))
    # ★ step0 이 만든 디코드 캐시. arm 6개가 각자 다시 디코드하던 걸 없앤다.
    _cache = s0.get("cache_dir", "") if getattr(Config, "CACHE", True) else ""
    if _cache:
        print(f"[cache] {_cache} 재사용 — arm 마다 다시 디코드하지 않는다\n")
    print(f"[pool] {pool}  n={s0['n_images']:,}장   [dial] config 값 그대로 mcs={mcs} ms={ms} "
          f"method={method} eps={eps}\n")

    champ = [p.strip() for p in str(Config.CHAMPION_PROJ).split(",") if p.strip()]
    use_champ = (not Config.SKIP_CHAMPION) and all(rel(p).exists() for p in champ)
    if not use_champ and not Config.SKIP_CHAMPION:
        missing = [p for p in champ if not rel(p).exists()]
        print(f"[warn] champion head 없음 -> frozen/z0 만 비교한다. 없는 파일: {missing}")
        print("       체크포인트를 받으면 다시 돌려라. (deploy/README.md 참조)\n")

    check_inputs(Config.BACKBONE, s0["image_root"],
                 champ if use_champ else None)

    out_root = step_dir(Config.OUT_ROOT, "step1_zeroshot")
    # (arm 이름, backbone, proj 목록)
    arms: list[tuple[str, str, list[str] | None]] = [("frozen", Config.BACKBONE, None)]
    # ★ z0 는 champion 과 head 수를 맞춘다. 용량이 다르면 비교가 무효다.
    n_heads = len(champ) if use_champ else 1
    arms.append((f"z0_x{n_heads}", Config.BACKBONE,
                 make_z0_set(out_root / "_z0", n_heads, int(Config.Z0_SEED))))
    if use_champ:
        arms.append(("champion", Config.BACKBONE, champ))

    # ★ palette 마스킹 효과를 이 pool 에서 직접 잰다.
    #   wafer PNG(mode="P")는 index 0~7 이 결함 grade, 그 위는 border/background 다.
    #   그 색이 clustering shortcut 이 될 수 있어 흰색 처리하는 전처리가 있는데,
    #   champion/b4 head 는 마스킹 **없이** 학습돼서 그 head 에 켜면 불일치가 된다.
    #   그래서 head 없는 frozen arm 으로만 켠 판을 만들어 **효과를 측정**한다.
    if Config.PALETTE_PROBE:
        arms.append(("frozen_masked", Config.BACKBONE, None))

    # ★ May 배포본은 자체 backbone 을 쓰는 독립 arm — FCMAE 와 섞으면 안 된다.
    b4bb, b4pj = rel(Config.B4_BACKBONE), rel(Config.B4_PROJ)
    if b4bb.exists() and b4pj.exists():
        arms.append(("contrastive_b4(May)", Config.B4_BACKBONE, [Config.B4_PROJ]))
        arms.append(("b4_backbone_only", Config.B4_BACKBONE, None))
        print("[info] May 배포본 b4 를 독립 arm 으로 추가 (자체 backbone 사용)")

    # composite 는 원본 6400x6400 을 다시 읽어 arm 당 ~70초를 먹는다(임베딩은 5초).
    # arm 을 고르는 판정에는 안 쓰이므로 기본은 비교 중엔 끄고 **이긴 arm 만** 만든다.
    # 전 arm 다 보고 싶으면 SITE_COMPOSITE_ALL_ARMS=1.
    skip_comp = not Config.COMPOSITE_ALL_ARMS
    if skip_comp:
        print("[composite] 비교 중에는 생략하고 **이긴 arm 만** 만든다 "
              "(arm 당 ~70초). 전 arm 이 필요하면 SITE_COMPOSITE_ALL_ARMS=1\n")

    results = {}
    for name, bb, projs in arms:
        out = out_root / name.replace("(", "_").replace(")", "")
        extra = {"UC_PALETTE_MASK": "1"} if name == "frozen_masked" else {"UC_PALETTE_MASK": "0"}
        rc = run(deploy_cmd(bb, pool, out, mcs, ms, projs,
                            Config.DEVICE, int(Config.BATCH), Config.REASSIGN, _cache,
                            skip_comp, Config.PARTITION_BY),
                 env_extra=extra, log_path=out_root / f"{out.name}.log")
        if rc != 0:
            print(f"[warn] {name} 종료코드 {rc} — summary.json 존재로 판정한다")
        results[name] = read_summary(out)

    print("\n" + "=" * 78)
    print("[결과] 같은 다이얼, 같은 pool, 같은 reassign — 1축 비교")
    print("=" * 78)
    for name, _, _ in arms:
        print(fmt_row(name, results.get(name)))

    verdict, note = judge(results, arms)
    print("\n" + "-" * 78)
    print(f"[판정] {verdict}")
    for line in note:
        print(f"  {line}")
    print("-" * 78)

    # ★ 이긴 arm 하나만 composite 를 만든다 — 6개 다 만드는 시간의 1/6.
    best = None
    if skip_comp:
        best = pick_best_arm(results, arms)
        if best:
            bname, bbb, bprojs = best
            bout = out_root / bname.replace("(", "_").replace(")", "")
            print(f"\n[composite] 이긴 arm '{bname}' 의 composite 를 만든다 "
                  f"(원본 6400x6400, 리사이즈 없음)")
            rc = run(deploy_cmd(bbb, pool, bout, mcs, ms, bprojs,
                                Config.DEVICE, int(Config.BATCH), Config.REASSIGN, _cache,
                                False, Config.PARTITION_BY),
                     env_extra={"UC_PALETTE_MASK": "1" if bname == "frozen_masked" else "0"},
                     log_path=out_root / f"{bout.name}_composite.log")
            if rc != 0:
                print(f"[warn] composite 재실행 종료코드 {rc}")
            print(f"[composite] -> {bout / 'composites'}")

    save_result(rel(Config.OUT_ROOT), "step1",
                {"dial": {"mcs": mcs, "ms": ms},
                 "arms": {k: results.get(k) for k, _, _ in arms},
                 "composite_arm": best[0] if best else ("전 arm" if not skip_comp else None),
                 "verdict": verdict, "notes": note, "config": cfg})
    print("\n다음:  python deploy/step2_recipe.py")
    print(f"\n[OUT] {out_root}")
    return 0


def judge(results: dict, arms: list) -> tuple[str, list[str]]:
    """label-free 판정. 라벨이 없으므로 seed_noise / k / coherence 로만 본다."""
    fz, notes = results.get("frozen"), []
    if not fz:
        return "판정 불가 (frozen 결과 없음)", ["frozen arm 이 실패했다. 로그를 봐라."]

    def sn(s):
        return s.get("seed_noise_pct", s.get("seed_noise")) if s else None

    fz_sn = sn(fz)
    notes.append(f"frozen 기준선: seed_noise={fz_sn}, k={fz.get('k')} "
                 f"— 이 값이 step2/step3 의 비교 대상이다.")

    z0key = next((a[0] for a in arms if a[0].startswith("z0")), None)
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

    b4 = results.get("contrastive_b4(May)")
    if b4 and sn(b4) is not None and ch_sn is not None:
        d = sn(b4) - ch_sn
        if d < -2.28:
            notes.append(f"★ May 배포본 b4 가 champion 보다 seed_noise {-d:.2f}pp 낮다 — b4 를 써라.")
        elif d > 2.28:
            notes.append(f"May 배포본 b4 는 champion 보다 {d:.2f}pp 나쁘다.")
        else:
            notes.append(f"May 배포본 b4 와 champion 은 잡음폭 안(Δ{d:+.2f}pp) — 둘 중 아무거나.")

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
