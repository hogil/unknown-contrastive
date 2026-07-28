#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 3 — 배포 사다리 ③: 여기서 만든 **recipe sweep** 으로 학습 후 predict.

  python deploy/step3_sweep.py

step2 의 고정 레시피를 이 pool 에 맞게 스윕한다. 1축씩만 바꾸는 controlled sweep.

★ 셀 선택은 **라벨 없이** `argmin(seed_noise)` 로 한다.
  실측(severstal 10셀, 다이얼 2개): 전체 순위 상관은 쓸모없지만
  (rho +0.30 / -0.29, 다이얼에 따라 부호가 뒤집힘) **argmin(seed_noise) 는
  두 다이얼 모두에서 정답 셀을 집어냈다.** 배포에 필요한 건 순위가 아니라 승자 하나다.
  단 증거가 약하니(pool 1개) 상위 셀은 반드시 multi-seed 로 재확인해라.

★ 우리 실측 승자: LR 2배(0.004 -> 0.008). base 대비 ARI +62%, seed_noise -12.2pp.
  그래서 LR 축을 맨 앞에 둔다 — 3셀만 돌려도 방향이 보인다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _site_common import (REPO, banner, check_inputs, deploy_cmd, die, env,  # noqa: E402
                          fmt_row, read_summary, rel, run, save_result,
                          run_root, live_dial, show_config, show_images, train_env)
from step2_recipe import rule_c, score_epochs  # noqa: E402


from config import Cluster, Paths, Runtime, Sweep, epochs, recipe  # noqa: E402


class Config:
    """★ 설정은 deploy/config.py 한 곳에서 관리한다. 여기는 참조만."""
    OUT_BASE = Paths.OUT_ROOT          # runs/site (캐시·latest.txt 가 여기)
    OUT_ROOT = Paths.OUT_ROOT          # main() 에서 타임스탬프 run 으로 교체된다
    BACKBONE = Paths.BACKBONE
    DEVICE = Runtime.DEVICE
    CACHE = Runtime.CACHE
    BATCH = Runtime.BATCH
    REASSIGN = Cluster.REASSIGN
    PARTITION_BY = Cluster.PARTITION_BY
    EPOCHS = epochs()
    K_PERCENTILE = Cluster.RULE_C_K_PERCENTILE
    CELLS = Sweep.CELLS
    ROUND1_SEED = Sweep.ROUND1_SEED
    TOP_N = Sweep.TOP_N
    ROUND2_SEEDS = Sweep.ROUND2_SEEDS


CELL_AXES = Sweep.AXES



def cell_recipe(cell: str) -> dict:
    if cell not in CELL_AXES:
        die(f"알 수 없는 셀: {cell}\n  사용 가능: {', '.join(CELL_AXES)}")
    r = recipe()
    r.update({k: v for k, v in CELL_AXES[cell].items() if v is not None})
    if CELL_AXES[cell].get("ignore_neg", "keep") is None:
        r["ignore_neg"] = 0.0     # NEG off
    return r


def load(out_root, step) -> dict:
    p = rel(out_root) / f"{step}_result.json"
    if not p.exists():
        die(f"{step} 결과가 없다. 먼저 실행해라.")
    return json.loads(p.read_text(encoding="utf-8"))


def train_and_score(cell: str, seed: int, pool: str, mcs: int, ms: int,
                    out_root: Path) -> dict | None:
    run_dir = out_root / f"{cell}_s{seed}"
    print(f"\n--- 셀 {cell} (seed {seed}) ---")
    rc = run([sys.executable, "-u", "_may_ablation.py", "B4"],
             env_extra=train_env(Config.BACKBONE, pool, run_dir, seed,
                                 int(Config.EPOCHS), cell_recipe(cell)),
             log_path=run_dir / "train.log")
    if rc != 0:
        print(f"[warn] 종료코드 {rc} — 체크포인트 존재로 판정")
    ck = run_dir / "checkpoints"
    if not ck.exists():
        cands = sorted(run_dir.glob("**/checkpoints"))
        ck = cands[-1] if cands else None
    if ck is None or not ck.exists():
        print(f"[error] {cell} s{seed}: 체크포인트 없음 -> 건너뜀")
        return None
    rows = score_epochs(pool, Config.BACKBONE, ck, mcs, ms, Config.DEVICE, int(Config.BATCH))
    sel = rule_c(rows, float(Config.K_PERCENTILE))
    if not sel:
        return None
    print(f"  ★ Rule C: {sel['selected']['ckpt']} seed_noise={sel['selected']['seed_noise_pct']}%")
    return {"cell": cell, "seed": seed, "ckpt_dir": str(ck.relative_to(REPO)),
            "recipe": cell_recipe(cell), "epochs": rows, **sel}


def main() -> int:
    banner("STEP 3", "recipe sweep + 무라벨 셀 선택", "③ 여기서 만든 recipe sweep 학습 후 predict")
    Config.OUT_ROOT = str(run_root(Config.OUT_BASE, create=False))
    cfg = show_config(Config)

    s0 = load(Config.OUT_ROOT, "step0")
    pool = s0["manifest"]
    mcs, ms, method, eps = live_dial()              # ★ config 에서 매번 읽는다 (얼리지 않음)
    check_inputs(Config.BACKBONE, s0["image_root"])
    show_images(s0["image_root"], pool, getattr(Config, "EXTS", ""))
    _cache = s0.get("cache_dir", "") if getattr(Config, "CACHE", True) else ""   # step0 이 만든 디코드 캐시
    print(f"[pool] {pool}  n={s0['n_images']:,}장   [dial] config 값 그대로 mcs={mcs} ms={ms} "
          f"method={method} eps={eps}")

    cells = [c.strip() for c in str(Config.CELLS).split(",") if c.strip()]
    print(f"[sweep] round-1: {len(cells)} 셀 x 1 seed({Config.ROUND1_SEED}) "
          f"x {Config.EPOCHS} epoch\n        순서: {cells}")

    out_root = rel(Config.OUT_ROOT) / "step3_sweep"
    r1 = {}
    for cell in cells:
        got = train_and_score(cell, int(Config.ROUND1_SEED), pool, mcs, ms, out_root)
        if got:
            r1[cell] = got
        # 중간 저장 — 도중에 끊겨도 진행분이 남는다
        save_result(rel(Config.OUT_ROOT), "step3_partial", {"round1": r1})

    if not r1:
        die("round-1 전 셀 실패. train.log 를 봐라.")

    ranked = sorted(r1.items(), key=lambda kv: kv[1]["selected"]["seed_noise_pct"])
    print("\n" + "=" * 78)
    print("[round-1] 무라벨 순위  (argmin seed_noise — 라벨 전혀 안 씀)")
    print("=" * 78)
    print(f"  {'cell':<10} {'ep':<5} {'seed_noise':<12} {'k':<6}")
    for cell, info in ranked:
        s = info["selected"]
        print(f"  {cell:<10} {s['epoch']:<5} {s['seed_noise_pct']:<12} {s['k']:<6}")
    win = ranked[0][0]
    print(f"\n  ★ 무라벨 승자: {win}  (축: {CELL_AXES[win] or 'base'})")

    # round-2: 상위 N 셀 multi-seed 재확인
    r2 = {}
    top_n = int(Config.TOP_N)
    if top_n > 0:
        seeds2 = [int(s) for s in str(Config.ROUND2_SEEDS).split(",") if s.strip()]
        top = [c for c, _ in ranked[:top_n]]
        print(f"\n[round-2] 상위 {top} 를 seed {seeds2} 로 재확인 "
              f"(1-seed 결과는 잡음폭 안에서 뒤집힐 수 있다)")
        for cell in top:
            for sd in seeds2:
                got = train_and_score(cell, sd, pool, mcs, ms, out_root)
                if got:
                    r2.setdefault(cell, []).append(got)
                save_result(rel(Config.OUT_ROOT), "step3_partial", {"round1": r1, "round2": r2})

    # multi-seed 집계
    agg = {}
    for cell, info in r1.items():
        vals = [info["selected"]["seed_noise_pct"]] + \
               [g["selected"]["seed_noise_pct"] for g in r2.get(cell, [])]
        agg[cell] = {"n_seeds": len(vals), "mean": round(sum(vals) / len(vals), 2),
                     "min": min(vals), "max": max(vals), "values": vals}
    final_rank = sorted(agg.items(), key=lambda kv: kv[1]["mean"])
    final_win = final_rank[0][0]

    print("\n" + "=" * 78)
    print("[최종] multi-seed 집계 (seed_noise, 낮을수록 좋음)")
    print("=" * 78)
    print(f"  {'cell':<10} {'seeds':<7} {'mean':<9} {'min':<9} {'max'}")
    for cell, a in final_rank:
        print(f"  {cell:<10} {a['n_seeds']:<7} {a['mean']:<9} {a['min']:<9} {a['max']}")
    if final_win != win:
        print(f"\n  ⚠ multi-seed 로 승자가 바뀌었다: {win} -> {final_win}. "
              f"1-seed 결과를 믿지 마라.")
    print(f"\n  ★★ 최종 승자: {final_win}   축: {CELL_AXES[final_win] or 'base'}")
    print(f"      레시피: {json.dumps(cell_recipe(final_win), ensure_ascii=False)}")

    # 최종 산출 (승자 셀의 전 seed 앙상블)
    ck_list = [str(Path(r1[final_win]["ckpt_dir"]) / r1[final_win]["selected"]["ckpt"])]
    ck_list += [str(Path(g["ckpt_dir"]) / g["selected"]["ckpt"]) for g in r2.get(final_win, [])]
    out = out_root / "final"
    # ★ 여기는 스윕 셀이 아니라 **최종 산출**이다 -> composite 를 만든다.
    #   (셀 비교 때는 끄는 게 맞지만 final 까지 끄면 그림이 하나도 안 남는다)
    run(deploy_cmd(Config.BACKBONE, pool, out, mcs, ms, ck_list,
                   Config.DEVICE, int(Config.BATCH), Config.REASSIGN, _cache, False,
                   Config.PARTITION_BY),
        log_path=out_root / "final.log")
    final_sum = read_summary(out)

    # step1/step2 대비
    print("\n" + "=" * 78)
    print("[사다리 비교] 같은 다이얼 / 같은 pool / 같은 reassign")
    print("=" * 78)
    s1 = load(Config.OUT_ROOT, "step1")["arms"] if (rel(Config.OUT_ROOT) / "step1_result.json").exists() else {}
    for k, v in s1.items():
        print(fmt_row(f"[①] {k}", v))
    p2 = rel(Config.OUT_ROOT) / "step2_result.json"
    if p2.exists():
        for k, v in json.loads(p2.read_text(encoding="utf-8"))["finals"].items():
            print(fmt_row(f"[②] {k}", v))
    print(fmt_row(f"[③] sweep {final_win}", final_sum))

    save_result(rel(Config.OUT_ROOT), "step3",
                {"dial": {"mcs": mcs, "ms": ms}, "round1": r1, "round2": r2,
                 "aggregate": agg, "label_free_winner_round1": win,
                 "final_winner": final_win, "final_recipe": cell_recipe(final_win),
                 "final_summary": final_sum, "config": cfg})
    print("\n다음:  python deploy/step5_temporal.py   (신규불량 시간축 감지 — 최종 산출물)")
    print("       사다리 ④(TAPT)는 ①~③ 이 부족할 때만:  python deploy/step4_tapt.py")
    print(f"\n[OUT] {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
