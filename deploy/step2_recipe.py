#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 2 — 배포 사다리 ②: 여기서 만든 **레시피로 학습** 후 predict.

  python deploy/step2_recipe.py

절차:
  1. B4 recipe 로 사내 pool 에 contrastive 적응 (seed 여러 개)
  2. ★ Rule C 로 **라벨 없이** epoch 선택
       gate 통과 -> k >= 자기 run 의 75th 퍼센타일 유지 -> argmin(seed_noise)
       (severstal 외부검증 통과: 라벨 없이 고른 ep17 이 오라클 ep12 와 0.25pp 차)
  3. 선택된 epoch 으로 최종 grouping 산출 (대표 이미지 + composite 포함)
  4. step1 의 frozen / z0 기준선과 비교

배경 정보를 한 번만 계산하고(backbone forward) 모든 epoch head 에 재사용하므로
epoch 수가 많아도 채점 비용은 거의 안 늘어난다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _site_common import (add_path, REPO, band, banner, baseline_arms, check_inputs, deploy_cmd, die, env,  # noqa: E402
                          fmt_row, read_summary, rel, run, save_result,
                          run_root, live_dial, step_dir, show_config, show_images, train_env)


from config import Cluster, Paths, Recipe, Runtime, epochs, recipe  # noqa: E402


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
    SEEDS = Recipe.SEEDS
    EPOCHS = epochs()
    K_PERCENTILE = Cluster.RULE_C_K_PERCENTILE
    SKIP_TRAIN = False



def load_step0(out_root) -> dict:
    p = rel(out_root) / "step0_result.json"
    if not p.exists():
        die("step0 결과가 없다. 먼저:  python deploy/step0_prepare.py")
    return json.loads(p.read_text(encoding="utf-8"))


def score_epochs(pool: str, backbone: str, ckpt_dir: Path, mcs: int, ms: int,
                 device: str, batch: int, cache_dir: str = "",
                 method: str = "", eps=None) -> list[dict]:
    """backbone feature 1회 계산 후 epoch head 별 label-free 지표.

    라벨을 전혀 쓰지 않는다 (사내엔 없다).

    ★ method/eps 를 안 주면 config.Cluster 에서 그때 읽는다. 예전엔 "leaf"/0.06 이
      하드코딩돼 있어서, config 로 METHOD=eom 을 켜도 **최종 grouping 만 eom 이고
      epoch 선택(Rule C)은 leaf 로 채점**됐다 — 고른 epoch 이 배포 다이얼 기준
      argmin 이 아니게 되는데 로그엔 아무 표시도 없었다 (260729 감사).
    """
    add_path(REPO)
    import torch
    import torch.nn.functional as F
    import grouping_deploy as gd

    if not method or eps is None:
        from config import Cluster as _C
        method = method or str(_C.METHOD)
        eps = _C.EPS if eps is None else eps
    _method, _eps = str(method), float(eps)
    print(f"  [score] HDBSCAN mcs={mcs} ms={ms} method={_method} eps={_eps}", flush=True)

    paths, _ = gd.collect_pool(str(rel(pool)))
    if not paths:
        die(f"pool 에 이미지가 없다: {pool}")
    device = gd.resolve_device(device)
    cache = gd.load_cache_for(cache_dir, paths)
    bb = gd.maybe_channels_last(gd.load_backbone(str(rel(backbone)), device))

    # backbone GAP feature 1회 (proj 없이) — 이후 모든 head 에 재사용.
    # ★ grouping_deploy.embed_backbone 과 **같은 경로**여야 한다.
    #   bb(x) 는 head norm 을 거쳐 분포가 달라지고(std 1.68 -> 0.23), proj head 는
    #   forward_features 기준으로 학습됐으므로 클러스터가 붕괴한다.
    #   직접 루프를 돌리면 캐시·스레드 디코드·UC_PALETTE_MASK 가 전부 빠진다.
    raw = gd.embed_backbone(paths, bb, device, batch, cache, "feat")
    print(f"  [feat] {len(paths)}/{len(paths)}  dim={raw.shape[1]}", flush=True)

    eps_files = sorted(Path(ckpt_dir).glob("proj_ep*.pt"),
                       key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))
    if not eps_files:
        die(f"epoch 체크포인트가 없다: {ckpt_dir}/proj_ep*.pt")

    rows = []
    for cp in eps_files:
        proj = gd.load_proj(str(cp), device)
        with torch.no_grad():
            z = F.normalize(proj(raw.to(device)), dim=1).cpu().numpy()
        pred = gd.hdbscan_predict(z, mcs, ms, _method, _eps)
        n = len(pred)
        k = len(set(pred[pred >= 0].tolist()))
        seed_noise = float((pred == -1).sum()) / n * 100.0
        rows.append({"ckpt": cp.name,
                     "epoch": int("".join(c for c in cp.stem if c.isdigit()) or 0),
                     "k": k, "seed_noise_pct": round(seed_noise, 2)})
        print(f"  [{cp.name}] k={k:<4} seed_noise={seed_noise:6.2f}%", flush=True)
    return rows


def rule_c(rows: list[dict], k_pct: float) -> dict:
    """Rule C — 라벨 없이 epoch 선택.
       gate 통과 -> k >= 자기 run 의 k_pct 퍼센타일 유지 -> argmin(seed_noise)

    k 하한이 핵심이다. 이게 없으면 noise 최소화가 '전부 한 덩어리로 병합' 이라는
    축퇴 해로 걸어간다(실측: frozen 이 k=2 로 뭉개면서 ARI 가 부풀려진 사례).
    """
    ok = [r for r in rows if r["k"] >= 2]
    if not ok:
        return {}
    ks = sorted(r["k"] for r in ok)
    idx = min(len(ks) - 1, int(round((k_pct / 100.0) * (len(ks) - 1))))
    kfloor = ks[idx]
    cand = [r for r in ok if r["k"] >= kfloor] or ok
    best = min(cand, key=lambda r: r["seed_noise_pct"])
    return {"selected": best, "k_floor": kfloor, "n_candidates": len(cand)}


def main() -> int:
    banner("STEP 2", "레시피 학습 + Rule C epoch 선택", "② 여기서 만든 레시피로 학습 후 predict")
    Config.OUT_ROOT = str(run_root(Config.OUT_BASE, create=False))
    cfg = show_config(Config)

    s0 = load_step0(Config.OUT_ROOT)
    pool = s0["manifest"]
    mcs, ms, method, eps = live_dial()              # ★ config 에서 매번 읽는다 (얼리지 않음)
    check_inputs(Config.BACKBONE, s0["image_root"])
    show_images(s0["image_root"], pool, getattr(Config, "EXTS", ""))
    _cache = s0.get("cache_dir", "") if getattr(Config, "CACHE", True) else ""   # step0 이 만든 디코드 캐시
    print(f"[pool] {pool}  n={s0['n_images']:,}장   [dial] config 값 그대로 mcs={mcs} ms={ms} "
          f"method={method} eps={eps}\n")

    out_root = step_dir(Config.OUT_ROOT, "step2_recipe")
    seeds = [int(s) for s in str(Config.SEEDS).split(",") if s.strip()]
    per_seed = {}

    for seed in seeds:
        run_dir = out_root / f"seed{seed}"
        ck = run_dir / "checkpoints"
        if not Config.SKIP_TRAIN:
            print(f"\n--- 학습 seed={seed} (epochs={Config.EPOCHS}) ---")
            rc = run([sys.executable, "-u", "_may_ablation.py", "B4"],
                     env_extra=train_env(Config.BACKBONE, pool, run_dir, seed,
                                         int(Config.EPOCHS), recipe()),
                     log_path=run_dir / "train.log")
            if rc != 0:
                print(f"[warn] 학습 종료코드 {rc} — 체크포인트 존재로 판정한다")
        # _may_ablation 이 REPRO_OUT 아래 run 폴더를 하나 더 만드는 경우 대비
        if not ck.exists():
            cands = sorted(run_dir.glob("**/checkpoints"))
            if cands:
                ck = cands[-1]
        if not ck.exists():
            print(f"[error] seed={seed} 체크포인트 없음 -> 건너뛴다 ({run_dir})")
            continue

        print(f"\n--- Rule C 채점 seed={seed} ---")
        rows = score_epochs(pool, Config.BACKBONE, ck, mcs, ms,
                            Config.DEVICE, int(Config.BATCH), _cache, method, eps)
        sel = rule_c(rows, float(Config.K_PERCENTILE))
        if not sel:
            print(f"[error] seed={seed} Rule C 선택 실패 (유효 epoch 없음)")
            continue
        print(f"\n  ★ Rule C 선택: {sel['selected']['ckpt']} "
              f"(k_floor={sel['k_floor']}, 후보 {sel['n_candidates']}개) "
              f"seed_noise={sel['selected']['seed_noise_pct']}%")
        per_seed[seed] = {"ckpt_dir": str(ck.relative_to(REPO)), "epochs": rows, **sel}

    if not per_seed:
        die("모든 seed 실패. train.log 를 봐라.")

    # 최종 산출: seed 별 선택 epoch 으로 grouping (대표 이미지/composite 포함)
    finals = {}
    for seed, info in per_seed.items():
        cp = Path(info["ckpt_dir"]) / info["selected"]["ckpt"]
        out = out_root / f"final_seed{seed}"
        run(deploy_cmd(Config.BACKBONE, pool, out, mcs, ms, [str(cp)],
                       Config.DEVICE, int(Config.BATCH), Config.REASSIGN, _cache, False, Config.PARTITION_BY),
            log_path=out_root / f"final_seed{seed}.log")
        finals[f"seed{seed}"] = read_summary(out)

    # 앙상블(전 seed concat+L2) — 배포 기본값이 앙상블이다
    ens_paths = [str(Path(i["ckpt_dir"]) / i["selected"]["ckpt"]) for i in per_seed.values()]
    ens = None
    if len(ens_paths) > 1:
        out = out_root / "final_ensemble"
        run(deploy_cmd(Config.BACKBONE, pool, out, mcs, ms, ens_paths,
                       Config.DEVICE, int(Config.BATCH), Config.REASSIGN, _cache, False, Config.PARTITION_BY),
            log_path=out_root / "final_ensemble.log")
        ens = read_summary(out)
        finals["ensemble"] = ens

    # 비교
    s1p = rel(Config.OUT_ROOT) / "step1_result.json"
    base = json.loads(s1p.read_text(encoding="utf-8"))["arms"] if s1p.exists() else {}
    print("\n" + "=" * 78)
    print("[결과] step1 기준선 대비 (같은 다이얼/pool/reassign)")
    print("=" * 78)
    # ★ 전 arm 을 찍되, 판정에 쓰는 기준선을 표시한다 (어느 게 같은 조건인지 보이게).
    _bfz, _bz0, _ = baseline_arms(base)
    for k, v in base.items():
        if not (k.startswith("frozen") or k.startswith("z0")):
            continue
        mark = " ←판정기준" if (v is _bfz or v is _bz0) else ""
        print(fmt_row(f"[step1] {k}{mark}", v))
    for name, s in finals.items():
        print(fmt_row(f"[step2] {name}", s))

    notes = judge(base, finals)
    print("\n" + "-" * 78)
    for line in notes:
        print(f"  {line}")
    print("-" * 78)

    save_result(rel(Config.OUT_ROOT), "step2",
                {"dial": {"mcs": mcs, "ms": ms}, "per_seed": per_seed,
                 "finals": finals, "notes": notes, "recipe": recipe(), "config": cfg})
    print("\n다음:  python deploy/step3_sweep.py")
    print(f"\n[OUT] {out_root}")
    return 0


def judge(base: dict, finals: dict) -> list[str]:
    _B = band("noise")          # ★ config.Judge.BAND 에서 live 로 읽는다
    sn = lambda s: (s or {}).get("seed_noise_pct", (s or {}).get("seed_noise"))
    out = []
    # ★ 내 palette 설정과 **같은 조건**의 기준선을 쓴다. 마스킹을 켜고 학습해놓고
    #   raw frozen 과 비교하면 마스킹 효과가 학습 효과로 둔갑한다 (260731 실측).
    fz, z0, _pnote = baseline_arms(base)
    out.append(f"기준선 palette = {_pnote}")
    best_name = max(finals, key=lambda k: -(sn(finals[k]) if sn(finals[k]) is not None else 1e9))
    best = finals[best_name]
    out.append(f"step2 최선: {best_name}  seed_noise={sn(best)}  k={(best or {}).get('k')}")
    _sfx = "_masked" if "같은 조건" in _pnote else ""
    for ref, label in ((fz, f"frozen{_sfx}"), (z0, f"z0(랜덤 head){_sfx}")):
        if ref is None or sn(ref) is None or sn(best) is None:
            continue
        d = sn(ref) - sn(best)
        if d > _B:
            out.append(f"  vs {label}: seed_noise {d:.2f}pp 개선 (잡음폭 {_B} 밖) — 유효")
        elif d < -_B:
            out.append(f"  vs {label}: {-d:.2f}pp 악화 — 이 pool 에서 레시피가 해롭다")
        else:
            out.append(f"  vs {label}: Δ{d:+.2f}pp — 잡음폭 안, 차이 없음")
    if z0 is not None and sn(z0) is not None and sn(best) is not None and sn(best) >= sn(z0) - _B:
        out.append("  ★ z0 를 못 이겼다 -> '학습이 개선했다'고 쓰지 마라. "
                   "projection head 의 기하 효과와 구분되지 않는다.")
    out.append("  ※ reassign 전(seed_noise)으로 판정했다. reassign 후 noise 는 "
               "어떤 임베딩에든 나오는 바닥값이라 판정에 쓰면 안 된다.")
    out.append("  다음: step3 로 레시피를 이 pool 에 맞게 스윕하면 더 오른다"
               "(우리 실측: LR 2배로 ARI +62%).")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
