#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 2 — 배포 사다리 ②: 여기서 만든 **레시피로 학습** 후 predict.

  python site/step2_recipe.py

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
from _site_common import (REPO, banner, check_inputs, deploy_cmd, die, env,  # noqa: E402
                          fmt_row, read_summary, rel, run, save_result,
                          show_config, train_env)


# ═══════════════════════════════════════════════════════════════════════════
class Config:
    """환경변수 우선, 없으면 default. 경로는 전부 프로젝트 루트 기준 상대경로."""

    OUT_ROOT = env("SITE_OUT_ROOT", "runs/site")
    BACKBONE = env("SITE_BACKBONE", "weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth")

    DEVICE = env("SITE_DEVICE", "cuda")
    BATCH = env("SITE_BATCH", 32)                 # 채점(inference) 배치
    REASSIGN = env("SITE_REASSIGN", "nearest_q90")

    SEEDS = env("SITE_SEEDS", "42,1,2")           # 3-seed 권장. 시간 없으면 "42"
    EPOCHS = env("SITE_EPOCHS", 20)

    # ── B4 recipe (여기서 만든 레시피) ────────────────────────────────────
    TEMP = env("SITE_TEMP", 0.20)
    QUEUE = env("SITE_QUEUE", 16384)
    IGNORE_NEG = env("SITE_IGNORE_NEG", 0.72)
    LR_HEAD = env("SITE_LR_HEAD", 0.004)
    TRAIN_BATCH = env("SITE_TRAIN_BATCH", 64)
    SAMPLING = env("SITE_SAMPLING", 0.25)
    USE_LOCAL = env("SITE_USE_LOCAL", True)

    # Rule C
    K_PERCENTILE = env("SITE_K_PERCENTILE", 75)

    SKIP_TRAIN = env("SITE_SKIP_TRAIN", False)    # 이미 학습돼 있으면 채점만
# ═══════════════════════════════════════════════════════════════════════════


def recipe_dict() -> dict:
    return {"temp": Config.TEMP, "queue": Config.QUEUE, "ignore_neg": Config.IGNORE_NEG,
            "lr_head": Config.LR_HEAD, "batch": Config.TRAIN_BATCH,
            "sampling": Config.SAMPLING, "use_local": bool(Config.USE_LOCAL)}


def load_step0(out_root) -> dict:
    p = rel(out_root) / "step0_result.json"
    if not p.exists():
        die("step0 결과가 없다. 먼저:  python site/step0_prepare.py")
    return json.loads(p.read_text(encoding="utf-8"))


def score_epochs(pool: str, backbone: str, ckpt_dir: Path, mcs: int, ms: int,
                 device: str, batch: int) -> list[dict]:
    """backbone feature 1회 계산 후 epoch head 별 label-free 지표.

    라벨을 전혀 쓰지 않는다 (사내엔 없다).
    """
    sys.path.insert(0, str(REPO))
    import numpy as np
    import torch
    import torch.nn.functional as F
    import grouping_deploy as gd

    paths, _ = gd.collect_pool(str(rel(pool)))
    if not paths:
        die(f"pool 에 이미지가 없다: {pool}")
    bb = gd.load_backbone(str(rel(backbone)), device)

    # backbone GAP feature 1회 (proj 없이) — 이후 모든 head 에 재사용
    tf = gd.T.Compose([gd.T.Resize((384, 384), interpolation=gd.T.InterpolationMode.BILINEAR),
                       gd.T.ToTensor(),
                       gd.T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            x = torch.stack([tf(gd.Image.open(p).convert("RGB")) for p in paths[i:i + batch]]).to(device)
            f = bb(x)
            f = f.mean(dim=(2, 3)) if f.ndim == 4 else f      # manual GAP
            feats.append(f.float().cpu())
            if i % (batch * 20) == 0:
                print(f"  [feat] {i}/{len(paths)}", flush=True)
    raw = torch.cat(feats)
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
        pred = gd.hdbscan_predict(z, mcs, ms, "leaf", 0.06)
        n = len(pred)
        k = len(set(pred[pred >= 0].tolist()))
        seed_noise = float((pred == -1).sum()) / n * 100.0
        stab = gd.per_group_stability(z, pred, mcs, ms, "leaf", 0.06)
        rows.append({"ckpt": cp.name,
                     "epoch": int("".join(c for c in cp.stem if c.isdigit()) or 0),
                     "k": k, "seed_noise_pct": round(seed_noise, 2),
                     "mean_stability": round(float(np.mean(list(stab.values()))) if stab else 0.0, 4)})
        print(f"  [{cp.name}] k={k:<4} seed_noise={seed_noise:6.2f}%  "
              f"stab={rows[-1]['mean_stability']}", flush=True)
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
    cfg = show_config(Config)

    s0 = load_step0(Config.OUT_ROOT)
    pool, mcs, ms = s0["manifest"], s0["dial"]["mcs"], s0["dial"]["ms"]
    check_inputs(Config.BACKBONE, s0["image_root"], s0["k_hat"])
    print(f"[step0] pool={pool}  n={s0['n_images']:,}  dial mcs={mcs} ms={ms}\n")

    out_root = rel(Config.OUT_ROOT) / "step2_recipe"
    seeds = [int(s) for s in str(Config.SEEDS).split(",") if s.strip()]
    per_seed = {}

    for seed in seeds:
        run_dir = out_root / f"seed{seed}"
        ck = run_dir / "checkpoints"
        if not Config.SKIP_TRAIN:
            print(f"\n--- 학습 seed={seed} (epochs={Config.EPOCHS}) ---")
            rc = run([sys.executable, "-u", "_may_ablation.py", "B4"],
                     env_extra=train_env(Config.BACKBONE, pool, run_dir, seed,
                                         int(Config.EPOCHS), recipe_dict()),
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
                            Config.DEVICE, int(Config.BATCH))
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
                       Config.DEVICE, int(Config.BATCH), Config.REASSIGN),
            log_path=out_root / f"final_seed{seed}.log")
        finals[f"seed{seed}"] = read_summary(out)

    # 앙상블(전 seed concat+L2) — 배포 기본값이 앙상블이다
    ens_paths = [str(Path(i["ckpt_dir"]) / i["selected"]["ckpt"]) for i in per_seed.values()]
    ens = None
    if len(ens_paths) > 1:
        out = out_root / "final_ensemble"
        run(deploy_cmd(Config.BACKBONE, pool, out, mcs, ms, ens_paths,
                       Config.DEVICE, int(Config.BATCH), Config.REASSIGN),
            log_path=out_root / "final_ensemble.log")
        ens = read_summary(out)
        finals["ensemble"] = ens

    # 비교
    s1p = rel(Config.OUT_ROOT) / "step1_result.json"
    base = json.loads(s1p.read_text(encoding="utf-8"))["arms"] if s1p.exists() else {}
    print("\n" + "=" * 78)
    print("[결과] step1 기준선 대비 (같은 다이얼/pool/reassign)")
    print("=" * 78)
    for name in ("frozen",):
        if base.get(name):
            print(fmt_row(f"[step1] {name}", base[name]))
    for k, v in base.items():
        if k.startswith("z0"):
            print(fmt_row(f"[step1] {k}", v))
    for name, s in finals.items():
        print(fmt_row(f"[step2] {name}", s))

    notes = judge(base, finals)
    print("\n" + "-" * 78)
    for line in notes:
        print(f"  {line}")
    print("-" * 78)

    save_result(rel(Config.OUT_ROOT), "step2",
                {"dial": {"mcs": mcs, "ms": ms}, "per_seed": per_seed,
                 "finals": finals, "notes": notes, "recipe": recipe_dict(), "config": cfg})
    print("\n다음:  python site/step3_sweep.py")
    print(f"\n[OUT] {out_root}")
    return 0


def judge(base: dict, finals: dict) -> list[str]:
    sn = lambda s: (s or {}).get("seed_noise_pct", (s or {}).get("seed_noise"))
    out = []
    fz, z0k = base.get("frozen"), next((k for k in base if k.startswith("z0")), None)
    z0 = base.get(z0k) if z0k else None
    best_name = max(finals, key=lambda k: -(sn(finals[k]) if sn(finals[k]) is not None else 1e9))
    best = finals[best_name]
    out.append(f"step2 최선: {best_name}  seed_noise={sn(best)}  k={(best or {}).get('k')}")
    for ref, label in ((fz, "frozen"), (z0, "z0(랜덤 head)")):
        if ref is None or sn(ref) is None or sn(best) is None:
            continue
        d = sn(ref) - sn(best)
        if d > 2.28:
            out.append(f"  vs {label}: seed_noise {d:.2f}pp 개선 (잡음폭 2.28 밖) — 유효")
        elif d < -2.28:
            out.append(f"  vs {label}: {-d:.2f}pp 악화 — 이 pool 에서 레시피가 해롭다")
        else:
            out.append(f"  vs {label}: Δ{d:+.2f}pp — 잡음폭 안, 차이 없음")
    if z0 is not None and sn(z0) is not None and sn(best) is not None and sn(best) >= sn(z0) - 2.28:
        out.append("  ★ z0 를 못 이겼다 -> '학습이 개선했다'고 쓰지 마라. "
                   "projection head 의 기하 효과와 구분되지 않는다.")
    out.append("  ※ reassign 전(seed_noise)으로 판정했다. reassign 후 noise 는 "
               "어떤 임베딩에든 나오는 바닥값이라 판정에 쓰면 안 된다.")
    out.append("  다음: step3 로 레시피를 이 pool 에 맞게 스윕하면 더 오른다"
               "(우리 실측: LR 2배로 ARI +62%).")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
