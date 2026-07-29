#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""★ backbone 부분 해동이 head-only 학습을 이기는가.

동기 (260729, `_b4_perturbation_control.py` 결과):
  배포본 b4 가 웨이퍼 pool 에서 이긴 이유는 backbone 을 **아주 약하게 방향성 있게**
  적응시킨 것이었다 (FCMAE 대비 중앙값 0.135%; 같은 크기 랜덤섭동 8개는 전부 못 미침).
  ★ 단 그건 **도메인 특화**다 — 철강(severstal)에선 b4 가 FCMAE 보다도, 랜덤섭동보다도
  나빴다(ARI 0.331 -> 0.212, P1 4/4 -> 3/4). 즉 "b4 를 쓰자"가 아니라
  **"우리가 대상 도메인에서 같은 종류의 적응을 직접 만들 수 있나"** 가 진짜 질문이다.
  그런데 현 레시피는 FREEZE_BACKBONE=True 라 그 축을 아예 못 건드렸다.

설계:
  - Rule C(에폭 선택)를 **쓰지 않는다.** per-epoch 로 backbone 을 저장하면 20 epoch 에
    7GB 라 비현실적이고, 무엇보다 arm 마다 다른 epoch 을 고르면 비교가 흐려진다.
    **모든 arm 을 같은 epoch 수로 학습해 마지막 체크포인트끼리 비교**한다 (apples-to-apples).
  - 학습된 전체 모델은 `checkpoints/last_training.pt` 의 `model` 에 들어 있다
    (386 텐서 = backbone 378 + head). 이걸 그대로 실어서 채점한다.
  - 참조선으로 FCMAE(학습 0)와 b4 를 같은 경로로 함께 찍는다.

사용:
  python _unfreeze_experiment.py --pool data/pools/mwm38_clean546.json --epochs 20
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.append(str(REPO / "scripts"))
import grouping_deploy as gd  # noqa: E402

FCMAE = "weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth"

# (tag, UNFREEZE_LAST_N, LR_BACKBONE, TRAIN_BATCH) — LR_BACKBONE 0 이면 LR_HEAD/100
# ★ 해동 stage 를 늘리면 activation 이 늘어 OOM 난다 (uf2 가 batch64 에서 CUDA OOM,
#   260729 실측). 학습 batch 만 낮춘다 — 채점 batch 는 arm 마다 동일하게 유지해야
#   비교가 안 흔들린다.
ARMS = [
    ("frozen_headonly", 0, 0.0,  64),   # 현행 레시피 = 기준선
    ("uf1_lr1e-5",      1, 1e-5, 64),   # 마지막 stage, head LR 의 1/100
    ("uf1_lr1e-4",      1, 1e-4, 64),   # 마지막 stage, 1/10  (더 크게 움직임)
    ("uf2_lr1e-5",      2, 1e-5, 24),   # 마지막 2 stage (batch 낮춰 OOM 회피)
    ("uf1_lr1e-3",      1, 1e-3, 64),   # 마지막 stage, head 와 동일 LR
    # ★ 1e-5<1e-4<1e-3 로 단조 개선했고 1e-3 이 그리드 최댓값이라 **경계에서 멈춤**.
    #   위를 열어 진짜 최적점이 내부에 있는지 본다 (backbone 이 부서지는 지점도 같이 확인).
    ("uf1_lr2e-3",      1, 2e-3, 64),
    ("uf1_lr4e-3",      1, 4e-3, 64),
]


def build_head(proj_sd, dev):
    """last_training.pt 의 `proj.net.*` 를 grouping_deploy 와 **같은 구조**로 되살린다.
    ★ 이걸 안 붙이면 학습한 head 를 버리고 backbone 만 채점하게 되어,
      head-only 학습 arm 이 학습 안 한 FCMAE 와 **소수점까지 같은 점수**로 나온다
      (260729 실측으로 걸린 함정)."""
    pj = {k[len("net."):] if k.startswith("net.") else k: v for k, v in proj_sd.items()}
    proj = gd.build_proj()
    proj.load_state_dict(pj)
    return proj.eval().to(dev)


def score_state(sd_backbone, proj_sd, paths, labels, dev, batch, mcs, ms, method, eps):
    """backbone state + (선택) proj 로 임베딩 -> HDBSCAN -> 라벨 채점."""
    import timm
    from eval_may37_checkpoints import (_summarize_predictions, _expand_ignored,
                                        _drop_megaclusters)
    m = timm.create_model("convnextv2_base.fcmae_ft_in22k_in1k_384",
                          pretrained=False, num_classes=0, global_pool="")
    miss, _ = m.load_state_dict(sd_backbone, strict=False)
    crit = [k for k in miss if not k.startswith("head.")]
    if crit:
        raise RuntimeError(f"backbone 가중치 누락 {len(crit)}개: {crit[:4]}")
    m = gd.maybe_channels_last(m.eval().to(dev))
    with torch.no_grad():
        raw = gd.embed_backbone(paths, m, dev, batch, None, "feat")
        if proj_sd:
            raw = build_head(proj_sd, dev)(raw.to(dev)).cpu()
    z = torch.nn.functional.normalize(raw.float(), dim=1).numpy().astype("float32")
    pred = gd.hdbscan_predict(z, mcs, ms, method, eps)
    lab = np.array([l if l is not None else "" for l in labels])
    ign = _expand_ignored(lab, {"Normal", "R", "Random"})
    meas = ~np.isin(lab, list(ign))
    fp = _drop_megaclusters(pred.copy(), 0.20)
    r = _summarize_predictions(z[meas], lab[meas], fp[meas], fp, lab, ign, "uf")
    return {"P1": r["P1_capture"], "P2_noise": r["P2_noise_pct"], "P3_comp": r["P3_completeness"],
            "P4_hom": r["P4_homogeneity"], "ARI": r["ARI"], "k": r["k"],
            "seed_noise": round(100.0 * float((pred == -1).sum()) / len(pred), 2)}


def load_sd(p):
    sd = torch.load(str(p), map_location="cpu", weights_only=False)
    for k in ("state_dict", "model"):
        if isinstance(sd, dict) and k in sd:
            sd = sd[k]
    return sd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="data/pools/mwm38_clean546.json")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--mcs", type=int, default=6)
    ap.add_argument("--ms", type=int, default=3)
    ap.add_argument("--method", default="leaf")
    ap.add_argument("--eps", type=float, default=0.06)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/unfreeze/report.json")
    ap.add_argument("--skip-train", action="store_true", help="이미 학습된 것만 채점")
    ap.add_argument("--arms", default="", help="콤마구분 tag 만 실행 (기본 전체)")
    a = ap.parse_args()

    dev = gd.resolve_device(a.device)
    paths, labels = gd.collect_pool(str(REPO / a.pool))
    print(f"[pool] {a.pool} n={len(paths)}  dial=mcs{a.mcs}/ms{a.ms}/{a.method}/{a.eps} "
          f"epochs={a.epochs} seed={a.seed}\n", flush=True)

    rows = {}
    print("[ref] FCMAE (학습 0)", flush=True)
    rows["FCMAE_frozen_nohead"] = score_state(load_sd(REPO / FCMAE), None, paths, labels,
                                              dev, a.batch, a.mcs, a.ms, a.method, a.eps)
    b4p = REPO / "weights/b4_may/b4_backbone.pth"
    if b4p.exists():
        print("[ref] b4 backbone (배포본)", flush=True)
        rows["b4_backbone_ref"] = score_state(load_sd(b4p), None, paths, labels,
                                              dev, a.batch, a.mcs, a.ms, a.method, a.eps)

    out_root = REPO / "runs/unfreeze"
    _only = {x.strip() for x in a.arms.split(",") if x.strip()}
    for tag, n_uf, lr_bb, tb in ARMS:
        if _only and tag not in _only:
            continue
        run_dir = out_root / (tag if a.seed == 42 else f"{tag}_s{a.seed}")
        ck_glob = sorted(run_dir.glob(f"abl_uf_{tag}_B4_*/checkpoints/last_training.pt"))
        if not ck_glob and not a.skip_train:
            print(f"\n=== 학습 {tag} (unfreeze={n_uf}, lr_bb={lr_bb or 'auto(1/100)'}) ===", flush=True)
            env = dict(os.environ)
            env.update({
                "REPRO_DATA": str(REPO / a.pool), "REPRO_BACKBONE": str(REPO / FCMAE),
                "REPRO_OUT": str(run_dir / "run"), "REPRO_BATCH": str(tb),
                "REPRO_EPOCHS": str(a.epochs), "REPRO_SEED": str(a.seed),
                "REPRO_TAG": f"_uf_{tag}", "REPRO_UNFREEZE": str(n_uf),
                "REPRO_LR_BACKBONE": str(lr_bb), "PYTHONIOENCODING": "utf-8",
            })
            subprocess.call([sys.executable, "-u", "_may_ablation.py", "B4"],
                            cwd=str(REPO), env=env)
            ck_glob = sorted(run_dir.glob(f"abl_uf_{tag}_B4_*/checkpoints/last_training.pt"))
        if not ck_glob:
            print(f"[error] {tag}: last_training.pt 없음 -> 건너뜀", flush=True)
            continue
        full = load_sd(ck_glob[-1])
        bb = {k[len("backbone."):]: v for k, v in full.items() if k.startswith("backbone.")}
        if not bb:
            print(f"[error] {tag}: 체크포인트에 backbone.* 가 없다 -> 건너뜀", flush=True)
            continue
        pj = {k[len("proj."):]: v for k, v in full.items() if k.startswith("proj.")}
        print(f"[score] {tag}  (backbone {len(bb)}개 + head {len(pj)}개)", flush=True)
        rows[tag] = score_state(bb, pj, paths, labels, dev, a.batch,
                                a.mcs, a.ms, a.method, a.eps)
        # 학습이 backbone 을 실제로 얼마나 움직였는지.
        # ★ 378개 **전체**의 중앙값을 쓰면 안 된다 — 해동한 건 stages.3 뿐(약 20%)이라
        #   중앙값이 항상 0 으로 나와 "안 움직였다"는 거짓 신호를 준다 (260729 실측).
        #   변한 텐서만 골라 중앙값을 내고, 몇 개가 변했는지도 같이 남긴다.
        fc = load_sd(REPO / FCMAE)
        d = [((bb[k].float() - fc[k].float()).abs().mean() / fc[k].float().abs().mean()).item()
             for k in bb if k in fc and bb[k].shape == fc[k].shape and fc[k].float().abs().mean() > 0]
        moved = [x for x in d if x > 1e-9]
        rows[tag]["bb_moved_tensors"] = f"{len(moved)}/{len(d)}"
        rows[tag]["bb_shift_median_pct"] = round(float(np.median(moved)) * 100, 4) if moved else 0.0

    print("\n" + "=" * 84)
    print(f"{'arm':<22}{'P1':>7}{'seed_noise':>12}{'ARI':>9}{'Comp':>8}{'Hom':>8}{'k':>5}"
          f"{'bbΔ%':>9}{'moved':>10}")
    print("-" * 84)
    for n, r in rows.items():
        print(f"{n:<22}{str(r['P1']):>7}{r['seed_noise']:>12}{r['ARI']:>9}"
              f"{r['P3_comp']:>8}{r['P4_hom']:>8}{r['k']:>5}"
              f"{str(r.get('bb_shift_median_pct','-')):>9}{str(r.get('bb_moved_tensors','-')):>10}")
    print("=" * 84)
    print("★ bbΔ% = **변한 텐서만**의 중앙값 / moved = 변한 텐서 수. b4 는 0.135%(378/378) 였다.")
    print("★ 판정: uf* 가 frozen_headonly 를 P1>P2_noise 순으로 이겨야 부분해동이 이득이다.")

    outp = REPO / a.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({"pool": a.pool, "n": len(paths), "epochs": a.epochs,
                                "seed": a.seed,
                                "dial": {"mcs": a.mcs, "ms": a.ms, "method": a.method, "eps": a.eps},
                                "arms": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OUT] {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
