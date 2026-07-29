#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""★ b4 backbone 의 우위가 **진짜 모델 품질인가, 혼돈계의 운 좋은 한 표본인가**.

배경 (260729 실측):
  weights/b4_may/b4_backbone.pth 를 FCMAE 원본과 텐서별로 비교하면
    - 378/378 텐서가 전부 다르지만 **중앙값 상대차 0.135%** 로 매우 작다 (>1% 인 건 31개뿐)
    - 가장 크게 바뀐 건 head.norm.bias (18.2%) 인데, 우리 임베딩 경로는
      `num_classes=0, global_pool=""` + `forward_features()` + 수동 GAP 이라
      **head.norm 을 아예 통과하지 않는다** -> 실제로 쓰는 부분의 차이는 0.135% 수준.
  그런데 이 프로젝트는 이미 "임베딩 5.5e-4 차이만으로 HDBSCAN 클러스터가 뒤집힌다"를
  실측했다 (grouping_deploy.amp_ctx 주석). 즉 0.1% 가중치 변화는 **혼돈 구간 안**일 수 있다.

질문:
  FCMAE 에 **b4 와 같은 크기의 무작위 섭동**을 주면 성능이 얼마나 흔들리는가?
  그 흔들림 폭이 b4 의 우위만 하다면, "b4 backbone 이 낫다"는 모델 품질이 아니라
  **뽑기 운**이다. (z0 랜덤-head 대조군과 같은 논리를 backbone 에 적용.)

판정:
  b4 의 지표가 섭동 분포 **안**에 들어오면  -> 우위는 재현 불가능한 표본 운. 채택 근거 무효.
  b4 가 섭동 분포 **밖**(꼬리 바깥)이면    -> 진짜 신호. 무엇이 그렇게 만들었는지 추적할 가치.

사용:
  python _b4_perturbation_control.py --pool data/pools/mwm38_clean546.json --n-seeds 8
  python _b4_perturbation_control.py --pool data/pools/anchor_avg30_repro.json --n-seeds 8 --mcs 12 --ms 4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.append(str(REPO / "scripts"))

import grouping_deploy as gd  # noqa: E402


def load_sd(p):
    sd = torch.load(str(p), map_location="cpu", weights_only=False)
    for k in ("state_dict", "model"):
        if isinstance(sd, dict) and k in sd:
            sd = sd[k]
    return sd


def per_tensor_rel_diff(a_sd, b_sd) -> dict:
    """b4 가 FCMAE 대비 텐서별로 **얼마나** 움직였는지 (상대 크기)."""
    out = {}
    for k, v in a_sd.items():
        if k in b_sd and hasattr(v, "shape") and v.shape == b_sd[k].shape:
            base = b_sd[k].float()
            n = base.abs().mean().item()
            if n > 0:
                out[k] = (v.float() - base).abs().mean().item() / n
    return out


def perturbed_sd(fc_sd, rel: dict, seed: int):
    """FCMAE 에 텐서별로 rel[k] 크기의 가우시안 섭동을 준다.
    E|N(0,s)| = s*sqrt(2/pi) 이므로 s = rel*mean|w| / sqrt(2/pi) 로 맞춘다."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    c = float(np.sqrt(2.0 / np.pi))
    for k, v in fc_sd.items():
        if k in rel and v.dtype.is_floating_point:
            w = v.float()
            s = rel[k] * w.abs().mean().item() / c
            out[k] = w + torch.randn(w.shape, generator=g) * s
        else:
            out[k] = v
    return out


def build_model(sd, device):
    import timm
    m = timm.create_model("convnextv2_base.fcmae_ft_in22k_in1k_384",
                          pretrained=False, num_classes=0, global_pool="")
    missing, unexpected = m.load_state_dict(sd, strict=False)
    crit = [k for k in missing if not k.startswith("head.")]
    if crit:
        raise RuntimeError(f"필수 가중치 누락 {len(crit)}개: {crit[:5]}")
    return gd.maybe_channels_last(m.eval().to(device))


def score(paths, labels, model, device, batch, cache, mcs, ms, method, eps):
    from eval_may37_checkpoints import (_summarize_predictions, _expand_ignored,
                                        _drop_megaclusters)
    with torch.no_grad():
        raw = gd.embed_backbone(paths, model, device, batch, cache, "feat")
    z = torch.nn.functional.normalize(raw.float(), dim=1).numpy().astype("float32")
    pred = gd.hdbscan_predict(z, mcs, ms, method, eps)
    lab = np.array([l if l is not None else "" for l in labels])
    ignored = _expand_ignored(lab, {"Normal", "R", "Random"})
    measured = ~np.isin(lab, list(ignored))
    fp = _drop_megaclusters(pred.copy(), 0.20)
    r = _summarize_predictions(z[measured], lab[measured], fp[measured], fp, lab, ignored, "perturb")
    return {"P1": r["P1_capture"], "P2_noise": r["P2_noise_pct"],
            "P3_comp": r["P3_completeness"], "P4_hom": r["P4_homogeneity"],
            "ARI": r["ARI"], "k": r["k"],
            "seed_noise": round(100.0 * float((pred == -1).sum()) / len(pred), 2)}


def p1num(v):
    try:
        return int(str(v).split("/")[0])
    except Exception:
        return -1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default="data/pools/mwm38_clean546.json")
    ap.add_argument("--fcmae", default="weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth")
    ap.add_argument("--b4", default="weights/b4_may/b4_backbone.pth")
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--mcs", type=int, default=6)
    ap.add_argument("--ms", type=int, default=3)
    ap.add_argument("--method", default="leaf")
    ap.add_argument("--eps", type=float, default=0.06)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="runs/b4_perturb/report.json")
    a = ap.parse_args()

    dev = gd.resolve_device(a.device)
    paths, labels = gd.collect_pool(str(REPO / a.pool))
    print(f"[pool] {a.pool}  n={len(paths)}  dial=mcs{a.mcs}/ms{a.ms}/{a.method}/eps{a.eps}", flush=True)

    fc_sd, b4_sd = load_sd(REPO / a.fcmae), load_sd(REPO / a.b4)
    rel = per_tensor_rel_diff(b4_sd, fc_sd)
    used = {k: v for k, v in rel.items() if not k.startswith("head.")}
    arr = np.array(list(used.values()))
    print(f"[b4 vs FCMAE] 매칭 텐서 {len(rel)}개 / head.* 제외(임베딩 미사용) {len(used)}개")
    print(f"              상대차 중앙값 {np.median(arr)*100:.4f}%  최대 {arr.max()*100:.3f}%", flush=True)

    rows = []
    print("\n[1/3] FCMAE 원본 채점", flush=True)
    rows.append(("FCMAE", score(paths, labels, build_model(fc_sd, dev), dev, a.batch, None,
                                a.mcs, a.ms, a.method, a.eps)))
    print("[2/3] b4 backbone 채점", flush=True)
    rows.append(("b4", score(paths, labels, build_model(b4_sd, dev), dev, a.batch, None,
                             a.mcs, a.ms, a.method, a.eps)))
    print(f"[3/3] 랜덤 섭동 {a.n_seeds}개 (b4 와 같은 크기, head.* 는 미사용이라 그대로)", flush=True)
    for s in range(1, a.n_seeds + 1):
        r = score(paths, labels, build_model(perturbed_sd(fc_sd, used, s), dev), dev,
                  a.batch, None, a.mcs, a.ms, a.method, a.eps)
        rows.append((f"perturb_s{s}", r))
        print(f"   s{s}: P1={r['P1']} noise={r['seed_noise']} ARI={r['ARI']}", flush=True)

    print("\n" + "=" * 76)
    print(f"{'arm':<14}{'P1':>7}{'seed_noise':>12}{'ARI':>9}{'Comp':>8}{'Hom':>8}{'k':>5}")
    print("-" * 76)
    for n, r in rows:
        print(f"{n:<14}{str(r['P1']):>7}{r['seed_noise']:>12}{r['ARI']:>9}"
              f"{r['P3_comp']:>8}{r['P4_hom']:>8}{r['k']:>5}")

    pert = [r for n, r in rows if n.startswith("perturb")]
    fc, b4 = rows[0][1], rows[1][1]
    print("\n" + "=" * 76)
    print("[판정] b4 의 우위가 '같은 크기 랜덤 섭동' 분포 안에 들어가는가?")
    for key, better_is_low in (("P1", False), ("seed_noise", True), ("ARI", False)):
        vals = [p1num(p[key]) if key == "P1" else p[key] for p in pert]
        bv = p1num(b4[key]) if key == "P1" else b4[key]
        fv = p1num(fc[key]) if key == "P1" else fc[key]
        lo, hi = min(vals), max(vals)
        inside = lo <= bv <= hi
        print(f"  {key:<11} FCMAE={fv:<9} b4={bv:<9} 섭동범위=[{lo}, {hi}]  "
              f"-> b4 {'분포 안 (운으로 설명됨)' if inside else '★분포 밖 (진짜 신호)'}")
    print("=" * 76)

    outp = REPO / a.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({
        "pool": a.pool, "n": len(paths),
        "dial": {"mcs": a.mcs, "ms": a.ms, "method": a.method, "eps": a.eps},
        "b4_vs_fcmae_rel_diff": {"median_pct": float(np.median(arr) * 100),
                                 "max_pct": float(arr.max() * 100),
                                 "n_tensors_used": len(used)},
        "arms": {n: r for n, r in rows},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OUT] {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
