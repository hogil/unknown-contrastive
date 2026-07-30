#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""자매 repo `known-cnn` 의 supervised 체크포인트에서 **TAPT backbone** 을 추출한다.

  python deploy/extract_tapt.py --src D:/project/known-cnn/models/iter116J_frozen/best_model.pth

★ TAPT 가 뭔가
  Task-Adaptive PreTraining — ImageNet FCMAE backbone 을 **우리 도메인(웨이퍼) 라벨로
  supervised 학습**시켜 도메인에 맞게 옮겨 놓은 backbone. 사다리 ④가 이걸 쓴다.
  ⚠ 라벨이 필요하다. 사내엔 라벨이 없으므로 (a) 소량이라도 라벨링해서 known-cnn 으로
    학습하거나 (b) 우리가 이미 만든 걸 이식해야 한다.

★ 왜 이 스크립트가 필요한가 (260730 실측)
  `weights/tapt/backbone_tapt.pth` 가 **FCMAE 와 378개 텐서 전부 동일**(상대차 0.0000%)
  이었다. TAPT 가 아니라 FCMAE 사본이었고, step4 는 그걸 모른 채 sweep 을 다시 돌려
  "③과 ④ 동률" 이라는 **가짜 결론**을 냈다 — FCMAE 를 FCMAE 와 비교한 것이다.
  그래서 이 스크립트는 **추출 후 반드시 FCMAE 와 비교해서, 안 움직였으면 저장을 거부**한다.

★ known-cnn 체크포인트 형식 (실측)
  { "model": <380 텐서 state_dict>, "classes": [...], "img_size": ..., "backbone": ...,
    "val_acc": ..., "epoch": ..., "variant": ..., "loss_name": ... }
  키는 timm convnextv2 그대로(`stem.*`, `stages.*`, `head.norm.*`, `head.fc.*`).
  ⚠ `grouping_deploy.load_backbone` 은 최상위 `"model"` 키를 풀어주지 않는다
    (`"state_dict"` 만 푼다). 그래서 여기서 **평평한 state_dict 로 저장**해야 한다.
  ⚠ `head.fc.*` 는 분류기라 뺀다 — 우리는 `num_classes=0, global_pool=""` 로 쓰고
    `forward_features()` + 수동 GAP 이라 head 를 통과하지도 않는다.

산출: weights/tapt/backbone_tapt.pth  -> `deploy/step4_tapt.py` 가 자동으로 쓴다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _site_common import banner, die, rel  # noqa: E402


def load_sd(p: Path) -> dict:
    d = torch.load(str(p), map_location="cpu", weights_only=False)
    meta = {}
    if isinstance(d, dict):
        meta = {k: v for k, v in d.items()
                if k in ("classes", "img_size", "backbone", "val_acc", "epoch",
                         "variant", "loss_name") }
        for k in ("model", "state_dict"):
            if k in d:
                d = d[k]
                break
    return d, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True,
                    help="known-cnn 의 best_model.pth (supervised 학습 결과)")
    ap.add_argument("--out", default="weights/tapt/backbone_tapt.pth")
    ap.add_argument("--fcmae", default="weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth")
    ap.add_argument("--force", action="store_true",
                    help="기존 파일을 덮어쓴다 (백업본을 남긴다)")
    a = ap.parse_args()

    banner("EXTRACT TAPT", "known-cnn supervised 체크포인트 -> TAPT backbone")

    src = Path(a.src)
    if not src.exists():
        die(f"src 가 없다: {src}\n"
            "  known-cnn 에서 supervised 학습을 먼저 해라 (라벨 필요).")
    sd, meta = load_sd(src)
    if not isinstance(sd, dict) or not sd:
        die(f"state_dict 를 못 찾았다: {src}")
    print(f"[src] {src}")
    if meta:
        print(f"      meta: {({k: (v if not isinstance(v, list) else f'{len(v)} classes') for k, v in meta.items()})}")
    print(f"      텐서 {len(sd)}개")

    # 분류기 head 제거 — 우리는 forward_features + 수동 GAP 이라 안 쓴다.
    drop = [k for k in sd if k.startswith("head.fc.")]
    bb = {k: v for k, v in sd.items() if not k.startswith("head.fc.")}
    print(f"[strip] head.fc.* {len(drop)}개 제거 -> {len(bb)}개")

    # ★ 핵심 검증: FCMAE 와 실제로 다른가. 같으면 TAPT 가 아니다.
    fcp = rel(a.fcmae)
    if not fcp.exists():
        die(f"FCMAE 기준본이 없다: {fcp}")
    fc, _ = load_sd(fcp)
    common = [k for k in bb if k in fc and hasattr(bb[k], "shape")
              and bb[k].shape == fc[k].shape]
    if not common:
        die("FCMAE 와 겹치는 텐서가 없다 — 아키텍처가 다른 체크포인트다.\n"
            "  convnextv2_base.fcmae_ft_in22k_in1k_384 로 학습된 것이어야 한다.")
    rel_d = []
    for k in common:
        base = fc[k].float().abs().mean().item()
        if base > 0:
            rel_d.append((bb[k].float() - fc[k].float()).abs().mean().item() / base)
    n_diff = sum(1 for x in rel_d if x > 1e-9)
    print(f"[check] FCMAE 대비: 매칭 {len(common)}개 중 **다른 텐서 {n_diff}개**  "
          f"중앙값 {np.median(rel_d)*100:.4f}%  최대 {max(rel_d)*100:.2f}%")
    if n_diff == 0:
        die("이 체크포인트의 backbone 이 **FCMAE 와 완전히 동일**하다 — TAPT 가 아니다.\n"
            "  supervised 학습에서 backbone 을 freeze 하고 head 만 학습한 결과로 보인다.\n"
            "  backbone 까지 학습(unfreeze)한 체크포인트를 써라. 그러지 않으면 사다리 ④가\n"
            "  FCMAE 를 FCMAE 와 비교하게 되어 판정이 무의미해진다 (260730 실측 전례).")
    print(f"        -> ★ 적응 확인. 참고로 May 배포본 b4 는 중앙값 0.135% 였다.")

    out = rel(a.out)
    if out.exists() and not a.force:
        die(f"이미 있다: {out}\n"
            "  덮어쓰려면 --force (기존 파일은 .bak 으로 백업된다).")
    if out.exists():
        bak = out.with_suffix(out.suffix + ".bak")
        out.replace(bak)
        print(f"[backup] 기존 파일 -> {bak}")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bb, out)
    print(f"\n[OUT] {out}  ({out.stat().st_size / 2**20:.0f} MiB)")

    # 실제로 우리 로더가 읽는지 확인 (형식 실수를 여기서 잡는다)
    try:
        sys.path.append(str(Path(__file__).resolve().parent.parent))
        import grouping_deploy as gd
        m = gd.load_backbone(str(out), "cpu")
        n_par = sum(p.numel() for p in m.parameters())
        print(f"[verify] grouping_deploy.load_backbone 성공 — {n_par/1e6:.1f}M params")
    except Exception as e:
        print(f"[warn] 로더 검증 실패 ({type(e).__name__}: {e}) — 형식을 확인해라")

    print("\n다음:  python deploy/step4_tapt.py   (③ FCMAE sweep 과 비교한다)")
    print("★ 실측 경고(260724): new-domain 에서는 no-TAPT FCMAE 가 TAPT 를 앞섰다.")
    print("  사다리는 낮은 순위가 우선이라, ④는 ③을 잡음폭 밖으로 이겨야만 채택한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
