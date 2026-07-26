#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""May 배포본 `contrastive_b4.pt` 에서 backbone / proj 를 분리 추출한다.

  python site/extract_b4.py

★ 왜 분리가 필요한가
  `contrastive_b4.pt` 는 `state_dict` 하나에 **backbone 378개 + proj 8개** 키를 함께 담고 있다.
  그리고 그 backbone 은 같은 폴더의 `backbone.pth` 와 **378개 텐서가 전부 다르다**(실측).
  즉 b4 는 우리 frozen FCMAE 와 부품을 섞을 수 없는 **독립 arm** 이다.
  b4 proj 를 FCMAE 위에 올리면 학습 때와 다른 feature 분포를 먹여서 결과가 무의미해진다.

산출:
  weights/b4_may/b4_backbone.pth   (timm convnextv2_base state_dict)
  weights/b4_may/b4_proj.pt        ({"proj": {...}} — grouping_deploy.load_proj 형식)

두 파일이 있으면 `site/step1_zeroshot.py` 가 자동으로 4번째 arm 으로 비교에 넣는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _site_common import banner, die, env, rel, show_config  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
class Config:
    """환경변수 우선, 없으면 default."""

    # 원본 번들. 프로젝트 밖 절대경로도 허용(외부에서 받은 파일이므로).
    SRC = env("SITE_B4_SRC", "weights/b4_may/contrastive_b4.pt")

    OUT_BACKBONE = env("SITE_B4_BACKBONE", "weights/b4_may/b4_backbone.pth")
    OUT_PROJ = env("SITE_B4_PROJ", "weights/b4_may/b4_proj.pt")

    # 같은 폴더의 backbone.pth 와 정말 다른지 검증 (있을 때만)
    COMPARE_WITH = env("SITE_B4_COMPARE", "weights/b4_may/backbone.pth")
# ═══════════════════════════════════════════════════════════════════════════


def main() -> int:
    banner("EXTRACT", "contrastive_b4.pt -> backbone / proj 분리")
    show_config(Config)
    import torch

    src = rel(Config.SRC)
    if not src.exists():
        die(f"원본이 없다: {src}\n"
            f"  contrastive_b4.pt 를 그 경로에 두거나 SITE_B4_SRC 로 지정해라.\n"
            f"  (원본 위치 예: <failure_agent>/checkpoints/contrastive_b4.pt)")

    d = torch.load(src, map_location="cpu", weights_only=False)
    sd = d.get("state_dict", d) if isinstance(d, dict) else d
    if not isinstance(sd, dict):
        die(f"예상과 다른 형식이다: {type(sd).__name__}")

    bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    pj = {k[len("proj."):]: v for k, v in sd.items() if k.startswith("proj.")}
    if not bb or not pj:
        die(f"backbone/proj 키를 못 찾았다. 최상위 prefix: "
            f"{sorted({k.split('.')[0] for k in sd})[:10]}")

    ob, op = rel(Config.OUT_BACKBONE), rel(Config.OUT_PROJ)
    ob.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bb, ob)
    torch.save({"proj": pj}, op)
    print(f"\n  backbone {len(bb):>4} keys -> {ob}")
    print(f"  proj     {len(pj):>4} keys -> {op}")
    print(f"    {list(pj)}")

    cmp_p = rel(Config.COMPARE_WITH)
    if cmp_p.exists():
        other = torch.load(cmp_p, map_location="cpu", weights_only=False)
        other = other.get("state_dict", other) if isinstance(other, dict) else other
        common = set(other) & set(bb)
        same = sum(1 for k in common if torch.equal(other[k], bb[k]))
        print(f"\n  [검증] {cmp_p.name} 와 공통 키 {len(common)}, 동일 텐서 {same}")
        if same == len(common) and common:
            print("    -> 같은 backbone 이다. b4 proj 를 그 backbone 위에 올려도 된다.")
        else:
            print("    -> ★ 다른 backbone 이다. **b4 proj 는 반드시 b4_backbone.pth 와 함께** 써라.")
            print("       FCMAE 위에 올리면 학습 때와 다른 feature 를 먹여서 결과가 무의미해진다.")

    print(f"\n[OUT] {ob.parent}")
    print("\n다음:  python site/step1_zeroshot.py   (b4 가 4번째 arm 으로 자동 포함된다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
