#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 0 — 사내 이미지 폴더 -> manifest + pool 기하 리포트 + 권장 다이얼.

  python site/step0_prepare.py

여기서 정한 다이얼(mcs/ms)을 이후 step1~5 가 전부 물려 쓴다. 이 값이 틀리면
그 뒤 모든 결론이 조용히 뒤집힌다(실측 전례 있음) — 반드시 K_HAT 을 현업 값으로 채워라.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _site_common import (REPO, banner, dial_search_range, die, env, rel,  # noqa: E402
                          recommend_dial, save_result, show_config)


# ═══════════════════════════════════════════════════════════════════════════
class Config:
    """환경변수가 있으면 그것, 없으면 아래 default. 경로는 전부 프로젝트 루트 기준 상대경로."""

    # 사내 이미지 루트. flat(`*.png`) / 중첩(`<any>/<any>/*.png`) 둘 다 됨.
    IMAGE_ROOT = env("SITE_IMAGE_ROOT", "data/site_images")

    # ★ 예상 불량 종수. 현업에서 받아야 한다. 이 값으로 다이얼이 정해진다.
    #   라벨 없이 다이얼을 고를 방법이 없다는 게 168셀 실측으로 확인됐다.
    K_HAT = env("SITE_K_HAT", 8)

    # 산출 루트
    OUT_ROOT = env("SITE_OUT_ROOT", "runs/site")

    # 이미지 확장자
    EXTS = env("SITE_EXTS", "png,jpg,jpeg,bmp,tif,tiff")

    # 이미 만들어둔 manifest 가 있으면 그걸 쓴다 (스캔 생략). IMAGE_ROOT 는 무시된다.
    # 사내에서 대상 이미지를 미리 골라둔 경우 유용.
    POOL_MANIFEST = env("SITE_POOL_MANIFEST", "")

    # 다이얼 수동 고정(0 이면 기하에서 자동 계산). 튜닝 목적 외에는 건드리지 마라.
    MCS_OVERRIDE = env("SITE_MCS", 0)
    MS_OVERRIDE = env("SITE_MS", 0)
# ═══════════════════════════════════════════════════════════════════════════


def main() -> int:
    banner("STEP 0", "manifest 생성 + pool 기하 -> 권장 다이얼")
    cfg = show_config(Config)

    # ── 기존 manifest 사용 경로 ──────────────────────────────────────────
    if str(Config.POOL_MANIFEST).strip():
        mp = rel(Config.POOL_MANIFEST)
        if not mp.exists():
            die(f"SITE_POOL_MANIFEST 가 없다: {mp}")
        man = json.loads(mp.read_text(encoding="utf-8"))
        n = len(man.get("files", []))
        if n == 0:
            die(f"manifest 에 files 가 없다: {mp}")
        root = Path(man["root"])
        k_hat = int(Config.K_HAT)
        mcs, ms = recommend_dial(n, k_hat)
        if int(Config.MCS_OVERRIDE) > 0:
            mcs = int(Config.MCS_OVERRIDE)
        if int(Config.MS_OVERRIDE) > 0:
            ms = int(Config.MS_OVERRIDE)
        per_class = n / k_hat
        print(f"\n[pool] 기존 manifest 사용: {mp}")
        print(f"[pool] n = {n:,} 장   root = {root}")
        print(f"[pool] k_hat = {k_hat}  ->  클래스당 예상 {per_class:.1f} 장")
        print(f"\n[dial] ★ 권장  mcs = {mcs} , ms = {ms}   (= n/k 의 {100*mcs/per_class:.1f}%)")
        out_root = rel(Config.OUT_ROOT)
        save_result(out_root, "step0", {
            "n_images": n, "k_hat": k_hat, "per_class_est": round(per_class, 1),
            "manifest": str(mp.relative_to(REPO)) if mp.is_relative_to(REPO) else str(mp),
            "image_root": root.as_posix(), "subdirs": [],
            "dial": {"mcs": mcs, "ms": ms, "method": "leaf", "eps": 0.06},
            "dial_search_mcs": dial_search_range(n, k_hat), "config": cfg})
        print("\n다음:  python site/step1_zeroshot.py")
        print(f"\n[OUT] {out_root}")
        return 0

    root = rel(Config.IMAGE_ROOT)
    if not root.exists():
        die(f"image_root 가 없다: {root}\n"
            f"  SITE_IMAGE_ROOT 환경변수로 지정하거나 Config.IMAGE_ROOT default 를 고쳐라.\n"
            f"  (프로젝트 루트 기준 상대경로 권장 — 예: data/site_images)")

    exts = {("." + e.strip().lstrip(".")).lower() for e in str(Config.EXTS).split(",") if e.strip()}
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)
    n = len(files)
    if n == 0:
        die(f"이미지가 하나도 없다: {root} (확장자 {sorted(exts)})")

    # 하위 폴더명이 있으면 참고용으로만 기록. 사내 무라벨 pool 이면 label=null 로 둔다.
    subdirs = sorted({p.parent.relative_to(root).as_posix() for p in files
                      if p.parent != root})
    entries = []
    for p in files:
        relp = p.relative_to(root).as_posix()
        entries.append({"path": relp, "label": None})

    manifest = {
        "root": root.as_posix(),
        "note": "site pool (unlabeled). label=null 은 의도된 것 — 부모폴더명으로 오염시키지 마라.",
        "files": entries,
    }
    out_root = rel(Config.OUT_ROOT)
    out_root.mkdir(parents=True, exist_ok=True)
    mpath = out_root / "site_pool.json"
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    k_hat = int(Config.K_HAT)
    mcs, ms = recommend_dial(n, k_hat)
    if int(Config.MCS_OVERRIDE) > 0:
        mcs = int(Config.MCS_OVERRIDE)
    if int(Config.MS_OVERRIDE) > 0:
        ms = int(Config.MS_OVERRIDE)
    search = dial_search_range(n, k_hat)

    per_class = n / k_hat
    print(f"\n[pool] n = {n:,} 장   |   하위폴더 {len(subdirs)}개"
          + (f"  (예: {', '.join(subdirs[:4])}{' ...' if len(subdirs) > 4 else ''})" if subdirs else ""))
    print(f"[pool] k_hat = {k_hat}  ->  클래스당 예상 {per_class:.1f} 장")
    print(f"\n[dial] ★ 권장  mcs = {mcs} , ms = {ms}   (= n/k 의 {100*mcs/per_class:.1f}%)")
    print(f"[dial] 좁은 sweep 범위(step3 에서 사용): mcs in {search}")
    print("\n  근거: mcs ~= (n/k)*0.10. mcs6 이 정상 작동한 pool 은 전부 9.9~11.4% 대역이었고,")
    print("        3.0% 로 어긋난 pool 에서는 결론이 통째로 뒤집혔다(적응이 '품질 희생'으로 잘못 판정).")
    if per_class < 30:
        print("\n  ⚠ 클래스당 예상 장수가 30 미만이다. 클러스터가 형성되기 어렵다 —")
        print("     이미지를 더 모으거나 k_hat 이 과대추정은 아닌지 현업과 확인해라.")

    payload = {
        "n_images": n, "k_hat": k_hat, "per_class_est": round(per_class, 1),
        "manifest": str(mpath.relative_to(REPO)),
        "image_root": root.as_posix(), "subdirs": subdirs[:50],
        "dial": {"mcs": mcs, "ms": ms, "method": "leaf", "eps": 0.06},
        "dial_search_mcs": search,
        "config": cfg,
    }
    save_result(out_root, "step0", payload)

    print("\n다음:  python site/step1_zeroshot.py")
    print("       (step0 결과의 다이얼을 자동으로 물려 쓴다)")
    print(f"\n[OUT] {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
