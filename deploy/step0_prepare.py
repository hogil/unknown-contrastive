#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 0 — 사내 이미지 폴더 -> manifest + pool 기하 리포트 + 권장 다이얼.

  python deploy/step0_prepare.py

여기서 정한 다이얼(mcs/ms)을 이후 step1~5 가 전부 물려 쓴다.

★ 불량 종수(k)는 입력하지 않는다. 모르는 게 전제이고, HDBSCAN 을 쓰는 이유가 k-free 라서다.
  다이얼은 **"몇 장 이상 뭉쳐야 하나의 그룹으로 볼 것인가"**(MIN_GROUP_SIZE)로 정한다 —
  이건 클래스 수가 아니라 **보고 가치가 있는 최소 그룹 크기**라 k 를 몰라도 답할 수 있다.
  그것도 정하기 싫으면 AUTO_DIAL=1 로 두면 bootstrap 안정성으로 라벨·k 없이 골라준다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _site_common import (add_path, REPO, banner, dial_from_min_group, dial_scan_range,  # noqa: E402
                          die, env, pick_dial_by_stability, rel, save_result,
                          show_config, show_images)


from config import Cluster, Paths, Runtime  # noqa: E402


class Config:
    """★ 설정은 deploy/config.py 한 곳에서 관리한다. 여기는 참조만."""
    IMAGE_ROOT = Paths.IMAGE_ROOT
    POOL_MANIFEST = Paths.POOL_MANIFEST
    EXTS = Paths.EXTS
    OUT_ROOT = Paths.OUT_ROOT
    BACKBONE = Paths.BACKBONE
    DEVICE = Runtime.DEVICE
    BATCH = Runtime.BATCH
    MIN_GROUP_SIZE = Cluster.MIN_GROUP_SIZE
    AUTO_DIAL = Cluster.AUTO_DIAL
    MCS_OVERRIDE = Cluster.MCS
    MS_OVERRIDE = Cluster.MS


def resolve_dial(n: int, pool_path: str):
    """k 를 쓰지 않고 (mcs, ms) 결정.
       AUTO_DIAL=0 -> MIN_GROUP_SIZE 를 그대로 min_cluster_size 로 사용
       AUTO_DIAL=1 -> 그 주변을 스캔해 bootstrap 안정성 최대인 값 선택 (라벨/k 무관)
       SITE_MCS / SITE_MS 를 주면 그게 최우선."""
    scan_rows = []
    mcs, ms = dial_from_min_group(int(Config.MIN_GROUP_SIZE))
    if Config.AUTO_DIAL:
        import torch
        import torch.nn.functional as F
        add_path(REPO)
        import grouping_deploy as gd
        bb_p = rel(Config.BACKBONE)
        if not bb_p.exists():
            die(f"AUTO_DIAL=1 이면 backbone 이 필요하다: {bb_p}\n"
                f"  또는 AUTO_DIAL=0 으로 두고 MIN_GROUP_SIZE 만 정해라.")
        paths, _ = gd.collect_pool(str(rel(pool_path)))
        bb = gd.load_backbone(str(bb_p), Config.DEVICE)
        tf = gd.T.Compose([gd.T.Resize((384, 384), interpolation=gd.T.InterpolationMode.BILINEAR),
                           gd.T.ToTensor(),
                           gd.T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        bs, fs = int(Config.BATCH), []
        print(f"[auto-dial] 임베딩 계산 중 ({len(paths):,} 장)...", flush=True)
        with torch.no_grad():
            for i in range(0, len(paths), bs):
                x = torch.stack([tf(gd.Image.open(q).convert("RGB"))
                                 for q in paths[i:i + bs]]).to(Config.DEVICE)
                fs.append(bb.forward_features(x).mean(dim=(2, 3)).float().cpu())
        z = F.normalize(torch.cat(fs), dim=1).numpy().astype("float32")
        scan = dial_scan_range(n, int(Config.MIN_GROUP_SIZE))
        print(f"[auto-dial] 스캔 mcs in {scan}  (bootstrap 안정성 최대 선택, 라벨/k 미사용)")
        mcs, ms, scan_rows = pick_dial_by_stability(
            z, scan, gd.hdbscan_predict, gd.per_group_stability)
        print(f"  {'mcs':<7}{'ms':<6}{'k':<6}{'noise%':<10}stability")
        for r in scan_rows:
            mark = "  <-- 선택" if r["mcs"] == mcs else ""
            print(f"  {r['mcs']:<7}{r['ms']:<6}{r['k']:<6}{r['noise_pct']:<10}"
                  f"{r['stability']}{mark}")
    if int(Config.MCS_OVERRIDE) > 0:
        mcs = int(Config.MCS_OVERRIDE)
    if int(Config.MS_OVERRIDE) > 0:
        ms = int(Config.MS_OVERRIDE)
    return mcs, ms, scan_rows


def main() -> int:
    banner("STEP 0", "manifest 생성 + pool 기하 -> 권장 다이얼")
    cfg = show_config(Config)
    show_images(Config.IMAGE_ROOT, Config.POOL_MANIFEST, Config.EXTS)

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
        mcs, ms, scan_rows = resolve_dial(n, str(mp))
        print(f"\n[pool] 기존 manifest 사용: {mp}")
        print(f"[pool] n = {n:,} 장   root = {root}")
        out_root = rel(Config.OUT_ROOT)
        save_result(out_root, "step0", {
            "n_images": n, "min_group_size": int(Config.MIN_GROUP_SIZE),
            "auto_dial": bool(Config.AUTO_DIAL), "dial_scan": scan_rows,
            "manifest": str(mp.relative_to(REPO)) if mp.is_relative_to(REPO) else str(mp),
            "image_root": root.as_posix(), "subdirs": [],
            "dial": {"mcs": mcs, "ms": ms, "method": "leaf", "eps": 0.06},
            "config": cfg})
        print("\n다음:  python deploy/step1_zeroshot.py")
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

    mcs, ms, scan_rows = resolve_dial(n, str(mpath))
    print(f"\n[pool] n = {n:,} 장   |   하위폴더 {len(subdirs)}개"
          + (f"  (예: {', '.join(subdirs[:4])}{' ...' if len(subdirs) > 4 else ''})" if subdirs else ""))

    payload = {
        "n_images": n, "min_group_size": int(Config.MIN_GROUP_SIZE),
        "auto_dial": bool(Config.AUTO_DIAL), "dial_scan": scan_rows,
        "manifest": str(mpath.relative_to(REPO)) if mpath.is_relative_to(REPO) else str(mpath),
        "image_root": root.as_posix(), "subdirs": subdirs[:50],
        "dial": {"mcs": mcs, "ms": ms, "method": "leaf", "eps": 0.06},
        "config": cfg,
    }
    save_result(out_root, "step0", payload)

    print("\n다음:  python deploy/step1_zeroshot.py")
    print("       (step0 결과의 다이얼을 자동으로 물려 쓴다)")
    print(f"\n[OUT] {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
