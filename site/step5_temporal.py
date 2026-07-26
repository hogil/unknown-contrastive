#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 5 — ★ 최종 산출물: 시간축 **신규불량 발생 감지**.

  python site/step5_temporal.py

정적 grouping 품질(step1~3)이 아니라 **"새 불량이 나타났을 때 알람이 뜨는가"** 를 만든다.
정적 지표만 보면 오경보 5배 격차를 놓친다(실측).

절차:
  1. 이미지를 **시간 배치**로 나눈다 (폴더명 = 날짜 권장, 없으면 mtime).
  2. 각 배치를 같은 다이얼로 클러스터링.
  3. **캘리브레이션 구간**(초기 배치들)에서 REF centroid 집합을 고정하고,
     "기존 그룹과의 최대 cosine" 분포의 P 퍼센타일을 임계로 잡는다.
  4. 이후 배치에서 임계 미만 + 크기 ≥ floor 인 클러스터가 **K 배치 연속** 나오면 알람.
  5. 순수 배경 구간(신규불량 없다고 아는 구간)에서 **FAR(오경보/배치)** 를 잰다.

★ 권장 운영점 (실측): P=10 / m_min = REF 크기 50th 퍼센타일 / K=2  -> FAR 0
  - **P1/P2 는 쓰지 마라.** 캘리브레이션 표본이 적으면 1st 퍼센타일은 통계량이 아니라
    관측 최솟값이고, 캘리브레이션 창을 한 배치 옮기면 임계가 전체 범위의 절반만큼 튄다.
    P10/P20 은 5~15배 안정적이다.
  - 절대값 임계 금지. 반드시 **자기 pool 의 분포 퍼센타일**로 잡아라.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _site_common import (REPO, banner, check_inputs, die, env, rel,  # noqa: E402
                          save_result, show_config)


# ═══════════════════════════════════════════════════════════════════════════
class Config:
    """환경변수 우선, 없으면 default. 경로는 전부 프로젝트 루트 기준 상대경로."""

    OUT_ROOT = env("SITE_OUT_ROOT", "runs/site")
    BACKBONE = env("SITE_BACKBONE", "weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth")

    # 사용할 head. 비우면 frozen backbone 만. 콤마 구분(앙상블).
    # 보통 step3 최종 승자 체크포인트를 넣는다. 비워두면 step3 -> step2 -> frozen 순 자동 탐색.
    PROJ = env("SITE_TEMPORAL_PROJ", "")

    DEVICE = env("SITE_DEVICE", "cuda")
    BATCH = env("SITE_BATCH", 32)

    # ── 시간 배치 ──────────────────────────────────────────────────────────
    # by_dir  : 이미지의 상위 폴더명을 배치 키로 (예: 20260701/ 20260702/ ...) — 권장
    # by_mtime: 파일 수정시각을 정렬해 BATCH_SIZE 장씩 끊음
    TIME_MODE = env("SITE_TIME_MODE", "by_dir")
    TIME_BATCH_SIZE = env("SITE_TIME_BATCH_SIZE", 200)   # by_mtime 일 때만
    # 폴더명에서 날짜만 뽑는 정규식 (첫 그룹이 정렬 키). 안 맞으면 폴더명 전체 사용.
    TIME_DIR_REGEX = env("SITE_TIME_DIR_REGEX", r"(\d{6,8})")

    # ── 운영점 ────────────────────────────────────────────────────────────
    N_CALIB = env("SITE_N_CALIB", 4)       # 앞 N개 배치로 REF + 임계 캘리브레이션
    N_BG = env("SITE_N_BG", 4)             # 그 다음 N개는 "신규불량 없음"으로 아는 구간 -> FAR 측정
    P_LIST = env("SITE_P_LIST", "10,20")   # 퍼센타일 후보 (★ 1,2 는 불안정하니 쓰지 마라)
    K_LIST = env("SITE_K_LIST", "1,2,3")   # persistence 후보
    MIN_SIZE_PCT = env("SITE_MIN_SIZE_PCT", "25,50,75")  # REF 크기 분포의 퍼센타일 -> 크기 하한
# ═══════════════════════════════════════════════════════════════════════════


def load(step: str) -> dict:
    p = rel(Config.OUT_ROOT) / f"{step}_result.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def resolve_proj() -> list[str]:
    if str(Config.PROJ).strip():
        return [p.strip() for p in str(Config.PROJ).split(",") if p.strip()]
    s3 = load("step3")
    if s3.get("final_winner"):
        w, r1, r2 = s3["final_winner"], s3.get("round1", {}), s3.get("round2", {})
        out = []
        if w in r1:
            out.append(str(Path(r1[w]["ckpt_dir"]) / r1[w]["selected"]["ckpt"]))
        for g in r2.get(w, []):
            out.append(str(Path(g["ckpt_dir"]) / g["selected"]["ckpt"]))
        if out:
            print(f"[proj] step3 승자({w}) 체크포인트 {len(out)}개 자동 사용")
            return out
    s2 = load("step2")
    if s2.get("per_seed"):
        out = [str(Path(i["ckpt_dir"]) / i["selected"]["ckpt"]) for i in s2["per_seed"].values()]
        print(f"[proj] step2 체크포인트 {len(out)}개 자동 사용")
        return out
    print("[proj] head 없음 -> frozen backbone 만 사용")
    return []


def time_key(p: Path, root: Path) -> str:
    if Config.TIME_MODE == "by_dir":
        relp = p.relative_to(root)
        d = relp.parts[0] if len(relp.parts) > 1 else "_flat"
        m = re.search(str(Config.TIME_DIR_REGEX), d)
        return m.group(1) if m else d
    return ""


def build_batches(paths: list[Path], root: Path) -> list[tuple[str, list[int]]]:
    if Config.TIME_MODE == "by_dir":
        g = defaultdict(list)
        for i, p in enumerate(paths):
            g[time_key(p, root)].append(i)
        if len(g) <= 1:
            die("by_dir 로 배치를 못 나눴다 (폴더가 하나뿐).\n"
                "  SITE_TIME_MODE=by_mtime 으로 바꾸거나, 날짜별 하위폴더 구조로 두어라.")
        return sorted(g.items(), key=lambda kv: kv[0])
    order = sorted(range(len(paths)), key=lambda i: paths[i].stat().st_mtime)
    bs = int(Config.TIME_BATCH_SIZE)
    return [(f"t{n+1:03d}", order[i:i + bs]) for n, i in enumerate(range(0, len(order), bs))]


def main() -> int:
    banner("STEP 5", "시간축 신규불량 감지 (최종 산출물)", "①~④ 중 채택된 모델로 운영")
    cfg = show_config(Config)

    s0 = load("step0")
    if not s0:
        die("step0 결과가 없다. 먼저:  python site/step0_prepare.py")
    mcs, ms = s0["dial"]["mcs"], s0["dial"]["ms"]
    check_inputs(Config.BACKBONE, s0["image_root"], s0["k_hat"])

    sys.path.insert(0, str(REPO))
    import numpy as np
    import torch
    import torch.nn.functional as F
    import grouping_deploy as gd

    root = Path(s0["image_root"])
    paths_s, _ = gd.collect_pool(str(rel(s0["manifest"])))
    paths = [Path(p) for p in paths_s]
    batches = build_batches(paths, root)
    print(f"\n[time] {len(batches)} 배치  (mode={Config.TIME_MODE})")
    for name, idx in batches[:12]:
        print(f"    {name}: {len(idx)} 장")
    if len(batches) > 12:
        print(f"    ... 외 {len(batches)-12} 배치")

    n_cal, n_bg = int(Config.N_CALIB), int(Config.N_BG)
    if len(batches) < n_cal + n_bg + 1:
        die(f"배치가 부족하다: {len(batches)}개. "
            f"캘리브레이션 {n_cal} + 배경 {n_bg} + 관측 1 이상 필요.\n"
            f"  SITE_N_CALIB / SITE_N_BG 를 줄이거나 데이터 기간을 늘려라.")

    # ── 임베딩 (1회) ──────────────────────────────────────────────────────
    dev = Config.DEVICE
    bb = gd.load_backbone(str(rel(Config.BACKBONE)), dev)
    tf = gd.T.Compose([gd.T.Resize((384, 384), interpolation=gd.T.InterpolationMode.BILINEAR),
                       gd.T.ToTensor(),
                       gd.T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    bs = int(Config.BATCH)
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            x = torch.stack([tf(gd.Image.open(p).convert("RGB")) for p in paths[i:i + bs]]).to(dev)
            f = bb(x)
            feats.append((f.mean(dim=(2, 3)) if f.ndim == 4 else f).float().cpu())
            if i % (bs * 20) == 0:
                print(f"  [feat] {i}/{len(paths)}", flush=True)
    raw = torch.cat(feats)

    projs = resolve_proj()
    if projs:
        zs = []
        for cp in projs:
            pr = gd.load_proj(str(rel(cp)), dev)
            with torch.no_grad():
                zs.append(F.normalize(pr(raw.to(dev)), dim=1).cpu())
        z = F.normalize(torch.cat(zs, dim=1), dim=1).numpy()
    else:
        z = F.normalize(raw, dim=1).numpy()
    print(f"[embed] {z.shape[0]} x {z.shape[1]}  (head {len(projs)}개)")

    # ── 배치별 클러스터 centroid ──────────────────────────────────────────
    per_batch = []
    for name, idx in batches:
        zb = z[idx]
        if len(idx) < mcs * 2:
            per_batch.append({"name": name, "n": len(idx), "clusters": []})
            continue
        pred = gd.hdbscan_predict(zb, mcs, ms, "leaf", 0.06)
        cl = []
        for c in sorted(set(pred[pred >= 0].tolist())):
            m = pred == c
            cen = zb[m].mean(axis=0)
            cen = cen / (np.linalg.norm(cen) + 1e-12)
            cl.append({"id": int(c), "size": int(m.sum()), "centroid": cen})
        per_batch.append({"name": name, "n": len(idx), "clusters": cl})
        print(f"  [{name}] n={len(idx):<5} k={len(cl):<4} noise={(pred==-1).sum()/len(idx)*100:5.1f}%")

    # ── REF (캘리브레이션 구간에서 고정) ──────────────────────────────────
    ref = [c for b in per_batch[:n_cal] for c in b["clusters"]]
    if not ref:
        die("캘리브레이션 구간에서 클러스터가 하나도 안 나왔다. "
            "다이얼(mcs)이 너무 큰지, 배치당 장수가 너무 적은지 확인해라.")
    R = np.stack([c["centroid"] for c in ref])
    ref_sizes = sorted(c["size"] for c in ref)
    print(f"\n[REF] 캘리브레이션 {n_cal} 배치 -> centroid {len(ref)}개, "
          f"크기 분포 min={ref_sizes[0]} p25={ref_sizes[len(ref_sizes)//4]} "
          f"p50={ref_sizes[len(ref_sizes)//2]} max={ref_sizes[-1]}")

    def max_sim(c):
        return float((R @ c["centroid"]).max())

    calib_sims = [max_sim(c) for b in per_batch[:n_cal] for c in b["clusters"]]
    print(f"[calib] 유사도 표본 {len(calib_sims)}개  "
          f"min={min(calib_sims):.5f} max={max(calib_sims):.5f}")
    if len(calib_sims) < 30:
        print("  ⚠ 표본이 30 미만이다. 퍼센타일 임계가 불안정할 수 있다 — "
              "캘리브레이션 배치를 늘려라.")

    def size_floor(pct):
        return ref_sizes[min(len(ref_sizes) - 1, int(round(pct / 100 * (len(ref_sizes) - 1))))]

    # ── 격자 평가 ─────────────────────────────────────────────────────────
    Ps = [float(x) for x in str(Config.P_LIST).split(",") if x.strip()]
    Ks = [int(x) for x in str(Config.K_LIST).split(",") if x.strip()]
    SPs = [float(x) for x in str(Config.MIN_SIZE_PCT).split(",") if x.strip()]
    bg_slice = per_batch[n_cal:n_cal + n_bg]
    obs_slice = per_batch[n_cal + n_bg:]

    grid = []
    for P in Ps:
        thr = float(np.percentile(calib_sims, P))
        for sp in SPs:
            floor = size_floor(sp)
            # 배치별 "신규 후보" 여부
            def novel_flags(slc):
                return [any(max_sim(c) < thr and c["size"] >= floor for c in b["clusters"])
                        for b in slc]
            bg_f, ob_f = novel_flags(bg_slice), novel_flags(obs_slice)
            for K in Ks:
                def alarms(flags):
                    out, streak = [], 0
                    for i, f in enumerate(flags):
                        streak = streak + 1 if f else 0
                        if streak >= K:
                            out.append(i)
                    return out
                far = len(alarms(bg_f)) / max(1, len(bg_f))
                oa = alarms(ob_f)
                grid.append({"P": P, "threshold": round(thr, 6), "size_pct": sp,
                             "min_size": floor, "K": K,
                             "FAR_per_bg_batch": round(far, 3),
                             "n_bg_batches": len(bg_f),
                             "obs_alarm_batches": [obs_slice[i]["name"] for i in oa],
                             "first_alarm": obs_slice[oa[0]]["name"] if oa else None})

    zero = [g for g in grid if g["FAR_per_bg_batch"] == 0]
    print("\n" + "=" * 78)
    print(f"[격자] FAR = 배경 {len(bg_slice)} 배치 기준 오경보/배치")
    print("=" * 78)
    print(f"  {'P':<5} {'size_pct':<9} {'min_size':<9} {'K':<4} {'FAR':<8} first_alarm")
    for g in grid:
        mark = " ★" if g["FAR_per_bg_batch"] == 0 else ""
        print(f"  {g['P']:<5} {g['size_pct']:<9} {g['min_size']:<9} {g['K']:<4} "
              f"{g['FAR_per_bg_batch']:<8} {g['first_alarm'] or '-'}{mark}")

    rec = None
    if zero:
        # 같은 FAR 0 이면 가장 느슨한 운영점(작은 K -> 작은 size floor -> 큰 P)을 고른다.
        rec = sorted(zero, key=lambda g: (g["K"], g["size_pct"], -g["P"]))[0]
        print(f"\n  ★★ 권장 운영점: P={rec['P']} / min_size={rec['min_size']}"
              f"(REF {rec['size_pct']}th pct) / K={rec['K']}  ->  FAR 0")
        print(f"      관측 구간 알람: {rec['obs_alarm_batches'] or '없음'}")
    else:
        best = min(grid, key=lambda g: (g["FAR_per_bg_batch"], g["K"]))
        print(f"\n  ⚠ FAR 0 조합이 없다. 최선: P={best['P']} min_size={best['min_size']} "
              f"K={best['K']} -> FAR {best['FAR_per_bg_batch']}/배치")
        print("    대응: (a) K 를 늘린다(감지가 늦어짐) (b) 캘리브레이션 배치를 늘린다")
        print("          (c) step3 로 임베딩을 개선한다 — 우리 실측에선 이게 가장 크게 들었다")
        rec = best

    print("\n  주의: 여기 FAR 은 '배경 구간에 신규불량이 없다'는 가정 위에서 잰 값이다.")
    print("        그 구간에 실제 신규불량이 있었다면 FAR 이 과대평가된다 — 현업과 확인해라.")

    save_result(rel(Config.OUT_ROOT), "step5",
                {"dial": {"mcs": mcs, "ms": ms}, "proj": projs,
                 "n_batches": len(batches),
                 "batches": [{"name": b["name"], "n": b["n"], "k": len(b["clusters"])} for b in per_batch],
                 "calib_batches": [b["name"] for b in per_batch[:n_cal]],
                 "bg_batches": [b["name"] for b in bg_slice],
                 "obs_batches": [b["name"] for b in obs_slice],
                 "n_calib_sims": len(calib_sims),
                 "ref_size_percentiles": {"min": ref_sizes[0], "p50": ref_sizes[len(ref_sizes)//2],
                                          "max": ref_sizes[-1]},
                 "grid": grid, "recommended": rec, "config": cfg})
    print(f"\n[OUT] {rel(Config.OUT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
