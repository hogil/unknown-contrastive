#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 5 — ★ 최종 산출물: 시간축 **신규불량 발생 감지**.

  python deploy/step5_temporal.py

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
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _site_common import (add_path, REPO, banner, check_inputs, die, env, rel,  # noqa: E402
                          save_result, run_root, show_config, show_images, step_dir)


from config import Cluster, Composite, Paths, Runtime, Temporal  # noqa: E402


class Config:
    """★ 설정은 deploy/config.py 한 곳에서 관리한다. 여기는 참조만."""
    OUT_BASE = Paths.OUT_ROOT          # runs/site (캐시·latest.txt 가 여기)
    OUT_ROOT = Paths.OUT_ROOT          # main() 에서 타임스탬프 run 으로 교체된다
    BACKBONE = Paths.BACKBONE
    DEVICE = Runtime.DEVICE
    BATCH = Runtime.BATCH
    PROJ = Temporal.PROJ
    TIME_MODE = Temporal.TIME_MODE
    TIME_BATCH_SIZE = Temporal.TIME_BATCH_SIZE
    TIME_DIR_REGEX = Temporal.TIME_DIR_REGEX
    BATCH_MIN_GROUP = Temporal.BATCH_MIN_GROUP
    N_CALIB = Temporal.N_CALIB
    N_BG = Temporal.N_BG
    P_LIST = Temporal.P_LIST
    K_LIST = Temporal.K_LIST
    MIN_SIZE_PCT = Temporal.MIN_SIZE_PCT


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
    Config.OUT_ROOT = str(run_root(Config.OUT_BASE, create=False))
    cfg = show_config(Config)

    s0 = load("step0")
    if not s0:
        die("step0 결과가 없다. 먼저:  python deploy/step0_prepare.py")
    check_inputs(Config.BACKBONE, s0["image_root"])
    show_images(s0["image_root"], s0.get("manifest", ""), getattr(Config, "EXTS", ""))

    add_path(REPO)
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

    # ── 배치 다이얼 (pool 다이얼과 별개) ──────────────────────────────────
    med_batch = sorted(len(idx) for _, idx in batches)[len(batches) // 2]
    if int(Config.BATCH_MIN_GROUP) > 0:
        mcs = max(3, int(Config.BATCH_MIN_GROUP))
    else:
        mcs = max(3, int(round(med_batch * 0.05)))
    ms = max(2, mcs // 4)
    print(f"\n[dial] 배치 단위: mcs={mcs} ms={ms}  (배치 중앙값 {med_batch} 장의 "
          f"{100*mcs/max(1,med_batch):.1f}%)")
    print(f"       pool 전체 다이얼(mcs={s0['dial']['mcs']}) 과 별개다 — 배치가 훨씬 작다.")
    if med_batch < mcs * 3:
        print(f"  ⚠ 배치당 {med_batch} 장은 mcs={mcs} 에 비해 작다. 클러스터가 잘 안 잡힌다 —")
        print(f"     SITE_TIME_BATCH_SIZE 를 키우거나 SITE_BATCH_MIN_GROUP 을 줄여라.")

    n_cal, n_bg = int(Config.N_CALIB), int(Config.N_BG)
    if len(batches) < n_cal + n_bg + 1:
        die(f"배치가 부족하다: {len(batches)}개. "
            f"캘리브레이션 {n_cal} + 배경 {n_bg} + 관측 1 이상 필요.\n"
            f"  SITE_N_CALIB / SITE_N_BG 를 줄이거나 데이터 기간을 늘려라.")

    # ── 임베딩 (1회) ──────────────────────────────────────────────────────
    # ★ grouping_deploy 와 **같은 경로** (device 진단 / 캐시 / 스레드 디코드 /
    #   UC_PALETTE_MASK 존중). forward_features -> GAP — bb(x) 는 head norm 이 껴서 다르다.
    dev = gd.resolve_device(Config.DEVICE)
    bb = gd.maybe_channels_last(gd.load_backbone(str(rel(Config.BACKBONE)), dev))
    cache = gd.load_cache_for(s0.get("cache_dir", "") if getattr(Config, "CACHE", True) else "",
                              paths)
    raw = gd.embed_backbone(paths, bb, dev, int(Config.BATCH), cache, "feat")

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
        if len(idx) < mcs * 2 or len(idx) < 5:
            per_batch.append({"name": name, "n": len(idx), "clusters": []})
            continue
        # ★ method/eps 를 하드코딩하면 config.Cluster.METHOD/EPS 를 켜도 시간축
        #   격자만 leaf/0.06 으로 돌아 배포 다이얼과 어긋난다 (260729 감사).
        pred = gd.hdbscan_predict(zb, mcs, ms, str(Cluster.METHOD), float(Cluster.EPS))
        cl = []
        _gidx = np.asarray(idx)
        for c in sorted(set(pred[pred >= 0].tolist())):
            m = pred == c
            cen = zb[m].mean(axis=0)
            cen = cen / (np.linalg.norm(cen) + 1e-12)
            # ★ 멤버의 **전역 인덱스**를 남긴다. 예전엔 centroid/size 만 저장해서
            #   알람이 떠도 "그 클러스터가 어떤 이미지인지" 되돌릴 수 없었다 —
            #   그래서 step5 는 composite/대표이미지를 만들 수 없었다 (260730).
            cl.append({"id": int(c), "size": int(m.sum()), "centroid": cen,
                       "members": _gidx[m].tolist()})
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

    # ★ 임계용 유사도는 **leave-one-batch-out** 으로 잰다.
    #   예전엔 캘리브레이션 클러스터를 REF 전체(자기 자신 포함)와 비교했다. centroid 는
    #   단위벡터라 자기와의 코사인이 정확히 1.0 이고, max 를 취하니 **표본이 전부 1.0** 이
    #   됐다. 그래서 어떤 퍼센타일을 잡아도 임계가 1.0 이 되고, 관측 클러스터는 거의 항상
    #   1.0 미만이라 **전부 신규로 판정**됐다 — 감지기가 늘 알람을 울리는 상태였다.
    #   실측(260731): threshold 1.0 / FAR 1.0·0.75·0.5 / 신규 클러스터 36개.
    #   운영 시점의 질문은 "**앞선 배치들**로 만든 REF 에 이 배치의 클러스터가 얼마나
    #   닮았나" 이므로, 캘리브레이션에서도 자기 배치를 뺀 REF 와 비교해야 같은 질문이 된다.
    _cal_batches = [b for b in per_batch[:n_cal] if b["clusters"]]
    calib_sims = []
    for i, b in enumerate(_cal_batches):
        others = [c for j, ob in enumerate(_cal_batches) if j != i for c in ob["clusters"]]
        if not others:
            continue
        Ro = np.stack([c["centroid"] for c in others])
        calib_sims += [float((Ro @ c["centroid"]).max()) for c in b["clusters"]]
    if not calib_sims:
        die("임계를 만들 수 없다: 클러스터가 나온 캘리브레이션 배치가 "
            f"{len(_cal_batches)}개뿐이다 (2개 이상 필요).\n"
            "  SITE_N_CALIB 를 늘리거나, 배치당 장수를 늘리거나, mcs 를 낮춰라.")
    print(f"[calib] 유사도 표본 {len(calib_sims)}개 (leave-one-batch-out)  "
          f"min={min(calib_sims):.5f} max={max(calib_sims):.5f}")
    if max(calib_sims) > 0.9999:
        print("  ⚠ 유사도 1.0 이 섞여 있다 — 배치 간 중복 이미지가 있는지 확인해라.")
    if len(calib_sims) < 30:
        print(f"  ⚠ 표본이 {len(calib_sims)}개(30 미만)다. 임계는 이 표본의 P 퍼센타일인데,"
              f" 표본이 적으면 그 값이 통계량이 아니라 관측 최솟값에 가까워진다.")
        print("     (실측: 같은 분포에서 N=10 이면 P10 이 0.067 폭으로 흔들리고,"
              " N=120 이면 0.026 으로 준다.)")
        print("     캘리브레이션 배치(SITE_N_CALIB)를 늘리거나 mcs 를 낮춰 클러스터를 늘려라.")

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
                # K 가 배경 배치 수에 비해 크면 FAR 0 이 자동으로 나온다(연속 K 회가 불가능).
                # 그런 셀은 '측정 불가'로 표시해 권장 후보에서 뺀다.
                unreliable = len(bg_f) < 2 * K
                oa = alarms(ob_f)
                grid.append({"P": P, "threshold": round(thr, 6), "size_pct": sp,
                             "min_size": floor, "K": K,
                             "FAR_per_bg_batch": round(far, 3),
                             "n_bg_batches": len(bg_f),
                             "obs_alarm_batches": [obs_slice[i]["name"] for i in oa],
                             "first_alarm": obs_slice[oa[0]]["name"] if oa else None,
                             "far_unreliable": unreliable})

    zero = [g for g in grid if g["FAR_per_bg_batch"] == 0 and not g["far_unreliable"]]
    zero_unrel = [g for g in grid if g["FAR_per_bg_batch"] == 0 and g["far_unreliable"]]
    if zero_unrel:
        print(f"\n  [주의] FAR 0 으로 보이지만 **측정 불가**인 셀 {len(zero_unrel)}개 —"
              f" 배경 배치 {len(bg_slice)}개에 K 가 너무 커서 연속 K 회 자체가 불가능하다.")
        print("         권장 후보에서 제외했다. SITE_N_BG 를 늘려라 (K 최대값의 2배 이상).")
    print("\n" + "=" * 78)
    print(f"[격자] FAR = 배경 {len(bg_slice)} 배치 기준 오경보/배치")
    print("=" * 78)
    print(f"  {'P':<5} {'size_pct':<9} {'min_size':<9} {'K':<4} {'FAR':<8} first_alarm")
    for g in grid:
        mark = (" (측정불가)" if g["far_unreliable"] and g["FAR_per_bg_batch"] == 0
                else (" ★" if g["FAR_per_bg_batch"] == 0 else ""))
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

    # ── ★ 신규 클러스터의 대표이미지 + composite ──────────────────────────
    #   예전엔 step5 가 FAR/lag 숫자만 냈다. 알람이 떠도 **그게 무슨 불량인지 볼 수단이
    #   없었다** — `docs/ABSOLUTE_RULES.md` §7("composite 와 실제 멤버 이미지로 눈으로
    #   확인해야 최종 판정")을 만족할 수 없는 상태였다 (260730).
    #   권장 운영점(rec) 기준으로 관측 구간에서 신규로 걸린 클러스터를 그대로 산출한다.
    novel_out = step_dir(Config.OUT_ROOT, "step5_novel")
    thr_rec, floor_rec = float(rec["threshold"]), int(rec["min_size"])
    novel_report = []
    n_made = 0
    for b in obs_slice:
        for c in b["clusters"]:
            if not (max_sim(c) < thr_rec and c["size"] >= floor_rec):
                continue
            # ★ lot 수도 넣는다 (파일명 첫 `_` 앞이 lot id). 신규 클러스터가
            #   한 lot 에서만 나왔는지 여러 lot 에 걸쳤는지가 대응을 가른다.
            _lots = {Path(paths_s[i]).stem.split("_", 1)[0] for i in c["members"]}
            tag = f"{b['name']}_c{c['id']:03d}_l{len(_lots)}_w{c['size']}"
            gdir = novel_out / tag
            (gdir / "representatives").mkdir(parents=True, exist_ok=True)
            # 중심에 가까운 순 (composite 상한도 이 순서로 자른다)
            cen = c["centroid"]
            mem = sorted(c["members"], key=lambda i: float(np.linalg.norm(z[i] - cen)))
            # ★ 하드코딩 12 였다 — config 를 고쳐도 step5 만 12장이었다.
            _reps = int(getattr(Composite, "MAX_REPS", 12))
            for rank, i in enumerate(mem[:_reps] if _reps > 0 else mem, 1):
                src = Path(paths_s[i])
                if src.exists():
                    # near<N> = 그룹 중심에서 가까운 순위 (mem 이 그 순으로 정렬돼 있다)
                    shutil.copy2(src, gdir / "representatives" / f"near{rank:02d}_{src.name}")
            mx = int(Composite.MAX_MEMBERS)
            use = mem if mx <= 0 else mem[:mx]
            try:
                from composite import write_group_composites
            except Exception:
                from deploy.composite import write_group_composites
            try:
                write_group_composites([paths_s[i] for i in use],
                                       gdir / "composites", prefix=f"{tag}_")
                n_made += 1
                print(f"  [novel] {tag}  max_sim={max_sim(c):.4f} < {thr_rec:.4f}  "
                      f"멤버 {c['size']} 중 {len(use)}장으로 composite", flush=True)
            except Exception as e:
                import traceback
                print(f"  [warn] ★ {tag} composite 실패: {type(e).__name__}: {e}", flush=True)
                if n_made == 0:
                    traceback.print_exc()
            novel_report.append({"batch": b["name"], "cluster": c["id"], "size": c["size"],
                                 "n_lots": len(_lots), "lots": sorted(_lots)[:20],
                                 "max_sim": round(max_sim(c), 6), "dir": str(gdir.name)})
    if novel_report:
        print(f"\n  ★ 신규 클러스터 {len(novel_report)}개 산출 (composite {n_made}개) -> {novel_out}")
        print("     ※ 알람의 근거다. 이 composite 와 representatives 를 눈으로 확인해야")
        print("        최종 판정이다 (ABSOLUTE_RULES §7 — 무라벨 지표만으로는 판정 불가).")
    else:
        print(f"\n  신규로 걸린 클러스터가 없다 (관측 구간 {len(obs_slice)} 배치). -> {novel_out}")

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
                 "grid": grid, "recommended": rec,
                 # ★ 알람의 근거 — 신규 클러스터별 산출 폴더까지 남긴다.
                 "novel_clusters": novel_report,
                 "novel_dir": str(novel_out),
                 "config": cfg})
    print(f"\n[OUT] {rel(Config.OUT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
