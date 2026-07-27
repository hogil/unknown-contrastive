#!/usr/bin/env python3
"""Temporal novelty-detection metrics — detection lag / FAR / min-detectable size.

배경: 기존 P1~P4/ARI 는 전부 정적("한 풀을 얼마나 잘 묶었나")이다. 이 스크립트는 시간축
지표(감지 지연, 그룹-수준 오경보율)를 처음으로 정의·측정한다. 시뮬 설계는
scripts/make_temporal_novelty_pools.py (배치 manifest) 참조. 임베딩은
scripts/run_temporal_novelty_embeddings.py 로 1회 캐시된 것을 슬라이싱만 한다.

다이얼: HDBSCAN(min_cluster_size=6, min_samples=3, cluster_selection_method="leaf",
cluster_selection_epsilon=0.06, metric="euclidean") — 팀리드 지시대로 고정, 배치마다
raw L2 임베딩 위에서 재클러스터링(다이얼 자체는 절대 조정하지 않음).

★ "새 그룹 탄생" 판정 (라벨 없이, 그 run 자신의 분포로 임계 결정):
  1. REF = t=4(calibration 종료) 누적 풀 클러스터링의 centroid 집합 — 1회 고정, 이후 안 바뀜.
  2. calibration 분포 = t=1,2,3 클러스터 각각을 REF 에 cosine 매칭한 유사도 값들
     (즉 "아직 데이터 적은 초기 클러스터가 최종 REF 와 자연히 얼마나 다른가"의 noise floor).
     THRESH = percentile(calibration 분포, P)  — P 는 sweep (1/2/5/10/20), 절대값 아님.
  3. 매 t(>=5) 마다 클러스터 c 를 REF 최근접과 비교: cos(c, REF) < THRESH 이고
     size(c) >= M_MIN 이면 "candidate". M_MIN 도 이 run 자신의 REF 클러스터 크기 분포의
     percentile 로 sweep(0/25/50/75/100 — 0=관측 최소, 100=관측 최대, 항상 mcs=6 이하로는
     안 내려감) — 절대값 아님(팀리드 260726: FAR 원인이 size~7 근처 소형 클러스터라는 관찰을
     검증하기 위한 축).
  4. candidate 를 배치 간 lineage 로 연결(직전 배치의 활성 lineage 와 cos>=THRESH 면 연장,
     아니면 새 lineage 시작). 연속 K 배치 지속되면 그 t 에서 ALARM.
  5. FAR = calibration 에 쓰이지 않은 held-out 순수-background 구간(t=5..8) 에서 발생한 alarm 수.
     detection lag = novel 주입 시작(t0=9) 부터 alarm 이 실제로 뜬 t 까지의 배치 수(+누적 novel 장수).

라벨은 사후 채점(정적 P1~P4 + alarm 이 정말 novel 이었는지)에만 사용 — 클러스터링/threshold 선택
어디에도 쓰지 않는다.

★ 260726: --sim-name 으로 novel 클래스 변형(unknown_novelty_sim_diagonalsmear 등) 을 선택 —
배경 스트림/다이얼/판정규칙은 전부 동일, novel 클래스만 바뀐 1-축 교차검증.

★ 260726 (leakage-fix v2): --emb-dir/--out-dir 로 v1(result_grouping/temporal_novelty_260726)과
분리된 v2 캐시(result_grouping/temporal_novelty_v2_260726)를 가리킬 수 있다. --arms 로 어느
arm 을 (재)계산할지 선택 — "frozen" 은 champion checkpoint 없이도 지금 바로 계산 가능. 이미
존재하는 report json 이 있으면 이번에 계산 안 한 arm 은 그대로 보존(merge).

★ 260726 (팀리드 재검토): FAR 은 t=5..8(순수 배경 구간)에서만 재므로 **arm 당 1회만 계산**한다
(4개 size 변형에 걸쳐 동일한 값이 반복 저장되던 구조를 고쳤다 — "크기별로 FAR 이 다르다"는
착시를 코드 구조로도 방지). novel_alarms(감지 지연)은 변형마다 다르므로 그대로 변형별 유지.
같은 이유로, 두 개의 novel-class sim(예: DiagonalSmear/RingDots)의 FAR 수치는 **독립 재현이
아니라 같은 배경 구간의 같은 계산**이다 — 검출 하한(min-detectable size)만 두 클래스에서
독립적으로 확인된 것이고 FAR 은 그렇지 않다. 이 캐비앗은 report json 의 top-level "_notes"
필드에도 그대로 기록된다(팀리드 지시: "두 리포트 모두에" 명시).

★ 260726 (z0 대조군): --arms 는 "frozen"/"champion" 외에 임의 이름을 받는다 — "frozen" 은
f0_frozen.npy, 그 외는 전부 f_<arm>.npy 규약(예: "z0_seed1" -> f_z0_seed1.npy). 랜덤 head
대조군(z0)을 champion 과 똑같은 파이프라인으로 채점하기 위함 — z0 도 학습된 champion 과 동일하게
"자기만의 최적 (P,m_min,K)" 를 탐색해야 한다(고정된 champion 의 최적점에 z0 를 끼워 맞추면
공정성 재검증에서 고친 실수를 반복하게 된다).
"""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from _common import resolve_pool  # noqa: E402
from cluster_metrics import capture_metrics  # noqa: E402

TEMPORAL_ROOT = REPO / "data/pools/temporal"
EMB_DIR = REPO / "result_grouping/temporal_novelty_260726"
OUT_DIR = EMB_DIR

HDB = dict(min_cluster_size=6, min_samples=3, cluster_selection_method="leaf", cluster_selection_epsilon=0.06)
EXCLUDE_P1 = {"Normal"}
PERSISTENCE_KS = [1, 2, 3]
THRESH_PERCENTILES = [1, 2, 5, 10, 20]
M_MIN_PERCENTILES = [0, 25, 50, 75, 100]  # 0=REF 관측 최소, 100=REF 관측 최대 (mcs=6 미만은 절대 불가)

FAR_CAVEAT_NOTE = (
    "CAVEAT (260726, team-lead review): far_grid is measured ONLY on t=5..8 (pure background), "
    "which is byte-identical across every novel-class sim variant sharing this batch design "
    "(confirmed via dict-equality across e.g. unknown_novelty_sim_v2_diagonalsmear and "
    "unknown_novelty_sim_v2_ringdots -- their far_grid is IDENTICAL, every cell). This means FAR "
    "is a SINGLE measurement, not independently replicated across novel classes -- do not treat it "
    "with the same confidence as detection lag / min-detectable-size, which genuinely ARE "
    "independently confirmed across different novel-class sims (see novel_grid per variant). "
    "Also: FAR is arm-level (one far_grid per arm), not per size-variant -- an older schema stored "
    "an identical FAR row 4x under size05/10/20/30, which looked like (but was not) 4 independent "
    "measurements; this schema fixes that by storing far_grid once per arm."
)


def l2(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def batch_indices(manifest_path: Path, path_to_row: dict) -> tuple[list[int], list[str]]:
    paths, labels = resolve_pool(manifest_path)
    idx = [path_to_row[p] for p in paths]
    return idx, labels


def static_metrics(emb: np.ndarray, labels: list[str]) -> dict:
    from sklearn.cluster import HDBSCAN
    from sklearn.metrics import completeness_score, homogeneity_score
    pred = HDBSCAN(metric="euclidean", **HDB).fit_predict(emb)
    labs = np.asarray(labels)
    cap = capture_metrics(pred, labs, excluded=EXCLUDE_P1)
    target = ~np.isin(labs, list(EXCLUDE_P1))
    pred_t, labs_t = pred[target], labs[target]
    noise_pct = float((pred_t == -1).mean() * 100)
    nn = pred_t != -1
    if nn.sum() > 1 and len(set(pred_t[nn])) > 1:
        comp = float(completeness_score(labs_t[nn], pred_t[nn]))
        hom = float(homogeneity_score(labs_t[nn], pred_t[nn]))
    else:
        comp = hom = 0.0
    # cluster centroids (L2-renormalized mean of members), non-noise only
    centroids, sizes, majority = {}, {}, {}
    for cid in sorted(set(pred[pred >= 0])):
        m = pred == cid
        c = emb[m].mean(0)
        c = c / (np.linalg.norm(c) + 1e-12)
        centroids[int(cid)] = c
        sizes[int(cid)] = int(m.sum())
        majority[int(cid)] = Counter(labs[m]).most_common(1)[0][0]
    return dict(
        pred=pred, capture=cap["capture_rate"], capture_count=cap["capture_count"],
        target_count=cap["target_class_count"], noise_pct=round(noise_pct, 2),
        completeness=round(comp, 4), homogeneity=round(hom, 4), n_clusters=len(centroids),
        centroids=centroids, sizes=sizes, majority=majority,
    )


def best_sim_to_set(c: np.ndarray, ref: dict[int, np.ndarray]) -> float:
    if not ref:
        return 0.0
    return max(float(c @ r) for r in ref.values())


def run_arm(emb_all: np.ndarray, labels_all: list[str], path_to_row: dict, arm_name: str, sim_root: Path) -> dict:
    """Cluster t=1..8 (shared) once, then per-variant t=9..14; derive REF/threshold grid."""
    bg_batches = [sim_root / f"batch_{t:02d}.json" for t in range(1, 9)]
    cum_idx: list[int] = []
    per_t: dict[int, dict] = {}
    for t, mpath in enumerate(bg_batches, start=1):
        idx, _ = batch_indices(mpath, path_to_row)
        cum_idx = cum_idx + idx  # cumulative, non-overlapping by construction
        sub_idx = np.array(cum_idx)
        res = static_metrics(emb_all[sub_idx], [labels_all[i] for i in sub_idx])
        per_t[t] = res
        print(f"  [{arm_name}] t={t:02d} (bg-only, n={len(sub_idx)}) cap={res['capture']:.3f} "
              f"noise%={res['noise_pct']:.1f} k={res['n_clusters']}", flush=True)

    ref = per_t[4]["centroids"]
    ref_sizes = list(per_t[4]["sizes"].values())
    m_mins = {
        mp: (max(HDB["min_cluster_size"], int(np.percentile(ref_sizes, mp))) if ref_sizes else HDB["min_cluster_size"])
        for mp in M_MIN_PERCENTILES
    }

    calib_sims = []
    for t in (1, 2, 3):
        for c in per_t[t]["centroids"].values():
            calib_sims.append(best_sim_to_set(c, ref))
    if not calib_sims:
        calib_sims = [0.5]

    thresholds = {p: float(np.percentile(calib_sims, p)) for p in THRESH_PERCENTILES}

    # ── FAR grid: arm-level, computed ONCE from t=5..8 (never touches novel-window data) ──
    far_grid: dict = {}
    for p in THRESH_PERCENTILES:
        thresh = thresholds[p]
        far_grid[p] = {}
        for mp in M_MIN_PERCENTILES:
            m_min = m_mins[mp]
            cands_by_t = {t: candidates_for_t(per_t[t], ref, thresh, m_min) for t in range(5, 9)}
            far_by_k = {k: persistence_chain(cands_by_t, thresh, k, 5, 8) for k in PERSISTENCE_KS}
            far_grid[p][mp] = {"m_min": m_min, "far_alarms": far_by_k}

    variants: dict[str, dict] = {}
    for size_tag in ("size05", "size10", "size20", "size30"):
        cum_idx_v = list(cum_idx)  # start from the shared bg(1..8) prefix
        v_per_t = dict(per_t)  # t=1..8 shared
        for t in range(9, 15):
            mpath = sim_root / size_tag / f"batch_{t:02d}.json"
            idx, _ = batch_indices(mpath, path_to_row)
            cum_idx_v = cum_idx_v + idx
            sub_idx = np.array(cum_idx_v)
            res = static_metrics(emb_all[sub_idx], [labels_all[i] for i in sub_idx])
            v_per_t[t] = res
            print(f"  [{arm_name}/{size_tag}] t={t:02d} (n={len(sub_idx)}) cap={res['capture']:.3f} "
                  f"noise%={res['noise_pct']:.1f} k={res['n_clusters']}", flush=True)
        variants[size_tag] = v_per_t

    return dict(per_t_bg=per_t, ref=ref, m_mins=m_mins, calib_sims=calib_sims,
                thresholds=thresholds, far_grid=far_grid, variants=variants)


def candidates_for_t(res_t: dict, ref: dict, thresh: float, m_min: int) -> list[dict]:
    out = []
    for cid, c in res_t["centroids"].items():
        sim = best_sim_to_set(c, ref)
        size = res_t["sizes"][cid]
        if sim < thresh and size >= m_min:
            out.append(dict(cid=cid, centroid=c, size=size, sim=sim, majority=res_t["majority"][cid]))
    return out


def persistence_chain(cands_by_t: dict[int, list[dict]], thresh: float, k: int, t_start: int, t_end: int):
    """Greedy consecutive-batch lineage tracking. Returns list of alarm events
    (confirm_t, birth_t, majority label at confirm_t, lineage streak)."""
    active: list[dict] = []  # each: {centroid, streak, birth_t, last_t}
    alarms = []
    for t in range(t_start, t_end + 1):
        cands = cands_by_t.get(t, [])
        matched_active_ids = set()
        new_active = []
        for cand in cands:
            best_j, best_s = None, -1.0
            for j, lin in enumerate(active):
                if j in matched_active_ids:
                    continue
                s = float(cand["centroid"] @ lin["centroid"])
                if s > best_s:
                    best_s, best_j = s, j
            if best_j is not None and best_s >= thresh:
                lin = active[best_j]
                matched_active_ids.add(best_j)
                new_active.append(dict(centroid=cand["centroid"], streak=lin["streak"] + 1,
                                        birth_t=lin["birth_t"], last_t=t, majority=cand["majority"],
                                        size=cand["size"]))
            else:
                new_active.append(dict(centroid=cand["centroid"], streak=1, birth_t=t, last_t=t,
                                        majority=cand["majority"], size=cand["size"]))
        active = new_active  # lineages not extended this t are dropped (persistence must be consecutive)
        for lin in active:
            if lin["streak"] == k:  # fires exactly once, the first t it reaches k
                alarms.append(dict(confirm_t=t, birth_t=lin["birth_t"], majority=lin["majority"],
                                    size=lin["size"]))
    return alarms


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-name", default="unknown_novelty_sim",
                     help="dir name under data/pools/temporal/ (default = original CrossScratch sim)")
    ap.add_argument("--emb-dir", default=None,
                     help="override embedding cache dir (default: result_grouping/temporal_novelty_260726, "
                          "the v1 cache). Use result_grouping/temporal_novelty_v2_260726 for the leakage-fixed v2 arms.")
    ap.add_argument("--out-dir", default=None, help="override report output dir (default: same as --emb-dir)")
    ap.add_argument("--arms", default="frozen,champion",
                     help="comma-separated arms to (re)compute this run. 'frozen' -> f0_frozen.npy, "
                          "'champion' -> f_champion.npy, anything else -> f_<arm>.npy (e.g. 'z0_seed1' -> "
                          "f_z0_seed1.npy -- used for random-head control arms, no training required). "
                          "If the report json already exists, arms not in this list are kept as-is "
                          "(merge, not overwrite) so earlier work is not wasted.")
    a = ap.parse_args()
    sim_root = TEMPORAL_ROOT / a.sim_name
    report_suffix = "" if a.sim_name == "unknown_novelty_sim" else f"_{a.sim_name.replace('unknown_novelty_sim_', '')}"
    emb_dir = Path(a.emb_dir) if a.emb_dir else EMB_DIR
    out_dir = Path(a.out_dir) if a.out_dir else emb_dir
    arms_requested = [s.strip() for s in a.arms.split(",") if s.strip()]

    def arm_file(arm_name: str) -> Path:
        if arm_name == "frozen":
            return emb_dir / "f0_frozen.npy"
        return emb_dir / f"f_{arm_name}.npy"

    all_embs = {}
    for arm_name in arms_requested:
        emb_path = arm_file(arm_name)
        if not emb_path.exists():
            raise SystemExit(
                f"--arms includes '{arm_name}' but {emb_path} does not exist yet "
                "(generate that embedding first, or drop it from --arms)."
            )
        all_embs[arm_name] = l2(np.load(emb_path))
    meta = json.loads((emb_dir / "paths_index.json").read_text(encoding="utf-8"))
    paths, labels = meta["paths"], meta["labels"]
    path_to_row = {p: i for i, p in enumerate(paths)}
    print(f"[load] {len(paths)} embedded images (cache={emb_dir}), sim={a.sim_name}, arms={arms_requested}", flush=True)

    results = {}
    for arm_name in arms_requested:
        print(f"\n=== arm: {arm_name} ({a.sim_name}) ===", flush=True)
        results[arm_name] = run_arm(all_embs[arm_name], labels, path_to_row, arm_name, sim_root)

    # ── build the report (merge with any existing arms already on disk) ──
    out_path = out_dir / f"temporal_novelty_report{report_suffix}.json"
    report = {"sim_name": a.sim_name, "threshold_grid": THRESH_PERCENTILES,
              "m_min_percentile_grid": M_MIN_PERCENTILES,
              "persistence_grid": PERSISTENCE_KS, "_notes": FAR_CAVEAT_NOTE, "arms": {}}
    if out_path.exists():
        prior = json.loads(out_path.read_text(encoding="utf-8"))
        report["arms"].update(prior.get("arms", {}))
        print(f"[merge] loaded {list(report['arms'].keys())} from existing {out_path}", flush=True)
    for arm_name, r in results.items():
        arm_report = {
            "static_bg": {t: {k: v for k, v in res.items()
                               if k not in ("pred", "centroids", "sizes", "majority")}
                          for t, res in r["per_t_bg"].items()},
            "m_mins": r["m_mins"], "thresholds": r["thresholds"],
            # ★ FAR is arm-level (t=5..8 only, never depends on the novel-window variant) — stored
            # once here, NOT duplicated per size-variant. See module docstring / top-level "_notes".
            "far_grid": r["far_grid"],
            "variants": {},
        }
        for size_tag, v_per_t in r["variants"].items():
            static_rows = {t: {k: v for k, v in res.items()
                                if k not in ("pred", "centroids", "sizes", "majority")}
                            for t, res in v_per_t.items()}
            novel_grid: dict = {}
            for p in THRESH_PERCENTILES:
                thresh = r["thresholds"][p]
                novel_grid[p] = {}
                for mp in M_MIN_PERCENTILES:
                    m_min = r["m_mins"][mp]
                    cands_by_t = {t: candidates_for_t(v_per_t[t], r["ref"], thresh, m_min)
                                  for t in range(5, 15)}
                    novel_by_k = {}
                    for k in PERSISTENCE_KS:
                        events = persistence_chain(cands_by_t, thresh, k, 5, 14)
                        # only alarms confirmed at t>=9 count as "detections" in the novel window
                        novel_by_k[k] = [e for e in events if e["confirm_t"] >= 9]
                    novel_grid[p][mp] = {"m_min": m_min, "novel_alarms": novel_by_k}
            arm_report["variants"][size_tag] = {"static": static_rows, "novel_grid": novel_grid}
        report["arms"][arm_name] = arm_report

    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[OUT] {out_path}", flush=True)


if __name__ == "__main__":
    main()
