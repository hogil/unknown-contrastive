#!/usr/bin/env python3
# ★ Task #20 (team-lead 260726): 라벨 없이 "셀(레시피)"을 고를 수 있는가 -- ladder-3 배포 여부를
# 가르는 분석. Rule C 는 라벨 없이 "epoch" 선택을 이미 풀었다(외부검증 통과) -- 이 스크립트는
# 라벨 없이 "cell" 선택이 되는지를 검증한다. 추가 학습/GPU 0 -- 이미 만들어진
# runs/clean546/severstal_pilot_*.json (Rule C 로 선택된 epoch 의 off/lf 지표) 만 사용.
#
# 프로토콜: (A) 라벨 순위(ARI desc)는 이 스크립트에서 라벨을 "여는" 유일한 지점이다.
# (B) 무라벨 후보(seed_noise/frag/stability/coherence/silhouette)는 전부 off/lf JSON 필드에서
# 이미 계산돼 있던 값을 그대로 읽는다 -- (B) 계산 도중 라벨을 다시 보는 코드 경로는 없다
# (off.ARI/P3_comp/P4_hom 은 (A) 출력에서만 읽고 (B) 순위 계산 함수에는 절대 넘기지 않는다).
#
# 다이얼/셀 정의 변경 금지 -- 이 스크립트는 순수 사후분석, runs/clean546/*.json 을 읽기만 한다.
import glob
import hashlib
import json
import os
import re
import sys

from scipy.stats import spearmanr

CLEAN546 = "runs/clean546"
NOISE_BAND_PP = 2.28   # runs/severstal/pre_registered_gates.json 과 동일
ARI_BAND = 0.019
COMP_BAND = 0.033
HOM_BAND = 0.005
RHO_DEPLOY_THRESHOLD = 0.7
EXPECTED_DIAL = {"mcs": 20, "ms": 5, "eps": 0.06, "method": "leaf"}
REQUIRED_CELL_TAGS = (
    "base", "t010", "t030", "neg060", "neg085", "q4096", "q32768", "lr002", "lr008",
)
LABEL_FREE_CANDIDATES = ("seed_noise", "k", "stability", "coherence", "over_merge")


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_cell(tag, path):
    path = os.path.abspath(path)
    d = json.load(open(path, encoding="utf-8"))
    if d.get("dial") != EXPECTED_DIAL:
        raise ValueError(f"{path}: expected dial {EXPECTED_DIAL}, got {d.get('dial')}")
    sel = d.get("selected")
    if not sel:
        return None
    off, lf = sel["off"], sel["lf"]
    return {
        "tag": tag,
        "selected_ep": d.get("selected_ep"),
        "P1": off["P1"],
        # (A) 라벨 지표 -- 아래 build_rankings() 의 rank_A() 에서만 사용
        "ARI": off["ARI"],
        "Comp": off["P3_comp"],
        "Hom": off["P4_hom"],
        # (B) 무라벨 후보 -- Rule C 선택(라벨 미사용) epoch 의 label-free 필드 그대로
        "seed_noise": lf["noise_pct"],       # 재배정 전 raw HDBSCAN noise%, 낮을수록 좋음
        "k": lf["k"],
        "stability": lf["stability"],        # bootstrap co-assignment, seed(재배정전) 기준, 높을수록 좋음
        "coherence": lf["coherence"],        # 재배정 후 멤버 pairwise cos sim, 높을수록 좋음
        "over_merge": lf["over_merge"],      # 낮을수록 좋음
        "path": path,
        "dial": d["dial"],
        "sha256": sha256(path),
    }


def collect_cells(clean546=CLEAN546):
    """Load only the preregistered full-20 mcs20 cell set; never mix mcs6."""
    paths = {"base": f"{clean546}/severstal_pilot_base_mcs20.json"}
    for path in glob.glob(f"{clean546}/severstal_pilot_*_mcs20.json"):
        tag = re.search(r"severstal_pilot_(.+)_mcs20\.json", path).group(1)
        if tag != "base":
            paths[tag] = path
    if set(paths) != set(REQUIRED_CELL_TAGS):
        raise ValueError(
            "full20 mcs20 cell set must contain exactly "
            f"{list(REQUIRED_CELL_TAGS)}; found {sorted(paths)}"
        )
    cells = {}
    for tag in REQUIRED_CELL_TAGS:
        cell = load_cell(tag, paths[tag])
        if cell is None:
            raise ValueError(f"{paths[tag]}: Rule C selection missing")
        cells[tag] = cell
    if len(cells) != 9 or len(set(cells)) != 9:
        raise ValueError(f"expected exactly 9 unique cells, got {list(cells)}")
    return cells


def main():
    try:
        cells = collect_cells()
    except ValueError as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        sys.exit(1)
    n = len(cells)
    print(f"[inputs] n={n}; expected_dial={EXPECTED_DIAL}; cells={list(cells)}")
    for tag in REQUIRED_CELL_TAGS:
        row = cells[tag]
        print(f"  {tag}: path={row['path']} sha256={row['sha256']} dial={row['dial']}")

    rows = list(cells.values())
    tags = [r["tag"] for r in rows]

    # ===== (A) 라벨 순위 -- ARI desc (secondary Comp, Hom). 이 지점 이후로 ARI/Comp/Hom 은
    # rank_A 출력물과 상관계수 계산에만 쓰인다 -- (B) 순위 함수엔 절대 전달하지 않는다. =====
    rank_A = sorted(tags, key=lambda t: (-cells[t]["ARI"], -cells[t]["Comp"], -cells[t]["Hom"]))
    ari_by_tag = {t: cells[t]["ARI"] for t in tags}

    # ===== (B) 무라벨 후보 -- 각자 "낮을수록 좋음" 인 지표는 부호를 뒤집어 oriented score 로 통일
    # (오리엔티드 score 가 클수록 "좋다"는 뜻이 되도록) =====
    candidates = {
        "seed_noise": {t: -cells[t]["seed_noise"] for t in tags},   # 낮을수록 좋음 -> 부호반전
        "k":          {t: cells[t]["k"] for t in tags},              # 높을수록 좋음 (사전등록 후보)
        "stability":  {t: cells[t]["stability"] for t in tags},      # 높을수록 좋음
        "coherence":  {t: cells[t]["coherence"] for t in tags},      # 높을수록 좋음
        "over_merge": {t: -cells[t]["over_merge"] for t in tags},    # 낮을수록 좋음
    }
    assert tuple(candidates) == LABEL_FREE_CANDIDATES

    print("\n=== 채점표 (Rule C 선택 epoch 기준, base 는 20ep 재사용, 나머지 9셀 20ep/seed42 신규) ===")
    hdr = f"{'tag':22s} selEp  P1     ARI    Comp   Hom    seed_noise  k    over  stab   coh"
    print(hdr)
    for t in rank_A:
        r = cells[t]
        print(f"{t:22s} {str(r['selected_ep']):5s}  {str(r['P1']):6s} {r['ARI']:.4f} {r['Comp']:.4f} "
              f"{r['Hom']:.4f} {r['seed_noise']:10.2f}  {r['k']:3d}  {r['over_merge']:4d}  "
              f"{r['stability']:.3f}  {r['coherence']:.3f}")

    print(f"\n=== (A) 라벨 순위 (ARI desc, n={n}) ===")
    for i, t in enumerate(rank_A, 1):
        print(f"  {i}. {t}  ARI={ari_by_tag[t]:.4f}")

    print(f"\n=== 사전등록 rho 판정 (rho>={RHO_DEPLOY_THRESHOLD}, n={n}; CI 넓음 -- 과대해석 금지) ===")
    a_vals = [ari_by_tag[t] for t in tags]
    rho_results = {}
    for name, oriented in candidates.items():
        b_vals = [oriented[t] for t in tags]
        rho, pval = spearmanr(a_vals, b_vals)
        rho_results[name] = rho
        verdict = "DEPLOY-CANDIDATE (rho>=0.7)" if rho >= RHO_DEPLOY_THRESHOLD else "insufficient alone"
        print(f"  {name:12s} rho={rho:+.3f}  p={pval:.3f}  -> {verdict}")

    # ===== 잡음폭 밖으로 갈리는 셀 쌍에 한정한 순위일치율 =====
    print(f"\n=== 잡음폭(ARI ±{ARI_BAND}) 밖으로 갈리는 셀 쌍만 -- concordance ===")
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = tags[i], tags[j]
            if abs(ari_by_tag[ti] - ari_by_tag[tj]) > ARI_BAND:
                pairs.append((ti, tj))
    n_pairs_total = n * (n - 1) // 2
    print(f"  전체 쌍 {n_pairs_total}개 중 ARI로 잡음폭 밖 분리되는 쌍 = {len(pairs)}개")
    if not pairs:
        print("  [경고] 잡음폭 밖으로 분리되는 쌍이 0개 -- 이 10셀은 ARI 축에서 서로 통계적으로 "
              "구분이 안 된다는 뜻. 아래 concordance 는 계산 불가/무의미.")
    else:
        for name, oriented in candidates.items():
            concordant = 0
            for ti, tj in pairs:
                a_order = ari_by_tag[ti] > ari_by_tag[tj]
                b_order = oriented[ti] > oriented[tj]
                if a_order == b_order:
                    concordant += 1
            rate = concordant / len(pairs)
            print(f"  {name:12s} concordant={concordant}/{len(pairs)} ({rate*100:.1f}%)")

    print(f"\n=== 사전등록 판정 (rho 기준만 사용, task #20) ===")
    best_name = max(rho_results, key=lambda k: rho_results[k])
    best_rho = rho_results[best_name]
    any_pass = any(r >= RHO_DEPLOY_THRESHOLD for r in rho_results.values())
    seed_noise_alone = rho_results["seed_noise"] >= RHO_DEPLOY_THRESHOLD
    print(f"  최고 후보: {best_name} (rho={best_rho:+.3f})")
    print(f"  seed_noise 단독 충분? {'YES' if seed_noise_alone else 'NO'} "
          f"(rho={rho_results['seed_noise']:+.3f}, 기준 {RHO_DEPLOY_THRESHOLD})")
    if any_pass:
        print("  -> DEPLOY-CANDIDATE 존재: ladder-3 성립 (라벨 없이 레시피 선택 가능한 후보 있음)")
    else:
        print("  -> FIRST-CLASS NEGATIVE: 모든 무라벨 후보 rho<0.7 -- 라벨 없이 레시피 선택 불가. "
              "사내 배포엔 소량 라벨링 예산이 필요하다는 결론.")
    print(f"  (주의: n={n} 셀, rho 신뢰구간 넓음. 위 concordance(잡음폭 밖 쌍 한정)를 같이 봐야 함)")
    print("\n=== 탐색적 top-1 일치 보고 (사전등록 rho 판정과 분리; 배포 근거 아님) ===")
    label_top1 = rank_A[0]
    for name, oriented in candidates.items():
        unlabeled_top1 = max(tags, key=lambda tag: oriented[tag])
        print(f"  {name:12s} top1={unlabeled_top1} label_top1={label_top1} "
              f"match={'YES' if unlabeled_top1 == label_top1 else 'NO'}")


if __name__ == "__main__":
    main()
