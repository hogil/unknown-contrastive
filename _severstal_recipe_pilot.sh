#!/usr/bin/env bash
# ★ Severstal recipe-sweep pilot. team-lead 260726 directive (+ 260726 correction: EPOCHS 8->20,
# 1-seed round-1 -> top 4-5 cells get seed 1/2 in round-2 -- cells cost ~5min/20ep here, not
# ~20min like 6400x6400 wafer, so full 20ep is affordable).
# champion recipe(TEMP0.20/QUEUE16384/NEG0.72/LR_HEAD4e-3/BATCH64/USE_LOCAL=on/SEED42) 가
# raw noise milestone(<=57.74) 미달 -> frag 를 키우지 않으면서 noise 를 더 줄이는 조합 탐색.
# 다이얼 mcs6/ms3/leaf/eps0.06 고정 (runs/severstal/pre_registered_gates*.json 과 동일, 변경 금지).
# 각 cell: baseline 에서 1축만 변경, 나머지는 champion 값 그대로 (coordinate-descent 1단계).
# "base" 행은 재실행하지 않는다 -- 기존 champion 20-epoch/seed42 런
# (runs/severstal/may_repro/abl_B4_260726_100437, severstal_adapt_ruleC.json 에 이미 채점됨)
# 을 그대로 기준행으로 재사용한다. base_e8(8-epoch 대조군, 260726 12:20 완료)도 별도 보관 --
# EPOCHS 8->20 정정의 근거 데이터(8ep 로 끊었을 때 순위가 바뀌는지)로 남겨둔다, 재실행 안 함.
# 순서: neg060/neg085 먼저 (팀리드 지시 -- IGNORE_NEG_SIM=0.72 는 이식값이라 재조정 여지 최대).
# ★ 알려진 버그: 학습 종료 로그에 cp949 UnicodeEncodeError 가 나서 python -u _may_ablation.py 의
#   exit code 가 오염될 수 있다 (체크포인트 저장 "이후" 발생이라 결과엔 영향 없음, camp-cache384 가
#   근본 수정 중). 이 스크립트는 exit code 를 검사하지 않고 proj_ep{N}.pt 존재로만 다음 단계로
#   진행한다.
# 사용: bash _severstal_recipe_pilot.sh                 (round-1, 9 cells, seed42, 20ep 기본)
#       bash _severstal_recipe_pilot.sh "tag t lr q neg local seed" ...   (round-2 커스텀)
set -u; cd "$(dirname "$0")"
POOL="data/pools/severstal_pilot260726.json"
FCMAE="weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth"
EPOCHS="${PILOT_EPOCHS:-20}"
export PYTHONIOENCODING=utf-8
# tag temp lr queue neg local(on/off) seed
if [ $# -eq 0 ]; then
  CONFIGS=(
    "neg060    0.20 4e-3 16384 0.60 on 42"
    "neg085    0.20 4e-3 16384 0.85 on 42"
    "t010      0.10 4e-3 16384 0.72 on 42"
    "t030      0.30 4e-3 16384 0.72 on 42"
    "q4096     0.20 4e-3 4096  0.72 on 42"
    "q32768    0.20 4e-3 32768 0.72 on 42"
    "lr002     0.20 2e-3 16384 0.72 on 42"
    "lr008     0.20 8e-3 16384 0.72 on 42"
    "nolocal   0.20 4e-3 16384 0.72 off 42"
  )
else
  CONFIGS=("$@")
fi
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
for cfg in "${CONFIGS[@]}"; do
  set -- $cfg; tag=$1; temp=$2; lr=$3; q=$4; neg=$5; local=$6; seed=${7:-42}
  localflag=1; [ "$local" = "off" ] && localflag=0
  runtag="$tag"; [ "$seed" != "42" ] && runtag="${tag}_s${seed}"
  say ">>> PILOT $runtag : temp=$temp lr=$lr queue=$q neg=$neg local=$local seed=$seed epochs=$EPOCHS"
  REPRO_DATA="$POOL" REPRO_BACKBONE="$FCMAE" REPRO_OUT="runs/severstal/pilot/run" \
    REPRO_BATCH=64 REPRO_TEMP="$temp" REPRO_LR="$lr" REPRO_QUEUE="$q" REPRO_IGNORE_NEG_SIM="$neg" \
    REPRO_USE_LOCAL="$localflag" REPRO_EPOCHS="$EPOCHS" REPRO_SEED="$seed" REPRO_TAG="_pilot_${runtag}" \
    python -u _may_ablation.py B4 2>&1 | grep -E "Epoch ${EPOCHS} done|ABLATION_DONE|Traceback|CUDA out|override"
  pdir="$(ls -dt runs/severstal/pilot/abl_pilot_${runtag}_B4_*/checkpoints 2>/dev/null | head -1)"
  if [ -z "$pdir" ] || [ ! -f "$pdir/proj_ep${EPOCHS}.pt" ]; then
    say "!!! $runtag : proj_ep${EPOCHS}.pt 없음 -- 학습 실패로 간주, eval 건너뜀"
    continue
  fi
  python -u _grouping_eval.py --backbone "$FCMAE" --pool "$POOL" --proj-dir "$pdir" \
    --tag "sevpilot_${runtag}" --mcs 6 --ms 3 --select-rule rich_noise --out-name "severstal_pilot_${runtag}" \
    2>&1 | grep -E "label-free selection|DONE_EVAL"
  say "<<< $runtag done"
done
say "=== PILOT LEADERBOARD (severstal, Rule C selected-epoch offline, mcs6/ms3/leaf/eps0.06) ==="
python - <<'PY'
import json, glob, re
rows = []
for jp in glob.glob("runs/clean546/severstal_pilot_*.json"):
    d = json.load(open(jp, encoding="utf-8"))
    tag = re.search(r"severstal_pilot_(.+)\.json", jp).group(1)
    sel = d.get("selected")
    if not sel:
        rows.append((tag, d.get("selected_ep"), "gate-fail", "-", "-", "-", "-", "-", "-", "-"))
        continue
    off, lf = sel["off"], sel["lf"]
    rows.append((tag, d.get("selected_ep"), off["P1"], lf["noise_pct"], off["P2_noise"],
                 off["P3_comp"], off["P4_hom"], off["ARI"], off["frag"], lf["k"]))

def p1v(r):
    try:
        return int(str(r[2]).split("/")[0])
    except Exception:
        return -1

rows.sort(key=lambda r: (-p1v(r), float(r[4]) if str(r[4]).replace(".", "").isdigit() else 999))
print(f"{'tag':12s} selEp  P1     lfNoise noiseP2 Comp   Hom    ARI    frag   k")
for r in rows:
    print(f"{r[0]:12s} {str(r[1]):5s}  {str(r[2]):6s} {str(r[3]):7s} {str(r[4]):7s} "
          f"{str(r[5]):6s} {str(r[6]):6s} {str(r[7]):6s} {str(r[8]):6s} {str(r[9])}")
print("(champion 기준행: base(20ep,seed42) lf 66.8/P2 69.43/Comp 0.364/Hom 0.771/ARI 0.226/frag 6.00/k 24 "
      "-- runs/clean546/severstal_adapt_ruleC.json 재사용, 재학습 안 함. base_e8(8ep,seed42) 은 "
      "runs/clean546/severstal_pilot_base_e8.json 에 별도 보관, EPOCHS 8->20 정정 근거로만 사용)")
PY
echo "[SEVERSTAL_PILOT_ROUND_DONE]"
