#!/usr/bin/env bash
# ★ ladder-3 블로커 해소용 **2번째 pool 재현**. (task #21 — 원래 "cca 7-class" 로 적혀 있었으나
#   cca 는 docs/ABSOLUTE_RULES.md §3 에서 금지된 데이터셋이라 **승인된 mwm38_clean546 으로 교체**한다.)
#
# 질문: severstal 에서 "라벨 없이 argmin(noise) 로 고른 셀"이 실제 ARI 1위였다.
#       이게 그 pool 한정 우연인가, 규칙으로 쓸 수 있는가?
#   severstal 실측(runs/clean546/severstal_pilot_*_mcs20.json):
#     라벨없이 1위 lr008(noise 57.89) -> ARI 0.6296 (실제로도 1위) ✅
#     라벨없이 2위 neg060(66.23)      -> ARI 0.2768 (실제 꼴찌급) ❌
#     라벨없이 3위 q32768(66.53)      -> ARI 0.6108 (실제 2위)   ✅
#   => 1위는 맞췄지만 중간순위는 엉망. **다른 pool 에서도 1위를 맞추는지**가 관건.
#
# ★ severstal pilot(_severstal_recipe_pilot.sh)과 **완전히 같은 9셀 · 같은 선택규칙**을 쓴다.
#   한 축이라도 바꾸면 재현이 아니라 새 실험이 된다.
# 다이얼: mcs6/ms3/leaf/eps0.06 — clean546 의 검증된 다이얼
#   (memory project_unlabeled_grouping_success_260724 "finer HDBSCAN mcs6ms3leaf" + 실측상
#    clean546 은 mcs 5~44 구간에서 ARI 평탄이라 다이얼 민감도가 낮다).
#   ⚠ severstal 은 mcs6/ms3 로 돌고 mcs20 재채점본이 따로 있다. 비교는 **같은 다이얼끼리**만.
#
# 사용: bash _clean546_recipe_pilot.sh                       (round-1, 9 cells, seed42, 20ep)
#       bash _clean546_recipe_pilot.sh "tag t lr q neg local seed" ...   (round-2 커스텀)
set -u; cd "$(dirname "$0")"
POOL="data/pools/mwm38_clean546.json"
FCMAE="weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth"
EPOCHS="${PILOT_EPOCHS:-20}"
MCS="${PILOT_MCS:-6}"
MS="${PILOT_MS:-3}"
export PYTHONIOENCODING=utf-8
# tag temp lr queue neg local(on/off) seed  — severstal pilot 과 동일 목록
if [ $# -eq 0 ]; then
  CONFIGS=(
    "base      0.20 4e-3 16384 0.72 on 42"
    "neg060    0.20 4e-3 16384 0.60 on 42"
    "neg085    0.20 4e-3 16384 0.85 on 42"
    "t010      0.10 4e-3 16384 0.72 on 42"
    "t030      0.30 4e-3 16384 0.72 on 42"
    "q4096     0.20 4e-3 4096  0.72 on 42"
    "q32768    0.20 4e-3 32768 0.72 on 42"
    "lr002     0.20 2e-3 16384 0.72 on 42"
    "lr008     0.20 8e-3 16384 0.72 on 42"
  )
else
  CONFIGS=("$@")
fi
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
say "=== clean546 recipe pilot : pool=$POOL dial=mcs${MCS}/ms${MS}/leaf/eps0.06 epochs=$EPOCHS ==="
for cfg in "${CONFIGS[@]}"; do
  set -- $cfg; tag=$1; temp=$2; lr=$3; q=$4; neg=$5; local=$6; seed=${7:-42}
  localflag=1; [ "$local" = "off" ] && localflag=0
  runtag="$tag"; [ "$seed" != "42" ] && runtag="${tag}_s${seed}"
  # 이미 끝난 셀은 건너뛴다 (중간에 끊겨도 재실행하면 남은 것만 돈다)
  if [ -f "runs/clean546/c546_pilot_${runtag}.json" ]; then
    say "--- $runtag : 이미 있음, skip"
    continue
  fi
  say ">>> PILOT $runtag : temp=$temp lr=$lr queue=$q neg=$neg local=$local seed=$seed epochs=$EPOCHS"
  REPRO_DATA="$POOL" REPRO_BACKBONE="$FCMAE" REPRO_OUT="runs/clean546/pilot/run" \
    REPRO_BATCH=64 REPRO_TEMP="$temp" REPRO_LR="$lr" REPRO_QUEUE="$q" REPRO_IGNORE_NEG_SIM="$neg" \
    REPRO_USE_LOCAL="$localflag" REPRO_EPOCHS="$EPOCHS" REPRO_SEED="$seed" REPRO_TAG="_c546_${runtag}" \
    python -u _may_ablation.py B4 2>&1 | grep -E "Epoch ${EPOCHS} done|ABLATION_DONE|Traceback|CUDA out|override"
  pdir="$(ls -dt runs/clean546/pilot/abl_c546_${runtag}_B4_*/checkpoints 2>/dev/null | head -1)"
  if [ -z "$pdir" ] || [ ! -f "$pdir/proj_ep${EPOCHS}.pt" ]; then
    say "!!! $runtag : proj_ep${EPOCHS}.pt 없음 -- 학습 실패로 간주, eval 건너뜀"
    continue
  fi
  python -u _grouping_eval.py --backbone "$FCMAE" --pool "$POOL" --proj-dir "$pdir" \
    --tag "c546pilot_${runtag}" --mcs "$MCS" --ms "$MS" --select-rule rich_noise \
    --out-name "c546_pilot_${runtag}" 2>&1 | grep -E "label-free selection|DONE_EVAL"
  say "<<< $runtag done"
done
say "=== CLEAN546 PILOT LEADERBOARD (Rule C selected-epoch, mcs${MCS}/ms${MS}/leaf/eps0.06) ==="
python - <<'PY'
import json, glob, re
rows = []
for jp in glob.glob("runs/clean546/c546_pilot_*.json"):
    if jp.endswith(".selection.json"):
        continue          # _grouping_eval 이 같이 쓰는 부산물 — leaderboard 행이 아니다
    d = json.load(open(jp, encoding="utf-8"))
    tag = re.search(r"c546_pilot_(.+)\.json", jp).group(1)
    sel = d.get("selected")
    if not sel:
        rows.append((tag, d.get("selected_ep"), "gate-fail", "-", "-", "-", "-", "-", "-", "-")); continue
    off, lf = sel["off"], sel["lf"]
    rows.append((tag, d.get("selected_ep"), off["P1"], lf["noise_pct"], off["P2_noise"],
                 off["P3_comp"], off["P4_hom"], off["ARI"], off["frag"], lf["k"]))
# ★ 라벨 없이 고르는 순서 그대로 = lfNoise 오름차순. 옆에 정답(ARI)을 붙여 규칙 적중을 본다.
def lfn(r):
    try: return float(r[3])
    except Exception: return 999.0
rows.sort(key=lfn)
print(f"{'rank':4s} {'tag':12s} selEp  {'lfNoise':7s} {'ARI':6s} {'P1':6s} {'Comp':6s} {'Hom':6s} {'frag':6s} k")
for i, r in enumerate(rows, 1):
    print(f"{i:<4d} {r[0]:12s} {str(r[1]):5s}  {str(r[3]):7s} {str(r[7]):6s} {str(r[2]):6s} "
          f"{str(r[5]):6s} {str(r[6]):6s} {str(r[8]):6s} {str(r[9])}")
ok = [r for r in rows if str(r[7]).replace('.','').replace('-','').isdigit()]
if len(ok) >= 2:
    best_lf = ok[0][0]
    best_ari = max(ok, key=lambda r: float(r[7]))[0]
    print()
    print(f"★ 라벨없이 고른 1위 : {best_lf}")
    print(f"★ 실제 ARI 1위     : {best_ari}")
    print(f"★ 규칙 적중?       : {'YES — 규칙이 정답을 집었다' if best_lf == best_ari else 'NO — 규칙이 틀렸다 (ladder-3 재설계 필요)'}")
PY
echo "[CLEAN546_PILOT_ROUND_DONE]"
