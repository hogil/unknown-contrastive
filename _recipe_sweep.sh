#!/usr/bin/env bash
# ★ recipe coordinate-descent sweep (master orchestration). clean546, FCMAE, B4 recipe 기반.
# 각 config: 학습(knob override) → _grouping_eval(무라벨 선택 + 오프라인) → eval_sweep_<tag>.json.
# 인자: config 들을 "tag TEMP LR QUEUE NEG" 형식으로. 없으면 round1(temperature).
set -u; cd "$(dirname "$0")"
CLEAN="D:/project/unknown-contrastive/data/images/mwm38_clean546"
FCMAE="D:/project/unknown-contrastive/weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth"
export PYTHONIOENCODING=utf-8
# 기본 round1: temperature. (LR/QUEUE/NEG 는 baseline 값)
if [ $# -eq 0 ]; then
  CONFIGS=("t05 0.05 1e-3 16384 0.72" "t07 0.07 1e-3 16384 0.72" "t10 0.10 1e-3 16384 0.72" "t15 0.15 1e-3 16384 0.72")
else
  CONFIGS=("$@")
fi
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
for cfg in "${CONFIGS[@]}"; do
  set -- $cfg; tag=$1; temp=$2; lr=$3; q=$4; neg=$5
  say ">>> SWEEP $tag : temp=$temp lr=$lr queue=$q neg=$neg"
  REPRO_DATA="$CLEAN" REPRO_BACKBONE="$FCMAE" REPRO_OUT="D:/project/unknown-contrastive/runs/sweep/run" \
    REPRO_BATCH=64 REPRO_TEMP="$temp" REPRO_LR="$lr" REPRO_QUEUE="$q" REPRO_NEG="$neg" REPRO_TAG="_sw_${tag}" \
    python -u _may_ablation.py B4 2>&1 | grep -E "Epoch 20 done|ABLATION_DONE|Traceback|CUDA out"
  pdir="$(ls -dt runs/sweep/abl_sw_${tag}_B4_*/checkpoints 2>/dev/null | head -1)"
  python -u _grouping_eval.py --backbone "$FCMAE" --pool "$CLEAN" --proj-dir "$pdir" --tag "sweep_${tag}" 2>&1 | grep -E "무라벨 선택|DONE_EVAL"
  say "<<< $tag done"
done
say "=== LEADERBOARD (clean546, selected-epoch offline) ==="
python - <<'PY'
import json, glob, re
rows=[]
for jp in glob.glob("runs/clean546/eval_sweep_*.json"):
    d=json.load(open(jp,encoding="utf-8")); tag=re.search(r"eval_sweep_(.+)\.json",jp).group(1)
    s=d.get("selected"); fr=d.get("frozen",{}).get("off",{})
    if not s: rows.append((tag,d.get("selected_ep"),"gate-fail","-","-","-")); continue
    o=s["off"]; rows.append((tag,d.get("selected_ep"),o["P1"],o["P2_noise"],o["P4_hom"],o["ARI"]))
def p1v(r):
    try: return int(str(r[2]).split("/")[0])
    except: return -1
rows.sort(key=lambda r:(-p1v(r), float(r[3]) if str(r[3]).replace('.','').isdigit() else 999))
print(f"{'tag':10s} selEp  P1     noise  P4hom  ARI")
for r in rows: print(f"{r[0]:10s} {str(r[1]):5s}  {str(r[2]):6s} {str(r[3]):6s} {str(r[4]):6s} {str(r[5])}")
print("(baseline frozen f = 4/7 noise60 / prev B4 t0.07 = 6/7 noise~28)")
PY
echo "[SWEEP_ROUND_DONE]"
