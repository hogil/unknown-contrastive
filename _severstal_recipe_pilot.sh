#!/usr/bin/env bash
# ★ Severstal recipe-sweep pilot (1-축 변경, EPOCHS=8, seed42). team-lead 260726 directive.
# champion recipe(TEMP0.20/QUEUE16384/NEG0.72/LR_HEAD4e-3/BATCH64/USE_LOCAL=on/SEED42)가
# raw noise milestone(<=57.74) 미달 -> frag 를 키우지 않으면서 noise 를 더 줄이는 조합 탐색.
# 다이얼 mcs6/ms3/leaf/eps0.06 고정 (runs/severstal/pre_registered_gates*.json 과 동일, 변경 금지).
# 각 cell: baseline 에서 1축만 변경, 나머지는 champion 값 그대로 (coordinate-descent 1단계).
# base_e8 은 champion recipe 를 EPOCHS=8 로 그대로 재학습한 대조행 -- LR 스케줄(warmup/cosine)이
# total-epoch 에 의존하므로, 기존 20-epoch 런의 ep8 체크포인트와 비교하면 스케줄이 달라 불공정.
set -u; cd "$(dirname "$0")"
POOL="data/pools/severstal_pilot260726.json"
FCMAE="weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth"
export PYTHONIOENCODING=utf-8
# tag temp lr queue neg local(on/off)
if [ $# -eq 0 ]; then
  CONFIGS=(
    "base_e8   0.20 4e-3 16384 0.72 on"
    "t010      0.10 4e-3 16384 0.72 on"
    "t030      0.30 4e-3 16384 0.72 on"
    "neg060    0.20 4e-3 16384 0.60 on"
    "neg085    0.20 4e-3 16384 0.85 on"
    "q4096     0.20 4e-3 4096  0.72 on"
    "q32768    0.20 4e-3 32768 0.72 on"
    "lr002     0.20 2e-3 16384 0.72 on"
    "lr008     0.20 8e-3 16384 0.72 on"
    "nolocal   0.20 4e-3 16384 0.72 off"
  )
else
  CONFIGS=("$@")
fi
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
for cfg in "${CONFIGS[@]}"; do
  set -- $cfg; tag=$1; temp=$2; lr=$3; q=$4; neg=$5; local=$6
  localflag=1; [ "$local" = "off" ] && localflag=0
  say ">>> PILOT $tag : temp=$temp lr=$lr queue=$q neg=$neg local=$local"
  REPRO_DATA="$POOL" REPRO_BACKBONE="$FCMAE" REPRO_OUT="runs/severstal/pilot/run" \
    REPRO_BATCH=64 REPRO_TEMP="$temp" REPRO_LR="$lr" REPRO_QUEUE="$q" REPRO_IGNORE_NEG_SIM="$neg" \
    REPRO_USE_LOCAL="$localflag" REPRO_EPOCHS=8 REPRO_SEED=42 REPRO_TAG="_pilot_${tag}" \
    python -u _may_ablation.py B4 2>&1 | grep -E "Epoch 8 done|ABLATION_DONE|Traceback|CUDA out|override"
  pdir="$(ls -dt runs/severstal/pilot/abl_pilot_${tag}_B4_*/checkpoints 2>/dev/null | head -1)"
  python -u _grouping_eval.py --backbone "$FCMAE" --pool "$POOL" --proj-dir "$pdir" \
    --tag "sevpilot_${tag}" --mcs 6 --ms 3 --select-rule rich_noise --out-name "severstal_pilot_${tag}" \
    2>&1 | grep -E "label-free selection|DONE_EVAL"
  say "<<< $tag done"
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
        rows.append((tag, d.get("selected_ep"), "gate-fail", "-", "-", "-", "-", "-", "-"))
        continue
    off, lf = sel["off"], sel["lf"]
    rows.append((tag, d.get("selected_ep"), off["P1"], off["P2_noise"], lf["noise_pct"],
                 off["P3_comp"], off["P4_hom"], off["ARI"], off["frag"]))

def p1v(r):
    try:
        return int(str(r[2]).split("/")[0])
    except Exception:
        return -1

rows.sort(key=lambda r: (-p1v(r), float(r[3]) if str(r[3]).replace(".", "").isdigit() else 999))
print(f"{'tag':10s} selEp  P1     noiseP2 lfNoise Comp   Hom    ARI    frag")
for r in rows:
    print(f"{r[0]:10s} {str(r[1]):5s}  {str(r[2]):6s} {str(r[3]):7s} {str(r[4]):7s} "
          f"{str(r[5]):6s} {str(r[6]):6s} {str(r[7]):6s} {str(r[8])}")
print("(champion 20-epoch 기준 ep17: P2 69.43 / lf 66.8 / Comp 0.364 / Hom 0.771 / ARI 0.226 / "
      "frag 6.00 -- base_e8 은 동일 recipe 의 8-epoch 재학습판, LR스케줄 공정비교용 대조행)")
PY
echo "[SEVERSTAL_PILOT_ROUND_DONE]"
