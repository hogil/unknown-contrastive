#!/usr/bin/env bash
# ★ FCMAE augmentation 단변량 (Codex 설계 v2, 260720): "spatial aug 가 FCMAE z 하락의 원인인가" 검증.
# 통제: FCMAE 백본 freeze (f == frozen f 체크섬 불변) + 순수 B0 (queue/local/ignore/neco 전부 off) + pdim 128 (May parity).
# 변수 = augmentation 한 축씩. 각 변형마다 z / f / f⊕z + ★ep0 대조군(random-head z) 채점 → aug 효과 vs random-proj 효과 분리.
# 판정선(FCMAE frozen, strict-novel 32cls): finch_p2 ARI 0.805 / louvain 0.871.
# 주의: noneonly 는 단변량 아님 = 최종 combo 행(4축 동시 off). 단변량 분해는 notrans/nocrop/norot/noscale.
set -u
cd "$(dirname "$0")"
LOG=_fcmae_aug_ablation.log
EMB=result_grouping/_fcmae_aug260720/embeddings
RES=docs/paper/FCMAE_AUG_ABLATION_260720.md
mkdir -p "$EMB" "$(dirname "$RES")"
TR=data/images/unknown_train_defectaware_260710
EV=data/images/unknown_eval100
FCMAE=convnextv2_base.fcmae_ft_in22k_in1k_384
FROZEN=result_grouping/_unknown_mixed260710/embeddings/frozen_unknown_fcmae.npy
EXCL="Normal,Random,R,Center_bank_boundary,Center_scratch,Donut_bank_boundary,Donut_fork,Edge-Ring_bank_boundary,Edge-Ring_scratch,Edge-Top_fork,Full_scratch,ParallelScratches,RingDots"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
export PYTHONIOENCODING=utf-8

# tag|추가flag|종류  (single=단변량 분해 / combo=최종조합)
VARIANTS=(
  "aug_current||baseline"
  "aug_notrans|--wafer-translate 0|single"
  "aug_nocrop|--wafer-crop-min 1.0|single"
  "aug_norot|--wafer-rot-deg 0|single"
  "aug_noscale|--wafer-scale-min 1.0|single"
  "aug_noneonly|--wafer-rot-deg 0 --wafer-translate 0 --wafer-scale-min 1.0 --wafer-crop-min 1.0|COMBO(4축off)"
)

score(){ # $1=npy $2=label
  python _score_umapfree.py "$1" --labels-from pool --pool "$EV" --exclude-classes "$EXCL" --skip-umap 2>>"$LOG" \
    | grep -iE "finch_p2|louvain" | sed "s/^/$2 /" | tee -a "$RES"
}
concat(){ # $1=f $2=z $3=out
  python -c "
import numpy as np
def l2(a): return a/(np.linalg.norm(a,axis=1,keepdims=True)+1e-9)
np.save(r'$3', np.concatenate([l2(np.load(r'$1').astype('float32')), l2(np.load(r'$2').astype('float32'))],1))" 2>>"$LOG"
}

say "=== FCMAE aug 단변량 v2 시작 (freeze bb, pure B0, pdim128, 6 variants x 5ep + ep0 대조) ==="
{
echo "# FCMAE Augmentation Univariate v2 (260720)"
echo "통제: FCMAE freeze(f불변) + pure B0(queue/local/ignore/neco off) + pdim128 + seed3. 변수=augmentation."
echo "판정선 FCMAE frozen: finch_p2 ARI 0.805 / louvain 0.871 (strict-novel 32cls)."
echo "노트: noneonly=최종 combo(4축 동시 off), 단변량 분해는 notrans/nocrop/norot/noscale. ep0=random-head 대조군."
echo ""
echo "## frozen (판정 기준)"
} > "$RES"
score "$FROZEN" "frozen_f"

for v in "${VARIANTS[@]}"; do
  IFS='|' read -r tag extra kind <<< "$v"
  if [ -f "$EMB/${tag}_ep5.npy" ]; then say "skip $tag"; else
    say ">>> $tag [$kind] $extra"
    python -u _ssl_methods.py --method simclr --timm "$FCMAE" --freeze-backbone --pdim 128 \
        --seed 3 --epochs 5 --batch 8 --ckpt-every 100 \
        --train-dir "$TR" --eval-dir "$EV" --out-dir "$EMB" --tag "$tag" $extra >>"$LOG" 2>&1
    say "<<< $tag done"
  fi
  echo "" >> "$RES"; echo "## $tag  [$kind]  $extra" >> "$RES"
  # ep0 대조군 (random-head): z 와 f+z(random) — 착시 baseline
  z0="$EMB/${tag}_ep0_proj.npy"; f0="$EMB/${tag}_ep0.npy"
  [ -f "$z0" ] && score "$z0" "${tag} ep0 z(random-head 대조)"
  if [ -f "$z0" ] && [ -f "$f0" ]; then concat "$f0" "$z0" "$EMB/${tag}_ep0_fz.npy"; score "$EMB/${tag}_ep0_fz.npy" "${tag} ep0 f+z(random 대조)"; fi
  # 학습 epoch
  for ep in 1 3 5; do
    z="$EMB/${tag}_ep${ep}_proj.npy"; f="$EMB/${tag}_ep${ep}.npy"
    [ -f "$z" ] && score "$z" "${tag} ep${ep} z"
    [ -f "$f" ] && score "$f" "${tag} ep${ep} f(sanity=frozen)"
    if [ -f "$z" ] && [ -f "$f" ]; then concat "$f" "$z" "$EMB/${tag}_ep${ep}_fz.npy"; score "$EMB/${tag}_ep${ep}_fz.npy" "${tag} ep${ep} f+z"; fi
  done
  mkdir -p /e/unknown-contrastive-archive/260720_fcmae_aug_ckpts
  mv "$EMB/${tag}_ckpt.pt" /e/unknown-contrastive-archive/260720_fcmae_aug_ckpts/ 2>/dev/null
done
say "=== FCMAE aug 단변량 v2 DONE — $RES ==="
