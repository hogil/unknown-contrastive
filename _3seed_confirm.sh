#!/usr/bin/env bash
# 3-seed fixed-epoch confirmation for the three cross-dataset headline rows.
# Regime: all-unfreeze + backbone f, with the discovery-stage epoch locked per dataset.
set -euo pipefail
cd "$(dirname "$0")"

LOG=_3seed_confirm.log
EMB=result_grouping/_field_robust/embeddings
RUN=runs/three_seed_confirm_260720
METRICS="$RUN/metrics"
EVENTS="$RUN/events"
RES=docs/paper/THREESEED_CONFIRM_260720.md
ARCHIVE=/e/unknown-contrastive-archive/260720_3seed_ckpts
mkdir -p "$EMB" "$METRICS" "$EVENTS" "$ARCHIVE" "$(dirname "$RES")"

WM_TR=data/images/wm811k_normal1500/all
WM_EV=data/images/wm811k_eval500_512/eval
RS=data/images/hf_resisc45
DTD=data/images/hf_dtd

say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
export PYTHONIOENCODING=utf-8

# name|flags|train|eval|batch|primary FINCH|fixed epoch|exclude|frozen embedding
# Fixed epochs are pre-registered from the seed-3 discovery trajectory, not selected per seed.
CELLS=(
  "wm_l03_ig75|--method simclr --use-queue --queue-size 4096 --ignore 0.75 --local 0.3|$WM_TR|$WM_EV|4|finch_p2|8|Normal,Random,R|$EMB/frozen_wm.npy"
  "rs75|--method simclr --natural-aug --use-queue --queue-size 4096 --ignore 0.75|$RS|$RS|8|finch_p1|6||$EMB/frozen_rs.npy"
  "dtd75|--method simclr --natural-aug --use-queue --queue-size 4096 --ignore 0.75|$DTD|$DTD|8|finch_p1|8||$EMB/frozen_dtd.npy"
)
declare -A SEEDS=( [wm_l03_ig75]="3 4 5" [rs75]="1 3 5" [dtd75]="1 3 5" )

score(){ # npy label eval exclude primary csv
  local npy="$1" label="$2" ev="$3" excl="$4" primary="$5" csv="$6"
  local -a exargs=()
  [ -n "$excl" ] && exargs=(--exclude-classes "$excl")
  python _score_umapfree.py "$npy" --labels-from pool --pool "$ev" \
    "${exargs[@]}" --skip-umap --out-csv "$csv" 2>>"$LOG" \
    | grep -iE "^(${primary}|louvain_res6)" \
    | sed "s/^/$label /" | tee -a "$RES"
}

write_event(){ # event dataset seed epoch embedding metrics baseline primary
  python - "$@" <<'PYEOF'
import json
import sys
from pathlib import Path

event, dataset, seed, epoch, embedding, metrics, baseline, primary = sys.argv[1:]
payload = {
    "status": "completed",
    "experiment": "three_seed_fixed_epoch_confirmation",
    "dataset": dataset,
    "seed": int(seed),
    "fixed_epoch": int(epoch),
    "embedding_mode": "backbone_f",
    "embedding": str(Path(embedding).resolve()),
    "metrics_csv": str(Path(metrics).resolve()),
    "frozen_embedding": str(Path(baseline).resolve()),
    "primary_clusterer": primary,
    "protocol": "same-folder transductive self-adaptation; labels used only for scoring",
}
Path(event).write_text(json.dumps(payload, indent=2), encoding="utf-8")
PYEOF
}

dispatch_analysis(){
  local event="$1"
  python -u scripts/run_result_analysis_agent.py \
    --event-file "$event" --context cross_dataset --max-attempts 3 \
    >>"$RUN/result_analysis_agent.log" 2>&1 &
}

: > "$LOG"
cat > "$RES" <<'EOF'
# 3-Seed Fixed-Epoch Confirmation (260720)

Protocol: all-unfreeze + backbone `f`; labels are used only for scoring. Epochs are locked from the seed-3 discovery trajectory: WM ep8, RESISC45 ep6, DTD ep8. No per-seed best-epoch selection.

Acceptance gate: all three seeds present; primary FINCH ARI mean improves over frozen and at least 2/3 seeds improve; every seed preserves P1; mean P2 does not worsen; mean P3/P4 do not worsen; mean fragmentation is no more than 1.5x frozen; Louvain mean ARI changes in the same positive direction.

Raw rows use the canonical metric order: P1 capture, recov, P2 noise, P3 completeness, P4 homogeneity, ARI, Silhouette, k(total/classes/background), fragment ratio.
EOF

say "=== 3-seed fixed-epoch confirmation started ==="
for c in "${CELLS[@]}"; do
  IFS='|' read -r name flags tr ev batch primary epoch excl frozen <<< "$c"
  [ -f "$frozen" ] || { say "missing frozen embedding: $frozen"; exit 2; }
  baseline_csv="$METRICS/${name}_frozen.csv"
  echo "" >> "$RES"
  echo "## $name (seeds ${SEEDS[$name]}, fixed ep$epoch f)" >> "$RES"
  score "$frozen" "$name frozen" "$ev" "$excl" "$primary" "$baseline_csv"

  for s in ${SEEDS[$name]}; do
    tag="${name}_s${s}"
    target="$EMB/${tag}_ep${epoch}.npy"
    csv="$METRICS/${tag}_ep${epoch}.csv"
    if [ ! -f "$target" ]; then
      say ">>> train $tag (seed $s, fixed score ep$epoch)"
      # shellcheck disable=SC2086
      python -u _ssl_methods.py $flags --seed "$s" --epochs 10 --batch "$batch" --ckpt-every 100 \
        --train-dir "$tr" --eval-dir "$ev" --out-dir "$EMB" --tag "$tag" >>"$LOG" 2>&1
      [ -f "$target" ] || { say "required embedding missing after training: $target"; exit 3; }
      say "<<< $tag done"
    else
      say "reuse $tag (existing fixed-epoch embedding)"
    fi

    score "$target" "$tag ep$epoch f" "$ev" "$excl" "$primary" "$csv"
    event="$EVENTS/${tag}_ep${epoch}.json"
    write_event "$event" "$name" "$s" "$epoch" "$target" "$csv" "$frozen" "$primary"
    dispatch_analysis "$event"

    ckpt="$EMB/${tag}_ckpt.pt"
    if [ -f "$ckpt" ]; then
      mv -n "$ckpt" "$ARCHIVE/"
    fi
  done
done

say "=== training/scoring complete; aggregate all canonical metrics ==="
python - "$METRICS" "$RES" <<'PYEOF'
import csv
import statistics
import sys
from pathlib import Path

metrics_dir = Path(sys.argv[1])
report = Path(sys.argv[2])
specs = {
    "wm_l03_ig75": {"seeds": (3, 4, 5), "epoch": 8, "primary": "finch_p2"},
    "rs75": {"seeds": (1, 3, 5), "epoch": 6, "primary": "finch_p1"},
    "dtd75": {"seeds": (1, 3, 5), "epoch": 8, "primary": "finch_p1"},
}
metric_fields = (
    "P1_capture", "recov", "P2_noise_pct", "P3_completeness",
    "P4_homogeneity", "ARI", "Sil", "k_total", "fragment_ratio",
)

def read_method(path, prefix):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if row["method"].startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {prefix} row in {path}, got {len(matches)}")
    row = matches[0]
    for key in metric_fields:
        row[key] = float(row[key])
    row["P1_capture_count"] = int(row["P1_capture_count"])
    row["P1_target_class_count"] = int(row["P1_target_class_count"])
    row["k_classes"] = int(row["k_classes"])
    row["k_noise"] = int(row["k_noise"])
    return row

def mean_sd(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0

def fmt(values, digits=3):
    mean, sd = mean_sd(values)
    return f"{mean:.{digits}f} +/- {sd:.{digits}f}"

with report.open("a", encoding="utf-8") as out:
    out.write("\n## Aggregate and acceptance gate\n")
    out.write("\n| dataset | clusterer | P1 count/total | P1 | P2 | P3 | P4 | ARI | Sil | k | fragment | delta ARI |\n")
    out.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    gates = []
    for name, spec in specs.items():
        epoch = spec["epoch"]
        seeds = spec["seeds"]
        baseline_csv = metrics_dir / f"{name}_frozen.csv"
        for clusterer in (spec["primary"], "louvain_res6"):
            baseline = read_method(baseline_csv, clusterer)
            rows = [read_method(metrics_dir / f"{name}_s{s}_ep{epoch}.csv", clusterer) for s in seeds]
            counts = [row["P1_capture_count"] for row in rows]
            total = rows[0]["P1_target_class_count"]
            ari_values = [row["ARI"] for row in rows]
            delta = statistics.mean(ari_values) - baseline["ARI"]
            out.write(
                f"| {name} | {clusterer} | {fmt(counts, 2)}/{total} | "
                f"{fmt([r['P1_capture'] for r in rows])} | {fmt([r['P2_noise_pct'] for r in rows])} | "
                f"{fmt([r['P3_completeness'] for r in rows])} | {fmt([r['P4_homogeneity'] for r in rows])} | "
                f"{fmt(ari_values)} | {fmt([r['Sil'] for r in rows])} | {fmt([r['k_total'] for r in rows], 2)} | "
                f"{fmt([r['fragment_ratio'] for r in rows])} | {delta:+.3f} |\n"
            )

        primary = spec["primary"]
        base = read_method(baseline_csv, primary)
        primary_rows = [read_method(metrics_dir / f"{name}_s{s}_ep{epoch}.csv", primary) for s in seeds]
        lv_base = read_method(baseline_csv, "louvain_res6")
        lv_rows = [read_method(metrics_dir / f"{name}_s{s}_ep{epoch}.csv", "louvain_res6") for s in seeds]
        checks = {
            "three_seeds": len(primary_rows) == 3,
            "ari_mean_up": statistics.mean(r["ARI"] for r in primary_rows) > base["ARI"],
            "ari_2_of_3_up": sum(r["ARI"] > base["ARI"] for r in primary_rows) >= 2,
            "p1_preserved_each_seed": all(r["P1_capture"] >= base["P1_capture"] for r in primary_rows),
            "p2_mean_not_worse": statistics.mean(r["P2_noise_pct"] for r in primary_rows) <= base["P2_noise_pct"],
            "p3_mean_not_worse": statistics.mean(r["P3_completeness"] for r in primary_rows) >= base["P3_completeness"],
            "p4_mean_not_worse": statistics.mean(r["P4_homogeneity"] for r in primary_rows) >= base["P4_homogeneity"],
            "fragment_guard": statistics.mean(r["fragment_ratio"] for r in primary_rows) <= 1.5 * base["fragment_ratio"],
            "louvain_ari_mean_up": statistics.mean(r["ARI"] for r in lv_rows) > lv_base["ARI"],
        }
        passed = all(checks.values())
        gates.append(passed)
        detail = ", ".join(f"{key}={'PASS' if value else 'FAIL'}" for key, value in checks.items())
        out.write(f"\n- **{name}: {'PASS' if passed else 'FAIL'}** - {detail}\n")
    out.write(f"\nOverall three-dataset gate: **{'PASS' if all(gates) else 'FAIL'}**\n")
PYEOF

say "=== 3-seed DONE: $RES ==="
