#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
cd "$ROOT"

SEED="${SEED:-1}"
BASE="$ROOT/runs/refuge2_phase0"
ALPHAS="${ALPHAS:-0,0.05,0.10,0.15,0.25,0.40}"
CRC_CONFIDENCE="${CRC_CONFIDENCE:-0.10}"
MAX_CAL_HARM_RATE="${MAX_CAL_HARM_RATE:-0.25}"

declare -a JOBS=(
  "dynk16_a010|$BASE/phase0_segformer_b0_dynk16_s${SEED}_residual_emasave_stagewise_a010/ckpt/best.pt|none|0.0|1.0|0.0"
  "dynk16_a025|$BASE/phase0_segformer_b0_dynk16_s${SEED}_residual_emasave_stagewise_a025/ckpt/best.pt|none|0.0|1.0|0.0"
)

for spec in "${JOBS[@]}"; do
  IFS='|' read -r tag ckpt gate floor power clip <<< "$spec"
  if [ ! -f "$ckpt" ]; then
    echo "skip missing checkpoint: $ckpt"
    continue
  fi
  for risk_score in change change_over_uncertainty change_times_confidence; do
    jid="$(
      CKPT="$ckpt" \
      VARIANT=dynk16 \
      ALPHAS="$ALPHAS" \
      GRAPH_SAFETY_GATE="$gate" \
      GRAPH_GATE_FLOOR="$floor" \
      GRAPH_GATE_POWER="$power" \
      GRAPH_RESIDUAL_CLIP="$clip" \
      CRC_CONFIDENCE="$CRC_CONFIDENCE" \
      MAX_CAL_HARM_RATE="$MAX_CAL_HARM_RATE" \
      RISK_SCORE="$risk_score" \
      RUN_TAG="${tag}_${risk_score}" \
      sbatch --parsable slurm/eval_refuge2_policy.sbatch
    )"
    echo "$tag risk_score=$risk_score policy job: $jid"
  done
done
