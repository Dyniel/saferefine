#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
cd "$ROOT"

HOST_CKPT="${HOST_CKPT:-$ROOT/runs/refuge2_phase0/phase0_segformer_b0_no_graph_s1_residual_emasave_host/ckpt/best.pt}"
PHASE0_CKPT="${PHASE0_CKPT:-$ROOT/runs/graphmembrane/graphmembrane_refuge2_s1_h48_steps6_phase0/ckpt/best.pt}"
HOSTFT_HARD_CKPT="${HOSTFT_HARD_CKPT:-$ROOT/runs/graphmembrane/graphmembrane_refuge2_s1_h48_steps6_hostft_hard/ckpt/best.pt}"
HOSTFT_SOFT_CKPT="${HOSTFT_SOFT_CKPT:-$ROOT/runs/graphmembrane/graphmembrane_refuge2_s1_h48_steps6_hostft_soft/ckpt/best.pt}"

STAMP="$(date '+%Y%m%d_%H%M%S')"
OUT_DIR="$ROOT/results/graphmembrane_host"
mkdir -p "$OUT_DIR"

declare -a deps=()

submit_eval() {
  local label="$1"
  local mode="$2"
  local ckpt="$3"
  local alphas="$4"
  local out_csv="$OUT_DIR/${label}_${mode}_${STAMP}.csv"
  local out_json="$OUT_DIR/${label}_${mode}_${STAMP}.json"
  local jid
  jid="$(
    INPUT_MODE="$mode" \
    HOST_CKPT="$HOST_CKPT" \
    MEMBRANE_CKPT="$ckpt" \
    RUN_TAG="$label" \
    ALPHAS="$alphas" \
    OUT_CSV="$out_csv" \
    OUT_SUMMARY="$out_json" \
    sbatch --parsable slurm/eval_graphmembrane_host.sbatch
  )"
  echo "geom host eval label=$label mode=$mode job=$jid csv=$out_csv"
  deps+=("$jid")
}

submit_eval "geom_phase0" "hard" "$PHASE0_CKPT" "${ALPHAS_PHASE0:-0,0.10,0.25,0.50,0.75,1.00}"
submit_eval "geom_phase0" "soft" "$PHASE0_CKPT" "${ALPHAS_PHASE0:-0,0.10,0.25,0.50,0.75,1.00}"
submit_eval "geom_hostft_hard" "hard" "$HOSTFT_HARD_CKPT" "${ALPHAS_HOSTFT:-0,0.05,0.10,0.25,0.50,0.75,1.00}"
submit_eval "geom_hostft_soft" "soft" "$HOSTFT_SOFT_CKPT" "${ALPHAS_HOSTFT:-0,0.05,0.10,0.25,0.50,0.75,1.00}"

dep_csv="$(IFS=:; echo "${deps[*]}")"
inputs="phase0_hard=$OUT_DIR/geom_phase0_hard_${STAMP}.csv,phase0_soft=$OUT_DIR/geom_phase0_soft_${STAMP}.csv,hostft_hard=$OUT_DIR/geom_hostft_hard_hard_${STAMP}.csv,hostft_soft=$OUT_DIR/geom_hostft_soft_soft_${STAMP}.csv"

for profile in practical strict; do
  if [ "$profile" = "strict" ]; then
    max_harm="${STRICT_MAX_CAL_HARM_RATE:-0.0}"
  else
    max_harm="${PRACTICAL_MAX_CAL_HARM_RATE:-0.25}"
  fi
  for risk_score in changed geom change_plus_geom topology; do
    jid="$(
      INPUTS="$inputs" \
      RISK_SCORE="$risk_score" \
      MAX_CAL_HARM_RATE="$max_harm" \
      RUN_TAG="geom_${profile}_portfolio" \
      sbatch --parsable --dependency=afterok:"$dep_csv" slurm/eval_graphmembrane_action_policy.sbatch
    )"
    echo "geom action policy profile=$profile risk_score=$risk_score max_harm=$max_harm job=$jid dependency=afterok:$dep_csv"
  done
done
