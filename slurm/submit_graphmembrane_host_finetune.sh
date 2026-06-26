#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
cd "$ROOT"

HOST_CKPT="${HOST_CKPT:-$ROOT/runs/refuge2_phase0/phase0_segformer_b0_no_graph_s1_residual_emasave_host/ckpt/best.pt}"
RESUME_MEMBRANE="${RESUME_MEMBRANE:-$ROOT/runs/graphmembrane/graphmembrane_refuge2_s1_h48_steps6_phase0/ckpt/best.pt}"

if [ ! -f "$HOST_CKPT" ]; then
  echo "ERROR missing HOST_CKPT=$HOST_CKPT"
  exit 20
fi
if [ ! -f "$RESUME_MEMBRANE" ]; then
  echo "ERROR missing RESUME_MEMBRANE=$RESUME_MEMBRANE"
  exit 21
fi

for mode in hard soft; do
  tag="hostft_${mode}"
  train_jid="$(
    SOURCE=host \
    HOST_INPUT_MODE="$mode" \
    HOST_CKPT="$HOST_CKPT" \
    RESUME_MEMBRANE="$RESUME_MEMBRANE" \
    EPOCHS="${EPOCHS:-20}" \
    LR="${LR:-5e-5}" \
    BATCH="${BATCH:-4}" \
    HIDDEN="${HIDDEN:-48}" \
    STEPS="${STEPS:-6}" \
    LAMBDA_ENERGY="${LAMBDA_ENERGY:-0.02}" \
    LAMBDA_RESIDUAL="${LAMBDA_RESIDUAL:-0.05}" \
    RUN_TAG="$tag" \
    sbatch --parsable slurm/train_graphmembrane_refuge2.sbatch
  )"
  echo "GraphMembrane host fine-tune mode=$mode train job: $train_jid"

  mem_ckpt="$ROOT/runs/graphmembrane/graphmembrane_refuge2_s1_h48_steps6_${tag}/ckpt/best.pt"
  eval_jid="$(
    INPUT_MODE="$mode" \
    HOST_CKPT="$HOST_CKPT" \
    MEMBRANE_CKPT="$mem_ckpt" \
    RUN_TAG="hostft_${mode}" \
    ALPHAS="${ALPHAS:-0,0.05,0.10,0.25,0.50,0.75,1.00}" \
    sbatch --parsable --dependency=afterok:"$train_jid" slurm/eval_graphmembrane_host.sbatch
  )"
  echo "GraphMembrane host fine-tune mode=$mode eval job: $eval_jid"
done
