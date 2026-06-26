#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
cd "$ROOT"

SEED="${SEED:-1}"
EPOCHS_HOST="${EPOCHS_HOST:-80}"
EPOCHS_GRM="${EPOCHS_GRM:-40}"
LR_HOST="${LR_HOST:-2.5e-4}"
LR_GRM="${LR_GRM:-1e-4}"

echo "Submitting REFUGE2 EMA-checkpoint rerun from $ROOT"
echo "Seed=$SEED host_epochs=$EPOCHS_HOST grm_epochs=$EPOCHS_GRM"

host_jid="$(
  VARIANT=no_graph \
  SEED="$SEED" \
  EPOCHS="$EPOCHS_HOST" \
  LR="$LR_HOST" \
  RUN_TAG=emasave_host \
  sbatch --parsable slurm/train_refuge2_variant.sbatch
)"

host_ckpt="$ROOT/runs/refuge2_phase0/phase0_segformer_b0_no_graph_s${SEED}_residual_emasave_host/ckpt/best.pt"
echo "host job: $host_jid"
echo "host checkpoint: $host_ckpt"

dyn_a010_jid="$(
  RESUME_HOST="$host_ckpt" \
  FREEZE_HOST=1 \
  VARIANT=dynk16 \
  SEED="$SEED" \
  ALPHA_GRAPH=0.10 \
  LR="$LR_GRM" \
  EPOCHS="$EPOCHS_GRM" \
  RUN_TAG=emasave_stagewise_a010 \
  sbatch --parsable --dependency=afterok:"$host_jid" slurm/train_refuge2_variant.sbatch
)"

dyn_a025_jid="$(
  RESUME_HOST="$host_ckpt" \
  FREEZE_HOST=1 \
  VARIANT=dynk16 \
  SEED="$SEED" \
  ALPHA_GRAPH=0.25 \
  LR="$LR_GRM" \
  EPOCHS="$EPOCHS_GRM" \
  RUN_TAG=emasave_stagewise_a025 \
  sbatch --parsable --dependency=afterok:"$host_jid" slurm/train_refuge2_variant.sbatch
)"

grid_a025_jid="$(
  RESUME_HOST="$host_ckpt" \
  FREEZE_HOST=1 \
  VARIANT=grid_only \
  SEED="$SEED" \
  ALPHA_GRAPH=0.25 \
  LR="$LR_GRM" \
  EPOCHS="$EPOCHS_GRM" \
  RUN_TAG=emasave_stagewise_a025 \
  sbatch --parsable --dependency=afterok:"$host_jid" slurm/train_refuge2_variant.sbatch
)"

echo "dynk16 alpha=0.10 train job: $dyn_a010_jid"
echo "dynk16 alpha=0.25 train job: $dyn_a025_jid"
echo "grid alpha=0.25 train job: $grid_a025_jid"

for train_alpha in 010 025; do
  if [ "$train_alpha" = "010" ]; then
    dep="$dyn_a010_jid"
    ckpt="$ROOT/runs/refuge2_phase0/phase0_segformer_b0_dynk16_s${SEED}_residual_emasave_stagewise_a010/ckpt/best.pt"
  else
    dep="$dyn_a025_jid"
    ckpt="$ROOT/runs/refuge2_phase0/phase0_segformer_b0_dynk16_s${SEED}_residual_emasave_stagewise_a025/ckpt/best.pt"
  fi

  for alpha in 0.00 0.05 0.10 0.15 0.25 0.40; do
    jid="$(
      CKPT="$ckpt" \
      VARIANT=dynk16 \
      ALPHA_GRAPH="$alpha" \
      sbatch --parsable --dependency=afterok:"$dep" slurm/eval_refuge2_ckpt.sbatch
    )"
    echo "eval train_alpha=$train_alpha eval_alpha=$alpha job: $jid"
  done
done

host_eval_jid="$(
  CKPT="$host_ckpt" \
  VARIANT=no_graph \
  ALPHA_GRAPH=0.00 \
  sbatch --parsable --dependency=afterok:"$host_jid" slurm/eval_refuge2_ckpt.sbatch
)"
echo "host eval calibration job: $host_eval_jid"

