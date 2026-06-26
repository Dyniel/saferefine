#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
cd "$ROOT"

smoke_jid="$(
  EPOCHS=2 \
  IMG_SIZE=256 \
  BATCH=2 \
  HIDDEN=32 \
  STEPS=4 \
  MAX_TRAIN_BATCHES=8 \
  MAX_VAL_BATCHES=8 \
  RUN_TAG=smoke \
  sbatch --parsable slurm/train_graphmembrane_refuge2.sbatch
)"
echo "graphmembrane smoke job: $smoke_jid"

full_jid="$(
  EPOCHS="${EPOCHS:-30}" \
  IMG_SIZE="${IMG_SIZE:-512}" \
  BATCH="${BATCH:-4}" \
  HIDDEN="${HIDDEN:-48}" \
  STEPS="${STEPS:-6}" \
  LR="${LR:-2e-4}" \
  RUN_TAG="${RUN_TAG:-phase0}" \
  sbatch --parsable --dependency=afterok:"$smoke_jid" slurm/train_graphmembrane_refuge2.sbatch
)"
echo "graphmembrane full job: $full_jid"

