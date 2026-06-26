#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
cd "$ROOT"

DATASET="${DATASET:?Set DATASET=polyp|skin|...}"
NPZ="${NPZ:?Set NPZ=/path/to/binary_dataset.npz}"
SPLIT_DIR="${SPLIT_DIR:?Set SPLIT_DIR=/path/to/splits}"
IDX_TRAIN="${IDX_TRAIN:-$SPLIT_DIR/train.txt}"
IDX_VAL="${IDX_VAL:-$SPLIT_DIR/val.txt}"
IDX_TEST="${IDX_TEST:-$SPLIT_DIR/test.txt}"
SEED="${SEED:-1}"
IMG_SIZE="${IMG_SIZE:-352}"
ARCH="${ARCH:-graphseg}"
BACKBONE="${BACKBONE:-segformer_b0}"
UNET_BASE="${UNET_BASE:-32}"
EPOCHS="${EPOCHS:-40}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-2}"
LR="${LR:-2.5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-3e-5}"
BOUNDARY_W="${BOUNDARY_W:-0.1}"
AMP="${AMP:-1}"
MORPH_KS="${MORPH_KS:-3,5,7,9,11}"
OUT_DIR="${OUT_DIR:-$ROOT/runs/binary_host}"
SUBMIT_POLICIES="${SUBMIT_POLICIES:-1}"

train_jid="$(
  DATASET="$DATASET" \
  NPZ="$NPZ" \
  IDX_TRAIN="$IDX_TRAIN" \
  IDX_VAL="$IDX_VAL" \
  IMG_SIZE="$IMG_SIZE" \
  ARCH="$ARCH" \
  BACKBONE="$BACKBONE" \
  UNET_BASE="$UNET_BASE" \
  SEED="$SEED" \
  EPOCHS="$EPOCHS" \
  BATCH="$BATCH" \
  WORKERS="$WORKERS" \
  LR="$LR" \
  WEIGHT_DECAY="$WEIGHT_DECAY" \
  BOUNDARY_W="$BOUNDARY_W" \
  AMP="$AMP" \
  OUT_DIR="$OUT_DIR" \
  RUN_TAG="${RUN_TAG:-phase0}" \
  sbatch --parsable slurm/train_binary_host.sbatch
)"
echo "binary host train job: $train_jid"

arch_tag="${ARCH}_${BACKBONE}"
if [ "$ARCH" = "unet_small" ] || [ "$ARCH" = "unet" ] || [ "$ARCH" = "simple_unet" ]; then
  arch_tag="unet_b${UNET_BASE}"
fi
run_name="${DATASET}_${arch_tag}_s${SEED}_${RUN_TAG:-phase0}"
ckpt="$OUT_DIR/$run_name/ckpt/best.pt"
safe_tag="${RUN_TAG:-phase0}"
eval_csv="$ROOT/results/binary_refinement/${DATASET}_${safe_tag}_eval.csv"
eval_json="$ROOT/results/binary_refinement/${DATASET}_${safe_tag}_eval.json"

eval_jid="$(
  DATASET="$DATASET" \
  NPZ="$NPZ" \
  IDX_EVAL="$IDX_TEST" \
  CKPT="$ckpt" \
  OUT_CSV="$eval_csv" \
  OUT_SUMMARY="$eval_json" \
  IMG_SIZE="$IMG_SIZE" \
  ARCH="$ARCH" \
  BACKBONE="$BACKBONE" \
  UNET_BASE="$UNET_BASE" \
  BATCH="$BATCH" \
  WORKERS="$WORKERS" \
  MORPH_KS="$MORPH_KS" \
  RUN_TAG="${DATASET}_${safe_tag}" \
  sbatch --parsable --dependency=afterok:"$train_jid" slurm/eval_binary_safe_refinement.sbatch
)"
echo "binary refinement eval job: $eval_jid"

if [ "$SUBMIT_POLICIES" = "1" ]; then
  DATASET="$DATASET" \
  EVAL_CSV="$eval_csv" \
  EVAL_JID="$eval_jid" \
  PRACTICAL_MAX_CAL_HARM_RATE="${PRACTICAL_MAX_CAL_HARM_RATE:-0.25}" \
  STRICT_MAX_CAL_HARM_RATE="${STRICT_MAX_CAL_HARM_RATE:-0.0}" \
  bash slurm/submit_binary_policies.sh
else
  echo "binary policy jobs skipped; submit later with:"
  echo "  DATASET=$DATASET EVAL_CSV=$eval_csv EVAL_JID=$eval_jid bash slurm/submit_binary_policies.sh"
fi
