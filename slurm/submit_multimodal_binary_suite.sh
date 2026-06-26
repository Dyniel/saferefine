#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
cd "$ROOT"

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
RUN_TAG="${RUN_TAG:-phase0}"
DATASETS="${DATASETS:-kvasir_seg kvasir_sessile ph2 polyps_official isic2018_task1}"
SUBMIT_POLICIES="${SUBMIT_POLICIES:-1}"

for dataset in $DATASETS; do
  npz="$ROOT/data_npz/${dataset}_${IMG_SIZE}.npz"
  split_dir="$ROOT/dataset_splits/${dataset}_${IMG_SIZE}"
  if [ ! -f "$npz" ]; then
    echo "skip ${dataset}: missing ${npz}"
    continue
  fi
  if [ ! -f "$split_dir/train.txt" ] || [ ! -f "$split_dir/val.txt" ]; then
    echo "skip ${dataset}: missing train/val split in ${split_dir}"
    continue
  fi
  echo "submit ${dataset}"
  DATASET="$dataset" \
  NPZ="$npz" \
  SPLIT_DIR="$split_dir" \
  IMG_SIZE="$IMG_SIZE" \
  ARCH="$ARCH" \
  BACKBONE="$BACKBONE" \
  UNET_BASE="$UNET_BASE" \
  EPOCHS="$EPOCHS" \
  BATCH="$BATCH" \
  WORKERS="$WORKERS" \
  LR="$LR" \
  WEIGHT_DECAY="$WEIGHT_DECAY" \
  BOUNDARY_W="$BOUNDARY_W" \
  AMP="$AMP" \
  MORPH_KS="$MORPH_KS" \
  RUN_TAG="$RUN_TAG" \
  SUBMIT_POLICIES="$SUBMIT_POLICIES" \
  bash slurm/submit_binary_safe_refinement.sh
done
