#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
cd "$ROOT"

# Two serial jobs: each trains one host, evaluates baseline refiners, evaluates
# the refiner zoo, and runs practical/strict policies for changed/geom/change+geom.
# This stays safely below the user's max-8 submitted-job preference.

submit_one() {
  local run_tag="$1"
  local arch="$2"
  local lr="$3"
  local weight_decay="$4"
  local unet_base="${5:-32}"

  DATASET=msd_heart_mri \
  RUN_TAG="$run_tag" \
  NPZ="${ROOT}/data_npz/msd_heart_mri_352.npz" \
  SPLIT_DIR="${ROOT}/dataset_splits/msd_heart_mri_352" \
  ARCH="$arch" \
  UNET_BASE="$unet_base" \
  LR="$lr" \
  WEIGHT_DECAY="$weight_decay" \
  BOUNDARY_W="${BOUNDARY_W:-0.0}" \
  EPOCHS="${EPOCHS:-120}" \
  BATCH="${BATCH:-8}" \
  sbatch slurm/run_binary_final_serial.sbatch
}

submit_one mediafinal_graphseg_mri_e120 graphseg 1e-4 3e-5
submit_one mediafinal_unet_mri_e120 unet_small 1e-3 1e-4 32
