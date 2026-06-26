#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
WAVE="${WAVE:-kvasir}"
cd "$ROOT"

submit_one() {
  local dataset="$1"
  local run_tag="$2"
  local arch="$3"
  local lr="$4"
  local weight_decay="$5"
  local unet_base="${6:-32}"

  DATASET="$dataset" \
  RUN_TAG="$run_tag" \
  ARCH="$arch" \
  UNET_BASE="$unet_base" \
  LR="$lr" \
  WEIGHT_DECAY="$weight_decay" \
  BOUNDARY_W=0.0 \
  EPOCHS=120 \
  sbatch slurm/run_binary_final_serial.sbatch
}

case "$WAVE" in
  kvasir)
    submit_one kvasir_seg mediafinal_graphseg_e120 graphseg 1e-4 3e-5
    submit_one kvasir_seg mediafinal_unet_e120 unet_small 1e-3 1e-4 32
    ;;
  polyps)
    submit_one polyps_official mediafinal_graphseg_e120 graphseg 1e-4 3e-5
    submit_one polyps_official mediafinal_unet_e120 unet_small 1e-3 1e-4 32
    ;;
  derm)
    submit_one ph2 mediafinal_unet_e120 unet_small 1e-3 1e-4 32
    submit_one isic2018_task1 mediafinal_unet_e120 unet_small 1e-3 1e-4 32
    ;;
  *)
    echo "Unknown WAVE=${WAVE}. Use kvasir, polyps, or derm." >&2
    exit 2
    ;;
esac
