#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
WAVE="${WAVE:-main}"
cd "$ROOT"

submit_one() {
  local dataset="$1"
  local run_tag="$2"
  local arch="$3"
  local ckpt="$4"
  local unet_base="${5:-32}"

  DATASET="$dataset" \
  RUN_TAG="$run_tag" \
  ARCH="$arch" \
  UNET_BASE="$unet_base" \
  CKPT="$ckpt" \
  sbatch slurm/run_refiner_zoo_one_serial.sbatch
}

case "$WAVE" in
  main)
    submit_one isic2018_task1 isic_graphseg_phase0 graphseg runs/binary_host/isic2018_task1_segformer_b0_s1_phase0_ampfix/ckpt/best.pt
    submit_one kvasir_seg kvasir_graphseg_e120 graphseg runs/binary_host/kvasir_seg_graphseg_segformer_b0_s1_endo_graphseg_e120/ckpt/best.pt
    submit_one kvasir_seg kvasir_unet_e120 unet_small runs/binary_host/kvasir_seg_unet_b32_s1_endo_unet_e120/ckpt/best.pt 32
    submit_one ph2 ph2_unet_e120 unet_small runs/binary_host/ph2_unet_b32_s1_unet_e120/ckpt/best.pt 32
    ;;
  stress)
    submit_one kvasir_sessile sessile_graphseg_e120 graphseg runs/binary_host/kvasir_sessile_graphseg_segformer_b0_s1_endo_graphseg_e120/ckpt/best.pt
    submit_one kvasir_sessile sessile_unet_e120 unet_small runs/binary_host/kvasir_sessile_unet_b32_s1_endo_unet_e120/ckpt/best.pt 32
    submit_one polyps_official polyps_graphseg_e120 graphseg runs/binary_host/polyps_official_graphseg_segformer_b0_s1_endo_graphseg_e120/ckpt/best.pt
    submit_one polyps_official polyps_unet_e120 unet_small runs/binary_host/polyps_official_unet_b32_s1_endo_unet_e120/ckpt/best.pt 32
    ;;
  *)
    echo "Unknown WAVE=${WAVE}. Use WAVE=main or WAVE=stress." >&2
    exit 2
    ;;
esac
