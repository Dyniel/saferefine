#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
cd "$ROOT"

run_one() {
  local dataset="$1"
  local run_tag="$2"
  local arch="$3"
  local ckpt="$4"
  local unet_base="${5:-32}"

  echo "submit zoo+policies: ${dataset} ${run_tag}"
  before="$(mktemp)"
  after="$(mktemp)"
  squeue -h -u "$USER" -o "%A" > "$before" || true

  DATASET="$dataset" \
  RUN_TAG="$run_tag" \
  ARCH="$arch" \
  UNET_BASE="$unet_base" \
  CKPT="$ckpt" \
  bash slurm/submit_refiner_zoo_one.sh

  squeue -h -u "$USER" -o "%A" > "$after" || true
  new_jobs="$(comm -13 <(sort "$before") <(sort "$after") | tr '\n' ',' | sed 's/,$//')"
  rm -f "$before" "$after"

  if [ -z "$new_jobs" ]; then
    echo "warning: could not detect new jobs; sleeping ${WAIT_SECONDS}s before next batch"
    sleep "$WAIT_SECONDS"
    return
  fi
  echo "waiting for jobs: $new_jobs"
  while squeue -h -j "$new_jobs" | grep -q .; do
    sleep "$WAIT_SECONDS"
  done
}

case "${WAVE:-main}" in
  main)
    run_one isic2018_task1 isic_graphseg_phase0 graphseg runs/binary_host/isic2018_task1_segformer_b0_s1_phase0_ampfix/ckpt/best.pt
    run_one kvasir_seg kvasir_graphseg_e120 graphseg runs/binary_host/kvasir_seg_graphseg_segformer_b0_s1_endo_graphseg_e120/ckpt/best.pt
    run_one kvasir_seg kvasir_unet_e120 unet_small runs/binary_host/kvasir_seg_unet_b32_s1_endo_unet_e120/ckpt/best.pt 32
    run_one ph2 ph2_unet_e120 unet_small runs/binary_host/ph2_unet_b32_s1_unet_e120/ckpt/best.pt 32
    ;;
  stress)
    run_one kvasir_sessile sessile_graphseg_e120 graphseg runs/binary_host/kvasir_sessile_graphseg_segformer_b0_s1_endo_graphseg_e120/ckpt/best.pt
    run_one kvasir_sessile sessile_unet_e120 unet_small runs/binary_host/kvasir_sessile_unet_b32_s1_endo_unet_e120/ckpt/best.pt 32
    run_one polyps_official polyps_graphseg_e120 graphseg runs/binary_host/polyps_official_graphseg_segformer_b0_s1_endo_graphseg_e120/ckpt/best.pt
    run_one polyps_official polyps_unet_e120 unet_small runs/binary_host/polyps_official_unet_b32_s1_endo_unet_e120/ckpt/best.pt 32
    ;;
  *)
    echo "Unknown WAVE=${WAVE}. Use WAVE=main or WAVE=stress." >&2
    exit 2
    ;;
esac
