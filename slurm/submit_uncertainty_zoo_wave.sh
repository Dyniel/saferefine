#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
cd "$ROOT"

IMG_SIZE="${IMG_SIZE:-352}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-2}"
OUT_DIR="${OUT_DIR:-${ROOT}/results/refiner_zoo_uncert}"
mkdir -p "$OUT_DIR"

submit_one() {
  local dataset="$1"
  local run_tag="$2"
  local arch="$3"
  local unet_base="${4:-32}"
  local backbone="${5:-segformer_b0}"

  local arch_tag="${arch}_${backbone}"
  if [ "$arch" = "unet_small" ] || [ "$arch" = "unet" ] || [ "$arch" = "simple_unet" ]; then
    arch_tag="unet_b${unet_base}"
  fi

  local npz="${ROOT}/data_npz/${dataset}_352.npz"
  local split_dir="${ROOT}/dataset_splits/${dataset}_352"
  local ckpt="${ROOT}/runs/binary_host/${dataset}_${arch_tag}_s1_${run_tag}/ckpt/best.pt"
  local out_csv="${OUT_DIR}/${dataset}_${run_tag}_zoo.csv"
  local out_json="${OUT_DIR}/${dataset}_${run_tag}_zoo.json"

  if [ ! -f "$ckpt" ]; then
    echo "missing checkpoint: $ckpt" >&2
    exit 2
  fi

  DATASET="$dataset" \
  RUN_TAG="${run_tag}_uncert" \
  NPZ="$npz" \
  IDX_EVAL="${split_dir}/test.txt" \
  CKPT="$ckpt" \
  ARCH="$arch" \
  BACKBONE="$backbone" \
  UNET_BASE="$unet_base" \
  IMG_SIZE="$IMG_SIZE" \
  BATCH="$BATCH" \
  WORKERS="$WORKERS" \
  OUT_CSV="$out_csv" \
  OUT_SUMMARY="$out_json" \
  sbatch slurm/eval_binary_refiner_zoo.sbatch
}

WAVE="${WAVE:-all}"
case "$WAVE" in
  core)
    submit_one isic2018_task1 mediafinal_unet_e120 unet_small 32
    submit_one kvasir_seg mediafinal_graphseg_e120 graphseg 32
    submit_one kvasir_seg mediafinal_unet_e120 unet_small 32
    submit_one ph2 mediafinal_unet_e120 unet_small 32
    ;;
  stress)
    submit_one polyps_official mediafinal_graphseg_e120 graphseg 32
    submit_one polyps_official mediafinal_unet_e120 unet_small 32
    submit_one msd_heart_mri mediafinal_graphseg_mri_e120 graphseg 32
    submit_one msd_heart_mri mediafinal_unet_mri_e120 unet_small 32
    ;;
  all)
    submit_one isic2018_task1 mediafinal_unet_e120 unet_small 32
    submit_one kvasir_seg mediafinal_graphseg_e120 graphseg 32
    submit_one kvasir_seg mediafinal_unet_e120 unet_small 32
    submit_one ph2 mediafinal_unet_e120 unet_small 32
    submit_one polyps_official mediafinal_graphseg_e120 graphseg 32
    submit_one polyps_official mediafinal_unet_e120 unet_small 32
    submit_one msd_heart_mri mediafinal_graphseg_mri_e120 graphseg 32
    submit_one msd_heart_mri mediafinal_unet_mri_e120 unet_small 32
    ;;
  *)
    echo "Unknown WAVE=${WAVE}. Use core, stress, or all." >&2
    exit 2
    ;;
esac
