#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
IMG_SIZE="${IMG_SIZE:-352}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-2}"
MORPH_KS="${MORPH_KS:-3,5,7,9,11}"
WAVE="${WAVE:-1}"

cd "$ROOT"

submit_one() {
  local dataset="$1"
  local arch="$2"
  local backbone="$3"
  local unet_base="$4"
  local run_tag="$5"
  local epochs="$6"
  local lr="$7"
  local boundary_w="$8"
  local weight_decay="$9"

  local npz="${ROOT}/data_npz/${dataset}_${IMG_SIZE}.npz"
  local split_dir="${ROOT}/dataset_splits/${dataset}_${IMG_SIZE}"
  if [ ! -f "$npz" ]; then
    echo "missing npz: $npz" >&2
    exit 2
  fi
  if [ ! -f "$split_dir/train.txt" ] || [ ! -f "$split_dir/test.txt" ]; then
    echo "missing split files in: $split_dir" >&2
    exit 2
  fi

  echo "submit ${dataset} ${arch} ${run_tag}"
  DATASET="$dataset" \
  NPZ="$npz" \
  SPLIT_DIR="$split_dir" \
  IMG_SIZE="$IMG_SIZE" \
  ARCH="$arch" \
  BACKBONE="$backbone" \
  UNET_BASE="$unet_base" \
  EPOCHS="$epochs" \
  BATCH="$BATCH" \
  WORKERS="$WORKERS" \
  LR="$lr" \
  WEIGHT_DECAY="$weight_decay" \
  BOUNDARY_W="$boundary_w" \
  MORPH_KS="$MORPH_KS" \
  RUN_TAG="$run_tag" \
  SUBMIT_POLICIES=0 \
  bash slurm/submit_binary_safe_refinement.sh
}

case "$WAVE" in
  1)
    # 4 configs x (train + eval) = 8 submitted jobs.
    submit_one kvasir_seg       graphseg   segformer_b0 32 endo_graphseg_e120 120 1e-4 0.0 3e-5
    submit_one polyps_official  graphseg   segformer_b0 32 endo_graphseg_e120 120 1e-4 0.0 3e-5
    submit_one kvasir_seg       unet_small segformer_b0 32 endo_unet_e120     120 1e-3 0.0 1e-4
    submit_one polyps_official  unet_small segformer_b0 32 endo_unet_e120     120 1e-3 0.0 1e-4
    ;;
  2)
    # 4 configs x (train + eval) = 8 submitted jobs.
    submit_one kvasir_sessile   graphseg   segformer_b0 32 endo_graphseg_e120 120 1e-4 0.0 3e-5
    submit_one kvasir_sessile   unet_small segformer_b0 32 endo_unet_e120     120 1e-3 0.0 1e-4
    submit_one ph2              unet_small segformer_b0 32 unet_e120          120 1e-3 0.0 1e-4
    submit_one isic2018_task1   unet_small segformer_b0 32 unet_e120          120 1e-3 0.0 1e-4
    ;;
  *)
    echo "Unknown WAVE=$WAVE. Use WAVE=1 or WAVE=2." >&2
    exit 2
    ;;
esac
