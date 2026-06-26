#!/bin/bash
#SBATCH --job-name=build_npz
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/users/scratch1/%u/logs/%x_%j.out

set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
STAGING="${STAGING:-/users/scratch1/dancies/datasets_staging/grm_media}"
PYTHON="${PYTHON:-/users/scratch1/dancies/conda_envs/py312/bin/python}"
IMG_SIZE="${IMG_SIZE:-352}"
COMPRESS="${COMPRESS:-0}"
BUILD_ISIC="${BUILD_ISIC:-1}"
BUILD_POLYPS="${BUILD_POLYPS:-1}"

cd "$ROOT"
mkdir -p data_npz dataset_splits "/users/scratch1/$USER/logs"

echo "[$(date '+%F %T')] build_npz ROOT=${ROOT}"
echo "[$(date '+%F %T')] build_npz STAGING=${STAGING}"
echo "[$(date '+%F %T')] build_npz PYTHON=${PYTHON}"
echo "[$(date '+%F %T')] build_npz IMG_SIZE=${IMG_SIZE} COMPRESS=${COMPRESS}"

if [ "$BUILD_ISIC" = "1" ]; then
  "$PYTHON" tools/build_binary_npz_from_split_dirs.py \
    --train_image_dir "$STAGING/isic/ISIC2018_Task1-2_Training_Input" \
    --train_mask_dir "$STAGING/isic/ISIC2018_Task1_Training_GroundTruth" \
    --val_image_dir "$STAGING/isic/ISIC2018_Task1-2_Validation_Input" \
    --val_mask_dir "$STAGING/isic/ISIC2018_Task1_Validation_GroundTruth" \
    --test_image_dir "$STAGING/isic/ISIC2018_Task1-2_Test_Input" \
    --test_mask_dir "$STAGING/isic/ISIC2018_Task1_Test_GroundTruth" \
    --out_npz data_npz/isic2018_task1_${IMG_SIZE}.npz \
    --split_dir dataset_splits/isic2018_task1_${IMG_SIZE} \
    --name isic2018_task1_${IMG_SIZE} \
    --img_size "$IMG_SIZE" \
    --compress "$COMPRESS"
fi

if [ "$BUILD_POLYPS" = "1" ]; then
  "$PYTHON" tools/build_binary_npz_from_split_dirs.py \
    --train_image_dir "$STAGING/polyps/NewTRimage" \
    --train_mask_dir "$STAGING/polyps/NewTRmask" \
    --test_image_dir "$STAGING/polyps_test_flat/images" \
    --test_mask_dir "$STAGING/polyps_test_flat/masks" \
    --out_npz data_npz/polyps_official_${IMG_SIZE}.npz \
    --split_dir dataset_splits/polyps_official_${IMG_SIZE} \
    --name polyps_official_${IMG_SIZE} \
    --img_size "$IMG_SIZE" \
    --val_from_train_frac 0.15 \
    --seed 1 \
    --compress "$COMPRESS"
fi

echo "[$(date '+%F %T')] build_npz done"
