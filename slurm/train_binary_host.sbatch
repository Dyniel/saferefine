#!/bin/bash
#SBATCH --job-name=bin_host
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/users/scratch1/%u/logs/%x_%j.out

set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
CONDA_ENV="${CONDA_ENV:-grm_media_py312}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-}"
PYTHON="${PYTHON:-}"

DATASET="${DATASET:-binary}"
NPZ="${NPZ:?Set NPZ=/path/to/binary_dataset.npz}"
IDX_TRAIN="${IDX_TRAIN:?Set IDX_TRAIN=/path/to/train.txt}"
IDX_VAL="${IDX_VAL:?Set IDX_VAL=/path/to/val.txt}"
OUT_DIR="${OUT_DIR:-${ROOT}/runs/binary_host}"

IMG_SIZE="${IMG_SIZE:-352}"
SEED="${SEED:-1}"
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
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"
RUN_TAG="${RUN_TAG:-}"
USE_MMSEG_BACKBONE="${USE_MMSEG_BACKBONE:-0}"
export USE_MMSEG_BACKBONE

cd "$ROOT"
mkdir -p "$OUT_DIR" "/users/scratch1/$USER/logs"

. rebuttal/slurm/common_env.sh

ARCH_TAG="${ARCH}_${BACKBONE}"
if [ "$ARCH" = "unet_small" ] || [ "$ARCH" = "unet" ] || [ "$ARCH" = "simple_unet" ]; then
  ARCH_TAG="unet_b${UNET_BASE}"
fi
RUN_NAME="${DATASET}_${ARCH_TAG}_s${SEED}"
if [ -n "$RUN_TAG" ]; then
  RUN_NAME="${RUN_NAME}_${RUN_TAG}"
fi

echo "[$(date '+%F %T')] bin_host ROOT=${ROOT}"
echo "[$(date '+%F %T')] bin_host DATASET=${DATASET} RUN_NAME=${RUN_NAME}"
echo "[$(date '+%F %T')] bin_host ARCH=${ARCH} BACKBONE=${BACKBONE} UNET_BASE=${UNET_BASE}"
echo "[$(date '+%F %T')] bin_host NPZ=${NPZ}"
echo "[$(date '+%F %T')] bin_host IDX_TRAIN=${IDX_TRAIN} IDX_VAL=${IDX_VAL}"

"$PYTHON" tools/train_binary_host.py \
  --npz_all "$NPZ" \
  --idx_train "$IDX_TRAIN" \
  --idx_val "$IDX_VAL" \
  --out_dir "$OUT_DIR" \
  --run_name "$RUN_NAME" \
  --arch "$ARCH" \
  --backbone "$BACKBONE" \
  --unet_base "$UNET_BASE" \
  --img_size "$IMG_SIZE" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --workers "$WORKERS" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --boundary_w "$BOUNDARY_W" \
  --seed "$SEED" \
  --amp "$AMP" \
  --max_train_batches "$MAX_TRAIN_BATCHES" \
  --max_val_batches "$MAX_VAL_BATCHES" \
  --data_verbose 1

echo "[$(date '+%F %T')] bin_host done checkpoint=${OUT_DIR}/${RUN_NAME}/ckpt/best.pt"
