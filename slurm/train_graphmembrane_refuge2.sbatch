#!/bin/bash
#SBATCH --job-name=graphmem_refuge2
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/users/scratch1/%u/logs/%x_%j.out

set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
CONDA_ENV="${CONDA_ENV:-grm_media_py312}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-}"
PYTHON="${PYTHON:-}"

NPZ="${NPZ:-/users/scratch1/dancies/jaskra_repo/jaskra/data/REFUGE2/refuge2_all_512_fixed.npz}"
IDX_TRAIN="${IDX_TRAIN:-${ROOT}/dataset_splits/refuge2_all512_fixed/train.txt}"
IDX_VAL="${IDX_VAL:-${ROOT}/dataset_splits/refuge2_all512_fixed/val.txt}"
OUT_DIR="${OUT_DIR:-${ROOT}/runs/graphmembrane}"

SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-4}"
WORKERS="${WORKERS:-0}"
IMG_SIZE="${IMG_SIZE:-512}"
LR="${LR:-2e-4}"
HIDDEN="${HIDDEN:-48}"
STEPS="${STEPS:-6}"
DT="${DT:-0.35}"
ALPHA="${ALPHA:-1.0}"
RESIDUAL_CLIP="${RESIDUAL_CLIP:-5.0}"
LAMBDA_ENERGY="${LAMBDA_ENERGY:-0.02}"
LAMBDA_RESIDUAL="${LAMBDA_RESIDUAL:-0.01}"
SOURCE="${SOURCE:-synthetic}"
MIXED_HOST_PROB="${MIXED_HOST_PROB:-0.5}"
HOST_CKPT="${HOST_CKPT:-}"
HOST_INPUT_MODE="${HOST_INPUT_MODE:-hard}"
HOST_SOFT_CLIP="${HOST_SOFT_CLIP:-5.0}"
RESUME_MEMBRANE="${RESUME_MEMBRANE:-}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"
RUN_TAG="${RUN_TAG:-}"
AMP="${AMP:-0}"

cd "$ROOT"
mkdir -p "/users/scratch1/$USER/logs" "$OUT_DIR"

. rebuttal/slurm/common_env.sh

RUN_NAME="graphmembrane_refuge2_s${SEED}_h${HIDDEN}_steps${STEPS}"
if [ -n "$RUN_TAG" ]; then
  RUN_NAME="${RUN_NAME}_${RUN_TAG}"
fi

echo "[$(date '+%F %T')] graphmem ROOT=${ROOT}"
echo "[$(date '+%F %T')] graphmem RUN_NAME=${RUN_NAME}"
echo "[$(date '+%F %T')] graphmem NPZ=${NPZ}"
echo "[$(date '+%F %T')] graphmem IDX_TRAIN=${IDX_TRAIN}"
echo "[$(date '+%F %T')] graphmem IDX_VAL=${IDX_VAL}"
echo "[$(date '+%F %T')] graphmem EPOCHS=${EPOCHS} BATCH=${BATCH} IMG_SIZE=${IMG_SIZE} LR=${LR}"
echo "[$(date '+%F %T')] graphmem HIDDEN=${HIDDEN} STEPS=${STEPS} DT=${DT} ALPHA=${ALPHA} CLIP=${RESIDUAL_CLIP} LAMBDA_ENERGY=${LAMBDA_ENERGY}"
echo "[$(date '+%F %T')] graphmem SOURCE=${SOURCE} HOST_INPUT_MODE=${HOST_INPUT_MODE} HOST_CKPT=${HOST_CKPT:-<none>} RESUME_MEMBRANE=${RESUME_MEMBRANE:-<none>}"

"$PYTHON" tools/train_graphmembrane_refiner.py \
  --npz_all "$NPZ" \
  --idx_train "$IDX_TRAIN" \
  --idx_val "$IDX_VAL" \
  --img_size "$IMG_SIZE" \
  --batch "$BATCH" \
  --workers "$WORKERS" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --hidden "$HIDDEN" \
  --steps "$STEPS" \
  --dt "$DT" \
  --alpha "$ALPHA" \
  --residual_clip "$RESIDUAL_CLIP" \
  --lambda_energy "$LAMBDA_ENERGY" \
  --lambda_residual "$LAMBDA_RESIDUAL" \
  --source "$SOURCE" \
  --mixed_host_prob "$MIXED_HOST_PROB" \
  --host_ckpt "$HOST_CKPT" \
  --host_input_mode "$HOST_INPUT_MODE" \
  --host_soft_clip "$HOST_SOFT_CLIP" \
  --resume_membrane "$RESUME_MEMBRANE" \
  --seed "$SEED" \
  --out_dir "$OUT_DIR" \
  --run_name "$RUN_NAME" \
  --amp "$AMP" \
  --max_train_batches "$MAX_TRAIN_BATCHES" \
  --max_val_batches "$MAX_VAL_BATCHES" \
  --data_verbose 1

echo "[$(date '+%F %T')] graphmem done checkpoint=${OUT_DIR}/${RUN_NAME}/ckpt/best.pt"
