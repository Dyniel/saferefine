#!/bin/bash
#SBATCH --job-name=bin_zoo
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/users/scratch1/%u/logs/%x_%j.out

set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
CONDA_ENV="${CONDA_ENV:-grm_media_py312}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-}"
PYTHON="${PYTHON:-}"

DATASET="${DATASET:?Set DATASET=name}"
NPZ="${NPZ:?Set NPZ=/path/to/dataset.npz}"
IDX_EVAL="${IDX_EVAL:?Set IDX_EVAL=/path/to/test.txt}"
CKPT="${CKPT:?Set CKPT=/path/to/best.pt}"
RUN_TAG="${RUN_TAG:-zoo}"

IMG_SIZE="${IMG_SIZE:-352}"
ARCH="${ARCH:-graphseg}"
BACKBONE="${BACKBONE:-segformer_b0}"
UNET_BASE="${UNET_BASE:-32}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-2}"
MORPH_KS="${MORPH_KS:-3,5,7,9,11}"
MIN_AREAS="${MIN_AREAS:-25,50,100,250}"
GAUSSIAN_SIGMAS="${GAUSSIAN_SIGMAS:-0.75,1.25,2.0}"
BILATERAL_SIGMAS="${BILATERAL_SIGMAS:-0.05,0.10,0.20}"
MAX_BATCHES="${MAX_BATCHES:-0}"
OUT_CSV="${OUT_CSV:-${ROOT}/results/refiner_zoo/${DATASET}_${RUN_TAG}_zoo.csv}"
OUT_SUMMARY="${OUT_SUMMARY:-${ROOT}/results/refiner_zoo/${DATASET}_${RUN_TAG}_zoo.json}"

cd "$ROOT"
mkdir -p "/users/scratch1/$USER/logs" results/refiner_zoo

. rebuttal/slurm/common_env.sh

echo "[$(date '+%F %T')] bin_zoo DATASET=${DATASET} RUN_TAG=${RUN_TAG}"
echo "[$(date '+%F %T')] bin_zoo CKPT=${CKPT}"
echo "[$(date '+%F %T')] bin_zoo ARCH=${ARCH} BACKBONE=${BACKBONE} UNET_BASE=${UNET_BASE}"
echo "[$(date '+%F %T')] bin_zoo OUT_CSV=${OUT_CSV}"

"$PYTHON" tools/eval_binary_refiner_zoo.py \
  --npz_all "$NPZ" \
  --idx_eval "$IDX_EVAL" \
  --ckpt "$CKPT" \
  --out_csv "$OUT_CSV" \
  --out_summary "$OUT_SUMMARY" \
  --arch "$ARCH" \
  --backbone "$BACKBONE" \
  --unet_base "$UNET_BASE" \
  --img_size "$IMG_SIZE" \
  --batch "$BATCH" \
  --workers "$WORKERS" \
  --morph_ks "$MORPH_KS" \
  --min_areas "$MIN_AREAS" \
  --gaussian_sigmas "$GAUSSIAN_SIGMAS" \
  --bilateral_sigmas "$BILATERAL_SIGMAS" \
  --max_batches "$MAX_BATCHES" \
  --data_verbose 1

echo "[$(date '+%F %T')] bin_zoo done"
