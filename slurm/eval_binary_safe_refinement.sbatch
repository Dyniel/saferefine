#!/bin/bash
#SBATCH --job-name=bin_refine
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

DATASET="${DATASET:-binary}"
NPZ="${NPZ:?Set NPZ=/path/to/binary_dataset.npz}"
IDX_EVAL="${IDX_EVAL:?Set IDX_EVAL=/path/to/eval.txt}"
CKPT="${CKPT:?Set CKPT=/path/to/binary_host/ckpt/best.pt}"

IMG_SIZE="${IMG_SIZE:-352}"
ARCH="${ARCH:-graphseg}"
BACKBONE="${BACKBONE:-segformer_b0}"
UNET_BASE="${UNET_BASE:-32}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-2}"
MORPH_KS="${MORPH_KS:-3,5,7,9,11}"
MAX_BATCHES="${MAX_BATCHES:-0}"
RUN_TAG="${RUN_TAG:-binary_refine}"
USE_MMSEG_BACKBONE="${USE_MMSEG_BACKBONE:-0}"
export USE_MMSEG_BACKBONE

cd "$ROOT"
mkdir -p "/users/scratch1/$USER/logs" results/binary_refinement

. rebuttal/slurm/common_env.sh

STAMP="$(date '+%Y%m%d_%H%M%S')"
SAFE_TAG="${RUN_TAG//\//_}_${DATASET}_${SLURM_JOB_ID:-local}_${STAMP}"
OUT_CSV="${OUT_CSV:-${ROOT}/results/binary_refinement/${SAFE_TAG}.csv}"
OUT_SUMMARY="${OUT_SUMMARY:-${ROOT}/results/binary_refinement/${SAFE_TAG}.json}"

echo "[$(date '+%F %T')] bin_refine ROOT=${ROOT}"
echo "[$(date '+%F %T')] bin_refine DATASET=${DATASET} CKPT=${CKPT}"
echo "[$(date '+%F %T')] bin_refine ARCH=${ARCH} BACKBONE=${BACKBONE} UNET_BASE=${UNET_BASE}"
echo "[$(date '+%F %T')] bin_refine NPZ=${NPZ} IDX_EVAL=${IDX_EVAL}"
echo "[$(date '+%F %T')] bin_refine MORPH_KS=${MORPH_KS}"
echo "[$(date '+%F %T')] bin_refine OUT_SUMMARY=${OUT_SUMMARY}"

"$PYTHON" tools/eval_binary_safe_refinement.py \
  --npz_all "$NPZ" \
  --idx_eval "$IDX_EVAL" \
  --ckpt "$CKPT" \
  --arch "$ARCH" \
  --backbone "$BACKBONE" \
  --unet_base "$UNET_BASE" \
  --img_size "$IMG_SIZE" \
  --batch "$BATCH" \
  --workers "$WORKERS" \
  --morph_ks "$MORPH_KS" \
  --max_batches "$MAX_BATCHES" \
  --out_csv "$OUT_CSV" \
  --out_summary "$OUT_SUMMARY" \
  --data_verbose 1

echo "[$(date '+%F %T')] bin_refine done"
