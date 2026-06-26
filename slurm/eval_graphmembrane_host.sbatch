#!/bin/bash
#SBATCH --job-name=graphmem_host
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/users/scratch1/%u/logs/%x_%j.out

set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
CONDA_ENV="${CONDA_ENV:-grm_media_py312}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-}"
PYTHON="${PYTHON:-}"

NPZ="${NPZ:-/users/scratch1/dancies/jaskra_repo/jaskra/data/REFUGE2/refuge2_all_512_fixed.npz}"
IDX_VAL="${IDX_VAL:-${ROOT}/dataset_splits/refuge2_all512_fixed/val.txt}"
HOST_CKPT="${HOST_CKPT:-${ROOT}/runs/refuge2_phase0/phase0_segformer_b0_no_graph_s1_residual_emasave_host/ckpt/best.pt}"
MEMBRANE_CKPT="${MEMBRANE_CKPT:-${ROOT}/runs/graphmembrane/graphmembrane_refuge2_s1_h48_steps6_phase0/ckpt/best.pt}"

INPUT_MODE="${INPUT_MODE:-hard}"
ALPHAS="${ALPHAS:-0,0.10,0.25,0.50,0.75,1.00}"
BATCH="${BATCH:-4}"
WORKERS="${WORKERS:-0}"
IMG_SIZE="${IMG_SIZE:-512}"
MAX_BATCHES="${MAX_BATCHES:-0}"
RUN_TAG="${RUN_TAG:-graphmem_host}"
USE_MMSEG_BACKBONE="${USE_MMSEG_BACKBONE:-0}"
export USE_MMSEG_BACKBONE

cd "$ROOT"
mkdir -p "/users/scratch1/$USER/logs" results/graphmembrane_host

. rebuttal/slurm/common_env.sh

STAMP="$(date '+%Y%m%d_%H%M%S')"
SAFE_TAG="${RUN_TAG//\//_}_${INPUT_MODE}_${SLURM_JOB_ID:-local}_${STAMP}"
OUT_CSV="${OUT_CSV:-${ROOT}/results/graphmembrane_host/${SAFE_TAG}.csv}"
OUT_SUMMARY="${OUT_SUMMARY:-${ROOT}/results/graphmembrane_host/${SAFE_TAG}.json}"

echo "[$(date '+%F %T')] graphmem_host ROOT=${ROOT}"
echo "[$(date '+%F %T')] graphmem_host HOST_CKPT=${HOST_CKPT}"
echo "[$(date '+%F %T')] graphmem_host MEMBRANE_CKPT=${MEMBRANE_CKPT}"
echo "[$(date '+%F %T')] graphmem_host INPUT_MODE=${INPUT_MODE} ALPHAS=${ALPHAS}"
echo "[$(date '+%F %T')] graphmem_host OUT_SUMMARY=${OUT_SUMMARY}"

"$PYTHON" tools/eval_graphmembrane_on_host.py \
  --npz_all "$NPZ" \
  --idx_val "$IDX_VAL" \
  --host_ckpt "$HOST_CKPT" \
  --membrane_ckpt "$MEMBRANE_CKPT" \
  --input_mode "$INPUT_MODE" \
  --alphas "$ALPHAS" \
  --img_size "$IMG_SIZE" \
  --batch "$BATCH" \
  --workers "$WORKERS" \
  --max_batches "$MAX_BATCHES" \
  --out_csv "$OUT_CSV" \
  --out_summary "$OUT_SUMMARY" \
  --data_verbose 1

echo "[$(date '+%F %T')] graphmem_host done"

