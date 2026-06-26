#!/bin/bash
#SBATCH --job-name=grm_smoke
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/users/scratch1/%u/logs/%x_%j.out

set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
CONDA_ENV="${CONDA_ENV:-grm_media_py312}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-}"
PYTHON="${PYTHON:-}"
GRAPH_OUTPUT_MODE="${GRAPH_OUTPUT_MODE:-residual}"
USE_MMSEG_BACKBONE="${USE_MMSEG_BACKBONE:-0}"
export USE_MMSEG_BACKBONE

cd "$ROOT"
mkdir -p "/users/scratch1/$USER/logs"

. rebuttal/slurm/common_env.sh

echo "[$(date '+%F %T')] smoke ROOT=${ROOT}"
nvidia-smi || true

"$PYTHON" -V
"$PYTHON" rebuttal/check_rebuttal_env.py
"$PYTHON" tools/unified_graphseg.py \
  --mode smoke \
  --backbone segformer_b0 \
  --img_size 128 \
  --graph_output_mode "$GRAPH_OUTPUT_MODE"

echo "[$(date '+%F %T')] smoke done"
