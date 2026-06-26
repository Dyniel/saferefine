#!/bin/bash
#SBATCH --job-name=grm_env312
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
ENV_PATH="${ENV_PATH:-/users/scratch1/$USER/conda_envs/grm_media_py312}"
PY_VER="${PY_VER:-3.12}"

cd "$ROOT"
mkdir -p "/users/scratch1/$USER/logs"

echo "[$(date '+%F %T')] setup env ROOT=${ROOT}"
echo "[$(date '+%F %T')] setup env ENV_PATH=${ENV_PATH}"
echo "[$(date '+%F %T')] setup env PY_VER=${PY_VER}"
nvidia-smi || true

ENV_PATH="$ENV_PATH" PY_VER="$PY_VER" bash rebuttal/setup_mmseg_env.sh

echo "[$(date '+%F %T')] setup env done"

