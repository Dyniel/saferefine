#!/bin/bash
#SBATCH --job-name=zoo_orch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=12:00:00
#SBATCH --output=/users/scratch1/%u/logs/%x_%j.out

set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
WAVE="${WAVE:-main}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"

cd "$ROOT"
mkdir -p "/users/scratch1/$USER/logs"

echo "[$(date '+%F %T')] zoo_orch WAVE=${WAVE}"
WAVE="$WAVE" WAIT_SECONDS="$WAIT_SECONDS" bash slurm/orchestrate_refiner_zoo.sh
echo "[$(date '+%F %T')] zoo_orch done"
