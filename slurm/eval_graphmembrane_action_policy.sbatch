#!/bin/bash
#SBATCH --job-name=graphmem_policy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/users/scratch1/%u/logs/%x_%j.out

set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
CONDA_ENV="${CONDA_ENV:-grm_media_py312}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-}"
PYTHON="${PYTHON:-}"

INPUTS="${INPUTS:-}"
RISK_SCORE="${RISK_SCORE:-changed}"
CAL_FRACTION="${CAL_FRACTION:-0.5}"
HARM_EPS="${HARM_EPS:-0.0}"
BETA_HARM="${BETA_HARM:-2.0}"
MAX_CAL_HARM_RATE="${MAX_CAL_HARM_RATE:-0.25}"
CRC_CONFIDENCE="${CRC_CONFIDENCE:-0.10}"
RUN_TAG="${RUN_TAG:-graphmem_policy}"

cd "$ROOT"
mkdir -p "/users/scratch1/$USER/logs" results/graphmembrane_policy

. rebuttal/slurm/common_env.sh

STAMP="$(date '+%Y%m%d_%H%M%S')"
SAFE_TAG="${RUN_TAG//\//_}_${RISK_SCORE}_${SLURM_JOB_ID:-local}_${STAMP}"
OUT_CSV="${OUT_CSV:-${ROOT}/results/graphmembrane_policy/${SAFE_TAG}.csv}"
OUT_SUMMARY="${OUT_SUMMARY:-${ROOT}/results/graphmembrane_policy/${SAFE_TAG}.json}"

echo "[$(date '+%F %T')] graphmem_policy ROOT=${ROOT}"
echo "[$(date '+%F %T')] graphmem_policy INPUTS=${INPUTS}"
echo "[$(date '+%F %T')] graphmem_policy RISK_SCORE=${RISK_SCORE} MAX_CAL_HARM_RATE=${MAX_CAL_HARM_RATE} CRC_CONFIDENCE=${CRC_CONFIDENCE}"
echo "[$(date '+%F %T')] graphmem_policy OUT_SUMMARY=${OUT_SUMMARY}"

"$PYTHON" tools/eval_safe_action_portfolio.py \
  --inputs "$INPUTS" \
  --risk_score "$RISK_SCORE" \
  --cal_fraction "$CAL_FRACTION" \
  --harm_eps "$HARM_EPS" \
  --beta_harm "$BETA_HARM" \
  --max_cal_harm_rate "$MAX_CAL_HARM_RATE" \
  --crc_confidence "$CRC_CONFIDENCE" \
  --out_csv "$OUT_CSV" \
  --out_summary "$OUT_SUMMARY"

echo "[$(date '+%F %T')] graphmem_policy done"
