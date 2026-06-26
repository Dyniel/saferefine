#!/bin/bash
#SBATCH --job-name=zoo_serial
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
RUN_TAG="${RUN_TAG:?Set RUN_TAG=name}"
NPZ="${NPZ:-${ROOT}/data_npz/${DATASET}_352.npz}"
SPLIT_DIR="${SPLIT_DIR:-${ROOT}/dataset_splits/${DATASET}_352}"
IDX_EVAL="${IDX_EVAL:-${SPLIT_DIR}/test.txt}"
CKPT="${CKPT:?Set CKPT=/path/to/best.pt}"

ARCH="${ARCH:-graphseg}"
BACKBONE="${BACKBONE:-segformer_b0}"
UNET_BASE="${UNET_BASE:-32}"
IMG_SIZE="${IMG_SIZE:-352}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-2}"
MORPH_KS="${MORPH_KS:-3,5,7,9,11}"
MIN_AREAS="${MIN_AREAS:-25,50,100,250}"
GAUSSIAN_SIGMAS="${GAUSSIAN_SIGMAS:-0.75,1.25,2.0}"
BILATERAL_SIGMAS="${BILATERAL_SIGMAS:-0.05,0.10,0.20}"
RISK_SCORES="${RISK_SCORES:-changed geom change_plus_geom}"
POLICY_MODES="${POLICY_MODES:-practical strict}"
PRACTICAL_MAX_CAL_HARM_RATE="${PRACTICAL_MAX_CAL_HARM_RATE:-0.25}"
STRICT_MAX_CAL_HARM_RATE="${STRICT_MAX_CAL_HARM_RATE:-0.0}"

cd "$ROOT"
mkdir -p "/users/scratch1/$USER/logs" results/refiner_zoo results/graphmembrane_policy

. rebuttal/slurm/common_env.sh

ZOO_LABEL="${DATASET}_${RUN_TAG}_zoo"
ZOO_CSV="${ROOT}/results/refiner_zoo/${ZOO_LABEL}.csv"
ZOO_JSON="${ROOT}/results/refiner_zoo/${ZOO_LABEL}.json"

echo "[$(date '+%F %T')] zoo_serial DATASET=${DATASET} RUN_TAG=${RUN_TAG}"
echo "[$(date '+%F %T')] zoo_serial CKPT=${CKPT}"
echo "[$(date '+%F %T')] zoo_serial OUT=${ZOO_CSV}"

"$PYTHON" tools/eval_binary_refiner_zoo.py \
  --npz_all "$NPZ" \
  --idx_eval "$IDX_EVAL" \
  --ckpt "$CKPT" \
  --out_csv "$ZOO_CSV" \
  --out_summary "$ZOO_JSON" \
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
  --data_verbose 1

for mode in $POLICY_MODES; do
  case "$mode" in
    practical) max_harm="$PRACTICAL_MAX_CAL_HARM_RATE" ;;
    strict) max_harm="$STRICT_MAX_CAL_HARM_RATE" ;;
    *) echo "unknown mode: $mode" >&2; exit 2 ;;
  esac
  for risk in $RISK_SCORES; do
    stamp="$(date '+%Y%m%d_%H%M%S')"
    out_csv="${ROOT}/results/graphmembrane_policy/${ZOO_LABEL}_binary_${mode}_${risk}_${SLURM_JOB_ID}_${stamp}.csv"
    out_json="${ROOT}/results/graphmembrane_policy/${ZOO_LABEL}_binary_${mode}_${risk}_${SLURM_JOB_ID}_${stamp}.json"
    echo "[$(date '+%F %T')] zoo_serial policy mode=${mode} risk=${risk}"
    "$PYTHON" tools/eval_safe_action_portfolio.py \
      --inputs "${ZOO_LABEL}=${ZOO_CSV}" \
      --risk_score "$risk" \
      --max_cal_harm_rate "$max_harm" \
      --out_csv "$out_csv" \
      --out_summary "$out_json"
  done
done

echo "[$(date '+%F %T')] zoo_serial done"
