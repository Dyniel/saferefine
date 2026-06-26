#!/bin/bash
#SBATCH --job-name=bin_final
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=08:00:00
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
IDX_TRAIN="${IDX_TRAIN:-${SPLIT_DIR}/train.txt}"
IDX_VAL="${IDX_VAL:-${SPLIT_DIR}/val.txt}"
IDX_EVAL="${IDX_EVAL:-${SPLIT_DIR}/test.txt}"

ARCH="${ARCH:-graphseg}"
BACKBONE="${BACKBONE:-segformer_b0}"
UNET_BASE="${UNET_BASE:-32}"
IMG_SIZE="${IMG_SIZE:-352}"
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-120}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-2}"
LR="${LR:-2.5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-3e-5}"
BOUNDARY_W="${BOUNDARY_W:-0.0}"
AMP="${AMP:-1}"
MORPH_KS="${MORPH_KS:-3,5,7,9,11}"
RISK_SCORES="${RISK_SCORES:-changed geom change_plus_geom}"
POLICY_MODES="${POLICY_MODES:-practical strict}"
PRACTICAL_MAX_CAL_HARM_RATE="${PRACTICAL_MAX_CAL_HARM_RATE:-0.25}"
STRICT_MAX_CAL_HARM_RATE="${STRICT_MAX_CAL_HARM_RATE:-0.0}"

cd "$ROOT"
mkdir -p "/users/scratch1/$USER/logs" runs/binary_host results/binary_refinement results/refiner_zoo results/graphmembrane_policy

. rebuttal/slurm/common_env.sh

ARCH_TAG="${ARCH}_${BACKBONE}"
if [ "$ARCH" = "unet_small" ] || [ "$ARCH" = "unet" ] || [ "$ARCH" = "simple_unet" ]; then
  ARCH_TAG="unet_b${UNET_BASE}"
fi
RUN_NAME="${DATASET}_${ARCH_TAG}_s${SEED}_${RUN_TAG}"
CKPT="${ROOT}/runs/binary_host/${RUN_NAME}/ckpt/best.pt"
EVAL_LABEL="${DATASET}_${RUN_TAG}"
EVAL_CSV="${ROOT}/results/binary_refinement/${EVAL_LABEL}_eval.csv"
EVAL_JSON="${ROOT}/results/binary_refinement/${EVAL_LABEL}_eval.json"
ZOO_LABEL="${DATASET}_${RUN_TAG}_zoo"
ZOO_CSV="${ROOT}/results/refiner_zoo/${ZOO_LABEL}.csv"
ZOO_JSON="${ROOT}/results/refiner_zoo/${ZOO_LABEL}.json"

echo "[$(date '+%F %T')] bin_final DATASET=${DATASET} RUN_NAME=${RUN_NAME}"
echo "[$(date '+%F %T')] bin_final ARCH=${ARCH} BACKBONE=${BACKBONE} UNET_BASE=${UNET_BASE}"

"$PYTHON" tools/train_binary_host.py \
  --npz_all "$NPZ" \
  --idx_train "$IDX_TRAIN" \
  --idx_val "$IDX_VAL" \
  --out_dir "${ROOT}/runs/binary_host" \
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
  --data_verbose 1

echo "[$(date '+%F %T')] bin_final eval baseline CKPT=${CKPT}"
"$PYTHON" tools/eval_binary_safe_refinement.py \
  --npz_all "$NPZ" \
  --idx_eval "$IDX_EVAL" \
  --ckpt "$CKPT" \
  --out_csv "$EVAL_CSV" \
  --out_summary "$EVAL_JSON" \
  --arch "$ARCH" \
  --backbone "$BACKBONE" \
  --unet_base "$UNET_BASE" \
  --img_size "$IMG_SIZE" \
  --batch "$BATCH" \
  --workers "$WORKERS" \
  --morph_ks "$MORPH_KS" \
  --data_verbose 1

for mode in $POLICY_MODES; do
  case "$mode" in
    practical) max_harm="$PRACTICAL_MAX_CAL_HARM_RATE" ;;
    strict) max_harm="$STRICT_MAX_CAL_HARM_RATE" ;;
    *) echo "unknown mode: $mode" >&2; exit 2 ;;
  esac
  for risk in $RISK_SCORES; do
    stamp="$(date '+%Y%m%d_%H%M%S')"
    "$PYTHON" tools/eval_safe_action_portfolio.py \
      --inputs "${EVAL_LABEL}=${EVAL_CSV}" \
      --risk_score "$risk" \
      --max_cal_harm_rate "$max_harm" \
      --out_csv "${ROOT}/results/graphmembrane_policy/${EVAL_LABEL}_binary_${mode}_${risk}_${SLURM_JOB_ID}_${stamp}.csv" \
      --out_summary "${ROOT}/results/graphmembrane_policy/${EVAL_LABEL}_binary_${mode}_${risk}_${SLURM_JOB_ID}_${stamp}.json"
  done
done

echo "[$(date '+%F %T')] bin_final refiner zoo"
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
  --data_verbose 1

for mode in $POLICY_MODES; do
  case "$mode" in
    practical) max_harm="$PRACTICAL_MAX_CAL_HARM_RATE" ;;
    strict) max_harm="$STRICT_MAX_CAL_HARM_RATE" ;;
  esac
  for risk in $RISK_SCORES; do
    stamp="$(date '+%Y%m%d_%H%M%S')"
    "$PYTHON" tools/eval_safe_action_portfolio.py \
      --inputs "${ZOO_LABEL}=${ZOO_CSV}" \
      --risk_score "$risk" \
      --max_cal_harm_rate "$max_harm" \
      --out_csv "${ROOT}/results/graphmembrane_policy/${ZOO_LABEL}_binary_${mode}_${risk}_${SLURM_JOB_ID}_${stamp}.csv" \
      --out_summary "${ROOT}/results/graphmembrane_policy/${ZOO_LABEL}_binary_${mode}_${risk}_${SLURM_JOB_ID}_${stamp}.json"
  done
done

echo "[$(date '+%F %T')] bin_final done"
