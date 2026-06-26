#!/bin/bash
#SBATCH --job-name=grm_refuge2
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/users/scratch1/%u/logs/%x_%j.out

set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
CONDA_ENV="${CONDA_ENV:-grm_media_py312}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-}"
PYTHON="${PYTHON:-}"

NPZ="${NPZ:-/users/scratch1/dancies/jaskra_repo/jaskra/data/REFUGE2/refuge2_all_512_fixed.npz}"
IDX_TRAIN="${IDX_TRAIN:-${ROOT}/dataset_splits/refuge2_all512_fixed/train.txt}"
IDX_VAL="${IDX_VAL:-${ROOT}/dataset_splits/refuge2_all512_fixed/val.txt}"
OUT_DIR="${OUT_DIR:-${ROOT}/runs/refuge2_phase0}"

VARIANT="${VARIANT:-dynk16}"       # no_graph | grid_only | dynk16
SEED="${SEED:-1}"
EPOCHS="${EPOCHS:-80}"
BATCH="${BATCH:-2}"
WORKERS="${WORKERS:-0}"
AMP="${AMP:-0}"
LR="${LR:-2.5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-3e-5}"
GRAPH_OUTPUT_MODE="${GRAPH_OUTPUT_MODE:-residual}"
GRAPH_SAFETY_GATE="${GRAPH_SAFETY_GATE:-none}"
GRAPH_GATE_FLOOR="${GRAPH_GATE_FLOOR:-0.0}"
GRAPH_GATE_POWER="${GRAPH_GATE_POWER:-1.0}"
GRAPH_RESIDUAL_CLIP="${GRAPH_RESIDUAL_CLIP:-0.0}"
USE_MMSEG_BACKBONE="${USE_MMSEG_BACKBONE:-0}"
export USE_MMSEG_BACKBONE
RUN_TAG="${RUN_TAG:-}"
RESUME_HOST="${RESUME_HOST:-}"
FREEZE_HOST="${FREEZE_HOST:-0}"
ALPHA_GRAPH_OVERRIDE="${ALPHA_GRAPH:-}"

cd "$ROOT"
mkdir -p rebuttal/logs "$OUT_DIR" "/users/scratch1/$USER/logs"

. rebuttal/slurm/common_env.sh

if [ "$VARIANT" = "no_graph" ]; then
  ALPHA_GRAPH="0.0"
  DYN_ON="feat"
  DYN_K="16"
  DYN_ON_EVAL="feat"
  EVAL_DYN_K="16"
elif [ "$VARIANT" = "grid_only" ]; then
  ALPHA_GRAPH="0.55"
  DYN_ON="none"
  DYN_K="0"
  DYN_ON_EVAL="none"
  EVAL_DYN_K="0"
elif [ "$VARIANT" = "dynk16" ]; then
  ALPHA_GRAPH="0.55"
  DYN_ON="feat"
  DYN_K="16"
  DYN_ON_EVAL="feat"
  EVAL_DYN_K="16"
else
  echo "ERROR unknown VARIANT=${VARIANT}; expected no_graph|grid_only|dynk16"
  exit 11
fi

if [ -n "$ALPHA_GRAPH_OVERRIDE" ]; then
  ALPHA_GRAPH="$ALPHA_GRAPH_OVERRIDE"
fi

if [ ! -f "$NPZ" ]; then
  echo "ERROR missing NPZ: $NPZ"
  echo "Set NPZ=/path/to/REFUGE2_512_sanitized.npz when submitting."
  exit 20
fi

"$PYTHON" rebuttal/check_rebuttal_env.py

RUN_NAME="phase0_segformer_b0_${VARIANT}_s${SEED}_${GRAPH_OUTPUT_MODE}"
if [ -n "$RUN_TAG" ]; then
  RUN_NAME="${RUN_NAME}_${RUN_TAG}"
fi

RESUME_HOST_ARGS=()
if [ -n "$RESUME_HOST" ]; then
  RESUME_HOST_ARGS=(--resume_host "$RESUME_HOST")
fi

echo "[$(date '+%F %T')] train ROOT=${ROOT}"
echo "[$(date '+%F %T')] train VARIANT=${VARIANT} SEED=${SEED} GRAPH_OUTPUT_MODE=${GRAPH_OUTPUT_MODE}"
echo "[$(date '+%F %T')] train GRAPH_SAFETY_GATE=${GRAPH_SAFETY_GATE} GRAPH_GATE_FLOOR=${GRAPH_GATE_FLOOR} GRAPH_GATE_POWER=${GRAPH_GATE_POWER} GRAPH_RESIDUAL_CLIP=${GRAPH_RESIDUAL_CLIP}"
echo "[$(date '+%F %T')] train ALPHA_GRAPH=${ALPHA_GRAPH} FREEZE_HOST=${FREEZE_HOST}"
echo "[$(date '+%F %T')] train RESUME_HOST=${RESUME_HOST:-<none>}"
echo "[$(date '+%F %T')] train NPZ=${NPZ}"
echo "[$(date '+%F %T')] train IDX_TRAIN=${IDX_TRAIN}"
echo "[$(date '+%F %T')] train IDX_VAL=${IDX_VAL}"
echo "[$(date '+%F %T')] train OUT_DIR=${OUT_DIR}/${RUN_NAME}"

"$PYTHON" tools/unified_graphseg.py \
  --mode train \
  --npz_all "$NPZ" \
  --idx_train "$IDX_TRAIN" \
  --idx_val "$IDX_VAL" \
  --backbone segformer_b0 \
  --img_size 512 \
  --feat_dim 256 \
  --hidden 512 \
  --depth 3 \
  --graph_down 2 \
  --grid4 1 \
  --dyn_on "$DYN_ON" \
  --dyn_k "$DYN_K" \
  --dyn_window 2 \
  --dyn_on_eval "$DYN_ON_EVAL" \
  --eval_dyn_k "$EVAL_DYN_K" \
  --alpha_graph "$ALPHA_GRAPH" \
  --graph_output_mode "$GRAPH_OUTPUT_MODE" \
  --graph_safety_gate "$GRAPH_SAFETY_GATE" \
  --graph_gate_floor "$GRAPH_GATE_FLOOR" \
  --graph_gate_power "$GRAPH_GATE_POWER" \
  --graph_residual_clip "$GRAPH_RESIDUAL_CLIP" \
  --batch "$BATCH" \
  --workers "$WORKERS" \
  --amp "$AMP" \
  --lr "$LR" \
  --weight_decay "$WEIGHT_DECAY" \
  --epochs "$EPOCHS" \
  --out_dir "$OUT_DIR" \
  --run_name "$RUN_NAME" \
  --seed "$SEED" \
  "${RESUME_HOST_ARGS[@]}" \
  --freeze_host "$FREEZE_HOST" \
  --data_verbose 1 \
  --wandb 0

echo "[$(date '+%F %T')] train done checkpoint=${OUT_DIR}/${RUN_NAME}/ckpt/best.pt"
