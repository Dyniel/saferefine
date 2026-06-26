#!/bin/bash
#SBATCH --job-name=grm_eval
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/users/scratch1/%u/logs/%x_%j.out

set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
CONDA_ENV="${CONDA_ENV:-grm_media_py312}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-}"
PYTHON="${PYTHON:-}"

NPZ="${NPZ:-/users/scratch1/dancies/jaskra_repo/jaskra/data/REFUGE2/refuge2_all_512_fixed.npz}"
IDX_VAL="${IDX_VAL:-${ROOT}/dataset_splits/refuge2_all512_fixed/val.txt}"
CKPT="${CKPT:?Set CKPT=/path/to/best.pt}"

VARIANT="${VARIANT:-dynk16}"       # no_graph | grid_only | dynk16
ALPHA_GRAPH="${ALPHA_GRAPH:-0.55}"
GRAPH_OUTPUT_MODE="${GRAPH_OUTPUT_MODE:-residual}"
GRAPH_SAFETY_GATE="${GRAPH_SAFETY_GATE:-none}"
GRAPH_GATE_FLOOR="${GRAPH_GATE_FLOOR:-0.0}"
GRAPH_GATE_POWER="${GRAPH_GATE_POWER:-1.0}"
GRAPH_RESIDUAL_CLIP="${GRAPH_RESIDUAL_CLIP:-0.0}"
BATCH="${BATCH:-4}"
WORKERS="${WORKERS:-0}"
USE_MMSEG_BACKBONE="${USE_MMSEG_BACKBONE:-0}"
export USE_MMSEG_BACKBONE

cd "$ROOT"
mkdir -p "/users/scratch1/$USER/logs"

. rebuttal/slurm/common_env.sh

if [ "$VARIANT" = "no_graph" ]; then
  DYN_ON="feat"
  DYN_K="16"
  DYN_ON_EVAL="feat"
  EVAL_DYN_K="16"
elif [ "$VARIANT" = "grid_only" ]; then
  DYN_ON="none"
  DYN_K="0"
  DYN_ON_EVAL="none"
  EVAL_DYN_K="0"
elif [ "$VARIANT" = "dynk16" ]; then
  DYN_ON="feat"
  DYN_K="16"
  DYN_ON_EVAL="feat"
  EVAL_DYN_K="16"
else
  echo "ERROR unknown VARIANT=${VARIANT}; expected no_graph|grid_only|dynk16"
  exit 11
fi

echo "[$(date '+%F %T')] eval ROOT=${ROOT}"
echo "[$(date '+%F %T')] eval CKPT=${CKPT}"
echo "[$(date '+%F %T')] eval VARIANT=${VARIANT} ALPHA_GRAPH=${ALPHA_GRAPH} GRAPH_OUTPUT_MODE=${GRAPH_OUTPUT_MODE}"
echo "[$(date '+%F %T')] eval GRAPH_SAFETY_GATE=${GRAPH_SAFETY_GATE} GRAPH_GATE_FLOOR=${GRAPH_GATE_FLOOR} GRAPH_GATE_POWER=${GRAPH_GATE_POWER} GRAPH_RESIDUAL_CLIP=${GRAPH_RESIDUAL_CLIP}"
echo "[$(date '+%F %T')] eval NPZ=${NPZ}"
echo "[$(date '+%F %T')] eval IDX_VAL=${IDX_VAL}"

"$PYTHON" tools/unified_graphseg.py \
  --mode eval \
  --npz_all "$NPZ" \
  --idx_val "$IDX_VAL" \
  --resume "$CKPT" \
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
  --amp 0 \
  --run_name "eval_${VARIANT}_a${ALPHA_GRAPH}" \
  --data_verbose 1

echo "[$(date '+%F %T')] eval done"
