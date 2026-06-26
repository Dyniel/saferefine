#!/bin/bash
#SBATCH --job-name=grm_policy
#SBATCH --partition=gpu-a100-80gb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/users/scratch1/%u/logs/%x_%j.out

set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
CONDA_ENV="${CONDA_ENV:-grm_media_py312}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-}"
PYTHON="${PYTHON:-}"

NPZ="${NPZ:-/users/scratch1/dancies/jaskra_repo/jaskra/data/REFUGE2/refuge2_all_512_fixed.npz}"
IDX_VAL="${IDX_VAL:-${ROOT}/dataset_splits/refuge2_all512_fixed/val.txt}"
CKPT="${CKPT:?Set CKPT=/path/to/best.pt}"

VARIANT="${VARIANT:-dynk16}"       # grid_only | dynk16
ALPHAS="${ALPHAS:-0,0.05,0.10,0.15,0.25,0.40}"
GRAPH_OUTPUT_MODE="${GRAPH_OUTPUT_MODE:-residual}"
GRAPH_SAFETY_GATE="${GRAPH_SAFETY_GATE:-none}"
GRAPH_GATE_FLOOR="${GRAPH_GATE_FLOOR:-0.0}"
GRAPH_GATE_POWER="${GRAPH_GATE_POWER:-1.0}"
GRAPH_RESIDUAL_CLIP="${GRAPH_RESIDUAL_CLIP:-0.0}"
BATCH="${BATCH:-4}"
WORKERS="${WORKERS:-0}"
CAL_FRACTION="${CAL_FRACTION:-0.5}"
HARM_EPS="${HARM_EPS:-0.0}"
BETA_HARM="${BETA_HARM:-2.0}"
MAX_CAL_HARM_RATE="${MAX_CAL_HARM_RATE:-0.25}"
CRC_CONFIDENCE="${CRC_CONFIDENCE:-0.10}"
RISK_SCORE="${RISK_SCORE:-change}"
RUN_TAG="${RUN_TAG:-policy}"
USE_MMSEG_BACKBONE="${USE_MMSEG_BACKBONE:-0}"
export USE_MMSEG_BACKBONE

cd "$ROOT"
mkdir -p "/users/scratch1/$USER/logs" results/refuge2_policy

. rebuttal/slurm/common_env.sh

if [ "$VARIANT" = "grid_only" ]; then
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
  echo "ERROR unknown VARIANT=${VARIANT}; expected grid_only|dynk16"
  exit 11
fi

STAMP="$(date '+%Y%m%d_%H%M%S')"
SAFE_TAG="${RUN_TAG//\//_}_${SLURM_JOB_ID:-local}_${STAMP}"
OUT_CSV="${OUT_CSV:-${ROOT}/results/refuge2_policy/${SAFE_TAG}.csv}"
OUT_SUMMARY="${OUT_SUMMARY:-${ROOT}/results/refuge2_policy/${SAFE_TAG}.json}"

echo "[$(date '+%F %T')] policy ROOT=${ROOT}"
echo "[$(date '+%F %T')] policy CKPT=${CKPT}"
echo "[$(date '+%F %T')] policy VARIANT=${VARIANT} ALPHAS=${ALPHAS} GRAPH_OUTPUT_MODE=${GRAPH_OUTPUT_MODE}"
echo "[$(date '+%F %T')] policy GRAPH_SAFETY_GATE=${GRAPH_SAFETY_GATE} GRAPH_GATE_FLOOR=${GRAPH_GATE_FLOOR} GRAPH_GATE_POWER=${GRAPH_GATE_POWER} GRAPH_RESIDUAL_CLIP=${GRAPH_RESIDUAL_CLIP}"
echo "[$(date '+%F %T')] policy CAL_FRACTION=${CAL_FRACTION} HARM_EPS=${HARM_EPS} BETA_HARM=${BETA_HARM} MAX_CAL_HARM_RATE=${MAX_CAL_HARM_RATE} CRC_CONFIDENCE=${CRC_CONFIDENCE} RISK_SCORE=${RISK_SCORE}"
echo "[$(date '+%F %T')] policy OUT_CSV=${OUT_CSV}"
echo "[$(date '+%F %T')] policy OUT_SUMMARY=${OUT_SUMMARY}"

"$PYTHON" tools/eval_refinement_policy.py \
  --npz_all "$NPZ" \
  --idx_val "$IDX_VAL" \
  --ckpt "$CKPT" \
  --alphas "$ALPHAS" \
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
  --graph_output_mode "$GRAPH_OUTPUT_MODE" \
  --graph_safety_gate "$GRAPH_SAFETY_GATE" \
  --graph_gate_floor "$GRAPH_GATE_FLOOR" \
  --graph_gate_power "$GRAPH_GATE_POWER" \
  --graph_residual_clip "$GRAPH_RESIDUAL_CLIP" \
  --batch "$BATCH" \
  --workers "$WORKERS" \
  --cal_fraction "$CAL_FRACTION" \
  --harm_eps "$HARM_EPS" \
  --beta_harm "$BETA_HARM" \
  --max_cal_harm_rate "$MAX_CAL_HARM_RATE" \
  --crc_confidence "$CRC_CONFIDENCE" \
  --risk_score "$RISK_SCORE" \
  --out_csv "$OUT_CSV" \
  --out_summary "$OUT_SUMMARY" \
  --data_verbose 1

echo "[$(date '+%F %T')] policy done"
