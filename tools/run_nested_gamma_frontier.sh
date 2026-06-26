#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
PYTHON="${PYTHON:-/users/scratch1/dancies/conda_envs/py312/bin/python}"
BASE_OUT_DIR="${BASE_OUT_DIR:-${ROOT}/results/tail_risk_nested_gamma_frontier}"
GAMMAS="${GAMMAS:-0.05 0.10 0.15 0.20 0.25}"
LOG_DIR="${LOG_DIR:-${BASE_OUT_DIR}/logs}"
TAIL_CONSTRAINT_MODE="${TAIL_CONSTRAINT_MODE:-bernoulli}"
MAX_CAL_DROP20_RATE="${MAX_CAL_DROP20_RATE:-1.0}"

mkdir -p "$BASE_OUT_DIR" "$LOG_DIR"

for gamma in $GAMMAS; do
  tag="${gamma/./p}"
  out_dir="${BASE_OUT_DIR}/gamma_${tag}"
  echo "nested gamma=${gamma} -> ${out_dir}"
  PYTHON="$PYTHON" \
  OUT_DIR="$out_dir" \
  TAIL_CONSTRAINT_MODE="$TAIL_CONSTRAINT_MODE" \
  MAX_CAL_DROP05_RATE="$gamma" \
  MAX_CAL_DROP20_RATE="$MAX_CAL_DROP20_RATE" \
  COPY_TABLES=0 \
  bash "${ROOT}/tools/run_nested_tail_risk_suite.sh" \
    > "${LOG_DIR}/gamma_${tag}.log" 2>&1
done

"$PYTHON" "${ROOT}/tools/summarize_certification_frontier.py" \
  --inputs "$BASE_OUT_DIR" \
  --out_prefix "${BASE_OUT_DIR}/nested_gamma_frontier"
