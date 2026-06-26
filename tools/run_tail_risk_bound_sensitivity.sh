#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
PYTHON="${PYTHON:-/users/scratch1/dancies/conda_envs/py312/bin/python}"
BASE_OUT="${BASE_OUT:-${ROOT}/results/tail_risk_bound_sensitivity}"
BOUND_MODES="${BOUND_MODES:-hoeffding empirical_bernstein clopper_pearson}"

cd "$ROOT"
mkdir -p "$BASE_OUT"

for bound in $BOUND_MODES; do
  OUT_DIR="${BASE_OUT}/${bound}" BOUND_MODE="$bound" \
    COPY_TABLES=0 PYTHON="$PYTHON" bash tools/run_tail_risk_primary_suite.sh > "${BASE_OUT}/${bound}.log" 2>&1
done

"$PYTHON" tools/summarize_tail_risk_bound_sensitivity.py \
  --input_root "$BASE_OUT" \
  --out_prefix "${BASE_OUT}/tail_risk_bound_sensitivity"

if [ -d "${ROOT}/docs/submission/full_submission/tables" ]; then
  cp "${BASE_OUT}"/tail_risk_bound_sensitivity_* "${ROOT}/docs/submission/full_submission/tables/"
fi
