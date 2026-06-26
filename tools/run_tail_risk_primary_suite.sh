#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
PYTHON="${PYTHON:-/users/scratch1/dancies/conda_envs/py312/bin/python}"
INPUT_DIR="${INPUT_DIR:-${ROOT}/results/refiner_zoo_uncert}"
OUT_DIR="${OUT_DIR:-${ROOT}/results/tail_risk_primary}"
RISK_SCORES="${RISK_SCORES:-changed geom change_plus_geom host_uncertainty}"

MAX_CAL_HARM_RATE="${MAX_CAL_HARM_RATE:-0.25}"
MAX_CAL_DROP05_RATE="${MAX_CAL_DROP05_RATE:-0.10}"
MAX_CAL_DROP20_RATE="${MAX_CAL_DROP20_RATE:-1.0}"
MAX_CAL_MEAN_HARM="${MAX_CAL_MEAN_HARM:-0.02}"
MEAN_HARM_SCALE="${MEAN_HARM_SCALE:-1.0}"
BOUND_MODE="${BOUND_MODE:-hoeffding}"
TAIL_CONSTRAINT_MODE="${TAIL_CONSTRAINT_MODE:-full}"
SELECTION_MODE="${SELECTION_MODE:-joint}"
HARM_EPS="${HARM_EPS:-0.0}"
NESTED_SELECT_FRACTION="${NESTED_SELECT_FRACTION:-0.25}"
NESTED_CAL_FRACTION="${NESTED_CAL_FRACTION:-0.25}"
REQUIRE_POSITIVE_UTILITY="${REQUIRE_POSITIVE_UTILITY:-0}"
MIN_CAL_UTILITY="${MIN_CAL_UTILITY:-0.0}"
COPY_TABLES="${COPY_TABLES:-1}"

cd "$ROOT"
mkdir -p "$OUT_DIR"

run_one() {
  local label="$1"
  local split_group="${2:-image}"
  local csv="${INPUT_DIR}/${label}.csv"
  if [ ! -f "$csv" ]; then
    echo "missing action CSV for ${label}: ${csv}" >&2
    exit 2
  fi
  for risk in $RISK_SCORES; do
    "$PYTHON" tools/eval_safe_action_portfolio.py \
      --inputs "${label}=${csv}" \
      --risk_score "$risk" \
      --split_group "$split_group" \
      --selection_mode "$SELECTION_MODE" \
      --harm_eps "$HARM_EPS" \
      --nested_select_fraction "$NESTED_SELECT_FRACTION" \
      --nested_cal_fraction "$NESTED_CAL_FRACTION" \
      --max_cal_harm_rate "$MAX_CAL_HARM_RATE" \
      --max_cal_drop05_rate "$MAX_CAL_DROP05_RATE" \
      --max_cal_drop20_rate "$MAX_CAL_DROP20_RATE" \
      --max_cal_mean_harm "$MAX_CAL_MEAN_HARM" \
      --mean_harm_scale "$MEAN_HARM_SCALE" \
      --bound_mode "$BOUND_MODE" \
      --tail_constraint_mode "$TAIL_CONSTRAINT_MODE" \
      $(if [ "$REQUIRE_POSITIVE_UTILITY" = "1" ]; then printf '%s' "--require_positive_utility"; fi) \
      --min_cal_utility "$MIN_CAL_UTILITY" \
      --out_csv "${OUT_DIR}/${label}_${risk}_tail_primary.csv" \
      --out_summary "${OUT_DIR}/${label}_${risk}_tail_primary.json"
  done
}

run_one isic2018_task1_mediafinal_unet_e120_zoo image
run_one kvasir_seg_mediafinal_graphseg_e120_zoo image
run_one kvasir_seg_mediafinal_unet_e120_zoo image
run_one ph2_mediafinal_unet_e120_zoo image
run_one polyps_official_mediafinal_graphseg_e120_zoo image
run_one polyps_official_mediafinal_unet_e120_zoo image
run_one msd_heart_mri_mediafinal_graphseg_mri_e120_zoo patient
run_one msd_heart_mri_mediafinal_unet_mri_e120_zoo patient

"$PYTHON" tools/summarize_tail_risk_primary.py \
  --input_dir "$OUT_DIR" \
  --out_prefix "${OUT_DIR}/tail_risk_primary_summary"

if [ "$COPY_TABLES" = "1" ] && [ -d "${ROOT}/docs/submission/full_submission/tables" ]; then
  cp "${OUT_DIR}"/tail_risk_primary_summary_* "${ROOT}/docs/submission/full_submission/tables/"
fi
