#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
PYTHON="${PYTHON:-/users/scratch1/dancies/conda_envs/py312/bin/python}"
INPUT_DIR="${INPUT_DIR:-${ROOT}/results/refiner_zoo_uncert}"
FALLBACK_INPUT_DIR="${FALLBACK_INPUT_DIR:-${ROOT}/results/refiner_zoo}"
OUT_DIR="${OUT_DIR:-${ROOT}/results/decision_baselines}"

cd "$ROOT"
mkdir -p "$OUT_DIR"

run_one() {
  local label="$1"
  local split_group="${2:-image}"
  local csv="${INPUT_DIR}/${label}.csv"
  if [ ! -f "$csv" ]; then
    csv="${FALLBACK_INPUT_DIR}/${label}.csv"
  fi
  if [ ! -f "$csv" ]; then
    echo "missing action CSV for ${label}" >&2
    exit 2
  fi
  "$PYTHON" tools/eval_decision_baselines.py \
    --input_csv "$csv" \
    --label "$label" \
    --split_group "$split_group" \
    --out_prefix "${OUT_DIR}/${label}"
}

run_one isic2018_task1_mediafinal_unet_e120_zoo image
run_one kvasir_seg_mediafinal_graphseg_e120_zoo image
run_one kvasir_seg_mediafinal_unet_e120_zoo image
run_one ph2_mediafinal_unet_e120_zoo image
run_one polyps_official_mediafinal_graphseg_e120_zoo image
run_one polyps_official_mediafinal_unet_e120_zoo image
run_one msd_heart_mri_mediafinal_graphseg_mri_e120_zoo patient
run_one msd_heart_mri_mediafinal_unet_mri_e120_zoo patient

"$PYTHON" tools/summarize_decision_baselines.py \
  --input_dir "$OUT_DIR" \
  --out_prefix "${OUT_DIR}/decision_baselines_summary"

if [ -d "${ROOT}/docs/submission/full_submission/tables" ]; then
  cp "${OUT_DIR}"/decision_baselines_summary_* "${ROOT}/docs/submission/full_submission/tables/"
fi
