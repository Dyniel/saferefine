#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-/users/scratch1/dancies/conda_envs/py312/bin/python}"
N_BOOT="${N_BOOT:-2000}"
OUT_DIR="${OUT_DIR:-results/bootstrap_ci_mediafinal}"
CAL_FRACTION="${CAL_FRACTION:-0.5}"
CRC_CONFIDENCE="${CRC_CONFIDENCE:-0.10}"
SEED="${SEED:-1}"

mkdir -p "$OUT_DIR"

labels=(
  "isic2018_task1_mediafinal_unet_e120"
  "kvasir_seg_mediafinal_graphseg_e120"
  "kvasir_seg_mediafinal_unet_e120"
  "ph2_mediafinal_unet_e120"
  "polyps_official_mediafinal_graphseg_e120"
  "polyps_official_mediafinal_unet_e120"
)

run_one() {
  local label="$1"
  local csv="$2"
  local mode="$3"
  local risk="$4"
  local budget="0.25"
  if [[ "$mode" == "strict" ]]; then
    budget="0.0"
  fi
  "$PY" tools/bootstrap_action_policy_ci.py \
    --inputs "${label}=${csv}" \
    --risk_score "$risk" \
    --cal_fraction "$CAL_FRACTION" \
    --max_cal_harm_rate "$budget" \
    --crc_confidence "$CRC_CONFIDENCE" \
    --n_boot "$N_BOOT" \
    --seed "$SEED" \
    --out_csv "${OUT_DIR}/${label}_${mode}_${risk}.csv"
}

for label in "${labels[@]}"; do
  base_csv="results/binary_refinement/${label}_eval.csv"
  zoo_label="${label}_zoo"
  zoo_csv="results/refiner_zoo/${zoo_label}.csv"
  for source_label in "$label" "$zoo_label"; do
    if [[ "$source_label" == "$label" ]]; then
      csv="$base_csv"
    else
      csv="$zoo_csv"
    fi
    if [[ ! -s "$csv" ]]; then
      echo "missing csv: $csv" >&2
      exit 1
    fi
    for mode in practical strict; do
      for risk in changed geom change_plus_geom; do
        echo "bootstrap ${source_label} ${mode} ${risk}"
        run_one "$source_label" "$csv" "$mode" "$risk"
      done
    done
  done
done

"$PY" tools/summarize_bootstrap_ci.py \
  --in_dir "$OUT_DIR" \
  --out_all "${OUT_DIR}/mediafinal_bootstrap_all_policies.csv" \
  --out_crc "${OUT_DIR}/mediafinal_bootstrap_crc_ablation.csv"
