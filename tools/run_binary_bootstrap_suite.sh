#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
PYTHON="${PYTHON:-/users/scratch1/dancies/conda_envs/py312/bin/python}"
DATASETS="${DATASETS:-kvasir_seg kvasir_sessile ph2 polyps_official isic2018_task1}"
RISK_SCORES="${RISK_SCORES:-changed geom change_plus_geom}"
N_BOOT="${N_BOOT:-2000}"

cd "$ROOT"
mkdir -p results/bootstrap_ci

for dataset in $DATASETS; do
  input="results/binary_refinement/${dataset}_phase0_eval.csv"
  if [ ! -f "$input" ]; then
    echo "skip ${dataset}: missing ${input}"
    continue
  fi
  for mode in practical strict; do
    if [ "$mode" = "practical" ]; then
      max_harm="${PRACTICAL_MAX_CAL_HARM_RATE:-0.25}"
    else
      max_harm="${STRICT_MAX_CAL_HARM_RATE:-0.0}"
    fi
    for risk in $RISK_SCORES; do
      out="results/bootstrap_ci/${dataset}_${mode}_${risk}.csv"
      "$PYTHON" tools/bootstrap_action_policy_ci.py \
        --inputs "${dataset}=${input}" \
        --risk_score "$risk" \
        --max_cal_harm_rate "$max_harm" \
        --n_boot "$N_BOOT" \
        --out_csv "$out"
    done
  done
done
