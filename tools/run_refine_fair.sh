#!/usr/bin/env bash

ROOT="/home/student2/jaskra"
export PYTHONPATH="$ROOT:${PYTHONPATH}"

RUN_DIR="$1"
if [ -z "$RUN_DIR" ]; then
  echo "usage: bash tools/run_refine_fair.sh /path/to/run_dir"
  exit 1
fi

echo "[A] TUNE (VAL-only) for MORPH + CRF (narrow grid)"
python "$ROOT/tools/refine_fair_eval.py" --run_dir "$RUN_DIR" --device cuda \
  --tune --crf_grid small --morph_ks 3,5,7,9 --learned off

echo
echo "[B] TRAIN learned refiner (TRAIN->VAL selection) + FINAL VAL report"
python "$ROOT/tools/refine_fair_eval.py" --run_dir "$RUN_DIR" --device cuda \
  --crf_grid small --morph_ks 3,5,7,9 --learned on --learned_epochs 20 --learned_lr 3e-4
