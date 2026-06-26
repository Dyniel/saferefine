#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
PYTHON="${PYTHON:-/users/scratch1/dancies/conda_envs/py312/bin/python}"
STAGING="${STAGING:-/users/scratch1/dancies/datasets_staging/grm_media}"
IMG_SIZE="${IMG_SIZE:-352}"
CHUNK_SIZE="${CHUNK_SIZE:-300}"
COMPRESS="${COMPRESS:-0}"

cd "$ROOT"

chunk_dir="$STAGING/chunks/isic2018_task1_${IMG_SIZE}"
out_npz="data_npz/isic2018_task1_${IMG_SIZE}.npz"
split_dir="dataset_splits/isic2018_task1_${IMG_SIZE}"

common=(
  --train_image_dir "$STAGING/isic/ISIC2018_Task1-2_Training_Input"
  --train_mask_dir "$STAGING/isic/ISIC2018_Task1_Training_GroundTruth"
  --val_image_dir "$STAGING/isic/ISIC2018_Task1-2_Validation_Input"
  --val_mask_dir "$STAGING/isic/ISIC2018_Task1_Validation_GroundTruth"
  --test_image_dir "$STAGING/isic/ISIC2018_Task1-2_Test_Input"
  --test_mask_dir "$STAGING/isic/ISIC2018_Task1_Test_GroundTruth"
  --out_npz "$out_npz"
  --split_dir "$split_dir"
  --chunk_dir "$chunk_dir"
  --name "isic2018_task1_${IMG_SIZE}"
  --img_size "$IMG_SIZE"
  --compress "$COMPRESS"
)

"$PYTHON" tools/build_binary_npz_chunks.py --mode plan --chunk_size "$CHUNK_SIZE" "${common[@]}"
"$PYTHON" - "$chunk_dir/plan.json" <<'PY' > "$chunk_dir/chunks.tsv"
import json
import sys

plan = json.load(open(sys.argv[1]))
for chunk in plan["chunks"]:
    print(chunk["start"], chunk["end"])
PY

while read -r start end; do
  chunk_file="$chunk_dir/chunk_$(printf '%06d' "$start")_$(printf '%06d' "$end").npz"
  if [ -f "$chunk_file" ]; then
    echo "skip existing chunk $start $end"
  else
    echo "build chunk $start $end"
    "$PYTHON" tools/build_binary_npz_chunks.py --mode chunk --start "$start" --end "$end" "${common[@]}"
  fi
done < "$chunk_dir/chunks.tsv"

"$PYTHON" tools/build_binary_npz_chunks.py --mode merge "${common[@]}"
