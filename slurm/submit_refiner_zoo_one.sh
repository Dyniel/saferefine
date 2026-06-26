#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
cd "$ROOT"

DATASET="${DATASET:?Set DATASET=name}"
RUN_TAG="${RUN_TAG:?Set RUN_TAG=host_run_tag}"
NPZ="${NPZ:-$ROOT/data_npz/${DATASET}_352.npz}"
SPLIT_DIR="${SPLIT_DIR:-$ROOT/dataset_splits/${DATASET}_352}"
IDX_EVAL="${IDX_EVAL:-$SPLIT_DIR/test.txt}"
CKPT="${CKPT:?Set CKPT=/path/to/best.pt}"
ARCH="${ARCH:-graphseg}"
BACKBONE="${BACKBONE:-segformer_b0}"
UNET_BASE="${UNET_BASE:-32}"
IMG_SIZE="${IMG_SIZE:-352}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-2}"
POLICY_MODES="${POLICY_MODES:-practical strict}"
RISK_SCORES="${RISK_SCORES:-changed geom change_plus_geom}"

out_csv="$ROOT/results/refiner_zoo/${DATASET}_${RUN_TAG}_zoo.csv"
out_json="$ROOT/results/refiner_zoo/${DATASET}_${RUN_TAG}_zoo.json"

zoo_jid="$(
  DATASET="$DATASET" \
  RUN_TAG="$RUN_TAG" \
  NPZ="$NPZ" \
  IDX_EVAL="$IDX_EVAL" \
  CKPT="$CKPT" \
  ARCH="$ARCH" \
  BACKBONE="$BACKBONE" \
  UNET_BASE="$UNET_BASE" \
  IMG_SIZE="$IMG_SIZE" \
  BATCH="$BATCH" \
  WORKERS="$WORKERS" \
  OUT_CSV="$out_csv" \
  OUT_SUMMARY="$out_json" \
  sbatch --parsable slurm/eval_binary_refiner_zoo.sbatch
)"
echo "refiner zoo eval job: $zoo_jid"

DATASET="${DATASET}_${RUN_TAG}_zoo" \
EVAL_CSV="$out_csv" \
EVAL_JID="$zoo_jid" \
POLICY_MODES="$POLICY_MODES" \
RISK_SCORES="$RISK_SCORES" \
POLICY_RUN_TAG="${DATASET}_${RUN_TAG}_zoo_binary" \
bash slurm/submit_binary_policies.sh
