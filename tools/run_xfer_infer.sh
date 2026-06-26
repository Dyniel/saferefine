#!/usr/bin/env bash
ROOT="${ROOT:-/home/student2/jaskra}"

TRAIN_PY="${TRAIN_PY:-$ROOT/train_refuge2_npz_unified.py}"

# datasety (pełne, nie cropped/square z kaggle)
ORIGA_IMG="${ORIGA_IMG:-$ROOT/data/glaucoma-datasets/ORIGA/Images}"
ORIGA_MSK="${ORIGA_MSK:-$ROOT/data/glaucoma-datasets/ORIGA/Masks}"

G1020_IMG="${G1020_IMG:-$ROOT/data/glaucoma-datasets/G1020/Images}"
G1020_MSK="${G1020_MSK:-$ROOT/data/glaucoma-datasets/G1020/Masks}"

OUTROOT="${OUTROOT:-$ROOT/runs/_xfer_infer_$(date '+%Y%m%d_%H%M%S')}"

RUNS_FILE="${RUNS_FILE:-$ROOT/_refine_run_dirs.txt}"  # albo podaj swój plik z run_dirami
IMG_SIZE="${IMG_SIZE:-512}"
DEVICE="${DEVICE:-cuda}"

# preprocess toggles
AUTOCROP="${AUTOCROP:-1}"   # 1/0
SQUARE="${SQUARE:-1}"       # 1/0
CLAHE_ORIGA="${CLAHE_ORIGA:-0}"
CLAHE_G1020="${CLAHE_G1020:-1}"

# evaluate only if masks exist
DO_EVAL="${DO_EVAL:-1}"     # 1/0

mkdir -p "$OUTROOT" || exit 2

echo "[cfg] OUTROOT=$OUTROOT"
echo "[cfg] RUNS_FILE=$RUNS_FILE"
echo "[cfg] IMG_SIZE=$IMG_SIZE DEVICE=$DEVICE"
echo "[cfg] ORIGA_IMG=$ORIGA_IMG"
echo "[cfg] G1020_IMG=$G1020_IMG"
echo

if [ ! -f "$RUNS_FILE" ]; then
  echo "[FAIL] RUNS_FILE not found: $RUNS_FILE"
  exit 2
fi

# helper: infer one dataset
run_one () {
  local ds="$1"
  local img="$2"
  local msk="$3"
  local clahe="$4"
  local run="$5"
  local ckpt="$6"

  local out="$OUTROOT/$ds/$run"
  mkdir -p "$out" || return 2

  echo "================================================================================"
  echo "[RUN] ds=$ds run=$run"
  echo "[RUN] ckpt=$ckpt"
  echo "[RUN] out =$out"
  echo "================================================================================"

  if [ "$DO_EVAL" = "1" ] && [ -d "$msk" ]; then
    python "$ROOT/tools/infer_folder.py" \
      --dataset "$ds" \
      --images "$img" \
      --masks "$msk" \
      --model-py "$TRAIN_PY" \
      --ckpt "$ckpt" \
      --outdir "$out" \
      --img-size "$IMG_SIZE" \
      --device "$DEVICE" \
      $( [ "$AUTOCROP" = "1" ] && echo --autocrop || echo --no-autocrop ) \
      $( [ "$SQUARE"   = "1" ] && echo --square   || echo --no-square ) \
      $( [ "$clahe"    = "1" ] && echo --clahe    || echo --no-clahe ) \
      --save-pred-masks \
      --save-overlays \
      --heartbeat 50
  else
    python "$ROOT/tools/infer_folder.py" \
      --dataset "$ds" \
      --images "$img" \
      --model-py "$TRAIN_PY" \
      --ckpt "$ckpt" \
      --outdir "$out" \
      --img-size "$IMG_SIZE" \
      --device "$DEVICE" \
      $( [ "$AUTOCROP" = "1" ] && echo --autocrop || echo --no-autocrop ) \
      $( [ "$SQUARE"   = "1" ] && echo --square   || echo --no-square ) \
      $( [ "$clahe"    = "1" ] && echo --clahe    || echo --no-clahe ) \
      --save-pred-masks \
      --save-overlays \
      --heartbeat 50
  fi
}

# loop runs
while IFS= read -r run_dir; do
  [ -z "$run_dir" ] && continue
  [ ! -d "$run_dir" ] && continue
  run="$(basename "$run_dir")"
  ckpt="$run_dir/ckpt/best.pt"
  [ ! -f "$ckpt" ] && continue

  run_one "origa" "$ORIGA_IMG" "$ORIGA_MSK" "$CLAHE_ORIGA" "$run" "$ckpt" || true
  run_one "g1020" "$G1020_IMG" "$G1020_MSK" "$CLAHE_G1020" "$run" "$ckpt" || true
done < "$RUNS_FILE"

echo
echo "[OK] done"
echo "[out] $OUTROOT"
echo
echo "Quick summaries:"
echo "  find \"$OUTROOT\" -name '*_infer.csv' | wc -l"
echo "  find \"$OUTROOT\" -name '*_infer.csv' -maxdepth 6 -print | head"
