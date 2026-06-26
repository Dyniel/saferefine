#!/usr/bin/env bash
ROOT="/home/student2/jaskra"
TRAIN_PY="$ROOT/train_refuge2_npz_unified.py"
NPZ="$ROOT/npz/REFUGE2_512_sanitized.npz"
IDX_VAL="$ROOT/idx_val.txt"
SPLITS_DIR="$ROOT/runs/eccv_small_data_segformer/splits"
OUTBASE="$ROOT/runs/eccv_small_data_segformer/runs"

mkdir -p "$OUTBASE"

BACKBONE="segformer_b0"
IMG=512
EPOCHS="${EPOCHS:-80}"
BS="${BS:-10}"
WORKERS="${WORKERS:-8}"

# graf warianty:
#  - no_graph: alpha_graph=0 (czyli zero miksu)
#  - grid: dyn_on=none, eval_dyn_k=0, alpha_graph=0.55
#  - dyn: dyn_on=feat, eval_dyn_k=16, alpha_graph=0.55
declare -a VARS=("no_graph" "grid" "dyn")

for SEED in 1 2 3; do
  for FR in 25 50 75; do
    IDX_TR="$SPLITS_DIR/idx_train_${FR}pct_seed${SEED}.txt"
    [ -f "$IDX_TR" ] || { echo "[FAIL] missing $IDX_TR"; exit 2; }

    for VAR in "${VARS[@]}"; do
      RUN="eccv_segformer_b0_${VAR}_sd${FR}_s${SEED}"
      OUTDIR="$OUTBASE/$RUN"
      mkdir -p "$OUTDIR"

      # parametry wariantu
      if [ "$VAR" = "no_graph" ]; then
        ALPHA="0.0"; DYN_ON="feat"; DYN_K="16"; EVAL_DYN_K="16"
      elif [ "$VAR" = "grid" ]; then
        ALPHA="0.55"; DYN_ON="none"; DYN_K="0"; EVAL_DYN_K="0"
      else
        ALPHA="0.55"; DYN_ON="feat"; DYN_K="16"; EVAL_DYN_K="16"
      fi

      echo "================================================================================"
      echo "[RUN] $RUN"
      echo "[cfg] idx_train=$IDX_TR"
      echo "[cfg] alpha=$ALPHA dyn_on=$DYN_ON dyn_k=$DYN_K eval_dyn_k=$EVAL_DYN_K"
      echo "================================================================================"

      python "$TRAIN_PY" \
        --npz_all "$NPZ" \
        --idx_train "$IDX_TR" \
        --idx_val "$IDX_VAL" \
        --backbone "$BACKBONE" \
        --img_size "$IMG" \
        --epochs "$EPOCHS" \
        --batch "$BS" \
        --workers "$WORKERS" \
        --seed "$SEED" \
        --alpha_graph "$ALPHA" \
        --dyn_on "$DYN_ON" \
        --dyn_k "$DYN_K" \
        --dyn_on_eval "$DYN_ON" \
        --eval_dyn_k "$EVAL_DYN_K" \
        --out_dir "$OUTBASE" \
        --run_name "$RUN" \
        2>&1 | tee "$OUTBASE/${RUN}.log"
    done
  done
done

echo "[OK] done"
echo "[out] $OUTBASE"
