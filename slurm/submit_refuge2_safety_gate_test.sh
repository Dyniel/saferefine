#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
cd "$ROOT"

SEED="${SEED:-1}"
HOST="${HOST:-$ROOT/runs/refuge2_phase0/phase0_segformer_b0_no_graph_s${SEED}_residual_emasave_host/ckpt/best.pt}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-3e-5}"
ALPHA_GRAPH="${ALPHA_GRAPH:-0.25}"
GRAPH_RESIDUAL_CLIP="${GRAPH_RESIDUAL_CLIP:-0.50}"
GRAPH_GATE_POWER="${GRAPH_GATE_POWER:-1.0}"
GRAPH_GATE_FLOOR="${GRAPH_GATE_FLOOR:-0.0}"

if [ ! -f "$HOST" ]; then
  echo "ERROR missing HOST checkpoint: $HOST"
  echo "Set HOST=/path/to/no_graph/best.pt or run slurm/submit_refuge2_ema_control_rerun.sh first."
  exit 20
fi

echo "Submitting REFUGE2 safety-gated GRM test from $ROOT"
echo "HOST=$HOST"
echo "SEED=$SEED EPOCHS=$EPOCHS LR=$LR ALPHA_GRAPH=$ALPHA_GRAPH"
echo "GRAPH_RESIDUAL_CLIP=$GRAPH_RESIDUAL_CLIP GRAPH_GATE_POWER=$GRAPH_GATE_POWER GRAPH_GATE_FLOOR=$GRAPH_GATE_FLOOR"

for gate in entropy margin; do
  tag="safety_${gate}_a${ALPHA_GRAPH}_lr${LR}_clip${GRAPH_RESIDUAL_CLIP}"
  tag="${tag//./p}"
  tag="${tag//- /}"

  train_jid="$(
    RESUME_HOST="$HOST" \
    FREEZE_HOST=1 \
    VARIANT=dynk16 \
    SEED="$SEED" \
    ALPHA_GRAPH="$ALPHA_GRAPH" \
    LR="$LR" \
    EPOCHS="$EPOCHS" \
    GRAPH_SAFETY_GATE="$gate" \
    GRAPH_GATE_FLOOR="$GRAPH_GATE_FLOOR" \
    GRAPH_GATE_POWER="$GRAPH_GATE_POWER" \
    GRAPH_RESIDUAL_CLIP="$GRAPH_RESIDUAL_CLIP" \
    RUN_TAG="$tag" \
    sbatch --parsable slurm/train_refuge2_variant.sbatch
  )"

  ckpt="$ROOT/runs/refuge2_phase0/phase0_segformer_b0_dynk16_s${SEED}_residual_${tag}/ckpt/best.pt"
  echo "$gate train job: $train_jid"
  echo "$gate checkpoint: $ckpt"

  for eval_alpha in 0.00 0.05 0.10 0.15 0.25 0.40; do
    eval_jid="$(
      CKPT="$ckpt" \
      VARIANT=dynk16 \
      ALPHA_GRAPH="$eval_alpha" \
      GRAPH_SAFETY_GATE="$gate" \
      GRAPH_GATE_FLOOR="$GRAPH_GATE_FLOOR" \
      GRAPH_GATE_POWER="$GRAPH_GATE_POWER" \
      GRAPH_RESIDUAL_CLIP="$GRAPH_RESIDUAL_CLIP" \
      sbatch --parsable --dependency=afterok:"$train_jid" slurm/eval_refuge2_ckpt.sbatch
    )"
    echo "$gate eval alpha=$eval_alpha job: $eval_jid"
  done
done

