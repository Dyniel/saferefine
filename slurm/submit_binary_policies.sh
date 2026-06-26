#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
cd "$ROOT"

DATASET="${DATASET:?Set DATASET=name}"
EVAL_CSV="${EVAL_CSV:-$ROOT/results/binary_refinement/${DATASET}_phase0_eval.csv}"
EVAL_JID="${EVAL_JID:-}"
POLICY_MODES="${POLICY_MODES:-practical strict}"
RISK_SCORES="${RISK_SCORES:-changed geom change_plus_geom}"
PRACTICAL_MAX_CAL_HARM_RATE="${PRACTICAL_MAX_CAL_HARM_RATE:-0.25}"
STRICT_MAX_CAL_HARM_RATE="${STRICT_MAX_CAL_HARM_RATE:-0.0}"
POLICY_RUN_TAG="${POLICY_RUN_TAG:-}"

dep_args=()
if [ -n "$EVAL_JID" ]; then
  dep_args=(--dependency=afterok:"$EVAL_JID")
fi

for mode in $POLICY_MODES; do
  case "$mode" in
    practical)
      max_harm="$PRACTICAL_MAX_CAL_HARM_RATE"
      ;;
    strict)
      max_harm="$STRICT_MAX_CAL_HARM_RATE"
      ;;
    *)
      echo "unknown POLICY_MODES entry: $mode" >&2
      exit 2
      ;;
  esac
  for risk_score in $RISK_SCORES; do
    policy_jid="$(
      INPUTS="${DATASET}=$EVAL_CSV" \
      RISK_SCORE="$risk_score" \
      MAX_CAL_HARM_RATE="$max_harm" \
      RUN_TAG="${POLICY_RUN_TAG:-${DATASET}_binary}_${mode}" \
      sbatch --parsable "${dep_args[@]}" slurm/eval_graphmembrane_action_policy.sbatch
    )"
    echo "binary ${mode} policy risk_score=${risk_score} job: ${policy_jid}"
  done
done
