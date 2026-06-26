#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
cd "$ROOT"

PHASE0_HARD="${PHASE0_HARD:-$ROOT/results/graphmembrane_host/phase0_hard_21430527_20260621_192426.csv}"
PHASE0_SOFT="${PHASE0_SOFT:-$ROOT/results/graphmembrane_host/phase0_soft_21430528_20260621_192427.csv}"
HOSTFT_HARD="${HOSTFT_HARD:-$ROOT/results/graphmembrane_host/hostft_hard_hard_21430544_20260621_195631.csv}"
HOSTFT_SOFT="${HOSTFT_SOFT:-$ROOT/results/graphmembrane_host/hostft_soft_soft_21430546_20260621_195528.csv}"

INPUTS="${INPUTS:-phase0_hard=$PHASE0_HARD,phase0_soft=$PHASE0_SOFT,hostft_hard=$HOSTFT_HARD,hostft_soft=$HOSTFT_SOFT}"

for risk_score in changed force_times_change changed_plus_force; do
  jid="$(
    INPUTS="$INPUTS" \
    RISK_SCORE="$risk_score" \
    RUN_TAG="graphmem_portfolio" \
    sbatch --parsable slurm/eval_graphmembrane_action_policy.sbatch
  )"
  echo "graphmembrane action policy risk_score=$risk_score job: $jid"
done
