#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/users/project1/pt01315/emnlp/grm_media}"
cd "$ROOT"

for mode in hard soft; do
  jid="$(
    INPUT_MODE="$mode" \
    RUN_TAG=phase0 \
    sbatch --parsable slurm/eval_graphmembrane_host.sbatch
  )"
  echo "graphmembrane host eval mode=$mode job: $jid"
done

