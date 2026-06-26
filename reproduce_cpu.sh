#!/usr/bin/env bash
# =============================================================================
# SafeRefine Tier-1 reproduction (exact, CPU-only, no dataset download).
#
# Reproduces the full SafeRefine certification analysis from the committed
# per-image action CSVs in results/refiner_zoo_uncert/. Requires only numpy and
# pandas (see requirements.txt).
#
# Usage:
#   ./reproduce_cpu.sh                 # uses `python3` on PATH
#   PYTHON=/path/to/python ./reproduce_cpu.sh
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
export ROOT="$REPO"
export PYTHON
export COPY_TABLES=0

echo "==> repo:   $REPO"
echo "==> python: $($PYTHON -c 'import sys;print(sys.executable, sys.version.split()[0])')"
$PYTHON -c "import numpy, pandas" || { echo "ERROR: need numpy+pandas (pip install -r requirements.txt)"; exit 1; }

step () { echo; echo "=============================================================="; echo "==> $1"; echo "=============================================================="; }

step "1/8  Primary tail-risk contract (full: harmed-rate + large-drop + mean-harm)"
TAIL_CONSTRAINT_MODE=full OUT_DIR="$REPO/results/tail_risk_primary" \
  bash "$REPO/tools/run_tail_risk_primary_suite.sh"

step "2/8  Bernoulli tail-risk variant (event risks only)"
TAIL_CONSTRAINT_MODE=bernoulli OUT_DIR="$REPO/results/tail_risk_bernoulli" \
  bash "$REPO/tools/run_tail_risk_primary_suite.sh"

step "3/8  Bound-sensitivity (Hoeffding / empirical-Bernstein / Clopper-Pearson)"
bash "$REPO/tools/run_tail_risk_bound_sensitivity.sh"

step "4/8  Nested low-multiplicity refusal (eps=0.01)"
bash "$REPO/tools/run_nested_tail_risk_suite.sh"

step "5/8  Certification frontier (gamma sweep 0.05..0.25)"
bash "$REPO/tools/run_nested_gamma_frontier.sh"

step "6/8  Decision baselines"
bash "$REPO/tools/run_decision_baseline_suite.sh"

step "7/8  Per-image action-selection (utility-frontier diagnostic)"
bash "$REPO/tools/run_per_image_action_selection_suite.sh"

step "8/8  Standalone-segmenter stress portfolio (UNet as alternative action)"
mkdir -p "$REPO/results/learned_refiner_stress"
"$PYTHON" "$REPO/tools/build_learned_refiner_stress_portfolio.py" \
  --out_dir "$REPO/results/learned_refiner_stress" \
  --python "$PYTHON" \
  --main_tex "$REPO/results/learned_refiner_stress/learned_refiner_stress.tex" \
  --supp_tex "$REPO/results/learned_refiner_stress/learned_refiner_stress_detail.tex" \
  --mri_volume_tex "$REPO/results/learned_refiner_stress/mri_volume_diagnostic.tex"

echo
echo "=============================================================="
echo "Done. Regenerated tables are under results/<suite>/."
echo "Primary refusal result:"
echo "  results/tail_risk_primary/tail_risk_primary_summary_best.tex"
echo "Compare against the committed reference tables in results/paper_tables/."
echo "=============================================================="
