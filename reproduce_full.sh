#!/usr/bin/env bash
# =============================================================================
# SafeRefine Tier-2 reproduction (full pipeline from raw public data).
#
# This regenerates the per-image action CSVs in results/refiner_zoo_uncert/ that
# Tier 1 (reproduce_cpu.sh) consumes. It requires a GPU, the public datasets,
# and requirements-full.txt. See REPRODUCE.md for dataset download + staging.
#
# This script is a documented skeleton: most heavy steps are SLURM batch jobs.
# Set STAGING to your staged dataset root, then run the steps you need.
# =============================================================================
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
STAGING="${STAGING:?set STAGING=/path/to/staged/datasets (see REPRODUCE.md)}"
export ROOT="$REPO" PYTHON

echo "==> repo:    $REPO"
echo "==> staging: $STAGING"

cat <<'EOF'

Tier-2 runs as SLURM jobs (single A100 per host run in the paper). Submit in
order; each step is also documented in REPRODUCE.md:

  1. Build NPZ bundles from staged data:
       ROOT=$PWD STAGING=$STAGING bash slurm/build_binary_npz_datasets.sbatch
       python tools/build_msd_heart_npz.py --help     # MSD Heart MRI

  2. Train hosts (GraphSeg + small UNet, seed 1, 120 epochs):
       ROOT=$PWD ARCH=graphseg   RUN_TAG=mediafinal_graphseg_e120 EPOCHS=120 \
         bash slurm/submit_multimodal_binary_suite.sh
       ROOT=$PWD ARCH=unet_small RUN_TAG=mediafinal_unet_e120     EPOCHS=120 \
         bash slurm/submit_multimodal_binary_suite.sh
       ROOT=$PWD bash slurm/submit_msd_heart_mri_final.sh

  3. Evaluate the refiner zoo -> results/refiner_zoo_uncert/*.csv :
       ROOT=$PWD bash slurm/eval_binary_refiner_zoo.sbatch

  4. Run Tier-1 analysis on the regenerated CSVs:
       ./reproduce_cpu.sh

Because step 3 overwrites results/refiner_zoo_uncert/, back up the committed
CSVs first if you want to compare against the paper's exact inputs.
EOF
