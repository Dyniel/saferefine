# Reproducing SafeRefine

There are two tiers. **Tier 1** reproduces every number in the paper exactly,
on a laptop, with no dataset download. **Tier 2** regenerates the Tier-1 inputs
from raw public data with a GPU.

---

## Tier 1 — exact, CPU-only (recommended starting point)

The per-image gain/harm/risk metrics for every candidate action on every dataset
are committed in [`results/refiner_zoo_uncert/`](results/refiner_zoo_uncert)
(8 CSVs, one per `dataset × host`). The entire SafeRefine certification analysis
is deterministic post-processing over these CSVs.

```bash
pip install -r requirements.txt        # numpy + pandas + scipy + pytest
./reproduce_cpu.sh                      # ~minutes
python -m pytest tests/ -q             # smoke test of the primary refusal claim
```

`scipy` is included so that the Clopper--Pearson bound-sensitivity analysis uses
the exact beta-quantile implementation rather than the Hoeffding fallback.

`reproduce_cpu.sh` writes regenerated tables under `results/<suite>/` and prints
a summary. Compare them against the committed reference tables in
[`results/paper_tables/`](results/paper_tables).

What it runs (each step is a self-contained script you can also run alone):

| Step | Script | Paper element |
|------|--------|---------------|
| Primary tail-risk contract (8 settings) | `tools/run_tail_risk_primary_suite.sh` | host-fallback refusal (main result) |
| Bernoulli variant | `run_tail_risk_primary_suite.sh` (`TAIL_CONSTRAINT_MODE=bernoulli`) | event-risk-only ablation |
| Bound sensitivity | `tools/run_tail_risk_bound_sensitivity.sh` | Hoeffding / emp-Bernstein / Clopper–Pearson |
| Nested low-multiplicity | `tools/run_nested_tail_risk_suite.sh` | reduced threshold multiplicity |
| Certification frontier | `tools/run_nested_gamma_frontier.sh` | γ ∈ {0.05…0.25} sweep |
| Decision baselines | `tools/run_decision_baseline_suite.sh` | comparison policies |
| Per-image action selection | `tools/run_per_image_action_selection_suite.sh` | utility-frontier diagnostic |
| Standalone-segmenter stress | `tools/build_learned_refiner_stress_portfolio.py` | UNet-as-alternative-action |

All scripts honor `ROOT` (repo path) and `PYTHON` overrides; `reproduce_cpu.sh`
sets these for you.

### The contract

The controller is [`tools/eval_safe_action_portfolio.py`](tools/eval_safe_action_portfolio.py).
A non-host action is certified only if, with the union-corrected upper bounds,

```
UCB[ Pr(g < 0)     ] <= alpha   (--max_cal_harm_rate,  default 0.25)
UCB[ Pr(g < -0.05) ] <= gamma   (--max_cal_drop05_rate, default 0.10)
UCB[ E max(0,-g)   ] <= eta     (--max_cal_mean_harm,   default 0.02)
```

`g_i` = Dice(action_i) − Dice(host_i). If no non-host action qualifies, the host
action is selected (`g = 0` on every case, revert-rate 1.0). `--bound_mode`
selects Hoeffding / empirical_bernstein / clopper_pearson; `--tail_constraint_mode`
selects `full` (all three), `bernoulli` (two event risks), or the `*_severe`
variants that add a `Pr(g < -0.20)` budget.

---

## Tier 2 — full pipeline from raw data (GPU)

```bash
pip install -r requirements-full.txt   # adds torch, opencv, matplotlib, Pillow, nibabel
```

### 1. Obtain the public datasets

None are redistributed here. Download each from its official source and stage it
under a single `$STAGING` directory.

| Dataset | Source |
|---------|--------|
| ISIC 2018 Task 1 (lesion boundary) | https://challenge.isic-archive.com/data/#2018 |
| PH2 | https://www.fc.up.pt/addi/ph2%20database.html |
| Kvasir-SEG | https://datasets.simula.no/kvasir-seg/ |
| Polyp benchmarks (CVC-300, CVC-ColonDB, ETIS-LaribPolypDB, Kvasir, CVC-ClinicDB) | standard PraNet train/test bundle (`TrainDataset`/`TestDataset`) |
| MSD Heart (Task02_Heart) | http://medicaldecathlon.com/ |

Expected staging layout (matches `slurm/build_binary_npz_datasets.sbatch`):

```
$STAGING/
  isic/ISIC2018_Task1-2_{Training,Validation,Test}_Input/
  isic/ISIC2018_Task1_{Training,Validation,Test}_GroundTruth/
  polyps/NewTRimage/  polyps/NewTRmask/            # PraNet train
  polyps_test_flat/images/  polyps_test_flat/masks/ # merged PraNet test
  kvasir/ ...   ph2/ ...   msd_heart/Task02_Heart/...
```

The exact split files used in the paper are committed in
[`dataset_splits/`](dataset_splits) so your build reuses the same train/val/test
partitions (image-level for 2D; patient-level for MSD Heart).

### 2. Build NPZ bundles

```bash
ROOT=$PWD STAGING=/path/to/staging \
  bash slurm/build_binary_npz_datasets.sbatch        # ISIC + polyps (and friends)
python tools/build_msd_heart_npz.py --help           # MSD Heart MRI -> NPZ (patient splits)
```

Outputs land in `data_npz/<dataset>_352.npz` (git-ignored).

### 3. Train hosts (GraphSeg + small UNet, seed 1, 120 epochs `mediafinal`)

```bash
ROOT=$PWD ARCH=graphseg  RUN_TAG=mediafinal_graphseg_e120 EPOCHS=120 \
  bash slurm/submit_multimodal_binary_suite.sh
ROOT=$PWD ARCH=unet_small RUN_TAG=mediafinal_unet_e120     EPOCHS=120 \
  bash slurm/submit_multimodal_binary_suite.sh
ROOT=$PWD bash slurm/submit_msd_heart_mri_final.sh        # MRI hosts
```

Or call `tools/train_binary_host.py` directly (see `--help`). Checkpoints are
written to `runs/binary_host/.../ckpt/best.pt` (git-ignored).

### 4. Evaluate the refiner zoo (produces the Tier-1 CSVs)

```bash
ROOT=$PWD bash slurm/eval_binary_refiner_zoo.sbatch
```

This writes `results/refiner_zoo_uncert/<dataset>_<host>_zoo.csv` — the exact
inputs Tier 1 consumes. From here, run `./reproduce_cpu.sh`.

### 5. Positive control and figures (optional)

```bash
python tools/run_positive_control_saferefine.py --help   # synthetic artifact-removal control
python tools/plot_media_submission_figures.py --help      # paper figures
```

---

## Notes on exactness

- Tier 1 is fully deterministic given the committed CSVs.
- Bootstrap confidence intervals use a fixed resample count (2000) over test
  images; the RNG seed is fixed in the bootstrap utilities.
- Host training (Tier 2) is subject to the usual GPU/cuDNN nondeterminism, so
  regenerated CSVs may differ at the 3rd–4th decimal; the certification
  *decisions* (host fallback under the conservative contract) are stable.
- MSD Heart metrics are slice-level under patient-disjoint splits and are a
  modality stress test, not volumetric clinical validation (see paper §Results).

## Hardware used in the paper

Host training used a single NVIDIA A100 (80 GB) per run via SLURM; see the
`slurm/*.sbatch` headers for partitions and resource requests. Tier 1 needs no
GPU.
