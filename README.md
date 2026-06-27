# SafeRefine

**Certification Frontiers for Risk-Controlled Segmentation Refinement**

SafeRefine is a model-agnostic calibration layer that wraps an existing host
segmenter and a portfolio of candidate refinement *actions* (morphology,
component filtering, probability smoothing, geometry-aware refiners, or an
alternative standalone segmenter). It accepts a non-host action **only** when
its risk score passes a learned threshold *and* the selected action–threshold
pair satisfies finite-sample upper-confidence constraints on harmed-rate,
large-drop rate, and mean harm. Otherwise it returns the host prediction
exactly. The output is always a single usable segmentation mask.

The paper's **primary result is a calibrated refusal**: under a conservative
tail-risk contract, no tested refiner is certifiable on the available
calibration evidence across eight settings, so the controller returns the host.
This repository reproduces that result — and the full certification-frontier,
bound-sensitivity, positive-control, and utility-diagnostic analyses around it.

---

## Two ways to reproduce

| Tier | What it reproduces | Needs | Time |
|------|--------------------|-------|------|
| **1 — exact, CPU-only** | Every SafeRefine table/number (calibration, tail-risk refusal, frontier, bound sensitivity, decision baselines, standalone-segmenter stress) | `numpy`, `pandas`, `scipy`, `pytest`. No GPU, **no dataset download**. | minutes |
| **2 — full pipeline** | Regenerates everything from scratch: build datasets → train hosts → evaluate refiners → produce the per-image CSVs that Tier 1 consumes | GPU, the public datasets, `requirements-full.txt` | hours–days |

Tier 1 is possible because the per-image gain/harm/risk metrics for every
candidate action are **committed** in [`results/refiner_zoo_uncert/`](results/refiner_zoo_uncert).
The certification analysis is deterministic post-processing over those CSVs.

### Quick start (Tier 1)

```bash
pip install -r requirements.txt
./reproduce_cpu.sh          # regenerates all certification tables under results/
python -m pytest tests/ -q  # fast smoke test of the primary refusal claim
```

See **[REPRODUCE.md](REPRODUCE.md)** for the full step-by-step guide, including
Tier 2 (datasets, host training, and the SLURM scripts).

---

## Repository layout

```
tools/                      core code (calibration, evaluation, summarizers, training)
  eval_safe_action_portfolio.py   <- SafeRefine controller: the calibration contract
  eval_binary_refiner_zoo.py      <- produces per-image action CSVs (Tier 2)
  build_learned_refiner_stress_portfolio.py  <- standalone-segmenter stress
  run_positive_control_saferefine.py         <- positive control
  summarize_*.py / run_*.sh       <- suite drivers + paper-table builders
slurm/                      SLURM batch scripts for the full GPU pipeline
dataset_splits/             train/val/test split files for the 5 datasets
results/
  refiner_zoo_uncert/       committed per-image action CSVs  (Tier 1 inputs)
  paper_tables/             reference LaTeX tables (expected Tier 1 outputs)
paper/                      manuscript, supplement, figures, bibliography
tests/                      CPU smoke test of the certification contract
reproduce_cpu.sh            Tier 1 orchestrator
reproduce_full.sh          Tier 2 orchestrator (documented skeleton)
```

## The certification contract in one place

The whole method is [`tools/eval_safe_action_portfolio.py`](tools/eval_safe_action_portfolio.py).
A non-host action is certified only if all three upper bounds hold:

```
UCB[ Pr(g < 0)        ] <= alpha   (harmed-rate,    default 0.25)
UCB[ Pr(g < -0.05)    ] <= gamma   (large-drop,     default 0.10)
UCB[ E max(0, -g)     ] <= eta     (mean harm,      default 0.02)
```

where `g` is the per-image Dice gain of the action over the host, and the UCB is
a Hoeffding / empirical-Bernstein / Clopper–Pearson bound with a union
correction over all tested action–threshold–risk triples. If no non-host action
qualifies, the host action is selected and `g = 0` exactly on every case.

## Data and licensing

This repo contains **code and lightweight reproducibility artifacts only**. The
public datasets are not redistributed; download instructions are in
[REPRODUCE.md](REPRODUCE.md). Code is MIT-licensed; datasets retain their own
licenses. SafeRefine is a research artifact and **not a clinical safety
guarantee**.
