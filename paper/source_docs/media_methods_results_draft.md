# MedIA Methods And Results Draft

Date: 2026-06-22

## Working Title

GraphMembrane / SafeRefine: model-agnostic harm-controlled refinement for
medical image segmentation.

The safest title shape is:

```text
SafeRefine: Model-Agnostic Harm-Controlled Refinement for Medical Image Segmentation
```

Use `GraphMembrane` as the geometry-aware mechanism name, not as the whole
paper claim.

## Paper Thesis

Modern medical segmentation pipelines often add post-processing or refinement to
improve masks, but a fixed refiner can make individual cases substantially
worse. We propose a deployment-time safety layer that wraps any host segmenter
and candidate refiner portfolio:

```text
host prediction -> candidate refiners -> calibrated action policy -> exact host fallback
```

The framework returns one operational segmentation mask. It is not an
uncertainty set and not a request for manual review. If refinement risk is too
high, the output is exactly the host prediction.

## Contributions

1. A model-agnostic safe refinement protocol for medical image segmentation,
   where arbitrary host segmenters and candidate refiners are converted into a
   calibrated action portfolio.
2. A harm-aware calibration objective and CRC-style safety filter that optimize
   expected gain while constraining harmful refinements and preserving an exact
   host fallback.
3. Geometry-aware risk scores based on prediction change, shape displacement,
   area shift, and connected-component change.
4. A reporting protocol centered on clinically legible safety quantities:
   mean gain, mean harm, worst drop, harmed-rate, and reverted-rate.
5. Multi-modality and cross-host evidence across retina, dermoscopy, and
   endoscopy, including GraphSeg and UNet hosts.

## Method

For an image `x`, a host segmenter produces a mask or logits `h(x)`. A candidate
action set is built as:

```text
A(x) = {host, r_1(h, x), ..., r_K(h, x)}
```

where `host` means returning `h(x)` exactly and each `r_k` is a candidate
refinement. In the current experiments, candidate actions include morphology
for binary segmentation and GraphMembrane/morphology-like refiners for REFUGE2.
Morphology is not the novelty claim; it is a controlled stressor and candidate
action.

On the calibration split, for each action `a` and threshold `tau`, the policy
is:

```text
pi_{a,tau}(x) = a(x)       if risk_a(x) <= tau
              = host(x)    otherwise
```

The utility used to choose among feasible policies is:

```text
utility = mean_gain - beta_harm * mean_harm
gain_i  = metric(pi(x_i), y_i) - metric(host(x_i), y_i)
harm_i  = max(0, -gain_i)
```

The practical mode allows a non-zero calibration harm-rate target
(`max_cal_harm_rate=0.25`). The strict mode sets `max_cal_harm_rate=0.0`; if no
candidate action can be certified, it returns the host exactly.

The CRC-style filter computes a Hoeffding upper confidence bound on calibration
harmed-rate, with a union correction over tested action-threshold candidates.
Only policies whose upper bound is below the requested harm-rate are feasible.

## Risk Scores

The ablation evaluates three risk scores:

- `changed`: fraction of pixels changed relative to the host.
- `geom`: geometry-aware risk

```text
geom = changed + area_delta + 0.25 * centroid_shift + 0.01 * component_delta
```

- `change_plus_geom`: `changed + geom`.

For retina/REFUGE2, the same policy template is used with graph/geometry-derived
candidate actions and risk summaries. The reporting metric is CPD. For binary
datasets, the metric is Dice.

## Experimental Protocol

Datasets:

- REFUGE2 retina optic disc/cup segmentation.
- ISIC 2018 Task 1 dermoscopic lesion segmentation.
- PH2 dermoscopic lesion segmentation.
- Kvasir-SEG endoscopic polyp segmentation.
- Kvasir-Sessile endoscopic sessile polyp segmentation.
- Polyp official aggregate external endoscopy stress set.

Hosts:

- GraphSeg/SegFormer-B0-style host.
- Dependency-free `unet_small` host for cross-host validation.

Splits:

- Binary NPZ datasets use fixed train/validation/test splits in
  `dataset_splits/*_352`.
- The action-policy evaluator splits the held-out evaluation set into
  calibration and test halves (`cal_fraction=0.5`).

Uncertainty:

- Bootstrap confidence intervals use 2000 resamples over test images.
- CI tables are in `results/bootstrap_ci/` and `results/bootstrap_ci_e120/`.
- Final internally consistent `mediafinal` CI tables are in
  `results/bootstrap_ci_mediafinal/` and should be treated as the paper source
  of truth for the final binary experiments.

## Main Results

Use this table in the main paper:

- Markdown: `results/paper_tables/main_safety_table.md`
- CSV: `results/paper_tables/main_safety_table.csv`
- LaTeX: `results/paper_tables/main_safety_table.tex`

Key readings:

- REFUGE2: fixed/calibrated refinement can improve CPD, while CRC reduces harm
  and worst drop; strict mode exactly reverts to host.
- ISIC 2018 GraphSeg: practical CRC keeps most of the fixed-refiner improvement
  with lower harm than naive fixed refinement; strict mode remains exact-host.
- Kvasir-SEG GraphSeg e120: geometry risk gives a positive practical gain
  while sharply reducing harm relative to fixed refinement.
- Kvasir-SEG UNet e120: a much stronger host shows the model-agnostic safety
  story; fixed refinement has some average gain but non-trivial worst-case
  harm, while CRC chooses mostly fallback.
- PH2 UNet e120: fixed refinement has a small positive signal; CRC is
  conservative on the small test set and preserves host safety.

## Cross-Host Evidence

Use this table to support the model-agnostic claim:

- Markdown: `results/paper_tables/cross_host_model_agnostic.md`
- CSV: `results/paper_tables/cross_host_model_agnostic.csv`
- LaTeX: `results/paper_tables/cross_host_model_agnostic.tex`

Critical interpretation:

- The framework is not tied to a single backbone: the same policy wrapper works
  for GraphSeg and UNet hosts.
- Strict mode gives exact fallback across all tested host/dataset combinations.
- Stronger hosts reduce the need for intervention; this is expected and should
  be presented as correct behavior, not as failure to improve Dice.

## Ablations

Use this table for risk-score and mode ablation:

- Markdown: `results/paper_tables/risk_ablation_crc.md`
- CSV: `results/paper_tables/risk_ablation_crc.csv`
- LaTeX: `results/paper_tables/risk_ablation_crc.tex`

Interpretation:

- `geom` is strongest in the Kvasir-SEG GraphSeg e120 setting.
- ISIC GraphSeg phase0 supports practical CRC as a utility mode.
- Strict mode consistently collapses to exact host fallback with zero measured
  harm.
- `changed` alone is a useful baseline, but geometry-aware risk is more
  defensible for MedIA because it links decisions to shape/structure changes.

## Cross-Dataset Policy Transfer

Use this table to answer the reviewer question "does the calibrated safety
policy generalize beyond within-dataset calibration?":

- Markdown: `results/paper_tables/cross_dataset_transfer.md`
- CSV: `results/paper_tables/cross_dataset_transfer.csv`
- LaTeX: `results/paper_tables/cross_dataset_transfer.tex`

Protocol:

- Calibrate the action policy on a source dataset.
- Apply the selected action and threshold to a target dataset without target-set
  recalibration.
- Canonicalize candidate actions by operation and strength, e.g.
  `binary_morph:a11.00`, so policies can transfer across datasets.

Key reading:

- ISIC -> PH2 GraphSeg gives positive dermoscopy transfer with very low harm.
- Kvasir -> Sessile is mixed but supports the fallback story.
- Kvasir -> Polyp official GraphSeg shows that practical transfer can be unsafe
  under external shift; this should be presented as a stress test.
- Strict mode gives exact host fallback in all transfer settings.

This is not a headline performance table. Its paper role is to show that the
framework exposes distribution-shift risk rather than hiding it.

## Harm-Budget Sweep

Use this table to show the safety-utility knob:

- Markdown: `results/paper_tables/harm_budget_sweep.md`
- CSV: `results/paper_tables/harm_budget_sweep.csv`
- LaTeX: `results/paper_tables/harm_budget_sweep.tex`

The sweep varies `max_cal_harm_rate` over:

```text
0.00, 0.01, 0.05, 0.10, 0.25
```

Key reading:

- At low budgets, the policy usually selects exact host fallback.
- At the practical budget (`0.25`), ISIC and Kvasir-SEG GraphSeg e120 admit
  useful refinements.
- This supports the deployment framing: users can choose conservative or
  utility-seeking behavior explicitly.

## Calibration-Set Size Sweep

Use this table to show how much calibration data the controller needs:

- Markdown: `results/paper_tables/calibration_fraction_sweep.md`
- CSV: `results/paper_tables/calibration_fraction_sweep.csv`
- LaTeX: `results/paper_tables/calibration_fraction_sweep.tex`

The sweep varies `cal_fraction` over:

```text
0.10, 0.20, 0.35, 0.50, 0.70
```

Key reading:

- With small calibration sets, the policy is conservative and mostly falls back
  to the host.
- With larger calibration sets, ISIC and Kvasir-SEG GraphSeg e120 admit useful
  practical refinements.
- This is useful for MedIA because it quantifies the validation-data requirement
  of the safety layer.

## Refiner Zoo Experiment

The refiner-zoo experiment removes the strongest remaining limitation:
dependence on a morphology-only candidate set. The evaluator generates a
broader candidate-action zoo:

- morphology close/open variants;
- `fill_holes`, `largest_cc`, and `lcc_fill`;
- small-component removal;
- probability Gaussian smoothing;
- probability Gaussian smoothing followed by largest-component/hole-fill;
- probability bilateral smoothing.

Scripts:

- evaluator: `tools/eval_binary_refiner_zoo.py`
- eval sbatch: `slurm/eval_binary_refiner_zoo.sbatch`
- one-run submitter: `slurm/submit_refiner_zoo_one.sh`
- max-8 orchestrator: `slurm/orchestrate_refiner_zoo.sh`
- Slurm wrapper for orchestrator: `slurm/orchestrate_refiner_zoo.sbatch`

Final `mediafinal` zoo outputs are available and should be cited from:

- `results/paper_tables/refiner_zoo_crc.{csv,md,tex}`
- `results/paper_tables/mediafinal_consistent_results.{csv,md,tex}`
- `results/paper_tables/mediafinal_bootstrap_crc_ablation.{csv,md,tex}`

Key reading:

- ISIC `mediafinal` gains become materially stronger with the zoo: the practical
  CRC selects `largest_cc` with positive bootstrap CI.
- Polyp official GraphSeg is rescued as a stress result: fixed morphology is
  harmful on average, while zoo CRC can select `close_only` with near-zero mean
  harm.
- Kvasir GraphSeg remains the cleanest geometry-risk positive case for the
  baseline candidate set.

## Stress And Negative Results

Use this table in supplement or a transparent limitations section:

- Markdown: `results/paper_tables/stress_negative_results.md`
- CSV: `results/paper_tables/stress_negative_results.csv`
- LaTeX: `results/paper_tables/stress_negative_results.tex`

Interpretation:

- Polyp official remains weak as a performance result. Use it as an external
  stress test, not a headline.
- Kvasir-Sessile is small and difficult. It supports the fallback story but
  should not carry the main performance claim.
- Negative fixed-refiner regimes are valuable because they show why a
  host-preserving safety layer is needed.

## Figures

Failure-analysis figures are in:

```text
results/figures/crc_reversion_cases/*.png
results/figures/mediafinal_crc_reversion_cases/*.png
```

Index:

- Markdown: `results/paper_tables/crc_reversion_figure_index.md`
- CSV: `results/paper_tables/crc_reversion_figure_index.csv`
- LaTeX: `results/paper_tables/crc_reversion_figure_index.tex`

Recommended main figure:

```text
host prediction | fixed refiner output | CRC output | risk/threshold annotation
```

Caption message:

> A fixed refiner can catastrophically remove or distort a clinically relevant
> mask. The calibrated risk policy identifies the refinement as unsafe and
> returns the host prediction exactly.

## What Not To Claim

Do not claim:

- morphology is novel;
- strict mode maximizes Dice;
- Polyp official is a performance success;
- the method replaces uncertainty quantification;
- GraphMembrane alone is the whole contribution.

Claim instead:

- safe action selection is the contribution;
- strict mode is a deployment safety setting;
- practical mode trades small measured risk for utility;
- failure and fallback are first-class outcomes;
- geometry-aware risk provides interpretable action gating.

## Current Completion Status

Done:

- improved endoscopy host runs;
- second host/backbone runs;
- full policy matrix for practical/strict and all three risk scores;
- bootstrap CI for phase0, e120, and final mediafinal runs;
- cross-dataset calibration-transfer experiments;
- harm-budget safety/utility sweep;
- calibration-fraction sweep;
- case figures for fixed-refiner harm and CRC reversion;
- refiner-zoo run for mediafinal candidate portfolios;
- reproducibility manifest with artifact hashes;
- main/cross-host/ablation/stress tables in CSV, Markdown, and LaTeX.

Optional before final submission:

- one stronger external-polyp host if we want Polyp official to be more than a
  stress test;
- one learned binary GraphMembrane candidate action if the zoo is not enough;
- one additional modality such as ultrasound or MRI if data access is easy.
