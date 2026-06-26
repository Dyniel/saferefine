# Submission Text Blocks

## Title Options

1. SafeRefine: Harm-Aware Action Selection for Safe Medical Segmentation Refinement
2. SafeRefine: Risk-Controlled Acceptance of Segmentation Refinement with Exact Host Fallback
3. Host-Preserving Safe Refinement for Medical Image Segmentation

Recommended: option 1.

## Highlights

- We formulate medical segmentation refinement as harm-aware action acceptance.
- The framework wraps arbitrary host segmenters and candidate refiners with exact host fallback.
- Practical and strict policies expose explicit utility-safety trade-offs.
- Safety value is measured as harm reduction and worst-drop improvement over fixed refinement.
- Final experiments span dermoscopy and endoscopy, GraphSeg and UNet hosts, and a refiner zoo.

## Short Abstract

Segmentation refiners can improve average medical image segmentation accuracy
but may severely harm individual predictions. We propose SafeRefine, a
model-agnostic safety layer that wraps a host segmenter and candidate refiners,
then calibrates when to apply refinement and when to return the host prediction
exactly. The policy uses a harm-aware utility and a conformal-risk-control style
filter over action thresholds. Across dermoscopy and endoscopy tasks, GraphSeg
and UNet hosts, and morphology plus a broader refiner zoo, strict policies
recover the host in all final runs, while practical policies improve selected
settings with explicit harm accounting. The framework reframes refinement as a
deployment decision: apply a candidate refiner only when calibrated risk is
acceptable, otherwise preserve the original model output.

## Graphical Abstract Concept

Left-to-right pipeline:

```text
medical image -> host segmenter -> host mask/logits
                         |
                         v
              candidate refiner portfolio
       host / morphology / component / smoothing / graph
                         |
                         v
             calibrated risk-action controller
              accept safe refinement or revert
                         |
                         v
                one output segmentation mask
```

Add small bottom strip:

```text
report: gain | harm | worst drop | harmed-rate | revert-rate
```

Visual message: the host path should be visually continuous and labeled "exact
fallback". The refiner path should pass through a risk gate.

## Cover Letter Draft

Dear Editors,

We submit our manuscript, "SafeRefine: Harm-Aware Action Selection for Safe
Medical Segmentation Refinement", for consideration in Medical Image Analysis.

Medical segmentation pipelines often use post-processing or refinement to
improve masks, yet fixed refiners can substantially degrade individual cases.
This manuscript addresses that deployment problem by formulating refinement as
calibrated safe action selection. Given a host segmenter and a portfolio of
candidate refiners, SafeRefine either applies a calibrated refinement action or
returns the host prediction exactly. The method outputs a single operational
segmentation mask and reports safety quantities that are clinically legible:
mean gain, mean harm, worst drop, harmed-rate, and revert-rate.

The contribution is methodological and model-agnostic rather than a new
segmentation backbone. We evaluate the same wrapper across multiple medical
segmentation settings, host families, and candidate-refiner portfolios. The
experiments emphasize safety value over fixed refinement: calibrated policies
prevent much of the mean harm and worst-case degradation caused by
unconditional refiners, while practical policies recover utility when the
calibration data support intervention.

We believe the work fits Medical Image Analysis because it combines a general
algorithmic framework, geometry-aware refinement/risk mechanisms, multi-domain
medical segmentation experiments, and a strong focus on deployment reliability
and reproducibility.

Sincerely,

[Authors]

## Data And Code Availability Draft

All scripts used to train hosts, evaluate candidate refiners, calibrate action
policies, compute bootstrap confidence intervals, generate figures, and build
paper tables are included in the accompanying repository. Dataset split files,
run manifests, and artifact hashes are provided to support reproducibility.
Public datasets are not redistributed; download/preprocessing scripts and split
definitions are provided where licensing permits.

Key reproducibility artifacts:

- `results/reproducibility/mediafinal_repro_manifest.json`
- `results/reproducibility/mediafinal_artifact_hashes.csv`
- `results/paper_tables/paper_table_manifest.json`

## Limitations Paragraph

SafeRefine is a safety layer around existing segmenters and refiners; it is not
intended to replace strong segmentation backbones. Strict mode can be
conservative and often returns the host prediction, especially on small
calibration sets or under external shift. Practical mode can improve utility but
does not eliminate all test-time harm, so worst-drop and harmed-rate must be
reported. The current refiner zoo includes simple morphology, component, and
probability-smoothing operations; learned graph-physical refiners can be added
as future candidate actions.
