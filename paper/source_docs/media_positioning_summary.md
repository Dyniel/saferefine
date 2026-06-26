# MedIA Positioning Summary

Date: 2026-06-22

## Core Claim

We should position the work as a model-agnostic safety layer for medical image
segmentation:

```text
host segmentation -> candidate refiners -> calibrated safe action policy -> host fallback
```

The contribution is not a new segmentation backbone and not morphology itself.
The contribution is an operational, host-preserving refinement framework that
decides when to apply a post-hoc refinement and when to revert to the original
host prediction. The key reported quantities are mean gain, mean harm,
worst-drop, harmed-rate, reverted-rate, and strict exact fallback.

## Current Experimental Coverage

| Modality/task | Dataset/run | Test n | Host metric | Best fixed refiner | Fixed gain | Fixed harm | Practical CRC | Strict CRC |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| Retina / optic disc-cup | REFUGE2 | 200 | 1.6388 CPD | Morph | +0.0096 | 0.0003 | +0.0026 gain, 0.00005 harm | exact host |
| Dermoscopy / lesion | ISIC 2018 GraphSeg | 1000 | 0.7705 Dice | Morph a11 | +0.0120 | 0.0113 | +0.0117 gain, 0.0040 harm | exact host |
| Dermoscopy / lesion | PH2 UNet e120 | 30 | 0.9078 Dice | Morph a7 | +0.0053 | 0.0004 | host fallback | exact host |
| Endoscopy / polyp | Kvasir-SEG GraphSeg e120 | 150 | 0.4877 Dice | Morph a9 | +0.0074 | 0.0424 | +0.0095 gain, 0.0016 harm | exact host |
| Endoscopy / polyp | Kvasir-SEG UNet e120 | 150 | 0.8412 Dice | Morph a11 | +0.0078 | 0.0082 | mostly fallback, near-zero harm | exact host |
| Endoscopy / sessile polyp | Kvasir-Sessile UNet e120 | 30 | 0.3296 Dice | Morph a5 | +0.0074 | 0.0210 | host fallback | exact host |
| Endoscopy / external polyp tests | Polyp official GraphSeg e120 | 798 | 0.2559 Dice | Morph a11 | -0.0211 | 0.0524 | host fallback | exact host |

Interpretation:

- The strict policy behaves as intended across all tested datasets: it reverts
  to the host and produces zero measured harm.
- The practical policy gives real positive results where the refiner is helpful
  and calibratable, most clearly ISIC GraphSeg and Kvasir-SEG GraphSeg e120.
- PH2 shows that useful fixed refinement exists, but the current calibration is
  conservative and falls back.
- Endoscopy now has a split role: Kvasir-SEG with the UNet e120 host shows that
  host weakness can be fixed, while Polyp official remains a negative external
  stress test rather than a performance headline.

## Position Relative To Prior Art

Conformal Risk Control is close but not the same claim. CRC controls expected
monotone risks and includes image segmentation examples, including polyp false
negative-rate control. Our angle is different: we use risk calibration to choose
between concrete segmentation actions/refiners and an exact host fallback, then
report harm and reverted-rate as first-class deployment metrics.

Conformal semantic segmentation and biomedical conformal confidence sets are
also close. They usually produce uncertainty sets, confidence sets, or coverage
maps. Our output remains a single operational segmentation decision: keep the
host, or safely apply a candidate refinement.

Deferred / human-AI segmentation frameworks are adjacent. They route uncertain
regions to a human or expert model. Our work is an autonomous post-hoc safety
portfolio: no human is required, and the fallback is the existing host model.

Morphological post-processing and conformalized morphology are prior art. We
must not claim dilation/erosion as novel. Morphology is only a candidate action
used to demonstrate the safety layer. The stronger novelty is the calibrated
action portfolio and harm-aware fallback contract.

Graph/physical positioning: GraphMembrane should be framed as a geometry-aware
risk and candidate-action mechanism, not as the sole performance driver. This
matches Medical Image Analysis' interest in geometrical, statistical, and
physical models for biomedical image algorithms.

## MedIA Fit

This is a plausible Medical Image Analysis submission if framed as a general
methodological safety paper rather than a narrow retina or polyp paper.

Strengths for MedIA:

- Multi-modality evidence: retina, dermoscopy, endoscopy.
- Model-agnostic deployment contract: works after an existing segmenter.
- Clinically legible safety metrics: harm, worst drop, harmed-rate, fallback
  rate, exact host equivalence in strict mode.
- Geometry-aware mechanism: graph/shape/risk scores are not just black-box
  uncertainty.
- Negative results are informative: unsafe refiners are rejected by design.

Risks:

- External polyp generalization is still weak, so Polyp official should be sold
  as a stress/safety negative rather than a performance improvement.
- Strict mode is safe but often conservative; the paper must explicitly sell
  this as a deployment option, not as a Dice maximizer.
- Prior art around CRC/conformal segmentation is close, so the novelty must be
  stated as safe action selection with exact fallback and harm accounting.

## Recommended Next Experiments

1. Improve endoscopy hosts before final tables. The current Kvasir/polyps
   numbers are useful as a safety stress test, but too weak as a primary
   performance result.
2. Run cross-host validation. The repository now supports a dependency-free
   `unet_small` host in addition to `graphseg/segformer_b0`, so the
   model-agnostic claim can be tested without adding external packages.
3. Use bootstrap confidence intervals for gain/harm/worst-drop in all main
   tables.
4. Report the full ablation: `changed` vs `geom` vs `change_plus_geom`, and
   practical vs strict policy.
5. Include failure-analysis panels showing fixed refinement harm and CRC host
   reversion.

## Added Paper Artifacts

Bootstrap confidence intervals were generated with 2000 resamples:

- All policies:
  `results/bootstrap_ci/binary_bootstrap_all_policies.csv`
- CRC ablation table:
  `results/bootstrap_ci/binary_bootstrap_crc_ablation.csv`
- Final mediafinal all policies:
  `results/bootstrap_ci_mediafinal/mediafinal_bootstrap_all_policies.csv`
- Final mediafinal CRC ablation:
  `results/bootstrap_ci_mediafinal/mediafinal_bootstrap_crc_ablation.csv`

Case figures were generated for Kvasir-SEG, Polyp official, and ISIC:

- Figures:
  `results/figures/crc_reversion_cases/*.png`
- Final mediafinal figures:
  `results/figures/mediafinal_crc_reversion_cases/*.png`
- Manifests:
  `results/figures/crc_reversion_cases/*_manifest.csv`
  and `results/figures/mediafinal_crc_reversion_cases/*_manifest.csv`

These figures instantiate the core safety story: fixed morphology can destroy a
host prediction, while the risk-calibrated policy reverts to the host.

## Slurm Plan With <=8 Jobs At Once

Wave 1 improves the weak endoscopy hosts and adds a second host on the two main
endoscopy datasets. It submits four train/eval pairs, i.e. eight jobs:

```bash
cd /users/project1/pt01315/emnlp/grm_media
WAVE=1 bash slurm/submit_endoscopy_host_wave.sh
```

Wave 2 extends the second-host claim to sessile polyps, PH2, and ISIC. It also
submits eight jobs:

```bash
cd /users/project1/pt01315/emnlp/grm_media
WAVE=2 bash slurm/submit_endoscopy_host_wave.sh
```

After each eval finishes, submit policy jobs one run at a time. Each command
submits six jobs, so it stays below the eight-job limit:

```bash
DATASET=kvasir_seg \
EVAL_CSV=results/binary_refinement/kvasir_seg_endo_graphseg_e120_eval.csv \
POLICY_RUN_TAG=kvasir_seg_endo_graphseg_e120_binary \
bash slurm/submit_binary_policies.sh

DATASET=polyps_official \
EVAL_CSV=results/binary_refinement/polyps_official_endo_graphseg_e120_eval.csv \
POLICY_RUN_TAG=polyps_official_endo_graphseg_e120_binary \
bash slurm/submit_binary_policies.sh

DATASET=kvasir_seg \
EVAL_CSV=results/binary_refinement/kvasir_seg_endo_unet_e120_eval.csv \
POLICY_RUN_TAG=kvasir_seg_endo_unet_e120_binary \
bash slurm/submit_binary_policies.sh

DATASET=polyps_official \
EVAL_CSV=results/binary_refinement/polyps_official_endo_unet_e120_eval.csv \
POLICY_RUN_TAG=polyps_official_endo_unet_e120_binary \
bash slurm/submit_binary_policies.sh
```

Repeat analogously for Wave 2 outputs:

- `kvasir_sessile_endo_graphseg_e120_eval.csv`
- `kvasir_sessile_endo_unet_e120_eval.csv`
- `ph2_unet_e120_eval.csv`
- `isic2018_task1_unet_e120_eval.csv`

## Current Local Artifacts

- Binary policy summary:
  `results/binary_multimodal_policy_summary.csv`
- Binary refinement summaries:
  `results/binary_refinement/*_phase0_eval.json`
- Binary policy outputs:
  `results/graphmembrane_policy/*binary*.json`
- REFUGE2 geometry-gated outputs:
  `results/graphmembrane_policy/geom_*_214306*.json`
- Bootstrap CI tables:
  `results/bootstrap_ci/*.csv`
- CRC reversion figures:
  `results/figures/crc_reversion_cases/*.png`

Final paper-table bundle:

- Main safety table:
  `results/paper_tables/main_safety_table.{csv,md,tex}`
- Cross-host/model-agnostic table:
  `results/paper_tables/cross_host_model_agnostic.{csv,md,tex}`
- Risk-score ablation:
  `results/paper_tables/risk_ablation_crc.{csv,md,tex}`
- Cross-dataset calibration-transfer:
  `results/paper_tables/cross_dataset_transfer.{csv,md,tex}`
- Harm-budget safety/utility sweep:
  `results/paper_tables/harm_budget_sweep.{csv,md,tex}`
- Calibration-fraction sweep:
  `results/paper_tables/calibration_fraction_sweep.{csv,md,tex}`
- Refiner-zoo hook after GPU jobs:
  `results/paper_tables/refiner_zoo_crc.{csv,md,tex}`
- Final internally consistent mediafinal results:
  `results/paper_tables/mediafinal_consistent_results.{csv,md,tex}`
- Final mediafinal bootstrap CI:
  `results/paper_tables/mediafinal_bootstrap_crc_ablation.{csv,md,tex}`
- Final reproducibility manifest:
  `results/paper_tables/mediafinal_reproducibility_manifest.{csv,md,tex}`
- Stress/negative results:
  `results/paper_tables/stress_negative_results.{csv,md,tex}`
- Methods/results draft:
  `docs/media_methods_results_draft.md`

Reproducibility artifacts:

- Run-level manifest:
  `results/reproducibility/mediafinal_repro_manifest.{csv,json}`
- Artifact hashes:
  `results/reproducibility/mediafinal_artifact_hashes.csv`
