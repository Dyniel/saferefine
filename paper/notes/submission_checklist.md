# MedIA Submission Checklist

## Must Finish Before Upload

- Convert `docs/submission/media_manuscript_draft.md` into LaTeX or Elsevier
  Word format.
- Add citations and BibTeX entries for conformal risk control, conformal
  segmentation, failure detection/deferral, classical morphology/CRF/active
  contours, learned mask refinement, and graph-based medical refinement.
- Insert final tables from `results/paper_tables/*.tex`.
- Insert submission figures from `results/submission_figures/` and case panels
  from `results/figures/mediafinal_crc_reversion_cases/`.
- Add dataset table: dataset name, modality, train/validation/test split,
  number of images, host architecture, metric, calibration/test protocol.
- Add exact implementation details: seeds, image size 352, calibration fraction
  0.5, bootstrap resamples 2000, practical harm budget 0.25, strict harm budget
  0.0, CRC confidence 0.10.
- Add code/data availability statement.
- Add declaration of competing interests.
- Add author contributions and funding acknowledgements.
- Prepare graphical abstract.
- Prepare highlights.

## Main Paper Figure Set

1. Method schematic:
   host segmenter -> candidate actions -> risk-action controller -> exact host
   fallback or accepted refinement.
2. Utility-safety scatter:
   `results/submission_figures/fig_gain_harm_tradeoff.png`.
3. Key bootstrap effects:
   `results/submission_figures/fig_key_bootstrap_effects.png`.
4. Strict fallback summary:
   `results/submission_figures/fig_strict_exact_fallback.png`.
5. Qualitative failure/reversion panels:
   choose one ISIC zoo, one Kvasir GraphSeg, and one external polyp case from
   `results/figures/mediafinal_crc_reversion_cases/`.

## Main Paper Tables

Primary:

- `results/paper_tables/mediafinal_consistent_results.tex`
- `results/paper_tables/mediafinal_bootstrap_crc_ablation.tex`
- `results/paper_tables/refiner_zoo_crc.tex`

Supplement:

- `results/paper_tables/cross_host_model_agnostic.tex`
- `results/paper_tables/risk_ablation_crc.tex`
- `results/paper_tables/cross_dataset_transfer.tex`
- `results/paper_tables/harm_budget_sweep.tex`
- `results/paper_tables/calibration_fraction_sweep.tex`
- `results/paper_tables/stress_negative_results.tex`
- `results/paper_tables/mediafinal_reproducibility_manifest.tex`

## Critical Narrative Rules

- Use `mediafinal` tables as the binary source of truth.
- Do not present stale phase0/e120 rows as final headline results.
- Present external polyp results as stress/safety evidence.
- Present strict mode as a conservative deployment mode, not a Dice optimizer.
- Present morphology as one candidate action, not the method.
- Present GraphMembrane/geometry as a mechanism inside the broader safety
  framework.

## Reviewer Risk Audit

- If a reviewer says "too conservative": answer that strict mode is meant for
  zero-harm deployment; practical mode is the utility setting.
- If a reviewer says "not SOTA segmentation": answer that this is a
  model-agnostic wrapper around host segmenters/refiners, not a backbone paper.
- If a reviewer says "just CRC": answer that CRC-like calibration is used to
  select concrete segmentation interventions with exact host fallback.
- If a reviewer says "morphology is old": agree; morphology is a candidate
  stressor, and the refiner zoo demonstrates broader action selection.
- If a reviewer asks about harm: point to mean harm, worst drop, harmed-rate,
  revert-rate, and case panels.

## Optional But Valuable If Time Remains

- Add one learned graph-physical refiner action to the zoo.
- Add one stronger external polyp host if the authors want external polyp to be
  a performance result instead of a stress result.
- Run n>1 seeds for selected final hosts if compute allows.
- Add a small runtime/cost table for candidate action generation and policy
  evaluation.
