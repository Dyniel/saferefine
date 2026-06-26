# Related-Work Defense For MedIA Submission

## Core Positioning Sentence

SafeRefine is not a new segmentation backbone, not a morphology paper, and not a
generic conformal prediction paper. It is a model-agnostic deployment layer that
turns arbitrary candidate refiners into calibrated segmentation actions with an
exact host fallback and explicit harm accounting.

## What Is Close

### Conformal risk control

Conformal risk control is the closest statistical ancestor. It provides
calibration machinery for bounding risks. Our distinction is the operational
object:

- CRC often calibrates prediction sets, thresholds, abstention, or risk-bounded
  outputs.
- SafeRefine calibrates an action from a concrete segmentation refiner
  portfolio.
- The output is one segmentation mask, not a set or map.
- The host action is always in the portfolio and returns the original host mask
  exactly.

Recommended manuscript wording:

> We use a CRC-style feasibility filter, but our contribution is not a new
> conformal bound. The contribution is a host-preserving refinement protocol in
> which risk calibration decides whether an actual segmentation intervention is
> applied.

### Conformal segmentation and uncertainty sets

Conformal semantic segmentation and biomedical conformal prediction methods
usually produce uncertainty sets, calibrated regions, or coverage guarantees.
SafeRefine is different because it produces a single usable segmentation mask.
It can be used downstream without asking a clinician or downstream algorithm to
interpret a set-valued output.

### Failure detection and deferral

Failure detection predicts whether the segmentation is unreliable. Deferral
systems route uncertain cases to a human or expert model. SafeRefine does not
only flag risk; it chooses among available actions. The fallback is not human
review but the original host prediction.

### Fixed post-processing and morphology

Morphology is prior art and must not be claimed as novel. In our paper,
morphology is a stress-test candidate action. The important observation is that
fixed morphology can help on average while causing severe case-level harm. The
method contribution is the calibrated decision rule that can apply, weaken, or
reject such actions.

### Graph/physical refinement

Graph and geometry-aware refinement are also occupied. The safest claim is:

- GraphMembrane/geometry supplies interpretable candidate actions and risk
  features.
- The paper's general method is the action-selection wrapper.
- The graph-physical language should support the mechanism, not replace the
  safety claim.

## Comparison Table To Include

| Family | Single mask output | Uses concrete refiners | Exact host fallback | Harm/gain reporting | Model-agnostic wrapper |
| --- | --- | --- | --- | --- | --- |
| Fixed morphology/post-processing | yes | yes | no | usually no | partly |
| Learned mask refiners | yes | yes | no | usually no | often |
| Conformal segmentation sets | no/varies | no | no | risk-specific | yes |
| Failure detection/deferral | no/varies | no/varies | defer/flag | often | varies |
| SafeRefine | yes | yes | yes | yes | yes |

## Claims We Can Safely Make

- SafeRefine converts post-hoc segmentation refinement into calibrated action
  selection.
- The host prediction is always recoverable exactly.
- Strict mode provides a conservative deployment setting that returns the host
  when safe refinement cannot be certified.
- Practical mode exposes an explicit safety-utility trade-off.
- Reporting mean gain without harm, worst drop, and revert-rate is insufficient
  for medical refinement.
- The framework is refiner-agnostic: morphology is only one candidate family.

## Claims To Avoid

- Do not claim a new morphology operator.
- Do not claim a new conformal guarantee.
- Do not claim strict mode improves Dice.
- Do not claim the external polyp stress set is a performance win for the host.
- Do not claim GraphMembrane alone is the full contribution.
- Do not overstate state-of-the-art segmentation performance.

## Reviewer-Anticipated Questions

### "Isn't this just conformal risk control?"

No. CRC supplies a calibration principle. SafeRefine defines the segmentation
intervention problem: choose between host and candidate refiners, output one
mask, report harm/gain/reversion, and preserve the host exactly when risk is too
high.

### "Isn't this just uncertainty-based rejection?"

No. Rejection flags or defers. SafeRefine selects a concrete segmentation action
or returns the host. It is a deployment policy, not only a detector.

### "Why are many strict rows host fallback?"

Because strict mode is intentionally a zero-harm deployment setting. If the
calibration set cannot certify a non-host action, returning the host is the
correct behavior. The practical mode is the utility-seeking setting.

### "Why include weak external polyp hosts?"

Because they show the failure mode the method is designed for: fixed refinement
can be harmful under shift. The external polyp setting is stress/safety
evidence, not a headline performance claim.

### "Why not compare against every modern refiner?"

The method is a wrapper around refiners, not a replacement for them. The refiner
zoo shows that the controller can operate beyond morphology. Additional learned
refiners can be added as actions in future work or supplement if compute allows.
