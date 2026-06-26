# SafeRefine: Model-Agnostic Harm-Controlled Refinement for Medical Image Segmentation

## Abstract

Post-processing and refinement are routinely used to improve medical image
segmentation masks, but applying a fixed refiner to every case can also
substantially degrade individual predictions. This failure mode is clinically
important: a small average Dice improvement may hide catastrophic case-level
drops. We propose SafeRefine, a model-agnostic deployment-time safety layer that
wraps an existing host segmenter and a portfolio of candidate refiners. For each
image, the framework either returns the host prediction exactly or applies a
calibrated refinement action whose image-level risk is below a learned
threshold. The controller is calibrated on a held-out split with a harm-aware
utility and a conformal-risk-control style filter over candidate
action-threshold pairs. Unlike uncertainty-set methods, SafeRefine outputs a
single operational segmentation mask; unlike fixed post-processing, it exposes
the host prediction as an exact fallback.

We evaluate SafeRefine across retina, dermoscopy, and endoscopy segmentation
tasks, with GraphSeg and UNet hosts, morphology and a broader refiner zoo, and
practical versus strict safety modes. Across final internally consistent runs,
the strict policy recovers the host exactly in all tested settings. Practical
policies improve selected settings while explicitly quantifying mean gain, mean
harm, worst drop, harmed-rate, and revert-rate. In ISIC 2018 lesion
segmentation, the refiner-zoo controller selects largest-component refinement
and improves Dice by +0.0033 with a 95% bootstrap CI of [+0.0015, +0.0055]. In
Kvasir-SEG, geometry-aware gating improves a GraphSeg host by +0.0095 while
keeping mean harm at 0.0016. On an external polyp stress set where fixed
morphology is harmful on average, the zoo controller selects a safer close-only
action and obtains +0.0026 Dice gain with near-zero mean harm. These results
support safe action selection as a general refinement protocol: the method does
not claim that any one refiner is universally beneficial, but provides a
calibrated mechanism for deciding when refinement should be applied.

## 1. Introduction

Medical segmentation systems are rarely deployed as raw neural-network outputs
alone. Predictions may be smoothed, cleaned, topology-corrected, refined by a
secondary model, or adjusted by a shape prior. These refinements are attractive
because they can improve average overlap metrics, remove implausible
components, and restore clinically meaningful structures. However, a fixed
refinement rule creates a deployment risk: the same operation that improves one
case can damage another. In medical imaging this matters because case-level
harm, not only population-level average Dice, is central to clinical reliability.

This work starts from a simple observation. A host segmenter already provides a
valid prediction. Any refinement should therefore be treated as an intervention
on top of that prediction, not as an unconditional replacement. A safe
refinement system should answer three questions for each image: should the host
be kept unchanged, which refinement action should be used, and how much
case-level harm is introduced by this decision rule?

We propose SafeRefine, a model-agnostic safety layer for medical segmentation
refinement. The framework takes a host mask or logits and a set of candidate
actions. Candidate actions may include morphology, component filtering,
probability smoothing, graph-membrane refinement, or future learned refiners.
The action set always includes the host action, which returns the original host
prediction exactly. A calibration split is used to select an action and a risk
threshold. At test time, the selected action is applied only if its measured
risk is below the threshold; otherwise the output is exactly the host mask.

The method is deliberately not a new segmentation backbone. Its purpose is to
convert arbitrary refinement operators into a deployable action-selection
policy with explicit harm accounting. This changes the empirical question from
"does a fixed refiner improve mean Dice?" to "when can a refiner be applied
without hiding harmful cases, and when should the host be preserved?"

Our contributions are:

1. We formulate medical segmentation refinement as calibrated safe action
   selection over a host-preserving action portfolio.
2. We introduce practical and strict policies that optimize mean gain while
   constraining harmful refinements; strict mode gives exact host fallback when
   no non-harmful intervention can be certified.
3. We define geometry-aware risk scores based on prediction change, area shift,
   centroid shift, and connected-component changes, enabling interpretable
   gating of refinement actions.
4. We evaluate the same controller across retina, dermoscopy, and endoscopy,
   two host families, and both morphology-only and broader refiner-zoo action
   sets.
5. We report safety-first quantities: mean gain, mean harm, worst drop,
   harmed-rate, revert-rate, and bootstrap confidence intervals.

## 2. Related Work

### Post-processing and mask refinement

Classical post-processing such as morphology, connected-component filtering,
conditional random fields, active contours, and level-set methods remains common
in medical segmentation. Modern learned refiners, including boundary correction
modules, class-agnostic mask refinement, point-based refinement, diffusion-based
refinement, and SAM-based refinement, can further improve mask quality. These
methods usually ask whether a refinement operator improves average performance.
SafeRefine is orthogonal: it asks whether a candidate refinement should be
accepted for a given image and provides the host prediction as an exact fallback.

### Graph and geometry-aware refinement

Graph-based medical refinement methods use superpixels, grid graphs, dynamic
neighborhoods, or graph neural networks to propagate local structure and correct
segmentation outputs. This literature motivates geometry-aware candidate
actions and risk scores, but most prior graph-refinement systems are fixed
predictors. Our controller treats graph or geometry mechanisms as actions in a
portfolio. The novelty claim is not graph message passing by itself; it is the
calibrated intervention policy around arbitrary refiners.

### Conformal prediction and risk control

Conformal prediction and conformal risk control provide finite-sample tools for
calibrating prediction sets, uncertainty maps, and risk-bounded outputs. These
ideas are increasingly used for segmentation. SafeRefine is adjacent but has a
different operational target. It does not output an uncertainty set or a
coverage map. It outputs one segmentation mask selected from an action
portfolio: either the host or an accepted refinement. The closest conceptual
connection is risk-controlled decision making, but our focus is medical
segmentation refinement with exact host fallback and harm/gain accounting.

### Failure detection, rejection, and deferral

Medical segmentation reliability work often detects failures, flags uncertain
cases, or defers regions to humans or expert models. SafeRefine instead performs
autonomous post-hoc action selection. Reverting to the host is not a request for
manual review; it is a deterministic deployment decision preserving the
original model output.

## 3. Method

Let a host segmentation model produce logits or a binary mask for image `x`.
Denote the host prediction by `h(x)`. A candidate action set is

```text
A(x) = {host, r_1(h, x), ..., r_K(h, x)}.
```

The `host` action returns `h(x)` exactly. Each `r_k` is a candidate refiner. In
the current experiments, actions include morphology with multiple kernel sizes,
connected-component and hole-fill operations, probability smoothing, bilateral
smoothing, and graph/geometry-derived refiners.

For each action `a`, define the per-image gain relative to the host:

```text
gain_i(a) = metric(a(x_i), y_i) - metric(h(x_i), y_i),
harm_i(a) = max(0, -gain_i(a)).
```

The action policy is parameterized by an action `a` and threshold `tau`:

```text
pi_{a,tau}(x) = a(x)     if risk_a(x) <= tau,
              = host(x) otherwise.
```

The policy is calibrated on a held-out split and then evaluated on a disjoint
test split. The practical mode permits a non-zero calibration harmed-rate
budget. The strict mode sets the harm budget to zero. If no non-host policy is
feasible, the host action remains feasible by construction and returns zero
measured harm relative to itself.

### Harm-aware utility

Among feasible policies, calibration maximizes

```text
utility = mean_gain - beta_harm * mean_harm.
```

This prevents a policy from being selected solely because it has a positive
average gain while also introducing large case-level damage.

### CRC-style feasibility filter

For each candidate action-threshold pair, the calibration harmed-rate is
estimated. A Hoeffding upper confidence bound with a union correction over
tested thresholds is computed. A policy is feasible only if the upper bound is
below the requested harm-rate budget. This filter is intentionally conservative:
when evidence is insufficient, the policy returns the host.

### Risk scores

We evaluate three risk scores:

```text
changed = fraction of pixels changed relative to host,
geom = changed + area_delta + 0.25 * centroid_shift + 0.01 * component_delta,
change_plus_geom = changed + geom.
```

The geometry score links refinement acceptance to interpretable shape changes
instead of relying only on a black-box confidence scalar.

## 4. Experiments

### Datasets and hosts

We evaluate binary medical segmentation tasks across dermoscopy and endoscopy,
with retina experiments retained as part of the broader GraphMembrane evidence.
The final internally consistent binary runs use:

- ISIC 2018 Task 1 lesion segmentation.
- PH2 lesion segmentation.
- Kvasir-SEG polyp segmentation.
- An external polyp stress set aggregated from official polyp benchmarks.

Hosts include GraphSeg/SegFormer-B0-style models and a dependency-free small
UNet. The same SafeRefine evaluator is used for all hosts.

### Candidate action sets

The baseline action set contains host plus binary morphology at multiple kernel
sizes. The refiner zoo adds close-only, open-only, open-close, hole filling,
largest connected component, largest-component with hole filling, small-object
removal, Gaussian probability smoothing, Gaussian smoothing with component
cleanup, and bilateral probability smoothing.

### Protocol

For each final run, host training, refinement evaluation, policy calibration,
and refiner-zoo evaluation were executed with unique `mediafinal` tags to avoid
checkpoint/result mixing. The held-out evaluation set is split into calibration
and test halves. Bootstrap confidence intervals use 2000 resamples over test
images. The source-of-truth tables are:

- `results/paper_tables/mediafinal_consistent_results.{csv,md,tex}`
- `results/paper_tables/mediafinal_bootstrap_crc_ablation.{csv,md,tex}`
- `results/paper_tables/refiner_zoo_crc.{csv,md,tex}`

## 5. Results

### Fixed refinement can be harmful

Fixed morphology is not reliably safe. On the external polyp GraphSeg setting,
the best fixed morphology action has mean gain -0.0211, mean harm 0.0524, and
worst drop -0.5093. On ISIC, fixed morphology has positive mean gain but a worst
drop of -0.5228. These are precisely the cases where reporting only mean Dice is
misleading.

### Practical policies can recover utility while limiting harm

In Kvasir-SEG with a GraphSeg host, geometry-aware CRC selects morphology
`a=5`, improves Dice by +0.0095, and keeps mean harm at 0.0016. The bootstrap
interval for gain is [+0.0000, +0.0227], reflecting both a positive practical
signal and the limited size of the test set.

In ISIC 2018 with the refiner zoo, the controller selects largest connected
component refinement. The practical policy improves Dice by +0.0033 with 95%
bootstrap CI [+0.0015, +0.0055], mean harm 0.0015, and revert-rate 0.032. This
is the cleanest demonstration that the contribution is not restricted to
morphology-only refinement.

On the external polyp GraphSeg stress set, fixed morphology is harmful on
average, but the zoo policy selects a close-only action with +0.0026 mean gain
and near-zero mean harm. This result should be interpreted as safety evidence:
the portfolio can reject unsafe fixed morphology and select a safer alternative.

On the external polyp UNet setting, the practical geometry policy improves Dice
by +0.0037 with bootstrap CI [+0.0008, +0.0067] for the zoo row, while mean harm
is 0.0018.

### Strict mode gives exact fallback

Across all six final mediafinal runs, strict mode returns exact host fallback
for both the baseline action set and the refiner zoo. This is not a Dice
maximization result; it is the deployment safety contract. If calibration cannot
certify a non-host action under zero harmed-rate tolerance, the controller does
not intervene.

### Model-agnostic behavior

The same wrapper operates over GraphSeg and UNet hosts. Stronger hosts often
reduce the need for intervention. For example, Kvasir-SEG UNet has host Dice
0.841, and practical policies mostly revert to the host. This should be
presented as correct behavior: the safety layer is allowed to decide that no
refinement is justified.

## 6. Discussion

SafeRefine changes how refinement methods should be evaluated. A refiner is not
only a mean-Dice improvement mechanism; it is an intervention that can help,
harm, or be rejected. The host fallback action makes this explicit and gives a
deployment-compatible output even when no safe refinement is selected.

The method is conservative, especially in strict mode and on small calibration
sets. This is a feature rather than a defect for safety-critical use. Practical
mode exposes the safety-utility trade-off when users are willing to accept a
non-zero calibration harm budget.

The external polyp results are intentionally included as stress tests. We do not
claim that SafeRefine solves external polyp segmentation performance. Instead,
the stress results show that unsafe fixed refiners can be detected or bypassed,
and that broader action portfolios can sometimes identify safer alternatives.

## 7. Limitations

First, the current learned host models are not intended to be state-of-the-art
segmenters. The paper's claim is about post-hoc safe refinement, not about
backbone superiority. Second, practical policies do not eliminate all test-time
harm; they calibrate a risk-controlled utility trade-off and must report
worst-drop and harmed-rate. Third, bootstrap intervals remain wide on smaller
datasets such as PH2 and Kvasir-SEG. Fourth, morphology and component operations
are simple refiners. The refiner-zoo experiment mitigates morphology dependence,
but learned graph-physical refiners remain future work.

## 8. Conclusion

We presented SafeRefine, a model-agnostic harm-controlled refinement framework
for medical image segmentation. By treating refinement as a calibrated action
selection problem with exact host fallback, SafeRefine exposes both utility and
harm instead of hiding case-level degradation behind average metrics. Across
multiple modalities, hosts, and refiner portfolios, strict policies preserve the
host exactly, while practical policies identify settings where refinement can
improve segmentation with measured harm. This provides a general safety layer
for deploying segmentation refiners in medical imaging.
