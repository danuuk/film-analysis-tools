# campaign.joint — the JCC joint-colour arc

**Disposition:** archive · **Modules:** 8 · **LOC:** 6,498 · **Legacy pin:** `obs-art-plugin@897aad09`
**Campaign:** `pulp_fiction_statistical_colour_calibration_v1`
**Documentation at time of archiving:** none — 0 of 8 modules appeared in any of the 183
legacy documents, which is why this arc was recorded first.

Status lines below are **quoted verbatim** from each stage's `summary.json` under
`findings/reference_statistical_colour_calibration/pulp_fiction_statistical_colour_calibration_v1/joint_colour/`.
Descriptions of intent are **inferred from the code**, since no specification survives.

---

## 1. What it was asking

Whether the three existing plugin control families, fitted **jointly** rather than one at a
time, could move the source population closer to the reference film — and whether the result
would survive translation to the native runtime.

The controls being fitted, from `CONTROL_DEFINITIONS`:

| Family | Owner | Controls |
|---|---|---|
| `tone_print_response` | profile | negative develop print density, master/red/green/blue printer lights, print contrast |
| `hue_colour_timing` | grade | print density, warm–cool, green–magenta, print contrast, plus shadow and highlight warm–cool and green–magenta |
| `chroma_response` | chroma gain | per-hue stops (R, Y, G, C, B, M) and per-luma-band stops |

Starting point was the `engaged_35mm_unified_v1` base profile with the
`pulp_fiction_chroma_gain_iterative_proxy_preview_v1` chroma seed.

## 2. How it tried to answer

Eight sequential stages, each a pipeline command with its own frozen implementation version:

| Stage | Module | Role |
|---|---|---|
| JCC1 | `basis_screen` | choose the candidate basis to fit within |
| JCC2 | `fit` | fit the joint control vector |
| JCC2R | `robust_selection` | refit under scene folds (`jcc2r_scene_fold_v1`) |
| — | `equivalence_audit` | scene-population interval against a frozen ±1% practical-equivalence band, via `delete_one_scene_jackknife_bayesian_pseudovalue_bootstrap_v1` |
| — | `visual_limit` | visual review of the bound |
| — | `candidate_frontier` | assemble the reviewable candidate frontier |
| JCC3 | `holdout_confirmation` | confirm the selected candidate on held-out scenes |
| JCC4 | `native_equivalence` | translate to the native runtime |

Objective components: sliced Wasserstein distance, a normalized conditional residual,
support-mismatch L1, and bounded non-negative chroma headroom.

## 3. What it concluded

| Stage | Status (verbatim) |
|---|---|
| JCC1 basis screen | `joint_colour_basis_selected` — candidate **C0**, "grounded neutral plus Pulp Chroma Gain seed" |
| JCC2 fit | `joint_colour_fit_candidate_blocked` — "failed development validation (-6.10% relative improvement) and must not proceed unchanged" |
| JCC2R robust selection | `joint_colour_robust_candidate_blocked` — "failed reused development validation (-0.57%)" |
| equivalence audit | `joint_colour_equivalence_audit_complete` — "The complete 90% scene-population interval lies inside the frozen +/-1% practical-equivalence band. The candidate may proceed only to a JCC3 diagnostic." |
| visual limit | `joint_colour_visual_limit_ready_for_review` (not decision-bearing) |
| candidate frontier | `joint_colour_frontier_conditional_review_available` |
| JCC3 holdout | `jcc3_selected_candidate_confirmed` — `timing_core_bounded_lbfgsb@0.25` |
| JCC4 native | **`jcc4_native_translation_blocked`** |

**The arc ended blocked.** The fit was blocked, the fold-robust refit was blocked, the
equivalence audit found the candidate practically indistinguishable from baseline and
permitted it only as a diagnostic, and the native translation was blocked.

### The JCC3 "confirmed" result deserves reading closely

The single stage reporting success did so on mixed evidence. From its own summary, on 80,000
held-out samples:

| Component | Relative improvement |
|---|---|
| sliced Wasserstein | **+1.67%** |
| conditional residual | **−0.59%** (regression) |
| bounded chroma headroom | **−2.22%** (regression) |

One metric improved by under two percent; two others got worse. The stage is nonetheless
recorded as `jcc3_selected_candidate_confirmed`, because confirmation was gated on the
sliced-Wasserstein component alone.

## 4. What is worth carrying forward

**The caution is the main finding.** This arc is the clearest instance in the legacy corpus of
the pattern the new system is built against: eight stages, scene folds, a jackknife-bootstrap
equivalence band, a holdout gate and a native-equivalence check — and the honest summary is
that the candidate never worked well enough to ship. The apparatus was elaborate enough to
*look* conclusive while a −6.10% development result, a −0.59% refit and two held-out
regressions sat inside it.

Concretely, for the rebuilt system:

- **A gate on one component of a multi-component objective is not confirmation.** JCC3 passed
  on sliced Wasserstein while regressing on conditional residual and chroma headroom. Report
  every component, and do not let one of them carry a verdict.
- **"Inside the practical-equivalence band" means the candidate did nothing.** The equivalence
  audit said so plainly and was right; the arc continued anyway. An equivalence result is a
  stop condition, not a permission.
- **Blocked stages should terminate an arc, not be routed around.** JCC2 and JCC2R both
  blocked, and four further stages ran regardless.
- The **control taxonomy is reusable** — three families with explicit owners (profile, grade,
  chroma gain), bounds and enabling gates. That structure migrates with `capabilities/fit`
  even though this campaign's results do not.
- **Joint fitting across all three families was never demonstrated to beat the baseline.** Any
  future attempt starts from that, not from an assumption that the approach is sound and the
  execution was at fault.

## 5. Where the evidence lives

Legacy artifacts (regenerable; retained under the retention policy in `MIGRATION_PLAN.md` §5.3):

```
findings/reference_statistical_colour_calibration/
  pulp_fiction_statistical_colour_calibration_v1/joint_colour/
    basis_screen_v1/ fit_v1/ robust_selection_v1/ equivalence_audit_v1/
    visual_limit_v1/ candidate_frontier_v1/ holdout_confirmation_v1/
    native_equivalence_v1/
```

Modules, all at the legacy pin under `mediachar/analyzers/film_grain/`:
`reference_joint_colour_basis_screen.py`, `_candidate_frontier.py`, `_equivalence_audit.py`,
`_fit.py`, `_holdout_confirmation.py`, `_native_equivalence.py`, `_robust_selection.py`,
`_visual_limit.py`.
