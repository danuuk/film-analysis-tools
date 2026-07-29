# Method, experiment and campaign catalogue

Status: first-pass inventory
Created: 2026-07-28
Source: `obs-art-plugin` @ `897aad09`

**Scope and honesty note.** This is a *mechanical* inventory — what exists, measured from the
code. What each experiment concluded and why it was created comes from the P1 archaeology pass
(`MIGRATION_PLAN.md` §11); where this document states a purpose, it is inferred from module and
document names unless marked otherwise. Counts are modules mentioning a technique, which is a
proxy for use, not a measure of importance.

---

## Part 1 — Statistical methods in use

### Heavily used — these are the working vocabulary

| Method | Modules | Used for |
|---|---:|---|
| quantiles / conditional quantiles | 57 | the primary comparison currency; `conditional_quantiles.csv` is the most common large artifact |
| robust scale (MAD, median-abs) | 37 | outlier-resistant spread; robust norms in `core/measure` |
| interpolation | 29 | curve resampling, LUT lookup, response surfaces |
| holdout / cross-validation | 28 | fit generalisation — already widespread, good instinct |
| bootstrap | 16 | uncertainty on fitted quantities |
| radial PSD / periodogram | 15 | grain and noise spectra (`core/noise`, `core/measure`) |
| monotonicity | 14 | tone-curve well-formedness |

### Moderately used

| Method | Modules | Used for |
|---|---:|---|
| numerical Jacobian | 11 | **parameter identifiability of fits**, not signal-space behaviour — see Part 4 |
| histograms | 11 | distribution comparison |
| clustering / medoids | 11 | representative sample selection, preview medoids |
| optimisers (`minimize`, `curve_fit`) | 10 | curve and parameter fitting |
| correlation (Pearson/Spearman) | 8 | coupling and dependence checks |
| regression (`lstsq`, `polyfit`) | 7 | linear response fits |

### Sparse — used once or twice, worth consolidating

| Method | Modules | Note |
|---|---:|---|
| Jensen–Shannon divergence | 5 | cohort/scene coverage scoring (`screening_cohort` uses JS + L1 greedy) |
| autocorrelation | 5 | temporal and spatial grain structure |
| Wasserstein / earth-mover | 4 | distribution distance |
| SVD / eigendecomposition | 4 | Jacobian conditioning, basis screening |
| permutation tests | 3 | **the null-control primitive — underused, should be default (plan §2.3)** |
| Mann–Whitney / Wilcoxon / t-test | 3 | nonparametric group comparison |
| differential evolution | 2 | global search in `reference_engaged_calibration_fit` (scipy pinned exactly for reproducibility) |

### Observations

- **Quantile-based comparison is the house style.** Any new framework should make conditional
  quantiles a first-class, one-line operation.
- **Cross-validation and holdout are already widespread** (28 modules) — the rigour instinct is
  present and only needs systematising.
- **Permutation/null controls are the weakest link at 3 modules.** This is the highest-value,
  lowest-cost gap, and plan §2.3 makes it the default output.
- **No formal multiple-comparison handling anywhere.** With cohorts sliced many ways (skin,
  foliage, neutral, …) this matters for honesty, though at tier 1 a note may suffice.

---

## Part 2 — Experiments and campaigns

### The statistical colour arc (the spine)

Sequential, each building on the last. Documented in ~55 `_spec` and 37 `_result` documents.

| Stage | Modules | Subject |
|---|---|---|
| STAT0/1 | `reference_colour_sample_pack` | stratified reservoir sampling into the sample table |
| STAT2 | `reference_statistical_screening_cohort`, `reference_statistical_colour_report` | scene coverage (JS + L1 greedy), gamut-safe tone screening |
| STAT2A | `reference_vector_tile_stat2a` (23k LOC), `reference_statistical_objective_audit`, `_residual_localization`, `_frame_replay`, `_control_capacity` | residual localisation, visibility, native-control capacity — stages VT0…VT8 |
| STAT3 | `reference_synthetic_colour_response_atlas` | synthetic point-renderer response atlas |
| Behaviour | `reference_behaviour_vector`, `_replay_queue`, `_visual_replay`, `_threshold_preflight` | per-axis behaviour vectors (the 2026-07-18 re-point) |

### Campaign arcs (ARCHIVE-bound, 58 modules / ~60k LOC)

| Arc | Modules | Subject | Documented |
|---|---:|---|---|
| `campaign.kodak` | 19 | 5219/2383 sensitometry, negative→print attribution, neutral construction, B1 toe/identity, Status-M camera pilot | 18/19 |
| `campaign.ptf` | 10 | practical target fit PTF1–PTF5: populations, attribution, visibility, sign capacity, absolute luma gate, basis screen | 7/10 |
| `campaign.b0-fsc1c` | 9 | B0-prime tone foundation, FSC1-C identity/integration, chroma gain capability and stress | 6/9 |
| `campaign.joint` | 8 | joint colour: basis screen, fit, robust selection, equivalence audit, holdout confirmation, candidate frontier, visual limit, native equivalence | **0/8 — do first** |
| `campaign.broad` | 5 | broad colour scan / coverage / review / freeze, global comparison | 1/5 |
| `campaign.vector-tile` | 4 | vector-tile pilot, response atlas, followup, curve analysis | 4/4 |
| `campaign.al5` | 1 | AL-5 synthetic optical calibration | 0/1 |
| misc | 2 | control probe, local colour experiment | 1/2 |

### Supporting programmes

- **Face/skin selection (SC1–SC4)** — 15 modules: MediaPipe FaceMesh provider, semantic
  probability fusion, continuous suitability, temporal grouping, attrition accounting,
  validation against frozen pixel labels, provider benchmarking.
- **Grain characterisation (MA-*)** — patch detection, temporal isolation, zone analysis,
  sparkle analysis, PSD refitting, slow-heterogeneity contract.
- **Optical layers (AL-2…AL-5)** — density→L100 placement, scanner/stock MTF, guarded halation,
  artistic optical layer.
- **Scene catalogue** — 11 modules; scene detection, 8 scored categories, face scouting,
  compare reports. See plan §5.4.

---

## Part 3 — Reusable capabilities already built

| Capability | Where | State |
|---|---|---|
| stratified sampling → tidy table | `reference_colour_sample_pack` | works; 54% ceremony to strip |
| colour feature extraction | `reference_colour_features` | **clean (20%)** — hue/sat/luma/opponent/chroma/headroom + cell binning |
| frame population selection | `reference_frame_population` | **clean (8%)**, in-degree 70 |
| scene catalogue + scoring | `scene_catalog/` | 8 categories, thresholds, face scouting |
| face/skin detection | `skin_selection/` | provider contract, FaceMesh, attrition, suitability |
| PSD / noise synthesis | `core/noise`, `core/measure` | clean, engine-free |
| patch grids, accumulators | `core/measure` | clean; covariance, residual bins, radial PSD |
| balanced/coverage selection | `core/selection` | clean |
| ffmpeg decode + extraction | `core/decode`, `core/sourcing` | works; `argparse.Namespace` coupling to remove |
| evidence hashing / identities | `reference_calibration_contract_v1` | over-built; keep ~6 of ~40 schemas |
| manifest pipeline runner | `reference_pipeline` | works; argv-driven, needs in-process port |

---

## Part 4 — Gaps: what Contour B needs and does not exist

Contour A ("how close to the reference") is well covered. Contour B ("is the transform itself
well-formed") is largely absent. Measured coverage:

| Contour B property | Existing coverage |
|---|---|
| tone-curve monotonicity | **partial** — FEE `LumaCurveParams._check_monotone` validates at authoring; nothing checks the composed chain |
| signal-space Jacobian (∂out/∂in) | **none** — the 11 Jacobian modules compute ∂residual/∂params (fit identifiability), a different question |
| Jacobian determinant sign (folding, hue inversion) | **none** |
| Jacobian SVD / conditioning (channel collapse, anisotropic squash) | **none in signal space** — SVD used for fit rank only |
| gradient smoothness, banding, posterisation | **near-none** — 3 modules mention smoothness, 3 banding, 0 posterisation |
| highlight collapse / shoulder behaviour | **indirect only** — via reference comparison, not as an intrinsic property |
| 3D LUT per-cell interpolation validity | **none** — 1 module touches LUTs at all (`reference_residual_lut`, 106 LOC) |
| gamut boundary behaviour | **partial** — `gamut` in 30 modules, but as a constraint during fitting, not as a property test |
| clipping/clamping audit | **scattered** — `clip|clamp` in 110 modules, no systematic accounting |

**The pattern:** where a Contour B concern exists at all, it exists as a *constraint applied
during fitting* rather than a *property tested on the transform*. That is why a badly-formed
transform can pass every existing check — the checks assume well-formedness rather than
verifying it.

### Proposed additions

Synthetic-input probes, no corpus required, cheap enough to run per-change:

1. **Ramp and gradient suite** — gentle luminance ramps, near-neutral gradients, hue sweeps at
   several saturations; measure output smoothness (second differences), banding, monotonicity of
   the composed chain.
2. **Signal-space Jacobian field** — numerical ∂out/∂in on a grid over the working cube.
   Report `det(J)` sign (folding), `cond(J)` from SVD (channel collapse), and per-axis gain.
3. **Highlight and shoulder probe** — behaviour approaching and beyond diffuse white; detect
   collapse, clipping, and hue shift under exposure sweep.
4. **Gamut boundary sweep** — cube-surface traversal; where does the transform leave the target
   gamut, and how is it brought back.
5. **3D LUT cell validity** — for any baked LUT: per-cell monotonicity and non-inversion,
   interpolation error against the analytic transform, worst-cell reporting.
6. **Clipping accounting** — how much of a representative input distribution clips, and where.

Each yields a scalar or small map plus a pass/fail. Each runs on synthetic input in seconds.
