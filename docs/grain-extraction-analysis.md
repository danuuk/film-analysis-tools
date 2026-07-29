# Legacy grain extraction: analysis, and a plan for the mature replacement

Created: 2026-07-30 · Analysed at `obs-art-plugin@897aad09`

## 1. What exists

~17k LOC across the grain path, plus 1.2k of shared measurement primitives:

| Module | LOC | Role |
|---|---:|---|
| `grain_compare` | 3,776 | real-vs-model comparison harness |
| `debug_real_footage` | 3,514 | diagnostic runner over real scenes |
| `analyze_patches` | 2,315 | **patch detection and residual extraction — the core** |
| `grain_properties` | 1,653 | property confirmation on static regions |
| `export_grain_preset` | 1,220 | fit and export of the runtime preset |
| `bake_grain_tiles` | 1,005 | deterministic noise tiles from a spectrum |
| `slow_heterogeneity_contract` | 763 | micro-uniformity evidence |
| `sparkle_analysis` / `_recalc` | 742 / 405 | sparkle statistics |
| `zone_analysis` | 735 | zone-wise temporal-isolated measurement |
| `build_scene_profile` | 376 | per-scene profile assembly |
| `refit_temporal_spectrum` | 364 | base spectrum refit from source PSDs |
| `core/measure` | 791 | accumulators: covariance, residual bins, radial PSD, autocorrelation |
| `core/noise` | 212 | shaped-noise synthesis |

## 2. How it actually works

Frames are decoded to 16-bit, a patch grid is laid down (256 px, stride 256), and per-patch
statistics accumulate over the sequence. The residual is formed as a **plain pixel-wise temporal
difference**, then split by spatial frequency:

```python
delta    = luma - prev_luma                      # no registration
delta_lp = box_blur_2d(delta, motion_blur_radius) # "motion": survives blurring
delta_hp = delta - delta_lp                       # "grain": spatially white
motion_energy  = sqrt(mean(delta_lp²))            # gate: reject patch if > 0.005
grain_hp_std   = sqrt(mean(delta_hp²) / 2.0)      # per-frame amplitude
temporal_grain_std = temporal_luma_rmse / sqrt(2)
```

Surviving patches are binned by luma (16 bins), reduced to 6 amplitude knots, and exported with a
radial PSD spectrum, an RGB-correlation block, and a confidence block.

## 3. What it gets right

This deserves saying first, because the design is more self-aware than its size suggests.

- **The motion/grain separator is a genuinely good idea.** Splitting the temporal difference into
  low-pass and high-pass parts, and gating on the low-pass energy, exploits a real asymmetry:
  translational motion is spatially structured and survives blurring, grain is white and does not.
  The reasoning is written down in a comment at the point of use.
- **Gate discipline is documented in the tool itself.** The `--max-motion-energy` help text states
  that a genuinely static scene sits near 0.001–0.003, that raising the gate mines the least-moving
  background "at higher contamination risk", and that if nothing passes, *the scene is not a
  grain-amplitude source*. The default (0.005) is tight, and the shipped preset is named
  `..._motion_gated`.
- **The colour limitation is handled honestly and repeatedly.** Not one caveat but a family:
  "4:2:0/4:2:2 or source-y: chroma unmeasurable (D-022)", "Do not treat upsampled delivery RGB
  residual covariance as film-grain chroma correlation", "would duplicate the luma residual into
  RGB and is not a chroma-grain model", and `rgb_correlation` is flagged null when any source
  profile is luma-only. Subsampling is detected from `pix_fmt` rather than assumed.
- **The fit reduces rather than inflates.** 16 measured luma bins become 6 exported knots. The
  runtime interpolates in log-luma and **clamps** outside the measured range instead of
  extrapolating a trend.
- **A confidence block ships with the preset**: source profile count, clean-patch point count,
  measured luma coverage, and free-text warnings including "provisional preset for pipeline
  testing, not a final physical film-stock model".
- **Temporal autocorrelation is measured** at lags 1–5, and patches are rejected when lag-1
  correlation exceeds 0.08.

## 4. Weaknesses, in severity order

### 4.1 ρ = 0 is baked into the estimator, although ρ is measured

`sqrt(var(Δ) / 2)` and `std(Δ)/sqrt(2)` recover the per-frame standard deviation **only if
consecutive frames are uncorrelated**. In general

$$\operatorname{Var}(X_t - X_{t-1}) = 2\sigma^2(1-\rho_1)$$

so the estimator understates σ by a factor of $\sqrt{1-\rho_1}$. The pipeline *does* measure ρ at
lags 1–5 in `grain_properties`, and *does* gate patches at ρ₁ ≤ 0.08 — which bounds the error at
roughly 4%. But the measured ρ is never substituted into the estimate, and the residual bias is
never propagated into the exported confidence. The information is collected and then disconnected
from the number it should correct.

Worse for a delivery master: inter-frame codec filtering can make ρ₁ **negative**, which inflates
σ rather than deflating it, and the gate is one-sided.

### 4.2 No registration, so fine texture is counted as grain

`delta = luma - prev_luma` is a raw difference. The low-pass/high-pass split separates motion by
*spatial frequency*, not by *alignment* — so sub-pixel translation, film weave, and gate jitter
move **fine detail**, and that difference lands in `delta_hp` and is scored as grain. Blur-then-
subtract cannot distinguish moving fine texture from static grain, because both are white in the
difference.

The system detected this and did not correct it. The shipped preset carries the warning:

> `scene_003_…: static textured residual std is 7.15x the static grain-dominated`

A **7×** discrepancy between textured and grain-dominated static patches is not a subtlety; it is
the dominant error term on any patch with detail in it. The legacy response was to prefer
grain-dominated patches and warn about the rest. Aligning on a low-pass image before differencing
attacks the cause instead.

### 4.3 The amplitude curve rests on 24 points covering a fifth of the range

The shipped preset's own numbers: `source_point_count: 24`, `amplitude_luma_coverage: 0.00012 to
0.282`. Six knots, the brightest at luma 0.222.

So **there is no grain measurement above roughly 28% luma at all.** Everything brighter — most of
a normally exposed image — receives the amplitude of the brightest measured knot, held flat. The
clamp is the right choice given no evidence, and the coverage *is* recorded, but the preset does
not say in the runtime-facing part that four fifths of its domain is unmeasured. It reads like a
curve; it is a curve over the shadows plus a constant.

This is a direct consequence of §4.2 and the motion gate interacting: static, untextured,
grain-dominated patches in those scenes were dark ones.

### 4.4 No holdout reconstruction

Verified absent across `grain_compare`, `grain_properties`, `export_grain_preset` and
`build_scene_profile`. `grain_compare` compares real against model, but on the same material the
preset was fitted from. The colour arc has holdout machinery in 10+ modules; the grain path has
none. So there is no evidence that a fitted grain profile generalises to a scene it never saw.

### 4.5 Per-knot coverage is not exported

The preset's knots carry `knot_index`, `luma`, `amplitude_luma_std` — and nothing else. Sample
counts exist at aggregate level in `confidence` and `coverage_summary`, but a consumer cannot tell
which knot rests on thousands of pixels and which on a handful. With 24 points over 6 knots, that
distinction matters: the mean is 4 points per knot.

### 4.6 Evidence types are entangled at the module level

The five kinds of evidence the replacement should separate — amplitude-versus-level, 2D
NPS/autocorrelation, distribution and tails, slow spatial heterogeneity, temporal behaviour — are
each measured somewhere, but spread across `analyze_patches`, `grain_properties`,
`slow_heterogeneity_contract`, `zone_analysis` and `refit_temporal_spectrum`, with overlapping
patch selection and no shared source record. Re-running one of them consistently with the others
requires knowing which flags they were originally given.

### 4.7 The source record is incomplete for repeatability

Scene manifests record source path, start, duration, frame count, pixel format and a stream probe
— but **no content hash of the source**, and the recorded absolute path has already gone stale
once. Crop is recorded as a separate derivative directory rather than as a field, and cadence is
implicit in the frame rate. A measurement cannot be reproduced from the record alone.

## 5. Plan for the mature replacement

Target: `film-analysis-tools`, following the eight-step workflow. Only the compiled numeric
profile crosses into the engine.

### Step 1 — Source record (`capabilities/source`)

A `SourceRecord` carrying edition, timestamp, crop rectangle, cadence as a rational, decode
contract, active-picture dimensions, and **source content hash**. This reuses what the camera
catalogue already established: hash identity with the path as a hint. Raw footage stays outside
both repositories. Fixes §4.7.

### Step 2 — Window selection (`capabilities/sample`)

Windows across shadow / midtone / highlight / texture / frame position, each carrying its own
gate measurements. Two rules the legacy tool advised but did not enforce:

- **A rejected window stays rejected.** The gate is not a tunable to be relaxed for yield; if a
  scene yields nothing, it is not a grain source. Enforced in code, not in help text.
- **Coverage is reported as a first-class result**, so a run that only found shadow windows says
  so loudly rather than silently exporting a shadow-only curve (§4.3).

### Step 3 — Residual extraction (`capabilities/measure`)

- **Align adjacent frames on a low-pass image before differencing.** This is the fix for §4.2 and
  the single biggest accuracy change. Estimate translation on a blurred copy, apply it, then
  difference.
- **Measure lags 1, 2 and 4** and estimate ρ, then use it: σ² = Var(Δ₁) / (2(1−ρ₁)), with lag 2
  and 4 as a consistency check. Report ρ per window. Fixes §4.1.
- Keep the low-pass/high-pass separator — it is sound — but as a *second* line of defence after
  alignment, not the only one.
- Validate on synthetic grain with **known** σ and **known** ρ. That is the null control this
  machinery has never had: inject grain at a stated amplitude and correlation, and confirm the
  extractor recovers both.

### Step 4 — Separate evidence types

Five independent producers, one shared source record and window set, each exportable alone:
amplitude-versus-level; 2D NPS and autocorrelation; distribution and tails; slow spatial
heterogeneity; temporal behaviour. Fixes §4.6.

### Step 5 — Colour limitation, enforced rather than annotated

The legacy caveats were right; the replacement makes them structural. A profile fitted from
subsampled material is *typed* as luma-appearance and **cannot** populate a colour-covariance
field — the field does not exist on that type. Colour grain requires 4:4:4 scans, laboratory
samples, or material granularity data, declared as such.

### Step 6 — Compact fit with honest coverage

A modest 1D amplitude curve and a compact 2D structure model, as now. Additionally:

- **Per-knot sample count and confidence interval** travel with each knot (§4.5).
- **The measured range is part of the runtime contract**, not only the confidence block, so a
  consumer must handle the unmeasured region deliberately.
- Refuse to emit a knot with fewer than a stated minimum of samples, rather than emitting a
  weakly-supported number that looks like the others.

### Step 7 — Holdout reconstruction

Real versus reconstructed grain on **scenes not used in the fit**, through the same measurement
lens and the same motion presentation. This is new (§4.4) and is what turns a fit into a claim.
Tier `study` in the ladder; below that a grain profile may not claim to generalise.

### Step 8 — Export one numeric profile

A single numeric profile into a generated catalogue. No source frames, no report screenshots into
the engine. The analytical report (curves, distributions, samples) stays here as HTML; only
numbers cross the boundary.

## 6. Sequencing

Steps 1–3 are the foundation and where the accuracy is won; 4–6 are mostly restructuring of
methods that already exist; 7 is new and is the gate that makes the rest trustworthy.

The order I would take: **3 first, on synthetic grain with known σ and ρ** — because if the
extractor cannot recover a known answer, nothing downstream of it is worth building. Then 1 and 2
to make it repeatable, then 4–6, then 7.

That is the same discipline the fast path used: prove the mechanism on a known input before
building the tool that depends on it.
