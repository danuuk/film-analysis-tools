# Spektrafilm negative-grain comparison v1

Status: completed bounded comparison; N2 design input, not an implementation source

Tier: comparison
Compact result: [`results/spektrafilm_negative_grain_comparison_v1.json`](results/spektrafilm_negative_grain_comparison_v1.json)

## Answer

The corrected conclusion holds: excluding Spektrafilm from the design discussion was too strong,
but copying its grain model would still be wrong. Its useful contribution is the *shape of the
question*: density-dependent particle variance, explicit sublayers, independent layer fields, and
measurement after print. Its field generator, temporal contract, layer reconstruction, default
amplitude, and blur are not suitable N2 inputs.

The reviewed source is
[`andreavolpato/spektrafilm@3bb2c2d`](https://github.com/andreavolpato/spektrafilm/commit/3bb2c2d2801ff68b92019cf1dbcbb133d60832bc),
committed 2026-06-07. The checkout was clean. The compact JSON pins the source files and the
Vision3 500T / Kodak 2383 profile hashes used by the oracle.

## Exact processing order

The reviewed runtime executes:

```text
input exposure
  -> normalized negative characteristic curves
  -> DIR-coupler exposure correction and redevelopment
  -> grain in negative CMY density
  -> spectral negative transmittance / enlarger exposure
  -> Kodak 2383 print development
  -> spectral scan
```

This is explicit in Spektrafilm's `develop()`: initial density formation is followed by DIR
couplers, then `apply_grain()`. Grain therefore does not feed back into coupler formation. The
pipeline then sends the grained CMY negative to print exposure, print development, and scan.

The layered branch interpolates each total dye-channel density into three sublayers. It samples
nine Poisson/binomial particle fields (three sublayers by three dye channels), with distinct fixed
seeds, sums sublayers by dye channel, applies per-channel multiplicative microstructure, and then a
final Gaussian blur. There is no shared scalar RGB field.

## Oracle method

A 256×256 synthetic linear-ProPhoto chart contained eight neutral levels from −5 to +2 stops
around 18%, six primary/secondary colour patches, and one warm skin proxy. The model used Kodak
Vision3 500T and Kodak 2383 at 21.420 µm/pixel, equivalent to a 35 mm frame sampled 1634 pixels
high. Camera halation, diffusion, print glare, and scanner sharpening were disabled so the measured
difference was negative formation, DIR couplers, grain, 2383, and scan.

The null was the same chain with grain disabled. It produced zero repeated difference. Canonical
Spektrafilm grain is bit-exact across repeated calls because every channel/sublayer resets NumPy's
global RNG to `channel + 10*sublayer`. The alternate `fixed_seed` branch is broken at the reviewed
pin: it sets the seed vector to `None` and then indexes it. Consequently the spatial comparison
uses 16 deterministic phase-resamples of the canonical fixed field. They are not represented as a
temporal ensemble.

The full ignored report contains every patch covariance matrix, drift vector, footprint, source
hash, and execution condition. The committed result reports effect ranges and spread.

## Results against N1/A1

### Density-dependent variance

The negative-density common-mode RMS stayed fairly flat across the neutral wedge
(0.00471–0.00660), while propagation through 2383 and scan made final luma RMS strongly
level-dependent:

| Input stop | Final linear Y | Spektrafilm luma RMS | Spektrafilm / A1 target |
|---:|---:|---:|---:|
| −5 | 0.000468 | 0.0000137 | 0.061 |
| −4 | 0.000931 | 0.0000442 | 0.119 |
| −3 | 0.002989 | 0.000192 | 0.220 |
| −2 | 0.013931 | 0.000934 | 0.346 |
| −1 | 0.060091 | 0.003051 | 0.388 |
| 0 | 0.173418 | 0.006015 | 0.352 |
| +1 | 0.349600 | 0.007632 | 0.440 |
| +2 | 0.544394 | 0.006130 | 0.353 |

This resembles N1's failure direction: negative-stage material modulation plus print propagation
does not reproduce A1's shadow envelope. Spektrafilm is not amplitude-calibrated to our chain, so
the ratios are structural evidence, not a fit or verdict on its realism.

### Shared versus independent fields and chromatic grain

Spektrafilm's neutral negative-density off-diagonal correlations span −0.0135 to +0.0222, with a
median of +0.0040: effectively independent dye-channel fields. Its negative opponent/common RMS
ratio is 2.23–2.33. That lands directly in N1a's predicted fully-independent range
(2.12–2.92), not its shared-field range (0.21–0.99).

Kodak 2383 changes that covariance. At neutral midgray, density correlations near zero become
approximately `(0.105, −0.001, 0.172)` in print density and `(0.319, −0.004, 0.379)` in final
linear RGB. Neutral final opponent/luma remains 1.44–1.83; colour patches span 1.05–2.93.
Independent layer fields therefore create real chromatic structure, while print mixes rather than
erases it. N2 needs an explicit shared-to-independent covariance control; fully independent is a
diagnostic bound, not the default conclusion.

### Spatial scale and sublayers

At the 1634-pixel reference height, Spektrafilm's final neutral footprint is nearly isotropic:

| Metric | Minimum | Median | Maximum | Accepted A1 at 1634 px |
|---|---:|---:|---:|---:|
| horizontal half-width | 0.941 | 0.981 | 1.007 | 1.216 |
| vertical half-width | 0.964 | 0.990 | 1.043 | 0.848 |
| anisotropy | 0.950 | 0.992 | 1.016 | 1.435 |

The midgray radius changes only from `(1.013, 0.998)` at negative density to `(0.981, 0.989)`
after 2383 and scan. Print does not justify replacing the accepted A1 footprint. Sublayers do
matter to amplitude: the layered midgray common RMS is 0.819× the composite model. That supports
explicit layer modulation, not Spektrafilm's Gaussian footprint.

### Mean-density drift

The particle law is mean-preserving in expectation, but the layered path reconstructs total
density by summing interpolated sublayer curves before returning grain. That reconstruction differs
from the input by as much as 0.004154 density. At neutral midgray its mean error is
`(−0.003385, −0.00000869, −0.001357)`, which explains almost all of the measured negative-density
drift. This is deterministic, precedes 2383, and must not enter N2.

N2 should always add a zero-mean perturbation to the original analytical density. Sublayer values
may modulate variance, but must never replace or reconstruct the mean signal. Nonlinear mean drift
created later by 2383 is measured and reported; it is not silently inverse-corrected.

## N2 equations

These equations are independently defined. Spektrafilm supplies comparison evidence only.

Let `D_c(x)` be the existing Kodak 5219 analytical density and `sigma_5219,c(D_c)` the N1
engineering-relative response. For three authored sublayers, define smooth nonnegative activation
scores from independently chosen thresholds and widths:

```text
r_lc(D) = softplus((D - t_lc) / tau_lc)
a_lc(D) = r_lc(D) / sum_j r_jc(D)
v_lc(D) = max(epsilon, a_lc(D) * (1 - u_lc * a_lc(D)) / N_lc)
w_lc(D) = sqrt(v_lc(D) / sum_j v_jc(D))
```

`t`, `tau`, `u`, and `N` are our inferred engineering parameters. No Spektrafilm defaults are
copied. The normalization guarantees `sum_l w_lc² = 1`, so layer modulation changes the mixture,
not the already-defined 5219 channel envelope.

Generate every field with the accepted A1 FIR, reference-height scale policy, PCG64
seed-plus-film-frame contract, and independent-frame cadence:

```text
F_0  = unit_variance(A1_FIR * epsilon_0)
F_lc = unit_variance(A1_FIR * epsilon_lc)
G_c  = sum_l w_lc(D_c) * F_lc
Phi_c(rho) = sqrt(rho) * F_0 + sqrt(1 - rho) * G_c
DeltaD_c = strength * sigma_5219,c(D_c) * Phi_c(rho)
D'_c = D_c + DeltaD_c
```

Locally, `Var(Phi_c)=1`; between dye channels, the common component gives covariance `rho` before
the spectral print transform. Because `D'_c` is formed by addition to `D_c`,
`E[D'_c | D_c] = D_c`. Layer activation is evaluated from ungrained density, so there is no
grain-to-amplitude feedback.

The bounded N2 comparison should use `rho = 1.0` as the current fully-shared control,
`rho = 0.75` and `0.5` as partial-correlation candidates, and `rho = 0.0` only as the
Spektrafilm-like independent diagnostic bound. The exact selected value is an experiment result,
not something this oracle can supply.

## Decision

- Keep A1 unchanged and accepted as the image-domain fallback.
- Keep the current fully-shared A1-field substitution as the `rho=1` N2 control, not the final
  layer model.
- Add only mean-preserving layer variance weights and a bounded covariance sweep in the next N2
  implementation.
- Measure covariance, opponent/luma RMS, mean drift, and footprint after 2383 as well as at negative
  density.
- Do not copy Spektrafilm source, parameters, fixed seeding, Gaussian blur, or layer reconstruction.

We were right not to copy Spektrafilm casually, but wrong to exclude it from our thinking. It is a
high-value external structural oracle for N2 precisely because the comparison shows both what to
borrow conceptually and what to reject explicitly.
