# Controlled N1/N2 synthetic strength selection v1

## Question

This comparison keeps two decisions separate:

1. Does N2's accepted structured field improve on N1's unfiltered white field when both use
   strength `1.0`?
2. Which bounded N2 strength—`0.75`, `1.0`, or `1.25`—is the best engineering default?

N2 `1.57501286` is retained only as a labelled positive control. Private shadow review already
found it too aggressive on skin and colour noise. If the synthetic sequence cannot expose more
opponent-colour activity at that value than at `1.0`, the study aborts before assembling a report.

## Ownership and domains

Film Emulation Engine owns the deterministic four-second, 24 fps, 1920x1080 scene, exact region
interiors, scene-linear Rec.709 to XYZ construction, exact forward Sony S-Log3/S-Gamut3.Cine
encoding, and compiled N0/N1/N2 floating-point renders. The scene includes neutral exposure
steps, a logarithmic gradient, equal-luminance colour patches, textured skin-colour detail,
tracked soft discs, a smoothly ramped disc, and unchanged controls.

Film Analysis Tools consumes those outputs only through `forward/`. It measures every N1/N2
variant against N0, assembles videos, and writes the report. It never subtracts N1 directly from
N2 because their differing random phase would dominate that image.

## Signed-delta rule

The exact delta is `grain_on - N0` in linear floating-point RGB. Some values are positive and
some negative, so placing the delta directly on black clips half the evidence. Every review video
therefore has three fixed columns:

1. normal processed output;
2. signed RGB delta with exact zero at neutral grey;
3. absolute RGB magnitude on black.

The signed and magnitude views share one authored linear scale (`0.05` by default) across all
frames and variants. There is no per-frame normalization. All numerical metrics use the original
signed float delta before display encoding or clipping.

## Measurements

For every static or tracked hard-interior region and every non-null variant, the report records:

- temporal luma, RGB, and opponent-colour RMS;
- opponent/luma ratio;
- temporal mean RGB and luma drift;
- p95 and p99 delta magnitude;
- adjacent-frame correlation with spread;
- spatial spectrum, horizontal/vertical correlation radii, and anisotropy;
- output floor and ceiling contacts;
- N2 `0.75`, `1.25`, and `1.57501286` RMS ratios relative to `1.0`.

Tracked masks exclude each soft object's antialiased edge ring, so geometric motion is not counted
as grain. The report plots RMS, chromatic/luma ratio, and drift against known scene exposure, plus
the N1/N2 strength-1 spectra.

## Run

Build the isolated FEE bundles from the Film Emulation Engine repository:

```sh
uv run --group spectral python -m tools.spectral.build_sony_negative_grain_n1_bundle \
  --output build/bundles/negative-grain-n1-synthetic-v1
uv run --group spectral python -m tools.spectral.build_sony_negative_grain_n2_bundle \
  --output build/bundles/negative-grain-n2-synthetic-v1
```

Then use the FEE environment while exposing this package's source tree:

```sh
PYTHONPATH=../film-analysis-tools/src .venv/bin/python -m film_analysis_tools.cli \
  negative-grain-synthetic \
  --n1-bundle build/bundles/negative-grain-n1-synthetic-v1 \
  --n2-bundle build/bundles/negative-grain-n2-synthetic-v1 \
  --report /private/tmp/fee-negative-grain-synthetic-strength-v1
```

The private output contains `index.html`, `metrics.json`, and one synchronized triptych video per
variant. No generated video or private source enters either repository.

The FullHD run defaults to four independent frame-render processes. Frame results are consumed in
index order, so process scheduling cannot change cadence, random seeds, measurements, or video
alignment. `--frame-workers` may be reduced on memory-constrained systems; increasing
`--variant-workers` adds threads inside every process and is normally less effective for this
CPU-bound reference chain.

## Decision boundary

The downsampled overview sequence can characterize the bounded strengths and expose spatial,
temporal, tonal, or chromatic instability; it does not by itself select between approximately
one-pixel candidates. Generate the final native-pixel review separately:

```sh
PYTHONPATH=../film-analysis-tools/src .venv/bin/python -m film_analysis_tools.cli \
  negative-grain-native-crops \
  --n1-bundle build/bundles/negative-grain-n1-synthetic-v1 \
  --n2-bundle build/bundles/negative-grain-n2-synthetic-v1 \
  --report /private/tmp/fee-negative-grain-native-crops-v1
```

That bounded deliverable renders only N2 0.75, 1.0, and 1.25 plus the internal N0 delta control.
It exports synchronized, unresized skin, deep-shadow, neutral-midtone, and tracked-colour crops.
Each video stacks the three strengths vertically and the normal, signed-delta, and magnitude views
horizontally. The streams are lossless RGB H.264 4:4:4; no 4:2:0 conversion or spatial resize is
used.

Synthetic skin colour and detail are not a human face. The selected value therefore receives only
a few natural clips as a final veto, not another broad corpus fit.

Standing constraints remain unchanged: N2 placement and shared A1 structure are accepted,
`1.57501286` is not a default, `1.0` leads, no hybrid is built, and sublayers/covariance remain
deferred.
