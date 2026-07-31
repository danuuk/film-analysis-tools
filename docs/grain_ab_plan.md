# Grain A/B: candidate vs legacy, image-domain

The measurement side is finished: the three-component candidate is fitted, validated held-out, and
materialised into a runtime kernel. What decides promotion is now visual, and this is the plan for
it. FAT owns everything except the legacy render.

## Ownership

| Owned by FAT | Owned by `obs-art-plugin` |
|---|---|
| clip selection from the catalogue | rendering its own authoritative grain shader |
| compile and execute the FEE appearance-grain model; assemble all variants | render the shipped grain-only pass from the supplied pre-grain float frames |
| assembly of the three variants into one comparison | — |
| metrics (per-clip RMS at a midtone, amplitude-match factor) | — |
| the report and the human questions | — |

**No new analysis machinery goes into `obs-art-plugin`.** The plugin renders its shipped shader
result and nothing more; FAT does the rest.

## The three variants

1. **No grain** — the pipeline with the grain stage disabled.
2. **Legacy grain** — the plugin's current shader at its shipped `0.3` strength. It is a larger
   composite effect than the candidate: it also carries sparkle, cluster modulation and a small
   chromatic component. That is acceptable for the native comparison but **must be stated in the
   report**, because it is not a like-for-like footprint comparison.
3. **Candidate grain** — the compiled FEE
   `sony_slog3_kodak_5219_2383_appearance_grain_v1` model at live strength `1.0`. FAT's
   `render_candidate` remains a compact numerical oracle; it is not the renderer used for human
   acceptance.

## What must be identical across variants

Colour pipeline, optical processing, cadence, render resolution, output encoding, and the exact
source frames and timing. Only the grain stage changes. FEE must expose the float32
`appearance_grain.input_image` tap immediately after `l100_to_relative`. That same array is the
no-grain output, the candidate's `image` and `amplitude_reference`, and the plugin grain pass's
image/amplitude source. There is no intermediate integer or sRGB round trip.

**Placement.** For the first comparison the candidate is injected in the **same final linear-image
location** as legacy grain — not in negative density. This tests grain *character* without
confounding it with negative-stage placement. The scalar luma delta is projected to RGB exactly as
the shader does (`native/.../film_look_native.effect`, `injectionDelta`):

```
dRGB = w * (dL / dot(w, w))      w = Rec.709 (0.2126, 0.7152, 0.0722)
```

which is `reconstruct.luma_delta_to_rgb`. Equal-RGB injection would add a tint unrelated to the
footprint and must not be used.

## Compiled FEE execution

Build the scoped bundle from the sibling FEE checkout:

```sh
uv run --project ../film-emulation-engine --group spectral python \
  -m tools.spectral.build_sony_appearance_grain_bundle \
  --root ../film-emulation-engine \
  --output build/bundles/sony-slog3-5219-2383-appearance-grain-v1
```

The A/B adapter must load and compatibility-check that bundle, request
`appearance_grain.input_image`, and render candidate/no-grain by changing only the live
`appearance_grain.strength`. It must record the bundle manifest/model identity in the result.
Calling FAT's prototype renderer in place of the compiled graph is a failed precondition.

The plugin branch receives the exact tapped float signal. A CPU twin is useful for diagnosis but
does not satisfy the final legacy condition: that branch must identify the shipped native
`PSWindowsGrain` pass or an OBS capture. If no native device is available, candidate/no-grain may
be prepared, but the acceptance result remains blocked rather than silently relabeling a twin.

## Resolution policy for the first review

The first Sony comparison is 1920x1080. FEE v1 scales the 1634-pixel kernel by bilinear coordinate
sampling and restores unit L2. FAT's independent discrete-kernel characterization is frozen in
[`docs/results/appearance_grain_scaling_v1.json`](results/appearance_grain_scaling_v1.json).
At 1080p the candidate is 0.774/0.602 px with 1.285 anisotropy. It is **not** described as the
1.435-anisotropy reference-height footprint. The sampled 720p and 2160p results are also recorded
so a future renderer change cannot masquerade as the same candidate.

## Two strength comparisons

* **Native.** Legacy at its shipped `0.3` versus the candidate at its measured `1.0`. Honest to how
  each ships, but the overall strengths differ.
* **Amplitude-matched.** Scale both to the same RMS at a representative midtone (say linear 0.05, a
  clean part of the measured range), then judge *spatial character and tonal modulation* rather than
  overall strength. The candidate's midtone RMS is `sigma(0.05) = 0.0616 * 0.05^0.732 ~ 0.0069`; the
  legacy midtone RMS is measured from its own render, and the match factor is their ratio.

Render at **normal strength for the acceptance judgement**, and keep **4x only as a labelled
diagnostic** — 8x is useful for a gradient sanity check but too extreme for preference.

## Focused equal-control review

After the broad native-strength catalogue pass, use the mixed-light C0055 interval for one
equal-control diagnostic at legacy `1.0` and measured-candidate `1.0`. This does not assert that
the two control scales are perceptually or statistically matched; it answers the narrower question
of what each implementation produces at its authored one-times setting.

Use two rows (legacy, measured candidate) and three native-resolution columns:

1. the identical compiled-FEE pre-grain float signal;
2. the corresponding grain-on result at strength `1.0`;
3. `grain_on - pre_grain`, shown as signed linear RGB at true `1x` over an 18% linear-grey
   carrier.

The first column is deliberately duplicated between rows to preserve spatial and temporal
alignment. The delta column must not use normalization, contrast stretching, per-frame scaling, or
another fitted multiplier. It is a visualization of the exact delta, not an image-domain signal or
a claim that legacy `1.0` and candidate `1.0` have equal RMS.

## Material

Five to ten seconds each, from real motion (not stills):

* dark / underexposed;
* ordinary mixed lighting with practical highlights;
* skin;
* fine fabric, hair, or detailed background;
* a gentle gradient or defocused surface.

The catalogue already indexes material for the first four (shadow, mixed, skin-labelled, textured
intervals); the gradient is synthetic.

## Human questions

Kept simple, at normal viewing distance:

* Does the grain feel natural in motion?
* Does it dirty or destabilise shadows?
* Does it sit convincingly on skin?
* Is its sampled horizontal structure (1.285:1 at 1080p) visible or objectionable?
* Does it preserve detail better or worse than legacy?
* Which version would you actually use?

## Decision

The broad five-scene motion review and focused equal-control C0055 review are complete. The project
owner judged legacy `0.3` quite subtle, found the measured candidate more interesting, and promoted
it to the next steps. The exact scoped decision is frozen in
[`docs/results/appearance_grain_motion_ab_decision_v1.json`](results/appearance_grain_motion_ab_decision_v1.json).

The candidate is accepted as the **image-domain statistical baseline at the reviewed scaling
policy**. It carries its 3840x1634 scale anchor and sampled multi-resolution outputs; promotion does
not imply resolution-invariant anisotropy, Kodak material identification, equal meaning for the two
`1.0` controls, plugin-default eligibility, or acceptance of the complete colour/optics model.

The negative-stage integration experiment may now begin, and it remains isolated: it
may reuse Gaussian innovation and independent cadence as starting hypotheses, but it must not
insert the accepted final-reference kernel unchanged before print. It starts with a provisional
latent kernel, propagates it through negative/print/projection/optics, and adjusts that latent
kernel until the rendered output footprint matches the accepted appearance target. The final-luma
amplitude law is never reused as a density law. That experiment must not simultaneously redesign
colour covariance or unrelated placement controls.
