# Grain A/B: candidate vs legacy, image-domain

The measurement side is finished: the three-component candidate is fitted, validated held-out, and
materialised into a runtime kernel. What decides promotion is now visual, and this is the plan for
it. FAT owns everything except the legacy render.

## Ownership

| Owned by FAT | Owned by `obs-art-plugin` |
|---|---|
| clip selection from the catalogue | rendering its own authoritative grain shader |
| the candidate render (`reconstruct.render_candidate` + `luma_delta_to_rgb`) | — |
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
3. **Candidate grain** — `render_candidate` with the fitted `sigma(L) = 0.0616 * L^0.732` and the
   materialised footprint kernel (support 15 is sufficient: it retains >99.8% of the footprint
   energy and reproduces the radii and anisotropy).

## What must be identical across variants

Colour pipeline, optical processing, cadence, render resolution and the 1634/1640 scale convention,
output encoding, and the exact source frames and timing. Only the grain stage changes.

**Placement.** For the first comparison the candidate is injected in the **same final linear-image
location** as legacy grain — not in negative density. This tests grain *character* without
confounding it with negative-stage placement. The scalar luma delta is projected to RGB exactly as
the shader does (`native/.../film_look_native.effect`, `injectionDelta`):

```
dRGB = w * (dL / dot(w, w))      w = Rec.709 (0.2126, 0.7152, 0.0722)
```

which is `reconstruct.luma_delta_to_rgb`. Equal-RGB injection would add a tint unrelated to the
footprint and must not be used.

## Two strength comparisons

* **Native.** Legacy at its shipped `0.3` versus the candidate at its measured `1.0`. Honest to how
  each ships, but the overall strengths differ.
* **Amplitude-matched.** Scale both to the same RMS at a representative midtone (say linear 0.05, a
  clean part of the measured range), then judge *spatial character and tonal modulation* rather than
  overall strength. The candidate's midtone RMS is `sigma(0.05) = 0.0616 * 0.05^0.732 ~ 0.0069`; the
  legacy midtone RMS is measured from its own render, and the match factor is their ratio.

Render at **normal strength for the acceptance judgement**, and keep **4x only as a labelled
diagnostic** — 8x is useful for a gradient sanity check but too extreme for preference.

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
* Is its horizontal structure (the ~1.44:1 anisotropy) visible or objectionable?
* Does it preserve detail better or worse than legacy?
* Which version would you actually use?

## Decision

If the candidate is consistently preferred or judged equivalent, with no recurring artefacts,
promote it as the **image-domain statistical profile**. The candidate carries its spatial-scale
anchor (the 3840x1634 active geometry) so runtime scaling stays explicit.

Only *after* that does the negative-stage integration experiment begin, and it is isolated: it
reuses the accepted temporal field and footprint unchanged, and varies only where and how amplitude
enters the layer-density model. It must not simultaneously redesign spatial character, colour
covariance, or placement.
