# Scene and window selection criteria for grain measurement

Created: 2026-07-30 · Catalogue examined: `pulp_fiction_1994_full`, 304 scenes

Criteria first, automation after. An automated selector encodes whatever criteria it is given, so
writing them down is the part that determines whether automation removes bias or industrialises
it.

## 1. What the current selection actually is

The shipped grain preset was fitted from **four** scenes — `scene_001` (00h21m17s), `scene_003`
(00h14m48s), `scene_004` (00h36m40s), `scene_005` (01h07m25s) — chosen by hand. The catalogue for
the same film holds **304**.

Measured against the catalogue, three things stand out.

**The 52 approved scenes are the wrong 52.** They were approved from `face_scout_report.json` —
selected for faces. Faces mean people, and people move:

| | of the 52 face-approved | of all 304 |
|---|---:|---:|
| `static_score >= 0.7` | **5** | 46 |
| labelled `grain_candidate` | **10** | 91 |
| mean `motion_score` | 0.403 | 0.394 |

The face-approved set is *no more static than the catalogue average*. Reusing it for grain would
import a selection bias built for the opposite requirement.

**The usable population is roughly ten times what is being used.** 46 scenes reach
`static_score >= 0.7`, and **42 of those are also labelled `grain_candidate`**. Four are in use.

**Highlights are genuinely scarce, not merely unselected.** Only **6 of 304** scenes carry a
`highlight` label, against 45 `shadow` and 139 `midtone`. Some of the legacy curve's missing upper
range is a property of the film, not of the selection — and no selector can conjure material that
was never shot. That distinction has to survive into the report rather than being papered over.

## 2. Two tiers, because they answer different questions

Conflating these is what produced a curve covering the bottom fifth of its domain.

**Scene admissibility** — *can this scene be measured at all?* A per-scene yes/no. A scene that
fails is unusable regardless of how much material it would otherwise contribute.

**Set coverage** — *does the accepted set span what the measurement needs?* A property of the
selection as a whole. No single scene can satisfy it, and a set of individually admissible scenes
can still fail it completely.

A measured detail makes the split concrete: by scene-level median luma, **272 of 304 scenes are
"midtone"**. Almost nothing is a shadow scene or a highlight scene, because a whole frame rarely
averages to an extreme. Yet **45 of the 46 static scenes span more than 0.2 of the luma range
within themselves** (median p10–p90 spread 0.437). A single scene with a practical lamp and a deep
corner supplies both highlight *and* shadow windows.

**Selecting scenes by their mean level is therefore the wrong instrument entirely.** Coverage is a
property of the selected *windows*, and must be assessed there.

## 3. Scene admissibility criteria

Each states what it protects and which measurement would be wrong without it.

| # | Criterion | Why it is required | Catalogue metric |
|---|---|---|---|
| A1 | **Static enough** | Amplitude is measured from temporal differences. Motion is not removable by frequency separation — sub-pixel motion of fine detail lands in the residual and is scored as grain. The legacy preset's own warning records one scene's textured residual at **7.15×** its grain-dominated residual. | `static_score`, `mean_frame_diff` |
| A2 | **No cut inside the measured span** | A difference across a cut is two different pictures subtracted, not a residual. One such pair can dominate a window's variance. | `max_scene_cut_score` |
| A3 | **No fade, dissolve or exposure ramp** | The estimator assumes stationarity. A ramp puts a deterministic trend into every difference, inflating variance in a way no gate distinguishes from grain. | luma trend across the scene |
| A4 | **Enough frames for the longest lag** | Correlation is estimated from lag-1, lag-2 and lag-4 variance ratios; lag 4 needs at least 5 frames, and a stable ratio needs more. Too few frames does not fail loudly — it produces a noisy ρ that looks like a measurement. | `analysis_frame_count`, `duration_sec` |
| A5 | **No heavy quantisation or banding** | If the signal is quantised near the grain amplitude, the residual measures the quantiser. In deep shadow this is the common case, and it is exactly where the legacy curve has most of its points. | `signal_bit_depth_proxy` |
| A6 | **Not clipped at the format ceiling or floor** | Clipping truncates the residual distribution: σ is biased low, kurtosis is biased negative, and the tail evidence becomes meaningless. This matters most for the highlight band, which is already the scarcest. | `luma_peak_code`, `luma_floor_code` |
| A7 | **No titles, overlays or synthetic graphics** | Composited elements carry no negative grain and different statistics. They are not rare in a feature. | labels / manual review |
| A8 | **Decode path free of resampling** | Scaling correlates noise spatially and destroys whiteness, so the spectrum measures the scaler. Not a scene property but a *source record* one, and it invalidates the same measurements. | `SourceRecord.decode.scale == "none"` |

## 4. What each measurement additionally requires

The five evidence types have different preconditions. A scene admissible for one may be useless
for another, so admissibility is per-measurement, not global.

**Amplitude versus level** — needs windows spread across the level range. Its failure mode is a
curve that is honest where measured and flat everywhere else. This is a *set* requirement (§5).

**2D noise power and autocorrelation** — needs **flat** windows. Texture contributes structure to
the spectrum and would be read as grain correlation, so the very windows most useful for detecting
drift are the worst for measuring the spectrum. Also requires A8 absolutely: any resampling makes
the measured spectrum the scaler's.

**Distribution and tails** — needs a large sample count and, critically, **no clipping** (A6).
Clipping is invisible in σ but destroys the tail evidence, which is the part that says whether σ
means anything at all.

**Slow spatial heterogeneity** — needs **at least two scenes with unrelated pictures**, because the
screen-anchored test is a cross-source comparison. A single scene cannot distinguish a pattern
fixed in the scan from one fixed in the shot. This is a set requirement that no per-scene score can
express.

**Temporal behaviour** — needs drift-free windows. Sub-pixel drift decorrelates consecutive frames
and *masks* real correlation: measured on synthetic material, a true ρ of 0.50 with 0.7 px of drift
reads back as 0.045. A drifting window does not merely add noise, it produces the wrong conclusion
— "temporal independence established" when it was not.

## 5. Set coverage criteria

Assessed over accepted windows, not scenes.

| # | Criterion | Why |
|---|---|---|
| S1 | **Minimum windows per luma band** | A knot supported by two windows and one supported by two hundred must not look alike in the exported curve. |
| S2 | **Both flat and textured windows present** | Flat windows carry the spectrum; textured ones are the only place drift is detectable. A set of only flat windows cannot prove it is drift-free. |
| S3 | **Centre and edge positions** | Lens falloff, scanner non-uniformity and gate illumination vary across the frame. A centre-only set cannot separate them from grain. |
| S4 | **At least two distinct scenes with unrelated content** | Required for the screen-anchored heterogeneity test. Without it, that question is `None` — unknown — rather than answered. |
| S5 | **A cap on windows per scene** | One long static scene can otherwise supply most of the set, making per-scene conditions — one lens, one lab pass, one exposure — look like general properties. |
| S6 | **Explicit unmeasured range** | Where the set has no windows, the report says so, and the fit refuses to present that region as measured. |

## 6. Two anti-bias rules

**Select on admissibility, never on appearance.** The catalogue's `grain_score` ranks scenes by how
grainy they *look*. Selecting on it over-samples high-amplitude grain and under-samples clean
highlights, biasing the amplitude curve upward in exactly the region where it has fewest points.
Admissibility asks whether a scene can be *measured*; appearance asks whether it will give a large
answer. Only the first is a valid criterion.

**Reject rather than relax.** Already enforced in window selection: the gate never widens itself,
and a widened gate is recorded and printed. The same must hold at scene level. If the admissible
set is too small, the answer is that the material cannot support the measurement — not a lower bar.

## 7. What the catalogue already provides

Most of this is measurable today. Per scene, across all 304, it already carries `static_score`,
`motion_score`, `mean_frame_diff`, `max_frame_diff`, `p90_frame_diff`, `max_scene_cut_score`,
`signal_bit_depth_proxy`, `luma_floor_code`, `luma_peak_code`, `luma_range_p10_p90`,
`shadow_depth_proxy`, `highlight_headroom_proxy`, per-band scores, and labels including `static`,
`low_motion`, `grain_candidate`.

Against §3 that leaves three genuine gaps:

- **A3 (fades and ramps)** — no monotonic-trend metric exists. A luma trend over the scene, and its
  linearity, would supply it.
- **A7 (titles and overlays)** — no detector. Manual review flags exist but are not systematic.
- **A6 (clipping)** — `luma_peak_code` and `luma_floor_code` exist, but the *fraction* of pixels at
  the limit is what matters, and that is not recorded.

## 8. Sequencing

1. Add the three missing scene metrics (A3, A6-fraction, A7), so admissibility is fully measurable.
2. Encode §3 and §4 as a per-measurement admissibility filter over the catalogue.
3. Encode §5 as a coverage-driven selector: choose the set that maximises band, texture and
   position coverage subject to §6's cap, rather than taking the highest-scoring scenes.
4. Report accepted, rejected and the reason for every rejection — and the unmeasured range.

Step 3 is where the actual gain sits. A greedy coverage selection over 42 admissible scenes should
comfortably beat four hand-picked ones, and — unlike a ranked list — it cannot quietly return
forty windows that are all shadow.
