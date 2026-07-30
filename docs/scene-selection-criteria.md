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
| A3 | **No fade, dissolve or exposure ramp** | The estimator assumes stationarity. A ramp puts a deterministic offset into every difference. See §7.1 — the motion gate does see a ramp, but only in absolute terms, and cannot know how damaging it is. | `ramp_evidence` |
| A4 | **Enough frames for the longest lag** | Correlation is estimated from lag-1, lag-2 and lag-4 variance ratios; lag 4 needs at least 5 frames, and a stable ratio needs more. Too few frames does not fail loudly — it produces a noisy ρ that looks like a measurement. | `analysis_frame_count`, `duration_sec` |
| A5 | **No heavy quantisation or banding** | If the signal is quantised near the grain amplitude, the residual measures the quantiser. In deep shadow this is the common case, and it is exactly where the legacy curve has most of its points. | `signal_bit_depth_proxy` |
| A6 | **Not clipped at the format ceiling or floor** | Clipping truncates the residual distribution: σ is biased low, kurtosis is biased negative, and the tail evidence becomes meaningless. This matters most for the highlight band, which is already the scarcest. | `clipping_evidence` |
| A7 | **No titles, overlays or synthetic graphics** | Composited elements carry no negative grain. A noise-free region drags measured amplitude down and, being perfectly static, passes every staticness gate. | `overlay_evidence` |
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

Against §3 that left three gaps. All three are now implemented in
`capabilities/measure/admissibility.py`, and §7.1 records what measuring them taught.

## 7.1 The three added metrics, and what measuring them corrected

Each is expressed as a **share of the quantity it would corrupt** rather than in absolute units,
so one threshold serves every source.

**A3 — `ramp_evidence`.** A whole-frame level change shows up in the *mean* of each temporal
difference, since grain has zero mean. The reported `variance_share` is
`slope² / (2σ²)`, verified numerically against the closed form, so it states directly how much of
the measured amplitude would be the fade rather than the film.

*A claim in the first draft of this document was wrong.* It said a motion gate "cannot see" a
fade. It can: a uniform level change is entirely low-frequency and lands wholly in motion energy.
Measured, the fixed 0.005 gate and a 5% ramp share flip at the same slope on σ = 0.01 material.

What A3 actually adds is that it **scales with the grain being measured**, which a fixed RMS gate
cannot. The same 0.003-per-frame ramp:

| σ | ramp share | fixed motion gate | A3 |
|---:|---:|---|---|
| 0.020 | 1.1% | pass | pass |
| 0.010 | 4.5% | pass | pass |
| 0.004 | **28.1%** | pass | **flags** |
| 0.002 | **100.0%** | pass | **flags** |

On clean, low-amplitude material the gate waves through a fade that is the entire measurement —
and clean low-amplitude material is exactly the highlight end the corpus is short of. A3 also
names the cause: the gate can only say "too much motion".

**A6 — `clipping_evidence`.** Reports the *fraction* of samples at the ceiling and floor, which
`luma_peak_code` cannot. A frame with one specular highlight and a frame blown across half its
area report the same peak; only the second destroys the distribution evidence.

**A7 — `overlay_evidence`.** Detects composited graphics by the noise they do **not** have: film
grain is everywhere in a photographed frame, so a region whose temporal residual is near zero was
added after the scan. More reliable than looking for edges or shapes, and it is the property that
matters, since a noise-free region drags amplitude down while passing every staticness test.

Building it corrected a design error. A small logo should not disqualify a scene — it should stop
a *window* being placed on it. So the evidence carries a block mask, `select_windows` accepts it,
and windows overlapping a composited region are rejected with that reason. The scene-level flag is
reserved for material that is largely graphics.

Its limit is recorded as a test: detection is block-wise, so a graphic thinner than `block_size`
is averaged away against the grain around it. That is why the block size is a parameter.

## 7.2 Screening the real catalogue

Run over all 304 scenes, the filter admits **46** for every measurement — the static population,
which matches §1. What it also established is that the catalogue is **less informative than its
field names suggest**, and that mattered more than the count.

Three of its metrics are not measurements at all:

| Field | Value across all 304 | What it actually is |
|---|---|---|
| `signal_bit_depth_proxy` | 10.0 | the container's bit depth |
| `luma_peak_code` | 940 | the legal-range ceiling |
| `luma_floor_code` | 64 | the legal-range floor |

Fed to a threshold, each satisfies it for every scene while measuring nothing. A first run scored
**1 of 304 admissible** and it took checking the distributions to see that this was an import
error, not a result — `max_scene_cut_score` is ffmpeg's scdet score on roughly 0–100 (median 11.4),
not a probability, so a 0.3 threshold rejected everything.

That produced a design change worth keeping. Screening now has three outcomes, not two:
**rejected**, **admissible**, and **needs a frame-level check**. A criterion the catalogue cannot
answer is recorded as unassessed rather than passed, so a scene admitted on a constant can never
look identical to one that was actually checked. A5 and A6 are unassessable from this catalogue and
now say so; the frame-level implementations from §7.1 are what resolve them.

Coverage potential is reported but is weakly selective here: 46/45/45 scenes could contribute
shadow/midtone/highlight windows, because nearly every scene spans a wide level range. That is the
§2 point restated — the potential exists almost everywhere, and whether *static* windows actually
land in each band is a window-level question that screening cannot answer.

## 7.3 Correction: the screening unit is wrong

`scene-extractor-reconstruction.md` traced the original extractor and found that scene detection
under-segments badly — 21 ranges longer than 120 s hold **52.8%** of the running time, and the
longest is 878 s. Every per-scene metric is an aggregate over such a range, so the `static_score`
that justified a five-second sample describes 20–60× more material than was measured.

The 46-of-304 result in §7.2 therefore screens the wrong objects. The criteria in §3 and §4 stand
unchanged — they are properties of a measured *interval*, and a two-second window is one — but
they should be applied to windows rather than detected scenes.

Re-aggregating the existing per-frame survey into 2 s windows at 1 s stride yields 8,659 cut-free
windows, of which 3,662 are stable, against the 20 seconds of material currently in use. No
re-decoding is needed; the survey output is already on disk.

## 8. Sequencing

1. ~~Add the three missing scene metrics~~ — done; see §7.1.
2. ~~Encode §3 and §4 as a per-measurement admissibility filter~~ — done, in
   `capabilities/measure/screening.py`. See §7.2 for what running it on the real catalogue found.
3. Re-aggregate the survey into fixed time windows and screen those, per §7.3 — then encode §5 as
   a coverage-driven selector over them, maximising band, texture and position coverage subject to
   §6's cap rather than taking the highest-scoring candidates.
4. Report accepted, rejected and the reason for every rejection — and the unmeasured range.

Step 3 is where the actual gain sits. A greedy coverage selection over 42 admissible scenes should
comfortably beat four hand-picked ones, and — unlike a ranked list — it cannot quietly return
forty windows that are all shadow.
