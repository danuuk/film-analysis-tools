# Reference analysis: system architecture

Created: 2026-07-30

This document exists because the preceding work went the wrong way round. Individual estimators
were built and tuned — residual extraction, window gating, evidence types, screening — before the
system they belong to was described. Each is defensible on its own and none of it is wasted, but
tuning a gear cannot tell you whether the clock keeps time.

## 1. Two sources, kept apart

Grain reconstruction draws on two kinds of knowledge, and they must not be blended:

| | Source | Derived from | Lives in |
|---|---|---|---|
| **Statistical, spatial, temporal behaviour** — amplitude against level, frame-to-frame dependence, cadence, distribution, spatial structure, motion | **measured from reference material** | scans and masters | `film-analysis-tools` |
| **Colour behaviour** — layer interaction, emulsion response, colour coupling | **derived from the spectral model and physical mechanism** | published data, spectral simulation | `film-emulation-engine` |

The separation is already enforced in one place and should be enforced everywhere:
`capabilities/measure/chroma.py` makes a profile fitted from subsampled material *typed* as
luma-appearance, with no colour-covariance field to populate. That is the right shape. A delivery
master can say how grain behaves; it cannot say how the layers couple, and nothing measured from
it should be allowed to look as though it did.

Practically: **the reference corpus answers "how does grain behave"; the spectral model answers
"what colour is it".** Every architectural decision below serves the first question only.

## 2. The system

```
  ┌─ SOURCE REGISTRY ──────────────────────────────────────────────┐
  │  irreplaceable material, identified by content hash            │
  │  films, camera shoots, charts. Lives outside every repository. │
  └────────────────────────────┬───────────────────────────────────┘
                               │  one cheap pass, once per source
  ┌────────────────────────────▼───────────────────────────────────┐
  │  SURVEY        reduced-resolution per-frame metrics            │
  │                motion, level, saturation, hue, bit depth, cuts │
  └────────────────────────────┬───────────────────────────────────┘
                               │  aggregation + detection
  ┌────────────────────────────▼───────────────────────────────────┐
  │  CATALOGUE     queryable index of INTERVALS and REGIONS        │
  │                the contract between material and analysis      │
  └────────────────────────────┬───────────────────────────────────┘
                               │  query: "give me material where …"
  ┌────────────────────────────▼───────────────────────────────────┐
  │  CORPUS        a named, traceable set assembled by query       │
  │                with a holdout reserved by construction         │
  └────────────────────────────┬───────────────────────────────────┘
                               │  materialise frames on demand
  ┌────────────────────────────▼───────────────────────────────────┐
  │  MEASUREMENT   the five evidence types, independently          │
  └────────────────────────────┬───────────────────────────────────┘
  ┌────────────────────────────▼───────────────────────────────────┐
  │  FIT → HOLDOUT → PROFILE   compact parameters, validated,      │
  │                            exported as numbers only            │
  └────────────────────────────────────────────────────────────────┘
```

**The catalogue is the architectural centre.** Everything above it is about material; everything
below is about measurement. The extractor should not *infer* which material suits a question — it
should *query* for it. Today it infers, which is why suitability judgements live scattered in
analysis scripts instead of in one place that can be inspected, corrected and reused.

## 3. The catalogue's subject: intervals and regions

The single most consequential correction. The catalogue currently indexes **scenes**, and scene
detection under-segments so badly that 21 ranges hold 52.8% of the running time and the longest is
878 seconds. A metric aggregated over such a range cannot describe a five-second sample inside it.

Three levels, each addressable and traceable:

**Source** — a film or shoot. Content hash, edition, cadence, decode contract, active picture.

**Interval** — a time span within a source, typically 1–3 seconds. *Not* a shot, not a detected
scene: a unit short enough that its aggregate statistics describe every frame in it. Intervals
overlap; they are cheap.

**Region** — a tile within an interval. Position, size, and its own statistics over the interval's
frames.

Grain is measured on regions across an interval. That is exactly the process described from the
start — *the same local area, over a second or several, with its mean level and the residual
around it* — and intervals plus regions are simply that process named.

### What each level carries

| Interval | Region |
|---|---|
| start/end frame, duration, cadence | position, size |
| motion estimate, cut-free flag | temporal stability across the interval |
| level distribution: p10, p50, p90 | mean level and its spread |
| shadow and highlight coverage | colour: hue, saturation, chroma |
| saturation and hue summary | texture / flatness |
| effective bit depth, clipping fraction | face membership (pixel-level, from mesh) |
| face presence, count, bounding box | overlay / composited flag |
| exposure and level ramp | clipping and quantisation locally |

The rule: **anything downstream analysis needs to know in order to choose material belongs in the
catalogue, not in the analysis.** If a question is being answered by reading scripts, that answer
should have been a column.

## 4. The query interface

The extractor's entire relationship with material is one call:

```
corpus = catalogue.regions(
    level_band     = "shadow",
    stability      = "high",
    min_duration_s = 1.0,
    inside         = "face_mesh",     # or None
    exclude        = ("overlay", "clipped", "ramping"),
    per_source_cap = ...,             # so one long take cannot dominate
)
```

and the answers to the questions already asked follow from it directly:

- *skin-tone grain* → regions inside a face mesh, stable, in intervals ≥ 1 s
- *shadow / highlight / saturated / neutral grain* → the same query with a different level or
  colour predicate
- *a new analytical question later* → a new predicate over existing columns, not a new pipeline

Every returned region carries its provenance: source hash, interval, frame range, tile rectangle,
and the criteria that admitted it. A fitted parameter must be traceable to the regions that
produced it, or it cannot be re-derived or disputed.

## 5. What already exists

More than the current use suggests. The survey pass has already been run over the whole film and
its output kept.

| Asset | Scale | Usable for |
|---|---|---|
| `per_frame_metrics.csv` | **37,084 rows @ 4/s**, whole film | interval construction — motion, level, saturation, hue, bit depth, cut score |
| `scene_catalog.json` | 304 ranges, 50+ metrics each | browsing and review; **not** selection (§3) |
| `face_scout_report` | 304 scenes probed; **216 with a detected face** | face-bearing interval discovery |
| — of those | **159 with face area > 2% and duration ≥ 1 s** | directly answers the 10–20 face-sequence question, an order of magnitude over |
| FaceMesh provider | pixel-level facial geometry | region-level skin membership |
| `selections/*_top_5.json` | grain, shadow, colour, bloom, halation, MTF | shows the categories were already conceived — but capped at five |
| per-scene assets | contact sheet, preview, thumbnails ×304 | human review of any query result |
| camera catalogue | 105 clips, 11 categories, hash-identified | the second source, already rebuilt |
| sample packs | ~260k rows, per-row source frame and pixel index | precedent: region-level provenance already works |

The `_top_5` selections are the narrowness in miniature: the categories existed, and each was
truncated to five. The corpus of five primaries plus one control was not a judgement about how
much material was needed — it was the shape of the selection tooling.

## 6. Gaps

1. **No interval level.** Scenes are the only temporal unit, and they are the wrong one.
2. ~~**No region catalogue.**~~ — closed by step 3. Tiles were recomputed inside each analysis run
   and discarded, so they could not be queried, compared across runs, or traced to.
3. **Face data is per-scene, not per-region.** One probe frame per scene gives presence and a
   bounding box; FaceMesh gives pixel geometry but is not joined to the catalogue.

   Step 2 joined what exists to intervals and, more usefully, *measured how thin it is*: of 8,660
   intervals only **352 contain the probe frame** and 488 sit within 2 s of one. The other 4,371
   "face" intervals inherit a verdict from up to 439 s away. So this is not merely a resolution
   gap to be refined later — **the face evidence for most of the film does not exist yet**, and
   the honest fix is to re-probe at interval cadence rather than to interpolate what is there.
   Cheap to do: the scout is one detector call per frame and only stable intervals need it.
4. **Categories are selections, not predicates.** `grain_top_5.json` is a frozen answer. A
   predicate over columns would let the same question be re-asked as criteria improve.
5. ~~**No traceability from parameter to region.**~~ — closed by step 3. A knot in the amplitude
   curve could not name the regions behind it; `Region.region_id` is that name.
6. **No holdout by construction.** The sixth control scene was chosen by hand and remembered.
7. ~~**Multi-source time axes were merged.**~~ — closed by review. Every temporal union pooled
   spans across sources, so two films each contributing 0–2 s reported two seconds of coverage
   instead of four, and `FaceObservation` carried no source identity at all — a probe from one
   film could mark intervals in another as carrying an *observed* face. Unions and joins now
   operate within `source_id`. Single-source work could never surface any of this, which is
   exactly why it needed finding before profiles combine films, scans or reels.
8. **Thresholds are scale-bound and nothing declares it.** Found while running step 3. Both
   `DEFAULT_BAND_EDGES` and `WindowGate.max_motion_energy` are absolute numbers compared against
   levels whose scale is chosen by the caller, so the same gate is 100× stricter or looser
   depending on what 1.0 means. Regions now *record* the scale, and a mismatch between edges and
   data is detected — but the gate itself still cannot check its own units. The fix is to declare
   thresholds against a named scale with a reference white, not to keep tuning the numbers.
9. **Slow heterogeneity answered the wrong question.** — closed by review. The cross-source test
   correlated blurred temporal *mean* images, which detects a shared **additive** fixed pattern
   (scanner shading, vignetting, dirt) and cannot see a **multiplicative** envelope modulating
   grain amplitude — the latter leaves the temporal mean untouched by construction. Measured on
   two unrelated sequences built with an identical envelope, it scored **−0.07** and declared the
   envelope not screen-anchored, while scoring **0.98** on a merely additive pattern. It now
   compares level-normalised grain-energy maps, and the mean-image detector survives under its
   accurate name, `additive_pattern_evidence`. The two are orthogonal on all four cases.

## 7. What survives from the work already done

None of the measurement layer is invalidated by this — it sits *below* the catalogue and does not
care how material was chosen:

- `measure/residual` — amplitude and correlation, validated against known σ and ρ
- `measure/windows` — region gating and coverage reporting
- `measure/evidence` — the five independent evidence types
- `measure/admissibility` — ramp, clipping and overlay checks
- `measure/chroma` — the luma/colour boundary from §1
- `measure/screening` — the criteria; correct, but applied to the wrong objects (§3)

What changes is what feeds them. Screening should filter intervals; window selection should run
inside chosen intervals and *record* its regions to the catalogue rather than discarding them.

## 8. Sequencing

1. ~~**Build the interval layer**~~ — done. `capabilities/catalogue/{survey,intervals}`. Run over
   the existing 37,084-row survey it yields **8,660 cut-free intervals covering 148.6 minutes**,
   of which **2,860 are stable at motion p90 ≤ 1.0, covering 57.5 minutes** — against 4 scenes ×
   5 s = 0.3 minutes in use. No decoding was needed.

   The within-frame decomposition earned its place immediately. By frame *mean* the film has 98
   highlight intervals; by what intervals *contain*, **6,265** do — and 1,846 of those are also
   stable. Highlight material exists after all; it was invisible to any statistic that judged an
   interval by its average.
2. ~~**Join face and colour metadata to intervals**~~ — done. `capabilities/catalogue/annotate`.

   The join exposed that **the two annotations are not equally trustworthy**, and the types now
   say so. Colour is *measured*: the survey sampled every frame, so an interval's saturation comes
   from its own frames. Face presence is *inherited*: the scout probed **one frame per scene, at
   the midpoint**. Median scene 6.6 s — but p90 is 49 s and the longest is 878 s, so an interval
   can sit **439 seconds** from the only frame ever checked. Only 54 of 216 face-bearing scenes
   are short enough for the probe to cover them.

   So each face annotation carries its distance to the observation and a confidence tier, and
   `FaceConfidence.usable_for_skin` excludes `INHERITED`. Over the film:

   | | intervals | of which stable |
   |---|---|---|
   | face reported, any confidence | 5,211 | 1,864 |
   | — observed (probe inside the interval) | 352 | **130** |
   | — near (≤ 2 s) | 488 | **150** |
   | — inherited (same range, far away) | 4,371 | 1,584 |

   A query for "stable intervals with a face" returns 1,864, but **1,584 of those rest on a probe
   frame that may be minutes away**. The honest skin corpus is 280 intervals, and 58 of those also
   carry a face large enough to sample (area ≥ 5%). Still an order of magnitude above the four
   scenes in use — and now the difference between 1,864 and 280 is visible instead of assumed.

   Saturation bands were calibrated against the measured distribution over 37,084 frames
   (p25 13.2, p75 21.4, p95 32.2) rather than guessed: neutral 1,437, low 4,932, moderate 1,712,
   saturated 579 — of which 416 neutral and 251 saturated are also stable. Hue is recorded but
   deliberately not offered as a selection predicate: `huemed` is a frame *median*, so it names a
   dominant cast and cannot say what a frame contains.

   `capabilities/catalogue/ingest` reads the existing survey and scout output into these types, so
   the numbers above are reproducible rather than the product of a one-off script. It declares the
   producer→survey column mapping instead of guessing it, and takes the face probe position as a
   parameter: the scout recorded a verdict per scene but not *when* it looked, so every
   `distance_s` rests on reconstructing that as the scene midpoint. Naming it makes it correctable.
3. ~~**Define the region record and index it.**~~ — done. `capabilities/catalogue/regions`.
   Validated end to end on the real 4K PQ master: 8 stable intervals spread across the film,
   decoded at native resolution, tiled and indexed as **629 regions**.

   A `Region` is a `Window` plus provenance — source, interval, frame range, gate, band edges and
   **level scale** — addressed by a readable `region_id` (`source@frame+n:x,y+size`), which closes
   gaps 2 and 5. Four things came out of running it that were not visible before:

   **A region count is not a sample size.** Those 629 regions are 8 disjoint spans totalling
   **16.0 seconds** — 79 tiles per span, all views of the same few frames. `independence()`
   reports the merged, disjoint time behind any selection, and `per_interval_cap` forces spread
   over depth. A corpus that reports 629 has said almost nothing about how much film it saw.

   **The level scale is load-bearing, not metadata.** Same footage, same tiles, same default band
   edges (0.02, 0.25) — normalised so 1.0 is the PQ peak, **737 of 739 regions land in "shadow"**;
   normalised so 1.0 is 100-nit diffuse white, the split is 467 shadow / 162 midtone. 162 midtone
   regions appeared from fixing units alone. That reads exactly like "this film has no midtones",
   and the two diagnoses need opposite fixes, so `RegionIndex.edges_bracket_the_data` now
   distinguishes them: when no band edge falls inside the measured range, every region is in one
   band *by construction* and the summary says so.

   **The motion gate is scale-bound too, and nothing declares it.** Motion energy is an RMS of
   temporal differences in level units, so it scales with the levels: measured medians 0.000011 →
   0.000110 → 0.001098 for level scales ×1 → ×10 → ×100, and acceptance 48 → 45 → 36 of 120 tiles.
   `WindowGate.max_motion_energy = 0.005` is documented as though it were absolute; it is not.
   This is gap 7 below — the record makes it visible but does not yet fix it.

   **A tile mean hides what the tile holds** — the step-1 finding one level down. By mean these
   629 regions contain **zero** highlight material; by content, **39 regions across 5 spans**. So
   `Window` now carries its 1st/99th percentile spread (§3's table always said a region carries
   "mean level *and its spread*"), and regions answer `contains_band` as intervals do.
   A practical consequence for the fit: `flat + contains highlight` returns **0** regions —
   a highlight inside a tile is an edge, so every highlight-bearing tile is textured. Highlight
   amplitude must come from aligned textured tiles or from tiles small enough to sit inside the
   bright area; it cannot come from flat ones.
4. ~~**Run one thin vertical slice end to end**~~ — done, and taken *ahead* of the remaining
   architecture steps deliberately. Three layers had been built and validated in isolation and
   nothing had ever run the whole way through, so nobody could say what fraction of a real film
   survives the chain. `studies/grain_slice` runs
   `SourceRecord → catalogue query → extraction → admissibility → window selection → residual
   measurement → evidence JSON + report` over two unrelated sources.

   **Actual coverage** after the corrections below, which is the deliverable rather than the
   parameters. Three different amounts of time are now reported separately, because catalogue
   support is not measurement support:

   | | intervals | admissible | windows | candidate s | decoded s | evidence s |
   |---|---|---|---|---|---|---|
   | Pulp Fiction UHD (PQ, 3840×1634 active) | 2,671 / 9,270 (28.8%) | 6 / 8 | 245 / 540 (45.4%) | 12.0 | 2.50 | **2.50** |
   | Sony ZV-E10 II C0014 (S-Log3, 1920×1080) | 4 / 4 (100%) | 4 / 4 | 94 / 128 (73.4%) | 5.0 | 1.67 | **1.67** |

   **Defects that existed only at the seams.** None was visible from inside any layer, and the
   last four were found by review after the first run:

   - `metadata=print` emits at INFO level, so `-loglevel error` discarded the entire survey.
   - **Admissibility was asked after the transfer.** Clipping is defined by the container's limits
     and overlay detection asks whether the *encoded* signal carries noise; both are
     container-domain questions, and asking them in linear light reported 87.6% clipped and 100%
     noise-free on good film.
   - **Letterbox bars were never cropped.** `SourceRecord.Crop` existed from the start with
     exactly this rationale and nothing filled it in; 23.9% of every frame sits at code 0 and read
     as hard-clipped black. Measured from row and column means — ffmpeg's `cropdetect` reported
     full frame on this 10-bit PQ master at every limit tried.
   - **Overlay was a veto rather than the mask `select_windows` already accepts.** 20–22% of 32 px
     blocks in the HEVC master have exactly zero temporal variation (encoder skip blocks).
   - **Limited-range PQ was normalised twice.** ffmpeg range-expands when converting limited-range
     YUV to `gray16le` — verified against a synthetic clip: code 64 arrives as 0, code 940 as
     1023 — and the study applied the 64..940 mapping again on top. This crushed low PQ values
     toward zero and distorted the linear levels, the amplitude-versus-level placement and the
     shadow tails. Now covered by an ffmpeg integration test with known codes; the previous unit
     test bypassed ffmpeg and so codified the wrong assumption.
   - **Evidence described one interval, not the corpus.** The run measured only the interval that
     yielded the most windows — 63 of 245 regions — which is also a selection bias toward whichever
     picture passes most easily. Every admissible interval is now measured independently and
     reported as a median with its full range and *n*.
   - **No per-evidence routing.** Every estimator got the same windows although they need
     different material. Spectrum now takes flat windows, distribution unclipped windows, temporal
     windows that can vouch for their own correlation; amplitude takes all, since `extract` aligns
     internally and each point carries its own trust. Spectrum and distribution also used *raw*
     lag-1 differences rather than the aligned residual, so `residual.aligned_residuals` is now
     public and both use it.
   - **Screen anchoring was framed as a cross-source test.** It needs unrelated picture content
     through the *same* acquisition geometry, so two distant intervals of one film answer it and
     two different cameras cannot. Comparing intervals 3 s apart in a tripod shot returned 0.93
     and 0.99 — a measurement of "same scene" — so a 60 s minimum separation is now enforced.

   **What the corrected run says**, after a second review round fixed four more defects:

   - **ρ was saturating.** The estimate is clamped to ±0.99, and two estimates pinned to the same
     edge agree perfectly — so a consistency check called them trustworthy while the amplitude
     correction `1/√(1−ρ)` multiplied σ by 10. Sony's six saturated points had median σ 0.00736
     against 0.000759 for the identified ones. A boundary solution is now `rho_saturated` and
     never `parameter_identified`, the unclamped solution is kept, and **0 saturated points are
     accepted** in the rerun.
   - **σ mixed trustworthy and untrustworthy points.** Split into `raw_sigma` (descriptive, all
     points) and `trusted_sigma` (identified ρ only).
   - **`trusted_fraction` was 1.0 by construction**, since only trustworthy windows were passed to
     the estimator — it read 100% where 1 of 60 accepted windows qualified. The share of *accepted*
     windows is now recorded instead.
   - **The Pulp survey was regenerated with the crop applied** (37,084 frames, ~50 min). It matters:
     the letterbox was damping every frame's motion, so cut-free intervals fell from 2,671 to
     **2,095** and detected cuts rose from 610 to 1,522.

   **The heavy tails: what actually causes them.** A third review round showed my previous
   explanation was wrong, and the four-way experiment that settles it is cheap. Running the *same*
   three tiles at two durations, singly and pooled, on the exact central 128 px crop:

   | excess kurtosis | A: 10f single | B: 60f single | C: 10f pooled | D: 60f pooled | duration (B−A) | pooling (D−B) |
   |---|---|---|---|---|---|---|
   | Pulp Fiction | +2.21 | +2.25 | +8.57 | +7.53 | **+0.03** | **+5.29** |
   | Sony C0014 | +0.68 | +0.66 | +0.85 | +0.94 | **−0.02** | **+0.29** |

   **Sequence length has no effect at all.** The earlier claim that "short duration explains the
   rest" is withdrawn — it was never isolated, because the wide/deep comparison changed four
   things at once. Pooling contributes +5.29 on Pulp, which matches the independently computed
   scale-mixing prediction of +5.5 almost exactly, so mixing is quantitatively confirmed as the
   pooling mechanism. The remaining gap to the wide pass (+24.8 Pulp, +666 Sony) is therefore
   **tile selection**: the wide pass pools every accepted tile, the deep probe takes the
   lowest-motion tile per interval.

   **Most deep probes are not trustworthy, and the report now says so.** Printing only
   "identified" while omitting trust presented five of six probes that had *failed* the drift gate
   as accepted measurements. With tiles now spread one per interval, **all three Pulp probes are
   rejected** — sub-pixel residual 0.63–1.10 with shifts at the search boundary — and one of three
   Sony probes is trusted. So *"Pulp is effectively independent" is not supportable*: the wide
   pass has 1 trustworthy point in 267, and no long Pulp tile passes the drift gate.

   | | trustworthy long tiles | ρ where trusted | σ |
   |---|---|---|---|
   | Pulp Fiction | **0 of 3** (all drifting) | — | — |
   | Sony C0014 | 1 of 3 | +0.372 | 0.00062 |

   **Zero inflation separates the two sources cleanly.** Pulp 10.9%, Sony 32.6% of frame-to-frame
   steps exactly zero, and the fingerprints differ:

   | | 16 px blocks all-frozen or all-live | zeros by texture (flat / rough) | step occupancy 0, ±1 |
   |---|---|---|---|
   | Pulp Fiction | **20.3%** | 11.1% / 10.4% | 10.9%, 9.5% |
   | Sony C0014 | **0.0%** | 32.8% / 32.4% | 32.6%, ~22% |

   Pulp's zeros clump into the coder's block grid — codec block freezing. Sony's show no block
   structure, no texture dependence and no long runs, but **76% of all steps fall within ±1 code**:
   the per-frame change is at or below the quantisation step. That is a quantisation-limited
   signal, not noise reduction and not block freezing.

   Also fixed in this round: saturation was counted inside the *trusted* set, where a saturated
   estimate can never appear, so it always read zero — the same by-construction error as
   `trusted_fraction`. It is now counted over all points (**3 Pulp, 27 Sony saturated, all
   rejected**), and `AmplitudePoint` carries `raw_rho` so a rejected point can say how far outside
   the model it fell. Per-window normalisation now uses `(r − mean)/std` rather than a floored
   MAD, which does not standardise variance and had been *re-introducing* the mixing it claimed to
   remove.

   Screen anchoring now runs over **15 interval pairs** up to 6,512 s apart: grain envelope median
   −0.003 (−0.085…+0.103), additive pattern median −0.006 (−0.133…+0.219). Stated as the bounded
   claim it is: *no strong scan-fixed envelope was detected between these sampled pairs*; a
   correlation near zero rejects a strong common pattern but does not rule out a weaker one, and
   no null distribution was computed.

   **The selection stage was rebuilt around the legacy geometry**, which is the empirical anchor
   the shipped profile provides. Verified in `presets/pulp_fiction_4scene_motion_gated.json`:
   **128 px patches on a 64 px stride, 1,392 candidate positions per 4K frame**, 72–120
   consecutive native-cadence frames, fixed coordinates, per-patch motion across the whole
   sequence — and its scene 004 kept **818 of 1,392** patches in a scene that was plainly not
   static. Finding stationary regions *inside* moving scenes is the job, and four things prevented
   it:

   - **Global motion was a hard reject.** `motion_p90 <= 0.01` on the whole frame discarded
     intervals before anything asked whether they held stable ground. Now a loose reject for cuts
     and large camera movement only: **5,906 of 9,270 intervals** pass, against 2,095 before.
   - **The search was sparse.** A 256 px stride examined 90 positions covering 23.5% of the
     picture. On the legacy grid the run now evaluates **1,416 positions per interval** — 7,080
     across five — and keeps 2,550.
   - **A two-second interval was judged on its first 0.417 s.** Scout frames are now spread evenly
     across the whole interval, so a tile cannot be accepted for being still during the first
     fifth of a span an actor crosses later.
   - **One aggregate could not express a brief disturbance.** Each tile now carries a motion
     *time series*: median, p90, the fraction of transitions above threshold, and the longest
     continuous unstable run. An actor crossing truncates a tile's usable run instead of
     disqualifying it.

   Two consequences arrived immediately. **Flat tiles exist again** — 797 of 2,550 on Pulp, 68 of
   1,531 on Sony — so the noise power spectrum is measurable at all, where the sparse grid had
   reported it unavailable on every interval. And the stratified shortlist reaches real midtones
   rather than the darkest corner: the Pulp deep tile at linear level **0.0871** returned excess
   kurtosis **+1.2**, where minimum-motion selection had been landing on tiles at 0.0001.

   **Both residual representations are computed, as the legacy path did.** `grain_properties`
   used `fields = frames - frames.mean(axis=0)` for spatial character and `deltas` for amplitude
   and temporal independence, and compared them. Forcing every statistic through the lagged
   difference made every spatial question depend on the temporal trust gate. The agreement is
   itself evidence — for a stationary independent field `std(fields) ≈ std(deltas)/√2` — and it
   earned its place on the first run:

   | | dual-residual ratio | reading |
   |---|---|---|
   | Pulp Fiction, 3 deep tiles | **1.02, 1.04, 1.06** | agree — no dominant persistent structure |
   | Sony C0014, deep tiles | **2.49, 2.40** | mean-removed residual far larger: persistent structure or slow drift |

   That is a warning about the Sony correlation estimates specifically. A slowly drifting tile
   produces a high apparent ρ, so "Sony ρ ≈ 0.4 is camera temporal processing" is **not** safe:
   the same tiles show 2.4× excess in the mean-removed residual, which is what drift looks like.
   The Pulp tiles agree on this test while still failing the drift gate — exactly the case for not
   routing every statistic through one residual.

   **The dual-residual verdict was wrong, and it made me overstate a conclusion.** For a
   stationary AR(1) field the ratio is `1/√(1−ρ)` with no drift whatever — 1.29 at ρ = 0.4 and
   2.58 at ρ = 0.85 — so reading a raw ratio above 1.25 as "persistent structure" mislabels every
   correlated field. It now reports `adjusted = ratio·√(1−ρ)` and stays *ambiguous* when ρ is
   untrusted. Corrected, the three Pulp tiles read **1.028, 1.026, 1.014 — agreement**, and my
   claim that "Sony ρ ≈ 0.4 is not safe" was itself too strong: only one of the three Sony tiles
   shows genuine excess beyond what its own correlation predicts. AR(1) null cases at ρ = 0.4 and
   0.85 now sit alongside the linear-drift case.

   Three implementation defects went with it. **Disturbed transitions were tolerated but never
   excluded** — a brief crossing was detected, allowed, and then measured anyway; measurement now
   requires `clean_throughout` while scouting keeps tolerating, which is the two-stage rule.
   **The 48-tile cap was documented and unused**, so one interval pooled 1,135 residual stacks;
   it is applied now (159 measured of 2,550 catalogued, reported as its own coverage stage).
   **The deep probe ran 2.5 s against a 2 s scouted span**, leaving its last half-second
   unscouted; it is 48 frames now.

   **The new scout was checked against the four legacy scenes** it is modelled on. Stable counts,
   against the legacy pass of 1,392 candidates:

   | scene | legacy | new, native cadence | new, 0.2 s spread |
   |---|---|---|---|
   | 001 (00:21:17) | 1,363 | 1,335 | 1,277 |
   | 003 (00:14:48) | 1,027 | 614 | 858 |
   | 004 (00:36:40) | 818 | 754 | 566 |
   | 005 (01:07:25) | 951 | 884 | 755 |

   Same order and the same ranking of scenes, with native cadence closest to legacy on three of
   four — as expected, since legacy differenced adjacent 24 fps frames. Scene 003 is the one real
   disagreement and is not yet explained. The scout is *not* numerically identical to the legacy
   moving box blur: it uses a block mean, and its 0.005 threshold now applies between frames
   ~0.2 s apart rather than adjacent ones.

   One documentation correction: the dual representation is a **scalar comparison on the deep
   probes only**. The reported NPS still comes from lag-one difference residuals, which is
   defensible but is not the fuller dual routing the previous wording implied.

   **The drift gate was the thing blocking every long Pulp measurement, and it was too blunt.**
   It rejected on the *maximum* sub-pixel residual across the sequence and on *any* boundary
   contact, so one unreliable registration in 47 pairs rejected an otherwise stationary tile — and
   the reported 0.46–0.62 px was the largest single-pair residual, not drift over the sequence.
   The gate now asks whether the per-pair shifts describe **one coherent motion**: it uses the
   90th-percentile residual and the boundary-*fraction*, and requires directional coherence
   (`|mean shift| / mean|shift|`) before calling anything drift. An isolated or scattered contact
   is registration failure on a hard tile, not demonstrated image motion. Coherent synthetic drift
   is still rejected; a stationary tile with one bad pair is not.

   With that fixed, **the three Pulp deep tiles pass** — and with them the amplitude points:

   | | before | after |
   |---|---|---|
   | trustworthy amplitude points | 2 / 159 | **111 / 159** |
   | intervals with a temporal estimate | 1 / 5 | **5 / 5** |
   | deep tiles trusted | 0 / 3 | **3 / 3** |

   This is the first provisional characterisation of Pulp Fiction grain the chain has produced,
   and it meets the standard set for accepting it: **three independent spans** (t = 2636, 3998,
   6470 s), each with unsaturated ρ, adjusted dual-residual agreement (1.026, 1.041, 1.026), and
   no coherent motion. Two conclusions, stated at the confidence they have earned:

   - **ρ is near zero** — median 0.005, range 0.003–0.024 across all five intervals. Independent
     per-frame grain is the correct initial runtime assumption.
   - The shadow tails are **strongly consistent with a quantisation / floor interaction**, but
     "confirmed not grain" would be too absolute. Two deep-shadow probes (level ~0.001) are
     heavy-tailed and quantised — σ ~0.0003, kurtosis +44 to +95 — while the one deep midtone
     probe (level 0.063) is approximately Gaussian — σ 0.0066, kurtosis +0.7. But that is *one*
     midtone probe and *no* highlight probe, so the honest statement is a **delivery-dominated
     shadow effect that should not be copied into the grain generator unless further evidence
     contradicts it**, not a proven property of the negative.

   Trusted σ runs **0.0004 to 0.0072** and rises with level — the amplitude-versus-level curve
   beginning to appear.

   **The first compact fit.** `capabilities/fit/amplitude` fits σ(level) on the 111 trustworthy
   points, deliberately narrow: the interval is the independent unit (not the tile, whose frames
   are shared), validation is **leave-one-interval-out**, intervals are weighted equally so the
   33-point interval does not outvote the 2-point one, and everything is in log-amplitude
   coordinates. Three candidates, chosen on *held-out* error:

   | model | params | in-sample log-RMSE | held-out log-RMSE |
   |---|---|---|---|
   | constant null | 1 | 1.476 | 1.648 |
   | **power/floor** | 3 | 0.197 | **0.206** ← chosen |
   | piecewise 4-knot | 4 | 0.170 | 0.181 |

   The compact model wins: the piecewise curve is better in-sample and only 0.025 better held out,
   below the 0.05 that would justify the extra knots. The result is a **pure power law,
   σ(L) = 0.0616 · L^0.732** — no amplitude floor is resolved within the measured range (σ₀
   collapsed to zero, so reporting a floor crossover would be false precision). Held-out log-RMSE
   0.206 means predictions land within about ±23% of measured σ. Predicted σ tracks the deep
   probes: 0.00039 at L = 0.001 (observed ~0.0003), 0.0082 at L = 0.063 (observed 0.0066).

   Two limits are stated in the output rather than left implicit. The **supported range is
   0.00016–0.177 linear** — there is *no* highlight evidence, and the curve must not be
   extrapolated above it. And "no amplitude floor" is a statement about σ only: the deepest-shadow
   *distribution* is still heavy-tailed, but that is a shape effect, not a floor in amplitude, and
   the region stays flagged as delivery-dominated.

   **The fitted domain is kept separate from density.** This curve is a final-reference
   **appearance envelope in decoded linear-light luma** — not a negative-density granularity
   curve. Injecting it directly into density space would be a category error. The intended path is

   ```
   Pulp reference → observed output-luma σ(level) → appearance-envelope fit (here)
       → negative-chain response/Jacobian → required layer-density modulation → runtime grain
   ```

   so the measured 5219 granularity response supplies the base physical shape while this fit is the
   aesthetic target or correction envelope on top of it. The first fit deliberately does **not**
   infer coloured grain or layer covariance, highlight behaviour, a non-Gaussian distribution from
   the shadow tails, temporal correlation beyond the near-independent baseline, or a spatial
   footprint — each is a later step on already-measured evidence, starting with normalising the
   trustworthy residuals by predicted amplitude and testing whether the spatial spectrum is stable
   across level bands before fitting one common footprint.

   Artefacts: [`amplitude_fit_pulp_2026-07-30.txt`](results/amplitude_fit_pulp_2026-07-30.txt) and
   its JSON.

   Three lesser corrections rode along. Deep probes were being selected from the *tolerated* tiles,
   not the clean ones, so a tile excluded from ordinary measurement could be chosen for the most
   important one; they now come from `clean`. The disturbance limit was counted in transitions, so
   the same rule was ~4.8× more tolerant by duration at the 5 fps spread scout than at native
   cadence — it is timed now (`MAX_DISTURBANCE_S`), which also removes one of the confounds behind
   the scene-003 discrepancy. And `rho_trusted` for the dual residual required only that ρ was
   identified; it now also requires lag-4 consistency, but deliberately **not** `not drifting` —
   the dual diagnostic exists to test the drift gate, so gating it on the drift verdict would be
   circular.

   **The spatial-footprint study.** `capabilities/measure/footprint` answers the one question that
   comes after amplitude: once predicted amplitude is divided out, is the grain *shape* stable
   enough across level and interval to use a single footprint? The shape metrics are
   scale-invariant by construction — the radial PSD is normalised to unit in-band mean, and grain
   radius, anisotropy and the block peak are all ratios — so **the footprint verdict does not
   depend on the amplitude fit at all**, which means it cannot be an artefact of amplitude-fit
   error. The independent unit is the interval, and the decision is made against a null:

   | comparison | log-spectral distance (median / p90) |
   |---|---|
   | split-half within interval+band | 0.043 / 0.064 |
   | between interval, same band | 0.076 / 0.116 |
   | **between level band, same interval** | **0.097 / 0.135** |

   Between-level variation (median 0.097) is within the between-interval baseline (p90 0.116):
   the footprint changes no more with level than it changes between scenes. **One common footprint
   suffices** — no level-dependent kernel is justified on this evidence. The shape itself, on 66
   clean windows across the two bands present (no highlight data):

   | band | windows / intervals | grain radius | anisotropy | block peak | level range |
   |---|---|---|---|---|---|
   | shadow | 38 / 5 | 1.23 px | ~2.0 | 1.08 | 0.0002–0.020 |
   | midtone | 28 / 3 | 1.20 px | ~2.0 | 1.07 | 0.021–0.149 |

   Two features are consistent across both bands. The grain is **~1.2 px** — near the resolution
   floor, close to spatially white — and it is **anisotropic, roughly 2:1 horizontal** (the
   autocorrelation is twice as wide across as down; the ratio is coarsely quantised to integer
   pixel lags, so "about 2:1" is the honest precision). The block peak near 1.07 says the encoder's
   block grid, which the zero-inflation study found freezing 20% of 16 px blocks, is **not**
   strongly imprinting a periodic pattern in the grain frequency band — the frozen blocks are a
   time-domain effect, not a spatial-frequency one. The shadow distribution shape stays out of this
   fit, as required: shadow windows contributed spectral evidence, but their heavy tails are not
   read as a particle law.

   The three components of the statistical grain candidate are now available *separately*, which
   was the point of keeping the questions apart:

   ```
   appearance amplitude : sigma(L) = 0.0616 * L^0.732   (appearance envelope, luma)
   spatial structure    : one common footprint, ~1.2 px, ~2:1 horizontal anisotropy
   temporal baseline    : independent frames (rho ~ 0.005)
   ```

   Colour and layer covariance remain a separate question: this luma reference cannot supply
   trustworthy chromatic grain evidence, and nothing here pretends to.

   One contract issue closed alongside: the amplitude model now takes an explicit out-of-range
   policy (`predict(level, outside=...)`), because the power law extrapolated while the piecewise
   curve clamped and a consumer must not get one behaviour by accident. Default is `clamp` (hold
   the envelope at the measured endpoints); `extrapolate` and `error` are opt-in. Artefacts:
   [`footprint_stability_pulp_2026-07-30.txt`](results/footprint_stability_pulp_2026-07-30.txt)
   and its JSON.
   Still open: the deep probe's alignment gate rejects almost everything at 60 frames — a
   sub-pixel residual of 0.63–1.10 px over 2.5 s is ordinary gate weave or scan drift, not a
   defect, but the current gate has no way to accept a tile that drifts slowly and steadily. Until
   that is resolved there is **no trustworthy long-sequence measurement of Pulp Fiction at all**,
   which is a sharper statement of why no fit should be attempted yet. The screen-anchoring guard
   is also still purely temporal; 60 s does not guarantee unrelated pictures, and an image
   similarity check would close it.

5. **Implement the query interface** over intervals and regions, with provenance on every result.
6. **Assemble corpora by query**, with a holdout reserved by construction rather than by memory.
7. **The compact fit** — begun. `capabilities/fit/amplitude` fits σ(level) on the 111 trustworthy
   Pulp points with leave-one-interval-out validation and a compact model chosen on held-out
   error: a pure power law σ(L) = 0.0616·L^0.732 over the supported range 0.00016–0.177 linear.
   The **spatial footprint** is now measured too (`capabilities/measure/footprint`): once
   amplitude is divided out, the grain shape is stable enough across level and interval to use one
   common footprint (~1.2 px, ~2:1 horizontal anisotropy). Still to come on this line: holdout
   residual reconstruction and several-second renders, then the map into the negative-density
   stage — kept explicitly separate from this appearance-envelope fit.

Steps 1-3 and 5-6 are architecture; step 4 is the slice that proves the architecture runs.
Only after them does refining an estimator have a defined meaning, because only then is there an
answer to "on what material, chosen how" — and, as step 4 showed, whether that material can carry
a measurement at all.
