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
2. **No region catalogue.** Tiles are recomputed inside each analysis run and discarded. They are
   never indexed, so they cannot be queried, compared across runs, or traced to.
3. **Face data is per-scene, not per-region.** One probe frame per scene gives presence and a
   bounding box; FaceMesh gives pixel geometry but is not joined to the catalogue.
4. **Categories are selections, not predicates.** `grain_top_5.json` is a frozen answer. A
   predicate over columns would let the same question be re-asked as criteria improve.
5. **No traceability from parameter to region.** A knot in the amplitude curve cannot name the
   regions behind it.
6. **No holdout by construction.** The sixth control scene was chosen by hand and remembered.

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
   the existing 37,084-row survey it yields **8,660 cut-free intervals covering 147.6 minutes**,
   of which **2,860 are stable at motion p90 ≤ 1.0, covering 55.0 minutes** — against 4 scenes ×
   5 s = 0.3 minutes in use. No decoding was needed.

   The within-frame decomposition earned its place immediately. By frame *mean* the film has 98
   highlight intervals; by what intervals *contain*, **6,265** do — and 1,846 of those are also
   stable. Highlight material exists after all; it was invisible to any statistic that judged an
   interval by its average.
2. **Join face and colour metadata to intervals**, so face-bearing and colour-conditioned
   intervals become queryable rather than inferred.
3. **Define the region record and index it.** Window selection already computes everything a
   region record needs; it currently throws it away.
4. **Implement the query interface** over intervals and regions, with provenance on every result.
5. **Assemble corpora by query**, with a holdout reserved by construction rather than by memory.
6. *Then* return to the statistical methods, inside a system that can say what they were run on.

Steps 1–4 are architecture. Only after them does refining an estimator have a defined meaning,
because only then is there an answer to "on what material, chosen how".
