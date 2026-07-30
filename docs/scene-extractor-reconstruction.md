# The original scene extractor: reconstruction, and what survives

Created: 2026-07-30 · Traced at `obs-art-plugin@897aad09` · Catalogue: `pulp_fiction_1994_full`

The intended process was: a cheap reduced-resolution FFmpeg pass to find scenes where neither
camera nor actors move for several seconds; within those, a search for tiles that stay stable
across the interval; neutral tones first, later widened to darks and highlights; then per tile,
the mean level and the residual noise around it.

That design is sound and it is **implemented**. What went wrong is one structural choice inside
it, and the damage it did is measurable.

## 1. The pipeline as built

| Stage | Implementation | State |
|---|---|---|
| Cheap survey | one FFmpeg pass: `scale=640:-2:bicubic`, `fps=6`, preblur σ 1.0, `scdet=threshold=10`, `signalstats` | **works, and its output was kept** |
| Scene segmentation | `scdet` cuts → 304 ranges | **broken — see §2** |
| Per-scene scoring | aggregate per-frame metrics over each range → 8 category scores + labels | **inherits §2's breakage** |
| Second pass | seek into source for selected scenes, previews and contact sheets | works |
| Frame extraction | `extract_frames` reads a table of `scene_id`/`start`/`duration` | works |
| Tile search | `analyze_patches`: 256 px grid, stride 256, low-pass/high-pass motion split, gate at 0.005 | works, analysed separately |

So the chain is connected end to end, and the survey pass genuinely is cheap and was genuinely
run. **37,084 per-frame rows at 4 samples/second across the whole 154.5-minute film are on disk**,
carrying `mafd`, `score`, `yavg`/`ylow`/`yhigh`, `satavg`, `huemed`, `ydif` and `ybitdepth`.

## 2. The structural fault: "scene" is the wrong unit

`scdet=10` under-segments badly. A feature averages 3–8 seconds per shot; this detection averages
**30.5**, and the tail is worse:

| | scenes | share of running time |
|---|---:|---:|
| longer than 30 s | 63 | **79.6%** |
| longer than 60 s | 37 | 67.7% |
| longer than 120 s | 21 | **52.8%** |
| longest single "scene" | — | **877.8 s** |

A 298-second range is not a shot. And because every per-scene metric is an *aggregate over the
whole range*, a `static_score` computed across 298 seconds says almost nothing about any
particular five-second interval inside it.

The four scenes that produced the shipped grain preset show exactly this:

| Extracted | Catalogue scene | Detected length | Extracted | Fraction the metric describes | Labels |
|---|---|---:|---:|---:|---|
| `scene_001` 00:21:17 | `scn_000016` | 298.8 s | 5 s | **1.67%** | static, grain_candidate |
| `scene_003` 00:14:48 | `scn_000014` | 271.0 s | 5 s | **1.85%** | static, grain_candidate |
| `scene_004` 00:36:40 | `scn_000039` | **4.5 s** | 5 s | **111%** | low_motion only |
| `scene_005` 01:07:25 | `scn_000128` | 144.5 s | 5 s | 3.46% | static, grain_candidate |

Two things fall out of that table.

**`scene_004` was never an admissible pick.** Its static score is 0.64, below the threshold for the
`static` label, and it carries only `low_motion`. Worse, the extracted 5 seconds is **longer than
the 4.5-second detected range**, so the extraction ran past a cut — criterion A2 violated in the
shipped data, not hypothetically.

**For the other three, the justifying metric describes 20–60× more material than was measured.**
That is not a selection error so much as a category error: the score answered a question about a
five-minute range and was used to license a five-second sample.

A related flattening: the scene aggregate reports `signal_bit_depth_proxy` as 10.0 for every one
of the 304 scenes, while the per-frame `ybitdepth` it was derived from ranges **1.0 to 10.0**. The
aggregate discarded the only per-scene quantisation evidence the survey actually captured.

## 3. What survives, and it is most of it

**The survey pass and its output are fully reusable, and re-aggregating them costs nothing** — no
decoding, no seeking. Replacing the aggregation unit is a pure post-processing change.

Re-aggregated into **2-second windows at 1-second stride**, rejecting any window containing a
`scdet` peak:

```
2s windows, no cut inside:                     8,659
  mafd <= 0.5   2,117 windows   ~35 min of material
  mafd <= 1.0   3,662 windows   ~61 min
  mafd <= 2.0   5,772 windows   ~96 min
```

Against **four scenes × 5 seconds = 20 seconds** actually used. The material was always there; the
unit hid it.

This also matches the original intent more closely than the implementation did. The design called
for stability "over several seconds — or even over a shorter interval such as one second". Fixed
time windows *are* that. Detected scenes were a proxy for it, and a poor one.

## 4. A finding that no selector can fix

Level coverage of the 8,659 cut-free windows, by frame-average luma on the legal range:

| Band | all windows | stable (mafd ≤ 1.0) | stable share | median mafd |
|---|---:|---:|---:|---:|
| deep shadow | 1,676 | 805 | 48.0% | 1.12 |
| shadow | 6,890 | 2,822 | 41.0% | 1.32 |
| midtone | 93 | 35 | 37.6% | 1.98 |
| **highlight** | **0** | **0** | — | — |

**Zero highlight windows, before any stability filter is applied.** And the stable *share* is
roughly constant across bands (48%, 41%, 38%), so stillness and brightness are not anticorrelated
here — the selection is not biased against bright material. The material does not exist.

This explains the shipped preset's amplitude curve spanning linear luma 0.00012 to 0.282 without
appealing to selection bias at all. That range is what stable Pulp Fiction *is*.

**One caveat, and it matters.** These are *frame-average* levels. A frame averaging 20% can still
contain a practical lamp or a window blowing out, and those are highlight *tiles*. So the correct
reading is: highlight coverage cannot come from scene- or window-level selection on this film, and
must come from the tile search — or from different material. It is not proof that no highlight
tile exists.

## 5. What to reuse, change, and discard

**Reuse as-is**
- The survey pass design and its parameters. Cheap, sufficient, already run.
- `per_frame_metrics.csv` — the whole basis for everything below, at 4 samples/second.
- `extract_frames` consuming a table of `scene_id`/`start`/`duration`; it does not care how the
  table was produced.
- The tile-level low-pass/high-pass motion separation, which is a genuinely good idea and is
  already carried into `capabilities/measure/residual`.

**Change**
- **Selection unit: fixed time windows, not detected scenes.** Score each window from the frames
  inside it. This is the single change that recovers 61 minutes from 20 seconds.
- **Cut handling: reject a window containing a `scdet` peak**, rather than trusting range
  boundaries — which failed on `scene_004`.
- **Keep `ybitdepth` per frame.** It is the A5 evidence, and aggregation destroyed it.

**Discard**
- Per-scene aggregate scores as a *selection* input. They remain fine for browsing and review.
- The `grain_score` ranking as a selection criterion, for the reason already recorded in
  `scene-selection-criteria.md` §6: it ranks by how grainy a scene looks.

**Still to verify** — the tile search itself was analysed only for its motion gate. Whether its
neutral-first bias persisted after the widening to darks and highlights, and how
`--coverage-color-metric` interacts with band coverage, has not been traced yet.

## 6. How this changes the plan

`scene-selection-criteria.md` §7.2 concluded that screening admits 46 of 304 scenes. That number
is now suspect for a reason that has nothing to do with the thresholds: it screens the wrong
objects. The criteria themselves stand — A1 through A8 are properties of a measured interval, and
a 2-second window is a measured interval — but they should be applied to windows.

The next step is therefore narrower and better founded than "a coverage-driven scene selector":
re-aggregate the existing survey into windows, screen those, and let the tile search run inside
the survivors.
