# Camera corpus: what the legacy catalogues held, and what replaced them

Created: 2026-07-29 · Investigated at `obs-art-plugin@897aad09`

The legacy catalogues were read **once**, to establish which source files existed. Nothing
about their structure, scoring or metadata survives into
[`sony_zve10ii_v1`](../src/film_analysis_tools/capabilities/catalogue/data/sony_zve10ii_v1.json).

## 1. What was found

**The source material is intact and outside the legacy repo.** 105 clips from a Sony ZV-E10 II,
each with its Sony XML sidecar, across four shoots in `~/Movies`:

| Shoot | Clips | Range | Size |
|---|---:|---|---:|
| `samples` | 30 | C0011–C0040 | 2.0 GB |
| `samples_dark` | 26 | C0062–C0087 | 1.6 GB |
| `samples_dd_dark` | 38 | C0088–C0136 | 2.5 GB |
| `samples_evening` | 11 | C0045–C0055 | 1.1 GB |

All S-Log3-cine / S-Gamut3.Cine, 1920×1080, 10-bit 4:2:2, full range, 23.98p. A fifth
directory, `obs_samples`, holds seven `.mkv` **renders** rather than camera material and is not
part of the corpus.

The XML sidecars carry only `CreationDate`, `Duration`, codec, and the gamma/primaries
declaration — no ISO, shutter or white balance. The capture timestamps are still useful: 30
clips shot at 16:00–17:00, 75 in the evening and after midnight.

## 2. The two generations, and why neither was enough

**Generation 1 — `findings/reference_camera_corpus/curated_v1`.** CSV-based: 656 candidate
frames scored down to 48 clips, plus four contact-sheet PNGs. References material by
`(clip, shoot_dir, time_sec)` with **no content hashes**. Its scoring columns — `dark_colour`,
`dark_coherence`, `sat_cov`, `quality`, `rank` — show it was optimised for one question (dark
colour) rather than for category coverage.

**Generation 2 — `findings/reference_slog3_source_intake/sony_zve10ii_samples_dark_v1`.** Much
more rigorous: `source_files.csv` with per-file sha256, a `source_lock.json` binding the file
list, extraction manifest and frame tree, and a recorded ffmpeg decode contract. But it covers
**only `samples_dark`** — 26 clips.

Together they catalogued 74 of 105 clips. **31 were never catalogued at all**, 29 of them from
`samples_dd_dark` — which turned out to contain much of the most useful material.

Generation 2's decode contract was worth keeping and is carried forward verbatim: full range,
`bt709` matrix, rgb48 output, no scaling.

## 3. The scenes you remembered

All recovered, and visible on a fresh 105-clip contact sheet:

- **Evening LED lights** — `C0051` (cyan strip over a pedal shelf; the most out-of-gamut clip in
  the corpus at 38% of pixels outside Rec.709) and `C0052` (red/magenta strip, poster wall).
- **LED lights with a face** — `C0055`, evening interior with a warm key and magenta LED spill.
  It is the only clip in the corpus with two genuinely competing sources.
- **Rubik's cube** — nine clips: `C0053` (bright), `C0065`, `C0088`, `C0106`, `C0107`, `C0118`,
  `C0119`, `C0129`, `C0134` (spot-lit against near-black).
- **Lamps** — `C0069`, `C0070`, `C0074`, `C0076`, `C0100`, `C0101`, `C0120`, `C0125`.
- **Indoor objects** — guitar pedals, hats, plants, cutlery, coloured pencils, clothes pegs,
  orchids, a colour chart, books with gold foil.
- **Lighting range** — rooftop daylight (`C0037`–`C0039`), hard window sunlight on a
  checkerboard (`C0023`, `C0024`), warm evening interiors, near-black rooms.

## 4. What replaced them

One catalogue, built by re-measuring all 105 clips directly: three probe frames each, decoded
S-Log3 → linear Rec.709, spatially sub-sampled 4×.

**Thresholds were chosen after measuring, not before** — and two first attempts failed outright,
which is the reason for saying so:

- A "near maximum code value" test for clipped highlights matched **nothing**. S-Log3 never
  reaches full range; the top code on this material sits near 0.72–0.76. Overexposure had to be
  expressed in linear terms instead, above 100% reflectance.
- A naive `(max-min)/max` saturation exceeded 1 on saturated sources, because S-Gamut3.Cine →
  Rec.709 legitimately produces **negative** channels. Saturation is now measured on clamped
  values, and out-of-gamut share is tracked separately as its own signal.
- The skin cast rules were initially absolute, which put 11 of 12 face clips in "mixed", and the
  green/magenta sign was inverted. This material has no white balance applied, so there is no
  absolutely neutral frame — warm/cool runs 0.22 to 1.15 across every face clip. The cut points
  are now corpus-relative.

### Category populations

| Category | Clips |
|---|---:|
| `normal_daylight_interior` | 24 |
| `deep_underexposure` | 31 |
| `overexposure_clipped` | 25 |
| `saturated_practical` | 44 |
| `low_saturation` | 13 |
| `difficult_shadows` | 31 |
| `motion_or_noise` | 41 |
| `skin_neutral` | 1 |
| `skin_warm` | 9 |
| `skin_green` | **0** |
| `skin_mixed` | 1 |
| *(no category — ordinary material, kept)* | 9 |

Membership overlaps deliberately; six clips are both deeply underexposed *and* contain saturated
practicals, which is where the interesting failures live.

## 5. Two honest gaps

**`skin_green` is empty.** The corpus contains no green-lit skin. Every face clip measures
magenta-leaning or neutral on the green/magenta axis. This is a **shooting gap, not a
classification problem** — no threshold change will produce material that was never captured. A
short shoot with a green practical or fluorescent source would fill it.

**Skin categories rest on 11 human-labelled clips.** Face *presence* was established by visual
review of the contact sheet, because detection does not exist here yet; the lighting sub-class is
measured. Every skin category is flagged `human_labelled` in the manifest so nothing downstream
mistakes the label for a measurement. When `capabilities/detect` lands, these should be re-derived
and the labels checked against it.

Also worth knowing: `skin_neutral` and `skin_mixed` have one clip each. That is enough to probe a
condition, not enough to characterise it.
