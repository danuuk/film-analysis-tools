# Film Analysis Tools — migration plan

Status: plan (v3)
Created: 2026-07-28
Source: `obs-art-plugin` @ `897aad09` (`mediachar/` + `scripts/` + `docs/` + `findings/`)
Siblings: `film-emulation-engine` (FEE), `obs-art-plugin` (legacy plugin)

---

## 1. What this repo is

The validation system outgrew the thing it validates. `mediachar/` is 221 modules and 233k LOC
— richer than the plugin itself — and it grew by accretion: a hypothesis appeared, a test was
added; the next reused and extended it; the one after forced a partial rewrite. It works and
covers real ground, but it is one large, loosely structured component inside a repository that
is now becoming the legacy version.

This repo gives it its own development path, so the statistical and scientific work can evolve
without every change being weighed against immediate plugin impact.

**"Independence" here means development independence** — separate repo, tests, cadence, review
bar. It does **not** mean standalone. FAT stays closely and appropriately dependent on the
plugin and the engine. There is no goal of running without them.

### 1.1 Three questions, three homes

"How accurately have we implemented the plugin?" is really **two** questions, and neither is the
one the system currently answers. Separating them settles what belongs where.

| Question | Contour | Needs | Home |
|---|---|---|---|
| **Is the transform well-formed?** monotone, no folding, no channel collapse, clean gradients, intact highlights, valid LUT cells | **B — intrinsic** | the transform + synthetic probes. **No corpus, no catalogue, no reference.** | **FEE** |
| **Does the shipped shader implement the model?** | parity | model + native pass bodies | **obs-art-plugin** (`shader_runner_parity`, the `native_*` modules) |
| **Does the model match the reference?** | **A — fidelity** | corpus, catalogue, sampling, statistics | **FAT** |

**Contour B belongs primarily in FEE**, for a reason that decides it: these checks must run
*early and automatically* — "poor colour rendering and collapsed highlights at the initial
stages" means CI on every mechanism change. Only FEE is positioned for that. It also already
does this in embryo: `LumaCurveParams._check_monotone` rejects a non-monotone curve at authoring
time, `synthetic/charts.py` and `l100_optical_charts.py` provide deterministic probe input, and
`tests/architecture/test_optical_acceptance.py` sets the acceptance-gate precedent.

**The seam is the shape of the output:**

- a **verdict on one model** → FEE gate. Synthetic input, seconds, per-commit, pass/fail.
- a **comparison across alternatives** → FAT study. "Which candidate keeps highlights cleanest
  while holding skin accuracy?" — that is tier ≥ 1 work and needs FAT's cohorts and statistics.

FAT consumes FEE's property checks through the `forward/` adapter (§8) and wraps them in
comparative studies. Dependency direction stays FAT → FEE, never the reverse.

This also gives §2.4 ("probe before you build") an institutional form. The scar — days building
a tool before discovering the plugin did not behave as assumed — is exactly what a Contour B
gate catches, automatically, before any study is written against a malformed transform.

Current coverage is measured in [`docs/method-catalogue.md`](docs/method-catalogue.md) Part 4.
The short version: Contour A is well covered; Contour B is largely absent, and where its concerns
appear at all they appear as *constraints applied during fitting* rather than *properties tested
on the transform*. That is why a badly-formed transform can pass every check that exists today —
the checks assume well-formedness rather than verifying it.

---

## 2. Design principles

These govern every decision below. Where a later section conflicts with this one, this one wins.

### 2.1 Ceremony verifies the procedure, not the answer

A hash chain proves the pipeline ran on the bytes it claims. It cannot tell you the mechanism
under test was barely working. Those are different questions, and the elaborate machinery has
been answering the cheap one while feeling like it answered the expensive one.

That is not merely wasted effort — it is **actively harmful**, because an elaborate, formal,
hard-to-reproduce procedure invites unearned confidence in its result. The system's own history
is the proof: the full apparatus never revealed that the underlying approach worked only under
limited conditions.

**Complexity is admitted only when it answers a specific question. It is never treated as
evidence that an answer is correct.**

### 2.2 Speed is the primary feature

The system exists to compare alternatives, change hypotheses quickly, and find directions worth
exploring. It is not a laboratory. Most results need to identify a *tendency*, not establish a
fact to publication standard.

Assembling "compare this hypothesis on skin tones, on foliage, on neutrals" should take minutes,
not half a day of reshaping data to fit a pipeline.

### 2.3 Cheap independent checks beat expensive internal proof

What actually catches a broken mechanism is a second, different route to the same answer:

- **null / negative control** — run the test where the answer must be "no effect". If it fires,
  the test is broken. This is the single highest-value check available and it is nearly free.
- **trivial baseline** — does a naive method do as well? Then the sophisticated one isn't earning
  its keep.
- **holdout** — does it survive data it wasn't fitted on?
- **perturbation** — does the conclusion survive small changes to corpus or parameters?

All four are fast. All four are more informative than any certificate chain. All four already
exist in the legacy code, scattered and unsystematised (null controls in 4 modules, holdout in
10+, effect sizes in 5+). Promoting them to first-class costs little and is most of the win.

### 2.4 Probe before you build

There is a scar here worth encoding: days spent building a tool, only to find the plugin did not
behave according to the assumptions the test was built on. The tool worked. The exercise was
wasted.

**Every study declares its assumptions about the system under test, and those assumptions
execute first.** Cheap, fast, and they abort with a clear message before the expensive work
starts. This is a required field, not a convention.

### 2.5 Rigour is opt-in and escalating

Simplification does not mean everything must stay simple. It means the default is cheap and
rigour is *added where a specific question needs it* — never charged upfront to every experiment.

### 2.6 Honesty by label, not by proof machinery

FEE already solved this the cheap way: evidence classes and an explicit claim level, so the
engine "never collapses these into a generic measured film profile". FAT mirrors that. A result
carries its tier and its claim; it does not carry a certificate. A label costs one field and is
read every time; a proof chain costs thousands of lines and is read never.

---

## 3. The evidence for §2

Ceremony share by module — keyword-based, so indicative rather than exact. The two clean modules
serve as the measure's own null control, which is why the top of the table is signal:

| Module | def-LOC | verification / ceremony |
|---|---:|---:|
| `reference_vector_tile_stat2a` | 21,143 | **81.9%** |
| `reference_colour_sample_pack` | 1,959 | **54.0%** |
| `reference_statistical_colour_report` | 3,208 | 35.8% |
| `reference_colour_features` | 273 | 20.1% |
| `reference_frame_population` | 318 | 8.2% |

Two consequences:

- The 23k-LOC monolith is roughly **17k LOC of ceremony around ~4k of statistics**. That phase
  becomes *salvage the statistics, drop the shell* — not *decompose a monolith*.
- The module that builds the sample table — the one thing that must be fast for §2.2 — spends
  **over half** its surface on `source_adapter_ledger_verified`,
  `ledger_matches_live_transform_contract`, `render_provenance_verified`,
  `require_source_adapter_canonical_domain`. That is the half-day, located.

The substrate worth keeping is already clean: `colour_feature_arrays()`, `hue_sector_indices()`,
`fixed_cell_arrays()`, `bin_indices()` in `reference_colour_features`, and
`reference_frame_population` at 8%.

---

## 4. What the fast path looks like

The target ergonomics for §2.2's example:

```python
corpus = samples("pulp_fiction/scenes_top10")  # sampled once, cached by input hash

skin = corpus.where(region="skin")
foliage = corpus.where(hue_sector="green", saturation__gt=0.15)
neutral = corpus.where(saturation__lt=0.05)

for name, cohort in [("skin", skin), ("foliage", foliage), ("neutral", neutral)]:
    r = compare(
        cohort,
        baseline=current_look,
        candidate=safe_tone_v1,
        metric=hue_drift,
        control=null_control.shuffled_labels,
    )
    print(r.summary())
```

```
skin      hue_drift  -0.021  (spread 0.014, n=4,812)   null: -0.001 ✓   tier: comparison
foliage   hue_drift  +0.004  (spread 0.019, n=1,203)   null: +0.002 ✓   tier: comparison
neutral   hue_drift  -0.038  (spread 0.008, n=9,447)   null: -0.000 ✓   tier: comparison
```

Three properties this has and the current system does not: **sample once, select many times**
(the corpus is a tidy table, not a per-hypothesis pipeline); **the null control is in the default
output**, not an opt-in extra; and **the report shows effect size and spread**, not a verdict —
because a pass/fail is exactly what hides "barely working".

---

## 5. Source data, catalogue, and regeneration

### 5.1 The current strategy, and why it fails

Today's model is **extract-and-store**: pull frames out of a source, keep them forever, derive
from them, keep that too. It has produced two problems, both now measured.

**Disk.** Of ~82 GB held in the legacy repo, roughly **79 GB is regenerable derived data**:

| | Size | Status |
|---|---:|---|
| `findings/reference_vector_tile_stat2a` | **31 GB** — 91% of `findings/` | derived; ~19 GB is `rasters/` (50 MB float64 full-frame renders); ARCHIVE-bound campaign |
| `film_samples/pulp_fiction` | **48 GB** — 12 scenes | derived; one scene is 4.6 GB of 3840×2160 rgb48le TIFF |
| rest of `findings/` | ~3 GB | mostly small tables — keep |
| `~/Movies/Pulp.Fiction.1994.mkv` | 89.7 GB | **source — irreplaceable** |
| `~/Movies/samples*`, `obs_samples` (SLOG3) | ~7.6 GB | **source — irreplaceable** |

The irreplaceable material lives *outside* the repo; the repo holds the disposable copy.

**Lost basis.** `film_samples/pulp_fiction/scene_001_.../manifest.json` records its source as
`/Users/dpakhomov/Movies/Pulp Fiction/Pulp.Fiction.1994.Hybrid.2160p.UHD.Blu-ray.Remux.DV.HDR10P.HEVC.DTS-HD.mkv`.
That path is dead. The file is at `~/Movies/Pulp.Fiction.1994.mkv` — moved and renamed. It was
identified only by matching the recorded probe block (hevc / Main 10 / 3840×2160 / yuv420p10le /
24000/1001). Recovered by luck, not by record. The same manifest records `scene_directory` under
the repository's *former* name. **Paths rot; content does not.**

### 5.2 The model: catalogue and regenerate

Keep the source. Describe it. Extract on demand. Delete freely.

1. **Source registry.** The irreplaceable inputs — film remuxes, camera SLOG3 shoots, artefact
   charts. **Identified by content hash, not path.** The registry records hash, size, probe
   metadata, and last-known locations as *hints*. A moved or renamed source is re-identified
   automatically; the §5.1 failure becomes impossible.
2. **Catalogue.** The meta-database of structures within each source: scenes with time ranges,
   per-scene metrics and labels — dark, bright, faces, skin, neutral, grain, halation. Small,
   text, version-controlled.
3. **Selection is a query, not a folder.** "Scenes where `skin_tone_candidate` is above threshold
   and `face_count ≥ 1`" resolves against the catalogue. Cohorts are named queries, so they are
   reproducible and comparable across studies.
4. **Extraction recipe.** A declarative, hashable spec: source hash, time range, frame indices,
   pixel format, crop, scale. Deterministic — same recipe against the same source yields the
   same bytes.
5. **Self-validating extraction.** Re-extraction verifies against a recorded digest of the
   *extracted result*. This is what makes deletion safe: the tool can prove it got the same
   picture back before anything depends on it.
6. **Lifecycle: materialise → analyse → evict.** Derived data is a **cache**, not an archive.
   Everything derived carries the recipe that regenerates it.

This is the third and best use of content hashing. §7 already repurposed it from *proof* to
*cache key* and *change detection*; regeneration identity is where it genuinely earns its place.
It is the one piece of the old ceremony that pays for itself, and §5.1 is the evidence.

### 5.3 Retention tiers

| Tier | What | Policy |
|---|---|---|
| **source** | remuxes, camera shoots, charts | keep forever; may live offline/external; identity is the hash |
| **catalogue** | scenes, metrics, labels, cohort queries | tiny; in-repo, version-controlled |
| **recipes** | extraction specs, study declarations | tiny; in-repo |
| **tables** | sample packs (~12 MB/pack), metric CSVs | keep — small and expensive to recompute |
| **materialised frames** | extracted TIFF/PNG sequences | cache; evict by size × staleness |
| **rasters** | full-frame renders | cache; evict aggressively |

Regeneration is not free — decoding 120 UHD frames costs real time — so the policy is not
"delete everything". It is: know precisely what is evictable, and evict the large and stale.
The 12 MB tables stay; the 19 GB of rasters do not.

### 5.4 What already exists

Roughly 80% of the model is present in the legacy code and needs promoting, not inventing:

- `scene_catalog/scoring.py` already scores and thresholds eight categories —
  `grain_candidate`, `halation_candidate`, `bloom_candidate`, `mtf_candidate`, `color_candidate`,
  `shadow_candidate`, `skin_tone_candidate`, `neutral_surface_candidate`.
- `scene_catalog/face_scout.py` records per-scene `face_count`, `best_face_area_ratio`,
  `max_detection_score` in a cheap catalog-wide pass.
- `manifest.json` already records source file, start, duration, frame rate, pixel format, frame
  count and the full source probe block — the extraction recipe, minus a content hash.
- Sample packs already carry `source_frame_index`, `source_pixel_index`, `stratum_id`,
  `sample_weight` — every row already knows which pixel of which frame it came from.

Missing: identity by hash rather than path; extraction driven *by* catalogue queries; verified
regeneration; and any eviction at all.

### 5.5 Sequencing

Not urgent, but it shapes `capabilities/source/` and `sample/`, so it must not be retrofitted.
The fast path (§4) reads through a **minimal** catalogue query from day one — hash-identified
sources plus the existing `scene_catalog` — so the old folder-oriented model is never baked in.
The full registry, verified extraction and eviction machinery follow as their own phase.

## 6. The escalation ladder

| Tier | Name | What it costs | What it may claim |
|---|---|---|---|
| 0 | **probe** | minutes, disposable, no artifacts | "the mechanism responds / does not respond" |
| 1 | **comparison** | default tier; null control required | "a tendency, on this cohort" |
| 2 | **study** | + holdout, + perturbation, + retained artifacts | "holds beyond the fitted data" |
| 3 | **frozen** | + full provenance, reproducible from pin | "an authority other work may build on" |

Tier 3 is where the surviving ceremony lives, and it is reserved for results that become
authorities — a promoted profile, a parity baseline. Nothing reaches tier 3 by default, and a
study may not claim more than its tier supports. That rule is the honest, one-field version of
what the Engine V1 machinery was groping toward.

---

## 7. What to keep and drop from Engine V1

`reference_calibration_contract_v1` (12 modules, 3,285 LOC) already specifies ~40 wire schemas.
Its consumer packages — `reference_calibration_{engine,coordinator,plugins,studies,verifier}_v1`
— are **1-line empty stubs**. The skeleton was laid out and never filled, which is itself
evidence that the design was too heavy to complete.

Separately, `reference_pipeline.py` (5,246 LOC) is a *working* manifest-driven runner with state,
resume, and human-in-the-loop gating. Its weakness is diagnostic: it drives stages by building
argv and shelling out, because every analyzer is an argparse program. **The argparse-everywhere
problem is what forced the orchestrator into subprocess design.**

**Keep — repurposed from proof to utility:**

| Mechanism | New purpose |
|---|---|
| input content hash | **cache key** — sample once, reuse across hypotheses (this is what makes §4 fast) |
| input content hash | **change detection** — "did the corpus move between these two runs?" |
| lightweight artifact record | what was produced, from which inputs, at which tier |
| cohort/population identity | so two results can say "same corpus" without re-deriving it |
| `reference_pipeline`'s state + resume semantics | proven; keep the behaviour, drop the argv layer |

**Drop:**

`verification_profile` · `verification_request` · `verification_report` ·
`verification_certificate` · nonces · `publication_record` · `publication_pointer` ·
`publication_evidence_closure` · `retention_closure` · `migration_equivalence_proof` ·
`recipe_migration_claim` · `recipe_migration_record` · `producer_plugin_declaration` ·
`verifier_plugin_declaration` · `ordered_task_key_sha256` · `task_identity_set_sha256` ·
`scientific_result_set_sha256` · task ordinal contiguity proofs.

**Collapse:** `execution_binding` + `fixture_projection` + `run_request` → one run config;
`task_manifest` + `task_plan` + `execution_plan` → one plan; `task_checkpoint` + `attempt_event`
+ `resume_ledger` + `result_ledger` → one run journal.

Roughly 40 schemas become about 6. Anything on the drop list can return later **for a specific
question**, at tier 3 only.

---

## 8. Structure

```
src/film_analysis_tools/
  core/            errors, typed IO, Workspace/output roots, parallel + thread control
  capabilities/    the reusable framework — composable, no CLI, no process exit
    source/        ffmpeg decode, frame extraction, scene info, input-domain intake
    sample/        the tidy sample table, cohorts, selectors, mining, curation
    detect/        face/skin providers, masks, attrition, suitability
    measure/       statistics, PSD/noise, patch grids, bucketing
    stock/         sensitometry, density models, datasheet extraction
    colour/        colour features, adapters, renderers, tile classification
    statistics/    the tests themselves + controls: null, baseline, holdout, perturbation
    fit/           parameter specs, calibration fits, behaviour vectors, cross-validation
    report/        summaries that lead with effect size and spread
  forward/         ForwardModel protocol + adapters (plugin runtime, FEE, synthetic)
  evidence/        content hashing as cache key; lightweight artifact records
  studies/         one package per declared study
  runner/          plan, dispatch, checkpoint, resume — in process
  cli/             every argparse parser; the only layer that may exit
```

### The study contract

```python
@dataclass(frozen=True)
class Study:
    id: str
    question: str  # the hypothesis, one sentence
    rationale: str  # why it exists; what prompted it
    assumptions: tuple[Assumption, ...]  # §2.4 — execute FIRST, abort on failure
    cohort: CohortSpec  # a selector over the sample table
    method: tuple[Step, ...]
    controls: tuple[Control, ...]  # null control required at tier >= 1
    tier: Literal["probe", "comparison", "study", "frozen"] = "comparison"
    falsified_by: str = ""  # what result would overturn this
    supersedes: tuple[str, ...] = ()
```

`assumptions`, `controls` and `falsified_by` are the anti-false-confidence fields; `rationale`
and `supersedes` are what stop the next round of accretion. All are cheap. None require a
certificate.

A tier-0 probe needs only `question` and `method` — that is the point.

---

## 9. Survey and triage

**Structural debt.** `argparse.Namespace` threaded through library primitives
(`core.decode.working_frame_from_u16(args, scene, frame)`); `SystemExit` as the error channel in
134/221 modules; argparse inside 132/221; hardcoded output paths in 90 (`findings/`), 59
(`calibration/`), 25 (`native/`); `parents[N]` in 39; 11 modules importing executable code out of
the `calibration/` **artifacts** tree, which has no `__init__.py`. `core.contract` — in-degree
**153** — is 107 lines of JSON/CSV IO.

**The product seam is narrow.** A static cut of the full dependency graph yields **26 violating
edges**, **17 pointing at three modules** (`render_grain_runtime`, `runtime_looks`,
`additional_layers`). `additional_layers.py` does two jobs — sensitometry (FAT's) and image
transforms (the plugin's) — and must be split, not moved.

**Triage is four-way, and settled.** [`ledger.toml`](ledger.toml) carries the disposition of
every legacy module with the reason for it, checked in CI by `tests/migration/test_ledger.py`.

| Disposition | Modules | LOC | Meaning |
|---|---:|---:|---|
| **keep** | 126 | 145,786 | migrates here, restructured on the way in |
| **archive** | 56 | 58,685 | one-shot campaign work; documented, then left readable at the pin |
| **plugin** | 22 | 27,867 | stays in `obs-art-plugin` |
| **remove** | 14 | 814 | dead scaffolding |
| total | **218** | **233,152** | |

The keep figure is an upper bound: retained volume falls well below it once §3's ceremony share
is removed. The 14 removals are the legacy CLI/registry plumbing (entry point, analyzer registry,
and the 657-line HOOKS table mapping 92 stage names onto argparse programs — the inventory it
encoded is now the ledger), two ~13-line re-export shims, and the eight one-line `__init__.py`
stubs of the Engine V1 consumer layer that was designed and never implemented. Nothing is
deleted from history; all of it stays readable at the pin.

**Archaeology is bounded.** 183 docs; **142** name a module; **111 / 164** modules are named in a
doc. Binding must be by content — docs use campaign codenames (`stat2a`, `ptf3`, `vt6`), modules
use descriptive names. The writing backlog is **19 records for KEEP, 20 for ARCHIVE**. The
`campaign.joint` arc (8 modules) has *zero* documentation and goes first, while the reasoning is
still recoverable.

---

## 10. Formalization rules

Enforced by architecture test. These serve manageability, not separation.

| # | Rule | Why |
|---|---|---|
| 1 | No `argparse` outside `cli/` | the reason the orchestrator had to shell out; blocks composition |
| 2 | No `SystemExit` outside `cli/` | library code must not kill its host |
| 3 | No artifact-directory string literals | 90 modules currently run in one repo layout only |
| 4 | No `parents[N]` path walking | behaviour must not depend on install location |
| 5 | Declared layer order; nothing imports `cli/` | keeps `studies/` composing `capabilities/` |
| 6 | Emulation-model imports confined to `forward/` | one seam against plugin drift, not twenty |
| 7 | A study may not claim above its tier | §2.6 — the one rule replacing the certificate chain |

---

## 11. Phases

The plan follows its own advice: a thin working fast path early, before heavy migration.

**P0 — Bootstrap.** `git init`, uv + Python 3.12, ruff/mypy/pytest, CI, `AGENTS.md`,
`ARCHITECTURE.md`. Layer packages with rules 1–7 enforced from the first commit.

**P1 — The fast path.** §4, end to end on real data: sample table, cohort selectors, `compare`,
null control, effect-size summary, CLI. Built fresh on the clean legacy substrate
(`colour_features`, `frame_population`), deliberately thin. Resolves packs by **name through a
workspace**, never by caller-supplied folder path (§5.5), so the folder-oriented model is never
baked in. **This is the product; everything after it is migration into a thing that already
works.**

Sequenced ahead of triage deliberately. It does not depend on triage, it delivers the speed that
is the point of the repo, and *using* it is what teaches us which legacy code is worth keeping —
which is §2.4 applied to the migration itself.

**P2 — Triage.** Dispositions only — KEEP / ARCHIVE / REMOVE / plugin, machine-checked in
`ledger.toml`. Cheap and mechanical, and better informed for following P1. The 39 written records
are *deferred and parallelised* across later phases so the ledger doesn't block working code.
One exception is pulled forward: the `campaign.joint` archaeology (8 modules, zero
documentation), because that reasoning is perishable in a way the rest is not.

**P3 — Core foundation.** Harden what P1 established: `errors`, typed IO (replacing
`core.contract`, in-degree 153), and `Workspace` gaining a **write root** — the concrete
replacement for 90 hardcoded `findings/…` literals and 39 `parents[N]` root-walks. Read and
write roots are separate so nothing writes into the corpus.

**Parallel/thread control was deliberately not built.** The legacy equivalent
(`core/native_threads.py`, in-degree 44) capped BLAS threads by setting environment variables at
*module import time*, which only works before NumPy is imported — a global side effect that is
part of why those modules could not be used as libraries. There is nothing to parallelise here
yet, so a worker pool would be speculation (§2.2). When a parallel workload arrives, the
environment capping belongs in `cli` as a pre-import concern and worker-plan resolution belongs
in `core` as a pure function.

What P1 *did* turn up, which no upfront design would have: the permutation null allocated the
whole `resamples × n` matrix, peaking at **1.25 GB** on a 260k-row cohort and growing linearly
with the corpus. Resampling is now batched to a memory budget — 189 MB and flat — and because a
batched draw consumes the generator in the same order, results are bit-identical. Writing
non-finite metrics as JSON was also emitting bare `NaN` tokens, which Python reads back happily
and stricter parsers reject; they are now `null`.

**P4 — Controls and tiers.** Promote the scattered null/baseline/holdout/perturbation code to
first-class `capabilities/statistics/`. Implement the tier ladder and rule 7. Permutation/null
control is the weakest existing primitive (3 modules) and the highest-value fix.

**P4b — Contour B gates (in FEE, §1.1).** A companion phase in the engine repo, not FAT: the
synthetic-probe suite from `docs/method-catalogue.md` Part 4 — ramp/gradient smoothness, signal-
space Jacobian (`det` sign for folding, SVD conditioning for channel collapse), highlight and
shoulder probe, gamut boundary sweep, 3D LUT cell validity, clipping accounting. Runs per-commit
on synthetic input; no corpus. FAT later consumes these through `forward/` for comparative
studies. Sequenced early because it protects everything after it.

**P5 — Behaviour lock.** Goldens at the legacy pin, for KEEP modules only, before they move.

**P6 — Source registry and catalogue (§5).** Hash-identified source registry; `scene_catalog`
promoted to first-class and generalised beyond one film; cohort queries; the self-validating
extraction tool with verified round-trip; retention tiers and eviction. Reclaims the ~79 GB and
makes it structurally impossible to lose a sample's basis again.

**P7 — Measurement capabilities.** `measure/`, `sample/`, `detect/`. De-Namespacing happens here.

**P8 — Forward model.** Protocol + adapters; split `additional_layers`; move `ffmpeg_bin` out of
the renderer. Resolves all 22 KEEP→plugin violations.

**P9 — `stock/`, `colour/`, `fit/`.** Promote `controlled_ramp_candidate` into `colour/` as real
library code; the artifacts tree stops being importable.

**P10 — Statistics salvage.** Extract the ~4k LOC of real statistics from STAT2A's 21k; drop the
verification shell per §7. Salvage, not decomposition. Goldens hold throughout.

**P11 — Studies backfill, records, CLI.** Express surviving pipelines as `Study` declarations;
finish the 39 records; single entry point. Legacy `mediachar/` referenced only as a pin.

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| Ceremony creeps back in | rule 7 + tier ladder; drop-list items return only for a named question at tier 3 |
| Cleanup silently changes numbers | P5 goldens before any KEEP module moves |
| The fast path stays a toy | P2 ships on real footage with a real cohort comparison, not fixtures |
| Simplification loses a genuinely needed check | nothing on the §7 drop list is deleted from history; it is reachable at the pin |
| Evicted data turns out not to be regenerable | eviction requires a *verified* round-trip first (§5.2); source registry keyed by hash, not path |
| A source goes missing or offline | registry records hash + probe metadata, so re-identification survives moves and renames; sources may live external |
| Regeneration cost makes eviction a false economy | tiered retention (§5.3): small expensive-to-recompute tables stay; large cheap-to-redraw rasters go |
| Archaeology outruns its value | 39 records, parallelised; REMOVE requires a record, not an opinion |
| STAT2A salvage discards real statistics with the shell | the 4k/21k split is verified module-by-module against goldens, not assumed from §3's keyword estimate |
| The system re-accretes | `rationale`, `supersedes`, `falsified_by` are required fields, checked in CI |

---

## 13. Open questions

- **`reference_obs_colour_contract_probe`** — product concern, or `capabilities/source/`? Two
  migrating SLog3 probes depend on it.
- **`display_camera_bench`** (5,005 LOC) — generator *and* analyzer; likely a split.
- **Sample table representation** — plain numpy record arrays keep dependencies minimal; a
  dataframe library would make §4's selectors more expressive. Worth deciding at P2, not before.
- **CLI name** — `film_analysis_tools`; command still open (`fat` is unfortunate).
