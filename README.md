# Film Analysis Tools

Measurement, fitting and validation tooling for film emulation work.

This is the analysis half of a three-repository system:

| Repo | Owns |
|---|---|
| `film-emulation-engine` | the forward model — what a look *does* to an image, and whether that transform is well-formed |
| **`film-analysis-tools`** | measurement, fitting, evidence — what real footage *is*, and which parameters reproduce it |
| `obs-art-plugin` | the shipping OBS product, and shader-to-model parity |

It has its own development path because the validation system outgrew the plugin it validates —
221 modules and 233k LOC in its legacy form, grown by accretion. Independence here means
**development** independence: separate repo, tests, cadence and review bar. It is not standalone,
and stays closely and appropriately dependent on the engine and the plugin.

## What it is for

**Speed first.** The system exists to compare alternatives, change hypotheses quickly, and find
directions worth exploring. Assembling "compare this hypothesis on skin tones, on foliage, on
neutrals" should take minutes.

```python
from film_analysis_tools.capabilities.colour import transforms
from film_analysis_tools.capabilities.sample import cohorts, load_pack
from film_analysis_tools.capabilities.statistics import compare_cohorts
from film_analysis_tools.core import Workspace

pack = load_pack("deployment_baseline_sample_pack", workspace=Workspace.from_env())
selected = cohorts.build(pack, ("neutral", "skin_like", "foliage_like"))

for result in compare_cohorts(
    selected,
    baseline=transforms.named("identity"),
    candidate=transforms.named("warm_gain"),
    metric="hue_drift",
):
    print(result.summary())
```

or the same thing from a shell:

```bash
film-analysis compare my_pack --candidate warm_gain --metric hue_drift --cohorts neutral,skin_like,foliage_like
```

```
cohort            metric        effect     |move|    spread        n      null      p  verdict
-----------------------------------------------------------------------------------------------
neutral           hue_drift   +54.4280 deg 117.3641  155.1099    7,515  -0.0017  0.005  moves, no net direction
skin_like         hue_drift    -1.7848 deg   2.5365    2.4565    2,693  +0.0054  0.005  clear of null
foliage_like      hue_drift   -12.9319 deg  12.9319    5.4241   25,896  -1.7111  0.005  clear of null

tier: comparison — a tendency on these cohorts, not a fact beyond them
NOTE: samples move but cancel out in ['neutral'] — the effect has no net direction here.
```

260,000 real samples, five cohorts, 200 permutation resamples each: **0.5 seconds**.

Three things that output does deliberately. The **null control always runs** — a comparison
that cannot show what "no effect" looks like is not evidence. Results lead with **effect size,
spread and sample count**, never a bare verdict, because a pass/fail is the shape that hides a
marginal mechanism. And it separates *no effect* from *effect with no net direction*: hue is
numerically meaningless near the achromatic axis, so a naive harness reports a confident-looking
`+54°` drift on neutrals. This one reports that the samples do not agree, and says why.

**Rigour is opt-in.** Four tiers — `probe`, `comparison`, `study`, `frozen` — and a result may
not claim above its tier. That one rule replaces the verification ceremony this system
accumulated in its previous life.

## Status

**Camera sample catalogue.** A clean, queryable index of 105 Sony ZV-E10 II clips across eleven
validation categories — exposure extremes, saturated practicals, low saturation, difficult
shadows, motion and noise, and skin under four lighting conditions. Built by re-measuring the
original camera files; it carries no dependency on the legacy catalogues, which were read once to
recover which sources existed and then discarded. Clips are identified by content hash, so a
renamed source is re-found rather than lost. See
[`docs/camera-corpus-recovery.md`](docs/camera-corpus-recovery.md).

```bash
film-analysis catalogue                                              # taxonomy and counts
film-analysis catalogue deep_underexposure saturated_practical --all # overlaps
film-analysis catalogue skin_mixed --paths --verify                  # resolve, hash-checked
```

**P3b — analytical reporting.** `--save NAME` now also writes a self-contained `report.html`:
tone and hue response curves, per-cohort metric distributions with the null band shaded, cohort
coverage maps, and before/after colour swatches. Hand-rolled SVG, so the dependency set stays at
NumPy alone; one file, no assets, no network, light and dark themes.

The hue response curve is the one worth looking at first — it plots drift against input hue and
states whether the curve changes sign, which is what turns "the median was zero" into an
explanation rather than a puzzle.

**P3 — core foundation.** Typed IO that raises instead of exiting, writes atomically, and never
coerces silently; a `Workspace` with separate read and write roots, so results are addressed by
name and nothing writes into the corpus. Two defects found by having built P1 first: the
permutation null allocated the full `resamples × n` matrix and peaked at **1.25 GB** on a
260k-row cohort — now batched to a budget, **189 MB and flat**, with bit-identical results — and
non-finite metrics were written as bare `NaN`, which is invalid JSON that Python reads back
happily and other parsers reject.

Parallel and native-thread control was deliberately not built; there is nothing to parallelise
yet, and the reasoning is recorded in `core/__init__.py` and the plan.

**P2 — triage complete.** [`ledger.toml`](ledger.toml) carries the disposition of all 218 legacy
modules with the reason for each: **keep 126** (145,786 LOC), **archive 56** (58,685),
**plugin 22** (27,867), **remove 14** (814). CI checks that every module is claimed exactly once,
that no disposition lacks a reason, that removals carry a justification, and — with a legacy
checkout available — that the ledger still covers the legacy tree. Archive records are tracked by
a ratchet, so a campaign cannot be archived unrecorded. The first record,
[`docs/archive/campaign-joint.md`](docs/archive/campaign-joint.md), covers the arc that had zero
documentation.

**P1 — the fast path, working.** Sample tables with cohort selectors, analytic transforms,
colour metrics, permutation null controls, comparison reporting, and a CLI. Verified against a
real 260k-row pack, including a cross-check that recomputed colour features reproduce the ones
the legacy pipeline stored.

Rigour is declared but not yet enforced beyond `comparison`: holdout and perturbation controls
arrive in P4. Cohorts are colour-defined — `skin_like` names the region skin usually occupies
and does not know what skin is; real detection arrives in P7. Emulation models arrive as
`forward/` adapters in P8, satisfying the same `Transform` protocol the built-ins already use.

Done: **P0** bootstrap — layer skeleton, tier vocabulary, seven formalization rules enforced
from the first commit. See [`MIGRATION_PLAN.md`](MIGRATION_PLAN.md) section 11 for what lands
when.

## Development

```sh
uv sync --group dev
uv run pytest
```

Full gate, matching CI:

```sh
uv run ruff format --check . && uv run ruff check . && uv run mypy src/film_analysis_tools && uv run pytest
```

## Documents

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — layers, the two contours, why the seams sit where they do
- [`MIGRATION_PLAN.md`](MIGRATION_PLAN.md) — design principles, survey, triage, phases
- [`docs/method-catalogue.md`](docs/method-catalogue.md) — statistical methods, experiments and campaigns already built
- [`AGENTS.md`](AGENTS.md) — contributor and agent rules
