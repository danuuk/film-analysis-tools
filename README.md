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
corpus = samples("pulp_fiction/scenes_top10")  # sampled once, cached by content hash

skin = corpus.where(region="skin")
foliage = corpus.where(hue_sector="green", saturation__gt=0.15)
neutral = corpus.where(saturation__lt=0.05)

for name, cohort in [("skin", skin), ("foliage", foliage), ("neutral", neutral)]:
    print(
        compare(
            cohort,
            baseline=current_look,
            candidate=safe_tone_v1,
            metric=hue_drift,
            control=null_control.shuffled_labels,
        ).summary()
    )
```

The null control is in the default output, and results lead with effect size and spread rather
than a verdict — a pass/fail is exactly the shape that hides a marginal mechanism.

**Rigour is opt-in.** Four tiers — `probe`, `comparison`, `study`, `frozen` — and a result may
not claim above its tier. That one rule replaces the verification ceremony this system
accumulated in its previous life.

## Status

**P0 — bootstrap.** Layer skeleton, tier vocabulary, and the seven formalization rules enforced
from the first commit. No capabilities migrated yet; see
[`MIGRATION_PLAN.md`](MIGRATION_PLAN.md) section 11 for what lands when.

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
