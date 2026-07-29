# Architecture

Film Analysis Tools measures real footage, fits parameters, and reports what it found. It
separates **the reusable framework** (capabilities) from **declared units of validation work**
(studies), and keeps every command-line concern in one leaf layer.

## Two axes

The legacy system had only a library axis, which is why the question a piece of code existed to
answer lived in a document nothing linked to.

- **Capabilities** — composable measurement components. Libraries, not programs.
- **Studies** — declared work: question, rationale, assumptions, cohort, method, controls, tier,
  what would falsify it, what it supersedes.

Studies compose capabilities. Capabilities never compose studies.

## Layers

Strictly downward. A module may import from its own layer or one below it, never above.

```
cli            argument parsing, process exit — the only layer that may do either
runner         plan, dispatch, checkpoint, resume (in process)
studies        declared validation units
forward        adapters onto emulation models — the only layer that may import one
capabilities   source · sample · detect · measure · stock · colour · statistics · fit · report
evidence       content identity, artifact records, provenance
core           errors, typed IO, workspace, parallel control, shared protocols
```

Enforced by `tests/architecture/test_layers.py`, from the first commit.

### Why the `ForwardModel` protocol lives in `core`, not `forward`

`capabilities.fit` needs the *abstraction* of a forward model, but must not depend on any
concrete adapter — otherwise fitting cannot be tested without an engine build. Putting the
protocol in `core` and the adapters in `forward` keeps the layer order intact and lets fits run
against a synthetic known-answer model with no engine and no footage.

### Why `forward` is a single layer

It is the only place permitted to import an emulation model. That is a maintenance boundary, not
a wall: this layer is *expected* to depend on the engine and the plugin runtime. Confining it
means one seam to keep in step with those repositories rather than twenty, which is what the
legacy code had.

## Two contours of validation

"How accurately did we implement this?" is two questions, and only one of them is this repo's.

| Question | Needs | Home |
|---|---|---|
| Is the transform well-formed — monotone, no folding, no channel collapse, clean gradients, intact highlights, valid LUT cells? | transform + synthetic probes; no corpus | **film-emulation-engine** |
| Does the shipped shader implement the model? | model + native pass bodies | **obs-art-plugin** |
| Does the model match the reference? | corpus, catalogue, sampling, statistics | **this repo** |

The seam is the shape of the output: a **verdict on one model** is an engine gate that runs
per-commit on synthetic input; a **comparison across alternatives** is a study here.

## Rigour is opt-in

`core.tiers.Tier` orders four levels — `probe`, `comparison`, `study`, `frozen`. The default is
`comparison`, which requires a null control. A result may not claim above its tier, and that
single rule stands in for the verification-certificate machinery the legacy system accumulated
and never finished.

Complexity is admitted when it answers a specific question. It is never treated as evidence that
an answer is correct.

## Data

Sources — film remuxes, camera shoots, charts — live outside the repository and are identified
by **content hash, not path**. The catalogue describes structures within them; extraction is a
declarative, verifiable recipe; derived data is a cache that can be evicted because it can be
regenerated and the regeneration is checked.

Content hashing therefore serves as cache key, change detector, and regeneration identity — not
as proof.

See [`MIGRATION_PLAN.md`](MIGRATION_PLAN.md) for the full reasoning and phasing, and
[`docs/method-catalogue.md`](docs/method-catalogue.md) for what the legacy system already
contains.
