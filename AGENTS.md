# Guidance for contributors and agents

Read [`ARCHITECTURE.md`](ARCHITECTURE.md) and [`MIGRATION_PLAN.md`](MIGRATION_PLAN.md) before
making changes. The plan's section 2 (design principles) overrides anything that conflicts with
it, including advice here.

## Hard rules

- **No ceremony without a question.** Do not add hashes, pointers, proofs, certificates,
  signatures, or verification-of-verification because it feels rigorous. Complexity is admitted
  only when it answers a specific question, never as evidence that an answer is correct. The
  legacy system reached ~82% verification vocabulary in its largest module and still failed to
  reveal that the approach under test barely worked.
- **A null control is not optional.** Any comparison at `Tier.COMPARISON` or above ships one. If
  a test cannot produce a case where it says "no effect", it is not informative.
- **Report effect size and spread, never a bare verdict.** A pass/fail is the shape that hides a
  marginal mechanism.
- **Probe before you build.** State what must be true of the system under test, check it first,
  and abort clearly when it is not. Days have been lost building a tool against a transform that
  never behaved as assumed.
- **Never claim above your tier.** Enforced by `tests/architecture/test_claims.py`.
- **Sources are identified by content hash, not path.** Never hardcode an artifact directory or
  an absolute path. Paths rot — a legacy manifest lost its source to a rename and was recovered
  only by matching a probe block.
- **Never delete derived data without a verified regeneration round-trip first.**
- **No argument parsing or process exit outside `cli/`.** This is what makes in-process
  orchestration possible; the legacy runner had to shell out because every analyzer was an
  argparse program.
- **Emulation-model imports stay in `forward/`.** Depending on the engine and plugin is
  expected and fine — concentrating that dependency in one seam is the point.
- **Do not** configure a Git remote, push, or select an open-source license without user
  authorization.

## Camera material comes from the catalogue, and only from the catalogue

`capabilities/catalogue` is the single route to camera samples. It was rebuilt by re-measuring the
original files; the legacy catalogues in `obs-art-plugin` — `findings/reference_camera_corpus/`
and `findings/reference_slog3_source_intake/` — are superseded and must not be read, imported, or
mirrored. They went through several generations, together covered only 74 of 105 clips, and the
older one has no content hashes at all.

```python
from film_analysis_tools.capabilities import catalogue

cat = catalogue.bundled()
for clip in cat.select("deep_underexposure", "saturated_practical", require_all=True):
    path = clip.locate()  # verified against its content hash
```

Other repositories and other agents consume it through the CLI rather than by importing:
`film-analysis catalogue <categories> --paths --verify --json`. That keeps the contract about
behaviour rather than code, and works from `obs-art-plugin` — which must not grow its own copy.

When working on it:

- **Never copy the manifest into another repository.** One source of truth. A duplicated
  catalogue drifts, and drift across generations is the exact problem this replaced.
- **A new category must name the failure it provokes**, not just describe a scene, and its
  threshold must be chosen *after* looking at the measured distribution. Absolute thresholds
  guessed in advance have failed here three times — see `docs/camera-corpus-recovery.md` §4.
- **Keep `human_labelled` honest.** Face presence is a human label until `capabilities/detect`
  exists. Anything not decided by measurement says so in the manifest.
- **Bump nothing silently.** Consumers record `catalogue_id` and `generated`; changing labels
  without changing those makes two incomparable result sets look comparable.

## Before you open a change

```sh
uv run ruff format --check .
uv run ruff check .
uv run mypy src/film_analysis_tools
uv run pytest
```

## Scope discipline

Migration order is in `MIGRATION_PLAN.md` section 11. Two habits matter most:

- **Behaviour-lock before restructuring.** Goldens at the legacy pin come first (P5); otherwise
  "cleaned up" and "broken" are indistinguishable.
- **Decompose and refactor in separate commits.** Never move code and change its logic at once.

One-shot campaign experiments are archived with a written record, not ported. Nothing is deleted
before its record exists.
