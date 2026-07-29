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
