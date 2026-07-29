"""Where data is read from and where results are written — both resolved, never hardcoded.

Datasets are addressed by *name*. The workspace maps a name to a location. That indirection is
what keeps the folder-oriented model out: no module names an artifact directory, and moving the
data edits no code.

This is the concrete replacement for the legacy pattern that put ``findings/…`` string literals
in 90 modules and walked ``parents[N]`` to find the repository root in 39, which together meant
the tools ran correctly only from inside one checkout.

Read and write roots are separate on purpose. Sources are large, external and read-only;
results are small and belong wherever the operator wants them. Nothing writes into the corpus.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from film_analysis_tools.core.errors import WorkspaceError

ENV_VAR = "FILM_ANALYSIS_WORKSPACE"
OUTPUT_ENV_VAR = "FILM_ANALYSIS_OUTPUT"
DEFAULT_OUTPUT_DIRNAME = "results"


def _contained(root: Path, name: str, *, kind: str) -> Path:
    """Resolve ``name`` under ``root``, refusing anything that climbs out."""
    if not name or name.startswith("/"):
        raise WorkspaceError(f"{kind} name must be relative and non-empty: {name!r}")
    candidate = (root / name).expanduser()
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(root.expanduser().resolve(strict=False))
    except ValueError as exc:
        raise WorkspaceError(f"{kind} name escapes the workspace: {name!r}") from exc
    return resolved


@dataclass(frozen=True)
class Workspace:
    """A read root holding named datasets, and optionally a write root for results.

    The write root may be absent: plenty of work only reads. Asking for an output path
    without one configured raises rather than guessing a location, because guessing is how
    tools end up writing into someone's corpus.
    """

    root: Path
    output_root: Path | None = None

    @classmethod
    def from_env(
        cls,
        override: Path | str | None = None,
        output_override: Path | str | None = None,
    ) -> Workspace:
        """Resolve both roots from arguments, then the environment.

        The read root has no default: guessing where someone's corpus lives is how tools end
        up hardcoding paths. The write root falls back to ``./results`` under the working
        directory, which is explicit enough to be predictable and harmless if wrong.
        """
        if override is not None:
            root = Path(override)
        else:
            raw = os.environ.get(ENV_VAR)
            if not raw:
                raise WorkspaceError(
                    f"no workspace configured: pass one explicitly or set {ENV_VAR}"
                )
            root = Path(raw)

        if output_override is not None:
            output_root = Path(output_override)
        else:
            output_root = Path(
                os.environ.get(OUTPUT_ENV_VAR) or Path.cwd() / DEFAULT_OUTPUT_DIRNAME
            )
        return cls(root=root.expanduser(), output_root=output_root.expanduser())

    # ------------------------------------------------------------------- reading

    def resolve(self, name: str) -> Path:
        """Resolve a dataset name to an existing directory inside the read root."""
        resolved = _contained(self.root, name, kind="dataset")
        if not resolved.is_dir():
            raise WorkspaceError(f"dataset {name!r} not found under {self.root.expanduser()}")
        return resolved

    def names(self, pattern: str = "*") -> list[str]:
        """Dataset names available for reading, for discovery and error messages."""
        root = self.root.expanduser()
        if not root.is_dir():
            raise WorkspaceError(f"workspace root does not exist: {root}")
        return sorted(str(path.relative_to(root)) for path in root.glob(pattern) if path.is_dir())

    # ------------------------------------------------------------------- writing

    def output(self, *parts: str, create: bool = True) -> Path:
        """A path under the write root. Parent directories are created by default.

        ``workspace.output("skin_headroom_v1", "summary.json")`` is how a study names its
        artifacts. It never names a directory that exists only in one repository.
        """
        if self.output_root is None:
            raise WorkspaceError(
                f"no result root configured: pass output_root or set {OUTPUT_ENV_VAR}"
            )
        if not parts:
            raise WorkspaceError("output() needs at least one path component")
        resolved = _contained(self.output_root, "/".join(parts), kind="output")
        if create:
            try:
                resolved.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise WorkspaceError(f"cannot create output directory: {exc}") from exc
        return resolved

    def describe(self) -> dict[str, str]:
        """Both roots, for recording alongside results so a run can be located later."""
        return {
            "read_root": str(self.root.expanduser()),
            "write_root": str(self.output_root.expanduser()) if self.output_root else "",
        }


__all__ = ["DEFAULT_OUTPUT_DIRNAME", "ENV_VAR", "OUTPUT_ENV_VAR", "Workspace"]
