"""Where data lives — resolved by name, never by a caller-supplied folder path.

Datasets are addressed by *name*. The workspace maps a name to a location. That indirection
is what keeps the folder-oriented model out: no module names an artifact directory, and
moving the data does not edit any code (``MIGRATION_PLAN.md`` sections 5 and 10).

The root comes from the environment or an explicit argument, so nothing here is tied to one
machine or one repository layout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from film_analysis_tools.core.errors import WorkspaceError

ENV_VAR = "FILM_ANALYSIS_WORKSPACE"


@dataclass(frozen=True)
class Workspace:
    """A root directory containing named datasets."""

    root: Path

    @classmethod
    def from_env(cls, override: Path | str | None = None) -> Workspace:
        if override is not None:
            root = Path(override)
        else:
            raw = os.environ.get(ENV_VAR)
            if not raw:
                raise WorkspaceError(
                    f"no workspace configured: pass one explicitly or set {ENV_VAR}"
                )
            root = Path(raw)
        return cls(root=root.expanduser())

    def resolve(self, name: str) -> Path:
        """Resolve a dataset name to a directory inside this workspace.

        Names are relative and may not escape the root; a name that climbs out is a bug or
        an injection, never a legitimate dataset.
        """
        if not name or name.startswith("/"):
            raise WorkspaceError(f"dataset name must be relative and non-empty: {name!r}")
        candidate = (self.root / name).expanduser()
        root = self.root.expanduser()
        try:
            resolved = candidate.resolve(strict=False)
            resolved.relative_to(root.resolve(strict=False))
        except ValueError as exc:
            raise WorkspaceError(f"dataset name escapes the workspace: {name!r}") from exc
        if not resolved.is_dir():
            raise WorkspaceError(f"dataset {name!r} not found under {root}")
        return resolved

    def names(self, pattern: str = "*") -> list[str]:
        """Dataset names available under this workspace, for discovery and error messages."""
        root = self.root.expanduser()
        if not root.is_dir():
            raise WorkspaceError(f"workspace root does not exist: {root}")
        return sorted(str(path.relative_to(root)) for path in root.glob(pattern) if path.is_dir())


__all__ = ["ENV_VAR", "Workspace"]
