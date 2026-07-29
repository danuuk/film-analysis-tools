"""Loading sample packs by name from a workspace.

A pack is one ``.npz`` of named columns per scene under ``samples/``. Packs are addressed by
name through the workspace, never by a caller-supplied folder path, so no module names an
artifact directory and moving the data edits no code.

Provenance travels with the rows: ``scene_id`` is attached during load, and packs already
carry ``source_frame_index`` and ``source_pixel_index``, so any row can be traced back to the
pixel it came from. That is the property the legacy extract-and-store approach lost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from film_analysis_tools.capabilities.sample.table import SampleTable
from film_analysis_tools.core import io
from film_analysis_tools.core.errors import DataError
from film_analysis_tools.core.workspace import Workspace

SAMPLES_DIRNAME = "samples"
MANIFEST_NAME = "sample_pack_manifest.json"


def load_pack(
    name: str, *, workspace: Workspace, scenes: tuple[str, ...] | None = None
) -> SampleTable:
    """Load a named sample pack into one table.

    ``scenes`` restricts the load to specific scene ids; omitting it loads the whole pack.
    """
    root = workspace.resolve(name)
    samples_dir = root / SAMPLES_DIRNAME
    if not samples_dir.is_dir():
        raise DataError(f"pack {name!r} has no {SAMPLES_DIRNAME}/ directory at {root}")

    paths = sorted(samples_dir.glob("*.npz"))
    if scenes is not None:
        wanted = set(scenes)
        paths = [path for path in paths if path.stem in wanted]
        missing = wanted - {path.stem for path in paths}
        if missing:
            raise DataError(f"pack {name!r} has no scenes: {sorted(missing)}")
    if not paths:
        raise DataError(f"pack {name!r} contains no scene files")

    per_scene = [_load_scene(path) for path in paths]
    shared = set(per_scene[0])
    for columns in per_scene[1:]:
        shared &= set(columns)
    if not shared:
        raise DataError(f"pack {name!r} scenes share no common columns")

    merged: dict[str, np.ndarray] = {
        key: np.concatenate([columns[key] for columns in per_scene], axis=0)
        for key in sorted(shared)
    }
    merged["scene_id"] = np.concatenate(
        [
            np.full(len(next(iter(columns.values()))), path.stem)
            for path, columns in zip(paths, per_scene, strict=True)
        ]
    )
    return SampleTable(columns=merged, name=name)


def _load_scene(path: Path) -> dict[str, np.ndarray]:
    return io.read_arrays(path)


def pack_manifest(name: str, *, workspace: Workspace) -> dict[str, Any]:
    """The pack's manifest, or an empty mapping when it has none."""
    path = workspace.resolve(name) / MANIFEST_NAME
    if not path.is_file():
        return {}
    return io.read_json(path)


def describe_pack(name: str, *, workspace: Workspace) -> dict[str, Any]:
    """Shape and provenance of a pack, without loading every column into memory twice."""
    table = load_pack(name, workspace=workspace)
    manifest = pack_manifest(name, workspace=workspace)
    return {
        "name": name,
        "rows": len(table),
        "scenes": len(set(table.column("scene_id").tolist())),
        "columns": table.column_names,
        "pack_id": manifest.get("pack_id", ""),
        "role": manifest.get("role", ""),
        "generated_at_utc": manifest.get("generated_at_utc", ""),
    }


__all__ = ["MANIFEST_NAME", "SAMPLES_DIRNAME", "describe_pack", "load_pack", "pack_manifest"]
