"""The catalogue: a queryable index of camera samples per validation category.

This is the clean boundary the analytical tool uses. It knows nothing about how the legacy
project organised its material — no legacy paths, no legacy metadata, no generation history.
It answers one question: *which samples exercise this condition, and where do I get them.*

Clips are identified by **content hash**. The recorded path is a hint, and
:meth:`CatalogueClip.locate` re-finds a clip that has been moved or renamed by matching its
digest — the failure that lost a legacy corpus its provenance when a source file was renamed.

The manifest itself is small metadata and lives in the repository. The media does not.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from film_analysis_tools.core.errors import DataError
from film_analysis_tools.core.io import read_json

SCHEMA_VERSION = 1
HASH_CHUNK = 4 << 20


@dataclass(frozen=True)
class CatalogueClip:
    """One camera clip, with the conditions it exercises and how to find it."""

    clip_id: str
    sha256: str
    byte_size: int
    shoot: str
    path_hint: str
    categories: tuple[str, ...]
    captured: str = ""
    duration_s: float = 0.0
    probe_times_s: tuple[float, ...] = ()
    stream: Mapping[str, Any] = field(default_factory=dict)
    measured: Mapping[str, Any] = field(default_factory=dict)
    notes: str = ""

    def locate(self, *, roots: Sequence[Path] = (), verify: bool = True) -> Path:
        """Find this clip on disk, by hint first and by content hash if the hint is stale.

        ``verify`` re-hashes the candidate. It costs a full read, so a caller streaming many
        clips may turn it off — but the default is on, because a silently wrong file is worse
        than a slow one.
        """
        candidates: list[Path] = []
        hint = Path(self.path_hint).expanduser()
        if hint.is_file():
            candidates.append(hint)
        for root in roots:
            candidates.extend(
                path for path in Path(root).expanduser().rglob(hint.name) if path.is_file()
            )
        for root in roots:
            candidates.extend(
                path
                for path in Path(root).expanduser().rglob("*")
                if path.is_file() and path.stat().st_size == self.byte_size
            )

        seen: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if candidate.stat().st_size != self.byte_size:
                continue
            if not verify or file_sha256(candidate) == self.sha256:
                return candidate

        raise DataError(
            f"clip {self.clip_id} not found; hint {self.path_hint!r} did not resolve and no "
            f"file under {[str(root) for root in roots]} matched sha256 {self.sha256[:12]}…"
        )

    def has(self, category: str) -> bool:
        return category in self.categories


@dataclass(frozen=True)
class Catalogue:
    """A queryable set of clips, grouped by validation category."""

    catalogue_id: str
    camera: Mapping[str, Any]
    decode: Mapping[str, Any]
    categories: Mapping[str, Mapping[str, Any]]
    clips: tuple[CatalogueClip, ...]
    generated: str = ""

    def __len__(self) -> int:
        return len(self.clips)

    def __iter__(self) -> Iterator[CatalogueClip]:
        return iter(self.clips)

    def category_ids(self) -> list[str]:
        return list(self.categories)

    def select(
        self,
        *categories: str,
        require_all: bool = False,
        shoot: str | None = None,
        limit: int = 0,
    ) -> list[CatalogueClip]:
        """Clips exercising the named categories.

        By default any of them matches, which is what a broad robustness sweep wants.
        ``require_all=True`` narrows to the overlaps, which is where the interesting failures
        live — dark *and* saturated, for instance.
        """
        unknown = [name for name in categories if name not in self.categories]
        if unknown:
            raise DataError(f"unknown categories {unknown}; available: {sorted(self.categories)}")
        chosen: list[CatalogueClip] = []
        for clip in self.clips:
            if shoot is not None and clip.shoot != shoot:
                continue
            if categories:
                matches = [clip.has(name) for name in categories]
                if not (all(matches) if require_all else any(matches)):
                    continue
            chosen.append(clip)
        return chosen[:limit] if limit > 0 else chosen

    def counts(self) -> dict[str, int]:
        return {name: sum(1 for clip in self.clips if clip.has(name)) for name in self.categories}

    def uncategorised(self) -> list[CatalogueClip]:
        """Clips no category claimed — ordinary material, kept rather than hidden."""
        return [clip for clip in self.clips if not clip.categories]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(HASH_CHUNK), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DataError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def load(path: Path) -> Catalogue:
    """Read a catalogue manifest."""
    payload = read_json(path)
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise DataError(f"{path}: unsupported catalogue schema_version {version!r}")

    clips = tuple(
        CatalogueClip(
            clip_id=str(row["clip_id"]),
            sha256=str(row["sha256"]),
            byte_size=int(row["byte_size"]),
            shoot=str(row.get("shoot", "")),
            path_hint=str(row.get("path_hint", "")),
            categories=tuple(row.get("categories", ())),
            captured=str(row.get("captured", "")),
            duration_s=float(row.get("duration_s", 0.0)),
            probe_times_s=tuple(float(t) for t in row.get("probe_times_s", ())),
            stream=row.get("stream", {}),
            measured=row.get("measured", {}),
            notes=str(row.get("notes", "")),
        )
        for row in payload.get("clips", [])
    )
    return Catalogue(
        catalogue_id=str(payload.get("catalogue_id", "")),
        camera=payload.get("camera", {}),
        decode=payload.get("decode", {}),
        categories={entry["id"]: entry for entry in payload.get("categories", [])},
        clips=clips,
        generated=str(payload.get("generated", "")),
    )


def bundled(name: str = "sony_zve10ii_v1") -> Catalogue:
    """Load a catalogue shipped with this package."""
    path = Path(__file__).with_name("data") / f"{name}.json"
    if not path.is_file():
        raise DataError(f"no bundled catalogue named {name!r}")
    return load(path)


__all__ = [
    "SCHEMA_VERSION",
    "Catalogue",
    "CatalogueClip",
    "bundled",
    "file_sha256",
    "load",
]
