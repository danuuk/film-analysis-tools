"""Source records: everything needed to reproduce a measurement from the record alone.

A measurement is only repeatable if the thing measured can be found again and decoded the same
way. The legacy scene manifests recorded the source *path*, start, duration and pixel format —
but no content hash, so when a source file was renamed the record pointed at nothing and the
material was recoverable only by matching a probe block by hand. Crop lived in the directory
name, and cadence was implicit in a frame rate string.

A :class:`SourceRecord` carries the seven things that make a measurement reproducible: edition,
timestamp, crop, cadence, decode contract, active-picture dimensions, and **content hash**. The
footage itself stays outside this repository and the engine; only the record travels.

Nothing here decodes anything. The record *describes* a decode so that two runs can be compared,
or one re-run, without the original command line being remembered.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from film_analysis_tools.core.errors import DataError, SelectionError
from film_analysis_tools.core.io import read_json, write_json

SCHEMA_VERSION = 1
HASH_CHUNK = 4 << 20


@dataclass(frozen=True)
class Cadence:
    """Frame rate as an exact rational, because 24000/1001 is not 23.98."""

    numerator: int
    denominator: int = 1

    def __post_init__(self) -> None:
        if self.numerator <= 0 or self.denominator <= 0:
            raise SelectionError(f"cadence must be positive: {self.numerator}/{self.denominator}")

    @classmethod
    def parse(cls, text: str) -> Cadence:
        """Accept ``24000/1001`` or ``25`` — the two forms ffprobe emits."""
        fraction = Fraction(text.strip())
        return cls(numerator=fraction.numerator, denominator=fraction.denominator)

    @property
    def fps(self) -> float:
        return self.numerator / self.denominator

    @property
    def frame_duration_s(self) -> float:
        return self.denominator / self.numerator

    def frame_at(self, seconds: float) -> int:
        return int(seconds * self.numerator // self.denominator)

    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator}"


@dataclass(frozen=True)
class Crop:
    """The active picture inside the coded frame.

    Recorded explicitly rather than encoded in a directory name, because letterbox bars and
    overscan are not scene content and measuring them silently corrupts every statistic.
    """

    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    def __post_init__(self) -> None:
        if self.width < 0 or self.height < 0 or self.x < 0 or self.y < 0:
            raise SelectionError(f"crop must be non-negative: {self}")

    @property
    def is_full_frame(self) -> bool:
        return self.x == 0 and self.y == 0

    def applied_to(self, coded_width: int, coded_height: int) -> tuple[int, int, int, int]:
        width = self.width or coded_width - self.x
        height = self.height or coded_height - self.y
        if self.x + width > coded_width or self.y + height > coded_height:
            raise DataError(f"crop {self} does not fit inside {coded_width}x{coded_height}")
        return self.x, self.y, width, height


@dataclass(frozen=True)
class DecodeContract:
    """How the source is turned into numbers. Statistics are meaningless without it."""

    input_range: str = "full"
    matrix: str = "bt709"
    transfer: str = "unknown"
    primaries: str = "unknown"
    output_pixel_format: str = "rgb48le"
    scale: str = "none"

    def as_record(self) -> dict[str, str]:
        return {
            "input_range": self.input_range,
            "matrix": self.matrix,
            "transfer": self.transfer,
            "primaries": self.primaries,
            "output_pixel_format": self.output_pixel_format,
            "scale": self.scale,
        }


@dataclass(frozen=True)
class SourceRecord:
    """A reproducible reference to measured material."""

    source_id: str
    edition: str
    """Which version of the material: a camera shoot, a remux, a scan, a grade."""

    sha256: str
    byte_size: int
    coded_width: int
    coded_height: int
    cadence: Cadence
    decode: DecodeContract
    crop: Crop = field(default_factory=Crop)
    timestamp: str = ""
    """When the material was captured or released — not when this record was written."""

    path_hint: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        if len(self.sha256) != 64:
            raise SelectionError(f"sha256 must be 64 hex characters: {self.sha256!r}")
        if self.coded_width <= 0 or self.coded_height <= 0:
            raise SelectionError(f"coded dimensions must be positive: {self}")

    @property
    def active_picture(self) -> tuple[int, int, int, int]:
        """``(x, y, width, height)`` of the region measurements may use."""
        return self.crop.applied_to(self.coded_width, self.coded_height)

    @property
    def identity(self) -> str:
        """Stable digest of the record itself, for binding results to the exact provenance.

        Two runs that agree on this measured the same material through the same decode. Two that
        do not are not comparable, whatever their numbers look like.
        """
        payload = "|".join(
            [
                self.sha256,
                str(self.byte_size),
                f"{self.coded_width}x{self.coded_height}",
                str(self.cadence),
                str(self.crop),
                str(sorted(self.decode.as_record().items())),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def locate(self, *, roots: tuple[Path, ...] = (), verify: bool = True) -> Path:
        """Find the material, by hint first and by content hash if the hint is stale."""
        candidates: list[Path] = []
        hint = Path(self.path_hint).expanduser() if self.path_hint else None
        if hint is not None and hint.is_file():
            candidates.append(hint)
        for root in roots:
            candidates.extend(
                path
                for path in Path(root).expanduser().rglob("*")
                if path.is_file() and path.stat().st_size == self.byte_size
            )
        for candidate in candidates:
            if not verify or file_sha256(candidate) == self.sha256:
                return candidate
        raise DataError(
            f"source {self.source_id} not found; hint {self.path_hint!r} did not resolve and "
            f"nothing under {[str(root) for root in roots]} matched sha256 {self.sha256[:12]}"
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "source_id": self.source_id,
            "edition": self.edition,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "coded_width": self.coded_width,
            "coded_height": self.coded_height,
            "cadence": str(self.cadence),
            "crop": {
                "x": self.crop.x,
                "y": self.crop.y,
                "width": self.crop.width,
                "height": self.crop.height,
            },
            "active_picture": list(self.active_picture),
            "decode": self.decode.as_record(),
            "timestamp": self.timestamp,
            "path_hint": self.path_hint,
            "notes": self.notes,
            "identity": self.identity,
        }

    def save(self, path: Path) -> None:
        write_json(path, self.as_record())


def from_record(payload: dict[str, Any]) -> SourceRecord:
    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise DataError(f"unsupported source record schema_version {version!r}")
    crop = payload.get("crop", {})
    decode = payload.get("decode", {})
    return SourceRecord(
        source_id=str(payload["source_id"]),
        edition=str(payload.get("edition", "")),
        sha256=str(payload["sha256"]),
        byte_size=int(payload["byte_size"]),
        coded_width=int(payload["coded_width"]),
        coded_height=int(payload["coded_height"]),
        cadence=Cadence.parse(str(payload["cadence"])),
        decode=DecodeContract(**{key: str(value) for key, value in decode.items()}),
        crop=Crop(
            x=int(crop.get("x", 0)),
            y=int(crop.get("y", 0)),
            width=int(crop.get("width", 0)),
            height=int(crop.get("height", 0)),
        ),
        timestamp=str(payload.get("timestamp", "")),
        path_hint=str(payload.get("path_hint", "")),
        notes=str(payload.get("notes", "")),
    )


def load(path: Path) -> SourceRecord:
    return from_record(read_json(path))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(HASH_CHUNK), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DataError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


__all__ = [
    "SCHEMA_VERSION",
    "Cadence",
    "Crop",
    "DecodeContract",
    "SourceRecord",
    "file_sha256",
    "from_record",
    "load",
]
