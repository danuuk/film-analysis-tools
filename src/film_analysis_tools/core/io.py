"""Typed file IO for the formats this system actually uses.

Replaces the legacy ``mediachar.core.contract`` — 107 lines that 153 modules depended on, whose
``fail()`` raised ``SystemExit`` from library code and so made every one of them unusable
outside a command line.

Three deliberate differences:

* **Errors are raised, not exited.** Every failure is a :class:`DataError` naming the path.
* **Writes are atomic.** Output goes to a temporary file in the destination directory and is
  renamed into place, so an interrupted run leaves either the old artifact or the new one —
  never a truncated file that a later stage reads as valid.
* **No implicit coercion.** The legacy CSV reader guessed types from string content, which
  silently turned an identifier like ``007`` into ``7``. Coercion is available but must be
  asked for.
"""

from __future__ import annotations

import csv
import io as _stdlib_io
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from film_analysis_tools.core.errors import DataError

JSON_INDENT = 2


def _prepare(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DataError(f"cannot create directory {path.parent}: {exc}") from exc


def _temporary_beside(path: Path, suffix: str) -> tuple[int, Path]:
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=suffix)
    return descriptor, Path(name)


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    _prepare(path)
    descriptor, temporary = _temporary_beside(path, ".tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise DataError(f"cannot write {path}: {exc}") from exc


# ----------------------------------------------------------------------------- JSON


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object. Anything that is not an object is an error, not a surprise later."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DataError(f"cannot read {path}: {exc}") from exc
    try:
        loaded: Any = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DataError(f"malformed JSON in {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise DataError(f"expected a JSON object in {path}, found {type(loaded).__name__}")
    return loaded


def write_json(path: Path, payload: Any) -> None:
    """Write JSON deterministically: stable indentation, no ASCII escaping, trailing newline.

    Non-finite numbers become ``null``. Python would otherwise emit bare ``NaN`` and
    ``Infinity`` tokens, which are **not valid JSON** — Python reads them back happily while
    stricter parsers reject the file, so the damage surfaces somewhere else entirely. Metrics
    legitimately produce non-finite values, so this has to be handled rather than forbidden.
    """
    try:
        text = json.dumps(
            _normalise(payload), indent=JSON_INDENT, ensure_ascii=False, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise DataError(f"cannot serialise payload for {path}: {exc}") from exc
    _atomic_write_bytes(path, (text + "\n").encode("utf-8"))


def _normalise(value: Any) -> Any:
    """Convert a payload into strictly JSON-representable values.

    ``np.float64`` subclasses ``float``, so ``json.dumps`` handles it through the float path
    and never consults a ``default=`` hook — which is why this is a walk rather than a hook.
    """
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, int):
        return value
    if isinstance(value, np.ndarray):
        return [_normalise(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _normalise(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_normalise(item) for item in value]
    raise TypeError(f"cannot serialise {type(value).__name__}")


# ------------------------------------------------------------------------------ CSV


def read_csv(path: Path, *, coerce: bool = False) -> list[dict[str, Any]]:
    """Read rows as dictionaries.

    ``coerce`` opts into the legacy heuristic that turns numeric-looking strings into numbers.
    It is lossy — identifiers with leading zeros and version strings do not survive — so it is
    off by default and should be used only when reading legacy artifacts.
    """
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise DataError(f"cannot read {path}: {exc}") from exc
    if not coerce:
        return [dict(row) for row in rows]
    return [{key: coerce_scalar(value) for key, value in row.items()} for row in rows]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write rows, with columns in first-seen order across the whole sequence."""
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    buffer = _stdlib_io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    if fieldnames:
        writer.writeheader()
        writer.writerows(rows)
    _atomic_write_bytes(path, buffer.getvalue().encode("utf-8"))


def coerce_scalar(value: str | None) -> Any:
    """Best-effort string-to-scalar conversion. Lossy; see :func:`read_csv`."""
    if value is None or value == "":
        return None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if any(marker in value for marker in (".", "e", "E")):
            number = float(value)
            return number if math.isfinite(number) else value
        return int(value)
    except ValueError:
        return value


# --------------------------------------------------------------------------- arrays


def read_arrays(path: Path) -> dict[str, np.ndarray]:
    """Read a ``.npz`` of named arrays. Pickled payloads are refused."""
    try:
        with np.load(path, allow_pickle=False) as handle:
            return {key: handle[key] for key in handle.files}
    except (OSError, ValueError) as exc:
        raise DataError(f"cannot read arrays from {path}: {exc}") from exc


def write_arrays(path: Path, arrays: Mapping[str, np.ndarray], *, compress: bool = True) -> None:
    """Write named arrays to ``.npz``, atomically."""
    _prepare(path)
    descriptor, temporary = _temporary_beside(path, ".npz")
    try:
        # numpy's stubs declare ``savez(file, *args, allow_pickle, **kwds)``, so a mapping
        # splat is checked against ``allow_pickle: bool`` and rejected. The call is correct;
        # the annotation is not.
        save: Any = np.savez_compressed if compress else np.savez
        with os.fdopen(descriptor, "wb") as handle:
            save(handle, **dict(arrays))
        os.replace(temporary, path)
    except (OSError, ValueError) as exc:
        temporary.unlink(missing_ok=True)
        raise DataError(f"cannot write arrays to {path}: {exc}") from exc


# ---------------------------------------------------------------------------- number


def as_float(value: Any, default: float = 0.0) -> float:
    """A finite float, or ``default``. For reading fields that may be absent or malformed."""
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def iter_json_lines(path: Path) -> Iterable[dict[str, Any]]:
    """Stream a JSON-lines file, so a long journal need not be held in memory."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record: Any = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise DataError(f"malformed JSON at {path}:{number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise DataError(f"expected an object at {path}:{number}")
                yield record
    except OSError as exc:
        raise DataError(f"cannot read {path}: {exc}") from exc


__all__ = [
    "as_float",
    "as_int",
    "coerce_scalar",
    "iter_json_lines",
    "read_arrays",
    "read_csv",
    "read_json",
    "write_arrays",
    "write_csv",
    "write_json",
]
