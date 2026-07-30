"""Source records: enough to reproduce a measurement from the record alone."""

from __future__ import annotations

from pathlib import Path

import pytest

from film_analysis_tools.capabilities.source import record as source_record
from film_analysis_tools.capabilities.source.record import (
    Cadence,
    Crop,
    DecodeContract,
    SourceRecord,
)
from film_analysis_tools.core.errors import DataError, SelectionError


def _record(**kwargs: object) -> SourceRecord:
    base: dict[str, object] = {
        "source_id": "pulp_scene_001",
        "edition": "UHD BD remux",
        "sha256": "a" * 64,
        "byte_size": 96_275_517_440,
        "coded_width": 3840,
        "coded_height": 2160,
        "cadence": Cadence(24000, 1001),
        "decode": DecodeContract(transfer="pq", primaries="bt2020"),
    }
    base.update(kwargs)
    return SourceRecord(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------- cadence


def test_cadence_is_exact_not_rounded() -> None:
    """23.98 is not a frame rate. 24000/1001 is."""
    cadence = Cadence.parse("24000/1001")
    assert (cadence.numerator, cadence.denominator) == (24000, 1001)
    assert cadence.fps == pytest.approx(23.976, abs=0.001)
    assert str(cadence) == "24000/1001"


def test_cadence_accepts_the_integer_form() -> None:
    assert Cadence.parse("25").fps == 25.0


def test_cadence_frame_index_uses_the_rational() -> None:
    cadence = Cadence(24000, 1001)
    assert cadence.frame_at(1.0) == 23
    assert cadence.frame_at(0.0) == 0


def test_impossible_cadence_is_refused() -> None:
    with pytest.raises(SelectionError):
        Cadence(0, 1)


# ------------------------------------------------------------------------------- crop


def test_crop_defaults_to_the_whole_coded_frame() -> None:
    assert _record().active_picture == (0, 0, 3840, 2160)


def test_crop_describes_the_active_picture() -> None:
    """Letterbox bars are not scene content; measuring them corrupts every statistic."""
    cropped = _record(crop=Crop(x=0, y=278, width=3840, height=1604))
    assert cropped.active_picture == (0, 278, 3840, 1604)


def test_a_crop_that_does_not_fit_is_an_error() -> None:
    with pytest.raises(DataError, match="does not fit"):
        _ = _record(crop=Crop(x=0, y=0, width=9999, height=100)).active_picture


# ----------------------------------------------------------------------------- identity


def test_identity_binds_material_and_decode_together() -> None:
    """Two runs agreeing on this measured the same material the same way."""
    assert _record().identity == _record().identity


@pytest.mark.parametrize(
    "change",
    [
        {"sha256": "b" * 64},
        {"cadence": Cadence(25, 1)},
        {"crop": Crop(x=0, y=278, width=3840, height=1604)},
        {"decode": DecodeContract(transfer="slog3")},
    ],
)
def test_identity_changes_when_the_measurement_would(change: dict[str, object]) -> None:
    assert _record(**change).identity != _record().identity


def test_identity_ignores_things_that_do_not_affect_the_measurement() -> None:
    """A note or a moved file does not make two results incomparable."""
    assert _record(notes="re-checked").identity == _record().identity
    assert _record(path_hint="/somewhere/else.mkv").identity == _record().identity


# ------------------------------------------------------------------- round trip and find


def test_round_trips_through_json(tmp_path: Path) -> None:
    original = _record(crop=Crop(0, 278, 3840, 1604), timestamp="1994-05-21")
    path = tmp_path / "source.json"
    original.save(path)
    restored = source_record.load(path)
    assert restored == original
    assert restored.identity == original.identity


def test_a_malformed_record_is_refused() -> None:
    with pytest.raises(SelectionError, match="64 hex"):
        _record(sha256="short")


def test_locate_finds_material_by_content_when_the_hint_is_stale(tmp_path: Path) -> None:
    """The legacy failure: a renamed source left its manifest pointing at nothing."""
    payload = b"not really a movie" * 32
    moved = tmp_path / "renamed.mkv"
    moved.write_bytes(payload)
    entry = _record(
        sha256=source_record.file_sha256(moved),
        byte_size=len(payload),
        path_hint="/gone/original.mkv",
    )
    assert entry.locate(roots=(tmp_path,)) == moved


def test_locate_reports_the_digest_when_nothing_matches(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="aaaaaaaaaaaa"):
        _record(byte_size=7, path_hint="/gone.mkv").locate(roots=(tmp_path,))


def test_the_record_carries_every_reproducibility_field() -> None:
    """Edition, timestamp, crop, cadence, decode, active picture, content hash."""
    payload = _record(timestamp="1994-05-21", crop=Crop(0, 278, 3840, 1604)).as_record()
    for key in (
        "edition",
        "timestamp",
        "crop",
        "cadence",
        "decode",
        "active_picture",
        "sha256",
        "identity",
    ):
        assert key in payload, key
    assert payload["decode"]["transfer"] == "pq"
