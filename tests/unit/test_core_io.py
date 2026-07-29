"""Typed IO: raises rather than exits, writes atomically, does not coerce behind your back."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from film_analysis_tools.core import io
from film_analysis_tools.core.errors import DataError

# ------------------------------------------------------------------------------ JSON


def test_json_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "payload.json"
    io.write_json(path, {"a": 1, "b": [1.5, "two"], "c": {"d": True}})
    assert io.read_json(path) == {"a": 1, "b": [1.5, "two"], "c": {"d": True}}


def test_json_write_creates_parent_directories(tmp_path: Path) -> None:
    io.write_json(tmp_path / "a" / "b" / "c.json", {})
    assert (tmp_path / "a" / "b" / "c.json").is_file()


def test_json_output_is_deterministic_and_newline_terminated(tmp_path: Path) -> None:
    payload = {"z": 1, "a": 2}
    first, second = tmp_path / "1.json", tmp_path / "2.json"
    io.write_json(first, payload)
    io.write_json(second, payload)
    assert first.read_bytes() == second.read_bytes()
    assert first.read_text(encoding="utf-8").endswith("}\n")


def test_json_serialises_numpy_scalars_and_arrays(tmp_path: Path) -> None:
    path = tmp_path / "numpy.json"
    io.write_json(
        path,
        {
            "int": np.int32(7),
            "float": np.float64(1.5),
            "bool": np.bool_(True),
            "array": np.asarray([1.0, 2.0]),
            "nonfinite": np.float64("nan"),
        },
    )
    loaded = io.read_json(path)
    assert loaded == {
        "int": 7,
        "float": 1.5,
        "bool": True,
        "array": [1.0, 2.0],
        "nonfinite": None,
    }


def test_reading_a_json_array_is_an_error_not_a_surprise_later(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(DataError, match="expected a JSON object"):
        io.read_json(path)


def test_malformed_json_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(DataError, match=r"bad\.json"):
        io.read_json(path)


def test_missing_file_raises_rather_than_exits(tmp_path: Path) -> None:
    """The whole point of replacing the legacy contract module."""
    with pytest.raises(DataError):
        io.read_json(tmp_path / "absent.json")


def test_an_unserialisable_payload_leaves_no_partial_file(tmp_path: Path) -> None:
    path = tmp_path / "partial.json"
    with pytest.raises(DataError):
        io.write_json(path, {"bad": object()})
    assert not path.exists()


def test_a_failed_write_does_not_destroy_the_previous_artifact(tmp_path: Path) -> None:
    """Atomic replace: an interrupted run leaves the old file, never a truncated one."""
    path = tmp_path / "artifact.json"
    io.write_json(path, {"generation": 1})
    with pytest.raises(DataError):
        io.write_json(path, {"bad": object()})
    assert io.read_json(path) == {"generation": 1}
    assert not list(tmp_path.glob(".*tmp*"))


# ------------------------------------------------------------------------------- CSV


def test_csv_round_trips_as_strings_by_default(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    io.write_csv(path, [{"name": "007", "value": "1.5"}])
    assert io.read_csv(path) == [{"name": "007", "value": "1.5"}]


def test_csv_does_not_silently_mangle_identifiers(tmp_path: Path) -> None:
    """The legacy reader coerced by default, turning '007' into 7."""
    path = tmp_path / "ids.csv"
    io.write_csv(path, [{"scene": "007"}])
    assert io.read_csv(path)[0]["scene"] == "007"
    assert io.read_csv(path, coerce=True)[0]["scene"] == 7


def test_csv_columns_follow_first_seen_order_across_all_rows(tmp_path: Path) -> None:
    path = tmp_path / "ragged.csv"
    io.write_csv(path, [{"a": 1, "b": 2}, {"b": 3, "c": 4}])
    assert io.read_csv(path)[0] == {"a": "1", "b": "2", "c": ""}


def test_empty_rows_write_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    io.write_csv(path, [])
    assert path.read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------- arrays


def test_arrays_round_trip_with_dtypes_intact(tmp_path: Path) -> None:
    path = tmp_path / "arrays.npz"
    original = {
        "rgb": np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64),
        "index": np.asarray([1, 2, 3], dtype=np.int16),
    }
    io.write_arrays(path, original)
    loaded = io.read_arrays(path)
    assert set(loaded) == set(original)
    for key, value in original.items():
        assert np.array_equal(loaded[key], value)
        assert loaded[key].dtype == value.dtype


def test_reading_a_non_npz_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "not.npz"
    path.write_text("definitely not an npz", encoding="utf-8")
    with pytest.raises(DataError, match=r"not\.npz"):
        io.read_arrays(path)


# ---------------------------------------------------------------------------- scalars


def test_number_helpers_tolerate_absent_and_malformed_values() -> None:
    assert io.as_float(None, 2.0) == 2.0
    assert io.as_float("nonsense", 2.0) == 2.0
    assert io.as_float(float("inf"), 2.0) == 2.0
    assert io.as_float("1.5") == 1.5
    assert io.as_int(None, 3) == 3
    assert io.as_int("nonsense", 3) == 3
    assert io.as_int("4") == 4


def test_json_lines_stream(tmp_path: Path) -> None:
    path = tmp_path / "journal.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert [record["a"] for record in io.iter_json_lines(path)] == [1, 2]


def test_json_lines_reports_the_offending_line(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(DataError, match=":2"):
        list(io.iter_json_lines(path))


def test_non_finite_numbers_become_null_not_invalid_json(tmp_path: Path) -> None:
    """Python emits bare NaN/Infinity tokens, which are not valid JSON.

    Python reads them back without complaint, so the file looks fine here and fails wherever
    it is consumed next. Non-finite values are written as null instead.
    """
    import json

    path = tmp_path / "nonfinite.json"
    io.write_json(path, {"nan": float("nan"), "inf": float("inf"), "ok": 1.5})
    raw = path.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert "Infinity" not in raw
    # A strict parser must accept it.
    assert json.loads(raw, parse_constant=_reject) == {"nan": None, "inf": None, "ok": 1.5}


def _reject(token: str) -> object:
    raise AssertionError(f"non-standard JSON token written: {token}")


def test_nested_and_numpy_payloads_normalise_throughout(tmp_path: Path) -> None:
    path = tmp_path / "nested.json"
    io.write_json(
        path,
        {
            "rows": [{"value": np.float64("nan")}, {"value": np.float32(2.5)}],
            "counts": np.asarray([1, 2], dtype=np.int16),
            "flag": np.bool_(False),
            "where": Path("/tmp/x"),
            "pairs": (1, 2),
        },
    )
    assert io.read_json(path) == {
        "rows": [{"value": None}, {"value": 2.5}],
        "counts": [1, 2],
        "flag": False,
        "where": "/tmp/x",
        "pairs": [1, 2],
    }
