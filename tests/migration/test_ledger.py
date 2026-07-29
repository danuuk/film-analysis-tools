"""The migration ledger must stay complete, consistent and honest.

A ledger written once and never checked is a snapshot that rots. Under CI it is a control
surface: a legacy module cannot be silently forgotten, an archive group cannot appear without
a plan to record it, and a removal cannot happen without a written justification.

Most checks are self-contained. Coverage of the legacy tree needs the legacy checkout, so it
runs only when ``FILM_ANALYSIS_LEGACY_ROOT`` points at it — this repo must not require the
legacy checkout to test, matching the boundary the engine repo already keeps.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
LEDGER = REPO / "ledger.toml"
LEGACY_ROOT_ENV = "FILM_ANALYSIS_LEGACY_ROOT"

DISPOSITIONS = {"keep", "archive", "remove", "plugin"}

#: Archive groups whose written record is still outstanding. A ratchet: writing a record, or
#: adding a new archive group, must update this set deliberately. It may only shrink.
ARCHIVE_RECORDS_PENDING = {
    "campaign.al5",
    "campaign.b0_fsc1c",
    "campaign.broad",
    "campaign.kodak",
    "campaign.ptf",
    "campaign.vector_tile",
}


def _ledger() -> dict[str, Any]:
    return tomllib.loads(LEDGER.read_text(encoding="utf-8"))


def _groups() -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = _ledger()["group"]
    return groups


# ------------------------------------------------------------------------ structure


def test_ledger_declares_its_source_and_pin() -> None:
    meta = _ledger()["meta"]
    assert meta["schema_version"] == 1
    assert meta["legacy_pin"], "the ledger must name the commit it classified"
    assert meta["legacy_package"]


def test_group_ids_are_unique() -> None:
    ids = [group["id"] for group in _groups()]
    assert len(ids) == len(set(ids)), "duplicate group ids"


def test_every_group_has_a_valid_disposition() -> None:
    for group in _groups():
        assert group["disposition"] in DISPOSITIONS, group["id"]


def test_declared_counts_match_the_listed_modules() -> None:
    for group in _groups():
        assert group["module_count"] == len(group["modules"]), group["id"]
        assert group["loc"] > 0, group["id"]


def test_no_module_is_claimed_by_two_groups() -> None:
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for group in _groups():
        for module in group["modules"]:
            if module in seen:
                duplicates.append(f"{module}: {seen[module]} and {group['id']}")
            seen[module] = group["id"]
    assert not duplicates, duplicates


# --------------------------------------------------------------------------- honesty


def test_every_group_states_why() -> None:
    """No disposition without a reason. This is what stops the ledger becoming a bare list."""
    for group in _groups():
        key = "justification" if group["disposition"] == "remove" else "reason"
        text = group.get(key, "")
        assert len(text) > 40, f"{group['id']} needs a substantive {key}"
        assert "UNCLASSIFIED" not in text, f"{group['id']} was never classified"


def test_removals_carry_a_justification_not_merely_a_reason() -> None:
    """Nothing is removed before its record exists; for dead scaffolding the record is the
    justification itself, since the code stays readable at the pin."""
    for group in _groups():
        if group["disposition"] == "remove":
            assert group.get("justification"), group["id"]
            assert "reason" not in group, f"{group['id']} should justify, not merely reason"


def test_archived_groups_either_have_a_record_or_declare_it_pending() -> None:
    for group in _groups():
        if group["disposition"] != "archive":
            continue
        has_record = "record" in group
        pending = group.get("record_pending", False)
        assert has_record != pending, f"{group['id']} must have exactly one of record/pending"
        if has_record:
            assert (REPO / group["record"]).is_file(), f"{group['id']} names a missing record"


def test_the_pending_record_ratchet_matches_reality() -> None:
    """Adding an archive group without a record, or writing one, must be deliberate."""
    pending = {
        group["id"]
        for group in _groups()
        if group["disposition"] == "archive" and group.get("record_pending", False)
    }
    assert pending == ARCHIVE_RECORDS_PENDING, (
        "archive record backlog changed; update ARCHIVE_RECORDS_PENDING deliberately. "
        f"newly pending: {sorted(pending - ARCHIVE_RECORDS_PENDING)}, "
        f"now recorded: {sorted(ARCHIVE_RECORDS_PENDING - pending)}"
    )


def test_kept_groups_name_where_they_are_going() -> None:
    for group in _groups():
        if group["disposition"] == "keep":
            assert group.get("target"), f"{group['id']} must name a target layer"


# ------------------------------------------------------- coverage of the legacy tree


@pytest.mark.skipif(not os.environ.get(LEGACY_ROOT_ENV), reason=f"{LEGACY_ROOT_ENV} is not set")
def test_the_ledger_covers_the_legacy_package_exactly_once() -> None:
    """The check that makes this a control surface rather than a one-time inventory."""
    ledger = _ledger()
    package = Path(os.environ[LEGACY_ROOT_ENV]) / ledger["meta"]["legacy_package"]
    if not package.is_dir():
        pytest.skip(f"legacy package not found at {package}")

    on_disk = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    # Trivial namespace markers carry no disposition of their own.
    on_disk = {
        rel
        for rel in on_disk
        if not (
            rel.endswith("__init__.py")
            and (package / rel).read_text(errors="replace").count("\n") < 2
            and "reference_calibration_" not in rel
        )
    }
    claimed = {module for group in _groups() for module in group["modules"]}

    assert not on_disk - claimed, (
        f"legacy modules absent from the ledger: {sorted(on_disk - claimed)}"
    )
    assert not claimed - on_disk, (
        f"ledger names modules that do not exist: {sorted(claimed - on_disk)}"
    )
