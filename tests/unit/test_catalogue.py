"""The catalogue and the S-Log3 decode it depends on."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from film_analysis_tools.capabilities import catalogue
from film_analysis_tools.capabilities.catalogue import categories, manifest
from film_analysis_tools.capabilities.source import slog3
from film_analysis_tools.core.errors import DataError

USER_REQUESTED = {
    "normal_daylight_interior",
    "skin_neutral",
    "skin_warm",
    "skin_green",
    "skin_mixed",
    "deep_underexposure",
    "overexposure_clipped",
    "saturated_practical",
    "low_saturation",
    "difficult_shadows",
    "motion_or_noise",
}


# ------------------------------------------------------------------------- S-Log3


def test_curve_hits_its_published_anchors() -> None:
    """18% grey at code 420, 90% white at code 598.

    The check that the transcription is right rather than merely plausible.
    """
    for code, expected in slog3.CURVE_ANCHORS:
        assert float(slog3.slog3_to_linear(np.asarray([code]))[0]) == pytest.approx(
            expected, abs=1e-3
        )


def test_curve_round_trips() -> None:
    codes = np.linspace(0.05, 0.75, 40)
    assert np.allclose(slog3.linear_to_slog3(slog3.slog3_to_linear(codes)), codes, atol=1e-6)


def test_curve_is_monotone() -> None:
    values = slog3.slog3_to_linear(np.linspace(0.0, 1.0, 500))
    assert np.all(np.diff(values) > 0)


def test_primaries_matrix_preserves_neutrals() -> None:
    """Rows summing to one is what keeps grey grey through the gamut rotation."""
    assert np.allclose(slog3.SGAMUT3CINE_TO_REC709.sum(axis=1), 1.0, atol=1e-4)
    grey = np.asarray([[0.4, 0.4, 0.4]])
    assert np.allclose(slog3.sgamut3cine_to_rec709(grey), grey, atol=1e-4)


def test_saturated_sources_land_outside_rec709() -> None:
    """Negative channels are information, not an error: the colour is outside Rec.709."""
    cyan = np.asarray([[0.05, 0.6, 0.6]])
    assert bool(slog3.out_of_gamut_mask(slog3.sgamut3cine_to_rec709(cyan))[0])


def test_slog3_never_reaches_full_code_range() -> None:
    """Why an absolute 'near maximum code' clipping test finds nothing on this material."""
    assert float(slog3.slog3_to_linear(np.asarray([0.75]))[0]) > 3.0


# ---------------------------------------------------------------------- taxonomy


def test_every_requested_category_exists() -> None:
    assert {category.id for category in categories.CATEGORIES} == USER_REQUESTED


def test_every_category_states_the_failure_it_provokes() -> None:
    """A category that cannot say what it is for is a scene description, not a test condition."""
    for category in categories.CATEGORIES:
        assert len(category.provokes) > 20, category.id
        assert len(category.rule) > 20, category.id


def test_skin_categories_are_marked_human_labelled() -> None:
    """Face presence is not measurable yet; the catalogue must not imply otherwise."""
    for category in categories.CATEGORIES:
        assert category.human_labelled == category.id.startswith("skin_")


def test_cast_sign_convention_is_the_documented_one() -> None:
    """Positive green_magenta is green. Inverting this labelled every magenta frame green."""
    green = {"cast_warm_cool": 0.1, "cast_green_magenta": 0.2, "cast_disagreement": 0.0}
    magenta = {"cast_warm_cool": 0.1, "cast_green_magenta": -0.2, "cast_disagreement": 0.0}
    assert categories._cast(green) == "green"
    assert categories._cast(magenta) != "green"


def test_skin_categories_need_a_labelled_face() -> None:
    measured = {"cast_warm_cool": 1.0, "cast_green_magenta": 0.0, "cast_disagreement": 0.0}
    assert "skin_warm" not in categories.classify(measured, clip_id="C9999")
    assert "skin_warm" in categories.classify(measured, clip_id=next(iter(categories.FACE_CLIPS)))


# --------------------------------------------------------------------- catalogue


@pytest.fixture(scope="module")
def cat() -> catalogue.Catalogue:
    return catalogue.bundled()


def test_the_bundled_catalogue_covers_the_whole_corpus(cat: catalogue.Catalogue) -> None:
    assert len(cat) == 105
    assert {clip.shoot for clip in cat} == {
        "samples",
        "samples_dark",
        "samples_dd_dark",
        "samples_evening",
    }


def test_every_clip_is_identified_by_content_not_path(cat: catalogue.Catalogue) -> None:
    for clip in cat:
        assert len(clip.sha256) == 64
        assert clip.byte_size > 0
    assert len({clip.sha256 for clip in cat}) == len(cat), "duplicate content in the corpus"


def test_categories_are_populated_and_overlapping(cat: catalogue.Catalogue) -> None:
    counts = cat.counts()
    assert set(counts) == USER_REQUESTED
    # Overlap is the point: the interesting failures live where conditions coincide.
    assert cat.select("deep_underexposure", "saturated_practical", require_all=True)


def test_selection_semantics(cat: catalogue.Catalogue) -> None:
    any_of = cat.select("deep_underexposure", "overexposure_clipped")
    all_of = cat.select("deep_underexposure", "overexposure_clipped", require_all=True)
    assert len(all_of) <= len(any_of)
    assert len(cat.select("deep_underexposure", limit=3)) == 3
    assert all(clip.shoot == "samples_evening" for clip in cat.select(shoot="samples_evening"))


def test_unknown_category_names_the_alternatives(cat: catalogue.Catalogue) -> None:
    with pytest.raises(DataError, match="unknown categories"):
        cat.select("not_a_category")


def test_ordinary_material_is_kept_rather_than_dropped(cat: catalogue.Catalogue) -> None:
    """Clips no category claimed still appear; silently losing them would be the legacy bug."""
    assert len(cat.select()) == len(cat)
    assert len(cat.uncategorised()) < len(cat)


def test_the_catalogue_declares_its_decode_contract(cat: catalogue.Catalogue) -> None:
    """Statistics are meaningless without knowing how the material was decoded."""
    assert cat.decode["transfer"] == "slog3_to_linear"
    assert cat.decode["primaries"] == "sgamut3cine_to_rec709"
    assert cat.camera["capture_gamma"] == "s-log3-cine"


def test_locate_reports_the_digest_when_a_clip_cannot_be_found() -> None:
    absent = manifest.CatalogueClip(
        clip_id="C0000",
        sha256="a" * 64,
        byte_size=123,
        shoot="nowhere",
        path_hint="/does/not/exist.MP4",
        categories=(),
    )
    with pytest.raises(DataError, match="aaaaaaaaaaaa"):
        absent.locate()


def test_locate_finds_a_clip_by_content_when_the_path_hint_is_stale(tmp_path: Path) -> None:
    """The failure that cost the legacy corpus its provenance: a source file was renamed."""
    payload = b"pretend this is an mp4" * 10
    moved = tmp_path / "renamed_by_someone.MP4"
    moved.write_bytes(payload)
    clip = manifest.CatalogueClip(
        clip_id="C0000",
        sha256=manifest.file_sha256(moved),
        byte_size=len(payload),
        shoot="test",
        path_hint="/stale/original_name.MP4",
        categories=(),
    )
    assert clip.locate(roots=[tmp_path]) == moved


# ------------------------------------------------------- the cross-repo JSON contract


def test_json_taxonomy_output_is_parseable_and_carries_identity() -> None:
    """Other repos consume the catalogue through this, so its shape is a contract."""
    import io as _io
    import json
    from contextlib import redirect_stdout

    from film_analysis_tools.cli import main

    buffer = _io.StringIO()
    with redirect_stdout(buffer):
        assert main(["catalogue", "--json"]) == 0
    payload = json.loads(buffer.getvalue())

    assert payload["catalogue_id"]
    assert payload["generated"], "consumers need the version to tell result sets apart"
    assert payload["clip_count"] == 105
    assert payload["decode"]["transfer"] == "slog3_to_linear"
    assert {entry["id"] for entry in payload["categories"]} == USER_REQUESTED
    assert all("count" in entry for entry in payload["categories"])
    assert "skin_green" in payload["empty_categories"]


def test_json_query_output_reports_the_query_it_answered() -> None:
    import io as _io
    import json
    from contextlib import redirect_stdout

    from film_analysis_tools.cli import main

    buffer = _io.StringIO()
    with redirect_stdout(buffer):
        assert (
            main(["catalogue", "deep_underexposure", "saturated_practical", "--all", "--json"]) == 0
        )
    payload = json.loads(buffer.getvalue())

    assert payload["query"]["require_all"] is True
    assert payload["query"]["categories"] == ["deep_underexposure", "saturated_practical"]
    assert payload["count"] == len(payload["clips"])
    for clip in payload["clips"]:
        assert len(clip["sha256"]) == 64
        assert {"deep_underexposure", "saturated_practical"} <= set(clip["categories"])


def test_json_output_never_contains_non_finite_tokens() -> None:
    """Measured values are embedded, so the strict-JSON guarantee has to hold here too."""
    import io as _io
    from contextlib import redirect_stdout

    from film_analysis_tools.cli import main

    buffer = _io.StringIO()
    with redirect_stdout(buffer):
        main(["catalogue", "motion_or_noise", "--json"])
    text = buffer.getvalue()
    assert "NaN" not in text
    assert "Infinity" not in text
