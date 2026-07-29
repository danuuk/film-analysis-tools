"""End-to-end checks against a real sample pack.

Skipped unless ``FILM_ANALYSIS_WORKSPACE`` points at a workspace containing a pack, because
corpora live outside the repository and CI has none. Run locally with:

    FILM_ANALYSIS_WORKSPACE=/path/to/workspace uv run pytest tests/integration

These assert *properties* that must hold on any real pack, not values from one particular
corpus — so the test survives the data being replaced.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from film_analysis_tools.capabilities.colour import transforms
from film_analysis_tools.capabilities.sample import cohorts, load_pack
from film_analysis_tools.capabilities.sample.table import RGB_COLUMN, SampleTable
from film_analysis_tools.capabilities.statistics import compare
from film_analysis_tools.core import Workspace
from film_analysis_tools.core.workspace import ENV_VAR

pytestmark = pytest.mark.skipif(
    not os.environ.get(ENV_VAR), reason=f"{ENV_VAR} is not set; no corpus available"
)


@pytest.fixture(scope="module")
def pack() -> SampleTable:
    workspace = Workspace.from_env()
    names = [name for name in workspace.names() if (workspace.root / name / "samples").is_dir()]
    if not names:
        pytest.skip("workspace contains no sample packs")
    return load_pack(names[0], workspace=workspace)


def test_a_real_pack_loads_with_rgb_and_provenance(pack: SampleTable) -> None:
    assert len(pack) > 0
    assert RGB_COLUMN in pack.column_names
    assert "scene_id" in pack.column_names
    assert pack.rgb.ndim == 2
    assert pack.rgb.shape[1] == 3


def test_stored_features_agree_with_recomputed_ones(pack: SampleTable) -> None:
    """Packs store luma; deriving it from RGB must reproduce what was stored.

    This is the cross-check that the migrated feature maths matches the legacy maths that
    wrote these packs — an independent route to the same number, which is worth more than
    any assertion that the pipeline ran.
    """
    if "luma_bt2020" not in pack.columns:
        pytest.skip("pack does not store luma")
    stored = np.asarray(pack.columns["luma_bt2020"], dtype=np.float64)
    derived = np.asarray(pack._derive("luma_bt2020"), dtype=np.float64)
    assert np.allclose(stored, derived, rtol=1e-9, atol=1e-12)


def test_identity_shows_no_effect_on_real_samples(pack: SampleTable) -> None:
    result = compare(
        pack, baseline=transforms.identity(), candidate=transforms.identity(), metric="hue_drift"
    )
    assert result.effect == 0.0
    assert result.magnitude == 0.0
    assert result.verdict == "no change"


def test_cohorts_select_real_and_disjoint_populations(pack: SampleTable) -> None:
    built = cohorts.build(pack, ("shadows", "highlights"))
    if len(built) < 2:
        pytest.skip("pack lacks both shadow and highlight samples")
    assert len(built["shadows"]) + len(built["highlights"]) <= len(pack)


def test_a_warm_gain_pulls_greens_toward_yellow(pack: SampleTable) -> None:
    """Direction, not magnitude: boosting red and cutting blue must decrease green hue."""
    foliage = cohorts.foliage_like(pack)
    if len(foliage) < 100:
        pytest.skip("pack has too few saturated greens")
    result = compare(
        foliage,
        baseline=transforms.identity(),
        candidate=transforms.named("warm_gain"),
        metric="hue_drift",
    )
    assert result.is_directional
    assert result.effect < 0.0
    assert result.exceeds_null


def test_hue_metrics_on_near_neutral_samples_are_flagged_as_unreliable(
    pack: SampleTable,
) -> None:
    """Hue is numerically meaningless near the achromatic axis.

    A naive harness reports a large, confident-looking hue drift there. This one must report
    that the samples do not agree on a direction, so the number is not mistaken for a finding.
    """
    neutral = cohorts.neutral(pack)
    if len(neutral) < 100:
        pytest.skip("pack has too few near-neutral samples")
    result = compare(
        neutral,
        baseline=transforms.identity(),
        candidate=transforms.named("warm_gain"),
        metric="hue_drift",
    )
    assert result.magnitude > abs(result.effect)
    assert not result.is_directional
