"""The reporting layer: valid, self-contained, and honest about what it is showing."""

from __future__ import annotations

import html.parser
import re

import numpy as np
import pytest

from film_analysis_tools.capabilities.colour import display, transforms
from film_analysis_tools.capabilities.report import charts, svg
from film_analysis_tools.capabilities.report import html as html_module
from film_analysis_tools.capabilities.sample.table import RGB_COLUMN, SampleTable
from film_analysis_tools.capabilities.statistics import compare
from film_analysis_tools.capabilities.statistics.compare import per_sample_metric

SELF_CLOSING = {"meta", "br", "img", "input", "line", "rect", "polyline", "use", "path"}


class _Balance(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.mismatched: list[str] = []

    def handle_startendtag(self, tag: str, attrs: object) -> None:
        """``<rect .../>`` is balanced by definition; do not let the default split it."""

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag not in SELF_CLOSING:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        else:
            self.mismatched.append(tag)


@pytest.fixture
def table() -> SampleTable:
    rgb = np.random.default_rng(4).uniform(0.02, 1.0, size=(1500, 3))
    return SampleTable(columns={RGB_COLUMN: rgb}, name="synthetic")


# ------------------------------------------------------------------------ primitives


def test_svg_escapes_untrusted_text() -> None:
    assert "&lt;script&gt;" in svg.text(0, 0, "<script>", fill="var(--ink)")


def test_frame_maps_data_space_to_pixels() -> None:
    frame = svg.Frame(width=100, height=100, left=10, right=10, top=10, bottom=10)
    assert frame.x(frame.x_min) == pytest.approx(10.0)
    assert frame.x(frame.x_max) == pytest.approx(90.0)
    assert frame.y(frame.y_max) == pytest.approx(10.0)


def test_non_finite_coordinates_do_not_produce_broken_svg() -> None:
    """A NaN in a coordinate would silently break the whole path."""
    rendered = svg.line(float("nan"), 0, float("inf"), 10, stroke="x")
    assert "NaN" not in rendered
    assert "inf" not in rendered.lower()


# ---------------------------------------------------------------------- display encode


def test_srgb_encode_brightens_linear_values() -> None:
    """Without the transfer function every swatch would read far too dark."""
    encoded = display.srgb_encode(np.asarray([[0.18, 0.18, 0.18]]))
    assert 0.4 < float(encoded[0, 0]) < 0.55


def test_hex_colours_are_well_formed() -> None:
    colours = display.hex_colours(np.asarray([[0.0, 0.5, 1.0], [1.0, 1.0, 1.0]]))
    assert all(re.fullmatch(r"#[0-9a-f]{6}", colour) for colour in colours)
    assert colours[1] == "#ffffff"


def test_clipping_is_measured_rather_than_hidden() -> None:
    assert display.clipped_fraction(np.asarray([[0.5, 0.5, 2.0]])) == pytest.approx(1 / 3)
    assert display.clipped_fraction(np.asarray([[0.5, 0.5, 0.5]])) == 0.0


# ---------------------------------------------------------------------------- charts


def test_charts_handle_an_empty_cohort_without_raising() -> None:
    empty = SampleTable(columns={RGB_COLUMN: np.zeros((0, 3))}, name="none")
    assert "empty cohort" in charts.cohort_coverage(empty)
    assert "empty cohort" in charts.sample_swatches(
        empty, baseline=transforms.identity(), candidate=transforms.identity()
    )


def test_hue_response_reports_a_sign_change_for_a_channel_gain() -> None:
    """The chart that explains a cancelling result — it must actually detect the crossing."""
    figure = charts.hue_response(transforms.identity(), transforms.channel_gain(red=1.15))
    assert "changes sign" in figure


def test_hue_response_reports_a_consistent_direction_when_there_is_one() -> None:
    figure = charts.hue_response(transforms.identity(), transforms.identity())
    assert "consistent direction" in figure or "changes sign" not in figure


def test_tone_response_draws_both_transforms(table: SampleTable) -> None:
    figure = charts.tone_response(transforms.identity(), transforms.tone_gamma(0.8))
    assert figure.count("<polyline") >= 3  # identity guide, baseline, candidate


def test_distribution_chart_reports_the_verdict_not_just_a_number(table: SampleTable) -> None:
    result = compare(
        table,
        baseline=transforms.identity(),
        candidate=transforms.saturate(1.2),
        metric="chroma_delta",
    )
    values = per_sample_metric(
        table,
        baseline=transforms.identity(),
        candidate=transforms.saturate(1.2),
        metric="chroma_delta",
    )
    figure = charts.metric_distribution(values, result)
    assert "null spread" in figure
    assert result.verdict in figure
    assert "<rect" in figure


# ----------------------------------------------------------------------- whole page


def _page(table: SampleTable) -> str:
    baseline, candidate = transforms.identity(), transforms.named("warm_gain")
    cohorts = {"all": table}
    results = [compare(table, baseline=baseline, candidate=candidate, metric="hue_drift")]
    per_sample = {
        results[0].cohort: per_sample_metric(
            table, baseline=baseline, candidate=candidate, metric="hue_drift"
        )
    }
    context = html_module.ReportContext(
        title="test run",
        pack="pack",
        baseline="identity",
        candidate="warm_gain",
        metric="hue_drift",
        resamples=32,
        seed=0,
        roots={"read_root": "/somewhere", "write_root": "/elsewhere"},
    )
    return html_module.comparison_report(
        context, results, cohorts, per_sample, baseline=baseline, candidate=candidate
    )


def test_report_is_balanced_html(table: SampleTable) -> None:
    parser = _Balance()
    parser.feed(_page(table))
    assert not parser.mismatched
    assert not parser.stack


def test_report_is_self_contained(table: SampleTable) -> None:
    """No assets, no network: it must survive being copied anywhere."""
    page = _page(table)
    assert "<script" not in page
    for pattern in ("http://", "https://", "src=", "@import"):
        assert pattern not in page


def test_report_states_its_tier_and_limits(table: SampleTable) -> None:
    """A report that looks authoritative without saying what it rests on is the failure mode."""
    page = _page(table)
    assert "comparison" in page
    assert "not a fact beyond them" in page
    assert "null control" in page


def test_report_never_emits_non_finite_tokens(table: SampleTable) -> None:
    page = _page(table)
    assert "NaN" not in page
    assert "Infinity" not in page


def test_report_records_provenance(table: SampleTable) -> None:
    page = _page(table)
    assert "/somewhere" in page
    assert "warm_gain" in page
    assert "seed 0" in page
