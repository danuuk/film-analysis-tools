"""Analytical charts: what the numbers in a comparison actually look like.

Each chart answers a question a summary row cannot:

* **Metric distribution** — *why* an effect is what it is. A median of zero over a broad cohort
  and a median of zero because nothing moved look identical in a table and nothing alike here.
* **Response curve** — what the transform does across the input range, independent of any
  corpus. This is where a cancelling effect becomes obvious: the drift curve crosses zero.
* **Cohort coverage** — where in colour space the cohort actually sits, so "skin_like" can be
  checked against what it selected rather than trusted because of its name.
* **Samples** — the actual colours, before and after.

Colours are emitted as CSS custom properties so the page themes itself.
"""

from __future__ import annotations

import numpy as np

from film_analysis_tools.capabilities.colour import display, features, metrics
from film_analysis_tools.capabilities.report import svg
from film_analysis_tools.capabilities.sample.table import SampleTable
from film_analysis_tools.capabilities.statistics.compare import Comparison
from film_analysis_tools.core.protocols import RGB, Transform

INK = "var(--ink)"
MUTED = "var(--muted)"
GRID = "var(--grid)"
ACCENT = "var(--accent)"
BASELINE = "var(--baseline)"
NULL_BAND = "var(--null-band)"

HISTOGRAM_BINS = 61
CURVE_SAMPLES = 128


def _figure(body: str, *, title: str, caption: str) -> str:
    return (
        f'<figure class="fig"><figcaption><b>{svg.escape(title)}</b>'
        f"<span>{svg.escape(caption)}</span></figcaption>{body}</figure>"
    )


def _nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    if not np.isfinite([low, high]).all() or high <= low:
        return [low]
    return [float(value) for value in np.linspace(low, high, count)]


# --------------------------------------------------------------- metric distribution


def metric_distribution(values: np.ndarray, result: Comparison, *, width: float = 460.0) -> str:
    """Histogram of per-sample metric values, with the observed effect and the null band."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return _figure("<p class='empty'>no finite samples</p>", title=result.cohort, caption="")

    extent = float(np.percentile(np.abs(finite), 99.0)) or 1.0
    counts, edges = np.histogram(finite, bins=HISTOGRAM_BINS, range=(-extent, extent))
    frame = svg.Frame(
        width=width,
        height=190.0,
        x_min=-extent,
        x_max=extent,
        y_min=0.0,
        y_max=float(counts.max()) or 1.0,
    )

    parts = [svg.open_svg(frame, label=f"{result.metric} distribution for {result.cohort}")]

    # Null band first, so bars sit on top of it.
    band = max(result.null.spread, extent * 0.002)
    parts.append(
        svg.rect(
            frame.clamped_x(-band),
            frame.top,
            frame.clamped_x(band) - frame.clamped_x(-band),
            frame.plot_height,
            fill=NULL_BAND,
            opacity=0.55,
        )
    )

    bar_width = frame.plot_width / len(counts)
    for index, count in enumerate(counts):
        centre = (edges[index] + edges[index + 1]) / 2.0
        height = frame.plot_height * (float(count) / (float(counts.max()) or 1.0))
        parts.append(
            svg.rect(
                frame.x(centre) - bar_width / 2.0,
                frame.top + frame.plot_height - height,
                bar_width * 0.9,
                height,
                fill=ACCENT,
                opacity=0.75,
            )
        )

    bottom = frame.top + frame.plot_height
    parts.append(svg.line(frame.x(0.0), frame.top, frame.x(0.0), bottom, stroke=MUTED, dash="3 3"))
    parts.append(
        svg.line(
            frame.clamped_x(result.effect),
            frame.top,
            frame.clamped_x(result.effect),
            bottom,
            stroke=BASELINE,
            width=2.0,
        )
    )
    parts.append(svg.axes(frame, stroke=GRID))
    parts.append(svg.x_ticks(frame, _nice_ticks(-extent, extent), stroke=GRID, fill=MUTED))
    parts.append(
        svg.text(
            frame.left + frame.plot_width,
            frame.top + 10,
            f"n={result.count:,}",
            fill=MUTED,
            anchor="end",
        )
    )
    parts.append(svg.close_svg())

    unit = f" {result.unit}" if result.unit else ""
    caption = (
        f"median {result.effect:+.4f}{unit} (solid), |move| {result.magnitude:.4f}, "
        f"null spread ±{result.null.spread:.4f} (shaded), p={result.null.p_value:.3f} "
        f"— {result.verdict}"
    )
    return _figure("".join(parts), title=result.cohort, caption=caption)


# ------------------------------------------------------------------- response curves


def _neutral_ramp(count: int = CURVE_SAMPLES) -> RGB:
    ramp = np.linspace(0.001, 1.0, count)
    return np.stack([ramp, ramp, ramp], axis=-1)


def _hue_sweep(count: int = CURVE_SAMPLES, *, luma: float = 0.25, chroma: float = 0.6) -> RGB:
    angles = np.linspace(0.0, 360.0, count, endpoint=False)
    radians = np.deg2rad(angles)
    base = np.stack(
        [
            0.5 + chroma * 0.5 * np.cos(radians),
            0.5 + chroma * 0.5 * np.cos(radians - 2.0944),
            0.5 + chroma * 0.5 * np.cos(radians + 2.0944),
        ],
        axis=-1,
    )
    scale = luma / np.maximum(features.luma(base), 1e-9)
    return np.clip(base * scale[..., None], 0.0, 4.0)


def tone_response(baseline: Transform, candidate: Transform, *, width: float = 460.0) -> str:
    """Output luma against input luma for both transforms, on a neutral ramp."""
    ramp = _neutral_ramp()
    inputs = features.luma(ramp)
    before = features.luma(baseline(ramp))
    after = features.luma(candidate(ramp))
    top = float(max(before.max(), after.max(), inputs.max())) or 1.0

    frame = svg.Frame(width=width, height=210.0, x_min=0.0, x_max=1.0, y_min=0.0, y_max=top)
    parts = [svg.open_svg(frame, label="tone response")]
    parts.append(
        svg.polyline(
            [(frame.x(x), frame.y(x)) for x in np.linspace(0.0, min(1.0, top), 2)],
            stroke=GRID,
            width=1.0,
        )
    )
    parts.append(
        svg.polyline(
            [(frame.x(x), frame.y(y)) for x, y in zip(inputs, before, strict=True)],
            stroke=BASELINE,
        )
    )
    parts.append(
        svg.polyline(
            [(frame.x(x), frame.y(y)) for x, y in zip(inputs, after, strict=True)], stroke=ACCENT
        )
    )
    parts.append(svg.axes(frame, stroke=GRID))
    parts.append(svg.x_ticks(frame, _nice_ticks(0.0, 1.0), stroke=GRID, fill=MUTED))
    parts.append(svg.y_ticks(frame, _nice_ticks(0.0, top), stroke=GRID, fill=MUTED))
    parts.append(svg.close_svg())
    return _figure(
        "".join(parts),
        title="Tone response",
        caption="input luma against output luma on a neutral ramp; "
        "baseline in grey, candidate in colour, identity faint",
    )


def hue_response(baseline: Transform, candidate: Transform, *, width: float = 460.0) -> str:
    """Hue drift as a function of input hue — the chart that explains a cancelling effect."""
    sweep = _hue_sweep()
    inputs = features.hue_degrees(baseline(sweep))
    drift = metrics.hue_drift(baseline(sweep), candidate(sweep))
    order = np.argsort(inputs)
    inputs, drift = inputs[order], drift[order]

    limit = float(np.max(np.abs(drift))) or 1.0
    frame = svg.Frame(width=width, height=210.0, x_min=0.0, x_max=360.0, y_min=-limit, y_max=limit)
    parts = [svg.open_svg(frame, label="hue response")]
    parts.append(
        svg.line(
            frame.left, frame.y(0.0), frame.left + frame.plot_width, frame.y(0.0), stroke=MUTED
        )
    )

    # A hue strip along the axis, so the reader can see which colours move which way.
    strip_height = 8.0
    for index in range(72):
        angle = index * 5.0
        patch = _hue_sweep(count=72)[index : index + 1]
        colour = display.hex_colours(patch)[0]
        parts.append(
            svg.rect(
                frame.x(angle),
                frame.top + frame.plot_height + 4,
                frame.plot_width / 72.0 + 0.5,
                strip_height,
                fill=colour,
            )
        )

    parts.append(
        svg.polyline(
            [(frame.x(x), frame.y(y)) for x, y in zip(inputs, drift, strict=True)],
            stroke=ACCENT,
            width=2.0,
        )
    )
    parts.append(svg.axes(frame, stroke=GRID))
    parts.append(svg.y_ticks(frame, _nice_ticks(-limit, limit), stroke=GRID, fill=MUTED))
    parts.append(svg.close_svg())

    crossings = int(np.count_nonzero(np.diff(np.sign(drift)) != 0))
    note = (
        "the curve changes sign, so a cohort spanning the whole circle cancels to zero"
        if crossings >= 2
        else "the curve keeps one sign, so the effect has a consistent direction"
    )
    return _figure(
        "".join(parts),
        title="Hue response",
        caption=f"hue drift in degrees against input hue — {note}",
    )


# ----------------------------------------------------------------- cohort coverage


def cohort_coverage(
    table: SampleTable, *, title: str = "Cohort coverage", width: float = 460.0
) -> str:
    """Where the cohort sits in hue and luma — checks a selector against what it selected."""
    if len(table) == 0:
        return _figure("<p class='empty'>empty cohort</p>", title=title, caption="")

    hue = np.asarray(table.column("hue_deg"), dtype=np.float64)
    luma = np.asarray(table.column("luma_bt2020"), dtype=np.float64)
    encoded = np.clip(luma, 0.0, 1.0) ** (1.0 / 2.4)  # perceptual-ish, so shadows are visible

    columns, rows = 48, 28
    counts, _, _ = np.histogram2d(
        hue, encoded, bins=[columns, rows], range=[[0.0, 360.0], [0.0, 1.0]]
    )
    peak = float(counts.max()) or 1.0

    frame = svg.Frame(width=width, height=210.0, x_min=0.0, x_max=360.0, y_min=0.0, y_max=1.0)
    parts = [svg.open_svg(frame, label=f"{title} coverage")]
    cell_w = frame.plot_width / columns
    cell_h = frame.plot_height / rows
    for column in range(columns):
        for row in range(rows):
            count = counts[column, row]
            if count <= 0:
                continue
            parts.append(
                svg.rect(
                    frame.left + column * cell_w,
                    frame.top + frame.plot_height - (row + 1) * cell_h,
                    cell_w + 0.4,
                    cell_h + 0.4,
                    fill=ACCENT,
                    opacity=0.12 + 0.88 * float(count) / peak,
                )
            )
    parts.append(svg.axes(frame, stroke=GRID))
    parts.append(
        svg.x_ticks(frame, [0.0, 90.0, 180.0, 270.0, 360.0], stroke=GRID, fill=MUTED, fmt="{:.0f}")
    )
    parts.append(svg.y_ticks(frame, [0.0, 0.5, 1.0], stroke=GRID, fill=MUTED, fmt="{:.1f}"))
    parts.append(svg.close_svg())
    return _figure(
        "".join(parts),
        title=title,
        caption=f"hue (degrees) against encoded luma, {len(table):,} samples",
    )


# ------------------------------------------------------------------------- samples


def sample_swatches(
    table: SampleTable,
    *,
    baseline: Transform,
    candidate: Transform,
    count: int = 24,
    title: str = "Samples",
) -> str:
    """Representative samples before and after, spread evenly across the cohort's luma range."""
    if len(table) == 0:
        return _figure("<p class='empty'>empty cohort</p>", title=title, caption="")

    luma = np.asarray(table.column("luma_bt2020"), dtype=np.float64)
    order = np.argsort(luma)
    picks = order[np.linspace(0, len(order) - 1, min(count, len(order))).astype(int)]
    chosen = table.rgb[picks]

    before = display.hex_colours(baseline(chosen))
    after_linear = candidate(chosen)
    after = display.hex_colours(after_linear)
    clipped = display.clipped_fraction(after_linear)

    size, gap = 22.0, 3.0
    height = size * 2 + gap + 16
    frame = svg.Frame(width=len(picks) * (size + gap) + 40, height=height, left=38, bottom=0, top=2)
    parts = [svg.open_svg(frame, label=f"{title} before and after")]
    parts.append(svg.text(0, size * 0.7, "before", fill=MUTED, size=9))
    parts.append(svg.text(0, size + gap + size * 0.7, "after", fill=MUTED, size=9))
    for index, (start, end) in enumerate(zip(before, after, strict=True)):
        x = 38 + index * (size + gap)
        parts.append(svg.swatch(x, 2, size, fill=start, stroke=GRID))
        parts.append(svg.swatch(x, 2 + size + gap, size, fill=end, stroke=GRID))
    parts.append(svg.close_svg())

    caption = f"{len(picks)} samples ordered by luma"
    if clipped > 0.0005:
        caption += f" — {clipped:.1%} of channel values fall outside [0,1] and are clipped here"
    return _figure("".join(parts), title=title, caption=caption)


__all__ = [
    "cohort_coverage",
    "hue_response",
    "metric_distribution",
    "sample_swatches",
    "tone_response",
]
