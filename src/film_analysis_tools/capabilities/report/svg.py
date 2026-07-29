"""Minimal SVG primitives for analytical charts.

Hand-rolled rather than plotted with a library, for three reasons: it keeps the dependency set
at NumPy alone, the output is text that diffs and embeds directly in a page, and the charts
this system needs are simple enough that a plotting framework would be more configuration than
drawing.

Everything here returns SVG element strings. Colours come from the caller so a page can theme
itself; nothing here decides what anything means.
"""

from __future__ import annotations

import html
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


def escape(text: object) -> str:
    return html.escape(str(text), quote=True)


def _n(value: float) -> str:
    """Format a coordinate compactly and deterministically."""
    if not (value == value and abs(value) != float("inf")):  # NaN or infinite
        return "0"
    return f"{value:.2f}".rstrip("0").rstrip(".") or "0"


@dataclass(frozen=True)
class Frame:
    """A drawing area with a data-space to pixel-space mapping."""

    width: float
    height: float
    left: float = 48.0
    right: float = 12.0
    top: float = 12.0
    bottom: float = 34.0
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0

    @property
    def plot_width(self) -> float:
        return max(1.0, self.width - self.left - self.right)

    @property
    def plot_height(self) -> float:
        return max(1.0, self.height - self.top - self.bottom)

    def x(self, value: float) -> float:
        span = self.x_max - self.x_min or 1.0
        return self.left + (value - self.x_min) / span * self.plot_width

    def y(self, value: float) -> float:
        span = self.y_max - self.y_min or 1.0
        return self.top + self.plot_height - (value - self.y_min) / span * self.plot_height

    def clamped_x(self, value: float) -> float:
        return min(max(self.x(value), self.left), self.left + self.plot_width)


def open_svg(frame: Frame, *, label: str) -> str:
    return (
        f'<svg viewBox="0 0 {_n(frame.width)} {_n(frame.height)}" '
        f'role="img" aria-label="{escape(label)}" class="chart">'
    )


def close_svg() -> str:
    return "</svg>"


def rect(
    x: float, y: float, width: float, height: float, *, fill: str, opacity: float = 1.0
) -> str:
    return (
        f'<rect x="{_n(x)}" y="{_n(y)}" width="{_n(max(0.0, width))}" '
        f'height="{_n(max(0.0, height))}" fill="{fill}" opacity="{_n(opacity)}"/>'
    )


def line(
    x1: float, y1: float, x2: float, y2: float, *, stroke: str, width: float = 1.0, dash: str = ""
) -> str:
    dasharray = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{_n(x1)}" y1="{_n(y1)}" x2="{_n(x2)}" y2="{_n(y2)}" '
        f'stroke="{stroke}" stroke-width="{_n(width)}"{dasharray}/>'
    )


def polyline(points: Iterable[tuple[float, float]], *, stroke: str, width: float = 1.5) -> str:
    rendered = " ".join(f"{_n(x)},{_n(y)}" for x, y in points)
    return (
        f'<polyline points="{rendered}" fill="none" stroke="{stroke}" '
        f'stroke-width="{_n(width)}" stroke-linejoin="round"/>'
    )


def text(
    x: float,
    y: float,
    body: object,
    *,
    fill: str,
    anchor: str = "start",
    size: float = 10.0,
    weight: str = "normal",
) -> str:
    return (
        f'<text x="{_n(x)}" y="{_n(y)}" fill="{fill}" font-size="{_n(size)}" '
        f'text-anchor="{anchor}" font-weight="{weight}">{escape(body)}</text>'
    )


def axes(frame: Frame, *, stroke: str) -> str:
    bottom = frame.top + frame.plot_height
    return line(frame.left, bottom, frame.left + frame.plot_width, bottom, stroke=stroke) + line(
        frame.left, frame.top, frame.left, bottom, stroke=stroke
    )


def x_ticks(
    frame: Frame, values: Sequence[float], *, stroke: str, fill: str, fmt: str = "{:.3g}"
) -> str:
    bottom = frame.top + frame.plot_height
    parts: list[str] = []
    for value in values:
        position = frame.x(value)
        parts.append(line(position, bottom, position, bottom + 4, stroke=stroke))
        parts.append(text(position, bottom + 15, fmt.format(value), fill=fill, anchor="middle"))
    return "".join(parts)


def y_ticks(
    frame: Frame, values: Sequence[float], *, stroke: str, fill: str, fmt: str = "{:.3g}"
) -> str:
    parts: list[str] = []
    for value in values:
        position = frame.y(value)
        parts.append(line(frame.left - 4, position, frame.left, position, stroke=stroke))
        parts.append(text(frame.left - 7, position + 3, fmt.format(value), fill=fill, anchor="end"))
    return "".join(parts)


def swatch(x: float, y: float, size: float, *, fill: str, stroke: str = "") -> str:
    outline = f' stroke="{stroke}" stroke-width="0.5"' if stroke else ""
    return (
        f'<rect x="{_n(x)}" y="{_n(y)}" width="{_n(size)}" height="{_n(size)}" '
        f'rx="2" fill="{fill}"{outline}/>'
    )


__all__ = [
    "Frame",
    "axes",
    "close_svg",
    "escape",
    "line",
    "open_svg",
    "polyline",
    "rect",
    "swatch",
    "text",
    "x_ticks",
    "y_ticks",
]
