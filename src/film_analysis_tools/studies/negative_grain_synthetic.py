"""Controlled N1/N2 field and strength comparison on an exact FEE synthetic chain.

FEE owns the scene-linear stimulus, Sony encoding, compiled graph execution, and original float
outputs. This study owns the signed-delta measurements, fixed-scale review encodings, video
assembly, and compact decision report. It is intentionally one bounded experiment rather than a
new validation framework.
"""

from __future__ import annotations

import math
import multiprocessing
import shutil
import subprocess
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from film_analysis_tools.capabilities.colour.display import srgb_encode
from film_analysis_tools.capabilities.report import svg
from film_analysis_tools.capabilities.report.html import STYLE
from film_analysis_tools.core.errors import DataError
from film_analysis_tools.core.io import write_json
from film_analysis_tools.core.tiers import Tier
from film_analysis_tools.forward.negative_grain_synthetic import (
    NegativeGrainFrame,
    NegativeGrainRegion,
    NegativeGrainSyntheticAdapter,
)

LUMA_WEIGHTS = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float64)
HISTOGRAM_MAX = 0.25
HISTOGRAM_BINS = 4096
SPECTRUM_BINS = 24
SPATIAL_FRAME_STRIDE = 4
FIXED_DELTA_DISPLAY_LIMIT = 0.05
POSITIVE_CONTROL_STRENGTH = 1.57501286


@dataclass(frozen=True, slots=True)
class _StudyDeclaration:
    question: str
    rationale: str
    assumptions: tuple[str, ...]
    controls: tuple[str, ...]
    tier: Tier
    falsified_by: str
    supersedes: tuple[str, ...] = ()


STUDY = _StudyDeclaration(
    question=(
        "Does N2's accepted structured field improve N1's white field, and which bounded N2 "
        "strength is the best engineering default?"
    ),
    rationale=(
        "Private motion review preferred N2 placement and tonal purity but found the earlier "
        "1.57501286 midtone match visibly too chromatic on skin."
    ),
    assumptions=(
        "FEE resolves the exact N1 v1 and N2 v2 bundles before rendering.",
        "The positive control exposes more chromatic activity than N2 strength 1.0.",
        "Every metric uses the original floating-point grain-on minus N0 delta.",
    ),
    controls=("null",),
    tier=Tier.COMPARISON,
    falsified_by=(
        "N2 fails to improve structured-field behaviour, the positive control is not more "
        "chromatic than 1.0, or a bounded strength introduces recurring tint or instability."
    ),
)


@dataclass(frozen=True, slots=True)
class SyntheticGrainRunConfig:
    n1_bundle: Path
    n2_bundle: Path
    output_dir: Path
    width: int = 1920
    height: int = 1080
    frame_count: int = 96
    seed: int = 20260731
    frame_workers: int = 4
    variant_workers: int = 1
    delta_display_limit: float = FIXED_DELTA_DISPLAY_LIMIT
    make_videos: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.delta_display_limit) or self.delta_display_limit <= 0.0:
            raise DataError("delta display limit must be finite and positive")
        if self.frame_workers < 1 or self.frame_workers > 12:
            raise DataError("frame workers must be between 1 and 12")
        if self.variant_workers < 1 or self.variant_workers > 6:
            raise DataError("variant workers must be between 1 and 6")


def _crossing(line: np.ndarray, threshold: float = 0.5) -> float:
    below = np.flatnonzero(line < threshold)
    if below.size == 0:
        return float(line.size)
    first = int(below[0])
    if first == 0:
        return 0.5
    high = float(line[first - 1])
    low = float(line[first])
    return float(first - 1 + (high - threshold) / max(high - low, 1.0e-12))


def _radial_profile(power: np.ndarray) -> tuple[list[float], list[float]]:
    height, width = power.shape
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    radius = np.hypot(fy, fx)
    edges = np.linspace(0.0, 0.5, SPECTRUM_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    profile = np.zeros(SPECTRUM_BINS, dtype=np.float64)
    for index in range(SPECTRUM_BINS):
        selected = (radius >= edges[index]) & (radius < edges[index + 1])
        profile[index] = float(np.mean(power[selected])) if np.any(selected) else np.nan
    band = (centers >= 0.05) & (centers <= 0.45) & np.isfinite(profile)
    normalizer = float(np.mean(profile[band])) if np.any(band) else 1.0
    profile /= max(normalizer, 1.0e-12)
    return centers.tolist(), profile.tolist()


def _spectrum_summary(power_sum: np.ndarray | None, frames: int) -> dict[str, Any]:
    if power_sum is None or frames == 0 or not np.any(power_sum > 0.0):
        return {
            "frames": frames,
            "frequencies_cycles_per_pixel": [],
            "radial_psd_normalized": [],
            "radius_h_px": 0.0,
            "radius_v_px": 0.0,
            "anisotropy_h_over_v": 1.0,
        }
    power = power_sum / frames
    autocorrelation = np.fft.ifft2(power).real
    autocorrelation /= max(float(autocorrelation[0, 0]), 1.0e-12)
    radius_h = _crossing(autocorrelation[0, : max(2, power.shape[1] // 2)])
    radius_v = _crossing(autocorrelation[: max(2, power.shape[0] // 2), 0])
    frequencies, radial = _radial_profile(power)
    return {
        "frames": frames,
        "frequencies_cycles_per_pixel": frequencies,
        "radial_psd_normalized": radial,
        "radius_h_px": radius_h,
        "radius_v_px": radius_v,
        "anisotropy_h_over_v": radius_h / max(radius_v, 1.0e-12),
    }


def _histogram_quantile(histogram: np.ndarray, quantile: float) -> float:
    total = int(np.sum(histogram))
    if total == 0:
        return 0.0
    target = max(1, math.ceil(total * quantile))
    index = int(np.searchsorted(np.cumsum(histogram), target, side="left"))
    return (index + 0.5) * HISTOGRAM_MAX / HISTOGRAM_BINS


@dataclass(slots=True)
class RegionAccumulator:
    """Streaming original-float metrics for one variant and tracked/static interior."""

    region_id: str
    category: str
    description: str
    motion: str
    count: int = 0
    rgb_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    rgb_square_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    luma_sum: float = 0.0
    luma_square_sum: float = 0.0
    opponent_sum: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    opponent_square_sum: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    output_contacts_floor: int = 0
    output_contacts_ceiling: int = 0
    histogram: np.ndarray = field(
        default_factory=lambda: np.zeros(HISTOGRAM_BINS + 1, dtype=np.int64)
    )
    adjacent_correlations: list[float] = field(default_factory=list)
    previous_luma: np.ndarray | None = None
    power_sum: np.ndarray | None = None
    spatial_frames: int = 0
    scene_luma_values: list[float] = field(default_factory=list)
    exposure_values: list[float] = field(default_factory=list)

    def update(
        self,
        delta: np.ndarray,
        output: np.ndarray,
        region: NegativeGrainRegion,
        *,
        frame_index: int,
    ) -> None:
        top, bottom, left, right = region.bounds_yxyx
        crop = np.asarray(delta[top:bottom, left:right], dtype=np.float64)
        rendered = np.asarray(output[top:bottom, left:right], dtype=np.float64)
        if crop.size == 0:
            raise DataError(f"measurement region {region.region_id!r} is empty")
        pixels = crop.reshape(-1, 3)
        luma = pixels @ LUMA_WEIGHTS
        opponents = np.stack((pixels[:, 0] - pixels[:, 1], pixels[:, 2] - pixels[:, 1]), axis=1)
        magnitude = np.sqrt(np.mean(pixels * pixels, axis=1))

        self.count += pixels.shape[0]
        self.rgb_sum += np.sum(pixels, axis=0)
        self.rgb_square_sum += np.sum(pixels * pixels, axis=0)
        self.luma_sum += float(np.sum(luma))
        self.luma_square_sum += float(luma @ luma)
        self.opponent_sum += np.sum(opponents, axis=0)
        self.opponent_square_sum += np.sum(opponents * opponents, axis=0)
        self.output_contacts_floor += int(np.count_nonzero(rendered <= 0.0))
        self.output_contacts_ceiling += int(np.count_nonzero(rendered >= 1.0))
        indices = np.minimum(
            np.floor(magnitude * HISTOGRAM_BINS / HISTOGRAM_MAX).astype(np.int64),
            HISTOGRAM_BINS,
        )
        self.histogram += np.bincount(indices, minlength=HISTOGRAM_BINS + 1)

        luma_crop = np.ascontiguousarray(crop @ LUMA_WEIGHTS)
        if self.previous_luma is not None and self.previous_luma.shape == luma_crop.shape:
            previous = self.previous_luma - float(np.mean(self.previous_luma))
            current = luma_crop - float(np.mean(luma_crop))
            denominator = float(np.sqrt(np.sum(previous**2) * np.sum(current**2)))
            if denominator > 1.0e-20:
                self.adjacent_correlations.append(float(np.sum(previous * current) / denominator))
        self.previous_luma = luma_crop.copy()

        if frame_index % SPATIAL_FRAME_STRIDE == 0:
            centered = luma_crop - float(np.mean(luma_crop))
            power = np.abs(np.fft.fft2(centered)) ** 2
            self.power_sum = power if self.power_sum is None else self.power_sum + power
            self.spatial_frames += 1
        self.scene_luma_values.append(region.scene_luma)
        if region.exposure_stops_from_grey is not None:
            self.exposure_values.append(region.exposure_stops_from_grey)

    def as_record(self) -> dict[str, Any]:
        if self.count == 0:
            raise DataError(f"measurement region {self.region_id!r} received no samples")
        rgb_mean = self.rgb_sum / self.count
        rgb_variance = np.maximum(self.rgb_square_sum / self.count - rgb_mean**2, 0.0)
        luma_mean = self.luma_sum / self.count
        luma_variance = max(self.luma_square_sum / self.count - luma_mean**2, 0.0)
        opponent_mean = self.opponent_sum / self.count
        opponent_variance = np.maximum(
            self.opponent_square_sum / self.count - opponent_mean**2,
            0.0,
        )
        luma_rms = math.sqrt(luma_variance)
        opponent_rms = math.sqrt(float(np.mean(opponent_variance)))
        correlations = np.asarray(self.adjacent_correlations, dtype=np.float64)
        total_output_channels = self.count * 3
        return {
            "region_id": self.region_id,
            "category": self.category,
            "description": self.description,
            "motion": self.motion,
            "samples": self.count,
            "scene_luma_median": float(np.median(self.scene_luma_values)),
            "scene_luma_range": [
                float(np.min(self.scene_luma_values)),
                float(np.max(self.scene_luma_values)),
            ],
            "exposure_stops_median": (
                float(np.median(self.exposure_values)) if self.exposure_values else None
            ),
            "temporal_luma_rms": luma_rms,
            "temporal_rgb_rms": np.sqrt(rgb_variance).tolist(),
            "opponent_rms": opponent_rms,
            "opponent_to_luma_ratio": opponent_rms / max(luma_rms, 1.0e-20),
            "mean_rgb_delta": rgb_mean.tolist(),
            "mean_luma_delta": luma_mean,
            "p95_delta_magnitude": _histogram_quantile(self.histogram, 0.95),
            "p99_delta_magnitude": _histogram_quantile(self.histogram, 0.99),
            "delta_histogram_overflow_fraction": float(self.histogram[-1])
            / float(np.sum(self.histogram)),
            "adjacent_frame_correlation": {
                "median": float(np.median(correlations)) if correlations.size else None,
                "p10": float(np.percentile(correlations, 10)) if correlations.size else None,
                "p90": float(np.percentile(correlations, 90)) if correlations.size else None,
                "pairs": int(correlations.size),
            },
            "output_floor_contact_fraction": self.output_contacts_floor / total_output_channels,
            "output_ceiling_contact_fraction": self.output_contacts_ceiling / total_output_channels,
            "spatial": _spectrum_summary(self.power_sum, self.spatial_frames),
        }


def signed_delta_view(delta: np.ndarray, *, limit: float) -> np.ndarray:
    """Fixed-scale signed RGB view: zero is neutral grey, never black."""
    return np.clip(0.5 + np.asarray(delta, dtype=np.float64) / (2.0 * limit), 0.0, 1.0)


def magnitude_delta_view(delta: np.ndarray, *, limit: float) -> np.ndarray:
    """Fixed-scale absolute RGB magnitude view on black."""
    return np.clip(np.abs(np.asarray(delta, dtype=np.float64)) / limit, 0.0, 1.0)


def _review_downsample(array: np.ndarray) -> np.ndarray:
    height, width = array.shape[:2]
    factor = max(1, math.gcd(width // min(width, 640), height // min(height, 360)))
    factor = min(factor, max(1, width // 640), max(1, height // 360))
    if factor <= 1:
        return np.asarray(array)
    cropped_height = height // factor * factor
    cropped_width = width // factor * factor
    cropped = np.asarray(array[:cropped_height, :cropped_width], dtype=np.float64)
    return cropped.reshape(
        cropped_height // factor,
        factor,
        cropped_width // factor,
        factor,
        3,
    ).mean(axis=(1, 3))


def review_triptych(output: np.ndarray, baseline: np.ndarray, *, limit: float) -> np.ndarray:
    """Normal / signed-grey / magnitude-black review frame at one immutable scale."""
    normal = srgb_encode(_review_downsample(output))
    delta = _review_downsample(np.asarray(output) - np.asarray(baseline))
    signed = signed_delta_view(delta, limit=limit)
    magnitude = magnitude_delta_view(delta, limit=limit)
    composite = np.concatenate((normal, signed, magnitude), axis=1)
    return np.rint(composite * 255.0).clip(0, 255).astype(np.uint8)


class _VideoWriter:
    def __init__(self, path: Path, shape: tuple[int, int], fps: float) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise DataError("ffmpeg is required to assemble the synthetic review videos")
        path.parent.mkdir(parents=True, exist_ok=True)
        height, width = shape
        self.path = path
        self._process = subprocess.Popen(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps:.12g}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if self._process.stdin is None:
            raise DataError(f"video writer for {self.path.name} has no input stream")
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            raise DataError(f"ffmpeg stopped while writing {self.path.name}") from exc

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        error = (
            self._process.stderr.read().decode("utf-8", errors="replace")
            if self._process.stderr is not None
            else ""
        )
        return_code = self._process.wait()
        if return_code != 0:
            raise DataError(f"ffmpeg failed for {self.path.name}: {error.strip()[:400]}")


def _probe_positive_control(adapter: NegativeGrainSyntheticAdapter) -> dict[str, float]:
    values: dict[str, list[np.ndarray]] = {"n2_100": [], "n2_1575": []}
    for frame_index in range(min(6, adapter.frame_count)):
        frame = adapter.render_frame(frame_index)
        baseline = frame.outputs["n0"]
        region = next(region for region in frame.regions if region.region_id == "skin_texture")
        top, bottom, left, right = region.bounds_yxyx
        for variant_id in values:
            delta = (
                frame.outputs[variant_id][top:bottom, left:right] - baseline[top:bottom, left:right]
            )
            opponent = np.stack(
                (delta[..., 0] - delta[..., 1], delta[..., 2] - delta[..., 1]),
                axis=-1,
            )
            values[variant_id].append(opponent)
    rms = {
        variant_id: float(np.sqrt(np.mean(np.stack(frames, axis=0) ** 2)))
        for variant_id, frames in values.items()
    }
    ratio = rms["n2_1575"] / max(rms["n2_100"], 1.0e-20)
    if ratio <= 1.20:
        raise DataError(
            "positive-control probe failed: N2 1.57501286 does not expose at least 20% more "
            f"skin-proxy opponent activity than N2 1.0 (ratio {ratio:.3f})"
        )
    return {**rms, "positive_control_to_native_ratio": ratio}


def _line_chart(
    title: str,
    subtitle: str,
    series: dict[str, list[tuple[float, float]]],
    *,
    x_label: str,
    y_label: str,
) -> str:
    points = [point for values in series.values() for point in values if np.all(np.isfinite(point))]
    if not points:
        return ""
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    y_min = min(0.0, min(ys))
    y_max = max(ys)
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0
    frame = svg.Frame(width=520, height=270, x_min=min(xs), x_max=max(xs), y_min=y_min, y_max=y_max)
    colours = ("#2563eb", "#d97706", "#059669", "#dc2626", "#7c3aed")
    parts = [svg.open_svg(frame, label=title), svg.axes(frame, stroke="#6b7280")]
    for (name, values), colour in zip(series.items(), colours, strict=False):
        ordered = sorted((x, y) for x, y in values if np.all(np.isfinite((x, y))))
        parts.append(svg.polyline(((frame.x(x), frame.y(y)) for x, y in ordered), stroke=colour))
        if ordered:
            parts.append(
                svg.text(frame.left + 8, frame.top + 14 * (len(parts) - 1), name, fill=colour)
            )
    parts.extend(
        (
            svg.text(frame.width / 2, frame.height - 4, x_label, fill="#6b7280", anchor="middle"),
            svg.text(4, 12, y_label, fill="#6b7280"),
            svg.close_svg(),
        )
    )
    return (
        f"<figure class='fig'><figcaption><b>{svg.escape(title)}</b>"
        f"<span>{svg.escape(subtitle)}</span></figcaption>{''.join(parts)}</figure>"
    )


def _metric_series(
    records: dict[str, dict[str, dict[str, Any]]],
    metric: str,
) -> dict[str, list[tuple[float, float]]]:
    wanted = ("n1_100", "n2_075", "n2_100", "n2_125", "n2_1575")
    result: dict[str, list[tuple[float, float]]] = {}
    for variant_id in wanted:
        values = []
        for record in records[variant_id].values():
            if record["category"] != "neutral_step":
                continue
            exposure = record["exposure_stops_median"]
            value = record[metric]
            if exposure is not None:
                values.append((float(exposure), float(value)))
        result[variant_id] = values
    return result


def _spectrum_chart(records: dict[str, dict[str, dict[str, Any]]]) -> str:
    series: dict[str, list[tuple[float, float]]] = {}
    for variant_id in ("n1_100", "n2_100"):
        spatial = records[variant_id]["neutral_step_05"]["spatial"]
        series[variant_id] = list(
            zip(
                spatial["frequencies_cycles_per_pixel"],
                spatial["radial_psd_normalized"],
                strict=True,
            )
        )
    return _line_chart(
        "N1 and N2 spatial spectra at strength 1.0",
        "Question: does N2 replace the white field with the accepted structure? Calculated from "
        "the original signed luma delta in the static 18% scene patch.",
        series,
        x_label="cycles per pixel",
        y_label="normalized PSD",
    )


def _strength_ratios(
    records: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for region_id, native in records["n2_100"].items():
        denominator = max(float(native["temporal_luma_rms"]), 1.0e-20)
        result[region_id] = {
            "n2_075_over_100": float(records["n2_075"][region_id]["temporal_luma_rms"])
            / denominator,
            "n2_125_over_100": float(records["n2_125"][region_id]["temporal_luma_rms"])
            / denominator,
            "n2_1575_over_100": float(records["n2_1575"][region_id]["temporal_luma_rms"])
            / denominator,
        }
    return result


def _report_html(
    *,
    adapter: NegativeGrainSyntheticAdapter,
    config: SyntheticGrainRunConfig,
    records: dict[str, dict[str, dict[str, Any]]],
    strength_ratios: dict[str, dict[str, float]],
    positive_probe: dict[str, float],
    videos: dict[str, str],
) -> str:
    rms_chart = _line_chart(
        "Output grain RMS versus known scene exposure",
        "Question: where do N1 and bounded N2 strengths place visible texture? Calculated from "
        "grain-on minus N0 before display encoding; predicts weak shadows or overactive "
        "highlights.",
        _metric_series(records, "temporal_luma_rms"),
        x_label="stops from 18% scene grey",
        y_label="luma RMS",
    )
    chroma_chart = _line_chart(
        "Chromatic-to-luma grain ratio versus exposure",
        "Question: does a strength become colour-noise dominated? The opponent RMS is divided by "
        "luma RMS; rising values predict coloured speckling and skin contamination.",
        _metric_series(records, "opponent_to_luma_ratio"),
        x_label="stops from 18% scene grey",
        y_label="opponent / luma",
    )
    drift_chart = _line_chart(
        "Mean luma drift versus exposure",
        "Question: does nonlinear density propagation move the temporal mean? A persistent trend "
        "predicts brightness bias; RGB drift remains available in metrics.json for tint diagnosis.",
        _metric_series(records, "mean_luma_delta"),
        x_label="stops from 18% scene grey",
        y_label="mean luma delta",
    )
    video_cards = "".join(
        f"<figure class='fig'><figcaption><b>{svg.escape(variant.label)}</b>"
        f"<span>{svg.escape(variant.role)}</span></figcaption>"
        "<video controls loop muted playsinline preload='metadata' "
        f"src='{svg.escape(videos[variant.variant_id])}'></video>"
        "<p class='sub'>Left: normal output. Centre: signed delta with zero at neutral grey. "
        "Right: absolute magnitude on black.</p></figure>"
        for variant in adapter.variants
        if variant.variant_id in videos
    )
    rows = "".join(
        "<tr>"
        f"<td>{svg.escape(region_id)}</td>"
        f"<td>{values['n2_075_over_100']:.3f}</td>"
        f"<td>{values['n2_125_over_100']:.3f}</td>"
        f"<td>{values['n2_1575_over_100']:.3f}</td>"
        "</tr>"
        for region_id, values in sorted(strength_ratios.items())
    )
    identity = adapter.bundle_identity
    body = f"""<main>
<h1>Controlled N1/N2 synthetic grain review</h1>
<p class='sub'><span class='tier'>comparison</span>&nbsp; Two questions, kept separate:
does N2's structured field improve N1 at strength 1.0; and which of N2 0.75/1.0/1.25 is the
best practical default? N2 1.57501286 is a labelled positive control, not a candidate.</p>
<dl class='meta'>
<div><dt>stimulus</dt><dd>{svg.escape(adapter.clip_id)}</dd></div>
<div><dt>geometry</dt><dd>{adapter.width}x{adapter.height}, {adapter.frame_count} frames</dd></div>
<div><dt>N1 model</dt><dd>{svg.escape(identity["n1"]["model_id"])}</dd></div>
<div><dt>N2 model</dt><dd>{svg.escape(identity["n2"]["model_id"])}</dd></div>
<div><dt>seed</dt><dd>{config.seed}</dd></div>
<div><dt>delta display scale</dt><dd>fixed ±{config.delta_display_limit:.4f} linear RGB</dd></div>
</dl>
<div class='note'><b>Signed-delta rule.</b> Black cannot display the negative half of grain.
All centre panels map an exact zero delta to neutral grey, and all right panels show
<code>abs(grain_on - N0)</code> on black. No frame is normalized independently. Every metric uses
the original floating-point signed delta before clipping or encoding.</div>
<div class='note'><b>Probe before report.</b> On the textured skin proxy, the positive control's
opponent RMS is {positive_probe["positive_control_to_native_ratio"]:.3f}x N2 1.0. The synthetic
test therefore exposes the known excess chromatic activity instead of hiding it.</div>
<h2>Synchronized review videos</h2>
<p class='sub'>View at ordinary scale. Synthetic skin colour and texture are controlled proxies,
not evidence that a strength looks natural on a real face.</p>
<div class='grid'>{video_cards}</div>
<h2>Exposure and colour behaviour</h2>
<div class='grid'>{rms_chart}{chroma_chart}{drift_chart}{_spectrum_chart(records)}</div>
<h2>N2 strength response by region</h2>
<p class='sub'>Ratios use original-float temporal luma RMS. This is the controlled evidence for
whether 0.75 becomes too weak or 1.25 becomes unnecessarily active.</p>
<div class='wrap'><table><tr><th>region</th><th>0.75 / 1.0</th><th>1.25 / 1.0</th>
<th>1.575 / 1.0</th></tr>{rows}</table></div>
<h2>Interpretation boundary</h2>
<p>This report can decide structured versus white field behaviour, bounded strength response,
temporal instability, persistent tint, and controlled chromatic activity. It cannot establish
natural skin appearance. After selecting one bounded value, only a few natural clips should be
used as a final veto—not another corpus-wide fit.</p>
<p><b>Standing architecture decision:</b> N2 placement and shared A1 structure remain accepted;
1.57501286 remains rejected for practical use; 1.0 remains the leading candidate pending this
controlled viewing decision. No hybrid, sublayers, or covariance modes are introduced.</p>
<footer>Generated {datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")} by film-analysis-tools.
Detailed original-float records: <a href='metrics.json'>metrics.json</a>.</footer>
</main>"""
    report_style = (
        STYLE + "\nvideo { width: 100%; height: auto; background: #000; display: block; }"
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Controlled N1/N2 synthetic grain review</title><style>{report_style}</style>"
        f"</head><body>{body}</body></html>\n"
    )


_FRAME_WORKER_ADAPTER: NegativeGrainSyntheticAdapter | None = None


def _initialize_frame_worker(
    n1_bundle: str,
    n2_bundle: str,
    width: int,
    height: int,
    frame_count: int,
    seed: int,
    variant_workers: int,
) -> None:
    global _FRAME_WORKER_ADAPTER
    _FRAME_WORKER_ADAPTER = NegativeGrainSyntheticAdapter(
        Path(n1_bundle),
        Path(n2_bundle),
        width=width,
        height=height,
        frame_count=frame_count,
        seed=seed,
        max_workers=variant_workers,
    )


def _render_frame_worker(frame_index: int) -> NegativeGrainFrame:
    if _FRAME_WORKER_ADAPTER is None:
        raise RuntimeError("negative-grain frame worker was not initialized")
    return _FRAME_WORKER_ADAPTER.render_frame(frame_index)


def _rendered_frames(
    adapter: NegativeGrainSyntheticAdapter,
    config: SyntheticGrainRunConfig,
) -> Iterator[NegativeGrainFrame]:
    if config.frame_workers == 1:
        for frame_index in range(adapter.frame_count):
            yield adapter.render_frame(frame_index)
        return

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=config.frame_workers,
        mp_context=context,
        initializer=_initialize_frame_worker,
        initargs=(
            str(config.n1_bundle),
            str(config.n2_bundle),
            config.width,
            config.height,
            config.frame_count,
            config.seed,
            config.variant_workers,
        ),
    ) as executor:
        pending: dict[int, Future[NegativeGrainFrame]] = {}
        next_to_submit = 0
        while next_to_submit < min(config.frame_workers, adapter.frame_count):
            pending[next_to_submit] = executor.submit(_render_frame_worker, next_to_submit)
            next_to_submit += 1
        for frame_index in range(adapter.frame_count):
            frame = pending.pop(frame_index).result()
            if next_to_submit < adapter.frame_count:
                pending[next_to_submit] = executor.submit(
                    _render_frame_worker,
                    next_to_submit,
                )
                next_to_submit += 1
            yield frame


def run(
    config: SyntheticGrainRunConfig,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Run the bounded comparison and write metrics, synchronized videos, and one report."""
    probe_adapter = NegativeGrainSyntheticAdapter(
        config.n1_bundle,
        config.n2_bundle,
        width=320,
        height=180,
        frame_count=24,
        seed=config.seed,
        max_workers=min(3, config.variant_workers),
    )
    positive_probe = _probe_positive_control(probe_adapter)
    adapter = NegativeGrainSyntheticAdapter(
        config.n1_bundle,
        config.n2_bundle,
        width=config.width,
        height=config.height,
        frame_count=config.frame_count,
        seed=config.seed,
        max_workers=1,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    variants = {variant.variant_id: variant for variant in adapter.variants}
    accumulators: dict[str, dict[str, RegionAccumulator]] = {
        variant_id: {} for variant_id in variants if variant_id != "n0"
    }
    writers: dict[str, _VideoWriter] = {}
    videos: dict[str, str] = {}
    try:
        for frame in _rendered_frames(adapter, config):
            frame_index = frame.frame_index
            baseline = frame.outputs["n0"]
            if config.make_videos:
                for variant_id, output in frame.outputs.items():
                    review = review_triptych(
                        output,
                        baseline,
                        limit=config.delta_display_limit,
                    )
                    writer = writers.get(variant_id)
                    if writer is None:
                        relative = f"{variant_id}.mp4"
                        writer = _VideoWriter(
                            config.output_dir / relative,
                            review.shape[:2],
                            adapter.cadence[0] / adapter.cadence[1],
                        )
                        writers[variant_id] = writer
                        videos[variant_id] = relative
                    writer.write(review)
            for variant_id, by_region in accumulators.items():
                output = frame.outputs[variant_id]
                delta = np.asarray(output) - np.asarray(baseline)
                for region in frame.regions:
                    accumulator = by_region.get(region.region_id)
                    if accumulator is None:
                        accumulator = RegionAccumulator(
                            region_id=region.region_id,
                            category=region.category,
                            description=region.description,
                            motion=region.motion,
                        )
                        by_region[region.region_id] = accumulator
                    accumulator.update(
                        delta,
                        output,
                        region,
                        frame_index=frame.frame_index,
                    )
            if progress is not None:
                progress(frame_index + 1, adapter.frame_count)
    finally:
        close_errors: list[Exception] = []
        for writer in writers.values():
            try:
                writer.close()
            except Exception as exc:  # preserve all encoder failures after the render loop
                close_errors.append(exc)
        if close_errors:
            raise close_errors[0]

    records = {
        variant_id: {
            region_id: accumulator.as_record()
            for region_id, accumulator in sorted(by_region.items())
        }
        for variant_id, by_region in accumulators.items()
    }
    full_positive_ratio = float(records["n2_1575"]["skin_texture"]["opponent_rms"]) / max(
        float(records["n2_100"]["skin_texture"]["opponent_rms"]),
        1.0e-20,
    )
    if full_positive_ratio <= 1.20:
        raise DataError(
            "full-resolution positive control failed: N2 1.57501286 does not expose at least "
            f"20% more skin-proxy opponent activity than N2 1.0 (ratio {full_positive_ratio:.3f})"
        )
    positive_probe["full_resolution_positive_control_to_native_ratio"] = full_positive_ratio
    strength_ratios = _strength_ratios(records)
    payload: dict[str, Any] = {
        "schema_version": "1",
        "study_id": "negative_grain_synthetic_strength_selection_v1",
        "tier": STUDY.tier.value,
        "question": STUDY.question,
        "rationale": STUDY.rationale,
        "controls": list(STUDY.controls),
        "bundle_identity": adapter.bundle_identity,
        "stimulus": {
            "clip_id": adapter.clip_id,
            "geometry": [adapter.width, adapter.height],
            "frame_count": adapter.frame_count,
            "cadence": list(adapter.cadence),
            "seed": config.seed,
            "parallel_frame_processes": config.frame_workers,
            "parallel_variant_workers_per_process": config.variant_workers,
            "authoring_domain": "scene_linear_rec709_then_scene_xyz",
            "camera_boundary": "exact_sony_slog3_sgamut3cine_forward",
        },
        "visualization": {
            "normal": "sRGB encoded processed output",
            "signed_delta": "zero at neutral grey",
            "magnitude": "absolute RGB delta on black",
            "fixed_linear_rgb_limit": config.delta_display_limit,
            "per_frame_normalization": False,
            "metrics_use_pre_encoding_float_signed_delta": True,
        },
        "variants": [
            {
                "variant_id": variant.variant_id,
                "label": variant.label,
                "family": variant.family,
                "strength": variant.strength,
                "role": variant.role,
            }
            for variant in adapter.variants
        ],
        "positive_control_probe": positive_probe,
        "region_metrics": records,
        "strength_response": strength_ratios,
        "decision_scope": {
            "can_decide": [
                "N2 structured field versus N1 white field",
                "bounded N2 default-strength response",
                "persistent tint and chromatic activity",
                "motion/exposure-transition instability",
            ],
            "cannot_decide": "natural appearance on a real human face",
            "standing": {
                "n2_placement": "accepted",
                "n2_structure": "accepted",
                "n2_1575": "rejected_for_practical_use_positive_control_only",
                "leading_default": 1.0,
                "hybrid": "not_built",
                "sublayers_and_covariance": "deferred",
            },
        },
    }
    write_json(config.output_dir / "metrics.json", payload)
    (config.output_dir / "index.html").write_text(
        _report_html(
            adapter=adapter,
            config=config,
            records=records,
            strength_ratios=strength_ratios,
            positive_probe=positive_probe,
            videos=videos,
        ),
        encoding="utf-8",
    )
    return payload


__all__ = [
    "FIXED_DELTA_DISPLAY_LIMIT",
    "STUDY",
    "RegionAccumulator",
    "SyntheticGrainRunConfig",
    "magnitude_delta_view",
    "review_triptych",
    "run",
    "signed_delta_view",
]
