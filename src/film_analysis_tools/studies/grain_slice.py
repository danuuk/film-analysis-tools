"""One thin vertical slice: source record to grain evidence, on real material.

Every layer below has been built and validated in isolation. That is exactly the failure mode
this study exists to correct — the parts were each defensible and nothing had ever run end to end,
so no one could say what fraction of a real film survives the whole chain.

The chain, in order, with nothing skipped::

    SourceRecord -> catalogue query -> frame extraction -> admissibility
                 -> window selection -> residual measurement -> evidence JSON + report

**Coverage is the deliverable, not the parameters.** At every stage this records what was
considered, what survived and why the rest did not. A run that measures four windows out of nine
thousand candidates has said something important, and the old pipeline would have reported only
the four.

Two deliberate choices about scope:

*The ffmpeg glue lives here, not in `capabilities`.* Surveying and decoding are needed to make
this slice run; generalising them into another capability layer before a single study has run end
to end is how the previous system grew. When a second study needs them, that is the moment to
promote them.

*Sources are put on a common level scale.* Gap 8: both the band edges and the motion gate are
absolute numbers compared against levels whose meaning the caller chooses, so two sources decoded
differently are not comparable at all. Each plan declares its transfer function and a reference
white, and every level in the output is linear with **1.0 = diffuse white**.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from film_analysis_tools.capabilities.catalogue import ingest
from film_analysis_tools.capabilities.catalogue import intervals as iv
from film_analysis_tools.capabilities.catalogue import regions as rg
from film_analysis_tools.capabilities.catalogue.survey import FrameSurvey
from film_analysis_tools.capabilities.measure import admissibility, evidence, residual, windows
from film_analysis_tools.capabilities.source.record import (
    Cadence,
    Crop,
    DecodeContract,
    SourceRecord,
    file_sha256,
)
from film_analysis_tools.capabilities.source.slog3 import slog3_to_linear
from film_analysis_tools.core.errors import DataError

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

#: Scene-linear reflectance of a diffuse white card. S-Log3 is scaled in reflectance, so this is
#: what puts its 1.0 at the same place as a display-referred master's.
DIFFUSE_WHITE_REFLECTANCE = 0.90

#: Display luminance a delivery master's diffuse white sits at, in nits. PQ is absolute, so this
#: is what converts "fraction of 10,000 nits" into "fraction of white".
DIFFUSE_WHITE_NITS = 100.0

#: Fraction of a window's samples that may sit at the container limits before it is unfit for a
#: distribution measurement. Tails are the first thing clipping distorts.
MAX_WINDOW_CLIPPED = 0.01

#: Minimum separation between the two intervals compared for screen anchoring.
#:
#: The test asks whether a pattern sits in screen coordinates rather than in the picture, so the
#: two intervals must hold *unrelated pictures*. Two intervals three seconds apart in a tripod
#: shot hold the same picture: comparing them returned 0.93 for the grain envelope and 0.99 for
#: the additive pattern, which is a measurement of "same scene", not of scan-fixed structure.
MIN_ANCHOR_SEPARATION_S = 60.0


@dataclass(frozen=True)
class SourcePlan:
    """One source, and everything needed to turn it into comparable numbers."""

    source_id: str
    path: Path
    edition: str
    transfer: str
    """``pq`` or ``slog3``. Determines the decode, and therefore what a level means."""

    sha256: str = ""
    """Content hash. Computed when absent, which is slow on a feature-length master."""

    survey_csv: Path | None = None
    """A survey already computed for this source, reused instead of decoding it again.

    The architecture calls for one cheap pass per source, kept. A feature-length 4K master takes
    far longer to survey than to measure, so re-running that pass inside every study would make
    the slice unusable on exactly the material it exists for.
    """

    survey_is_cropped: bool = False
    """Whether ``survey_csv`` was produced with the active-picture crop already applied."""

    survey_fps: float = 4.0
    survey_width: int = 640
    window_s: float = 2.0
    stride_s: float = 1.0
    max_motion: float = 0.01
    """Motion gate as a **fraction of full scale**, so it means the same on every source.

    Motion arrives as ``ydif`` in code values, which is bit-depth dependent and roughly 10.2x
    ``mafd`` on the same footage — the gap-8 defect on the motion axis. Dividing by the code
    ceiling makes the gate dimensionless; 0.01 corresponds to the ``mafd <= 1.0`` used earlier.
    """
    intervals_to_measure: int = 8
    frames_per_interval: int = 10
    tile_size: int = 128
    tile_stride: int = 256

    deep_frames: int = 60
    """Frames to follow a single tile for, in the deep probe."""

    deep_tiles: int = 3
    """How many tiles to follow. Spatial coverage traded for temporal support."""

    def __post_init__(self) -> None:
        if self.transfer not in ("pq", "slog3"):
            raise DataError(f"unsupported transfer {self.transfer!r}; expected 'pq' or 'slog3'")


@dataclass(frozen=True)
class StageCount:
    """What a stage saw, what it passed, and why it dropped the rest."""

    stage: str
    considered: int
    accepted: int
    rejected: dict[str, int] = field(default_factory=dict)

    @property
    def survival(self) -> float:
        return self.accepted / self.considered if self.considered else 0.0

    def as_record(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "considered": self.considered,
            "accepted": self.accepted,
            "survival": self.survival,
            "rejected": dict(self.rejected),
        }

    def line(self) -> str:
        reasons = "; ".join(f"{name} {count}" for name, count in sorted(self.rejected.items()))
        return (
            f"  {self.stage:<22} {self.accepted:>7,} / {self.considered:>7,} "
            f"({self.survival:6.1%}){'   ' + reasons if reasons else ''}"
        )


# ------------------------------------------------------------------ ffmpeg glue


def probe(path: Path) -> dict[str, Any]:
    """Stream geometry and cadence, straight from ffprobe."""
    result = subprocess.run(
        [
            FFPROBE, "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,pix_fmt,r_frame_rate,color_range,color_transfer,color_primaries",
            "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    if result.returncode != 0:
        raise DataError(f"ffprobe failed on {path}: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    if not payload.get("streams"):
        raise DataError(f"no video stream in {path}")
    stream = payload["streams"][0]
    stream["duration_s"] = float(payload.get("format", {}).get("duration", 0.0))
    return stream


_METADATA = re.compile(r"lavfi\.(?:signalstats\.(\w+)|scd\.(score))=([-\d.eE+]+)")
#: Container level below which a row or column is treated as letterbox rather than picture.
#: About code 4 of 1023 — bars sit at exactly 0, real shadow detail does not.
CROP_FLOOR = 0.004


def _crop_filter(crop: Crop | None, stream: dict[str, Any]) -> str:
    """``crop=...,`` prefix for a filter chain, or empty when the whole frame is active."""
    if crop is None or (crop.is_full_frame and not crop.width):
        return ""
    x, y, width, height = crop.applied_to(int(stream["width"]), int(stream["height"]))
    return f"crop={width}:{height}:{x}:{y},"


def detect_crop(plan: SourcePlan, stream: dict[str, Any], *, probes: int = 5) -> Crop:
    """The active picture, found by sampling several points in the source.

    A 2.35:1 film in a 16:9 frame is a quarter letterbox bars, and those bars sit at code 0. Left
    in, they read as hard-clipped black across 24% of every frame and disqualify the entire
    source — which is exactly what happened before this existed. :class:`Crop` was in the source
    record from the start for this reason; nothing had been filling it in.

    Measured from decoded frames rather than ffmpeg's ``cropdetect``, which reported full frame on
    this 10-bit PQ master at every limit tried while the bars were plainly at code 0. Row and
    column means are unambiguous and cost two frames per probe.

    The widest box seen across the probes wins, so a genuinely dark scene cannot narrow the crop
    onto real picture content.
    """
    duration = float(stream.get("duration_s", 0.0))
    width, height = int(stream["width"]), int(stream["height"])
    if duration <= 0:
        return Crop()

    top, bottom, left, right = height, 0, width, 0
    seen = False
    for fraction in np.linspace(0.15, 0.85, probes):
        try:
            frames = decode_signal(plan, stream, duration * fraction, 2)
        except DataError:
            continue
        rows = np.where(frames.mean(axis=(0, 2)) > CROP_FLOOR)[0]
        columns = np.where(frames.mean(axis=(0, 1)) > CROP_FLOOR)[0]
        if not rows.size or not columns.size:
            continue  # an entirely black probe says nothing about the active picture
        seen = True
        top, bottom = min(top, int(rows[0])), max(bottom, int(rows[-1]))
        left, right = min(left, int(columns[0])), max(right, int(columns[-1]))

    if not seen:
        return Crop()
    keep_height = (bottom - top + 1) // 2 * 2
    keep_width = (right - left + 1) // 2 * 2
    if keep_height >= height and keep_width >= width:
        return Crop()
    return Crop(x=left, y=top, width=keep_width, height=keep_height)


def survey(plan: SourcePlan, stream: dict[str, Any], crop: Crop | None = None) -> FrameSurvey:
    """A reduced-resolution per-frame pass — the same cheap survey the catalogue is built on.

    ``metadata=print`` is written to a file rather than parsed from the log: it emits at INFO
    level, so any run quiet enough to be usable discards it silently and the survey comes back
    empty.
    """
    with tempfile.TemporaryDirectory() as workspace:
        dump = Path(workspace) / "signalstats.txt"
        result = subprocess.run(
            [
                FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(plan.path),
                "-map", "0:v:0", "-an", "-sn", "-dn",
                "-vf",
                _crop_filter(crop, stream)
                + f"scale={plan.survey_width}:-2:flags=bicubic,fps={plan.survey_fps},"
                "gblur=sigma=1,scdet=threshold=10,signalstats,"
                f"metadata=mode=print:file={dump}",
                "-f", "null", "-",
            ],
            capture_output=True, text=True, check=False,
        )  # fmt: skip
        if result.returncode != 0:
            raise DataError(f"survey pass failed on {plan.path}: {result.stderr.strip()[:400]}")
        emitted = dump.read_text() if dump.exists() else ""

    times: list[float] = []
    rows: list[dict[str, float]] = []
    current: dict[str, float] = {}
    for line in emitted.splitlines():
        if line.startswith("frame:"):
            if current:
                rows.append(current)
            current = {}
            match = re.search(r"pts_time:([-\d.]+)", line)
            times.append(float(match.group(1)) if match else float(len(times)) / plan.survey_fps)
            continue
        found = _METADATA.search(line)
        if found:
            key = found.group(1) or found.group(2)
            current[key] = float(found.group(3))
    if current:
        rows.append(current)
    if not rows:
        raise DataError(f"survey produced no frames for {plan.source_id}")

    times = times[: len(rows)]
    take = lambda key: np.asarray([row.get(key, 0.0) for row in rows], dtype=np.float64)  # noqa: E731
    peak = 1023.0 if "10" in str(stream.get("pix_fmt", "")) else 255.0
    scale = 1023.0 / peak  # express every level on a 10-bit code axis

    columns = {
        "time_s": np.asarray(times, dtype=np.float64),
        "motion": take("YDIF") * scale / 1023.0,  # fraction of full scale, not code values
        "cut_score": take("score"),
        "level_mean": take("YAVG") * scale,
        "level_low": take("YLOW") * scale,
        "level_high": take("YHIGH") * scale,
        "saturation_mean": take("SATAVG"),
        "hue_median": take("HUEMED"),
    }
    full_range = str(stream.get("color_range", "tv")) == "pc"
    return FrameSurvey(
        source_id=plan.source_id,
        columns=columns,
        cadence=Cadence.parse(str(stream.get("r_frame_rate", "24000/1001"))),
        sample_rate_hz=plan.survey_fps,
        code_floor=0.0 if full_range else 64.0,
        code_ceiling=1023.0 if full_range else 940.0,
        notes={"pass": "scale/fps/gblur/scdet/signalstats", "transfer": plan.transfer},
    )


def load_survey(plan: SourcePlan, stream: dict[str, Any], crop: Crop | None = None) -> FrameSurvey:
    """The kept survey when the plan names one, otherwise a fresh pass.

    Both paths use ``ydif`` for motion, normalised to a fraction of full scale. The kept survey
    also carries ``mafd``, which is about 10.2x smaller on the same footage — mixing the two would
    silently change what the motion gate means between sources.
    """
    if plan.survey_csv is None:
        return survey(plan, stream, crop)

    loaded = ingest.read_survey(
        plan.survey_csv,
        source_id=plan.source_id,
        cadence=Cadence.parse(str(stream.get("r_frame_rate", "24000/1001"))),
        sample_rate_hz=plan.survey_fps,
        columns={**ingest.SIGNALSTATS_COLUMNS, "motion": "ydif"},
        notes={"reused": str(plan.survey_csv)},
    )
    columns = dict(loaded.columns)
    columns["motion"] = columns["motion"] / loaded.code_ceiling
    return replace(loaded, columns=columns)


def decode_signal(
    plan: SourcePlan,
    stream: dict[str, Any],
    start_s: float,
    count: int,
    crop: Crop | None = None,
) -> np.ndarray:
    """Native-resolution luma for one interval, as the **encoded signal** on 0..1.

    Decoded at full resolution on purpose: the survey is downscaled and pre-blurred, which
    destroys exactly the high-frequency detail grain lives in.

    Admissibility is asked here, before any transfer. Clipping is defined by the container's own
    limits, and overlay detection asks whether the encoded signal carries noise — both are
    code-domain questions. Asking them after a PQ EOTF gave 87.6% "clipped at the floor" and 100%
    "noise-free" on ordinary film, because the EOTF compresses shadow codes to values that
    underflow to zero. Nothing was wrong with the footage.
    """
    active = crop or Crop()
    _, _, width, height = active.applied_to(int(stream["width"]), int(stream["height"]))
    filters = _crop_filter(crop, stream) + "format=gray16le"
    result = subprocess.run(
        [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-ss", f"{start_s:.3f}", "-i", str(plan.path), "-map", "0:v:0",
            "-frames:v", str(count), "-vf", filters,
            "-f", "rawvideo", "-pix_fmt", "gray16le", "-",
        ],
        capture_output=True, check=False,
    )  # fmt: skip
    needed = count * width * height * 2
    if len(result.stdout) < needed:
        raise DataError(
            f"{plan.source_id}: decoded {len(result.stdout) // (width * height * 2)} of {count} "
            f"frames at {start_s:.2f}s"
        )
    raw = np.frombuffer(result.stdout[:needed], dtype="<u2").reshape(count, height, width)
    code = raw.astype(np.float64) / 64.0  # 16-bit container back to the 10-bit code axis
    return code / 1023.0


def to_linear(plan: SourcePlan, container: np.ndarray) -> np.ndarray:
    """The container signal as linear light, with 1.0 at diffuse white on every source.

    **No range mapping happens here, because ffmpeg has already done it.** Converting a
    limited-range YUV stream to ``gray16le`` range-expands: measured against a synthetic clip with
    known codes, 64 arrives as 0 and 940 as 1023. Applying the 64..940 mapping again on top of
    that crushed every low PQ value toward zero and distorted the linear levels, the
    amplitude-versus-level placement, the residual distributions and the shadow tails.

    The same measurement shows the conversion also **clamps**: codes 0 and 64 both arrive as 0,
    940 and 1023 both as 1023. Sub-legal footroom and super-white headroom do not survive this
    decode, so "clipped at the floor" here means "at or below legal black" rather than "at code
    zero". That is a defensible definition of crushed black but it is not the same statement, and
    recovering the difference would need a decode that does not normalise.

    Measurement happens here rather than in the code domain: amplitude against *level* only means
    something on a scale where equal steps are equal light.
    """
    if plan.transfer == "pq":
        return _pq_to_linear(container) * (10000.0 / DIFFUSE_WHITE_NITS)
    return np.asarray(slog3_to_linear(container)) / DIFFUSE_WHITE_REFLECTANCE


def _pq_to_linear(signal: np.ndarray) -> np.ndarray:
    """SMPTE ST 2084 EOTF, normalised so 1.0 is the 10,000-nit peak."""
    m1, m2, c1, c2, c3 = 0.1593017578125, 78.84375, 0.8359375, 18.8515625, 18.6875
    powered = np.clip(signal, 0.0, 1.0) ** (1.0 / m2)
    return np.asarray((np.maximum(powered - c1, 0.0) / (c2 - c3 * powered)) ** (1.0 / m1))


def source_record(plan: SourcePlan, stream: dict[str, Any], crop: Crop) -> SourceRecord:
    digest = plan.sha256 or file_sha256(plan.path)
    return SourceRecord(
        source_id=plan.source_id,
        edition=plan.edition,
        sha256=digest,
        byte_size=plan.path.stat().st_size,
        coded_width=int(stream["width"]),
        coded_height=int(stream["height"]),
        cadence=Cadence.parse(str(stream.get("r_frame_rate", "24000/1001"))),
        decode=DecodeContract(
            input_range=str(stream.get("color_range", "tv")),
            transfer=plan.transfer,
            primaries=str(stream.get("color_primaries", "unknown")),
            output_pixel_format="gray16le",
            scale="linear_diffuse_white",
        ),
        crop=crop,
        path_hint=str(plan.path),
        notes="grain_slice vertical study",
    )


# ------------------------------------------------------------------- the slice


@dataclass(frozen=True)
class Spread:
    """A scalar measured independently on several intervals — never reported as one number."""

    values: tuple[float, ...]

    @property
    def count(self) -> int:
        return len(self.values)

    @property
    def median(self) -> float:
        return float(np.median(self.values)) if self.values else 0.0

    @property
    def low(self) -> float:
        return float(np.min(self.values)) if self.values else 0.0

    @property
    def high(self) -> float:
        return float(np.max(self.values)) if self.values else 0.0

    def as_record(self) -> dict[str, Any]:
        return {
            "median": self.median,
            "min": self.low,
            "max": self.high,
            "n": self.count,
            "values": list(self.values),
        }

    def line(self) -> str:
        if not self.values:
            return "none"
        return f"median {self.median:+.4f}  range {self.low:+.4f}..{self.high:+.4f}  n={self.count}"


@dataclass(frozen=True)
class IntervalEvidence:
    """Evidence from one interval, with the window set each estimator was actually given.

    The estimators need different material and the project's own selection criteria say so, so
    they do not all get the same windows:

    * **spectrum** — flat windows only. A textured window's residual carries the picture's edges,
      and a noise power spectrum reads those as grain structure.
    * **distribution** — unclipped windows. A clipped sample is not a residual, and tails are what
      clipping distorts first.
    * **temporal** — windows whose own correlation estimate is trustworthy. A drifting window
      reports near-zero correlation whatever the truth.
    * **amplitude** — every accepted window; ``extract`` aligns internally and each point carries
      its own trust.
    """

    interval_start_s: float
    windows: int
    flat_windows: int
    unclipped_windows: int
    trustworthy_windows: int
    amplitude: evidence.AmplitudeEvidence | None
    spectrum: evidence.SpectrumEvidence | None
    distribution: evidence.DistributionEvidence | None
    heterogeneity: evidence.HeterogeneityEvidence | None
    temporal: evidence.TemporalEvidence | None
    distribution_trustworthy: evidence.DistributionEvidence | None = None
    """The same measurement restricted to windows that pass the temporal trust gate.

    Alignment removes integer translation; it does not remove sub-pixel deformation, local motion
    or model failure. "Heavy tails survive alignment" and "heavy tails survive contamination
    rejection" are different claims, and only this one supports the second.
    """

    tails: TailDiagnostics | None = None
    skipped: tuple[str, ...] = ()

    def as_record(self) -> dict[str, Any]:
        return {
            "interval_start_s": self.interval_start_s,
            "windows": self.windows,
            "routed": {
                "amplitude": self.windows,
                "spectrum_flat": self.flat_windows,
                "distribution_unclipped": self.unclipped_windows,
                "temporal_trustworthy": self.trustworthy_windows,
            },
            "temporal_trusted_share_of_accepted": (
                self.trustworthy_windows / self.windows if self.windows else 0.0
            ),
            "note": (
                "temporal.trusted_fraction is 1.0 by construction: only trustworthy windows are "
                "passed to the estimator. Use temporal_trusted_share_of_accepted instead."
            ),
            "amplitude": self.amplitude.as_record() if self.amplitude else None,
            "spectrum": self.spectrum.as_record() if self.spectrum else None,
            "distribution": self.distribution.as_record() if self.distribution else None,
            "distribution_trustworthy": (
                self.distribution_trustworthy.as_record() if self.distribution_trustworthy else None
            ),
            "tails": self.tails.as_record() if self.tails else None,
            "heterogeneity": self.heterogeneity.as_record() if self.heterogeneity else None,
            "temporal": self.temporal.as_record() if self.temporal else None,
            "skipped": list(self.skipped),
        }


@dataclass(frozen=True)
class Seconds:
    """Three different amounts of time, none of which may stand in for another.

    ``candidate`` is what the catalogue offered, ``decoded`` is what was actually pulled off disk,
    and ``evidence`` is what the estimators saw. Reporting the first as though it were the third
    overstates measurement support by an order of magnitude: ten frames at 23.976 fps is 0.417 s,
    not the two seconds the interval spans.
    """

    candidate: float
    decoded: float
    evidence: float

    def as_record(self) -> dict[str, Any]:
        return {
            "candidate_s": self.candidate,
            "decoded_s": self.decoded,
            "evidence_s": self.evidence,
        }

    def line(self) -> str:
        return (
            f"  seconds                candidate {self.candidate:6.1f}   "
            f"decoded {self.decoded:5.2f}   evidence {self.evidence:5.2f}"
        )


@dataclass(frozen=True)
class SourceOutcome:
    """Everything one source contributed, and everything it failed to."""

    plan: SourcePlan
    record: SourceRecord
    stages: tuple[StageCount, ...]
    regions: rg.RegionIndex
    seconds: Seconds
    per_interval: tuple[IntervalEvidence, ...]
    deep: tuple[DeepProbe, ...] = ()
    screen_anchoring: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def measured(self) -> bool:
        return any(one.amplitude is not None for one in self.per_interval)

    def _spread(self, pick: Any) -> Spread:
        values = [pick(one) for one in self.per_interval]
        return Spread(values=tuple(v for v in values if v is not None))

    def raw_sigma(self) -> Spread:
        """Median sigma over **every** point, trustworthy or not.

        A descriptive median of all residual estimates, not a validated grain amplitude. Reported
        under its own name because 3 of 245 Pulp points and 28 of 94 Sony points were trustworthy,
        and quoting the all-points median as "sigma" implies support that does not exist.
        """
        return self._spread(
            lambda one: (
                float(np.median([p.sigma for p in one.amplitude.points]))
                if one.amplitude and one.amplitude.points
                else None
            )
        )

    def trusted_sigma(self) -> Spread:
        """Median sigma over points that are trustworthy *and* whose correlation was identified.

        A point pinned to the correlation bound is excluded: it receives a ``1/sqrt(1-rho)``
        correction of 10x that the data never justified. On the previous run Sony's six saturated
        points had median sigma 0.00736 against 0.000759 for the nineteen identified ones.
        """
        return self._spread(
            lambda one: (
                float(np.median([p.sigma for p in _identified(one)])) if _identified(one) else None
            )
        )

    def whiteness(self) -> Spread:
        return self._spread(lambda one: one.spectrum.whiteness if one.spectrum else None)

    def kurtosis(self) -> Spread:
        return self._spread(
            lambda one: one.distribution.excess_kurtosis if one.distribution else None
        )

    def rho(self) -> Spread:
        return self._spread(lambda one: one.temporal.rho if one.temporal else None)

    def kurtosis_normalised(self) -> Spread:
        return self._spread(lambda one: one.tails.kurtosis_normalised if one.tails else None)

    def mixing_prediction(self) -> Spread:
        return self._spread(lambda one: one.tails.mixing_prediction if one.tails else None)

    def exact_zero(self) -> Spread:
        return self._spread(lambda one: one.tails.exact_zero_fraction if one.tails else None)

    def skew(self) -> Spread:
        return self._spread(lambda one: one.tails.skew_pooled if one.tails else None)

    def kurtosis_trustworthy(self) -> Spread:
        return self._spread(
            lambda one: (
                one.distribution_trustworthy.excess_kurtosis
                if one.distribution_trustworthy
                else None
            )
        )

    def envelope(self) -> Spread:
        return self._spread(
            lambda one: one.heterogeneity.envelope_ratio if one.heterogeneity else None
        )

    def trusted_points(self) -> tuple[int, int, int]:
        """``(identified, trustworthy, total)`` amplitude points."""
        identified = trusted = total = 0
        for one in self.per_interval:
            if one.amplitude:
                identified += len(_identified(one))
                trusted += len(one.amplitude.trusted)
                total += len(one.amplitude.points)
        return identified, trusted, total

    def saturated_points(self) -> int:
        return sum(
            1
            for one in self.per_interval
            if one.amplitude
            for point in one.amplitude.trusted
            if abs(point.rho) >= residual.RHO_BOUND - 1e-9
        )

    def as_record(self) -> dict[str, Any]:
        identified, trusted, total = self.trusted_points()
        return {
            "source": self.record.as_record(),
            "plan": {
                "transfer": self.plan.transfer,
                "level_scale": "linear, 1.0 = diffuse white",
                "window_s": self.plan.window_s,
                "frames_per_interval": self.plan.frames_per_interval,
                "tile_size": self.plan.tile_size,
                "intervals_requested": self.plan.intervals_to_measure,
            },
            "coverage": [stage.as_record() for stage in self.stages],
            "seconds": self.seconds.as_record(),
            "regions": self.regions.as_record(),
            "intervals_measured": len(self.per_interval),
            "amplitude_points": {
                "identified": identified,
                "trustworthy": trusted,
                "total": total,
                "saturated_rho": self.saturated_points(),
            },
            "aggregate": {
                "raw_sigma": self.raw_sigma().as_record(),
                "trusted_sigma": self.trusted_sigma().as_record(),
                "whiteness": self.whiteness().as_record(),
                "excess_kurtosis": self.kurtosis().as_record(),
                "rho": self.rho().as_record(),
                "envelope_ratio": self.envelope().as_record(),
                "kurtosis_normalised": self.kurtosis_normalised().as_record(),
                "kurtosis_trustworthy": self.kurtosis_trustworthy().as_record(),
                "mixing_prediction": self.mixing_prediction().as_record(),
                "skew": self.skew().as_record(),
                "exact_zero_fraction": self.exact_zero().as_record(),
            },
            "per_interval": [one.as_record() for one in self.per_interval],
            "deep_probes": [one.as_record() for one in self.deep],
            "screen_anchoring": self.screen_anchoring,
            "notes": list(self.notes),
        }


# ------------------------------------------------------- why the tails are heavy


@dataclass(frozen=True)
class TailDiagnostics:
    """What is actually producing the heavy tails, separated into candidate causes.

    Excess kurtosis of +90 on Pulp Fiction and +2989 on the Sony clip is a real property of the
    extracted residual arrays. It is not automatically a property of grain, and four different
    mechanisms produce it:

    * **scale mixing** — pooling windows of different amplitudes makes a narrow peak with broad
      tails even when every window is perfectly Gaussian. ``mixing_prediction`` is the excess
      kurtosis expected from the observed per-window variance spread *alone*.
    * **quantisation and zero inflation** — pixels that do not change between frames pile up at
      exactly zero. ``exact_zero_fraction`` and ``step_occupancy`` measure it in code units.
    * **isolated events** — one bad frame pair dominating. ``outlier_fraction_per_pair`` shows
      whether the tails are spread across pairs or concentrated in one transition.
    * **genuinely non-Gaussian grain** — what is left after the other three are accounted for.

    ``kurtosis_normalised`` is the one that answers "after accounting for its changing strength,
    what shape does this residual have": each window is divided by its own robust scale before
    pooling, so mixing cannot contribute.
    """

    kurtosis_pooled: float
    kurtosis_normalised: float
    kurtosis_per_window: Spread
    mixing_prediction: float
    skew_pooled: float
    exact_zero_fraction: float
    step_occupancy: dict[str, float]
    outlier_fraction_per_pair: tuple[float, ...]
    windows_used: int

    @property
    def mixing_explains(self) -> float:
        """Share of the observed excess kurtosis the variance spread alone would produce."""
        return self.mixing_prediction / self.kurtosis_pooled if self.kurtosis_pooled > 0 else 0.0

    def as_record(self) -> dict[str, Any]:
        return {
            "kurtosis_pooled": self.kurtosis_pooled,
            "kurtosis_normalised": self.kurtosis_normalised,
            "kurtosis_per_window": self.kurtosis_per_window.as_record(),
            "mixing_prediction": self.mixing_prediction,
            "mixing_explains": self.mixing_explains,
            "skew_pooled": self.skew_pooled,
            "exact_zero_fraction": self.exact_zero_fraction,
            "step_occupancy": self.step_occupancy,
            "outlier_fraction_per_pair": list(self.outlier_fraction_per_pair),
            "windows_used": self.windows_used,
        }


def _excess_kurtosis(values: np.ndarray) -> float:
    centred = values - values.mean()
    scale = float(np.std(centred))
    if scale <= 1e-15:
        return 0.0
    return float(np.mean((centred / scale) ** 4) - 3.0)


def _skew(values: np.ndarray) -> float:
    centred = values - values.mean()
    scale = float(np.std(centred))
    if scale <= 1e-15:
        return 0.0
    return float(np.mean((centred / scale) ** 3))


def _robust_scale(values: np.ndarray) -> float:
    """1.4826 x MAD, with a floor at a fraction of the standard deviation.

    MAD collapses toward zero on a zero-inflated window — where a third of pixels do not change
    between frames, the median absolute deviation can be a single quantisation step or less.
    Dividing by that *amplifies* the tails instead of removing the scale, which is exactly what
    happened on Pulp Fiction: per-window normalisation reported a median excess kurtosis of
    +1481 against +24.8 pooled. The floor keeps the normalisation a scale correction rather than
    an outlier magnifier.
    """
    median = float(np.median(values))
    mad = float(1.4826 * np.median(np.abs(values - median)))
    return max(mad, 0.25 * float(np.std(values)))


def tail_diagnostics(
    linear: np.ndarray,
    container: np.ndarray,
    selected: Sequence[windows.Window],
) -> TailDiagnostics:
    """Decompose the residual's heavy tails into the mechanisms that could produce them."""
    stacks = [residual.aligned_residuals(window.slice_of(linear)) for window in selected]
    pooled = np.concatenate([stack.ravel() for stack in stacks])

    variances = np.asarray([float(np.var(stack)) for stack in stacks])
    # A scale mixture of zero-mean Gaussians has kurtosis 3 E[v^2]/E[v]^2, so the excess produced
    # by mixing alone is that minus 3 -- with no contribution from the shape of any component.
    mean_variance = float(np.mean(variances)) if variances.size else 0.0
    mixing = (
        3.0 * float(np.mean(variances**2)) / (mean_variance**2) - 3.0
        if mean_variance > 1e-30
        else 0.0
    )

    normalised = np.concatenate(
        [
            (stack / scale).ravel()
            for stack, scale in ((s, _robust_scale(s.ravel())) for s in stacks)
            if scale > 1e-15
        ]
        or [np.zeros(1)]
    )

    # Quantisation is only visible in code units: the transfer smears the steps.
    code_deltas = np.diff(container * 1023.0, axis=0)
    tiles = np.concatenate([window.slice_of(code_deltas).ravel() for window in selected])
    rounded = np.rint(tiles)
    occupancy = {
        str(int(step)): float(np.mean(rounded == step)) for step in (-2.0, -1.0, 0.0, 1.0, 2.0)
    }

    scale = _robust_scale(pooled)
    per_pair: list[float] = []
    for index in range(stacks[0].shape[0] if stacks else 0):
        pair = np.concatenate([stack[index].ravel() for stack in stacks])
        per_pair.append(float(np.mean(np.abs(pair) > 5.0 * scale)) if scale > 1e-15 else 0.0)

    return TailDiagnostics(
        kurtosis_pooled=_excess_kurtosis(pooled),
        kurtosis_normalised=_excess_kurtosis(normalised),
        kurtosis_per_window=Spread(tuple(_excess_kurtosis(s.ravel()) for s in stacks)),
        mixing_prediction=mixing,
        skew_pooled=_skew(pooled),
        exact_zero_fraction=float(np.mean(tiles == 0.0)),
        step_occupancy=occupancy,
        outlier_fraction_per_pair=tuple(per_pair),
        windows_used=len(selected),
    )


# --------------------------------------------------------------- the deep probe


@dataclass(frozen=True)
class DeepProbe:
    """One tile, followed for far longer than the wide pass can afford.

    Ten frames is 0.417 s at 23.976 fps, and it is too short to say anything firm about temporal
    behaviour: an AR(1) correlation estimated from nine difference pairs is barely constrained,
    which is part of why so many estimates ran to the clamp. Following a handful of tiles for
    48-72 frames costs about the same decode as the wide pass and answers the temporal question
    properly, at the price of spatial coverage.
    """

    interval_start_s: float
    x: int
    y: int
    size: int
    frames: int
    level: float
    sigma: float
    legacy_sigma: float
    rho: float
    raw_rho: float
    rho_from_lag4: float
    identified: bool
    trustworthy: bool
    tails: TailDiagnostics

    def as_record(self) -> dict[str, Any]:
        return {
            "interval_start_s": self.interval_start_s,
            "tile": {"x": self.x, "y": self.y, "size": self.size},
            "frames": self.frames,
            "level": self.level,
            "sigma": self.sigma,
            "legacy_sigma": self.legacy_sigma,
            "rho": self.rho,
            "raw_rho": self.raw_rho,
            "rho_from_lag4": self.rho_from_lag4,
            "parameter_identified": self.identified,
            "trustworthy": self.trustworthy,
            "tails": self.tails.as_record(),
        }

    def line(self) -> str:
        flag = "identified" if self.identified else "SATURATED"
        return (
            f"    t={self.interval_start_s:7.0f}s ({self.x},{self.y})  {self.frames}f  "
            f"level {self.level:.4f}  sigma {self.sigma:.5f}  rho {self.rho:+.3f} "
            f"(raw {self.raw_rho:+.3f}, {flag})  kurt {self.tails.kurtosis_pooled:+.1f} -> "
            f"{self.tails.kurtosis_normalised:+.1f} norm  zeros "
            f"{self.tails.exact_zero_fraction:.1%}"
        )


def decode_tile(
    plan: SourcePlan,
    stream: dict[str, Any],
    crop: Crop,
    window: windows.Window,
    start_s: float,
    count: int,
    *,
    margin: int = 16,
) -> np.ndarray:
    """A long run of frames for one tile, in the container domain.

    Cropping in ffmpeg rather than decoding whole frames is what makes the long run affordable:
    72 frames of one 160 px tile is a rounding error beside 72 frames of 4K.
    """
    origin_x, origin_y, active_w, active_h = crop.applied_to(
        int(stream["width"]), int(stream["height"])
    )
    left = max(0, window.x - margin)
    top = max(0, window.y - margin)
    width = min(window.size + 2 * margin, active_w - left)
    height = min(window.size + 2 * margin, active_h - top)
    result = subprocess.run(
        [
            FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-ss", f"{start_s:.3f}", "-i", str(plan.path), "-map", "0:v:0",
            "-frames:v", str(count),
            "-vf", f"crop={width}:{height}:{origin_x + left}:{origin_y + top},format=gray16le",
            "-f", "rawvideo", "-pix_fmt", "gray16le", "-",
        ],
        capture_output=True, check=False,
    )  # fmt: skip
    needed = count * width * height * 2
    if len(result.stdout) < needed:
        raise DataError(
            f"{plan.source_id}: deep probe decoded "
            f"{len(result.stdout) // max(width * height * 2, 1)} of {count} frames"
        )
    raw = np.frombuffer(result.stdout[:needed], dtype="<u2").reshape(count, height, width)
    return raw.astype(np.float64) / 64.0 / 1023.0


def deep_probe(
    plan: SourcePlan,
    stream: dict[str, Any],
    crop: Crop,
    interval: iv.Interval,
    window: windows.Window,
) -> DeepProbe:
    """Follow one tile for ``plan.deep_frames`` frames and re-ask the temporal question."""
    container = decode_tile(plan, stream, crop, window, interval.start_s, plan.deep_frames)
    linear = to_linear(plan, container)
    estimate = residual.extract(linear)
    whole = windows.Window(
        x=0,
        y=0,
        size=min(linear.shape[1], linear.shape[2]),
        level=float(linear.mean()),
        motion_energy=estimate.motion_energy,
        structure_snr=estimate.structure_snr,
        subpixel_residual=estimate.subpixel_residual,
        band=window.band,
        texture=window.texture,
        position=window.position,
    )
    return DeepProbe(
        interval_start_s=interval.start_s,
        x=window.x,
        y=window.y,
        size=window.size,
        frames=int(linear.shape[0]),
        level=float(linear.mean()),
        sigma=estimate.sigma,
        legacy_sigma=estimate.legacy_sigma,
        rho=estimate.rho,
        raw_rho=estimate.raw_rho,
        rho_from_lag4=estimate.rho_from_lag4,
        identified=estimate.parameter_identified,
        trustworthy=estimate.correlation_trustworthy,
        tails=tail_diagnostics(linear, container, [whole]),
    )


def _identified(one: IntervalEvidence) -> list[evidence.AmplitudePoint]:
    """Trustworthy points whose correlation was determined rather than pinned to the clamp."""
    if not one.amplitude:
        return []
    return [point for point in one.amplitude.trusted if abs(point.rho) < residual.RHO_BOUND - 1e-9]


def _clipped_fraction(container: np.ndarray, window: windows.Window) -> float:
    tile = window.slice_of(container)
    return float(((tile <= 0.0) | (tile >= 1.0)).mean())


def _measure_interval(
    start_s: float,
    linear: np.ndarray,
    container: np.ndarray,
    accepted: Sequence[windows.Window],
) -> IntervalEvidence:
    """Every estimator on one interval, each given the windows it actually needs."""
    skipped: list[str] = []
    amplitude = evidence.amplitude_evidence(linear, accepted)

    flat = [window for window in accepted if window.texture == "flat"]
    unclipped = [w for w in accepted if _clipped_fraction(container, w) <= MAX_WINDOW_CLIPPED]
    trustworthy = [
        window
        for window, point in zip(accepted, amplitude.points, strict=True)
        if point.trustworthy
    ]

    spectrum = None
    if flat:
        spectrum = evidence.spectrum_evidence(linear, flat)
    else:
        skipped.append("spectrum: no flat window (every accepted tile carries picture structure)")

    distribution = None
    if unclipped:
        distribution = evidence.distribution_evidence(linear, unclipped)
    else:
        skipped.append("distribution: every window clipped")

    temporal = None
    if trustworthy:
        temporal = evidence.temporal_evidence(linear, trustworthy)
    else:
        skipped.append("temporal: no window could vouch for its own correlation estimate")

    # temporal_evidence reports the trusted fraction *of the windows it was given*. Since only
    # trustworthy windows are passed, that is 1.0 by construction — the previous run recorded
    # "100% trusted" for an interval where 1 of 60 accepted windows qualified. The honest figure
    # is the share of accepted windows, and it is recorded here rather than left to be misread.

    clean = [w for w in trustworthy if _clipped_fraction(container, w) <= MAX_WINDOW_CLIPPED]
    distribution_trustworthy = evidence.distribution_evidence(linear, clean) if clean else None
    if not clean:
        skipped.append(
            "distribution (trustworthy): no window is both unclipped and temporally trustworthy"
        )

    return IntervalEvidence(
        interval_start_s=start_s,
        windows=len(accepted),
        flat_windows=len(flat),
        unclipped_windows=len(unclipped),
        trustworthy_windows=len(trustworthy),
        amplitude=amplitude,
        spectrum=spectrum,
        distribution=distribution,
        heterogeneity=evidence.heterogeneity_evidence(linear, accepted),
        temporal=temporal,
        distribution_trustworthy=distribution_trustworthy,
        tails=tail_diagnostics(linear, container, unclipped or accepted),
        skipped=tuple(skipped),
    )


def run_source(plan: SourcePlan) -> SourceOutcome:
    """The whole chain for one source, recording what was lost at every stage."""
    stream = probe(plan.path)
    active = detect_crop(plan, stream)
    record = source_record(plan, stream, active)
    fps = record.cadence.fps
    notes: list[str] = []
    stages: list[StageCount] = []

    frame_survey = load_survey(plan, stream, active)
    if plan.survey_csv is not None and not active.is_full_frame and not plan.survey_is_cropped:
        notes.append(
            "the reused survey is a CODED-FRAME survey: its motion and level statistics include "
            f"the {active.y * 2 / int(stream['height']):.1%} of each frame that is letterbox, "
            "while extraction uses the active picture. Regenerating the survey with the crop "
            "applied would remove the discrepancy."
        )
    built = iv.build_intervals(
        frame_survey, window_s=plan.window_s, stride_s=plan.stride_s, drop_cuts=False
    )
    cut_free = [interval for interval in built if interval.cut_free]
    stable = [interval for interval in cut_free if interval.motion_p90 <= plan.max_motion]
    stages.append(
        StageCount(
            stage="intervals",
            considered=len(built),
            accepted=len(stable),
            rejected={
                "contains a cut": len(built) - len(cut_free),
                "motion above gate": len(cut_free) - len(stable),
            },
        )
    )
    empty_seconds = Seconds(candidate=0.0, decoded=0.0, evidence=0.0)
    if not stable:
        notes.append("no cut-free, low-motion interval exists in this source")
        return _empty(plan, record, stages, notes, empty_seconds)

    picks = np.linspace(0, len(stable) - 1, plan.intervals_to_measure).astype(int)
    chosen = list({stable[i].start_s: stable[i] for i in picks}.values())

    admissible: list[tuple[iv.Interval, np.ndarray, np.ndarray, admissibility.OverlayEvidence]] = []
    admissibility_rejects: dict[str, int] = {}
    decode_failures = 0
    noise_free_masked = 0
    for interval in chosen:
        try:
            signal = decode_signal(
                plan, stream, interval.start_s, plan.frames_per_interval, crop=active
            )
        except DataError:
            decode_failures += 1
            continue
        # Admissibility in the container domain, where 0 and 1 are the format's real limits.
        verdict = admissibility.scene_admissibility(signal, ceiling=1.0, floor=0.0)

        # Overlay is a *mask*, not a veto. A compressed delivery master carries large blocks the
        # encoder froze -- 20-22% of 32px blocks in this one have exactly zero temporal variation,
        # which is real grain loss, not a title card. Rejecting the interval throws away the 78%
        # that is still measurable; excluding those blocks from window selection does not.
        blocking = [
            reason for reason in verdict.reasons if "carries no temporal noise" not in reason
        ]
        if verdict.overlay.has_overlay:
            noise_free_masked += 1
        if blocking:
            for reason in blocking:
                key = reason.split(" contributes")[0].split(" of ")[-1][:28]
                admissibility_rejects[key] = admissibility_rejects.get(key, 0) + 1
        elif verdict.overlay.noise_free_fraction >= 1.0:
            admissibility_rejects["frame is entirely static"] = (
                admissibility_rejects.get("frame is entirely static", 0) + 1
            )
        else:
            admissible.append((interval, to_linear(plan, signal), signal, verdict.overlay))
    if decode_failures:
        admissibility_rejects["decode failed"] = decode_failures
    if noise_free_masked:
        notes.append(
            f"{noise_free_masked} of {len(chosen)} intervals carry frozen, noise-free blocks "
            "(encoder skip blocks); those blocks were masked out of window selection rather than "
            "used to reject the interval"
        )
    stages.append(
        StageCount(
            stage="admissibility",
            considered=len(chosen),
            accepted=len(admissible),
            rejected=admissibility_rejects,
        )
    )
    decoded_s = len(admissible) * plan.frames_per_interval / fps
    if not admissible:
        notes.append("every decoded interval was ruled inadmissible")
        return _empty(plan, record, stages, notes, Seconds(0.0, 0.0, 0.0))

    collected: list[rg.Region] = []
    considered = accepted_count = 0
    window_rejects: dict[str, int] = {}
    measurable: list[tuple[iv.Interval, np.ndarray, np.ndarray, list[windows.Window]]] = []
    for interval, linear, container, overlay in admissible:
        report_ = windows.select_windows(
            linear, size=plan.tile_size, stride=plan.tile_stride, overlay=overlay
        )
        considered += len(report_.accepted) + len(report_.rejected)
        accepted_count += len(report_.accepted)
        for reason, count in report_.rejection_reasons().items():
            window_rejects[reason] = window_rejects.get(reason, 0) + count
        collected.extend(
            rg.regions_from_report(
                report_,
                source_id=plan.source_id,
                interval_start_s=interval.start_s,
                interval_end_s=interval.end_s,
                start_frame=record.cadence.frame_at(interval.start_s),
                level_scale="linear",
                source_identity=record.identity,
            )
        )
        if report_.accepted:
            measurable.append((interval, linear, container, list(report_.accepted)))
    stages.append(
        StageCount(
            stage="windows",
            considered=considered,
            accepted=accepted_count,
            rejected=window_rejects,
        )
    )
    index = rg.index(collected)
    candidate_s = index.independence().span_seconds
    if not measurable:
        notes.append("no tile passed the window gate in any admissible interval")
        return _empty(
            plan, record, stages, notes, Seconds(candidate_s, decoded_s, 0.0), regions=index
        )

    # Every admissible interval is measured, independently. Measuring only the interval that
    # yielded the most windows -- as the first run did -- reports one picture's statistics as the
    # source's, and selects for whichever picture happens to pass most easily.
    per_interval = [
        _measure_interval(interval.start_s, linear, container, accepted)
        for interval, linear, container, accepted in measurable
    ]
    evidence_s = len(per_interval) * plan.frames_per_interval / fps

    # Follow a few tiles for much longer. Ten frames gives nine difference pairs, which is why so
    # many correlation estimates ran to the clamp; deep_frames gives an actually constrained one.
    deep: list[DeepProbe] = []
    candidates: list[tuple[iv.Interval, windows.Window]] = []
    for interval, _, _, accepted in measurable:
        ranked = sorted(accepted, key=lambda w: w.motion_energy)
        candidates.extend((interval, window) for window in ranked[:2])
    for interval, window in candidates[: plan.deep_tiles]:
        try:
            deep.append(deep_probe(plan, stream, active, interval, window))
        except DataError as error:
            notes.append(f"deep probe skipped: {error}")

    return SourceOutcome(
        plan=plan,
        record=record,
        stages=tuple(stages),
        regions=index,
        seconds=Seconds(candidate=candidate_s, decoded=decoded_s, evidence=evidence_s),
        per_interval=tuple(per_interval),
        deep=tuple(deep),
        screen_anchoring=_screen_anchoring(measurable),
        notes=tuple(notes),
    )


def _screen_anchoring(
    measurable: Sequence[tuple[iv.Interval, np.ndarray, np.ndarray, list[windows.Window]]],
) -> dict[str, Any]:
    """Compare the two most widely separated intervals *of the same source*.

    Screen anchoring asks whether a pattern sits in screen coordinates rather than in the picture.
    That needs unrelated picture content through the **same acquisition or scan geometry** — which
    two distant intervals of one film satisfy exactly, and two different cameras do not satisfy at
    all. Comparing Pulp Fiction against a Sony clip could never have answered it, geometry aside.

    "Unrelated" is the load-bearing word, and it is enforced: see :data:`MIN_ANCHOR_SEPARATION_S`.
    """
    if len(measurable) < 2:
        return {"available": False, "reason": "needs two admissible intervals from one source"}
    first, last = measurable[0], measurable[-1]
    separation = abs(last[0].start_s - first[0].start_s)
    pairs: list[dict[str, Any]] = []
    for left in range(len(measurable)):
        for right in range(left + 1, len(measurable)):
            gap = abs(measurable[right][0].start_s - measurable[left][0].start_s)
            if gap < MIN_ANCHOR_SEPARATION_S:
                continue
            a, b = measurable[left], measurable[right]
            pairs.append(
                {
                    "interval_a_s": a[0].start_s,
                    "interval_b_s": b[0].start_s,
                    "separation_s": gap,
                    "grain_envelope": evidence.heterogeneity_evidence(
                        a[1], a[3], other_source=b[1]
                    ).as_record(),
                    "additive_pattern": evidence.additive_pattern_evidence(
                        a[1], other_source=b[1]
                    ).as_record(),
                }
            )
    if separation < MIN_ANCHOR_SEPARATION_S:
        return {
            "available": False,
            "separation_s": separation,
            "reason": (
                f"the two intervals are only {separation:.0f}s apart, below the "
                f"{MIN_ANCHOR_SEPARATION_S:.0f}s needed for unrelated pictures; a correlation "
                "here would measure the same scene, not screen-anchored structure"
            ),
        }
    envelopes = [p["grain_envelope"]["screen_anchored_correlation"] for p in pairs]
    additives = [p["additive_pattern"]["cross_source_correlation"] for p in pairs]
    return {
        "available": True,
        "pairs": len(pairs),
        "widest_separation_s": separation,
        "grain_envelope": Spread(tuple(v for v in envelopes if v is not None)).as_record(),
        "additive_pattern": Spread(tuple(v for v in additives if v is not None)).as_record(),
        "per_pair": pairs,
        "claim": (
            "No strong scan-fixed envelope was detected between these sampled interval pairs. "
            "A correlation near zero rejects a strong common pattern; it does not rule out a "
            "weaker scan-fixed component, and no null distribution was computed."
        ),
    }


def _empty(
    plan: SourcePlan,
    record: SourceRecord,
    stages: Sequence[StageCount],
    notes: Sequence[str],
    seconds: Seconds,
    *,
    regions: rg.RegionIndex | None = None,
) -> SourceOutcome:
    return SourceOutcome(
        plan=plan,
        record=record,
        stages=tuple(stages),
        regions=regions or rg.index([]),
        seconds=seconds,
        per_interval=(),
        notes=tuple(notes),
    )


@dataclass(frozen=True)
class StudyResult:
    """Every source, measured interval by interval."""

    outcomes: tuple[SourceOutcome, ...]

    def as_record(self) -> dict[str, Any]:
        return {
            "study": "grain_slice",
            "question": (
                "Does the whole chain — source record, catalogue query, extraction, "
                "admissibility, window selection, residual measurement — run on real material, "
                "and what fraction of it survives?"
            ),
            "level_scale": "linear, 1.0 = diffuse white, for every source",
            "sources": [outcome.as_record() for outcome in self.outcomes],
        }


def run(plans: Sequence[SourcePlan]) -> StudyResult:
    """Run the slice over every plan.

    Screen anchoring is answered *within* each source, between its own distant intervals, so this
    no longer needs two sources to be meaningful — but two unrelated sources remain the point of
    the slice, because a chain that works on one kind of material has not been shown to work.
    """
    if len(plans) < 2:
        raise DataError(
            f"the slice needs at least two unrelated sources; got {len(plans)}. "
            "A chain shown to work on one kind of material has not been shown to work."
        )
    return StudyResult(outcomes=tuple(run_source(plan) for plan in plans))


# ------------------------------------------------------------------ human report


def report(result: StudyResult) -> str:
    lines = [
        "grain slice — source record to evidence, end to end",
        "=" * 78,
        "level scale: linear, 1.0 = diffuse white (every source)",
        "",
    ]
    for outcome in result.outcomes:
        record = outcome.record
        lines += [
            f"{record.source_id}   [{record.edition}]",
            f"  {record.coded_width}x{record.coded_height}  {record.cadence} fps  "
            f"{outcome.plan.transfer}  sha256 {record.sha256[:12]}",
            f"  active picture {record.active_picture[2]}x{record.active_picture[3]}"
            + ("" if record.crop.is_full_frame else f" (letterbox cropped {record.crop.y}px)"),
            "  coverage:",
        ]
        lines += [stage.line() for stage in outcome.stages]
        lines.append(outcome.seconds.line())
        if len(outcome.regions):
            lines.append("  " + outcome.regions.summary().replace("\n", "\n  "))

        if outcome.measured:
            identified, trusted, total = outcome.trusted_points()
            lines += [
                f"  measurements ({len(outcome.per_interval)} intervals, each measured "
                "independently):",
                f"    raw sigma    {outcome.raw_sigma().line()}   (all points — descriptive only)",
                f"    trusted sig  {outcome.trusted_sigma().line()}   (identified rho only)",
                f"    whiteness    {outcome.whiteness().line()}",
                f"    kurtosis     {outcome.kurtosis().line()}   (pooled, all unclipped)",
                f"      normalised {outcome.kurtosis_normalised().line()}   "
                "(per-window scale removed)",
                f"      mixing pred{outcome.mixing_prediction().line()}   "
                "(expected from variance spread alone)",
                f"      trustworthy{outcome.kurtosis_trustworthy().line()}",
                f"    skew         {outcome.skew().line()}",
                f"    exact zeros  {outcome.exact_zero().line()}",
                f"    rho          {outcome.rho().line()}",
                f"    envelope     {outcome.envelope().line()}",
                f"    amplitude    {identified} identified / {trusted} trustworthy / {total} "
                f"points   ({outcome.saturated_points()} rejected at the rho bound)",
            ]
            routed = [
                f"{one.flat_windows}f/{one.unclipped_windows}u/{one.trustworthy_windows}t "
                f"of {one.windows}"
                for one in outcome.per_interval
            ]
            lines.append(f"    routed       flat/unclipped/trustworthy: {'; '.join(routed)}")
            for one in outcome.per_interval:
                for reason in one.skipped:
                    lines.append(f"      t={one.interval_start_s:.0f}s  {reason}")
        else:
            lines.append("  measurements: NONE — nothing survived the chain")

        if outcome.deep:
            lines.append(f"  deep probe ({outcome.deep[0].frames} frames per tile):")
            lines += [one.line() for one in outcome.deep]

        anchoring = outcome.screen_anchoring
        if anchoring.get("available"):
            lines += [
                f"  screen anchoring ({anchoring['pairs']} interval pairs, up to "
                f"{anchoring['widest_separation_s']:.0f}s apart):",
                "    grain envelope    "
                + Spread(tuple(anchoring["grain_envelope"]["values"])).line(),
                "    additive pattern  "
                + Spread(tuple(anchoring["additive_pattern"]["values"])).line(),
                f"    {anchoring['claim']}",
            ]
        elif anchoring:
            lines.append(f"  screen anchoring: unavailable — {anchoring.get('reason')}")
        for note in outcome.notes:
            lines.append(f"  note: {note}")
        lines.append("")
    return "\n".join(lines)


def _redact(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip absolute paths, leaving the content hash as the identity.

    The material lives outside every repository, so a committed result must not carry a path into
    someone's home directory. The sha256 is what makes the source findable anyway.
    """
    for source in payload.get("sources", []):
        record = source.get("source", {})
        if record.get("path_hint"):
            record["path_hint"] = Path(record["path_hint"]).name
    return payload


def write_outputs(result: StudyResult, directory: Path) -> tuple[Path, Path]:
    """Evidence JSON and the human report, side by side."""
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "grain_slice.json"
    text_path = directory / "grain_slice.txt"
    json_path.write_text(json.dumps(_redact(result.as_record()), indent=2, allow_nan=False))
    text_path.write_text(report(result) + "\n")
    return json_path, text_path


__all__ = [
    "MAX_WINDOW_CLIPPED",
    "MIN_ANCHOR_SEPARATION_S",
    "DeepProbe",
    "IntervalEvidence",
    "Seconds",
    "SourceOutcome",
    "SourcePlan",
    "Spread",
    "StageCount",
    "StudyResult",
    "TailDiagnostics",
    "decode_signal",
    "decode_tile",
    "deep_probe",
    "detect_crop",
    "load_survey",
    "probe",
    "report",
    "run",
    "run_source",
    "source_record",
    "survey",
    "tail_diagnostics",
    "to_linear",
    "write_outputs",
]
