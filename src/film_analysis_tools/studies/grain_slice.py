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
from film_analysis_tools.capabilities.measure import admissibility, evidence, windows
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


def survey(plan: SourcePlan, stream: dict[str, Any]) -> FrameSurvey:
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
                f"scale={plan.survey_width}:-2:flags=bicubic,fps={plan.survey_fps},"
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


def load_survey(plan: SourcePlan, stream: dict[str, Any]) -> FrameSurvey:
    """The kept survey when the plan names one, otherwise a fresh pass.

    Both paths use ``ydif`` for motion, normalised to a fraction of full scale. The kept survey
    also carries ``mafd``, which is about 10.2x smaller on the same footage — mixing the two would
    silently change what the motion gate means between sources.
    """
    if plan.survey_csv is None:
        return survey(plan, stream)

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
    x, y, width, height = active.applied_to(int(stream["width"]), int(stream["height"]))
    filters = (
        "format=gray16le"
        if active.is_full_frame and not active.width
        else (f"crop={width}:{height}:{x}:{y},format=gray16le")
    )
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


def to_linear(plan: SourcePlan, container: np.ndarray, stream: dict[str, Any]) -> np.ndarray:
    """The container signal as linear light, with 1.0 at diffuse white on every source.

    The legal-range offset is applied here, not in :func:`decode_signal`, so the container domain
    stays the one place where 0 and 1 are the format's real limits. Measuring clipping against the
    *legal* range instead counted ordinary footroom as clipped black — 42.8% of one Pulp Fiction
    interval, which is simply how much of that frame sits below code 64.

    Measurement happens here rather than in the code domain: amplitude against *level* only means
    something on a scale where equal steps are equal light.
    """
    full_range = str(stream.get("color_range", "tv")) == "pc"
    signal = container if full_range else (container * 1023.0 - 64.0) / (940.0 - 64.0)
    if plan.transfer == "pq":
        return _pq_to_linear(signal) * (10000.0 / DIFFUSE_WHITE_NITS)
    return np.asarray(slog3_to_linear(signal)) / DIFFUSE_WHITE_REFLECTANCE


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
class SourceOutcome:
    """Everything one source contributed, and everything it failed to."""

    plan: SourcePlan
    record: SourceRecord
    stages: tuple[StageCount, ...]
    regions: rg.RegionIndex
    frames_measured: int
    amplitude: evidence.AmplitudeEvidence | None
    spectrum: evidence.SpectrumEvidence | None
    distribution: evidence.DistributionEvidence | None
    heterogeneity: evidence.HeterogeneityEvidence | None
    temporal: evidence.TemporalEvidence | None
    notes: tuple[str, ...] = ()

    @property
    def measured(self) -> bool:
        return self.amplitude is not None

    def as_record(self) -> dict[str, Any]:
        return {
            "source": self.record.as_record(),
            "plan": {
                "transfer": self.plan.transfer,
                "level_scale": "linear, 1.0 = diffuse white",
                "window_s": self.plan.window_s,
                "tile_size": self.plan.tile_size,
                "intervals_requested": self.plan.intervals_to_measure,
            },
            "coverage": [stage.as_record() for stage in self.stages],
            "frames_measured": self.frames_measured,
            "regions": self.regions.as_record(),
            "evidence": {
                "amplitude": self.amplitude.as_record() if self.amplitude else None,
                "spectrum": self.spectrum.as_record() if self.spectrum else None,
                "distribution": self.distribution.as_record() if self.distribution else None,
                "heterogeneity": self.heterogeneity.as_record() if self.heterogeneity else None,
                "temporal": self.temporal.as_record() if self.temporal else None,
            },
            "notes": list(self.notes),
        }


def run_source(plan: SourcePlan) -> SourceOutcome:
    """The whole chain for one source, recording what was lost at every stage."""
    stream = probe(plan.path)
    active = detect_crop(plan, stream)
    record = source_record(plan, stream, active)
    notes: list[str] = []
    stages: list[StageCount] = []

    frame_survey = load_survey(plan, stream)
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
    if not stable:
        notes.append("no cut-free, low-motion interval exists in this source")
        return _empty(plan, record, stages, notes)

    chosen = [
        stable[i] for i in np.linspace(0, len(stable) - 1, plan.intervals_to_measure).astype(int)
    ]
    chosen = list({interval.start_s: interval for interval in chosen}.values())

    admissible: list[tuple[iv.Interval, np.ndarray, admissibility.OverlayEvidence]] = []
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
        # encoder froze — 20-22% of 32px blocks in this one have exactly zero temporal variation,
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
            admissible.append((interval, to_linear(plan, signal, stream), verdict.overlay))
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
    if not admissible:
        notes.append("every decoded interval was ruled inadmissible")
        return _empty(plan, record, stages, notes)

    collected: list[rg.Region] = []
    considered = accepted = 0
    window_rejects: dict[str, int] = {}
    measurable: list[tuple[windows.SelectionReport, np.ndarray]] = []
    for interval, frames, overlay in admissible:
        report = windows.select_windows(
            frames, size=plan.tile_size, stride=plan.tile_stride, overlay=overlay
        )
        considered += len(report.accepted) + len(report.rejected)
        accepted += len(report.accepted)
        for reason, count in report.rejection_reasons().items():
            window_rejects[reason] = window_rejects.get(reason, 0) + count
        collected.extend(
            rg.regions_from_report(
                report,
                source_id=plan.source_id,
                interval_start_s=interval.start_s,
                interval_end_s=interval.end_s,
                start_frame=record.cadence.frame_at(interval.start_s),
                level_scale="linear",
                source_identity=record.identity,
            )
        )
        if report.accepted:
            measurable.append((report, frames))
    stages.append(
        StageCount(
            stage="windows",
            considered=considered,
            accepted=accepted,
            rejected=window_rejects,
        )
    )
    index = rg.index(collected)
    if not measurable:
        notes.append("no tile passed the window gate in any admissible interval")
        return _empty(plan, record, stages, notes, regions=index)

    # Measure on the interval that yielded the most windows: the evidence producers take one
    # frame stack, and pooling stacks from different intervals would mix unrelated pictures.
    best_report, best_frames = max(measurable, key=lambda pair: len(pair[0].accepted))
    selected = list(best_report.accepted)
    return SourceOutcome(
        plan=plan,
        record=record,
        stages=tuple(stages),
        regions=index,
        frames_measured=int(best_frames.shape[0]),
        amplitude=evidence.amplitude_evidence(best_frames, selected),
        spectrum=evidence.spectrum_evidence(best_frames, selected),
        distribution=evidence.distribution_evidence(best_frames, selected),
        heterogeneity=evidence.heterogeneity_evidence(best_frames, selected),
        temporal=evidence.temporal_evidence(best_frames, selected),
        notes=tuple(notes),
    )


def _empty(
    plan: SourcePlan,
    record: SourceRecord,
    stages: Sequence[StageCount],
    notes: Sequence[str],
    *,
    regions: rg.RegionIndex | None = None,
) -> SourceOutcome:
    return SourceOutcome(
        plan=plan,
        record=record,
        stages=tuple(stages),
        regions=regions or rg.index([]),
        frames_measured=0,
        amplitude=None,
        spectrum=None,
        distribution=None,
        heterogeneity=None,
        temporal=None,
        notes=tuple(notes),
    )


@dataclass(frozen=True)
class StudyResult:
    """Both sources, and the questions only two unrelated sources can answer."""

    outcomes: tuple[SourceOutcome, ...]
    cross_source: dict[str, Any] = field(default_factory=dict)

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
            "cross_source": self.cross_source,
        }


def run(plans: Sequence[SourcePlan]) -> StudyResult:
    """Run the slice over every plan, then the cross-source checks.

    Screen anchoring is the one question a single source cannot answer at all: a pattern fixed in
    screen coordinates is indistinguishable from scene content until unrelated material is
    compared against it.
    """
    if len(plans) < 2:
        raise DataError(
            f"the slice needs at least two unrelated sources; got {len(plans)}. "
            "Screen anchoring cannot be assessed from one source."
        )
    outcomes = [run_source(plan) for plan in plans]
    return StudyResult(outcomes=tuple(outcomes), cross_source=_cross_source(plans, outcomes))


def _cross_source(plans: Sequence[SourcePlan], outcomes: Sequence[SourceOutcome]) -> dict[str, Any]:
    measured = [
        (plan, outcome) for plan, outcome in zip(plans, outcomes, strict=True) if outcome.measured
    ]
    if len(measured) < 2:
        return {
            "screen_anchoring": None,
            "note": "needs two sources that both produced measurements",
        }
    sigmas = {
        outcome.plan.source_id: float(
            np.median([point.sigma for point in outcome.amplitude.points])
        )
        for _, outcome in measured
        if outcome.amplitude
    }
    return {
        "screen_anchoring": None,
        "note": (
            "geometry differs between these sources, so the screen-anchored test cannot run; "
            "it needs two sources of identical frame geometry"
        ),
        "median_sigma_by_source": sigmas,
    }


# ------------------------------------------------------------------ human report


def report(result: StudyResult) -> str:
    lines = [
        "grain slice — source record to evidence, end to end",
        "=" * 72,
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
        if len(outcome.regions):
            lines.append("  " + outcome.regions.summary().replace("\n", "\n  "))
        if outcome.measured and outcome.amplitude:
            low, high = outcome.amplitude.level_range()
            sigmas = [point.sigma for point in outcome.amplitude.points]
            trusted = len(outcome.amplitude.trusted)
            lines += [
                "  measurements:",
                f"    sigma        median {np.median(sigmas):.5f}  "
                f"range {min(sigmas):.5f}..{max(sigmas):.5f}",
                f"    levels       {low:.4f}..{high:.4f}   "
                f"({trusted}/{len(sigmas)} points trustworthy)",
            ]
            if outcome.spectrum:
                lines.append(
                    f"    spectrum     whiteness {outcome.spectrum.whiteness:.3f} "
                    f"({'white' if outcome.spectrum.is_white else 'structured'})"
                )
            if outcome.distribution:
                lines.append(
                    f"    distribution kurtosis {outcome.distribution.excess_kurtosis:+.3f} "
                    f"({'gaussian' if outcome.distribution.is_gaussian else 'heavy-tailed'})"
                )
            if outcome.temporal:
                established = (
                    "established"
                    if outcome.temporal.independence_established
                    else "NOT established"
                )
                lines.append(
                    f"    temporal     rho {outcome.temporal.rho:+.4f}  independence {established}"
                )
            if outcome.heterogeneity:
                lines.append(
                    f"    envelope     ratio {outcome.heterogeneity.envelope_ratio:.4f}  "
                    "screen-anchoring unknown (needs matching geometry)"
                )
        else:
            lines.append("  measurements: NONE — nothing survived the chain")
        for note in outcome.notes:
            lines.append(f"  note: {note}")
        lines.append("")

    lines.append("cross-source:")
    for key, value in result.cross_source.items():
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def write_outputs(result: StudyResult, directory: Path) -> tuple[Path, Path]:
    """Evidence JSON and the human report, side by side."""
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "grain_slice.json"
    text_path = directory / "grain_slice.txt"
    json_path.write_text(json.dumps(result.as_record(), indent=2, allow_nan=False))
    text_path.write_text(report(result) + "\n")
    return json_path, text_path


__all__ = [
    "SourceOutcome",
    "SourcePlan",
    "StageCount",
    "StudyResult",
    "decode_signal",
    "detect_crop",
    "load_survey",
    "probe",
    "report",
    "run",
    "run_source",
    "source_record",
    "survey",
    "to_linear",
    "write_outputs",
]
