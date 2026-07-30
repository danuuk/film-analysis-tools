"""The vertical slice: the parts that can be checked without the private sources.

The ffmpeg stages are exercised by running the study over real material, not here. What is worth
pinning down in a test is the reasoning the run depends on — which domain each question is asked
in, and whether coverage is reported honestly.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from film_analysis_tools.capabilities.measure import admissibility
from film_analysis_tools.core.errors import DataError
from film_analysis_tools.studies import grain_slice as gs

PQ_STREAM = {"width": 3840, "height": 2160, "color_range": "tv", "r_frame_rate": "24000/1001"}
SLOG_STREAM = {"width": 1920, "height": 1080, "color_range": "pc", "r_frame_rate": "24000/1001"}


def _plan(**kwargs: object) -> gs.SourcePlan:
    base: dict[str, object] = {
        "source_id": "s",
        "path": gs.Path("/nowhere.mkv"),
        "edition": "e",
        "transfer": "pq",
    }
    base.update(kwargs)
    return gs.SourcePlan(**base)  # type: ignore[arg-type]


def test_an_unknown_transfer_is_refused() -> None:
    with pytest.raises(DataError, match="unsupported transfer"):
        _plan(transfer="rec709")


# ------------------------------------------------------------- domains


def test_admissibility_is_asked_in_the_container_domain() -> None:
    """The defect that stopped the whole film measuring.

    Shadow codes survive as ordinary small numbers in the container, but a PQ EOTF compresses
    them until they underflow to zero. Asking about clipping after the transfer reported 87.6%
    of an ordinary film interval as clipped black, and 100% of the frame as carrying no noise.
    """
    rng = np.random.default_rng(0)
    bright = np.clip(0.90 + rng.normal(0.0, 0.004, (8, 128, 128)), 0.0, 1.0)

    in_container = admissibility.clipping_evidence(bright, ceiling=1.0, floor=0.0)
    linear = gs.to_linear(_plan(), bright)
    after_transfer = admissibility.clipping_evidence(linear, ceiling=1.0, floor=0.0)

    # Nothing here is clipped: it is an ordinary bright highlight, well inside the container.
    assert not in_container.is_clipped
    # But 1.0 no longer means "the top" after a PQ EOTF -- it means diffuse white, and a highlight
    # sits far above it. Asking the same question after the transfer calls the highlight clipped.
    assert linear.mean() > 10.0
    assert after_transfer.is_clipped
    assert after_transfer.total_fraction > in_container.total_fraction


def test_levels_land_on_a_common_scale_whatever_the_transfer() -> None:
    """1.0 must mean diffuse white for every source, or no two are comparable (gap 8).

    Checked end to end against a real decode in the integration tests below; this only pins the
    transfer maths, which is why it cannot stand on its own.
    """
    pq_white = gs.to_linear(_plan(transfer="pq"), np.full((2, 8, 8), 0.5073))
    slog_white = gs.to_linear(_plan(transfer="slog3"), np.full((2, 8, 8), 0.5977))
    assert pq_white.mean() == pytest.approx(1.0, rel=0.05)
    assert slog_white.mean() == pytest.approx(1.0, rel=0.15)


# ------------------------------------------------------------- coverage


def test_a_stage_reports_what_it_dropped_and_why() -> None:
    stage = gs.StageCount(
        stage="windows", considered=540, accepted=245, rejected={"motion above gate": 202}
    )
    assert stage.survival == pytest.approx(245 / 540)
    assert "245" in stage.line() and "540" in stage.line()
    assert "motion above gate 202" in stage.line()
    assert stage.as_record()["survival"] == pytest.approx(0.4537, abs=1e-3)


def test_a_stage_that_saw_nothing_does_not_divide_by_zero() -> None:
    assert gs.StageCount(stage="x", considered=0, accepted=0).survival == 0.0


def test_the_slice_refuses_to_run_on_one_source() -> None:
    """Screen anchoring is the question a single source cannot answer at all."""
    with pytest.raises(DataError, match="at least two unrelated sources"):
        gs.run([_plan()])


# ------------------------------------------------------------- reporting


def _outcome(**kwargs: object) -> gs.SourceOutcome:
    plan = _plan()
    record = gs.source_record(
        gs.SourcePlan(
            source_id="s", path=gs.Path(__file__), edition="e", transfer="pq", sha256="a" * 64
        ),
        PQ_STREAM,
        gs.Crop(x=0, y=263, width=3840, height=1634),
    )
    base: dict[str, object] = {
        "plan": plan,
        "record": record,
        "stages": (gs.StageCount(stage="intervals", considered=10, accepted=3),),
        "regions": gs.rg.index([]),
        "seconds": gs.Seconds(candidate=12.0, decoded=2.5, evidence=0.417),
        "per_interval": (),
    }
    base.update(kwargs)
    return gs.SourceOutcome(**base)  # type: ignore[arg-type]


def test_a_source_that_measured_nothing_says_so_plainly() -> None:
    text = gs.report(gs.StudyResult(outcomes=(_outcome(),)))
    assert "NONE — nothing survived the chain" in text
    assert "intervals" in text


def test_the_report_names_the_active_picture() -> None:
    """Letterbox bars sit at code 0 and read as hard-clipped black across a quarter of the
    frame; the crop that removes them has to be visible in the output."""
    text = gs.report(gs.StudyResult(outcomes=(_outcome(),)))
    assert "3840x1634" in text
    assert "letterbox cropped 263px" in text


def test_the_record_carries_provenance_and_coverage() -> None:
    record = gs.StudyResult(outcomes=(_outcome(),)).as_record()
    assert record["study"] == "grain_slice"
    assert "diffuse white" in record["level_scale"]
    source = record["sources"][0]
    assert source["source"]["sha256"] == "a" * 64
    assert source["source"]["active_picture"] == [0, 263, 3840, 1634]
    assert source["coverage"][0]["considered"] == 10
    assert source["intervals_measured"] == 0
    assert source["per_interval"] == []


# ------------------------------------------------- ffmpeg range handling (integration)


def _synthetic_clip(directory: Path, codes: Sequence[int], *, color_range: str) -> Path:
    """A 10-bit clip whose luma bands hold exactly the given codes."""
    width = height = 64
    frames = 4
    band = height // len(codes)
    luma = np.zeros((frames, height, width), dtype="<u2")
    for index, code in enumerate(codes):
        luma[:, index * band : (index + 1) * band, :] = code
    chroma = np.full((frames, height // 2, width // 2), 512, dtype="<u2")
    raw = b"".join(
        luma[f].tobytes() + chroma[f].tobytes() + chroma[f].tobytes() for f in range(frames)
    )
    path = directory / f"synthetic_{color_range}.mkv"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-s", f"{width}x{height}", "-r", "24",
            "-i", "-", "-c:v", "ffv1", "-pix_fmt", "yuv420p10le",
            "-color_range", color_range, "-colorspace", "bt2020nc",
            "-color_trc", "smpte2084", "-color_primaries", "bt2020", str(path),
        ],
        input=raw, check=True, capture_output=True,
    )  # fmt: skip
    return path


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_ffmpeg_normalises_limited_range_so_the_study_must_not(tmp_path: Path) -> None:
    """The defect this exists for, pinned against the installed ffmpeg.

    Converting limited-range YUV to gray16le **range-expands**: legal black 64 arrives as 0 and
    legal white 940 as 1023. The study applied the 64..940 mapping again on top of that, which
    crushed low PQ values toward zero and distorted every level, distribution and tail derived
    from this source. The previous unit test bypassed ffmpeg and so codified the wrong assumption.
    """
    codes = [64, 502, 940]  # legal black, mid, legal white
    clip = _synthetic_clip(tmp_path, codes, color_range="tv")
    plan = _plan(path=clip, transfer="pq")
    stream = gs.probe(clip)

    container = gs.decode_signal(plan, stream, 0.0, 2)
    band = container.shape[1] // len(codes)
    seen = [float(container[:, i * band + band // 2, :].mean()) for i in range(len(codes))]

    assert seen[0] == pytest.approx(0.0, abs=0.005), "legal black must arrive already at 0"
    assert seen[2] == pytest.approx(1.0, abs=0.005), "legal white must arrive already at 1"
    assert seen[1] == pytest.approx(0.5, abs=0.02)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_diffuse_white_lands_at_one_through_the_real_decode(tmp_path: Path) -> None:
    """PQ code 508 is 100 nits. End to end, that must come out at 1.0 = diffuse white."""
    codes = [64, 508, 940]  # 508 is 100 nits on the legal-range PQ axis
    clip = _synthetic_clip(tmp_path, codes, color_range="tv")
    plan = _plan(path=clip, transfer="pq")
    stream = gs.probe(clip)

    linear = gs.to_linear(plan, gs.decode_signal(plan, stream, 0.0, 2))
    band = linear.shape[1] // len(codes)
    black = float(linear[:, band // 2, :].mean())
    white = float(linear[:, band + band // 2, :].mean())

    assert black == pytest.approx(0.0, abs=1e-4)
    assert white == pytest.approx(1.0, rel=0.10)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_a_full_range_source_is_unaffected(tmp_path: Path) -> None:
    codes = [0, 512, 1023]
    clip = _synthetic_clip(tmp_path, codes, color_range="pc")
    plan = _plan(path=clip, transfer="slog3")
    stream = gs.probe(clip)

    container = gs.decode_signal(plan, stream, 0.0, 2)
    band = container.shape[1] // len(codes)
    assert float(container[:, band // 2, :].mean()) == pytest.approx(0.0, abs=0.005)
    assert float(container[:, -band // 2, :].mean()) == pytest.approx(1.0, abs=0.005)


# ------------------------------------------------------------- honest accounting


def test_three_different_seconds_are_reported_separately() -> None:
    """Catalogue support is not measurement support.

    Ten frames at 23.976 fps is 0.417 s, not the two seconds the interval spans. Reporting the
    span as though it were the measurement overstated support by roughly 5x per interval.
    """
    text = gs.report(gs.StudyResult(outcomes=(_outcome(),)))
    assert "candidate   12.0" in text
    assert "decoded  2.50" in text
    assert "evidence  0.42" in text
    assert _outcome().seconds.as_record() == {
        "candidate_s": 12.0,
        "decoded_s": 2.5,
        "evidence_s": 0.417,
    }


def test_a_spread_never_collapses_to_a_bare_number() -> None:
    spread = gs.Spread(values=(0.1, 0.2, 0.9))
    assert spread.median == pytest.approx(0.2)
    assert (spread.low, spread.high, spread.count) == (0.1, 0.9, 3)
    assert "n=3" in spread.line()
    assert spread.as_record()["values"] == [0.1, 0.2, 0.9]


def test_an_empty_spread_is_safe() -> None:
    assert gs.Spread(values=()).median == 0.0
    assert gs.Spread(values=()).line() == "none"


def test_screen_anchoring_refuses_intervals_that_share_a_picture() -> None:
    """Measured on the Sony clip: two intervals 3 s apart in a tripod shot returned 0.93 for the
    grain envelope and 0.99 for the additive pattern. That is the same scene twice, not a
    scan-fixed pattern, and the test must decline rather than report it."""
    assert gs.MIN_ANCHOR_SEPARATION_S >= 30.0
