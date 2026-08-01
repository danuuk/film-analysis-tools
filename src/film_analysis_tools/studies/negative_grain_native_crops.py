"""Native-pixel motion review for the bounded N2 default-strength decision.

This is deliberately a visual deliverable, not another measurement study. FEE renders the exact
FullHD float chain. Worker processes immediately extract a few declared 1:1 crops, and FAT forms
normal / signed-delta / magnitude rows without any spatial resampling.
"""

from __future__ import annotations

import math
import multiprocessing
import shutil
import subprocess
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

from film_analysis_tools.capabilities.colour.display import srgb_encode
from film_analysis_tools.capabilities.report.html import STYLE
from film_analysis_tools.core.errors import DataError
from film_analysis_tools.core.io import write_json
from film_analysis_tools.forward.negative_grain_synthetic import (
    NegativeGrainFrame,
    NegativeGrainSyntheticAdapter,
)
from film_analysis_tools.studies.negative_grain_synthetic import (
    FIXED_DELTA_DISPLAY_LIMIT,
    magnitude_delta_view,
    signed_delta_view,
)

NATIVE_VARIANT_IDS = ("n0", "n2_075", "n2_100", "n2_125")
DISPLAY_VARIANT_IDS = ("n2_075", "n2_100", "n2_125")


@dataclass(frozen=True, slots=True)
class NativeCropSpec:
    crop_id: str
    label: str
    question: str
    region_id: str
    width: int
    height: int
    tracking: str


NATIVE_CROPS: tuple[NativeCropSpec, ...] = (
    NativeCropSpec(
        "skin_proxy",
        "Skin proxy with low-contrast detail",
        "Does the selected strength preserve subtle warm texture without chromatic crawling?",
        "skin_texture",
        320,
        180,
        "fixed scene region",
    ),
    NativeCropSpec(
        "deep_shadow",
        "Deep neutral shadow",
        "Is grain visible and coherent at -8 stops without becoming sparse coloured dots?",
        "neutral_step_01",
        160,
        96,
        "fixed scene region",
    ),
    NativeCropSpec(
        "neutral_midtone",
        "Neutral 18% midtone",
        "Do 0.75, 1.0, and 1.25 separate cleanly at the reference midtone?",
        "neutral_step_05",
        160,
        96,
        "fixed scene region",
    ),
    NativeCropSpec(
        "moving_colour",
        "Tracked coloured disc over a midtone",
        "Does motion expose chromatic boil or a persistent tint around coloured structure?",
        "moving_midtone_disc",
        256,
        128,
        "crop centre follows the exact FEE interior mask",
    ),
)


@dataclass(frozen=True, slots=True)
class NativeCropRunConfig:
    n1_bundle: Path
    n2_bundle: Path
    output_dir: Path
    frame_count: int = 96
    seed: int = 20260731
    frame_workers: int = 4
    delta_display_limit: float = FIXED_DELTA_DISPLAY_LIMIT

    def __post_init__(self) -> None:
        if self.frame_count < 24:
            raise DataError("native crop review requires at least one second (24 frames)")
        if self.frame_workers < 1 or self.frame_workers > 12:
            raise DataError("frame workers must be between 1 and 12")
        if not math.isfinite(self.delta_display_limit) or self.delta_display_limit <= 0.0:
            raise DataError("delta display limit must be finite and positive")


def _crop_bounds(
    region_bounds: tuple[int, int, int, int],
    *,
    crop_width: int,
    crop_height: int,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    top, bottom, left, right = region_bounds
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    crop_left = max(0, min(frame_width - crop_width, center_x - crop_width // 2))
    crop_top = max(0, min(frame_height - crop_height, center_y - crop_height // 2))
    return crop_top, crop_top + crop_height, crop_left, crop_left + crop_width


def native_triptych(output: np.ndarray, baseline: np.ndarray, *, limit: float) -> np.ndarray:
    """Make one unresized normal / signed-grey / magnitude-black RGB8 row."""
    candidate = np.asarray(output, dtype=np.float64)
    control = np.asarray(baseline, dtype=np.float64)
    if candidate.shape != control.shape or candidate.ndim != 3 or candidate.shape[2] != 3:
        raise DataError("native crop candidate and baseline must be matching RGB images")
    delta = candidate - control
    composite = np.concatenate(
        (
            srgb_encode(candidate),
            signed_delta_view(delta, limit=limit),
            magnitude_delta_view(delta, limit=limit),
        ),
        axis=1,
    )
    return np.rint(composite * 255.0).clip(0, 255).astype(np.uint8)


def _native_crop_frames(
    frame: NegativeGrainFrame,
    *,
    limit: float,
) -> dict[str, np.ndarray]:
    regions = {region.region_id: region for region in frame.regions}
    baseline = frame.outputs["n0"]
    frame_height, frame_width = baseline.shape[:2]
    result: dict[str, np.ndarray] = {}
    for spec in NATIVE_CROPS:
        region = regions.get(spec.region_id)
        if region is None:
            raise DataError(f"FEE frame omitted native crop region {spec.region_id!r}")
        top, bottom, left, right = _crop_bounds(
            region.bounds_yxyx,
            crop_width=spec.width,
            crop_height=spec.height,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        control_crop = baseline[top:bottom, left:right]
        rows = [
            native_triptych(
                frame.outputs[variant_id][top:bottom, left:right],
                control_crop,
                limit=limit,
            )
            for variant_id in DISPLAY_VARIANT_IDS
        ]
        result[spec.crop_id] = np.ascontiguousarray(np.concatenate(rows, axis=0))
    return result


class _LosslessRgbVideoWriter:
    """Raw RGB8 to lossless H.264 RGB 4:4:4; no chroma subsampling or spatial resize."""

    def __init__(self, path: Path, shape: tuple[int, int], fps: float) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise DataError("ffmpeg is required to assemble native crop videos")
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
                "libx264rgb",
                "-preset",
                "medium",
                "-crf",
                "0",
                "-pix_fmt",
                "rgb24",
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
            raise DataError(f"ffmpeg failed for {self.path.name}: {error.strip()}")


_FRAME_ADAPTER: NegativeGrainSyntheticAdapter | None = None
_FRAME_LIMIT = FIXED_DELTA_DISPLAY_LIMIT


def _initialize_worker(
    n1_bundle: str,
    n2_bundle: str,
    frame_count: int,
    seed: int,
    limit: float,
) -> None:
    global _FRAME_ADAPTER, _FRAME_LIMIT
    _FRAME_ADAPTER = NegativeGrainSyntheticAdapter(
        Path(n1_bundle),
        Path(n2_bundle),
        width=1920,
        height=1080,
        frame_count=frame_count,
        seed=seed,
        max_workers=1,
        variant_ids=NATIVE_VARIANT_IDS,
    )
    _FRAME_LIMIT = limit


def _render_worker(frame_index: int) -> tuple[int, dict[str, np.ndarray]]:
    if _FRAME_ADAPTER is None:
        raise RuntimeError("native crop frame worker was not initialized")
    frame = _FRAME_ADAPTER.render_frame(frame_index)
    return frame_index, _native_crop_frames(frame, limit=_FRAME_LIMIT)


def _rendered_frames(config: NativeCropRunConfig) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=config.frame_workers,
        mp_context=context,
        initializer=_initialize_worker,
        initargs=(
            str(config.n1_bundle),
            str(config.n2_bundle),
            config.frame_count,
            config.seed,
            config.delta_display_limit,
        ),
    ) as executor:
        pending: dict[int, Future[tuple[int, dict[str, np.ndarray]]]] = {}
        next_to_submit = 0
        while next_to_submit < min(config.frame_workers, config.frame_count):
            pending[next_to_submit] = executor.submit(_render_worker, next_to_submit)
            next_to_submit += 1
        for frame_index in range(config.frame_count):
            rendered = pending.pop(frame_index).result()
            if next_to_submit < config.frame_count:
                pending[next_to_submit] = executor.submit(_render_worker, next_to_submit)
                next_to_submit += 1
            yield rendered


def _report_html(payload: dict[str, Any]) -> str:
    cards: list[str] = []
    for crop in payload["crops"]:
        cards.append(
            "<section class='card'>"
            f"<h2>{escape(crop['label'])}</h2>"
            f"<p><strong>Question:</strong> {escape(crop['question'])}</p>"
            f"<p class='small'>Native panel: {crop['width']}x{crop['height']} px; "
            f"{escape(crop['tracking'])}. No resize.</p>"
            f"<video controls loop muted playsinline preload='metadata' "
            f"src='{escape(crop['video'])}'></video>"
            "</section>"
        )
    rows = "".join(
        f"<tr><td>{escape(row['strength'])}</td><td>{escape(row['role'])}</td></tr>"
        for row in payload["row_order"]
    )
    body = (
        "<main><h1>Native-pixel N2 strength review</h1>"
        "<p class='lede'>A bounded visual decision aid. Every source frame is rendered by the "
        "exact 1920x1080 FEE chain. Crop pixels are never resized, and the review streams use "
        "lossless RGB H.264 4:4:4 rather than 4:2:0.</p>"
        "<div class='callout'><strong>Each video matrix:</strong> rows are N2 strengths; columns "
        "are normal output, signed delta on neutral grey, and absolute magnitude on black. "
        "All delta views use the same fixed ±0.05 linear-RGB scale.</div>"
        f"<table><thead><tr><th>Row</th><th>Meaning</th></tr></thead><tbody>{rows}</tbody></table>"
        "<p><button id='play'>Play all</button> <button id='pause'>Pause all</button> "
        "<button id='restart'>Restart all</button></p>"
        f"{''.join(cards)}"
        "<section class='card'><h2>Decision rule</h2><p>If 1.0 looks clearly balanced, freeze it. "
        "If the three remain indistinguishable, retain 1.0 as the semantically clean default and "
        "stop optimizing. Natural clips are only a final veto for objectionable skin or shadow "
        "behaviour.</p></section></main>"
        "<script>const vs=[...document.querySelectorAll('video')];"
        "document.querySelector('#play').onclick=async()=>{const t=vs[0]?.currentTime||0;"
        "vs.forEach(v=>v.currentTime=t);await Promise.allSettled(vs.map(v=>v.play()));};"
        "document.querySelector('#pause').onclick=()=>vs.forEach(v=>v.pause());"
        "document.querySelector('#restart').onclick=()=>vs.forEach(v=>{v.pause();v.currentTime=0;});"
        "setInterval(()=>{if(vs.length&& !vs[0].paused){const t=vs[0].currentTime;"
        "vs.slice(1).forEach(v=>{if(Math.abs(v.currentTime-t)>.06)v.currentTime=t;});}},200);"
        "</script>"
    )
    extra_style = (
        "video{display:block;max-width:100%;height:auto;background:#111}"
        ".small{color:#667085}.callout{padding:1rem;background:#eef4ff;border-radius:.5rem}"
        "button{padding:.5rem .8rem;margin-right:.3rem}"
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Native-pixel N2 strength review</title><style>{STYLE}{extra_style}</style>"
        f"</head><body>{body}</body></html>\n"
    )


def run(
    config: NativeCropRunConfig,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Render four synchronized native crops and their self-contained review report."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_adapter = NegativeGrainSyntheticAdapter(
        config.n1_bundle,
        config.n2_bundle,
        width=1920,
        height=1080,
        frame_count=config.frame_count,
        seed=config.seed,
        max_workers=1,
        variant_ids=NATIVE_VARIANT_IDS,
    )
    fps = metadata_adapter.cadence[0] / metadata_adapter.cadence[1]
    writers: dict[str, _LosslessRgbVideoWriter] = {}
    try:
        for frame_index, crops in _rendered_frames(config):
            for spec in NATIVE_CROPS:
                frame = crops[spec.crop_id]
                writer = writers.get(spec.crop_id)
                if writer is None:
                    writer = _LosslessRgbVideoWriter(
                        config.output_dir / f"{spec.crop_id}.mp4",
                        frame.shape[:2],
                        fps,
                    )
                    writers[spec.crop_id] = writer
                writer.write(frame)
            if progress is not None:
                progress(frame_index + 1, config.frame_count)
    finally:
        for writer in writers.values():
            writer.close()

    variants = {variant.variant_id: variant for variant in metadata_adapter.variants}
    payload: dict[str, Any] = {
        "schema_version": "negative-grain-native-crop-review/v1",
        "generated": datetime.now(UTC).isoformat(),
        "source": {
            "geometry": [1920, 1080],
            "frame_count": config.frame_count,
            "cadence": list(metadata_adapter.cadence),
            "seed": config.seed,
            "parallel_frame_processes": config.frame_workers,
            "bundle_identity": metadata_adapter.bundle_identity,
        },
        "encoding": {
            "codec": "H.264/AVC libx264rgb",
            "mode": "lossless CRF 0",
            "chroma": "RGB 4:4:4; no chroma subsampling",
            "spatial_resampling": False,
            "visualization_quantization": "explicit RGB8 after fixed display mapping",
        },
        "matrix": {
            "columns": ["normal", "signed_delta_neutral_grey", "magnitude_on_black"],
            "fixed_delta_limit_linear_rgb": config.delta_display_limit,
            "per_frame_normalization": False,
        },
        "row_order": [
            {
                "variant_id": variant_id,
                "strength": variants[variant_id].label,
                "role": variants[variant_id].role,
            }
            for variant_id in DISPLAY_VARIANT_IDS
        ],
        "crops": [
            {
                "crop_id": spec.crop_id,
                "label": spec.label,
                "question": spec.question,
                "region_id": spec.region_id,
                "width": spec.width,
                "height": spec.height,
                "tracking": spec.tracking,
                "video": f"{spec.crop_id}.mp4",
                "composite_geometry": [spec.width * 3, spec.height * 3],
            }
            for spec in NATIVE_CROPS
        ],
        "decision_rule": {
            "balanced": "freeze 1.0",
            "indistinguishable": "retain semantic default 1.0 and stop optimizing",
            "natural_material": "few clips as veto only, not another corpus fit",
        },
    }
    write_json(config.output_dir / "manifest.json", payload)
    (config.output_dir / "index.html").write_text(_report_html(payload), encoding="utf-8")
    return payload


__all__ = [
    "DISPLAY_VARIANT_IDS",
    "NATIVE_CROPS",
    "NATIVE_VARIANT_IDS",
    "NativeCropRunConfig",
    "NativeCropSpec",
    "native_triptych",
    "run",
]
