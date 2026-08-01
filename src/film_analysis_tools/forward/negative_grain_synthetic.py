"""Single adapter onto FEE's controlled N0/N1/N2 float renderer.

The engine dependency is imported only when this adapter is constructed. Keeping that optional
runtime seam here lets the rest of film-analysis-tools remain usable without an engine checkout,
while preserving the architectural rule that no other layer imports an emulation model.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Literal

import numpy as np

from film_analysis_tools.core.errors import DataError

RegionMotion = Literal["static", "tracked"]


@dataclass(frozen=True, slots=True)
class NegativeGrainVariant:
    variant_id: str
    label: str
    family: str
    strength: float
    role: str


@dataclass(frozen=True, slots=True)
class NegativeGrainRegion:
    region_id: str
    category: str
    description: str
    motion: RegionMotion
    bounds_yxyx: tuple[int, int, int, int]
    scene_luma: float
    exposure_stops_from_grey: float | None


@dataclass(frozen=True, slots=True)
class NegativeGrainFrame:
    frame_index: int
    elapsed_time_seconds: float
    regions: tuple[NegativeGrainRegion, ...]
    outputs: dict[str, np.ndarray]


class NegativeGrainSyntheticAdapter:
    """Typed FAT view of FEE's exact bundles, stimulus, masks, and float outputs."""

    def __init__(
        self,
        n1_bundle: Path,
        n2_bundle: Path,
        *,
        width: int = 1920,
        height: int = 1080,
        frame_count: int = 96,
        seed: int = 20260731,
        max_workers: int = 1,
        variant_ids: Collection[str] | None = None,
    ) -> None:
        try:
            engine = import_module("film_emulation_engine.synthetic")
        except ModuleNotFoundError as exc:
            raise DataError(
                "the synthetic negative-grain study needs film-emulation-engine installed; "
                "run it from the FEE environment with film-analysis-tools on PYTHONPATH"
            ) from exc
        clip = engine.negative_grain_synthetic_clip(
            width=width,
            height=height,
            frame_count=frame_count,
        )
        try:
            self._renderer: Any = engine.NegativeGrainValidationRenderer.from_bundle_paths(
                Path(n1_bundle),
                Path(n2_bundle),
                clip=clip,
                seed=seed,
                max_workers=max_workers,
                variant_ids=variant_ids,
            )
        except (OSError, ValueError) as exc:
            raise DataError(f"cannot initialize FEE negative-grain renderer: {exc}") from exc
        self.variants = tuple(
            NegativeGrainVariant(
                variant_id=str(variant.variant_id),
                label=str(variant.label),
                family=str(variant.family),
                strength=float(variant.strength),
                role=str(variant.role),
            )
            for variant in self._renderer.variants
        )

    @property
    def width(self) -> int:
        return int(self._renderer.clip.width)

    @property
    def height(self) -> int:
        return int(self._renderer.clip.height)

    @property
    def frame_count(self) -> int:
        return int(self._renderer.clip.frame_count)

    @property
    def cadence(self) -> tuple[int, int]:
        return (
            int(self._renderer.clip.cadence_numerator),
            int(self._renderer.clip.cadence_denominator),
        )

    @property
    def clip_id(self) -> str:
        return str(self._renderer.clip.clip_id)

    @property
    def bundle_identity(self) -> dict[str, dict[str, str]]:
        return {name: dict(identity) for name, identity in self._renderer.bundle_identity.items()}

    def render_frame(self, frame_index: int) -> NegativeGrainFrame:
        rendered = self._renderer.render_frame(frame_index)
        source = rendered.source
        regions_list: list[NegativeGrainRegion] = []
        for region in source.regions:
            bounds = tuple(int(value) for value in region.bounds_yxyx)
            if len(bounds) != 4:
                raise DataError(f"FEE region {region.region_id!r} returned invalid bounds")
            regions_list.append(
                NegativeGrainRegion(
                    region_id=str(region.region_id),
                    category=str(region.category),
                    description=str(region.description),
                    motion=region.motion,
                    bounds_yxyx=(bounds[0], bounds[1], bounds[2], bounds[3]),
                    scene_luma=float(region.scene_luma),
                    exposure_stops_from_grey=(
                        None
                        if region.exposure_stops_from_grey is None
                        else float(region.exposure_stops_from_grey)
                    ),
                )
            )
        regions = tuple(regions_list)
        return NegativeGrainFrame(
            frame_index=int(source.frame_index),
            elapsed_time_seconds=float(source.elapsed_time_seconds),
            regions=regions,
            outputs={name: np.asarray(array) for name, array in rendered.outputs.items()},
        )


__all__ = [
    "NegativeGrainFrame",
    "NegativeGrainRegion",
    "NegativeGrainSyntheticAdapter",
    "NegativeGrainVariant",
]
