"""Source access: decode, frame extraction, scene info, input-domain intake.

Reads through catalogue queries against hash-identified sources, never through
folder layout (MIGRATION_PLAN.md section 5). Sheds the legacy argparse.Namespace
coupling: decode settings are explicit typed objects, not CLI argument bags.
"""

from __future__ import annotations

from film_analysis_tools.capabilities.source import record, slog3
from film_analysis_tools.capabilities.source.record import (
    Cadence,
    Crop,
    DecodeContract,
    SourceRecord,
)

__all__ = [
    "Cadence",
    "Crop",
    "DecodeContract",
    "SourceRecord",
    "record",
    "slog3",
]
