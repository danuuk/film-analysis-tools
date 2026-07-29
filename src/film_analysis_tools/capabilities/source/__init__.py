"""Source access: decode, frame extraction, scene info, input-domain intake.

Reads through catalogue queries against hash-identified sources, never through
folder layout (MIGRATION_PLAN.md section 5). Sheds the legacy argparse.Namespace
coupling: decode settings are explicit typed objects, not CLI argument bags.
"""

from __future__ import annotations

__all__: list[str] = []
