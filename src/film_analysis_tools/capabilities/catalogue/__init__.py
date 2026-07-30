"""The sample catalogue: which material exercises which validation condition.

This is the clean boundary between the analytical tool and the camera corpus. It replaces the
legacy catalogues rather than wrapping them — nothing here reads legacy paths, legacy metadata,
or legacy scoring, and none of the legacy generations' structure survives.

Three properties the generations it replaces lacked:

* **Content-hash identity.** A clip is its digest; the recorded path is a hint. A renamed or
  moved source is re-found rather than lost, which is exactly how the legacy corpus lost the
  basis of its samples.
* **One taxonomy, chosen after measurement.** Categories name the failure they provoke, and
  their thresholds sit against the observed distribution of the whole corpus.
* **Complete coverage.** Every clip appears, including the ordinary ones no category claimed,
  so nothing is silently dropped.

Typical use::

    from film_analysis_tools.capabilities import catalogue

    cat = catalogue.bundled()
    for clip in cat.select("deep_underexposure", "saturated_practical", require_all=True):
        path = clip.locate()               # verified against its content hash
"""

from __future__ import annotations

from film_analysis_tools.capabilities.catalogue import (
    annotate,
    categories,
    ingest,
    intervals,
    regions,
    survey,
)
from film_analysis_tools.capabilities.catalogue.categories import (
    CATEGORIES,
    Category,
    classify,
)
from film_analysis_tools.capabilities.catalogue.manifest import (
    SCHEMA_VERSION,
    Catalogue,
    CatalogueClip,
    bundled,
    file_sha256,
    load,
)

__all__ = [
    "CATEGORIES",
    "SCHEMA_VERSION",
    "Catalogue",
    "CatalogueClip",
    "Category",
    "annotate",
    "bundled",
    "categories",
    "classify",
    "file_sha256",
    "ingest",
    "intervals",
    "load",
    "regions",
    "survey",
]
