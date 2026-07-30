"""Parameter specifications, calibration fits, behaviour vectors, cross-validation.

Depends on the ForwardModel protocol in core, never on a specific adapter, so fits
are testable against a synthetic known-answer model with no engine and no footage.
"""

from __future__ import annotations

from film_analysis_tools.capabilities.fit import amplitude

__all__ = ["amplitude"]
