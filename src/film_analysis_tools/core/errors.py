"""Exception hierarchy.

Library code raises; only ``cli`` decides what that means for the process. The legacy
system routed errors through ``SystemExit`` in 134 of 221 modules, which made every analyzer
unusable as a library and forced the orchestrator to drive stages as subprocesses.
"""

from __future__ import annotations


class FilmAnalysisError(Exception):
    """Base for every error this package raises deliberately."""


class WorkspaceError(FilmAnalysisError):
    """The workspace is unset, missing, or the requested name does not resolve inside it."""


class DataError(FilmAnalysisError):
    """Data on disk is absent, malformed, or inconsistent with what it claims to be."""


class SelectionError(FilmAnalysisError):
    """A cohort selector names an unknown column or an unsupported operator."""


class ControlError(FilmAnalysisError):
    """A comparison was asked to claim at a tier whose required controls are missing."""


__all__ = [
    "ControlError",
    "DataError",
    "FilmAnalysisError",
    "SelectionError",
    "WorkspaceError",
]
