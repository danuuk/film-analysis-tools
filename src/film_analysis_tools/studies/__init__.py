"""Declared units of validation work.

A study states what it is asking and what would overturn it. The fields exist to stop the
accretion that produced the legacy system — a hypothesis appeared, a test was added, the
next reused and extended it, and the reasoning ended up in a document nothing linked to.

Required of every study above ``Tier.PROBE``:

``question``
    The hypothesis, in one sentence.
``rationale``
    Why it exists, and what prompted it.
``assumptions``
    What must be true of the system under test for this study to mean anything. These
    **execute first** and abort with a clear message — the antidote to building a tool for
    days before discovering the transform never behaved as assumed.
``controls``
    At minimum a null control; more at higher tiers.
``falsified_by``
    What result would overturn the conclusion.
``supersedes``
    Which earlier studies this replaces.
"""

from __future__ import annotations

__all__: list[str] = []
