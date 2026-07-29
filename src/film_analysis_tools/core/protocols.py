"""Shared abstractions.

``Transform`` lives here rather than in ``forward`` so that capabilities can depend on the
*abstraction* without depending on any concrete adapter. A fit or a comparison must be
runnable against a synthetic, known-answer transform with no engine build and no footage;
that is what keeps the statistical layers testable (``ARCHITECTURE.md``).

The richer ``ForwardModel`` — parameter spaces, render context, model identity — arrives with
the adapters in P8 and will extend this same seam.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

RGB = npt.NDArray[np.float64]
"""An ``(N, 3)`` array of linear Rec.709 samples, or an ``(H, W, 3)`` image."""


@runtime_checkable
class Transform(Protocol):
    """Anything that maps RGB to RGB without changing shape.

    Implementations must be pure and deterministic: the same input yields the same output.
    """

    def __call__(self, rgb: RGB, /) -> RGB: ...


__all__ = ["RGB", "Transform"]
