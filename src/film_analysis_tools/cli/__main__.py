"""Allow ``python -m film_analysis_tools.cli`` in cross-repository environments."""

from __future__ import annotations

from . import main

if __name__ == "__main__":
    raise SystemExit(main())
