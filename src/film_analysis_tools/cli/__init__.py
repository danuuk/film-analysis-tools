"""Command-line entry points.

The only layer permitted to parse arguments or exit the process. Everything below is a
library that raises exceptions and returns values.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from film_analysis_tools import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="film-analysis",
        description="Measurement, fitting and validation tooling for film emulation work.",
    )
    parser.add_argument("--version", action="version", version=f"film-analysis {__version__}")
    parser.set_defaults(handler=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.handler is None:
        parser.print_help(sys.stderr)
        return 2
    handler: object = args.handler
    assert callable(handler)
    result = handler(args)
    return int(result)


__all__ = ["build_parser", "main"]
