"""Bootstrap smoke: the package imports, the CLI runs, the tier vocabulary behaves."""

from __future__ import annotations

import pytest

import film_analysis_tools
from film_analysis_tools.cli import build_parser, main
from film_analysis_tools.core import Tier


def test_package_exposes_a_version() -> None:
    assert film_analysis_tools.__version__


def test_cli_reports_version() -> None:
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["--version"])
    assert excinfo.value.code == 0


def test_cli_without_a_subcommand_returns_usage_exit_code() -> None:
    assert main([]) == 2


def test_tiers_compare_by_rigour() -> None:
    assert Tier.PROBE < Tier.FROZEN
    assert max(Tier, key=lambda tier: tier.rank) is Tier.FROZEN
    assert Tier("comparison") is Tier.COMPARISON
