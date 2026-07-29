"""Command-line entry points.

The only layer permitted to parse arguments or exit the process. Everything below is a
library that raises exceptions and returns values — which is what makes in-process
orchestration possible, and what the legacy design could not do.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from film_analysis_tools import __version__
from film_analysis_tools.capabilities import report, sample
from film_analysis_tools.capabilities.colour import metrics, transforms
from film_analysis_tools.capabilities.sample import cohorts as cohorts_module
from film_analysis_tools.capabilities.statistics import compare_cohorts
from film_analysis_tools.capabilities.statistics.compare import per_sample_metric
from film_analysis_tools.core import FilmAnalysisError, Tier, Workspace, io

DEFAULT_COHORTS = "neutral,skin_like,foliage_like,shadows,highlights"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="film-analysis",
        description="Measurement, fitting and validation tooling for film emulation work.",
    )
    parser.add_argument("--version", action="version", version=f"film-analysis {__version__}")
    parser.add_argument(
        "--workspace",
        default=None,
        help="dataset root; defaults to the FILM_ANALYSIS_WORKSPACE environment variable",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="result root; defaults to FILM_ANALYSIS_OUTPUT, else ./results",
    )
    parser.set_defaults(handler=None)

    subparsers = parser.add_subparsers(dest="command")

    listing = subparsers.add_parser("packs", help="list sample packs in the workspace")
    listing.add_argument("--glob", default="*", help="restrict to names matching this pattern")
    listing.set_defaults(handler=_packs)

    describe = subparsers.add_parser("describe", help="shape and provenance of one pack")
    describe.add_argument("pack")
    describe.set_defaults(handler=_describe)

    cohorts = subparsers.add_parser("cohorts", help="row counts per cohort for one pack")
    cohorts.add_argument("pack")
    cohorts.set_defaults(handler=_cohorts)

    comparison = subparsers.add_parser(
        "compare",
        help="compare two transforms across cohorts, with a null control",
    )
    comparison.add_argument("pack")
    comparison.add_argument("--baseline", default="identity", help="baseline transform")
    comparison.add_argument("--candidate", required=True, help="candidate transform")
    comparison.add_argument("--metric", default="hue_drift")
    comparison.add_argument("--cohorts", default=DEFAULT_COHORTS)
    comparison.add_argument("--resamples", type=int, default=200)
    comparison.add_argument("--seed", type=int, default=0)
    comparison.add_argument(
        "--save",
        default=None,
        metavar="NAME",
        help="also write summary.json and comparisons.csv under the result root",
    )
    comparison.set_defaults(handler=_compare)

    return parser


def _workspace(args: argparse.Namespace) -> Workspace:
    return Workspace.from_env(args.workspace, args.output)


def _packs(args: argparse.Namespace) -> int:
    for name in _workspace(args).names(args.glob):
        print(name)
    return 0


def _describe(args: argparse.Namespace) -> int:
    description = sample.describe_pack(args.pack, workspace=_workspace(args))
    print(report.format_pack(description))
    return 0


def _cohorts(args: argparse.Namespace) -> int:
    table = sample.load_pack(args.pack, workspace=_workspace(args))
    print(f"{args.pack}: {len(table):,} rows")
    for name, selector in sorted(cohorts_module.BUILT_INS.items()):
        selected = selector(table)
        share = 100.0 * len(selected) / len(table) if len(table) else 0.0
        print(f"  {name:<16} {len(selected):>9,}  {share:5.1f}%")
    return 0


def _compare(args: argparse.Namespace) -> int:
    workspace = _workspace(args)
    table = sample.load_pack(args.pack, workspace=workspace)
    names = tuple(name.strip() for name in args.cohorts.split(",") if name.strip())
    selected = cohorts_module.build(table, names)
    if not selected:
        print("no cohort produced any samples", file=sys.stderr)
        return 1

    metrics.named(args.metric)  # resolve early so an unknown name fails before the work
    results = compare_cohorts(
        selected,
        baseline=transforms.named(args.baseline),
        candidate=transforms.named(args.candidate),
        metric=args.metric,
        tier=Tier.COMPARISON,
        resamples=args.resamples,
        seed=args.seed,
    )
    title = f"{args.pack}: {args.baseline} -> {args.candidate}"
    print(report.format_comparisons(results, title=title))

    if args.save:
        baseline_fn = transforms.named(args.baseline)
        candidate_fn = transforms.named(args.candidate)
        per_sample = {
            name: per_sample_metric(
                table, baseline=baseline_fn, candidate=candidate_fn, metric=args.metric
            )
            for name, table in selected.items()
        }
        page = report.comparison_report(
            report.ReportContext(
                title=f"{args.pack}: {args.baseline} → {args.candidate}",
                pack=args.pack,
                baseline=args.baseline,
                candidate=args.candidate,
                metric=args.metric,
                resamples=args.resamples,
                seed=args.seed,
                roots=workspace.describe(),
            ),
            results,
            selected,
            per_sample,
            baseline=baseline_fn,
            candidate=candidate_fn,
        )
        workspace.output(args.save, "report.html").write_text(page, encoding="utf-8")

        records = [result.as_record() for result in results]
        io.write_csv(workspace.output(args.save, "comparisons.csv"), records)
        io.write_json(
            workspace.output(args.save, "summary.json"),
            {
                "pack": args.pack,
                "baseline": args.baseline,
                "candidate": args.candidate,
                "metric": args.metric,
                "cohorts": list(selected),
                "tier": Tier.COMPARISON.value,
                "resamples": args.resamples,
                "seed": args.seed,
                "workspace": workspace.describe(),
                "results": records,
            },
        )
        directory = workspace.output(args.save, "summary.json", create=False).parent
        print(f"\nwrote {directory}")
        print(f"open  {directory / 'report.html'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.handler is None:
        parser.print_help(sys.stderr)
        return 2
    try:
        return int(args.handler(args))
    except FilmAnalysisError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


__all__ = ["build_parser", "main"]
