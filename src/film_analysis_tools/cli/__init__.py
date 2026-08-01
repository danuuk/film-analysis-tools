"""Command-line entry points.

The only layer permitted to parse arguments or exit the process. Everything below is a
library that raises exceptions and returns values — which is what makes in-process
orchestration possible, and what the legacy design could not do.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from film_analysis_tools import __version__
from film_analysis_tools.capabilities import catalogue, report, sample
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

    cat = subparsers.add_parser("catalogue", help="query the camera sample catalogue")
    cat.add_argument(
        "category",
        nargs="*",
        help="validation categories; omit to list the taxonomy and its counts",
    )
    cat.add_argument("--all", action="store_true", help="require every category, not any")
    cat.add_argument("--shoot", default=None, help="restrict to one shoot")
    cat.add_argument("--limit", type=int, default=0)
    cat.add_argument("--paths", action="store_true", help="resolve and print file paths")
    cat.add_argument(
        "--verify",
        action="store_true",
        help="re-hash resolved files; slower, but proves the sample is the one recorded",
    )
    cat.add_argument(
        "--json",
        action="store_true",
        help="machine-readable output, for another tool or agent consuming the catalogue",
    )
    cat.set_defaults(handler=_catalogue)

    grain = subparsers.add_parser(
        "negative-grain-synthetic",
        help="render and measure the controlled FEE N0/N1/N2 synthetic comparison",
    )
    grain.add_argument("--n1-bundle", type=Path, required=True)
    grain.add_argument("--n2-bundle", type=Path, required=True)
    grain.add_argument("--report", type=Path, required=True, help="private report directory")
    grain.add_argument("--width", type=int, default=1920)
    grain.add_argument("--height", type=int, default=1080)
    grain.add_argument("--frames", type=int, default=96, help="four seconds at 24 fps")
    grain.add_argument("--seed", type=int, default=20260731)
    grain.add_argument(
        "--frame-workers",
        "--workers",
        dest="frame_workers",
        type=int,
        default=4,
        help="independent frame-render processes (legacy alias: --workers)",
    )
    grain.add_argument(
        "--variant-workers",
        type=int,
        default=1,
        help="variant-render threads inside each frame process",
    )
    grain.add_argument("--delta-limit", type=float, default=0.05)
    grain.add_argument("--no-video", action="store_true", help="metrics/report only")
    grain.set_defaults(handler=_negative_grain_synthetic)

    native_grain = subparsers.add_parser(
        "negative-grain-native-crops",
        help="render the bounded N2 0.75/1.0/1.25 native-pixel motion review",
    )
    native_grain.add_argument("--n1-bundle", type=Path, required=True)
    native_grain.add_argument("--n2-bundle", type=Path, required=True)
    native_grain.add_argument("--report", type=Path, required=True)
    native_grain.add_argument("--frames", type=int, default=96, help="four seconds at 24 fps")
    native_grain.add_argument("--seed", type=int, default=20260731)
    native_grain.add_argument("--frame-workers", type=int, default=4)
    native_grain.add_argument("--delta-limit", type=float, default=0.05)
    native_grain.set_defaults(handler=_negative_grain_native_crops)

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


def _resolved_path(clip: catalogue.CatalogueClip, *, verify: bool) -> tuple[str, str]:
    """``(path, error)`` — resolution failure is reported, never raised past the caller."""
    try:
        return str(clip.locate(verify=verify)), ""
    except FilmAnalysisError as error:
        return "", str(error)


def _catalogue(args: argparse.Namespace) -> int:
    cat = catalogue.bundled()
    counts = cat.counts()

    # Every response carries the catalogue identity. The skin labels will change when face
    # detection lands, so a consumer that does not record which version it queried cannot tell
    # two incomparable result sets apart.
    identity = {
        "catalogue_id": cat.catalogue_id,
        "generated": cat.generated,
        "clip_count": len(cat),
    }

    if not args.category:
        empty = [name for name, count in counts.items() if count == 0]
        if args.json:
            print(
                io.json_text(
                    {
                        **identity,
                        "camera": dict(cat.camera),
                        "decode": dict(cat.decode),
                        "categories": [
                            {**dict(entry), "count": counts.get(name, 0)}
                            for name, entry in cat.categories.items()
                        ],
                        "empty_categories": empty,
                        "uncategorised_count": len(cat.uncategorised()),
                    }
                )
            )
            return 0
        print(
            f"{cat.catalogue_id} ({cat.generated}): {len(cat)} clips, "
            f"{cat.camera.get('model', '?')}"
        )
        print(f"decode: {cat.decode.get('transfer')} / {cat.decode.get('primaries')}\n")
        for name, entry in cat.categories.items():
            flag = " (human-labelled)" if entry.get("human_labelled") else ""
            print(f"  {name:26s} {counts.get(name, 0):3d}{flag}")
            print(f"  {'':26s}     provokes: {entry.get('provokes', '')}")
        if empty:
            print(f"\nno material for: {', '.join(empty)} — a gap in the corpus, not a bug")
        print(f"uncategorised (ordinary material): {len(cat.uncategorised())}")
        return 0

    clips = cat.select(*args.category, require_all=args.all, shoot=args.shoot, limit=args.limit)

    if args.json:
        rows = []
        for clip in clips:
            row: dict[str, object] = {
                "clip_id": clip.clip_id,
                "shoot": clip.shoot,
                "sha256": clip.sha256,
                "byte_size": clip.byte_size,
                "duration_s": clip.duration_s,
                "probe_times_s": list(clip.probe_times_s),
                "categories": list(clip.categories),
                "notes": clip.notes,
                "measured": dict(clip.measured),
            }
            if args.paths:
                path, error = _resolved_path(clip, verify=args.verify)
                row["path"] = path
                if error:
                    row["unresolved"] = error
            rows.append(row)
        print(
            io.json_text(
                {
                    **identity,
                    "query": {
                        "categories": list(args.category),
                        "require_all": bool(args.all),
                        "shoot": args.shoot,
                        "limit": args.limit,
                        "verified": bool(args.paths and args.verify),
                    },
                    "decode": dict(cat.decode),
                    "count": len(rows),
                    "clips": rows,
                }
            )
        )
        return 0

    joiner = " AND " if args.all else " OR "
    print(f"{joiner.join(args.category)} -> {len(clips)} clips")
    for clip in clips:
        line = f"  {clip.clip_id}  {clip.shoot:16s} {clip.duration_s:5.1f}s  {clip.notes}"
        if args.paths:
            path, error = _resolved_path(clip, verify=args.verify)
            line += f"\n      {path}" if path else f"\n      UNRESOLVED: {error}"
        print(line)
    return 0


def _negative_grain_synthetic(args: argparse.Namespace) -> int:
    # Imported only for this command: the FEE runtime is an optional forward-model dependency,
    # and ordinary FAT catalogue/measurement commands must remain usable without it installed.
    from film_analysis_tools.studies.negative_grain_synthetic import (
        SyntheticGrainRunConfig,
        run,
    )

    run(
        SyntheticGrainRunConfig(
            n1_bundle=args.n1_bundle,
            n2_bundle=args.n2_bundle,
            output_dir=args.report,
            width=args.width,
            height=args.height,
            frame_count=args.frames,
            seed=args.seed,
            frame_workers=args.frame_workers,
            variant_workers=args.variant_workers,
            delta_display_limit=args.delta_limit,
            make_videos=not args.no_video,
        ),
        progress=lambda done, total: print(f"rendered {done}/{total}", flush=True),
    )
    print(f"wrote {args.report}")
    print(f"open  {args.report / 'index.html'}")
    return 0


def _negative_grain_native_crops(args: argparse.Namespace) -> int:
    from film_analysis_tools.studies.negative_grain_native_crops import (
        NativeCropRunConfig,
        run,
    )

    run(
        NativeCropRunConfig(
            n1_bundle=args.n1_bundle,
            n2_bundle=args.n2_bundle,
            output_dir=args.report,
            frame_count=args.frames,
            seed=args.seed,
            frame_workers=args.frame_workers,
            delta_display_limit=args.delta_limit,
        ),
        progress=lambda done, total: print(f"rendered {done}/{total}", flush=True),
    )
    print(f"wrote {args.report}")
    print(f"open  {args.report / 'index.html'}")
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
