"""Formalization rules 1-6 (``MIGRATION_PLAN.md`` section 10).

These are enforced mechanically rather than by review discipline, and they are enforced from
the first commit — while there is no code to violate them — so the legacy structural debt
cannot be reintroduced. They serve manageability, not separation from the plugin.

Legacy violation counts these rules would have caught, for scale:
``SystemExit`` in library code 134/221 modules, ``argparse`` in analyzer modules 132/221,
hardcoded artifact paths 90, ``parents[N]`` walking 39.

Each rule is a pure ``find_*`` function over parsed modules, so the same logic runs against
the real tree *and* against a synthetic violating module. That second run is the null control
for these tests: a rule that has quietly decayed into a no-op fails
``test_every_rule_still_fires_on_a_known_violation`` rather than passing in silence.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Sequence
from pathlib import Path

import film_analysis_tools

SRC = Path(film_analysis_tools.__file__).resolve().parent
PACKAGE = "film_analysis_tools"

Parsed = tuple[Path, str, ast.Module]

# Rule 5: strictly downward. A module may import only from its own layer or one below it.
# Top-level modules sit below everything and may import nothing internal.
LAYER_ORDER: tuple[str, ...] = (
    "",  # top-level package modules
    "core",
    "evidence",
    "capabilities",
    "forward",
    "studies",
    "runner",
    "cli",
)

# Rule 3: paths that tie a module to one repository layout, or to one machine.
FORBIDDEN_PATH_PREFIXES: tuple[str, ...] = (
    "findings",
    "calibration",
    "film_samples",
    "scene_catalogs",
    "reference_candidate_db",
    "presets",
    "datasheets",
    "native",
    "shaders",
    "mediachar",
)
FORBIDDEN_ABSOLUTE_PREFIXES: tuple[str, ...] = ("/Users/", "/home/", "C:\\", "/private/")

# Rule 6: emulation models are reachable only through the adapter layer.
EMULATION_PACKAGES: tuple[str, ...] = ("film_emulation_engine", "mediachar")

PARENTS_WALK = re.compile(r"\.parents\[")


# --------------------------------------------------------------------------- helpers


def _layer_of(path: Path) -> str:
    relative = path.relative_to(SRC)
    return relative.parts[0] if len(relative.parts) > 1 else ""


def _rank(layer: str) -> int:
    return LAYER_ORDER.index(layer)


def _imported_modules(tree: ast.AST) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.append(node.module)
    return modules


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """Ids of ``Constant`` nodes that are docstrings, so rule 3 does not scan prose."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                ids.add(id(body[0].value))
    return ids


def _parsed() -> list[Parsed]:
    parsed: list[Parsed] = []
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        parsed.append((path, text, ast.parse(text, filename=str(path))))
    return parsed


# ------------------------------------------------------------------------ rule logic


def find_rule_1(modules: Sequence[Parsed]) -> list[str]:
    """Argument parsing outside ``cli/`` — why the legacy runner had to shell out."""
    return [
        str(path.relative_to(SRC))
        for path, _text, tree in modules
        if _layer_of(path) != "cli"
        and any(module.split(".")[0] == "argparse" for module in _imported_modules(tree))
    ]


def find_rule_2(modules: Sequence[Parsed]) -> list[str]:
    """Process exit outside ``cli/`` — library code must not terminate its host."""
    offenders: list[str] = []
    for path, _text, tree in modules:
        if _layer_of(path) == "cli":
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise):
                exc = node.exc
                target = exc.func if isinstance(exc, ast.Call) else exc
                if isinstance(target, ast.Name) and target.id == "SystemExit":
                    offenders.append(f"{path.relative_to(SRC)}: raise SystemExit")
            elif isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "exit"
                    and isinstance(func.value, ast.Name)
                    and func.value.id in {"sys", "os"}
                ):
                    offenders.append(f"{path.relative_to(SRC)}: {func.value.id}.exit()")
    return offenders


def find_rule_3(modules: Sequence[Parsed]) -> list[str]:
    """Artifact or machine paths — 90 legacy modules only ran in one repository layout."""
    offenders: list[str] = []
    for path, _text, tree in modules:
        skip = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in skip:
                continue
            value = node.value
            if value.startswith(FORBIDDEN_ABSOLUTE_PREFIXES):
                offenders.append(f"{path.relative_to(SRC)}: absolute path {value!r}")
                continue
            head = value.split("/")[0]
            if head in FORBIDDEN_PATH_PREFIXES and (value == head or value.startswith(head + "/")):
                offenders.append(f"{path.relative_to(SRC)}: artifact path {value!r}")
    return offenders


def find_rule_4(modules: Sequence[Parsed]) -> list[str]:
    """Repository-root walking — behaviour must not depend on install location."""
    return [
        str(path.relative_to(SRC)) for path, text, _tree in modules if PARENTS_WALK.search(text)
    ]


def find_rule_5_upward(modules: Sequence[Parsed]) -> list[str]:
    offenders: list[str] = []
    for path, _text, tree in modules:
        layer = _layer_of(path)
        for module in _imported_modules(tree):
            parts = module.split(".")
            if parts[0] != PACKAGE:
                continue
            target = parts[1] if len(parts) > 1 and parts[1] in LAYER_ORDER else ""
            if _rank(target) > _rank(layer):
                offenders.append(
                    f"{path.relative_to(SRC)} ({layer or 'top-level'}) imports {module} ({target})"
                )
    return offenders


def find_rule_5_cli_is_a_leaf(modules: Sequence[Parsed]) -> list[str]:
    offenders: list[str] = []
    for path, _text, tree in modules:
        if _layer_of(path) == "cli":
            continue
        offenders.extend(
            f"{path.relative_to(SRC)} imports {module}"
            for module in _imported_modules(tree)
            if module == f"{PACKAGE}.cli" or module.startswith(f"{PACKAGE}.cli.")
        )
    return offenders


def find_rule_6(modules: Sequence[Parsed]) -> list[str]:
    """One seam to maintain against engine and plugin drift, not twenty."""
    offenders: list[str] = []
    for path, _text, tree in modules:
        if _layer_of(path) == "forward":
            continue
        offenders.extend(
            f"{path.relative_to(SRC)} imports {module}"
            for module in _imported_modules(tree)
            if module.split(".")[0] in EMULATION_PACKAGES
        )
    return offenders


# ----------------------------------------------------------------- rules on real code


def test_rule_1_argument_parsing_only_in_cli() -> None:
    assert not find_rule_1(_parsed())


def test_rule_2_process_exit_only_in_cli() -> None:
    assert not find_rule_2(_parsed())


def test_rule_3_no_artifact_or_machine_paths() -> None:
    assert not find_rule_3(_parsed())


def test_rule_4_no_repository_root_walking() -> None:
    assert not find_rule_4(_parsed())


def test_rule_5_layers_are_strictly_downward() -> None:
    assert not find_rule_5_upward(_parsed())


def test_rule_5_nothing_imports_cli() -> None:
    assert not find_rule_5_cli_is_a_leaf(_parsed())


def test_rule_6_emulation_models_only_in_forward() -> None:
    assert not find_rule_6(_parsed())


def test_every_layer_in_the_declared_order_exists() -> None:
    """Guards against a layer being added on disk without being ranked here."""
    on_disk = {
        entry.name
        for entry in SRC.iterdir()
        if entry.is_dir() and (entry / "__init__.py").exists() and entry.name != "__pycache__"
    }
    undeclared = on_disk - set(LAYER_ORDER)
    assert not undeclared, f"packages present but absent from LAYER_ORDER: {sorted(undeclared)}"


# ----------------------------------------------------------- null control for the rules

VIOLATING_SOURCE = '''
"""A module that breaks every rule. Never imported; parsed as text only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mediachar
from film_analysis_tools.runner import something

ROOT = Path(__file__).resolve().parents[3]
OUT = "findings/some_experiment"
HOME = "/Users/someone/data"


def go() -> None:
    argparse.ArgumentParser()
    if not OUT:
        raise SystemExit("bad")
    sys.exit(1)
'''


def test_every_rule_still_fires_on_a_known_violation() -> None:
    """The null control. Without it, a rule that decayed into a no-op would pass silently."""
    fake_path = SRC / "capabilities" / "_synthetic_violation.py"
    modules: list[Parsed] = [(fake_path, VIOLATING_SOURCE, ast.parse(VIOLATING_SOURCE))]

    silent = [
        name
        for name, find in (
            ("rule 1 argparse", find_rule_1),
            ("rule 2 process exit", find_rule_2),
            ("rule 3 hardcoded paths", find_rule_3),
            ("rule 4 parents[N]", find_rule_4),
            ("rule 5 upward import", find_rule_5_upward),
            ("rule 6 emulation import", find_rule_6),
        )
        if not find(modules)
    ]
    assert not silent, f"rules no longer detect their own violation: {silent}"


def test_cli_leaf_rule_fires_on_a_known_violation() -> None:
    source = "from film_analysis_tools.cli import main\n"
    fake_path = SRC / "core" / "_synthetic_violation.py"
    assert find_rule_5_cli_is_a_leaf([(fake_path, source, ast.parse(source))])
