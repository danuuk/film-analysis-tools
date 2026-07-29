"""Result paths are resolved, never hardcoded — and never written into the corpus."""

from __future__ import annotations

from pathlib import Path

import pytest

from film_analysis_tools.core import Workspace, WorkspaceError
from film_analysis_tools.core.workspace import ENV_VAR, OUTPUT_ENV_VAR


def test_read_and_write_roots_are_separate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sources are large, external and read-only; results belong elsewhere."""
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "corpus"))
    monkeypatch.setenv(OUTPUT_ENV_VAR, str(tmp_path / "results"))
    workspace = Workspace.from_env()
    assert workspace.root != workspace.output_root
    assert workspace.output("study", "summary.json").is_relative_to(tmp_path / "results")


def test_output_root_falls_back_to_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    monkeypatch.delenv(OUTPUT_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    assert Workspace.from_env().output_root == tmp_path / "results"


def test_output_creates_parent_directories(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path, output_root=tmp_path / "out")
    path = workspace.output("a", "b", "c.json")
    assert path.parent.is_dir()
    assert not path.exists()


def test_output_can_decline_to_create(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path, output_root=tmp_path / "out")
    path = workspace.output("a", "b.json", create=False)
    assert not path.parent.exists()


def test_output_names_that_escape_the_write_root_are_refused(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path, output_root=tmp_path / "out")
    for hostile in ("../elsewhere", "/etc/passwd", "a/../../b"):
        with pytest.raises(WorkspaceError):
            workspace.output(hostile)


def test_output_requires_at_least_one_component(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceError):
        Workspace(root=tmp_path, output_root=tmp_path).output()


def test_describe_records_both_roots_for_provenance(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path / "in", output_root=tmp_path / "out")
    described = workspace.describe()
    assert described["read_root"].endswith("in")
    assert described["write_root"].endswith("out")


def test_explicit_arguments_beat_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "env_in"))
    monkeypatch.setenv(OUTPUT_ENV_VAR, str(tmp_path / "env_out"))
    workspace = Workspace.from_env(tmp_path / "arg_in", tmp_path / "arg_out")
    assert workspace.root == tmp_path / "arg_in"
    assert workspace.output_root == tmp_path / "arg_out"


def test_writing_without_a_configured_root_fails_loudly(tmp_path: Path) -> None:
    """Absent is fine for read-only work; guessing a write location is not."""
    with pytest.raises(WorkspaceError, match="no result root configured"):
        Workspace(root=tmp_path).output("study", "summary.json")
