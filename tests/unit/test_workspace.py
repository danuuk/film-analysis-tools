"""Datasets resolve by name through a workspace, never by a caller-supplied path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from film_analysis_tools.capabilities.sample import describe_pack, load_pack
from film_analysis_tools.core import DataError, Workspace, WorkspaceError
from film_analysis_tools.core.workspace import ENV_VAR


def _write_pack(root: Path, name: str, scenes: int = 2, rows: int = 50) -> None:
    samples = root / name / "samples"
    samples.mkdir(parents=True)
    generator = np.random.default_rng(0)
    for index in range(scenes):
        np.savez(
            samples / f"c{index:04d}.npz",
            rgb_display_linear_l100=generator.uniform(0.05, 1.0, size=(rows, 3)),
            source_frame_index=np.arange(rows, dtype=np.int32),
        )
    (root / name / "sample_pack_manifest.json").write_text(
        '{"pack_id": "test_pack", "role": "unit_test"}', encoding="utf-8"
    )


def test_from_env_reads_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    assert Workspace.from_env().root == tmp_path


def test_missing_workspace_says_how_to_set_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    with pytest.raises(WorkspaceError, match=ENV_VAR):
        Workspace.from_env()


def test_names_that_escape_the_workspace_are_refused(tmp_path: Path) -> None:
    workspace = Workspace(root=tmp_path)
    for hostile in ("../elsewhere", "/etc", "a/../../b"):
        with pytest.raises(WorkspaceError):
            workspace.resolve(hostile)


def test_unknown_dataset_reports_the_root_it_looked_in(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceError, match="not found under"):
        Workspace(root=tmp_path).resolve("absent")


def test_load_pack_merges_scenes_and_tags_their_origin(tmp_path: Path) -> None:
    _write_pack(tmp_path, "pack_a", scenes=3, rows=40)
    table = load_pack("pack_a", workspace=Workspace(root=tmp_path))
    assert len(table) == 120
    assert sorted(set(table.column("scene_id").tolist())) == ["c0000", "c0001", "c0002"]


def test_load_pack_can_restrict_to_named_scenes(tmp_path: Path) -> None:
    _write_pack(tmp_path, "pack_b", scenes=3, rows=10)
    table = load_pack("pack_b", workspace=Workspace(root=tmp_path), scenes=("c0001",))
    assert len(table) == 10
    assert set(table.column("scene_id").tolist()) == {"c0001"}


def test_requesting_an_absent_scene_is_an_error(tmp_path: Path) -> None:
    _write_pack(tmp_path, "pack_c", scenes=1)
    with pytest.raises(DataError, match="no scenes"):
        load_pack("pack_c", workspace=Workspace(root=tmp_path), scenes=("missing",))


def test_a_directory_without_samples_is_not_a_pack(tmp_path: Path) -> None:
    (tmp_path / "not_a_pack").mkdir()
    with pytest.raises(DataError, match="no samples/ directory"):
        load_pack("not_a_pack", workspace=Workspace(root=tmp_path))


def test_describe_reports_shape_and_provenance(tmp_path: Path) -> None:
    _write_pack(tmp_path, "pack_d", scenes=2, rows=25)
    description = describe_pack("pack_d", workspace=Workspace(root=tmp_path))
    assert description["rows"] == 50
    assert description["scenes"] == 2
    assert description["pack_id"] == "test_pack"
