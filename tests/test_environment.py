"""Path staging and rank helpers.

The staging bug pinned here (``lstrip`` used as a prefix strip) silently
rewrote every /Volumes path and could collide two distinct sources onto one
staging directory, so the assertions are deliberately literal.
"""

from __future__ import annotations

import pytest

from src.utils.environment import (
    is_rank_zero,
    stage_data_to_local,
    volumes_staging_path,
)


@pytest.mark.parametrize(
    "volumes_path, expected",
    [
        ("/Volumes/main/cv/images/", "/tmp/staged/main/cv/images"),
        ("/Volumes/users/east/imgs", "/tmp/staged/users/east/imgs"),
        ("/Volumes/main/schema/vol/data/", "/tmp/staged/main/schema/vol/data"),
        # Segments whose leading characters appear in the literal "/Volumes/"
        # are exactly what the old lstrip() implementation destroyed.
        ("/Volumes/mount/x", "/tmp/staged/mount/x"),
        ("/Volumes/sales/y", "/tmp/staged/sales/y"),
        ("/Volumes/lume/z", "/tmp/staged/lume/z"),
    ],
)
def test_staging_path_preserves_every_segment(volumes_path, expected):
    assert volumes_staging_path(volumes_path, "/tmp/staged") == expected


def test_distinct_volume_paths_do_not_collide():
    a = volumes_staging_path("/Volumes/mount/data", "/tmp/staged")
    b = volumes_staging_path("/Volumes/nt/data", "/tmp/staged")
    assert a != b


def test_non_volumes_paths_pass_through():
    assert stage_data_to_local("/dbfs/foo") == "/dbfs/foo"
    assert stage_data_to_local("relative/path") == "relative/path"


def test_stage_copies_a_directory_tree(tmp_path, monkeypatch):
    source = tmp_path / "Volumes" / "main" / "cv"
    source.mkdir(parents=True)
    (source / "one.txt").write_text("hello")

    # Rebase the /Volumes root onto the tmp tree so the copy runs for real.
    monkeypatch.setattr(
        "src.utils.environment._VOLUMES_ROOT", str(tmp_path / "Volumes")
    )
    local_root = str(tmp_path / "staged")

    out = stage_data_to_local("/Volumes/main/cv", local_root)

    assert out == str(tmp_path / "staged" / "main" / "cv")
    assert (tmp_path / "staged" / "main" / "cv" / "one.txt").read_text() == "hello"


def test_stage_is_idempotent(tmp_path, monkeypatch):
    source = tmp_path / "Volumes" / "main" / "cv"
    source.mkdir(parents=True)
    (source / "one.txt").write_text("hello")
    monkeypatch.setattr(
        "src.utils.environment._VOLUMES_ROOT", str(tmp_path / "Volumes")
    )
    local_root = str(tmp_path / "staged")

    first = stage_data_to_local("/Volumes/main/cv", local_root)
    second = stage_data_to_local("/Volumes/main/cv", local_root)
    assert first == second


def test_rank_zero_uses_global_rank(monkeypatch):
    monkeypatch.delenv("RANK", raising=False)
    assert is_rank_zero() is True

    # Second node, first local process: LOCAL_RANK is 0 but RANK is not, so
    # this must NOT be treated as the writer of runs/checkpoints.
    monkeypatch.setenv("RANK", "4")
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert is_rank_zero() is False

    monkeypatch.setenv("RANK", "0")
    assert is_rank_zero() is True
