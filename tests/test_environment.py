"""Path staging, process topology and rank helpers.

The staging bug pinned here (``lstrip`` used as a prefix strip) silently
rewrote every /Volumes path and could collide two distinct sources onto one
staging directory, so the assertions are deliberately literal.
"""

from __future__ import annotations

import pytest

from src.utils.environment import (
    gpus_per_node,
    is_distributed_worker,
    is_rank_zero,
    num_nodes,
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


@pytest.fixture
def volume(tmp_path, monkeypatch):
    """A fake /Volumes root with a small nested image tree."""
    root = tmp_path / "Volumes"
    source = root / "main" / "cv"
    (source / "train" / "a").mkdir(parents=True)
    (source / "train" / "a" / "one.jpg").write_text("1")
    (source / "train" / "two.jpg").write_text("2")
    (source / "ann.json").write_text("{}")
    # Rebase the /Volumes root onto the tmp tree so the copy runs for real.
    monkeypatch.setattr("src.utils.environment._VOLUMES_ROOT", str(root))
    return tmp_path


def test_stage_copies_a_nested_directory_tree(volume):
    out = stage_data_to_local("/Volumes/main/cv/train", str(volume / "staged"))

    assert out == str(volume / "staged" / "main" / "cv" / "train")
    assert (volume / "staged/main/cv/train/a/one.jpg").read_text() == "1"
    assert (volume / "staged/main/cv/train/two.jpg").read_text() == "2"


def test_stage_is_idempotent(volume):
    local_root = str(volume / "staged")
    first = stage_data_to_local("/Volumes/main/cv/train", local_root)
    # A second call must not recopy: prove it by removing the source.
    (volume / "Volumes/main/cv/train/two.jpg").unlink()
    second = stage_data_to_local("/Volumes/main/cv/train", local_root)

    assert first == second
    assert (volume / "staged/main/cv/train/two.jpg").exists()


def test_a_partial_copy_is_redone(volume):
    """A run killed mid-copy must not leave a half-staged dataset behind."""
    partial = volume / "staged/main/cv/train"
    partial.mkdir(parents=True)
    (partial / "leftover.jpg").write_text("stale")

    stage_data_to_local("/Volumes/main/cv/train", str(volume / "staged"))

    assert not (partial / "leftover.jpg").exists()
    assert (partial / "two.jpg").read_text() == "2"


def test_a_single_file_is_staged(volume):
    out = stage_data_to_local("/Volumes/main/cv/ann.json", str(volume / "staged"))

    assert out == str(volume / "staged/main/cv/ann.json")
    assert (volume / "staged/main/cv/ann.json").read_text() == "{}"
    assert not (volume / "staged/main/cv/ann.json.partial").exists()


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


def test_worker_detection_ignores_the_node_level_world_size(monkeypatch):
    """AI Runtime sets WORLD_SIZE on every node before any launcher runs."""
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "16")
    assert is_distributed_worker() is False

    monkeypatch.setenv("LOCAL_RANK", "3")
    assert is_distributed_worker() is True


def test_topology_reads_ai_runtime_variables(monkeypatch):
    monkeypatch.setenv("NUM_NODES", "2")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    assert num_nodes() == 2
    assert gpus_per_node() == 8

    monkeypatch.delenv("NUM_NODES")
    assert num_nodes() == 1
