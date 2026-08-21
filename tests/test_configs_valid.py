"""Every shipped config must load, and must not contain inert settings.

The configs are the user-facing surface of the framework: a key that
validates but is never read is worse than no key at all, because it looks
like a working control.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config.schema import load_config
from src.tasks.augmentation import unknown_augmentation_keys

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
CONFIGS = sorted(CONFIG_DIR.glob("*.yaml"))


def test_there_are_configs_to_check():
    assert CONFIGS, f"no configs found in {CONFIG_DIR}"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_loads_and_validates(path):
    cfg = load_config(path)
    assert cfg.model.model_name
    assert cfg.model.task_type in {"detection", "classification", "segmentation"}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_checkpoints_are_not_written_directly_to_a_volume(path):
    """UC Volumes are a FUSE mount with poor rename/random-write behaviour.

    Training writes to local disk; volume_checkpoint_dir mirrors afterwards.
    """
    cfg = load_config(path)
    assert not cfg.training.checkpoint_dir.startswith("/Volumes/"), (
        f"{path.name} writes checkpoints straight to a Volume. Point "
        f"checkpoint_dir at local disk and let volume_checkpoint_dir persist them."
    )


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_augmentation_keys_are_all_implemented(path):
    cfg = load_config(path)
    unknown = unknown_augmentation_keys(cfg.data.augmentations)
    assert not unknown, (
        f"{path.name} configures augmentations {sorted(unknown)} that nothing "
        f"implements — they would be silently ignored."
    )


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_monitor_metric_matches_monitor_mode(path):
    """A loss tracked with mode 'max' would select the worst checkpoint."""
    cfg = load_config(path)
    metric = cfg.training.monitor_metric
    if "loss" in metric:
        assert cfg.training.monitor_mode == "min", (
            f"{path.name} monitors '{metric}' with mode "
            f"'{cfg.training.monitor_mode}'"
        )
    if any(k in metric for k in ("map", "accuracy", "iou", "f1")):
        assert cfg.training.monitor_mode == "max", (
            f"{path.name} monitors '{metric}' with mode "
            f"'{cfg.training.monitor_mode}'"
        )
