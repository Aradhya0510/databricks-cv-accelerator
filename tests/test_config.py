"""Config schema: loading, coercion, and cross-field validation."""

from __future__ import annotations

import copy

import pytest
import yaml
from pydantic import ValidationError

from src.config.schema import PipelineConfig, load_config

BASE = {
    "model": {
        "model_name": "facebook/detr-resnet-50",
        "task_type": "detection",
        "num_classes": 3,
    },
    "data": {
        "train_data_path": "/Volumes/c/s/v/train",
        "val_data_path": "/Volumes/c/s/v/val",
    },
}


def _write(tmp_path, **overrides):
    """Deep-merge *overrides* into BASE and write it as YAML."""
    cfg = copy.deepcopy(BASE)
    for section, values in overrides.items():
        cfg.setdefault(section, {}).update(values)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_loads_a_minimal_config(tmp_path):
    cfg = load_config(_write(tmp_path))
    assert cfg.model.model_name == "facebook/detr-resnet-50"
    assert cfg.model.num_classes == 3
    assert cfg.training.max_epochs > 0
    assert cfg.mlflow.experiment_name


@pytest.mark.parametrize(
    "contents, expected",
    [
        ("", "an empty file"),
        ("# only a comment\n", "an empty file"),
        ("- model\n- data\n", "a list"),
    ],
)
def test_a_config_that_is_not_a_mapping_says_so(tmp_path, contents, expected):
    """These used to surface as 'argument after ** must be a mapping', which
    named neither the file nor the reason."""
    p = tmp_path / "broken.yaml"
    p.write_text(contents)

    with pytest.raises(ValueError, match="not a YAML mapping") as exc:
        load_config(p)

    assert expected in str(exc.value)
    assert "broken.yaml" in str(exc.value)


def test_string_numbers_are_coerced(tmp_path):
    cfg = load_config(_write(tmp_path, model={"learning_rate": "1e-4", "epochs": "10"}))
    assert isinstance(cfg.model.learning_rate, float)
    assert cfg.model.learning_rate == pytest.approx(1e-4)
    assert cfg.model.epochs == 10


def test_image_size_list_collapses_to_scalar(tmp_path):
    cfg = load_config(_write(tmp_path, model={"image_size": [800, 800]}))
    assert cfg.model.image_size_scalar == 800


def test_deprecated_training_keys_are_stripped(tmp_path):
    cfg = load_config(_write(tmp_path, training={
        "max_epochs": 5, "use_ray": True, "resources_per_worker": 2, "learning_rate": 0.1,
    }))
    assert cfg.training.max_epochs == 5
    assert not hasattr(cfg.training, "use_ray")
    # learning_rate lives on model; a stale training copy must not shadow it.
    assert not hasattr(cfg.training, "learning_rate")


def test_unknown_task_type_is_rejected(tmp_path):
    with pytest.raises(ValidationError, match="task_type"):
        load_config(_write(tmp_path, model={"task_type": "pose_estimation"}))


def test_class_names_length_must_match_num_classes(tmp_path):
    with pytest.raises(ValidationError, match="num_classes"):
        load_config(_write(tmp_path, model={
            "task_type": "classification", "num_classes": 3, "class_names": ["cat", "dog"],
        }))


def test_matching_class_names_are_accepted(tmp_path):
    cfg = load_config(_write(tmp_path, model={
        "task_type": "classification", "num_classes": 2, "class_names": ["cat", "dog"],
    }))
    assert cfg.model.class_names == ["cat", "dog"]


def test_monitor_mode_is_validated(tmp_path):
    with pytest.raises(ValidationError, match="monitor_mode"):
        load_config(_write(tmp_path, training={"monitor_mode": "maximise"}))


def test_seed_is_available_and_defaulted(tmp_path):
    cfg = load_config(_write(tmp_path))
    assert isinstance(cfg.training.seed, int)


def test_round_trips_through_dict(tmp_path):
    cfg = load_config(_write(tmp_path))
    again = PipelineConfig(**cfg.to_dict())
    assert again.model.model_name == cfg.model.model_name
    assert again.training.max_epochs == cfg.training.max_epochs
