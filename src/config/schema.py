"""Pydantic v2 configuration models for the CV pipeline.

Accepts the same YAML format as the existing ``src/config.py`` dataclasses,
but adds validation, type coercion, and a convenient ``from_yaml`` loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Section configs
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    model_config = {"extra": "allow", "protected_namespaces": ()}

    model_name: str
    num_classes: int = 80
    pretrained: bool = True

    # Training hyper-parameters (kept here to match existing YAML layout)
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    scheduler_params: Optional[Dict[str, Any]] = None
    epochs: int = 100

    # Task
    task_type: str = "detection"
    class_names: Optional[List[str]] = None
    segmentation_type: Optional[str] = None

    # Detection-specific
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.5
    max_detections: int = 100
    image_size: Union[int, List[int]] = 640

    # Classification-specific
    dropout: float = 0.2
    mixup_alpha: float = 0.2

    # Segmentation-specific
    aux_loss_weight: float = 0.4
    mask_threshold: float = 0.5

    @field_validator("image_size", mode="before")
    @classmethod
    def _coerce_image_size(cls, v: Any) -> Union[int, List[int]]:
        if isinstance(v, list):
            return v[0] if len(v) == 1 else v
        return int(v)

    @field_validator("num_classes", "epochs", "max_detections", mode="before")
    @classmethod
    def _coerce_int(cls, v: Any) -> int:
        return int(v)

    @field_validator("learning_rate", "weight_decay", "confidence_threshold", "iou_threshold", mode="before")
    @classmethod
    def _coerce_float(cls, v: Any) -> float:
        return float(v)

    @field_validator("pretrained", mode="before")
    @classmethod
    def _coerce_bool(cls, v: Any) -> bool:
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes")
        return bool(v)

    @field_validator("task_type")
    @classmethod
    def _validate_task_type(cls, v: str) -> str:
        allowed = {"detection", "classification", "segmentation"}
        if v not in allowed:
            raise ValueError(
                f"task_type must be one of {sorted(allowed)}, got '{v}'"
            )
        return v

    @model_validator(mode="after")
    def _check_class_names_length(self) -> "ModelConfig":
        """``class_names`` must line up with ``num_classes``.

        These were previously independent, so a mismatch surfaced either as a
        silently oversized model head or as an index error deep inside the loss.
        """
        if self.class_names is not None and len(self.class_names) != self.num_classes:
            raise ValueError(
                f"class_names has {len(self.class_names)} entries but "
                f"num_classes is {self.num_classes} — they must match. "
                f"class_names={self.class_names}"
            )
        return self

    @property
    def image_size_scalar(self) -> int:
        """Return image size as a single int (takes first element if list)."""
        if isinstance(self.image_size, list):
            return self.image_size[0]
        return self.image_size


class DataConfig(BaseModel):
    model_config = {"extra": "allow", "protected_namespaces": ()}

    train_data_path: str
    val_data_path: str
    train_annotation_file: Optional[str] = None
    val_annotation_file: Optional[str] = None
    test_data_path: Optional[str] = None
    test_annotation_file: Optional[str] = None

    batch_size: int = 16
    num_workers: int = 4
    model_name: str = ""

    image_size: Union[int, List[int], Tuple[int, int]] = (512, 512)
    normalize_mean: Optional[List[float]] = None
    normalize_std: Optional[List[float]] = None

    augment: Union[bool, Dict[str, Any]] = True
    augmentations: Optional[Dict[str, Any]] = None

    # Copy /Volumes inputs to local disk once before training.  Image datasets
    # are many small files re-read every epoch, the access pattern UC Volumes
    # serve worst; see src/utils/environment.py.
    stage_to_local: bool = True
    local_cache_dir: str = "/tmp/cv_data"

    @field_validator("batch_size", "num_workers", mode="before")
    @classmethod
    def _coerce_int(cls, v: Any) -> int:
        return int(v)


class TrainingConfig(BaseModel):
    max_epochs: int = 100
    early_stopping_patience: int = 10
    monitor_metric: str = "val_loss"
    monitor_mode: str = "min"
    checkpoint_dir: str = "/tmp/checkpoints"
    volume_checkpoint_dir: Optional[str] = None
    save_top_k: int = 3
    log_every_n_steps: int = 50
    use_gpu: bool = True

    # Reproducibility: seeds torch, numpy and the dataloaders, and is used for
    # any dataset splitting.  Logged as an MLflow param so a run can be repeated.
    seed: int = 42

    # Mixed precision.  ``auto`` picks bf16 when the GPU supports it, else fp16,
    # else fp32 — V100 and T4 do not support bf16 and used to fail outright.
    precision: str = "auto"

    # ``"latest"`` resumes from the newest complete checkpoint in
    # ``volume_checkpoint_dir`` and starts fresh when there is none, so the
    # first attempt of an AI Runtime task trains normally and its retries pick
    # up where it stopped.  Any other value is a checkpoint path.  Use a fresh
    # ``volume_checkpoint_dir`` per training run, or "latest" will resume
    # someone else's.
    resume_from_checkpoint: Optional[str] = None

    model_config = {"extra": "allow"}

    @field_validator("monitor_mode")
    @classmethod
    def _validate_monitor_mode(cls, v: str) -> str:
        if v not in {"min", "max"}:
            raise ValueError(f"monitor_mode must be 'min' or 'max', got '{v}'")
        return v

    @field_validator("precision")
    @classmethod
    def _validate_precision(cls, v: str) -> str:
        allowed = {"auto", "bf16", "fp16", "fp32"}
        if v not in allowed:
            raise ValueError(f"precision must be one of {sorted(allowed)}, got '{v}'")
        return v

    @field_validator("max_epochs", "early_stopping_patience", "save_top_k", "log_every_n_steps", mode="before")
    @classmethod
    def _coerce_int(cls, v: Any) -> int:
        if v is None:
            return 0
        return int(v)

    @field_validator("use_gpu", mode="before")
    @classmethod
    def _coerce_bool(cls, v: Any) -> bool:
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes")
        return bool(v)

    @model_validator(mode="after")
    def _check_resume_source(self) -> "TrainingConfig":
        if self.resume_from_checkpoint == "latest" and not self.volume_checkpoint_dir:
            raise ValueError(
                "resume_from_checkpoint: latest needs volume_checkpoint_dir — "
                "local checkpoints do not survive the AI Runtime node, so there "
                "is nothing to resume from."
            )
        return self


class MLflowConfig(BaseModel):
    experiment_name: str = "cv_training"
    run_name: str = "default_run"
    log_model: Union[bool, str] = True
    tags: Dict[str, str] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class OutputConfig(BaseModel):
    results_dir: str = "/tmp/results"
    save_predictions: bool = True
    visualization: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Serving & monitoring configs
# ---------------------------------------------------------------------------

class ServingConfig(BaseModel):
    model_config = {"extra": "allow"}
    registered_model_name: Optional[str] = None  # catalog.schema.model_name
    endpoint_name: Optional[str] = None
    workload_size: str = "Small"
    scale_to_zero: bool = True


class MonitoringConfig(BaseModel):
    model_config = {"extra": "allow"}
    drift_threshold: float = 0.1
    error_rate_threshold: float = 0.05
    latency_p95_threshold_ms: float = 500


# ---------------------------------------------------------------------------
# Top-level pipeline config
# ---------------------------------------------------------------------------

class PipelineConfig(BaseModel):
    model_config = {"extra": "allow", "protected_namespaces": ()}

    model: ModelConfig
    data: DataConfig
    training: TrainingConfig = TrainingConfig()
    mlflow: MLflowConfig = MLflowConfig()
    output: OutputConfig = OutputConfig()
    serving: ServingConfig = ServingConfig()
    monitoring: MonitoringConfig = MonitoringConfig()

    @model_validator(mode="before")
    @classmethod
    def _clean_training_section(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Strip deprecated keys from the training section."""
        training = values.get("training")
        if isinstance(training, dict):
            for removed_key in [
                "distributed", "use_ray", "num_workers", "resources_per_worker",
                "master_port", "preferred_strategy", "preferred_devices",
                "learning_rate", "weight_decay", "scheduler", "scheduler_params",
            ]:
                training.pop(removed_key, None)
        return values

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "PipelineConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        # An empty or truncated file parses to None, and a bare list parses to
        # a list; both reach ``cls(**raw)`` as "argument after ** must be a
        # mapping", which says nothing about which file or why.
        if not isinstance(raw, dict):
            found = "an empty file" if raw is None else f"a {type(raw).__name__}"
            raise ValueError(
                f"{path} is not a YAML mapping — parsed as {found}. A config "
                f"must be a mapping with model/data/training sections; an empty "
                f"file usually means it was truncated in transit."
            )

        return cls(**raw)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Convenience loader (drop-in for ``from src.config import load_config``)
# ---------------------------------------------------------------------------

def load_config(config_path: Union[str, Path]) -> PipelineConfig:
    """Load a YAML config and return a validated ``PipelineConfig``."""
    return PipelineConfig.from_yaml(config_path)
