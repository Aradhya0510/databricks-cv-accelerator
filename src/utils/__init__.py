"""Utility modules for the computer vision pipeline.

Kept deliberately import-light: pulling ``pycocotools`` in at package import
time meant any ``import src.utils`` paid for it, including from the serving
wrappers that never touch COCO.  Import ``src.utils.coco`` directly where it
is needed.
"""

from .environment import (
    ensure_runtime_requirements,
    get_gpu_count,
    is_databricks,
    is_databricks_job,
    is_databricks_notebook,
    is_rank_zero,
    resolve_precision,
    setup_nccl_env,
    stage_data_to_local,
    volumes_staging_path,
)
from .labels import apply_label_names, label_names_from_config

__all__ = [
    "ensure_runtime_requirements",
    "get_gpu_count",
    "is_databricks",
    "is_databricks_job",
    "is_databricks_notebook",
    "is_rank_zero",
    "resolve_precision",
    "setup_nccl_env",
    "stage_data_to_local",
    "volumes_staging_path",
    "apply_label_names",
    "label_names_from_config",
]
