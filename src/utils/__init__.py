"""
Utility Modules

This module provides utility functions for the computer vision pipeline.
"""

from .coco import COCOAnnotationType, COCODataSource, detect_annotation_type
from .coco_eval import (
    coco_evaluate_bbox,
    coco_evaluate_segm,
    mask_to_rle,
    xyxy_to_xywh,
)
from .environment import (
    get_gpu_count,
    is_databricks,
    is_databricks_job,
    is_databricks_notebook,
    setup_nccl_env,
    stage_data_to_local,
)
