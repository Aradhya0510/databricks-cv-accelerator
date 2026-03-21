"""Standardised COCO evaluation via ``pycocotools.COCOeval``.

Provides thin wrappers that accept model predictions in a common format
and return official COCO metrics.  Two evaluation types are supported:

  - **bbox**: bounding-box mAP for object detection.
  - **segm**: mask AP for instance segmentation (predictions are
    binary masks encoded as RLE).

Semantic segmentation (mIoU) is *not* a COCO metric and is computed
separately by the segmentation task's eval function.

Usage
-----
::

    from src.utils.coco_eval import coco_evaluate_bbox, coco_evaluate_segm

    # Detection — predictions are [{"image_id", "category_id", "bbox", "score"}, ...]
    metrics = coco_evaluate_bbox(coco_gt, predictions)

    # Instance segmentation — predictions include RLE-encoded masks
    metrics = coco_evaluate_segm(coco_gt, predictions)
"""

from __future__ import annotations

import copy
import io
import sys
from contextlib import redirect_stdout
from typing import Any, Dict, List, Optional

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from .coco import COCODataSource


def _run_coco_eval(
    coco_gt: COCO,
    predictions: List[Dict[str, Any]],
    iou_type: str,
) -> Dict[str, float]:
    """Run COCOeval and return a metrics dict.

    Parameters
    ----------
    coco_gt : COCO
        Ground-truth COCO object (from ``COCODataSource.get_coco_api()``).
    predictions : list of dicts
        Each dict must have ``image_id``, ``category_id``, ``score``, and
        either ``bbox`` (for ``iou_type="bbox"``) or ``segmentation``
        (for ``iou_type="segm"``).
    iou_type : str
        ``"bbox"`` or ``"segm"``.
    """
    if not predictions:
        return {
            "AP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "AP_small": 0.0,
            "AP_medium": 0.0,
            "AP_large": 0.0,
            "AR_max1": 0.0,
            "AR_max10": 0.0,
            "AR_max100": 0.0,
            "AR_small": 0.0,
            "AR_medium": 0.0,
            "AR_large": 0.0,
        }

    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()

    # Capture the printed summary
    buf = io.StringIO()
    with redirect_stdout(buf):
        coco_eval.summarize()

    stats = coco_eval.stats
    return {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "AR_max1": float(stats[6]),
        "AR_max10": float(stats[7]),
        "AR_max100": float(stats[8]),
        "AR_small": float(stats[9]),
        "AR_medium": float(stats[10]),
        "AR_large": float(stats[11]),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def coco_evaluate_bbox(
    coco_gt: COCO,
    predictions: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute official COCO bounding-box metrics.

    Parameters
    ----------
    coco_gt : COCO
        Ground truth (``COCODataSource.get_coco_api()``).
    predictions : list of dicts
        Each: ``{"image_id": int, "category_id": int,
        "bbox": [x, y, w, h], "score": float}``.
        **Important**: ``bbox`` must be in COCO format ``[x, y, w, h]``
        (not ``[x1, y1, x2, y2]``).  Use :func:`xyxy_to_xywh` to convert.

    Returns
    -------
    dict
        Official COCO AP / AR metrics.
    """
    return _run_coco_eval(coco_gt, predictions, iou_type="bbox")


def coco_evaluate_segm(
    coco_gt: COCO,
    predictions: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute official COCO instance-segmentation metrics.

    Parameters
    ----------
    coco_gt : COCO
        Ground truth (``COCODataSource.get_coco_api()``).
    predictions : list of dicts
        Each: ``{"image_id": int, "category_id": int,
        "segmentation": RLE_dict, "score": float}``.
        Use :func:`mask_to_rle` to encode binary masks.

    Returns
    -------
    dict
        Official COCO AP / AR metrics (mask IoU).
    """
    return _run_coco_eval(coco_gt, predictions, iou_type="segm")


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Convert ``[x1, y1, x2, y2]`` boxes to COCO ``[x, y, w, h]`` format."""
    xywh = boxes.copy()
    xywh[..., 2] = boxes[..., 2] - boxes[..., 0]
    xywh[..., 3] = boxes[..., 3] - boxes[..., 1]
    return xywh


def mask_to_rle(binary_mask: np.ndarray) -> Dict[str, Any]:
    """Encode a binary mask as COCO RLE for ``COCOeval`` with ``iou_type="segm"``.

    Uses ``pycocotools.mask.encode`` under the hood.
    """
    from pycocotools import mask as mask_utils

    fortran_mask = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(fortran_mask)
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle
