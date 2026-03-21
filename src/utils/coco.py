"""Shared COCO data source — unified pycocotools interface for all CV tasks.

``COCODataSource`` wraps ``pycocotools.COCO`` and provides task-specific
accessors so that detection and segmentation share the same annotation
parsing, category mapping, and mask-decoding logic.  This is the single
place where pycocotools is used for data loading; individual dataset
classes delegate to this source.

Design principle: **COCO as the standard annotation framework**.
``instances_*.json`` supports both detection (bounding boxes) and
instance/semantic segmentation (per-object polygon masks via
``annToMask``).  A single annotation file can drive multiple tasks.
``panoptic_*.json`` is the standard for panoptic segmentation.
ADE20K-style image+mask pairs remain as a zero-annotation fallback.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from pycocotools.coco import COCO


class COCOAnnotationType(Enum):
    """Annotation file type, determined by inspecting the JSON structure."""
    INSTANCES = "instances"
    PANOPTIC = "panoptic"


def detect_annotation_type(annotation_file: str) -> COCOAnnotationType:
    """Determine whether a COCO JSON is instances or panoptic format.

    - **Instances**: ``annotations[0]`` has ``"bbox"`` or ``"segmentation"``
    - **Panoptic**: ``annotations[0]`` has ``"segments_info"``
    """
    import json

    with open(annotation_file, "r") as f:
        data = json.load(f)

    annotations = data.get("annotations", [])
    if not annotations:
        raise ValueError(
            f"No annotations found in {annotation_file}. "
            "Cannot determine annotation type."
        )

    sample = annotations[0]
    if "segments_info" in sample:
        return COCOAnnotationType.PANOPTIC
    if "bbox" in sample or "segmentation" in sample:
        return COCOAnnotationType.INSTANCES
    raise ValueError(
        f"Cannot determine annotation type from {annotation_file}. "
        f"First annotation has keys: {list(sample.keys())}"
    )


class COCODataSource:
    """Unified COCO annotation interface for all CV tasks.

    Wraps ``pycocotools.COCO`` and provides:
      - **Detection**: ``get_detection_target()`` → bboxes + labels
      - **Instance masks**: ``get_instance_masks()`` → per-object binary masks
        via ``coco.annToMask()``
      - **Semantic map**: ``get_semantic_map()`` → single-channel class-index
        map (flattened from instance masks)
      - **Panoptic-from-instances**: ``get_panoptic_from_instances()`` →
        creates a panoptic-style ID map + segments_info that HF universal
        segmentation processors accept directly

    Parameters
    ----------
    annotation_file : str
        Path to a COCO ``instances_*.json`` file.
    """

    def __init__(self, annotation_file: str):
        self.coco = COCO(annotation_file)
        self.image_ids = list(sorted(self.coco.imgs.keys()))

        cats = self.coco.loadCats(self.coco.getCatIds())
        self.class_names: List[str] = [cat["name"] for cat in cats]
        self.cat_to_idx: Dict[int, int] = {
            cat["id"]: idx for idx, cat in enumerate(cats)
        }
        self.idx_to_cat: Dict[int, int] = {
            idx: cat_id for cat_id, idx in self.cat_to_idx.items()
        }
        self.num_classes: int = len(cats)

    def __len__(self) -> int:
        return len(self.image_ids)

    # ------------------------------------------------------------------
    # Image info
    # ------------------------------------------------------------------
    def get_image_info(self, image_id: int) -> dict:
        return self.coco.loadImgs(image_id)[0]

    def get_image_path(self, image_id: int, root_dir: str) -> Path:
        info = self.get_image_info(image_id)
        return Path(root_dir) / info["file_name"]

    # ------------------------------------------------------------------
    # Raw annotations
    # ------------------------------------------------------------------
    def get_annotations(self, image_id: int) -> List[dict]:
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        return self.coco.loadAnns(ann_ids)

    # ------------------------------------------------------------------
    # Detection accessors
    # ------------------------------------------------------------------
    def get_detection_target(
        self,
        image_id: int,
        skip_crowd: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Return detection target: boxes ``[x1, y1, x2, y2]`` + labels.

        Boxes are in absolute pixel coordinates.  Crowd annotations are
        skipped by default (standard COCO evaluation practice).
        """
        anns = self.get_annotations(image_id)
        boxes: List[List[float]] = []
        labels: List[int] = []

        for ann in anns:
            if skip_crowd and ann.get("iscrowd", 0):
                continue
            bbox = ann["bbox"]  # COCO format: [x, y, w, h]
            boxes.append([
                bbox[0],
                bbox[1],
                bbox[0] + bbox[2],
                bbox[1] + bbox[3],
            ])
            labels.append(self.cat_to_idx[ann["category_id"]])

        if not boxes:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros(0, dtype=torch.int64)
        else:
            boxes_t = torch.as_tensor(boxes, dtype=torch.float32)
            labels_t = torch.as_tensor(labels, dtype=torch.int64)

        return {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([image_id]),
        }

    # ------------------------------------------------------------------
    # Segmentation accessors
    # ------------------------------------------------------------------
    def get_instance_masks(
        self,
        image_id: int,
        skip_crowd: bool = True,
    ) -> Tuple[List[np.ndarray], List[int], Tuple[int, int]]:
        """Return per-instance binary masks + class labels via ``annToMask()``.

        Returns
        -------
        masks : list of (H, W) uint8 ndarrays
            One binary mask per instance.
        labels : list of int
            Contiguous class index for each instance.
        image_size : (height, width)
        """
        anns = self.get_annotations(image_id)
        img_info = self.get_image_info(image_id)
        h, w = img_info["height"], img_info["width"]

        masks: List[np.ndarray] = []
        labels: List[int] = []

        for ann in anns:
            if skip_crowd and ann.get("iscrowd", 0):
                continue
            mask = self.coco.annToMask(ann)
            masks.append(mask)
            labels.append(self.cat_to_idx[ann["category_id"]])

        return masks, labels, (h, w)

    def get_semantic_map(
        self,
        image_id: int,
        ignore_index: int = 255,
    ) -> np.ndarray:
        """Flatten instance masks to a class-index semantic map.

        Later annotations overwrite earlier ones at overlapping pixels.
        Unlabeled pixels get ``ignore_index``.
        """
        masks, labels, (h, w) = self.get_instance_masks(image_id)

        semantic = np.full((h, w), fill_value=ignore_index, dtype=np.int64)
        for mask, label in zip(masks, labels):
            semantic[mask > 0] = label

        return semantic

    def get_panoptic_from_instances(
        self,
        image_id: int,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Create a panoptic-style ID map from instance annotations.

        Each instance gets a unique ID (starting from 1).  The returned
        ``segments_info`` list has the format expected by HF universal
        segmentation processors::

            [{"id": 1, "category_id": 5}, {"id": 2, "category_id": 12}, ...]

        Returns
        -------
        panoptic_map : (H, W) int32 ndarray
            Pixel value = instance ID (0 = unlabeled).
        segments_info : list of dicts
            Per-segment metadata with ``id`` and ``category_id`` keys.
        """
        masks, labels, (h, w) = self.get_instance_masks(image_id)

        panoptic_map = np.zeros((h, w), dtype=np.int32)
        segments_info: List[Dict[str, Any]] = []

        for instance_id, (mask, label) in enumerate(
            zip(masks, labels), start=1,
        ):
            panoptic_map[mask > 0] = instance_id
            segments_info.append({
                "id": instance_id,
                "category_id": label,
            })

        return panoptic_map, segments_info

    def get_coco_api(self) -> COCO:
        """Return the raw ``pycocotools.COCO`` object for ``COCOeval``."""
        return self.coco
