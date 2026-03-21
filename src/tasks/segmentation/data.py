"""Segmentation datasets — three standard formats, COCO-first.

**COCO instances** (preferred — same ``instances_*.json`` used for detection)::

    data/
    ├── train2017/                    # images
    └── instances_train2017.json      # COCO instances with polygon masks

    Uses ``pycocotools.COCO.annToMask()`` to extract per-instance binary
    masks, then composes them into a panoptic-style ID map that HF
    universal processors (Mask2Former, etc.) accept directly.  For semantic
    models, the adapter flattens it to a class-index map.

**COCO panoptic** (for panoptic benchmarks)::

    data/
    ├── train2017/                    # images
    ├── panoptic_train2017/           # RGB-encoded panoptic masks
    └── panoptic_train2017.json       # segments_info per image

**ADE20K-style** (zero-annotation fallback)::

    root_dir/
    ├── images/
    │   ├── 0001.jpg
    │   └── ...
    └── masks/
        ├── 0001.png   # single-channel, pixel value = class ID
        └── ...

All three formats work with semantic models (SegFormer, UperNet, BEiT, DPT)
and universal models (Mask2Former, MaskFormer, OneFormer) — the HF processor
handles format conversion internally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ...utils.coco import COCODataSource


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ---------------------------------------------------------------------------
# COCO instances dataset (preferred path)
# ---------------------------------------------------------------------------

class COCOInstanceSegmentationDataset(torch.utils.data.Dataset):
    """Segmentation dataset backed by a COCO ``instances_*.json``.

    Uses the shared :class:`COCODataSource` to extract per-instance binary
    masks via ``pycocotools.COCO.annToMask()``, then composes them into a
    panoptic-style ID map + ``segments_info`` list.  This preserves
    instance-level information for universal models (Mask2Former,
    MaskFormer, OneFormer) while remaining compatible with semantic
    models (the adapter flattens to a class-index map automatically).

    This is the **same** ``instances_*.json`` that detection uses —
    a single annotation file drives both tasks.
    """

    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        transform: Optional[Any] = None,
    ):
        self.image_dir = Path(image_dir)
        self.source = COCODataSource(annotation_file)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_id = self.source.image_ids[idx]
        image = Image.open(
            self.source.get_image_path(image_id, str(self.image_dir))
        ).convert("RGB")

        panoptic_map, segments_info = self.source.get_panoptic_from_instances(
            image_id
        )

        if self.transform:
            return self.transform(image, panoptic_map, segments_info)

        return {
            "image": image,
            "panoptic_map": panoptic_map,
            "segments_info": segments_info,
        }


# ---------------------------------------------------------------------------
# ADE20K-style dataset (zero-annotation fallback)
# ---------------------------------------------------------------------------

class SemanticSegmentationDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for semantic segmentation with image + mask pairs."""

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Any] = None,
        image_subdir: str = "images",
        mask_subdir: str = "masks",
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform

        image_dir = self.root_dir / image_subdir
        mask_dir = self.root_dir / mask_subdir

        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")
        if not mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

        self.samples: List[Tuple[Path, Path]] = []
        for img_path in sorted(image_dir.iterdir()):
            if img_path.suffix.lower() not in _IMAGE_EXTENSIONS:
                continue
            mask_path = mask_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                mask_path = mask_dir / f"{img_path.stem}{img_path.suffix}"
            if mask_path.exists():
                self.samples.append((img_path, mask_path))

        if not self.samples:
            raise ValueError(
                f"No matching image/mask pairs found in {root_dir}. "
                f"Expected {image_subdir}/<name>.jpg + {mask_subdir}/<name>.png"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, mask_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform:
            return self.transform(image, mask)

        return {"image": image, "mask": mask}


# ---------------------------------------------------------------------------
# COCO panoptic dataset
# ---------------------------------------------------------------------------

class COCOPanopticSegmentationDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for COCO panoptic segmentation.

    Expects the standard COCO panoptic layout:

    - *image_dir*: directory of RGB images (``train2017/``).
    - *annotation_file*: panoptic JSON (``panoptic_train2017.json``).
    - *panoptic_dir*: directory of RGB-encoded panoptic PNGs.  When
      ``None``, derived from the annotation file name by stripping
      ``.json`` (the standard COCO convention — ``panoptic_train2017.json``
      → ``panoptic_train2017/``).

    The RGB panoptic PNGs encode instance IDs as
    ``R + G*256 + B*256*256``.  The annotation JSON maps these IDs to
    category IDs via ``segments_info``.
    """

    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        transform: Optional[Any] = None,
        panoptic_dir: Optional[str] = None,
    ):
        self.image_dir = Path(image_dir)
        self.annotation_file = Path(annotation_file)
        self.transform = transform

        if panoptic_dir is not None:
            self.panoptic_dir = Path(panoptic_dir)
        else:
            self.panoptic_dir = self.annotation_file.parent / self.annotation_file.stem

        if not self.panoptic_dir.exists():
            raise FileNotFoundError(
                f"Panoptic mask directory not found: {self.panoptic_dir}. "
                f"Expected a directory alongside {self.annotation_file} named "
                f"'{self.annotation_file.stem}/'."
            )

        with open(self.annotation_file, "r") as f:
            panoptic_data = json.load(f)

        id_to_image = {img["id"]: img for img in panoptic_data["images"]}

        categories = panoptic_data.get("categories", [])
        self.cat_to_idx: Dict[int, int] = {}
        for i, cat in enumerate(sorted(categories, key=lambda c: c["id"])):
            self.cat_to_idx[cat["id"]] = i

        self.samples: List[Dict[str, Any]] = []
        for ann in panoptic_data["annotations"]:
            image_id = ann["image_id"]
            image_info = id_to_image.get(image_id)
            if image_info is None:
                continue

            img_path = self.image_dir / image_info["file_name"]
            panoptic_path = self.panoptic_dir / ann["file_name"]

            if img_path.exists() and panoptic_path.exists():
                self.samples.append({
                    "image_path": img_path,
                    "panoptic_path": panoptic_path,
                    "segments_info": ann["segments_info"],
                })

        if not self.samples:
            raise ValueError(
                f"No valid image/panoptic pairs found. "
                f"image_dir={image_dir}, annotation_file={annotation_file}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        panoptic_png = Image.open(sample["panoptic_path"]).convert("RGB")

        panoptic_map = _decode_panoptic_png(panoptic_png)
        segments_info = sample["segments_info"]

        if self.transform:
            return self.transform(image, panoptic_map, segments_info)

        return {
            "image": image,
            "panoptic_map": panoptic_map,
            "segments_info": segments_info,
        }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _decode_panoptic_png(panoptic_png: Image.Image) -> np.ndarray:
    """Decode an RGB panoptic PNG into an integer ID map.

    COCO panoptic encodes IDs as ``R + G*256 + B*256*256``.
    """
    arr = np.array(panoptic_png, dtype=np.int32)
    return arr[:, :, 0] + arr[:, :, 1] * 256 + arr[:, :, 2] * 256 * 256


def panoptic_to_semantic(
    panoptic_map: np.ndarray,
    segments_info: List[Dict[str, Any]],
    cat_to_idx: Optional[Dict[int, int]] = None,
    ignore_index: int = 255,
) -> np.ndarray:
    """Convert a COCO panoptic ID map to a semantic class map.

    Pixels that don't belong to any segment get ``ignore_index``.
    """
    semantic = np.full(panoptic_map.shape, fill_value=ignore_index, dtype=np.int64)
    for seg in segments_info:
        mask = panoptic_map == seg["id"]
        cat_id = seg["category_id"]
        if cat_to_idx is not None:
            cat_id = cat_to_idx.get(cat_id, cat_id)
        semantic[mask] = cat_id
    return semantic
