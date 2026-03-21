"""Segmentation model-family adapters — config, not class hierarchies.

Each segmentation model family's quirks (model API type, pixel mask,
label format, postprocessing) are captured in a ``SegmentationFamilyConfig``
dataclass.  ``detect_segmentation_family()`` selects the right config by
substring match on the model name.

Adding a new segmentation architecture means adding an entry to
``_FAMILY_CONFIGS`` — no new class, no inheritance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor


# ---------------------------------------------------------------------------
# Family config dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SegmentationFamilyConfig:
    """Declares how a segmentation model family handles inputs and outputs.

    ``model_type`` is the critical fork:
      - ``"semantic"``: uses ``AutoModelForSemanticSegmentation``, labels are
        a single ``(H, W)`` class-index tensor per image.
      - ``"universal"``: uses ``AutoModelForUniversalSegmentation``
        (Mask2Former, MaskFormer, OneFormer), labels are per-instance
        ``mask_labels`` + ``class_labels`` lists.
    """

    model_type: str  # "semantic" | "universal"
    requires_pixel_mask: bool
    reduce_labels: bool
    output_logits_attr: str


# ---------------------------------------------------------------------------
# Family registry — extend the framework by adding an entry here
# ---------------------------------------------------------------------------

_FAMILY_CONFIGS: Dict[str, SegmentationFamilyConfig] = {
    # Universal segmentation models (instance / panoptic / semantic)
    "mask2former": SegmentationFamilyConfig(
        model_type="universal",
        requires_pixel_mask=True,
        reduce_labels=False,
        output_logits_attr="class_queries_logits",
    ),
    "oneformer": SegmentationFamilyConfig(
        model_type="universal",
        requires_pixel_mask=True,
        reduce_labels=False,
        output_logits_attr="class_queries_logits",
    ),
    "maskformer": SegmentationFamilyConfig(
        model_type="universal",
        requires_pixel_mask=True,
        reduce_labels=False,
        output_logits_attr="class_queries_logits",
    ),
    # Semantic segmentation models
    "segformer": SegmentationFamilyConfig(
        model_type="semantic",
        requires_pixel_mask=False,
        reduce_labels=False,
        output_logits_attr="logits",
    ),
    "upernet": SegmentationFamilyConfig(
        model_type="semantic",
        requires_pixel_mask=False,
        reduce_labels=False,
        output_logits_attr="logits",
    ),
    "beit": SegmentationFamilyConfig(
        model_type="semantic",
        requires_pixel_mask=False,
        reduce_labels=False,
        output_logits_attr="logits",
    ),
    "dpt": SegmentationFamilyConfig(
        model_type="semantic",
        requires_pixel_mask=False,
        reduce_labels=False,
        output_logits_attr="logits",
    ),
}

_DEFAULT_CONFIG = SegmentationFamilyConfig(
    model_type="semantic",
    requires_pixel_mask=False,
    reduce_labels=False,
    output_logits_attr="logits",
)


def detect_segmentation_family(model_name: str) -> Tuple[str, SegmentationFamilyConfig]:
    """Select family config by substring match on model name."""
    model_lower = model_name.lower()
    for family, cfg in _FAMILY_CONFIGS.items():
        if family in model_lower:
            return family, cfg
    return "generic", _DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Input adapter (single class, config-driven)
# ---------------------------------------------------------------------------

class SegmentationInputAdapter:
    """Unified input adapter for all segmentation model families.

    Handles two data formats transparently:
      - **ADE20K-style**: ``(image, mask_pil)`` — 2-arg call.
      - **COCO panoptic**: ``(image, panoptic_id_map, segments_info)`` — 3-arg call.

    The HF processor handles per-family differences (label conversion,
    pixel mask generation).
    """

    def __init__(
        self,
        model_name: str,
        image_size: int,
        family_cfg: SegmentationFamilyConfig,
    ):
        self.family_cfg = family_cfg
        self.processor = AutoImageProcessor.from_pretrained(
            model_name,
            size={"height": image_size, "width": image_size},
            do_resize=True,
            do_reduce_labels=family_cfg.reduce_labels,
        )

    def __call__(
        self,
        image: Image.Image,
        mask_or_panoptic,
        segments_info: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if segments_info is not None:
            return self._process_coco_panoptic(image, mask_or_panoptic, segments_info)
        return self._process_semantic_mask(image, mask_or_panoptic)

    def _process_semantic_mask(
        self, image: Image.Image, mask: Image.Image,
    ) -> Dict[str, Any]:
        """ADE20K-style: mask is a PIL Image (pixel value = class ID)."""
        seg_map = np.array(mask)
        inputs = self.processor(
            images=image,
            segmentation_maps=seg_map,
            return_tensors="pt",
        )
        return _squeeze_batch_dim(inputs)

    def _process_coco_panoptic(
        self,
        image: Image.Image,
        panoptic_map: np.ndarray,
        segments_info: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """COCO panoptic: panoptic_map is a decoded int32 ID map."""
        id_to_semantic = {
            seg["id"]: seg["category_id"] for seg in segments_info
        }
        # Background pixels (value 0) must be in the mapping for HF processors
        # that iterate over all unique segmentation map values.
        if 0 not in id_to_semantic:
            id_to_semantic[0] = 0

        if self.family_cfg.model_type == "universal":
            # Universal processors (Mask2Former, etc.) accept panoptic maps
            # with an instance-to-semantic mapping and produce mask_labels +
            # class_labels automatically.
            inputs = self.processor(
                images=image,
                segmentation_maps=panoptic_map,
                instance_id_to_semantic_id=id_to_semantic,
                return_tensors="pt",
            )
        else:
            # Semantic processors need a plain class-index map.
            from .data import panoptic_to_semantic
            semantic_map = panoptic_to_semantic(panoptic_map, segments_info)
            inputs = self.processor(
                images=image,
                segmentation_maps=semantic_map,
                return_tensors="pt",
            )
        return _squeeze_batch_dim(inputs)


def _squeeze_batch_dim(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Remove the batch dimension added by ``return_tensors='pt'``."""
    result: Dict[str, Any] = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.squeeze(0)
        elif isinstance(v, list) and len(v) > 0:
            result[k] = v[0]
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Output helpers for evaluation
# ---------------------------------------------------------------------------

def postprocess_semantic(
    logits: torch.Tensor,
    target_sizes: List[Tuple[int, int]],
) -> List[torch.Tensor]:
    """Upsample semantic logits and argmax to produce per-image class maps."""
    upsampled = torch.nn.functional.interpolate(
        logits, size=target_sizes[0], mode="bilinear", align_corners=False,
    )
    return [seg.argmax(dim=0) for seg in upsampled]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_input_adapter(
    model_name: str, image_size: int = 512,
) -> SegmentationInputAdapter:
    """Get the appropriate input adapter for a segmentation model."""
    _, cfg = detect_segmentation_family(model_name)
    return SegmentationInputAdapter(model_name, image_size, cfg)
