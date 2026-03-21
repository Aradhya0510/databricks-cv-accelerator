"""Detection model-family adapters — config, not class hierarchies.

Each detection model family's quirks (pixel mask requirements, box format,
output attribute names) are captured in a ``DetectionFamilyConfig`` dataclass.
``detect_detection_family()`` selects the right config by substring match on
the model name — the same pattern the SLM accelerator uses for LoRA target
modules.

Adding a new detection architecture means adding an entry to
``_FAMILY_CONFIGS`` — no new class, no inheritance.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchvision.transforms.functional as F
from PIL import Image
from transformers import AutoImageProcessor


# ---------------------------------------------------------------------------
# Family config dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DetectionFamilyConfig:
    """Declares how a detection model family handles inputs and outputs.

    Captures behavioural differences between detection architectures so a
    single adapter class can handle all of them.  Each field maps to a
    concrete difference in preprocessing or postprocessing logic.
    """

    requires_pixel_mask: bool
    box_format: str  # "cxcywh_normalized" | "xyxy_absolute"
    output_logits_attr: str
    output_boxes_attr: str


# ---------------------------------------------------------------------------
# Family registry — extend the framework by adding an entry here
# ---------------------------------------------------------------------------

_FAMILY_CONFIGS: Dict[str, DetectionFamilyConfig] = {
    # More specific patterns first — order matters for substring matching.
    # "conditional-detr" and "rt-detr" must precede "detr" so they don't
    # fall through to the generic DETR entry.
    "yolos": DetectionFamilyConfig(
        requires_pixel_mask=False,
        box_format="cxcywh_normalized",
        output_logits_attr="logits",
        output_boxes_attr="pred_boxes",
    ),
    "deta": DetectionFamilyConfig(
        requires_pixel_mask=True,
        box_format="cxcywh_normalized",
        output_logits_attr="logits",
        output_boxes_attr="pred_boxes",
    ),
    "conditional-detr": DetectionFamilyConfig(
        requires_pixel_mask=True,
        box_format="cxcywh_normalized",
        output_logits_attr="logits",
        output_boxes_attr="pred_boxes",
    ),
    "rt-detr": DetectionFamilyConfig(
        requires_pixel_mask=False,
        box_format="cxcywh_normalized",
        output_logits_attr="logits",
        output_boxes_attr="pred_boxes",
    ),
    "rtdetr": DetectionFamilyConfig(
        requires_pixel_mask=False,
        box_format="cxcywh_normalized",
        output_logits_attr="logits",
        output_boxes_attr="pred_boxes",
    ),
    "detr": DetectionFamilyConfig(
        requires_pixel_mask=True,
        box_format="cxcywh_normalized",
        output_logits_attr="logits",
        output_boxes_attr="pred_boxes",
    ),
}

_DEFAULT_CONFIG = DetectionFamilyConfig(
    requires_pixel_mask=False,
    box_format="xyxy_absolute",
    output_logits_attr="logits",
    output_boxes_attr="pred_boxes",
)


def detect_detection_family(model_name: str) -> Tuple[str, DetectionFamilyConfig]:
    """Select family config by substring match on model name.

    Returns ``(family_name, config)``.  The first match wins, so more
    specific patterns should come before general ones in ``_FAMILY_CONFIGS``.
    """
    model_lower = model_name.lower()
    for family, cfg in _FAMILY_CONFIGS.items():
        if family in model_lower:
            return family, cfg
    return "generic", _DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Input adapter (single class, config-driven)
# ---------------------------------------------------------------------------

class DetectionInputAdapter:
    """Unified input adapter parameterised by :class:`DetectionFamilyConfig`."""

    def __init__(
        self,
        model_name: str,
        image_size: int,
        family_cfg: DetectionFamilyConfig,
    ):
        self.family_cfg = family_cfg
        self.image_size = image_size
        self.processor: Optional[AutoImageProcessor] = None

        if family_cfg.box_format == "cxcywh_normalized":
            self.processor = AutoImageProcessor.from_pretrained(
                model_name,
                size={"height": image_size, "width": image_size},
                do_resize=True,
                do_rescale=True,
                do_normalize=True,
                do_pad=True,
            )

    def __call__(
        self, image: Image.Image, target: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        if self.family_cfg.box_format == "xyxy_absolute":
            target["class_labels"] = target["labels"]
            return F.to_tensor(image), target

        return self._preprocess_detr_family(image, target)

    def _preprocess_detr_family(
        self, image: Image.Image, target: Dict,
    ) -> Tuple[torch.Tensor, Dict]:
        w, h = image.size
        boxes_xyxy = target["boxes"]

        if boxes_xyxy.shape[0] == 0:
            boxes_cxcywh = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes_cxcywh = _xyxy_to_cxcywh(boxes_xyxy)

        boxes_normalized = boxes_cxcywh / torch.tensor(
            [w, h, w, h], dtype=torch.float32,
        )

        adapted_target = {
            "boxes": boxes_normalized,
            "class_labels": target["labels"],
            "image_id": target["image_id"],
            "orig_size": torch.tensor([h, w]),
            "size": torch.tensor([h, w]),
        }

        processed = self.processor(
            image,
            return_tensors="pt",
            do_resize=True,
            do_rescale=True,
            do_normalize=True,
            do_pad=True,
        )

        return processed.pixel_values.squeeze(0), adapted_target


# ---------------------------------------------------------------------------
# Output adapter (single class, config-driven)
# ---------------------------------------------------------------------------

class DetectionOutputAdapter:
    """Unified output adapter parameterised by :class:`DetectionFamilyConfig`."""

    def __init__(
        self,
        model_name: str,
        image_size: int,
        family_cfg: DetectionFamilyConfig,
    ):
        self.family_cfg = family_cfg
        self.image_size = image_size
        self.processor = AutoImageProcessor.from_pretrained(
            model_name,
            size={"height": image_size, "width": image_size},
        )

    def adapt_output(self, outputs: Any) -> Dict[str, Any]:
        """Normalise model output to a standard dict."""
        loss = getattr(outputs, "loss", None)
        if loss is None and isinstance(outputs, dict):
            loss = outputs.get("loss")

        return {
            "loss": loss,
            "pred_boxes": getattr(outputs, self.family_cfg.output_boxes_attr),
            "pred_logits": getattr(outputs, self.family_cfg.output_logits_attr),
            "loss_dict": getattr(outputs, "loss_dict", {}),
        }

    def format_predictions(
        self,
        outputs: Dict[str, Any],
        batch: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """Format model outputs for metric computation."""
        # HF processors access .logits and .pred_boxes by attribute —
        # a SimpleNamespace satisfies this without importing model-specific
        # output classes.
        container = SimpleNamespace(
            logits=outputs["pred_logits"],
            pred_boxes=outputs["pred_boxes"],
        )

        target_sizes = self._resolve_target_sizes(outputs, batch)
        processed_outputs = self.processor.post_process_object_detection(
            container, threshold=0.7, target_sizes=target_sizes,
        )

        preds = []
        for i, po in enumerate(processed_outputs):
            image_id = torch.tensor([i])
            if batch and "labels" in batch and i < len(batch["labels"]):
                image_id = batch["labels"][i].get(
                    "image_id", torch.tensor([i]),
                )

            preds.append({
                "boxes": po["boxes"],
                "scores": po["scores"],
                "labels": po["labels"],
                "image_id": image_id,
            })
        return preds

    def format_targets(
        self, targets: List[Dict[str, torch.Tensor]],
    ) -> List[Dict[str, torch.Tensor]]:
        """Convert targets to [x1,y1,x2,y2] absolute pixels for metrics."""
        if self.family_cfg.box_format == "xyxy_absolute":
            return [
                {
                    "boxes": t["boxes"],
                    "labels": t.get("class_labels", t.get("labels")),
                    "image_id": t.get("image_id", torch.tensor([i])),
                }
                for i, t in enumerate(targets)
            ]

        formatted = []
        for i, target in enumerate(targets):
            boxes = target["boxes"]
            labels = target["class_labels"]
            h, w = target["size"]
            image_id = target.get("image_id", torch.tensor([i]))

            boxes_xyxy = torch.zeros_like(boxes)
            boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w  # x1
            boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h  # y1
            boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w  # x2
            boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h  # y2

            formatted.append({
                "boxes": boxes_xyxy,
                "labels": labels,
                "image_id": image_id,
            })
        return formatted

    def _resolve_target_sizes(
        self, outputs: Dict[str, Any], batch: Optional[Dict[str, Any]],
    ) -> List[List[int]]:
        if batch and "labels" in batch:
            return [
                t["size"].tolist()
                if "size" in t
                else [self.image_size, self.image_size]
                for t in batch["labels"]
            ]
        batch_size = outputs["pred_logits"].shape[0]
        return [[self.image_size, self.image_size]] * batch_size


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _xyxy_to_cxcywh(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    """Convert [x1, y1, x2, y2] to [center_x, center_y, width, height]."""
    x1, y1, x2, y2 = boxes_xyxy.unbind(1)
    return torch.stack(
        [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1,
    )


# ---------------------------------------------------------------------------
# Public API — backward-compatible factory functions
# ---------------------------------------------------------------------------

def get_input_adapter(
    model_name: str, image_size: int = 800,
) -> DetectionInputAdapter:
    """Get the appropriate input adapter for a model."""
    _, cfg = detect_detection_family(model_name)
    return DetectionInputAdapter(model_name, image_size, cfg)


def get_output_adapter(
    model_name: str, image_size: int = 800,
) -> DetectionOutputAdapter:
    """Get the appropriate output adapter for a model."""
    _, cfg = detect_detection_family(model_name)
    return DetectionOutputAdapter(model_name, image_size, cfg)


def get_adapter(
    model_name: str, image_size: int = 800,
) -> DetectionInputAdapter:
    """Get the appropriate adapter for a model (backward compatibility)."""
    return get_input_adapter(model_name, image_size)
