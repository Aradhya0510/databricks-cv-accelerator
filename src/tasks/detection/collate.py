"""Standalone collate function for object detection batches.

Extracted from ``DetectionDataModule._collate_fn`` so it can be used
by both Lightning DataModules and HF Trainer data collators.
"""

import torch
from typing import Dict, List, Tuple, Any


def detection_collate_fn(batch: List[Tuple[Any, Dict[str, torch.Tensor]]]) -> Dict[str, Any]:
    """Collate detection samples into a batch dict.

    Args:
        batch: List of ``(image_tensor, target_dict)`` tuples.

    Returns:
        ``{"pixel_values": Tensor[B,C,H,W], "labels": List[dict]}``
    """
    images = []
    targets = []

    for image, target in batch:
        images.append(image)
        targets.append(target)

    pixel_values = torch.stack(images, dim=0)

    return {
        "pixel_values": pixel_values,
        "labels": targets,
    }
