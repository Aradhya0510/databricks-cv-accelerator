"""Collate function for classification batches."""

import torch
from typing import Any, Dict, List, Tuple


def classification_collate_fn(
    batch: List[Tuple[torch.Tensor, int]],
) -> Dict[str, Any]:
    """Collate classification samples into a batch dict.

    Args:
        batch: List of ``(image_tensor, label_int)`` tuples.

    Returns:
        ``{"pixel_values": Tensor[B,C,H,W], "labels": Tensor[B]}``
    """
    images = []
    labels = []

    for image, label in batch:
        images.append(image)
        labels.append(label)

    return {
        "pixel_values": torch.stack(images, dim=0),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
