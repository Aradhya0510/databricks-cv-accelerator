"""Collate functions for segmentation batches.

Handles two distinct label formats:
  - **Semantic**: ``labels`` is a ``(H, W)`` class-index tensor → stack into ``(B, H, W)``.
  - **Universal**: ``mask_labels`` + ``class_labels`` are per-image lists → keep as lists.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def segmentation_collate_fn(
    batch: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Collate segmentation samples into a batch dict.

    The adapter produces dicts with varying keys depending on the model type.
    This collate function stacks tensors where shapes match and keeps lists
    where they don't (instance masks have variable numbers of objects).
    """
    keys = batch[0].keys()
    collated: Dict[str, Any] = {}

    for key in keys:
        values = [sample[key] for sample in batch]

        if key == "pixel_values":
            collated[key] = torch.stack(values, dim=0)
        elif key == "pixel_mask":
            collated[key] = torch.stack(values, dim=0)
        elif key == "labels":
            # Semantic segmentation: (H, W) tensors → stack
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values, dim=0)
            else:
                collated[key] = values
        elif key in ("mask_labels", "class_labels"):
            # Universal segmentation: variable-length per image → keep as list
            collated[key] = values
        else:
            collated[key] = values

    return collated
