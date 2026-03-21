"""COCO-format detection dataset backed by the shared COCODataSource."""

from typing import Any, Dict, Optional, Tuple

import torch
from pathlib import Path
from PIL import Image

from ...utils.coco import COCODataSource


class COCODetectionDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset for COCO-formatted object detection datasets.

    Delegates all annotation parsing to :class:`COCODataSource`, which
    wraps ``pycocotools.COCO`` and provides category mapping, bounding
    box conversion, and crowd-annotation filtering.
    """

    def __init__(
        self,
        root_dir: str,
        annotation_file: str,
        transform: Optional[Any] = None,
    ):
        self.root_dir = Path(root_dir)
        self.source = COCODataSource(annotation_file)
        self.transform = transform

        self.class_names = self.source.class_names
        self.cat_to_idx = self.source.cat_to_idx

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, torch.Tensor]]:
        image_id = self.source.image_ids[idx]
        image = Image.open(
            self.source.get_image_path(image_id, str(self.root_dir))
        ).convert("RGB")

        target = self.source.get_detection_target(image_id)

        if self.transform:
            image, target = self.transform(image, target)

        return image, target
