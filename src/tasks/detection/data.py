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
        augmentations: Optional[Any] = None,
    ):
        """
        Args:
            root_dir: Directory holding the images named in the annotations.
            annotation_file: COCO ``instances_*.json``.
            transform: The model-family input adapter (resize, normalise,
                box-format conversion).  Always applied last.
            augmentations: Optional albumentations pipeline, applied to the
                raw image and boxes *before* the adapter so the two stay in
                sync.  Training split only.
        """
        self.root_dir = Path(root_dir)
        self.source = COCODataSource(annotation_file)
        self.transform = transform
        self.augmentations = augmentations

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

        if self.augmentations is not None:
            image, target = self._augment(image, target)

        if self.transform:
            image, target = self.transform(image, target)

        return image, target

    def _augment(self, image, target):
        """Co-transform the image and its boxes, then rebuild the target."""
        import numpy as np

        boxes = target["boxes"].tolist()
        labels = target["labels"].tolist()

        result = self.augmentations(
            image=np.array(image), bboxes=boxes, labels=labels,
        )

        image = Image.fromarray(result["image"])

        aug_boxes = result["bboxes"]
        aug_labels = result["labels"]
        if len(aug_boxes) == 0:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros(0, dtype=torch.int64)
        else:
            target["boxes"] = torch.as_tensor(aug_boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(aug_labels, dtype=torch.int64)

        return image, target
