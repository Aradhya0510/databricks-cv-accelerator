"""COCO-format detection dataset."""

from typing import Any, Dict, Optional, Tuple

import torch
from pathlib import Path
from pycocotools.coco import COCO
from PIL import Image


class COCODetectionDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset for COCO-formatted object detection datasets.

    This class uses the pycocotools library to load and parse annotations.
    It is designed to be generic for any dataset in the COCO format.
    """

    def __init__(
        self,
        root_dir: str,
        annotation_file: str,
        transform: Optional[Any] = None,
    ):
        self.root_dir = Path(root_dir)
        self.coco = COCO(annotation_file)
        self.transform = transform
        self.ids = list(sorted(self.coco.imgs.keys()))

        # Load class names and create category to index mapping
        self.class_names = [cat["name"] for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.cat_to_idx = {cat["id"]: idx for idx, cat in enumerate(self.coco.loadCats(self.coco.getCatIds()))}

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, torch.Tensor]]:
        # Load image
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.root_dir / img_info['file_name']
        image = Image.open(img_path).convert("RGB")

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        # Prepare boxes and labels
        boxes = []
        labels = []

        for ann in anns:
            bbox = ann['bbox']  # [x, y, w, h]
            # Convert to [x1, y1, x2, y2] format in absolute pixels
            boxes.append([
                bbox[0],              # x1
                bbox[1],              # y1
                bbox[0] + bbox[2],    # x2
                bbox[1] + bbox[3],    # y2
            ])
            # Convert category ID to zero-based index
            labels.append(self.cat_to_idx[ann['category_id']])

        # Handle empty boxes case
        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,  # [x1, y1, x2, y2] in absolute pixels
            'labels': labels,
            'image_id': torch.tensor([img_id]),
        }

        # Apply transforms
        if self.transform:
            image, target = self.transform(image, target)

        return image, target
