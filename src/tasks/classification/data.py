"""ImageFolder-style classification dataset.

Expects directory structure:
    root_dir/
        class_0/
            img001.jpg
            img002.jpg
        class_1/
            img003.jpg
        ...

Class names are derived from sorted subdirectory names, or can be explicitly
provided via ``class_names``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoImageProcessor


class ImageFolderClassificationDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset for image classification from a folder structure."""

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(
        self,
        root_dir: str,
        processor: Optional[AutoImageProcessor] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.processor = processor

        # Discover classes from subdirectories
        subdirs = sorted(
            [d for d in self.root_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )

        if class_names:
            self.class_names = class_names
            self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        else:
            self.class_names = [d.name for d in subdirs]
            self.class_to_idx = {d.name: idx for idx, d in enumerate(subdirs)}

        # Collect all image paths and their labels
        self.samples: List[Tuple[Path, int]] = []
        for subdir in subdirs:
            class_name = subdir.name
            if class_name not in self.class_to_idx:
                continue
            label = self.class_to_idx[class_name]
            for img_path in sorted(subdir.iterdir()):
                if img_path.suffix.lower() in self.EXTENSIONS:
                    self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.processor is not None:
            processed = self.processor(image, return_tensors="pt")
            pixel_values = processed.pixel_values.squeeze(0)
        else:
            import torchvision.transforms.functional as F

            pixel_values = F.to_tensor(image)

        return pixel_values, label
