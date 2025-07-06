from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import lightning as pl
from pycocotools.coco import COCO
import cv2
import numpy as np
from PIL import Image
from .adapters import get_input_adapter

@dataclass
class DetectionDataConfig:
    """Configuration for detection data module."""
    # Separate paths for train/val/test splits
    train_data_path: str
    train_annotation_file: str
    val_data_path: str
    val_annotation_file: str
    test_data_path: Optional[str] = None
    test_annotation_file: Optional[str] = None
    
    batch_size: int = 8
    num_workers: int = 4
    model_name: Optional[str] = None
    image_size: Optional[List[int]] = None
    normalize_mean: Optional[List[float]] = None
    normalize_std: Optional[List[float]] = None
    augment: Optional[Dict[str, Any]] = None

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
                bbox[0],  # x1
                bbox[1],  # y1
                bbox[0] + bbox[2],  # x2
                bbox[1] + bbox[3]   # y2
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
            'image_id': torch.tensor([img_id])
        }
        
        # Apply transforms
        if self.transform:
            image, target = self.transform(image, target)
        
        return image, target

class DetectionDataModule(pl.LightningDataModule):
    def __init__(self, config: Union[Dict[str, Any], DetectionDataConfig]):
        super().__init__()
        if isinstance(config, dict):
            config = DetectionDataConfig(**config)
        self.config = config
        self.adapter = None  # Will be set after initialization
    
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = COCODetectionDataset(
                root_dir=self.config.train_data_path,
                annotation_file=self.config.train_annotation_file,
                transform=None  # Transform will be set by the adapter
            )
            
            self.val_dataset = COCODetectionDataset(
                root_dir=self.config.val_data_path,
                annotation_file=self.config.val_annotation_file,
                transform=None  # Transform will be set by the adapter
            )
        
        if stage == 'test':
            if self.config.test_data_path is not None and self.config.test_annotation_file is not None:
                self.test_dataset = COCODetectionDataset(
                    root_dir=self.config.test_data_path,
                    annotation_file=self.config.test_annotation_file,
                    transform=None  # Transform will be set by the adapter
                )
            else:
                # Use validation data for testing if test data not provided
                self.test_dataset = COCODetectionDataset(
                    root_dir=self.config.val_data_path,
                    annotation_file=self.config.val_annotation_file,
                    transform=None  # Transform will be set by the adapter
                )
        
        # Set the adapter after datasets are created
        if self.adapter is not None:
            if hasattr(self, 'train_dataset'):
                self.train_dataset.transform = self.adapter
            if hasattr(self, 'val_dataset'):
                self.val_dataset.transform = self.adapter
            if hasattr(self, 'test_dataset'):
                self.test_dataset.transform = self.adapter
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False
        )
    
    @staticmethod
    def _collate_fn(batch):
        """Collate function for object detection.
        
        Args:
            batch: List of tuples (image, target) where image is a tensor and target is a dict
            
        Returns:
            Dictionary with 'pixel_values' and 'labels' keys
        """
        # Separate images and targets
        images = []
        targets = []
        
        for image, target in batch:
            images.append(image)
            targets.append(target)
        
        # Stack images into a batch tensor
        pixel_values = torch.stack(images, dim=0)
        
        # Create batch dictionary
        batch_dict = {
            'pixel_values': pixel_values,
            'labels': targets
        }
        
        return batch_dict
    
    @property
    def class_names(self) -> List[str]:
        """Get list of class names."""
        return self.train_dataset.class_names 