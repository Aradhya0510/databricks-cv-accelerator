from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pycocotools.coco import COCO
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoFeatureExtractor
from .adapters import get_instance_adapter

@dataclass
class InstanceSegmentationDataConfig:
    """Configuration for instance segmentation data module."""
    # Separate paths for train/val/test splits
    train_data_path: str
    train_annotation_file: str
    val_data_path: str
    val_annotation_file: str
    test_data_path: Optional[str] = None
    test_annotation_file: Optional[str] = None
    
    image_size: int = 512
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    batch_size: int = 4  # Smaller batch size for instance segmentation
    num_workers: int = 4
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation: int = 30
    brightness_contrast: float = 0.2
    hue_saturation: float = 0.2
    model_name: Optional[str] = None

class COCOInstanceSegmentationDataset(torch.utils.data.Dataset):
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
        
        # Load class names
        self.class_names = [cat["name"] for cat in self.coco.loadCats(self.coco.getCatIds())]
        
    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.root_dir / img_info['file_name']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image for adapter
        image_pil = Image.fromarray(image)
        
        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Create instance masks and bounding boxes
        instance_masks = []
        bounding_boxes = []
        class_labels = []
        
        for ann in anns:
            if ann['segmentation']:
                # Create instance mask
                mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
                for seg in ann['segmentation']:
                    points = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [points], 1)
                
                instance_masks.append(mask)
                bounding_boxes.append(ann['bbox'])  # [x, y, width, height]
                class_labels.append(ann['category_id'])
        
        # Convert to tensors
        if instance_masks:
            instance_masks = torch.stack([torch.as_tensor(mask, dtype=torch.long) for mask in instance_masks])
            bounding_boxes = torch.as_tensor(bounding_boxes, dtype=torch.float32)
            class_labels = torch.as_tensor(class_labels, dtype=torch.long)
        else:
            # Empty image case
            instance_masks = torch.empty((0, img_info['height'], img_info['width']), dtype=torch.long)
            bounding_boxes = torch.empty((0, 4), dtype=torch.float32)
            class_labels = torch.empty((0,), dtype=torch.long)
        
        # Create target dictionary
        target = {
            "instance_masks": instance_masks,
            "bounding_boxes": bounding_boxes,
            "class_labels": class_labels,
            "image_id": torch.tensor(img_id)
        }
        
        # Apply transforms
        if self.transform:
            image_pil, target = self.transform(image_pil, target)
        
        return {
            "pixel_values": image_pil,
            "labels": target
        }

class InstanceSegmentationDataModule(pl.LightningDataModule):
    def __init__(self, config: Union[Dict[str, Any], InstanceSegmentationDataConfig]):
        super().__init__()
        if isinstance(config, dict):
            config = InstanceSegmentationDataConfig(**config)
        self.config = config
        self.adapter = None  # Will be set after initialization
    
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = COCOInstanceSegmentationDataset(
                root_dir=self.config.train_data_path,
                annotation_file=self.config.train_annotation_file,
                transform=None  # Transform will be set by the adapter
            )
            
            self.val_dataset = COCOInstanceSegmentationDataset(
                root_dir=self.config.val_data_path,
                annotation_file=self.config.val_annotation_file,
                transform=None  # Transform will be set by the adapter
            )
        
        if stage == 'test':
            if self.config.test_data_path is not None and self.config.test_annotation_file is not None:
                self.test_dataset = COCOInstanceSegmentationDataset(
                    root_dir=self.config.test_data_path,
                    annotation_file=self.config.test_annotation_file,
                    transform=None  # Transform will be set by the adapter
                )
            else:
                # Use validation data for testing if test data not provided
                self.test_dataset = COCOInstanceSegmentationDataset(
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
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Custom collate function to handle variable number of instances."""
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        
        # Handle variable number of instances
        labels = {
            "instance_masks": [item["labels"]["instance_masks"] for item in batch],
            "bounding_boxes": [item["labels"]["bounding_boxes"] for item in batch],
            "class_labels": [item["labels"]["class_labels"] for item in batch],
            "image_id": torch.stack([item["labels"]["image_id"] for item in batch])
        }
        
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }
    
    @property
    def class_names(self) -> List[str]:
        """Get class names from the dataset."""
        if hasattr(self, 'train_dataset'):
            return self.train_dataset.class_names
        else:
            # Default COCO class names
            return [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush'
            ] 