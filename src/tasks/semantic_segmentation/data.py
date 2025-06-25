from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pycocotools.coco import COCO
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoFeatureExtractor
from .adapters import get_semantic_adapter

@dataclass
class SemanticSegmentationDataConfig:
    """Configuration for semantic segmentation data module."""
    data_path: str
    annotation_file: str
    image_size: int = 512
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    batch_size: int = 8
    num_workers: int = 4
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation: int = 30
    brightness_contrast: float = 0.2
    hue_saturation: float = 0.2
    model_name: Optional[str] = None

class COCOSemanticSegmentationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str,
        annotation_file: str,
        transform: Optional[Union[A.Compose, Any]] = None,
        image_size: int = 512,
        model_name: Optional[str] = None
    ):
        self.root_dir = Path(root_dir)
        self.coco = COCO(annotation_file)
        self.transform = transform
        self.image_size = image_size
        self.model_name = model_name
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        # Initialize adapter if using Hugging Face model
        if model_name:
            self.adapter = get_semantic_adapter(model_name, image_size=image_size)
        else:
            self.adapter = None
        
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
        
        # Create semantic mask
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in anns:
            if ann['segmentation']:
                for seg in ann['segmentation']:
                    points = np.array(seg).reshape(-1, 2).astype(np.int32)
                    cv2.fillPoly(mask, [points], ann['category_id'])
        
        # Create target dictionary
        target = {
            "semantic_masks": torch.as_tensor(mask, dtype=torch.long),
            "image_id": torch.tensor(img_id)
        }
        
        # Apply transforms
        if self.adapter:
            # Use adapter for preprocessing
            processed_image, adapted_target = self.adapter(image_pil, target)
            return {
                "pixel_values": processed_image,
                "labels": adapted_target
            }
        else:
            # Use Albumentations transforms
            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            return {
                "pixel_values": image,
                "labels": target
            }

class SemanticSegmentationDataModule(pl.LightningDataModule):
    def __init__(self, config: Union[Dict[str, Any], SemanticSegmentationDataConfig]):
        super().__init__()
        if isinstance(config, dict):
            config = SemanticSegmentationDataConfig(**config)
        self.config = config
        
        # Initialize adapter if using Hugging Face model
        if config.model_name:
            self.adapter = get_semantic_adapter(config.model_name, image_size=config.image_size)
            self.train_transform = None
            self.val_transform = None
        else:
            # Define transforms
            self.train_transform = A.Compose([
                A.Resize(config.image_size, config.image_size),
                A.HorizontalFlip(p=0.5 if config.horizontal_flip else 0),
                A.VerticalFlip(p=0.5 if config.vertical_flip else 0),
                A.Rotate(limit=config.rotation, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=config.brightness_contrast,
                    contrast_limit=config.brightness_contrast,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=config.hue_saturation,
                    sat_shift_limit=config.hue_saturation,
                    val_shift_limit=config.hue_saturation,
                    p=0.5
                ),
                A.Normalize(
                    mean=config.mean,
                    std=config.std
                ),
                ToTensorV2()
            ])
            
            self.val_transform = A.Compose([
                A.Resize(config.image_size, config.image_size),
                A.Normalize(
                    mean=config.mean,
                    std=config.std
                ),
                ToTensorV2()
            ])
    
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = COCOSemanticSegmentationDataset(
                root_dir=self.config.data_path,
                annotation_file=self.config.annotation_file,
                transform=self.train_transform,
                image_size=self.config.image_size,
                model_name=self.config.model_name
            )
            
            self.val_dataset = COCOSemanticSegmentationDataset(
                root_dir=self.config.data_path,
                annotation_file=self.config.annotation_file,
                transform=self.val_transform,
                image_size=self.config.image_size,
                model_name=self.config.model_name
            )
        
        if stage == 'test':
            self.test_dataset = COCOSemanticSegmentationDataset(
                root_dir=self.config.data_path,
                annotation_file=self.config.annotation_file,
                transform=self.val_transform,
                image_size=self.config.image_size,
                model_name=self.config.model_name
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
    
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