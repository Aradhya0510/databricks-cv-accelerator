from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from pathlib import Path
from transformers import AutoFeatureExtractor
from .adapters import get_adapter

@dataclass
class ClassificationDataConfig:
    """Configuration for classification data module."""
    data_path: str
    image_size: int = 224
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    batch_size: int = 32
    num_workers: int = 4
    horizontal_flip: bool = True
    vertical_flip: bool = False
    rotation: int = 30
    brightness_contrast: float = 0.2
    hue_saturation: float = 0.2
    model_name: Optional[str] = None

class ClassificationDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        transform: Optional[Union[A.Compose, Any]] = None,
        is_train: bool = True,
        model_name: Optional[str] = None,
        image_size: int = 224
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        self.is_train = is_train
        self.model_name = model_name
        self.image_size = image_size
        
        # Initialize adapter if using Hugging Face model
        if model_name:
            self.adapter = get_adapter(model_name, image_size=image_size)
        else:
            self.adapter = None
        
        # Load image paths and labels
        self.image_paths = []
        self.labels = []
        self.class_names = []
        
        # Assuming data is organized in class folders
        for class_idx, class_name in enumerate(sorted(self.data_path.iterdir())):
            if class_name.is_dir():
                self.class_names.append(class_name.name)
                for img_path in class_name.glob("*.jpg"):
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Create target dictionary
        target = {
            "class_labels": torch.tensor(label, dtype=torch.long),
            "image_id": torch.tensor(idx)
        }
        
        # Apply transformations
        if self.adapter:
            # Use adapter for preprocessing
            processed_image, adapted_target = self.adapter(image, target)
            return {
                "pixel_values": processed_image,
                "labels": adapted_target
            }
        else:
            # Use Albumentations transforms
            image = np.array(image)
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed["image"]
            return {
                "pixel_values": image,
                "labels": target
            }

class ClassificationDataModule(pl.LightningDataModule):
    def __init__(self, config: Union[Dict[str, Any], ClassificationDataConfig]):
        super().__init__()
        if isinstance(config, dict):
            config = ClassificationDataConfig(**config)
        self.config = config
        
        # Initialize adapter if using Hugging Face model
        if config.model_name:
            self.adapter = get_adapter(config.model_name, image_size=config.image_size)
            self.train_transform = None
            self.val_transform = None
        else:
            # Define Albumentations transforms
            self.train_transform = A.Compose([
                A.RandomResizedCrop(config.image_size, config.image_size),
                A.HorizontalFlip(p=0.5 if config.horizontal_flip else 0),
                A.VerticalFlip(p=0.5 if config.vertical_flip else 0),
                A.Rotate(limit=config.rotation),
                A.RandomBrightnessContrast(
                    brightness_limit=config.brightness_contrast,
                    contrast_limit=config.brightness_contrast
                ),
                A.Normalize(mean=config.mean, std=config.std),
                ToTensorV2(),
            ])
            
            self.val_transform = A.Compose([
                A.Resize(config.image_size, config.image_size),
                A.Normalize(mean=config.mean, std=config.std),
                ToTensorV2(),
            ])
    
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = ClassificationDataset(
                self.config.data_path,
                transform=self.train_transform,
                is_train=True,
                model_name=self.config.model_name,
                image_size=self.config.image_size
            )
            self.val_dataset = ClassificationDataset(
                self.config.data_path,
                transform=self.val_transform,
                is_train=False,
                model_name=self.config.model_name,
                image_size=self.config.image_size
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = ClassificationDataset(
                self.config.data_path,
                transform=self.val_transform,
                is_train=False,
                model_name=self.config.model_name,
                image_size=self.config.image_size
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=True
        )
    
    @property
    def class_names(self) -> List[str]:
        """Get list of class names."""
        if hasattr(self, 'train_dataset'):
            return self.train_dataset.class_names
        return [] 