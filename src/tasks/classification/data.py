from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
import lightning as pl
from PIL import Image
import numpy as np
from pathlib import Path
from transformers import AutoFeatureExtractor
from .adapters import get_input_adapter

@dataclass
class ClassificationDataConfig:
    """Configuration for classification data module."""
    # Separate paths for train/val/test splits
    train_data_path: str
    val_data_path: str
    test_data_path: Optional[str] = None
    
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
    augmentations: Optional[Dict[str, Any]] = None

class ClassificationDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        transform: Optional[Any] = None,
    ):
        self.data_path = Path(data_path)
        self.transform = transform
        
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
        
        # Apply transforms
        if self.transform:
            image, target = self.transform(image, target)
        else:
            # Convert PIL image to tensor if no transform is provided
            import torchvision.transforms.functional as F
            image = F.to_tensor(image)
        
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
        self.save_hyperparameters(config.__dict__)
        self.adapter = None  # Will be set after initialization
    
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = ClassificationDataset(
                self.config.train_data_path,
                transform=None  # Transform will be set by the adapter
            )
            self.val_dataset = ClassificationDataset(
                self.config.val_data_path,
                transform=None  # Transform will be set by the adapter
            )
        
        if stage == "test" or stage is None:
            # Use test data if available, otherwise use validation data
            test_path = self.config.test_data_path if self.config.test_data_path is not None else self.config.val_data_path
            self.test_dataset = ClassificationDataset(
                test_path,
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
            num_workers=self.config.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True if self.config.num_workers > 0 else False
        )
    
    @property
    def class_names(self) -> List[str]:
        """Get list of class names."""
        if hasattr(self, 'train_dataset'):
            return self.train_dataset.class_names
        return [] 