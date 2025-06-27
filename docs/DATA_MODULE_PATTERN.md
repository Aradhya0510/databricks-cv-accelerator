# Data Module Pattern

This document explains the consistent pattern that all data modules follow in the Databricks Computer Vision reference implementation.

## Overview

All data modules now follow the same clean, consistent pattern established by the `DetectionDataModule`. This pattern ensures:

1. **Separation of concerns**: Data loading is separate from data transformation
2. **Adapter-based transforms**: All transforms are handled through adapters
3. **Consistent interface**: All data modules have the same structure
4. **Minimal complexity**: Simple, readable code

## Pattern Structure

### 1. Configuration Class

```python
@dataclass
class TaskDataConfig:
    """Configuration for task data module."""
    # Separate paths for train/val/test splits
    train_data_path: str
    train_annotation_file: str  # Required for detection/segmentation tasks
    val_data_path: str
    val_annotation_file: str    # Required for detection/segmentation tasks
    test_data_path: Optional[str] = None
    test_annotation_file: Optional[str] = None  # Optional for detection/segmentation tasks
    
    # Common parameters
    batch_size: int = 8
    num_workers: int = 4
    model_name: Optional[str] = None
    image_size: Optional[List[int]] = None
```

### 2. Dataset Class

```python
class TaskDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: str,
        annotation_file: str,  # Optional for classification
        transform: Optional[Any] = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        # ... dataset initialization ...
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # ... load image and annotations ...
        
        # Apply transforms
        if self.transform:
            image, target = self.transform(image, target)
        
        return {
            "pixel_values": image,
            "labels": target
        }
```

### 3. Data Module Class

```python
class TaskDataModule(pl.LightningDataModule):
    def __init__(self, config: Union[Dict[str, Any], TaskDataConfig]):
        super().__init__()
        if isinstance(config, dict):
            config = TaskDataConfig(**config)
        self.config = config
        self.adapter = None  # Will be set after initialization
    
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_dataset = TaskDataset(
                root_dir=self.config.train_data_path,
                annotation_file=self.config.train_annotation_file,
                transform=None  # Transform will be set by the adapter
            )
            
            self.val_dataset = TaskDataset(
                root_dir=self.config.val_data_path,
                annotation_file=self.config.val_annotation_file,
                transform=None  # Transform will be set by the adapter
            )
        
        if stage == 'test':
            # Handle optional test data
            if self.config.test_data_path is not None:
                self.test_dataset = TaskDataset(
                    root_dir=self.config.test_data_path,
                    annotation_file=self.config.test_annotation_file,
                    transform=None  # Transform will be set by the adapter
                )
            else:
                # Use validation data for testing
                self.test_dataset = TaskDataset(
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
            collate_fn=self._collate_fn  # Optional, task-specific
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn  # Optional, task-specific
        )
    
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn  # Optional, task-specific
        )
    
    @property
    def class_names(self) -> List[str]:
        """Get list of class names."""
        if hasattr(self, 'train_dataset'):
            return self.train_dataset.class_names
        return []
```

## Key Principles

### 1. Adapter-Based Transforms

- **No built-in transforms**: Data modules don't define their own transforms
- **Adapter injection**: Transforms are injected via adapters after initialization
- **Consistent interface**: All adapters follow the same `(image, target) -> (image, target)` pattern

### 2. Separation of Concerns

- **Data loading**: Handled by dataset classes
- **Data transformation**: Handled by adapters
- **Configuration**: Handled by config classes
- **Data module**: Orchestrates the components

### 3. Consistent Interface

- **Same setup pattern**: All data modules follow identical setup logic
- **Same dataloader pattern**: Consistent dataloader configuration
- **Same adapter pattern**: All use the same adapter injection mechanism

## Usage Pattern

### 1. Create Data Module

```python
# Create configuration
config = {
    "train_data_path": "/path/to/train/",
    "train_annotation_file": "/path/to/train_annotations.json",
    "val_data_path": "/path/to/val/",
    "val_annotation_file": "/path/to/val_annotations.json",
    "batch_size": 16,
    "model_name": "facebook/detr-resnet-50"
}

# Create data module
data_module = DetectionDataModule(config)
```

### 2. Set Adapter

```python
# Create and set adapter
adapter = DetectionAdapter(model_name="facebook/detr-resnet-50", image_size=800)
data_module.adapter = adapter
```

### 3. Setup and Use

```python
# Setup datasets
data_module.setup()

# Get dataloaders
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
test_loader = data_module.test_dataloader()
```

## Benefits

### 1. **Simplicity**
- Clean, readable code
- No complex conditional logic
- Minimal configuration

### 2. **Consistency**
- All data modules work the same way
- Predictable behavior
- Easy to understand and maintain

### 3. **Flexibility**
- Easy to swap adapters
- Support for different model types
- Extensible architecture

### 4. **Maintainability**
- Single pattern to maintain
- Clear separation of responsibilities
- Easy to debug and test

## Task-Specific Variations

### Classification
- **No annotation files**: Uses folder-based organization
- **Simple dataset**: No complex annotation parsing
- **No collate function**: Standard PyTorch collation

### Detection/Segmentation
- **COCO annotations**: Uses pycocotools for parsing
- **Complex targets**: Bounding boxes, masks, etc.
- **Custom collate**: Handles variable-length targets

### Instance/Panoptic Segmentation
- **Complex masks**: Multiple instance masks per image
- **Custom collate**: Handles variable number of instances
- **Specialized processing**: Instance ID management

## Migration Guide

If you have existing data modules that don't follow this pattern:

1. **Remove built-in transforms**: Delete Albumentations transforms
2. **Simplify dataset constructor**: Remove transform parameters
3. **Add adapter injection**: Add the adapter setting pattern
4. **Update setup method**: Use the consistent setup pattern
5. **Clean up imports**: Remove unused dependencies

This pattern ensures all data modules are consistent, maintainable, and follow best practices for separation of concerns. 