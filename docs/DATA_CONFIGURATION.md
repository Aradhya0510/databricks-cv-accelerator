# Data Configuration Guide

This document explains how to configure data paths for different dataset splits (train/validation/test) in the Databricks Computer Vision reference implementation.

## Overview

The data modules require separate paths for train/validation/test splits to ensure proper dataset separation and prevent data leakage.

## Configuration Format

### Object Detection, Segmentation Tasks

For tasks that require annotation files (detection, semantic segmentation, instance segmentation, panoptic segmentation):

```yaml
data:
  # Separate paths for train/val/test splits
  train_data_path: "/Volumes/<catalog>/<schema>/<volume>/<path>/data/train2017/"
  train_annotation_file: "/Volumes/<catalog>/<schema>/<volume>/<path>/data/instances_train2017.json"
  val_data_path: "/Volumes/<catalog>/<schema>/<volume>/<path>/data/val2017/"
  val_annotation_file: "/Volumes/<catalog>/<schema>/<volume>/<path>/data/instances_val2017.json"
  test_data_path: "/Volumes/<catalog>/<schema>/<volume>/<path>/data/test2017/"
  test_annotation_file: "/Volumes/<catalog>/<schema>/<volume>/<path>/data/instances_test2017.json"
  
  # Other parameters
  batch_size: 16
  num_workers: 4
  model_name: "facebook/detr-resnet-50"
  image_size: [800, 800]
```

### Classification Tasks

For classification tasks that use folder-based organization:

```yaml
data:
  # Separate paths for train/val/test splits
  train_data_path: "/Volumes/<catalog>/<schema>/<volume>/<path>/data/train/"
  val_data_path: "/Volumes/<catalog>/<schema>/<volume>/<path>/data/val/"
  test_data_path: "/Volumes/<catalog>/<schema>/<volume>/<path>/data/test/"
  
  # Other parameters
  batch_size: 32
  num_workers: 4
  model_name: "microsoft/resnet-50"
  image_size: 224
```

## Configuration Classes

### DetectionDataConfig
```python
@dataclass
class DetectionDataConfig:
    # Required paths
    train_data_path: str
    train_annotation_file: str
    val_data_path: str
    val_annotation_file: str
    
    # Optional test paths
    test_data_path: Optional[str] = None
    test_annotation_file: Optional[str] = None
    
    # Other parameters
    batch_size: int = 8
    num_workers: int = 4
    model_name: Optional[str] = None
```

### ClassificationDataConfig
```python
@dataclass
class ClassificationDataConfig:
    # Required paths
    train_data_path: str
    val_data_path: str
    
    # Optional test path
    test_data_path: Optional[str] = None
    
    # Other parameters
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    model_name: Optional[str] = None
```

## Usage Examples

### Python Configuration

```python
# Configuration with separate train/val/test paths
config = {
    "train_data_path": "/Volumes/<catalog>/<schema>/<volume>/<path>/data/train2017/",
    "train_annotation_file": "/Volumes/<catalog>/<schema>/<volume>/<path>/data/instances_train2017.json",
    "val_data_path": "/Volumes/<catalog>/<schema>/<volume>/<path>/data/val2017/",
    "val_annotation_file": "/Volumes/<catalog>/<schema>/<volume>/<path>/data/instances_val2017.json",
    "test_data_path": "/Volumes/<catalog>/<schema>/<volume>/<path>/data/test2017/",
    "test_annotation_file": "/Volumes/<catalog>/<schema>/<volume>/<path>/data/instances_test2017.json",
    "batch_size": 16,
    "model_name": "facebook/detr-resnet-50"
}

# Create data module
data_module = DetectionDataModule(config)
data_module.setup()

# Access datasets
print(f"Train: {len(data_module.train_dataset)} samples")
print(f"Val: {len(data_module.val_dataset)} samples")
print(f"Test: {len(data_module.test_dataset)} samples")
```

### YAML Configuration

```yaml
# config.yaml
data:
  train_data_path: "/Volumes/<catalog>/<schema>/<volume>/<path>/data/train2017/"
  train_annotation_file: "/Volumes/<catalog>/<schema>/<volume>/<path>/data/instances_train2017.json"
  val_data_path: "/Volumes/<catalog>/<schema>/<volume>/<path>/data/val2017/"
  val_annotation_file: "/Volumes/<catalog>/<schema>/<volume>/<path>/data/instances_val2017.json"
  test_data_path: "/Volumes/<catalog>/<schema>/<volume>/<path>/data/test2017/"
  test_annotation_file: "/Volumes/<catalog>/<schema>/<volume>/<path>/data/instances_test2017.json"
  batch_size: 16
  model_name: "facebook/detr-resnet-50"

training:
  max_epochs: 50
  learning_rate: 1e-4
```

```python
# Load configuration
import yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Create data module
data_module = DetectionDataModule(config["data"])
data_module.setup()
```

## Best Practices

1. **Always use separate train/val/test paths** for proper dataset splitting
2. **Test data is optional** - if not provided, validation data will be used for testing
3. **Use descriptive path names** that clearly indicate the dataset split
4. **Keep annotation files organized** with clear naming conventions
5. **Validate paths exist** before training to avoid runtime errors

## Dataset Organization

### Recommended Directory Structure

```
/Volumes/<catalog>/<schema>/<volume>/<path>/
├── data/
│   ├── train2017/
│   │   ├── 000000000001.jpg
│   │   ├── 000000000002.jpg
│   │   └── ...
│   ├── val2017/
│   │   ├── 000000000139.jpg
│   │   ├── 000000000285.jpg
│   │   └── ...
│   ├── test2017/
│   │   ├── 000000000001.jpg
│   │   ├── 000000000002.jpg
│   │   └── ...
│   └── annotations/
│       ├── instances_train2017.json
│       ├── instances_val2017.json
│       └── instances_test2017.json
├── configs/
├── checkpoints/
├── logs/
└── results/
```

### Classification Dataset Structure

```
/Volumes/<catalog>/<schema>/<volume>/<path>/
├── data/
│   ├── train/
│   │   ├── class1/
│   │   │   ├── image1.jpg
│   │   │   └── image2.jpg
│   │   └── class2/
│   │       ├── image3.jpg
│   │       └── image4.jpg
│   ├── val/
│   │   ├── class1/
│   │   └── class2/
│   └── test/
│       ├── class1/
│       └── class2/
```

## Troubleshooting

### Common Issues

1. **Path not found errors**: Ensure all specified paths exist
2. **Annotation file errors**: Verify annotation files are valid JSON
3. **Empty datasets**: Check that directories contain the expected data
4. **Memory issues**: Reduce batch size or number of workers

### Validation

```python
# Validate configuration
def validate_data_config(config):
    paths_to_check = [
        config.get("train_data_path"),
        config.get("val_data_path"),
        config.get("test_data_path")
    ]
    
    for path in paths_to_check:
        if path and not Path(path).exists():
            raise ValueError(f"Path does not exist: {path}")
    
    # Check annotation files for detection tasks
    if "train_annotation_file" in config:
        annotation_files = [
            config.get("train_annotation_file"),
            config.get("val_annotation_file"),
            config.get("test_annotation_file")
        ]
        
        for ann_file in annotation_files:
            if ann_file and not Path(ann_file).exists():
                raise ValueError(f"Annotation file does not exist: {ann_file}")
```

### Required vs Optional Fields

| Field | Required | Description |
|-------|----------|-------------|
| `train_data_path` | ✅ | Path to training data directory |
| `train_annotation_file` | ✅ | Path to training annotations (detection/segmentation only) |
| `val_data_path` | ✅ | Path to validation data directory |
| `val_annotation_file` | ✅ | Path to validation annotations (detection/segmentation only) |
| `test_data_path` | ❌ | Path to test data directory (optional) |
| `test_annotation_file` | ❌ | Path to test annotations (optional) |

**Note**: If test paths are not provided, the validation data will be used for testing. 