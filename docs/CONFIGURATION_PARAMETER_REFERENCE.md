# Configuration Parameter Reference

**Complete parameter guide for the Databricks CV Reference Framework**

---

## Overview

This document provides a complete reference for all configuration parameters used in the framework. Each parameter is documented with its type, default value, purpose, and usage context.

---

## Configuration Structure

```yaml
model:           # Model architecture and training hyperparameters
data:            # Dataset paths and data loading parameters  
training:        # Training control and distributed training settings
mlflow:          # MLflow experiment tracking configuration
output:          # Output paths and visualization settings
```

---

## Model Configuration

Controls model architecture, task type, and training hyperparameters.

### Core Model Parameters

#### `model_name` (string, **required**)
- **Description:** Hugging Face model identifier
- **Examples:**
  - `"facebook/detr-resnet-50"` - Detection
  - `"microsoft/resnet-50"` - Classification
  - `"nvidia/mit-b0"` - Segmentation
- **Used by:** ModelClass, DataModule, Adapters, UnifiedTrainer
- **Notes:** Must be valid HF model name for the task

#### `task_type` (string, **required**)
- **Description:** Computer vision task type
- **Valid values:**
  - `"classification"`
  - `"detection"`
  - `"semantic_segmentation"`
  - `"instance_segmentation"`
  - `"universal_segmentation"`
- **Used by:** UnifiedTrainer, task modules
- **Notes:** Determines which task pipeline to use

#### `num_classes` (integer, **required**)
- **Description:** Number of output classes
- **Examples:**
  - `1000` - ImageNet classification
  - `80` - COCO detection
  - `19` - Cityscapes segmentation
- **Used by:** ModelClass
- **Notes:** Must match dataset annotations

#### `pretrained` (boolean, default: `true`)
- **Description:** Whether to load pretrained weights
- **Used by:** ModelClass
- **Notes:** Recommended for transfer learning

### Training Hyperparameters (in model section)

#### `learning_rate` (float, default: `1e-4`)
- **Description:** Initial learning rate for optimizer
- **Range:** Typically `1e-5` to `1e-3`
- **Used by:** ModelClass (optimizer config)
- **Notes:** Also duplicated in training section (redundant)

#### `weight_decay` (float, default: `1e-4`)
- **Description:** L2 regularization coefficient
- **Range:** Typically `1e-5` to `1e-2`
- **Used by:** ModelClass (optimizer config)

#### `scheduler` (string, default: `"cosine"`)
- **Description:** Learning rate scheduler type
- **Valid values:** `"cosine"`, `"linear"`, `"step"`, `"exponential"`
- **Used by:** ModelClass

#### `scheduler_params` (dict, optional)
- **Description:** Parameters for the scheduler
- **Example:**
  ```yaml
  scheduler_params:
    T_max: 100
    eta_min: 1e-6
  ```
- **Used by:** ModelClass

#### `epochs` (integer, default: `100`)
- **Description:** **DEPRECATED** - Use `training.max_epochs` instead
- **Notes:** Legacy parameter, should not be used

### Task-Specific Parameters

#### Classification

##### `dropout` (float, default: `0.2`)
- **Description:** Dropout rate for classification head
- **Range:** `0.0` to `0.5`
- **Used by:** ClassificationModel

##### `mixup_alpha` (float, default: `0.2`)
- **Description:** Mixup augmentation strength
- **Range:** `0.0` to `1.0`
- **Used by:** ClassificationModel

#### Detection

##### `confidence_threshold` (float, default: `0.5`)
- **Description:** Minimum confidence for detection
- **Range:** `0.0` to `1.0`
- **Used by:** DetectionModel

##### `iou_threshold` (float, default: `0.5`)
- **Description:** IOU threshold for NMS
- **Range:** `0.0` to `1.0`
- **Used by:** DetectionModel

##### `max_detections` (integer, default: `100`)
- **Description:** Maximum detections per image
- **Used by:** DetectionModel

#### Segmentation

##### `segmentation_type` (string, optional)
- **Description:** Type of segmentation
- **Valid values:** `"semantic"`, `"instance"`, `"universal"`
- **Used by:** Segmentation models

##### `aux_loss_weight` (float, default: `0.4`)
- **Description:** Weight for auxiliary losses
- **Range:** `0.0` to `1.0`
- **Used by:** Segmentation models

##### `mask_threshold` (float, default: `0.5`)
- **Description:** Threshold for mask binarization
- **Range:** `0.0` to `1.0`
- **Used by:** Segmentation models

---

## Data Configuration

Controls dataset paths, data loading, and augmentations.

### Dataset Paths

#### `train_data_path` (string, **required**)
- **Description:** Path to training data directory
- **Format:** Volume path like `/Volumes/<catalog>/<schema>/<volume>/data/train`
- **Used by:** DataModule

#### `val_data_path` (string, **required**)
- **Description:** Path to validation data directory
- **Used by:** DataModule

#### `test_data_path` (string, optional)
- **Description:** Path to test data directory
- **Used by:** DataModule (for evaluation)

#### `train_annotation_file` (string, optional for classification, required for detection/segmentation)
- **Description:** Path to training annotations (COCO JSON format)
- **Example:** `/Volumes/<catalog>/<schema>/<volume>/data/train/annotations.json`
- **Used by:** DataModule

#### `val_annotation_file` (string, optional for classification, required for detection/segmentation)
- **Description:** Path to validation annotations
- **Used by:** DataModule

#### `test_annotation_file` (string, optional)
- **Description:** Path to test annotations
- **Used by:** DataModule

### Data Loading Parameters

#### `batch_size` (integer, default: `16`)
- **Description:** Batch size for training
- **Recommendations:**
  - Classification: 32-64
  - Detection: 8-16
  - Segmentation: 4-8
- **Used by:** DataModule
- **Notes:** Adjust based on GPU memory

#### `num_workers` (integer, default: `4`)
- **Description:** Number of DataLoader worker processes **per GPU**
- **Used by:** DataModule
- **Notes:** NOT the same as `training.num_workers` (distributed workers)
- **Recommendations:** 2-8 depending on CPU cores

#### `model_name` (string, required)
- **Description:** Model name for adapter initialization
- **Used by:** DataModule (to get correct adapter)
- **Notes:** Should match `model.model_name`

### Image Processing

#### `image_size` (list or integer, default: `[512, 512]`)
- **Description:** Target image dimensions [height, width] or single value
- **Examples:**
  - `[224, 224]` - Classification
  - `[800, 800]` - Detection
  - `[512, 512]` - Segmentation
- **Used by:** DataModule, Adapters
- **Format:** List `[height, width]` or single integer

#### `normalize_mean` (tuple, default: `[0.485, 0.456, 0.406]`)
- **Description:** Mean values for normalization (ImageNet stats)
- **Used by:** DataModule

#### `normalize_std` (tuple, default: `[0.229, 0.224, 0.225]`)
- **Description:** Std values for normalization (ImageNet stats)
- **Used by:** DataModule

### Augmentation

#### `augment` (boolean, default: `true`)
- **Description:** Enable data augmentation
- **Used by:** DataModule

#### `augmentations` (dict, optional)
- **Description:** Augmentation configuration
- **Example:**
  ```yaml
  augmentations:
    horizontal_flip: true
    vertical_flip: false
    rotation: 15
    color_jitter:
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
      hue: 0.1
    random_crop: true
    random_resize: [0.8, 1.2]
  ```
- **Used by:** DataModule

---

## Training Configuration

Controls training execution, checkpointing, and distributed training.

### Basic Training Parameters

#### `max_epochs` (integer, default: `100`)
- **Description:** Maximum number of training epochs
- **Used by:** UnifiedTrainer
- **Notes:** Replaces deprecated `model.epochs`

#### `learning_rate` (float, default: `1e-4`)
- **Description:** **REDUNDANT** - Already in model section
- **Notes:** Currently duplicated for backward compatibility

#### `weight_decay` (float, default: `1e-4`)
- **Description:** **REDUNDANT** - Already in model section
- **Notes:** Currently duplicated for backward compatibility

### Early Stopping

#### `early_stopping_patience` (integer, default: `10`)
- **Description:** Epochs to wait before early stopping
- **Used by:** UnifiedTrainer (EarlyStopping callback)

#### `monitor_metric` (string, default: `"val_loss"`)
- **Description:** Metric to monitor for early stopping
- **Valid values:**
  - `"val_loss"` - Validation loss
  - `"val_map"` - Mean Average Precision (detection)
  - `"val_miou"` - Mean IOU (segmentation)
  - `"val_acc"` - Accuracy (classification)
- **Used by:** UnifiedTrainer

#### `monitor_mode` (string, default: `"min"`)
- **Description:** Whether to minimize or maximize metric
- **Valid values:** `"min"`, `"max"`
- **Used by:** UnifiedTrainer

### Checkpointing

#### `checkpoint_dir` (string, **required**)
- **Description:** Directory for local checkpoints
- **Format:** Volume path like `/Volumes/<catalog>/<schema>/<volume>/checkpoints`
- **Used by:** UnifiedTrainer
- **Notes:** Used during training

#### `volume_checkpoint_dir` (string, optional)
- **Description:** Directory for persistent volume checkpoints
- **Format:** Volume path like `/Volumes/<catalog>/<schema>/<volume>/volume_checkpoints`
- **Used by:** UnifiedTrainer (VolumeCheckpoint callback)
- **Notes:** For long-term storage, survives cluster termination

#### `save_top_k` (integer, default: `3`)
- **Description:** Number of best checkpoints to keep
- **Used by:** UnifiedTrainer (ModelCheckpoint callback)

### Logging

#### `log_every_n_steps` (integer, default: `50`)
- **Description:** Frequency of metric logging
- **Used by:** UnifiedTrainer

### Distributed Training

#### `distributed` (boolean, default: `false`)
- **Description:** Enable distributed training
- **Used by:** UnifiedTrainer
- **Notes:** Automatically enabled in multi-GPU jobs

#### `use_ray` (boolean, default: `false`)
- **Description:** Use Ray for multi-node training (vs Databricks DDP)
- **Valid values:**
  - `false` - Use Databricks DDP (single-node multi-GPU)
  - `true` - Use Ray (multi-node)
- **Used by:** UnifiedTrainer

#### `num_workers` (integer, default: `1`)
- **Description:** Number of distributed training workers (GPUs/nodes)
- **Used by:** UnifiedTrainer (for Ray training)
- **Notes:** **DIFFERENT** from `data.num_workers` (DataLoader processes)!

#### `use_gpu` (boolean, default: `true`)
- **Description:** Whether to use GPUs
- **Used by:** UnifiedTrainer

#### `resources_per_worker` (dict, default: `{CPU: 4, GPU: 1}`)
- **Description:** Resources to allocate per Ray worker
- **Example:**
  ```yaml
  resources_per_worker:
    CPU: 4
    GPU: 1
  ```
- **Used by:** UnifiedTrainer (Ray configuration)

#### `master_port` (integer, optional)
- **Description:** Port for DDP communication
- **Used by:** UnifiedTrainer (DDP setup)
- **Notes:** Auto-selected if not specified

### Strategy Overrides

#### `preferred_strategy` (string, optional)
- **Description:** Override automatic strategy selection
- **Valid values:**
  - `"ddp"` - Distributed Data Parallel
  - `"auto"` - Auto-select based on environment
  - `"ddp_notebook"` - DDP in notebook (experimental)
- **Used by:** UnifiedTrainer (strategy selection)
- **Notes:** Typically set by job scripts, not YAML configs

#### `preferred_devices` (string or integer, optional)
- **Description:** Override automatic device selection
- **Valid values:**
  - `"auto"` - Auto-detect GPUs
  - `1` - Use 1 GPU
  - `4` - Use 4 GPUs
- **Used by:** UnifiedTrainer (device selection)
- **Notes:** Typically set by job scripts, not YAML configs

---

## MLflow Configuration

Controls MLflow experiment tracking and logging.

### Experiment Tracking

#### `experiment_name` (string, **required**)
- **Description:** MLflow experiment name (absolute workspace path)
- **Format:** Must start with `/`
- **Examples:**
  - `/Users/email@databricks.com/cv_detection_detr`
  - `/Shared/cv_experiments/classification`
  - `/Workspace/Users/email@databricks.com/experiments/segmentation`
- **Used by:** Jobs, Notebooks (for create_databricks_logger)
- **Notes:** ⚠️ Must be in `mlflow` section, NOT `training` section!

#### `run_name` (string, default: `"default_run"`)
- **Description:** MLflow run name
- **Examples:**
  - `"detr_resnet50"`
  - `"resnet50_imagenet"`
- **Used by:** Jobs, Notebooks (optional)

#### `log_model` (boolean, default: `true`)
- **Description:** Whether to log model to MLflow
- **Used by:** Jobs, Notebooks

#### `tags` (dict, optional)
- **Description:** Tags to add to MLflow run
- **Example:**
  ```yaml
  tags:
    framework: "lightning"
    model: "detr"
    dataset: "coco"
  ```
- **Used by:** Jobs, Notebooks

---

## Output Configuration

Controls output paths and visualization settings.

### Output Paths

#### `results_dir` (string, default: `/Volumes/.../results`)
- **Description:** Directory for results and predictions
- **Used by:** Evaluation scripts

#### `save_predictions` (boolean, default: `true`)
- **Description:** Whether to save predictions
- **Used by:** Evaluation scripts

### Visualization

#### `visualization` (dict, optional)
- **Description:** Visualization settings
- **Example:**
  ```yaml
  visualization:
    save_images: true
    confidence_threshold: 0.5
    max_boxes: 20
    max_images: 10
  ```
- **Used by:** Evaluation and inference scripts

---

## Complete Example Configs

### Classification Example

```yaml
model:
  model_name: "microsoft/resnet-50"
  task_type: "classification"
  num_classes: 1000
  pretrained: true
  dropout: 0.2
  mixup_alpha: 0.2
  learning_rate: 1e-4
  weight_decay: 1e-4
  scheduler: "cosine"
  scheduler_params:
    T_max: 100
    eta_min: 1e-6

data:
  train_data_path: "/Volumes/catalog/schema/volume/data/train"
  val_data_path: "/Volumes/catalog/schema/volume/data/val"
  batch_size: 32
  num_workers: 4
  model_name: "microsoft/resnet-50"
  image_size: [224, 224]
  augment: true

training:
  max_epochs: 100
  early_stopping_patience: 10
  monitor_metric: "val_loss"
  monitor_mode: "min"
  checkpoint_dir: "/Volumes/catalog/schema/volume/checkpoints"
  volume_checkpoint_dir: "/Volumes/catalog/schema/volume/volume_checkpoints"
  save_top_k: 3
  log_every_n_steps: 50
  distributed: false
  use_ray: false
  num_workers: 1
  use_gpu: true
  resources_per_worker:
    CPU: 4
    GPU: 1
  master_port: null
  preferred_strategy: null
  preferred_devices: null

mlflow:
  experiment_name: "/Users/email@databricks.com/cv_classification_resnet"
  run_name: "resnet50"
  log_model: true
  tags:
    framework: "lightning"
    model: "resnet"
    dataset: "imagenet"

output:
  results_dir: "/Volumes/catalog/schema/volume/results"
  save_predictions: true
```

### Detection Example

```yaml
model:
  model_name: "facebook/detr-resnet-50"
  task_type: "detection"
  num_classes: 80
  pretrained: true
  confidence_threshold: 0.5
  iou_threshold: 0.5
  max_detections: 100
  learning_rate: 1e-4
  weight_decay: 1e-4
  scheduler: "cosine"

data:
  train_data_path: "/Volumes/catalog/schema/volume/data/train"
  train_annotation_file: "/Volumes/catalog/schema/volume/data/train/annotations.json"
  val_data_path: "/Volumes/catalog/schema/volume/data/val"
  val_annotation_file: "/Volumes/catalog/schema/volume/data/val/annotations.json"
  batch_size: 16
  num_workers: 4
  model_name: "facebook/detr-resnet-50"
  image_size: [800, 800]

training:
  max_epochs: 50
  early_stopping_patience: 20
  monitor_metric: "val_map"
  monitor_mode: "max"
  checkpoint_dir: "/Volumes/catalog/schema/volume/checkpoints"
  volume_checkpoint_dir: "/Volumes/catalog/schema/volume/volume_checkpoints"
  save_top_k: 3
  log_every_n_steps: 50
  distributed: false
  use_ray: false
  num_workers: 1
  use_gpu: true
  master_port: null

mlflow:
  experiment_name: "/Users/email@databricks.com/cv_detection_detr"
  run_name: "detr_resnet50"
  log_model: true
  tags:
    framework: "lightning"
    model: "detr"
    dataset: "coco"

output:
  results_dir: "/Volumes/catalog/schema/volume/results"
  save_predictions: true
```

---

## Common Pitfalls

### 1. `experiment_name` in Wrong Section ❌

**Wrong:**
```yaml
training:
  experiment_name: "/Users/email@databricks.com/experiment"
```

**Correct:**
```yaml
mlflow:
  experiment_name: "/Users/email@databricks.com/experiment"
```

### 2. Confusing Two `num_workers` ❌

**Different meanings:**
```yaml
data:
  num_workers: 4  # DataLoader worker processes per GPU

training:
  num_workers: 1  # Number of distributed training workers (GPUs/nodes)
```

### 3. Relative Paths Instead of Absolute ❌

**Wrong:**
```yaml
mlflow:
  experiment_name: "cv_detection"  # Relative
```

**Correct:**
```yaml
mlflow:
  experiment_name: "/Users/email@databricks.com/cv_detection"  # Absolute
```

### 4. Using Deprecated `model.epochs` ❌

**Deprecated:**
```yaml
model:
  epochs: 100  # Old
```

**Correct:**
```yaml
training:
  max_epochs: 100  # New
```

---

## Parameter Migration Guide

If you have old configs, follow this migration:

### Step 1: Move `experiment_name`

```yaml
# OLD
training:
  experiment_name: "my_experiment"

# NEW
mlflow:
  experiment_name: "/Users/email@databricks.com/my_experiment"
```

### Step 2: Replace `model.epochs`

```yaml
# OLD
model:
  epochs: 50

# NEW
training:
  max_epochs: 50
```

### Step 3: Add New Parameters

Add to training section:
```yaml
training:
  # ... existing params ...
  volume_checkpoint_dir: "/Volumes/.../volume_checkpoints"
  num_workers: 1
  master_port: null
  preferred_strategy: null
  preferred_devices: null
```

---

## Validation

Use the built-in validator to check your configs:

```bash
cd /Workspace/Repos/your-repo/Databricks_CV_ref
python configs/validate_configs.py
```

Or in Python:

```python
from databricks_cv_accelerator.config import load_config
from databricks_cv_accelerator.utils.config_validator import validate_config_for_simplified_mlflow

config = load_config("configs/detection_detr_config.yaml")
validated_config = validate_config_for_simplified_mlflow(config)
```

---

## Summary

**Total Parameters:** 50+

**By Section:**
- Model: 20+ parameters
- Data: 15+ parameters
- Training: 15+ parameters
- MLflow: 4 parameters
- Output: 3+ parameters

**Critical Parameters:** 
- `model.model_name` ✅
- `model.task_type` ✅
- `model.num_classes` ✅
- `data.train_data_path` ✅
- `data.val_data_path` ✅
- `training.checkpoint_dir` ✅
- `mlflow.experiment_name` ✅

**All parameters documented with:**
- Type
- Default value (if applicable)
- Description
- Usage context
- Examples
- Common pitfalls

---

**Last Updated:** October 18, 2025  
**Version:** 1.0  
**Status:** Complete and Validated

