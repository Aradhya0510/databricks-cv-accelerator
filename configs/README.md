# Configuration Files

This directory contains standardized configuration files for all computer vision tasks in the Databricks CV framework.

## Naming Convention

Configuration files follow this naming pattern:
- `{task}_{model_type}_config.yaml`

### Task Types:
- `detection` - Object detection tasks
- `classification` - Image classification tasks  
- `semantic_segmentation` - Semantic segmentation tasks
- `instance_segmentation` - Instance segmentation tasks
- `panoptic_segmentation` - Panoptic segmentation tasks

### Model Types:
- `detr` - DETR-based models
- `yolos` - YOLOS models
- `vit` - Vision Transformer models
- `resnet` - ResNet models
- `segformer` - SegFormer models
- `mask2former` - Mask2Former models

## Configuration Structure

All config files follow this standardized structure:

```yaml
# Model Configuration
model:
  model_name: "model/name"
  task_type: "task_name"
  num_classes: 80
  pretrained: true
  # Task-specific parameters...

# Data Configuration  
data:
  # Dataset paths
  train_data_path: "/path/to/train"
  train_annotation_file: "/path/to/train/annotations.json"
  val_data_path: "/path/to/val"
  val_annotation_file: "/path/to/val/annotations.json"
  test_data_path: "/path/to/test"
  test_annotation_file: "/path/to/test/annotations.json"
  
  # Data loading parameters
  batch_size: 16
  num_workers: 4
  model_name: "model/name"  # For adapter initialization
  
  # Image processing
  image_size: [800, 800]
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
  
  # Augmentation
  augment: true
  augmentations:
    horizontal_flip: true
    # ... other augmentation settings

# Training Configuration
training:
  # Basic training
  max_epochs: 50
  learning_rate: 1e-4
  weight_decay: 1e-4
  scheduler: "cosine"
  scheduler_params:
    T_max: 300
    eta_min: 1e-6
  
  # Early stopping
  early_stopping_patience: 20
  monitor_metric: "val_map"
  monitor_mode: "max"
  
  # Checkpointing
  checkpoint_dir: "/path/to/checkpoints"
  save_top_k: 3
  
  # Logging
  log_every_n_steps: 50
  
  # Distributed training
  distributed: false
  use_gpu: true
  resources_per_worker:
    CPU: 4
    GPU: 1

# MLflow Configuration
mlflow:
  experiment_name: "task_training"
  run_name: "model_name"
  log_model: true
  tags:
    framework: "lightning"
    model: "model_type"
    dataset: "dataset_name"

# Output Configuration
output:
  results_dir: "/path/to/results"
  save_predictions: true
  visualization:
    save_images: true
    confidence_threshold: 0.5
    max_boxes: 20
```

## Usage

Load a configuration file:

```python
from databricks_cv_accelerator.config import load_config

config = load_config("configs/detection_detr_config.yaml")
```

## Task-Specific Parameters

### Detection Tasks
- `confidence_threshold`: Minimum confidence for detections
- `iou_threshold`: IoU threshold for NMS
- `max_detections`: Maximum number of detections per image

### Classification Tasks  
- `dropout`: Dropout rate for classification head
- `mixup_alpha`: Mixup augmentation alpha

### Segmentation Tasks
- `aux_loss_weight`: Weight for auxiliary loss
- `mask_threshold`: Threshold for mask predictions
- `segmentation_type`: "semantic", "instance", or "panoptic"

## Environment Variables

Replace placeholder paths with your actual Unity Catalog paths:
- `<catalog>`: Your Unity Catalog catalog name
- `<schema>`: Your Unity Catalog schema name  
- `<volume>`: Your Unity Catalog volume name
- `<path>`: Your project path within the volume 