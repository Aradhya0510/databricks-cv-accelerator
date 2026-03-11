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
- `universal_segmentation` - Universal/panoptic segmentation tasks

### Model Types:
- `detr` - DETR-based models
- `yolos` - YOLOS models
- `vit` - Vision Transformer models
- `resnet` - ResNet models
- `segformer` - SegFormer models
- `mask2former` - Mask2Former models
- `mitb0` - MiT-B0 models

## Configuration Structure

All config files follow this standardized structure:

```yaml
# Model Configuration
model:
  model_name: "model/name"
  task_type: "task_name"
  num_classes: 80
  pretrained: true
  learning_rate: 1e-4
  weight_decay: 1e-4
  scheduler: cosine
  scheduler_params:
    T_max: 300
    eta_min: 1e-6
  # Task-specific parameters...

# Data Configuration
data:
  train_data_path: "/path/to/train"
  train_annotation_file: "/path/to/train/annotations.json"
  val_data_path: "/path/to/val"
  val_annotation_file: "/path/to/val/annotations.json"
  batch_size: 16
  num_workers: 4
  model_name: "model/name"
  image_size: [800, 800]
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
  augment: true
  augmentations:
    horizontal_flip: true

# Training Configuration
training:
  max_epochs: 50
  early_stopping_patience: 20
  monitor_metric: "val_map"
  monitor_mode: "max"
  checkpoint_dir: "/path/to/checkpoints"
  volume_checkpoint_dir: "/path/to/volume_checkpoints"
  save_top_k: 3
  log_every_n_steps: 50
  use_gpu: true

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

Note: Learning rate, weight decay, scheduler, and scheduler_params are defined in the `model` section only. The `training` section focuses on training loop control (epochs, early stopping, checkpointing). Multi-GPU DDP is handled automatically by the Trainer based on environment detection.

## Usage

Load a configuration file:

```python
from src.config import load_config

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
- `segmentation_type`: "semantic", "instance", or "universal"

## Environment Variables

Replace placeholder paths with your actual Unity Catalog paths:
- `<catalog>`: Your Unity Catalog catalog name
- `<schema>`: Your Unity Catalog schema name
- `<volume>`: Your Unity Catalog volume name
- `<path>`: Your project path within the volume
