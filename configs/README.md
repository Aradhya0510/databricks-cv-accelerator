# Configuration Files

This directory contains consolidated configuration files for all computer vision tasks. These configs work for both **standard notebooks** and **serverless GPU notebooks**.

## Config Structure

### Sections

#### 1. `model:` - Model Configuration
Defines the model architecture and task-specific parameters.

```yaml
model:
  model_name: facebook/detr-resnet-50  # HuggingFace model identifier
  task_type: detection                  # Task: detection, classification, segmentation
  num_classes: 80                       # Number of classes
  pretrained: true                      # Use pretrained weights
  image_size: 800                       # Input image size
  learning_rate: 1e-4                   # Learning rate
  weight_decay: 1e-4                    # Weight decay
```

#### 2. `data:` - Data Configuration
Specifies data paths, preprocessing, and augmentations.

```yaml
data:
  train_data_path: /Volumes/.../train2017/
  train_annotation_file: /Volumes/.../instances_train2017.json
  val_data_path: /Volumes/.../val2017/
  val_annotation_file: /Volumes/.../instances_val2017.json
  batch_size: 16
  num_workers: 4
  image_size: [800, 800]              # Can be int or [height, width]
  normalize_mean: [0.485, 0.456, 0.406]
  normalize_std: [0.229, 0.224, 0.225]
  augment: true
  augmentations:
    horizontal_flip: true
    rotation: 10
```

#### 3. `training:` - Training Configuration
**Used by `UnifiedTrainer`** - configures the training process.

```yaml
training:
  max_epochs: 50
  learning_rate: 1e-4
  weight_decay: 1e-4
  early_stopping_patience: 20
  monitor_metric: val_map
  monitor_mode: max
  checkpoint_dir: /Volumes/.../checkpoints
  save_top_k: 3
  log_every_n_steps: 50
  
  # Distributed training settings (used by UnifiedTrainer)
  distributed: false              # Set to true for distributed training
  use_ray: false                  # Set to true for Ray multi-node
  use_gpu: true
  num_workers: 1
  resources_per_worker:
    CPU: 4
    GPU: 1
  preferred_strategy: null        # Options: "ddp", "auto", "ddp_notebook"
  preferred_devices: null         # Options: "auto", 1, 2, 4, etc.
```

#### 4. `serverless:` - Serverless GPU Configuration (NEW)
**Notebook-level orchestration only** - NOT used by `UnifiedTrainer`.
These parameters are used by the `@distributed` decorator in serverless notebooks.

```yaml
serverless:
  enabled: false                  # Set to true to enable serverless GPU training
  gpu_type: A10                   # Options: A10, H100
  gpu_count: 4                    # Number of GPUs (H100 only supports 1)
```

**Important Notes:**
- These parameters are **read at the notebook level** by the `@distributed` decorator
- They are **NOT passed to `UnifiedTrainer`**
- Perfect for UI-based config generation in lakehouse apps

#### 5. `mlflow:` - MLflow Configuration
Defines experiment tracking settings.

```yaml
mlflow:
  experiment_name: /Users/<email>/cv_detection_detr
  run_name: detr_resnet50
  log_model: true
  tags:
    framework: lightning
    model: detr
    dataset: coco
```

#### 6. `output:` - Output Configuration
Specifies where to save results and visualizations.

```yaml
output:
  results_dir: /Volumes/.../results/detection
  save_predictions: true
  visualization:
    save_images: true
    confidence_threshold: 0.5
    max_boxes: 20
```

## Usage

### Standard Notebooks (Single/Multi-GPU)
```python
from databricks_cv_accelerator.config import load_config

config = load_config("configs/detection_detr_config.yaml")

# Set distributed=false for single GPU
config['training']['distributed'] = False

# Or set distributed=true for multi-GPU DDP
config['training']['distributed'] = True
```

### Serverless GPU Notebooks
```python
from databricks_cv_accelerator.config import load_config

config = load_config("configs/detection_detr_config.yaml")

# Enable serverless GPU
config['serverless']['enabled'] = True
config['serverless']['gpu_type'] = 'A10'
config['serverless']['gpu_count'] = 4

# Use in @distributed decorator
from serverless_gpu import distributed

@distributed(
    gpus=config['serverless']['gpu_count'],
    gpu_type=config['serverless']['gpu_type'],
    remote=True
)
def distributed_train(...):
    # Training code
    pass
```

## Migration from configs_serverless/

**Old structure (deprecated):**
```
configs/                    # Standard GPU configs
configs_serverless/         # Serverless GPU configs (slightly different)
```

**New structure (current):**
```
configs/                    # Unified configs for all scenarios
```

### Changes Made:
1. **Consolidated** `configs/` and `configs_serverless/` into single `configs/` directory
2. **Added** `serverless:` section for notebook-level orchestration
3. **Removed** serverless params from `training:` section (they weren't used by UnifiedTrainer anyway)
4. **Cleaner** separation of concerns: UnifiedTrainer only sees training params it actually uses

### Benefits:
- ✅ Single source of truth for configs
- ✅ Cleaner code (no redundant params in UnifiedTrainer)
- ✅ Easier UI generation for lakehouse apps
- ✅ Better documentation and maintainability

## Available Configs

- `classification_resnet_config.yaml` - ResNet image classification
- `classification_vit_config.yaml` - Vision Transformer classification
- `detection_detr_config.yaml` - DETR object detection
- `detection_yolos_config.yaml` - YOLOS object detection
- `semantic_segmentation_segformer_config.yaml` - SegFormer segmentation
- `semantic_segmentation_mitb0_config.yaml` - MiT-B0 segmentation
- `instance_segmentation_mask2former_config.yaml` - Mask2Former instance segmentation
- `universal_segmentation_mask2former_config.yaml` - Mask2Former universal segmentation

## For Lakehouse App UI

The `serverless:` section provides placeholders for UI-based config generation:

```python
# In your lakehouse app UI
serverless_config = {
    "enabled": st.checkbox("Enable Serverless GPU"),
    "gpu_type": st.selectbox("GPU Type", ["A10", "H100"]),
    "gpu_count": st.slider("GPU Count", 1, 8, 4)
}

# Write to config
config['serverless'] = serverless_config
```

This makes it easy to generate configs dynamically from user input.
