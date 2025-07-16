# Configuration Compatibility Guide for Simplified MLflow Integration

## Overview

This guide explains the configuration requirements and compatibility for the simplified MLflow integration approach. The new approach requires specific fields in your configuration files to work properly with the `UnifiedTrainer`.

## ✅ **Updated Configuration Structure**

### Required Fields for UnifiedTrainerConfig

The `UnifiedTrainerConfig` dataclass requires these fields:

```yaml
# These fields are required at the top level or can be mapped from nested sections
task: "detection"                    # From model.task_type
model_name: "facebook/detr-resnet-50"  # From model.model_name
max_epochs: 50                      # From training.max_epochs
log_every_n_steps: 50              # From training.log_every_n_steps
monitor_metric: "val_map"           # From training.monitor_metric
monitor_mode: "max"                 # From training.monitor_mode
early_stopping_patience: 20         # From training.early_stopping_patience
checkpoint_dir: "/path/to/checkpoints"  # From training.checkpoint_dir
distributed: false                  # From training.distributed

# Optional fields with defaults
volume_checkpoint_dir: "/path/to/volume_checkpoints"  # Optional
save_top_k: 3                       # Default: 3
```

### Complete Configuration Template

```yaml
# Model Configuration
model:
  model_name: "facebook/detr-resnet-50"
  task_type: "detection"
  num_classes: 80
  pretrained: true
  learning_rate: 1e-4
  weight_decay: 1e-4

# Data Configuration
data:
  train_data_path: "/Volumes/<catalog>/<schema>/<volume>/data/train/"
  train_annotation_file: "/Volumes/<catalog>/<schema>/<volume>/data/train/annotations.json"
  val_data_path: "/Volumes/<catalog>/<schema>/<volume>/data/val/"
  val_annotation_file: "/Volumes/<catalog>/<schema>/<volume>/data/val/annotations.json"
  batch_size: 16
  num_workers: 4
  image_size: [800, 800]

# Training Configuration
training:
  max_epochs: 50
  learning_rate: 1e-4
  weight_decay: 1e-4
  early_stopping_patience: 20
  monitor_metric: "val_map"
  monitor_mode: "max"
  checkpoint_dir: "/Volumes/<catalog>/<schema>/<volume>/checkpoints"
  volume_checkpoint_dir: "/Volumes/<catalog>/<schema>/<volume>/volume_checkpoints"
  save_top_k: 3
  log_every_n_steps: 50
  distributed: false

# Output Configuration
output:
  results_dir: "/Volumes/<catalog>/<schema>/<volume>/results"
  save_predictions: true
```

## 🔄 **Migration from Old Configuration**

### What Changed

1. **Removed MLflow Section**: The `mlflow` section is no longer needed
2. **Added Volume Checkpoint Directory**: New field for persistent storage
3. **Field Mapping**: The trainer now maps nested fields to top-level requirements

### Automatic Field Mapping

The configuration validator automatically maps these fields:

| UnifiedTrainerConfig Field | Config Path | Description |
|----------------------------|-------------|-------------|
| `task` | `model.task_type` | Task type (detection, classification, etc.) |
| `model_name` | `model.model_name` | Model identifier |
| `max_epochs` | `training.max_epochs` | Maximum training epochs |
| `log_every_n_steps` | `training.log_every_n_steps` | Logging frequency |
| `monitor_metric` | `training.monitor_metric` | Metric to monitor for checkpointing |
| `monitor_mode` | `training.monitor_mode` | Monitor mode (min/max) |
| `early_stopping_patience` | `training.early_stopping_patience` | Early stopping patience |
| `checkpoint_dir` | `training.checkpoint_dir` | Checkpoint directory |
| `distributed` | `training.distributed` | Distributed training flag |

## 🛠️ **Configuration Validation**

### Using the Configuration Validator

```python
from utils.config_validator import (
    validate_config_for_simplified_mlflow,
    print_config_compatibility_report
)

# Load your config
config = load_config("path/to/config.yaml")

# Print compatibility report
print_config_compatibility_report(config)

# Validate and update config
validated_config = validate_config_for_simplified_mlflow(config)
```

### Validation Features

1. **Field Mapping**: Automatically maps nested fields to top-level requirements
2. **Default Values**: Adds missing optional fields with sensible defaults
3. **Volume Directory Inference**: Automatically infers volume checkpoint directory
4. **Deprecation Warnings**: Warns about deprecated MLflow sections
5. **Type Validation**: Ensures all fields have correct types

## ❌ **Common Issues and Solutions**

### Issue 1: Missing Required Fields

**Error**: `Missing required field: task`

**Solution**: Ensure your config has `model.task_type` or add the field directly:

```yaml
# Option 1: Add to model section
model:
  task_type: "detection"

# Option 2: Add at top level
task: "detection"
```

### Issue 2: Deprecated MLflow Section

**Warning**: `Found deprecated 'mlflow' section - this is no longer needed`

**Solution**: Remove the MLflow section from your config:

```yaml
# Remove this section
# mlflow:
#   experiment_name: "detection_training"
#   run_name: "detr_resnet50"
#   log_model: true
```

### Issue 3: Missing Volume Checkpoint Directory

**Warning**: `No volume_checkpoint_dir specified`

**Solution**: Add the volume checkpoint directory:

```yaml
training:
  checkpoint_dir: "/Volumes/catalog/schema/volume/checkpoints"
  volume_checkpoint_dir: "/Volumes/catalog/schema/volume/volume_checkpoints"
```

## ✅ **Updated DETR Configuration**

The `detection_detr_config.yaml` has been updated with:

1. **Added `volume_checkpoint_dir`**: For persistent storage
2. **Removed `mlflow` section**: No longer needed
3. **Added `num_workers`**: For distributed training configuration
4. **Proper field structure**: All required fields are present

## 🔍 **Configuration Testing**

### Test Your Configuration

```python
# Test configuration compatibility
from utils.config_validator import get_config_compatibility_report

report = get_config_compatibility_report(config)
if report['compatible']:
    print("✅ Configuration is ready for simplified MLflow integration!")
else:
    print("❌ Configuration needs updates:")
    for issue in report['issues']:
        print(f"   - {issue}")
```

### Validation Checklist

- [ ] All required sections present (`model`, `data`, `training`)
- [ ] All required model fields present (`model_name`, `task_type`, `num_classes`)
- [ ] All required training fields present (`max_epochs`, `learning_rate`, `monitor_metric`, `monitor_mode`)
- [ ] `volume_checkpoint_dir` specified (optional but recommended)
- [ ] No deprecated `mlflow` section
- [ ] Proper data types for all fields

## 🚀 **Usage in Notebooks**

The training notebook now automatically validates your configuration:

```python
# In 02_model_training.py
from utils.config_validator import validate_config_for_simplified_mlflow, print_config_compatibility_report

# Print compatibility report
print_config_compatibility_report(config)

# Validate and update config
validated_config = validate_config_for_simplified_mlflow(config)
```

This ensures your configuration is compatible with the simplified MLflow integration before training begins.

## 📋 **Summary**

The updated configuration approach provides:

1. **Automatic Validation**: Configuration is validated before training
2. **Field Mapping**: Nested fields are automatically mapped to requirements
3. **Default Values**: Missing optional fields get sensible defaults
4. **Deprecation Warnings**: Clear guidance on what to remove/update
5. **Better Error Messages**: Specific guidance on what needs to be fixed

Your `detection_detr_config.yaml` is now fully compatible with the simplified MLflow integration approach! 