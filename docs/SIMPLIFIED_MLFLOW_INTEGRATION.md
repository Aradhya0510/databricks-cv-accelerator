# Simplified MLflow Integration for Databricks Computer Vision Accelerator

## Overview

This document describes the simplified MLflow integration approach that removes redundant checkpoint logging and relies on the native integration between MLFlowLogger and Lightning's ModelCheckpoint callback.

## Key Changes

### 🔴 Removed Redundant Checkpoint Logging

The custom `MLflowCheckpointCallback` has been removed because it was manually trying to perform checkpoint logging, which directly conflicts with the modern, built-in functionality of Lightning's MLFlowLogger. This creates redundancy and unpredictable behavior.

### ✅ New Simplified Approach

The code has been significantly simplified and made more robust by:

1. **Removing custom checkpointing callback** - Rely on the native integration between MLFlowLogger and Lightning's ModelCheckpoint callback
2. **Using `log_model="all"`** - When you configure MLFlowLogger with `log_model="all"`, it automatically hooks into the Trainer's ModelCheckpoint callback and logs the saved checkpoints to MLflow artifacts
3. **Centralized logger creation** - Using `create_databricks_logger_for_task()` for consistent logger setup

## Key Issues Resolved

### ❌ Previous Issues:
- **Redundant Checkpoint Logging**: Custom callback conflicted with built-in functionality
- **Brittle best_model_path Logic**: Searching through trainer.callbacks was not robust
- **Inefficient Hooks**: Logging after every training epoch was inefficient
- **Scattered Parameter Logging**: Using global mlflow API instead of integrated approach

### ✅ New Benefits:
- **Automatic Checkpoint Logging**: MLFlowLogger handles all checkpoint logging automatically
- **Robust Integration**: Uses Lightning's native ModelCheckpoint callback
- **Efficient Logging**: Only logs when checkpoints are actually saved
- **Centralized Parameters**: LightningModule's `save_hyperparameters()` is automatically picked up

## Usage Examples

### Basic Usage

```python
from training.trainer import UnifiedTrainer, UnifiedTrainerConfig
from utils.logging import create_databricks_logger_for_task
from tasks.detection.model import DETRModel
from tasks.detection.data import DETRDataModule

# 1. Define Configuration
config = {
    "task": "detection",
    "model_name": "detr-resnet50",
    "max_epochs": 20,
    "log_every_n_steps": 50,
    "monitor_metric": "val_loss",
    "monitor_mode": "min",
    "early_stopping_patience": 5,
    "checkpoint_dir": "./checkpoints",
    "volume_checkpoint_dir": "/dbfs/mnt/my_volume/project_checkpoints",
    "save_top_k": 3,
    "distributed": False,
}
trainer_config = UnifiedTrainerConfig(**config)

# 2. ✨ Create the MLFlowLogger First ✨
mlf_logger = create_databricks_logger_for_task(
    task=trainer_config.task,
    model_name=trainer_config.model_name,
    run_name=f"{trainer_config.model_name}-run-{trainer_config.task}",
    log_model="all"  # Automatically log all ModelCheckpoint artifacts
)

# 3. Initialize Model and Data
model = DETRModel(num_classes=91, learning_rate=0.0001)
data_module = DETRDataModule(data_dir="/path/to/data", batch_size=4)

# 4. Initialize the UnifiedTrainer
unified_trainer = UnifiedTrainer(
    config=trainer_config,
    model=model,
    data_module=data_module,
    logger=mlf_logger
)

# 5. Start Training
result = unified_trainer.train()
```

### Advanced Usage with Custom Tags

```python
# Create logger with additional tags
mlf_logger = create_databricks_logger_for_task(
    task="detection",
    model_name="detr-resnet50",
    run_name="custom-run-name",
    log_model="all",
    additional_tags={
        'dataset': 'coco',
        'architecture': 'transformer',
        'experiment_type': 'baseline'
    }
)
```

## Updated Components

### 1. `utils/logging.py`

**New Functions:**
- `create_databricks_logger()` - Simplified logger creation
- `create_databricks_logger_for_task()` - Task-specific logger with common patterns
- `VolumeCheckpoint` - Lightweight callback for volume copying (no MLflow interaction)

**Removed:**
- Custom MLflowCheckpointCallback
- Redundant parameter logging functions

### 2. `training/trainer.py`

**Updated UnifiedTrainer:**
- Logger is now optional in constructor (auto-created if not provided)
- Simplified callback initialization
- Automatic integration with MLFlowLogger
- Better error handling and logging

### 3. `notebooks/02_model_training.py`

**Updated Training Flow:**
- Uses `create_databricks_logger_for_task()` for consistent setup
- Simplified monitoring functions
- Updated evaluation and model saving functions
- Removed redundant MLflow calls

## Best Practices

### 1. LightningModule Hyperparameters

Use `self.save_hyperparameters()` in your LightningModule constructor:

```python
class DETRModel(pl.LightningModule):
    def __init__(self, num_classes, learning_rate, weight_decay):
        super().__init__()
        self.save_hyperparameters()  # This is automatically logged by MLFlowLogger
        # ... rest of initialization
```

### 2. Checkpoint Configuration

The ModelCheckpoint callback is automatically configured by UnifiedTrainer:

```python
checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename=f"{model_name}-{{epoch}}-{{{monitor_metric}:.2f}}",
    monitor=monitor_metric,
    mode=monitor_mode,
    save_top_k=save_top_k,
    save_last=True
)
```

### 3. Volume Checkpointing

For persistent storage, use the VolumeCheckpoint callback:

```python
# This is automatically added if volume_checkpoint_dir is provided
if config.volume_checkpoint_dir:
    callbacks.append(VolumeCheckpoint(volume_dir=config.volume_checkpoint_dir))
```

## Migration Guide

### From Old Approach to New Approach

**Old (Complex):**
```python
# Manual MLflow setup
mlflow.set_experiment(experiment_name)
mlf_logger = MLFlowLogger(experiment_name=experiment_name, log_model=True)

# Custom checkpoint callback
callbacks.append(MLflowCheckpointCallback())

# Manual parameter logging
mlflow.log_param("learning_rate", learning_rate)
```

**New (Simplified):**
```python
# Automatic logger creation
mlf_logger = create_databricks_logger_for_task(
    task="detection",
    model_name="detr-resnet50",
    log_model="all"
)

# No custom checkpoint callback needed
# Parameters automatically logged from LightningModule
```

## Troubleshooting

### Common Issues and Solutions

1. **MLflow runs showing as "FAILED"**
   - Check that the MLFlowLogger is properly initialized
   - Ensure the experiment name is correct
   - Verify that the run is properly ended by the trainer

2. **Checkpoints not appearing in MLflow**
   - Ensure `log_model="all"` is set in the logger
   - Check that ModelCheckpoint callback is properly configured
   - Verify checkpoint directory permissions

3. **Parameters not logged**
   - Use `self.save_hyperparameters()` in your LightningModule
   - Ensure the logger is passed to the trainer
   - Check that the run is active during training

4. **Volume checkpointing not working**
   - Verify the volume directory exists and is writable
   - Check that VolumeCheckpoint callback is added to callbacks
   - Ensure ModelCheckpoint is saving files first

## Benefits of the New Approach

1. **Simplified Code**: Less custom code, more reliance on built-in functionality
2. **Better Reliability**: Native integration is more stable than custom implementations
3. **Automatic Logging**: Parameters, metrics, and checkpoints are logged automatically
4. **Consistent Behavior**: Same behavior across different environments
5. **Easier Debugging**: Fewer moving parts means easier troubleshooting

## Example Files

- `examples/simplified_training_example.py` - Complete working example
- `notebooks/02_model_training.py` - Updated training notebook
- `src/utils/logging.py` - Simplified logging utilities
- `src/training/trainer.py` - Updated UnifiedTrainer

This simplified approach provides a more robust, maintainable, and reliable MLflow integration for the Databricks Computer Vision Accelerator. 