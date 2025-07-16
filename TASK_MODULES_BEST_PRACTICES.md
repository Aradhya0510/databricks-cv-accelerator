# Task Modules Best Practices Alignment

This document outlines the best practices that should be consistently applied across all task modules (`classification`, `detection`, `semantic_segmentation`, `instance_segmentation`, `panoptic_segmentation`) to ensure uniformity and maintainability.

## ✅ Completed Updates

### 1. Configuration Classes
All task modules now include:
- `num_workers: int = 1` parameter
- `scheduler_params: Optional[Dict[str, Any]] = None` parameter
- `sync_dist_flag` property that returns `True` if `num_workers > 1`

### 2. Optimizer Configuration
All modules now use the advanced `configure_optimizers` method with:
- Backbone parameter separation with different learning rates
- Proper scheduler configuration with `eta_min=1e-6`
- Consistent return format with optimizer and scheduler dictionaries

### 3. Logging Improvements
All modules now include:
- `sync_dist=self.config.sync_dist_flag` in all logging calls
- `batch_size` extraction from input data with safe fallbacks
- Proper error handling for batch size extraction

### 4. Checkpoint Management
All modules now use consistent checkpoint saving/loading:
- Save optimizer parameters instead of full config dict
- Load and restore optimizer parameters properly
- Save class names for model reconstruction

## 🔧 Remaining Issues to Address

### 1. Import Issues
Some modules have linter warnings about:
- `PreTrainedModel` import (should be removed)
- `Dice` import from torchmetrics (may need version-specific import)

### 2. Adapter Interface Issues
All modules have a consistent issue with:
- `format_targets` method expecting `Dict[str, Tensor]` but receiving `Tensor`
- This suggests the adapter interface needs standardization

### 3. Metric Logging Issues
Some modules have issues with:
- Logging dictionary values instead of scalar metrics
- Need to ensure all logged values are proper scalars

## 📋 Best Practices Summary

### Configuration Standardization
```python
@dataclass
class TaskModelConfig:
    # ... existing fields ...
    scheduler_params: Optional[Dict[str, Any]] = None
    num_workers: int = 1
    
    @property
    def sync_dist_flag(self) -> bool:
        return self.num_workers > 1
```

### Optimizer Configuration
```python
def configure_optimizers(self):
    # Check for backbone and create parameter groups
    if hasattr(self.model, 'backbone'):
        # Separate backbone and task-specific parameters
        # Use different learning rates
    else:
        # Standard optimizer
    
    # Configure scheduler with proper parameters
    if self.config.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": {...}}
```

### Logging Standardization
```python
# Get batch size safely
try:
    batch_size = batch["pixel_values"].shape[0] if "pixel_values" in batch else None
except (KeyError, AttributeError, IndexError):
    batch_size = None

# Log with sync_dist and batch_size
self.log("metric_name", value, sync_dist=self.config.sync_dist_flag, batch_size=batch_size)
```

### Checkpoint Management
```python
def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
    checkpoint["class_names"] = self.config.class_names
    checkpoint["optimizer_params"] = {
        "learning_rate": self.config.learning_rate,
        "weight_decay": self.config.weight_decay,
        "scheduler": self.config.scheduler,
        "scheduler_params": self.config.scheduler_params
    }

def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
    if "class_names" in checkpoint:
        self.config.class_names = checkpoint["class_names"]
    if "optimizer_params" in checkpoint:
        params = checkpoint["optimizer_params"]
        # Restore optimizer parameters
```

## 🎯 Next Steps

1. **Fix Import Issues**: Remove unused imports and fix torchmetrics imports
2. **Standardize Adapter Interface**: Ensure all adapters have consistent input/output formats
3. **Fix Metric Logging**: Ensure all logged values are proper scalars
4. **Add Memory Management**: Ensure all modules have proper memory cleanup
5. **Add Comprehensive Testing**: Test all modules with the updated configurations

## 📊 Status by Module

| Module | Config Updated | Optimizer Updated | Logging Updated | Checkpoint Updated | Issues |
|--------|---------------|-------------------|-----------------|-------------------|---------|
| detection | ✅ | ✅ | ✅ | ✅ | None |
| classification | ✅ | ✅ | ✅ | ✅ | Import/Adapter |
| semantic_segmentation | ✅ | ✅ | ✅ | ✅ | Import/Adapter |
| instance_segmentation | ✅ | ✅ | ⚠️ | ⚠️ | Import/Adapter/Metrics |
| panoptic_segmentation | ✅ | ✅ | ⚠️ | ⚠️ | Import/Adapter/Metrics |

Legend:
- ✅ Complete
- ⚠️ Partial (needs additional updates)
- ❌ Not started 