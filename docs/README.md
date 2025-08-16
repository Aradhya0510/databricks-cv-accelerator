# Documentation Overview

This directory contains the essential documentation for the Databricks Computer Vision Framework.

## 📚 Available Documentation

### Core Documentation
- **[SUPPORTED_TASKS_AND_MODELS.md](SUPPORTED_TASKS_AND_MODELS.md)** - Complete guide to supported tasks and available models
- **[ADAPTER_OVERVIEW.md](ADAPTER_OVERVIEW.md)** - Comprehensive guide to the adapter system architecture
- **[DATA_CONFIGURATION.md](DATA_CONFIGURATION.md)** - Guide for configuring data paths and dataset splits
- **[SIMPLIFIED_MLFLOW_INTEGRATION.md](SIMPLIFIED_MLFLOW_INTEGRATION.md)** - Simplified MLflow integration approach
- **[CONFIG_COMPATIBILITY_GUIDE.md](CONFIG_COMPATIBILITY_GUIDE.md)** - Configuration compatibility and migration guide

## 🗑️ Purged Documentation

The following documentation was removed during cleanup as it was redundant, outdated, or too specific:

### Redundant Documentation
- `MLFLOW_TROUBLESHOOTING.md` - Redundant with SIMPLIFIED_MLFLOW_INTEGRATION.md
- `TASK_MODULES_BEST_PRACTICES.md` - Covered in other documentation
- `DATA_MODULE_PATTERN.md` - Redundant with DATA_CONFIGURATION.md

### One-time Fixes
- `MODEL_SERVING_FIX.md` - Specific to a one-time serving issue, not general documentation

### Overly Detailed Adapter Documentation
- `CLASSIFICATION_ADAPTERS.md` - Too detailed, covered in ADAPTER_OVERVIEW.md
- `DETECTION_ADAPTERS.md` - Too detailed, covered in ADAPTER_OVERVIEW.md
- `SEMANTIC_SEGMENTATION_ADAPTERS.md` - Too detailed, covered in ADAPTER_OVERVIEW.md
- `INSTANCE_SEGMENTATION_ADAPTERS.md` - Too detailed, covered in ADAPTER_OVERVIEW.md
- `PANOPTIC_SEGMENTATION_ADAPTERS.md` - Too detailed, covered in ADAPTER_OVERVIEW.md

## 📖 Documentation Philosophy

The remaining documentation follows these principles:

1. **Essential Only** - Only core, frequently-needed information
2. **No Redundancy** - Each document serves a unique purpose
3. **Practical Focus** - Emphasis on usage rather than implementation details
4. **Framework-Specific** - Focused on our framework's unique features
5. **Maintainable** - Easy to keep up-to-date with framework changes

## 🔗 Related Resources

- **Source Code**: Implementation details are in `src/tasks/[task_type]/adapters.py`
- **Examples**: Working examples in `examples/` directory
- **Notebooks**: Interactive tutorials in `notebooks/` directory
- **Configs**: Configuration templates in `configs/` directory
