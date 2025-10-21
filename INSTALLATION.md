# Installation Guide

## Quick Installation

### Development Mode (Recommended for Development)

Install the package in editable mode so changes to the source code are immediately reflected:

```bash
cd /path/to/Databricks_CV_ref
pip install -e .
```

### Standard Installation

Install the package normally:

```bash
cd /path/to/Databricks_CV_ref
pip install .
```

### Installation with Optional Dependencies

Install with development dependencies:

```bash
pip install -e ".[dev]"
```

Install with documentation dependencies:

```bash
pip install -e ".[docs]"
```

Install with all optional dependencies:

```bash
pip install -e ".[all]"
```

## Verification

Verify the installation:

```python
import databricks_cv_accelerator
print(f"Version: {databricks_cv_accelerator.__version__}")

from databricks_cv_accelerator.config import load_config
from databricks_cv_accelerator.training.trainer import UnifiedTrainer

print("✓ Installation successful!")
```

## Importing Modules

After installation, you can import modules from anywhere:

```python
# Configuration
from databricks_cv_accelerator.config import load_config

# Training
from databricks_cv_accelerator.training.trainer import UnifiedTrainer

# Task-specific modules
from databricks_cv_accelerator.tasks.detection.model import DetectionModel
from databricks_cv_accelerator.tasks.detection.data import DetectionDataModule

# Inference
from databricks_cv_accelerator.inference.batch_inference import BatchInference

# Utils
from databricks_cv_accelerator.utils.logging import create_databricks_logger

# Alternative import styles
import databricks_cv_accelerator.tasks.detection as db_detection
from databricks_cv_accelerator import tasks
```

## Databricks-Specific Notes

When using in Databricks, you have several options:

### Option 1: Install from Workspace Files
```bash
%pip install -e /Workspace/path/to/Databricks_CV_ref
```

### Option 2: Install from Git Repository
```bash
%pip install git+https://github.com/Aradhya0510/databricks-cv-architecture.git
```

### Option 3: Install from Volume
If you've copied the repository to a Unity Catalog volume:
```bash
%pip install -e /Volumes/catalog/schema/volume/Databricks_CV_ref
```

## Dependencies

The package will automatically install the following core dependencies:

- `lightning>=2.5.2` - PyTorch Lightning for training
- `optuna>=3.2.0` - Hyperparameter optimization
- `pycocotools>=2.0.6` - COCO dataset utilities
- `timm>=0.9.0` - PyTorch Image Models
- `albumentations==2.0.8` - Image augmentation library
- `opencv-python>=4.8.0` - Computer vision utilities
- `numpy==1.26.4` - Numerical computing

**Note:** The following packages are provided by Databricks ML Runtime and don't need to be installed:
- torch, torchvision, torchmetrics
- transformers, tokenizers
- pandas, pillow, requests
- pyyaml, tqdm

## Troubleshooting

### Issue: Module not found after installation

Make sure you installed the package correctly and restart your Python kernel:

```python
import sys
print(sys.path)
```

### Issue: Version conflicts

If you encounter dependency conflicts, try installing in a fresh virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

### Issue: Import errors in Databricks

Restart the Python execution environment after installation:

```python
dbutils.library.restartPython()
```

## Uninstallation

To uninstall the package:

```bash
pip uninstall databricks-cv-accelerator
```

