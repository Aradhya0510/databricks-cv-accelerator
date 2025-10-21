# Package Structure

## Overview

This repository contains the `databricks_cv_accelerator` Python package along with supporting materials for usage and development.

## Directory Structure

```
Databricks_CV_ref/
├── databricks_cv_accelerator/    # 📦 INSTALLABLE PACKAGE
│   ├── __init__.py
│   ├── config.py
│   ├── config_serverless.py
│   ├── inference/
│   ├── tasks/
│   │   ├── classification/
│   │   ├── detection/
│   │   ├── instance_segmentation/
│   │   ├── semantic_segmentation/
│   │   └── universal_segmentation/
│   ├── training/
│   └── utils/
│
├── configs/                       # 📋 Configuration examples (standard compute)
├── configs_serverless/            # 📋 Configuration examples (serverless GPU)
├── notebooks/                     # 📓 Example notebooks (standard compute)
├── notebooks_serverless/          # 📓 Example notebooks (serverless GPU)
├── jobs/                         # 🚀 Production job scripts
├── examples/                     # 💡 Example usage scripts
├── lakehouse_app/                # 🏠 Streamlit Lakehouse app
├── tests/                        # 🧪 Test suite
├── docs/                         # 📚 Documentation
│
├── setup.py                      # 🔧 Package installation config
├── MANIFEST.in                   # 📄 Non-Python file inclusion
├── requirements.txt              # 📦 Dependencies
├── LICENSE                       # ⚖️ MIT License
├── README.md                     # 📖 Project overview
└── INSTALLATION.md               # 📥 Installation guide
```

## What Gets Installed

When you run `pip install -e .`, **only** the `databricks_cv_accelerator/` directory is installed as a Python package.

**Installed packages:**
- `databricks_cv_accelerator`
- `databricks_cv_accelerator.inference`
- `databricks_cv_accelerator.tasks`
- `databricks_cv_accelerator.tasks.classification`
- `databricks_cv_accelerator.tasks.detection`
- `databricks_cv_accelerator.tasks.instance_segmentation`
- `databricks_cv_accelerator.tasks.semantic_segmentation`
- `databricks_cv_accelerator.tasks.universal_segmentation`
- `databricks_cv_accelerator.training`
- `databricks_cv_accelerator.utils`

**Not installed (remain as project files):**
- `configs/` - Configuration examples for users to copy and customize
- `configs_serverless/` - Serverless-specific configurations
- `notebooks/` - Interactive development notebooks
- `notebooks_serverless/` - Serverless GPU notebooks
- `jobs/` - Production job scripts
- `examples/` - Standalone example scripts
- `lakehouse_app/` - Streamlit application
- `tests/` - Test suite for development
- `docs/` - Documentation files

## Usage Patterns

### After Installation

Once installed, you can import the package from anywhere:

```python
# Main package
import databricks_cv_accelerator
from databricks_cv_accelerator import tasks

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

# Aliased imports
import databricks_cv_accelerator.tasks.detection as db_detection
```

### Using Configuration Files

Configuration files remain in the repository for users to reference:

```python
from databricks_cv_accelerator.config import load_config

# Load a config file from the repository
config = load_config("configs/detection_detr_config.yaml")
```

### Using Notebooks

Notebooks import the installed package:

```python
# In notebooks/02_model_training.py
from databricks_cv_accelerator.config import load_config
from databricks_cv_accelerator.training.trainer import UnifiedTrainer
```

## Development Workflow

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd Databricks_CV_ref
   ```

2. **Install in development mode**
   ```bash
   pip install -e .
   ```

3. **Develop the package**
   - Edit files in `databricks_cv_accelerator/`
   - Changes are immediately available (no reinstall needed)

4. **Use examples and notebooks**
   - Run notebooks from `notebooks/` or `notebooks_serverless/`
   - Reference configs from `configs/` or `configs_serverless/`
   - Test with examples from `examples/`

5. **Test your changes**
   ```bash
   pytest tests/
   ```

## Benefits of This Structure

✅ **Clean imports**: Use `databricks_cv_accelerator` instead of generic `src`

✅ **Separation of concerns**: Package code separate from usage examples

✅ **Development flexibility**: Edit package code without reinstalling

✅ **Distribution ready**: Can publish to PyPI or internal package index

✅ **Clear organization**: Users know what to import vs. what to reference

## Migration from Old Structure

If you have existing code using `from src import ...`, update to:

```python
# Old
from src.config import load_config
from src.training.trainer import UnifiedTrainer
from src.tasks.detection.model import DetectionModel

# New
from databricks_cv_accelerator.config import load_config
from databricks_cv_accelerator.training.trainer import UnifiedTrainer
from databricks_cv_accelerator.tasks.detection.model import DetectionModel
```

