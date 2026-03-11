# Databricks Computer Vision Accelerator

A config-driven framework for training and deploying computer vision models on Databricks. Specify a model name + dataset path in YAML, and the framework handles everything else: data loading, training, multi-GPU DDP, experiment tracking, and deployment.

**Supported tasks:** Object Detection, Image Classification, Semantic Segmentation, Instance Segmentation, Universal Segmentation

**Built on:** PyTorch Lightning, Hugging Face Transformers, MLflow, Optuna, Albumentations

---

## Quick Start

```python
# 1. Load config
from config import load_config
config = load_config("configs/detection_detr_config.yaml")

# 2. Create model and data
from tasks.detection.model import DetectionModel
from tasks.detection.data import DetectionDataModule
from tasks.detection.adapters import get_input_adapter

model = DetectionModel(config["model"])
data_module = DetectionDataModule(config["data"])
data_module.adapter = get_input_adapter(config["model"]["model_name"], image_size=800)
data_module.setup("fit")

# 3. Train
from training.trainer import Trainer
from utils.logging import create_databricks_logger

logger = create_databricks_logger(experiment_name="/Users/you@company.com/detection")
trainer = Trainer(
    config={
        "task": "detection",
        "model_name": config["model"]["model_name"],
        "max_epochs": config["training"]["max_epochs"],
        "log_every_n_steps": 50,
        "monitor_metric": "val_map",
        "monitor_mode": "max",
        "early_stopping_patience": 20,
        "checkpoint_dir": "/Volumes/catalog/schema/volume/checkpoints",
        "use_gpu": True,
    },
    model=model,
    data_module=data_module,
    logger=logger,
)
metrics = trainer.train()
```

To switch models, change one line in your config:
```yaml
model:
  model_name: "hustvl/yolos-tiny"  # was "facebook/detr-resnet-50"
```

---

## Getting Started

### Prerequisites

- Databricks workspace with **Unity Catalog** enabled
- Dataset in **COCO format** (image folder + `annotations.json`) uploaded to a UC Volume
- GPU compute (single GPU for notebooks, multi-GPU cluster for Jobs)
- **Databricks Runtime ML 16.4+**

### 1. Clone the repo

```bash
git clone https://github.com/Aradhya0510/databricks-cv-accelerator.git
```

### 2. Organize your data in Unity Catalog

```
/Volumes/your_catalog/your_schema/your_volume/
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── annotations.json
│   ├── val/
│   │   ├── images/
│   │   └── annotations.json
│   └── test/  (optional)
├── checkpoints/
└── results/
```

### 3. Pick a config and customize it

Choose from `configs/` based on your task and model:

| Config file | Task | Model |
|---|---|---|
| `classification_resnet_config.yaml` | Classification | ResNet-50 |
| `classification_vit_config.yaml` | Classification | ViT-Base |
| `detection_detr_config.yaml` | Detection | DETR ResNet-50 |
| `detection_yolos_config.yaml` | Detection | YOLOS |
| `semantic_segmentation_mitb0_config.yaml` | Segmentation | MiT-B0 |
| `semantic_segmentation_segformer_config.yaml` | Segmentation | SegFormer |
| `instance_segmentation_mask2former_config.yaml` | Instance Seg | Mask2Former |
| `universal_segmentation_mask2former_config.yaml` | Universal Seg | Mask2Former |

Update the data paths, checkpoint directory, and MLflow experiment name to match your environment. See [`configs/README.md`](configs/README.md) for the full config reference.

### 4. Run the notebooks

Walk through the notebooks in `notebooks/` in order:

| Notebook | Purpose |
|---|---|
| `00_setup_and_config.py` | Validate environment, load config, verify GPU + data |
| `01_data_preparation.py` | Explore and validate dataset |
| `02_model_training.py` | Train model with MLflow tracking |
| `03_hparam_tuning.py` | Hyperparameter search with Optuna (optional) |
| `04_model_evaluation.py` | Evaluate model, analyze metrics and errors |
| `05_model_registration_deployment.py` | Register in UC, deploy as serving endpoint |
| `06_model_monitoring.py` | Monitor drift and endpoint health |

Notebooks run on **single GPU** interactively. For multi-GPU training, use Jobs (next section).

---

## Two Training Paths

### Notebooks (interactive, single GPU)

Best for development and experimentation. Run cells interactively, inspect results, iterate quickly.

### Databricks Jobs (production, multi-GPU DDP)

Best for production training. The `Trainer` automatically detects the Jobs environment and uses all available GPUs with DDP.

**Why Jobs?** Lightning limits notebooks to 1 GPU:
```
Trainer will use only 1 of 4 GPUs because it is running inside an interactive environment.
```

**How to set up a training Job:**

1. Go to **Workflows > Create Job** in Databricks
2. Set task type to **Python script**, path to `jobs/model_training.py`
3. Add parameters:
   ```
   --config_path /Volumes/.../configs/detection_detr_config.yaml
   --src_path /Workspace/.../src
   ```
4. Select a multi-GPU cluster (e.g. `g5.12xlarge` with 4x A10G)
5. Run — all GPUs are used automatically

| | Notebooks | Jobs |
|---|---|---|
| **GPUs** | Single | All available (DDP) |
| **Best for** | Development | Production training |
| **Scheduling** | Manual | Cron, triggers, chaining |
| **Config** | Code cells | CLI parameters |

**Recommended workflow:**
1. Develop and iterate in **notebooks** (single GPU)
2. Launch production training via **Jobs** (multi-GPU)
3. Deploy with `jobs/model_registration_deployment.py`

---

## Supported Tasks and Models

The framework uses Hugging Face Auto-classes, so any compatible model works by just changing `model_name` in the config.

| Task | Models | Monitor Metric |
|---|---|---|
| **Object Detection** | DETR, YOLOS, Table Transformer | `val_map` |
| **Image Classification** | ViT, ResNet, ConvNeXT, Swin | `val_loss` |
| **Semantic Segmentation** | SegFormer, MiT, Mask2Former | `val_miou` |
| **Instance Segmentation** | Mask2Former | `val_map` |
| **Universal Segmentation** | Mask2Former (panoptic) | `val_miou` |

See [docs/SUPPORTED_TASKS_AND_MODELS.md](docs/SUPPORTED_TASKS_AND_MODELS.md) for the full list.

---

## Project Structure

```
├── configs/              # YAML config files for each task/model
├── notebooks/            # Interactive Databricks notebooks (step-by-step)
├── jobs/                 # Python scripts for Databricks Jobs (multi-GPU)
├── lakehouse_app/        # Streamlit UI for launching training from a Lakehouse App
├── src/
│   ├── config.py         # Config dataclasses and YAML loading
│   ├── training/
│   │   └── trainer.py    # Trainer with auto DDP (Jobs) / single GPU (notebooks)
│   ├── tasks/
│   │   ├── classification/
│   │   ├── detection/
│   │   ├── semantic_segmentation/
│   │   ├── instance_segmentation/
│   │   └── universal_segmentation/
│   │       ├── model.py      # LightningModule for the task
│   │       ├── data.py       # LightningDataModule + Dataset
│   │       ├── adapters.py   # Input/output adapters per model
│   │       ├── evaluate.py   # Task-specific metrics
│   │       └── inference.py  # Prediction utilities
│   └── utils/
│       ├── logging.py            # MLflow logger + volume checkpointing
│       └── config_validator.py   # Config validation
├── tests/
└── docs/
```

For a deep dive into the architecture, data flow, and adapter system, see [`src/README.md`](src/README.md).

---

## Configuration

Configs have four sections. Learning rate, optimizer, and scheduler live in `model` (since they're model-specific). The `training` section controls the training loop.

```yaml
model:
  model_name: "facebook/detr-resnet-50"
  task_type: detection
  num_classes: 80
  learning_rate: 1e-4
  weight_decay: 1e-4
  scheduler: cosine

data:
  train_data_path: "/Volumes/.../train/"
  train_annotation_file: "/Volumes/.../annotations_train.json"
  val_data_path: "/Volumes/.../val/"
  val_annotation_file: "/Volumes/.../annotations_val.json"
  batch_size: 16
  num_workers: 4

training:
  max_epochs: 50
  early_stopping_patience: 20
  monitor_metric: val_map
  monitor_mode: max
  checkpoint_dir: "/Volumes/.../checkpoints"
  use_gpu: true

mlflow:
  experiment_name: "/Users/you@company.com/cv_detection"
```

Full reference: [`configs/README.md`](configs/README.md) and [docs/CONFIGURATION_PARAMETER_REFERENCE.md](docs/CONFIGURATION_PARAMETER_REFERENCE.md)

---

## Adding New Models

To add a new Hugging Face model to an existing task, create an adapter:

```python
# In src/tasks/detection/adapters.py
class YourModelInputAdapter(BaseAdapter):
    def __init__(self, model_name, image_size=800):
        self.processor = AutoImageProcessor.from_pretrained(model_name, size={"height": image_size, "width": image_size})

    def __call__(self, image, target):
        processed = self.processor(image, return_tensors="pt")
        return processed.pixel_values.squeeze(0), adapted_target

# Register it in get_input_adapter()
def get_input_adapter(model_name, image_size=800):
    if "your_model" in model_name:
        return YourModelInputAdapter(model_name, image_size)
    ...
```

Then use it in your config:
```yaml
model:
  model_name: "your-org/your-model"
```

For adding entirely new task types, see the [architecture guide](src/README.md).

---

## Lakehouse App

The `lakehouse_app/` directory contains a Streamlit-based UI that lets users configure training, launch Jobs, and monitor runs without writing code. Deploy it as a Databricks Lakehouse App.

See [`lakehouse_app/README.md`](lakehouse_app/README.md) for setup instructions.

---

## Further Reading

| Doc | What it covers |
|---|---|
| [`src/README.md`](src/README.md) | Architecture deep dive: data flow, adapters, trainer internals |
| [`src/tasks/README.md`](src/tasks/README.md) | Task module structure and how to extend |
| [`configs/README.md`](configs/README.md) | Config file format and all available keys |
| [`docs/CONFIGURATION_PARAMETER_REFERENCE.md`](docs/CONFIGURATION_PARAMETER_REFERENCE.md) | Every config parameter explained |
| [`docs/SIMPLIFIED_MLFLOW_INTEGRATION.md`](docs/SIMPLIFIED_MLFLOW_INTEGRATION.md) | How MLflow integration works |
| [`jobs/README.md`](jobs/README.md) | Jobs setup for multi-GPU training |
| [`lakehouse_app/README.md`](lakehouse_app/README.md) | Lakehouse App deployment guide |

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
