# Databricks Computer Vision Accelerator

A config-driven framework for fine-tuning object detection models on Databricks. Specify a model name + dataset path in YAML, and the framework handles data loading, training, multi-GPU DDP, and experiment tracking.

**Supported models:** Any Hugging Face `AutoModelForObjectDetection` — DETR, YOLOS, Table Transformer, etc.

**Built on:** HF Trainer, Hugging Face Transformers, MLflow, torchmetrics

---

## Quick Start

```python
from src.config.schema import load_config
from src.engine import TrainingEngine

config = load_config("configs/detection_detr_config.yaml")
engine = TrainingEngine(config)
metrics = engine.train()
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
- GPU compute (single or multi-GPU cluster)
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

| Config file | Model |
|---|---|
| `detection_detr_config.yaml` | DETR ResNet-50 |
| `detection_yolos_config.yaml` | YOLOS Base |
| `test_detection_yolos_sanity.yaml` | YOLOS Tiny (quick sanity check) |

Update the data paths, checkpoint directory, and MLflow experiment name to match your environment.

### 4. Train

**Option A — Notebook (interactive)**

Open `notebooks/02_model_training.py` in Databricks and run the cells.

**Option B — Databricks Job (production, multi-GPU)**

```bash
python jobs/train.py --config_path configs/detection_detr_config.yaml
python jobs/train.py --config_path configs/detection_detr_config.yaml --num_gpus 4
```

Set up as a Databricks Job:
1. Go to **Workflows > Create Job**
2. Set task type to **Python script**, path to `jobs/train.py`
3. Add parameters: `--config_path /Volumes/.../configs/your_config.yaml`
4. Select a multi-GPU cluster (e.g. `g5.12xlarge` with 4x A10G)
5. Run — all GPUs are used automatically via native DDP

---

## Multi-GPU Strategy

- **Single-node multi-GPU (default):** HF Trainer handles DDP natively. No Spark overhead. Pass `--num_gpus N` or let it auto-detect.
- **Multi-node (opt-in):** Pass `--distributed torchd` to use TorchDistributor across Spark workers. Only for scaling beyond a single machine.

---

## Project Structure

```
src/
├── config/
│   └── schema.py              # Pydantic v2 config models + YAML loader
├── registry.py                # TaskRegistry with @register decorator
├── engine/
│   ├── engine.py              # TrainingEngine: config → train → metrics
│   ├── trainer.py             # CVTrainer — generic HF Trainer with task hooks
│   └── callbacks.py           # VolumeCheckpointCallback, EarlyStoppingCallback
├── tasks/
│   └── detection/
│       ├── __init__.py        # DetectionTask (model, data, loss, eval hooks)
│       ├── adapters.py        # DETR/YOLOS input/output adapters
│       ├── collate.py         # Detection collate function
│       └── data.py            # COCODetectionDataset
└── utils/
    └── environment.py         # GPU detection, NCCL setup, data staging
configs/                       # YAML configs per model
notebooks/                     # Interactive Databricks notebooks
jobs/
└── train.py                   # CLI entry point for Jobs
lakehouse_app/                 # Streamlit UI for launching training
```

---

## Configuration

Configs have four sections. Learning rate, optimizer, and scheduler live in `model` (model-specific). The `training` section controls the training loop.

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

mlflow:
  experiment_name: "/Users/you@company.com/cv_detection"
```

---

## Adding New Detection Models

To add a new Hugging Face detection model, create an adapter in `src/tasks/detection/adapters.py`:

```python
class YourModelInputAdapter(BaseAdapter):
    def __init__(self, model_name, image_size=800):
        self.processor = AutoImageProcessor.from_pretrained(model_name, size={"height": image_size, "width": image_size})

    def __call__(self, image, target):
        processed = self.processor(image, return_tensors="pt")
        return processed.pixel_values.squeeze(0), adapted_target
```

Register it in `get_input_adapter()`, then use it in your config:
```yaml
model:
  model_name: "your-org/your-model"
```

---

## Lakehouse App

The `lakehouse_app/` directory contains a Streamlit UI for configuring training, launching Jobs, and monitoring runs. Deploy it as a Databricks App — see [`lakehouse_app/README.md`](lakehouse_app/README.md).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
