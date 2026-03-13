# Databricks CV Accelerator

A production-ready framework for fine-tuning computer vision models on Databricks. Drop in a config, point it at your data, and get a trained, evaluated, and deployed model — with full MLflow tracking, multi-GPU support, and a Streamlit UI.

## Why This Framework

Fine-tuning CV models on Databricks involves gluing together data loading, HuggingFace models, distributed training, experiment tracking, model packaging, and serving. This framework does it for you:

- **Config-driven.** One YAML file controls the entire pipeline — model, data, training, serving. No boilerplate code to write.
- **Task-agnostic.** Detection and classification work today. Add new tasks by implementing a single class. The engine, evaluation, serving, and monitoring layers all adapt automatically.
- **Databricks-native.** Unity Catalog Volumes for data, MLflow for tracking, Model Serving for deployment, system tables for monitoring. Everything wired together.
- **Multi-GPU out of the box.** HF Trainer handles DDP natively. Pass `--num_gpus 4` and it works. No Spark orchestration overhead for single-node.
- **Full lifecycle.** Train, evaluate (mAP/accuracy + error analysis + latency benchmarks), register to Unity Catalog, deploy to Model Serving, and monitor — all from the same framework.

## What You Can Do

| Task | Models | Data Format |
|---|---|---|
| Object Detection | DETR, YOLOS, any `AutoModelForObjectDetection` | COCO JSON (images + annotations.json) |
| Image Classification | ViT, ResNet, any `AutoModelForImageClassification` | ImageFolder (class_name/image.jpg) |

To switch models, change one line:

```yaml
model:
  model_name: "google/vit-base-patch16-224"  # or "facebook/detr-resnet-50", "hustvl/yolos-base", etc.
  task_type: classification                   # or "detection"
```

## Architecture

```
Config YAML ─→ PipelineConfig (Pydantic) ─→ TrainingEngine ─→ CVTrainer (HF Trainer)
                                                │
                                   TaskRegistry │ dispatches to:
                                                ├── DetectionTask
                                                └── ClassificationTask
```

The framework separates **what** (task-specific logic) from **how** (training loop, evaluation, serving):

- **Tasks** (`src/tasks/`) provide model loading, data handling, loss, and eval metrics
- **Engine** (`src/engine/`) runs the training loop and DDP — task-agnostic
- **Evaluation** (`src/evaluation/`) runs standalone eval, error analysis, benchmarks — task-aware
- **Serving** (`src/serving/`) wraps models as PyFunc for Model Serving — task-aware
- **Monitoring** (`src/monitoring/`) queries system tables for endpoint observability

## Project Structure

```
src/
├── config/schema.py              # Pydantic v2 config + YAML loader
├── registry.py                   # TaskRegistry (@register decorator)
├── engine/
│   ├── engine.py                 # TrainingEngine: config → train → metrics
│   ├── trainer.py                # CVTrainer (HF Trainer + task hooks)
│   └── callbacks.py              # Checkpoint + early stopping callbacks
├── tasks/
│   ├── detection/                # DetectionTask, COCO dataset, DETR/YOLOS adapters
│   └── classification/           # ClassificationTask, ImageFolder dataset
├── evaluation/
│   └── engine.py                 # EvaluationEngine: metrics, error analysis, benchmarks
├── serving/
│   ├── pyfunc.py                 # DetectionPyFuncModel, ClassificationPyFuncModel
│   ├── registration.py           # register_model() → Unity Catalog
│   └── deployment.py             # deploy_endpoint() → Model Serving
└── monitoring/
    └── endpoint_monitor.py       # Health, request metrics, prediction distribution

jobs/
├── train.py                      # Training CLI
├── evaluate.py                   # Evaluation CLI
├── deploy.py                     # Registration + deployment CLI
└── monitor.py                    # Monitoring report CLI

notebooks/
├── 01_data_exploration.py        # EDA: stats, class distribution, quality checks
├── 02_model_training.py          # Interactive training
├── 03_model_evaluation.py        # Evaluation + error analysis
├── 04_model_deployment.py        # PyFunc test → register → deploy → smoke test
└── 05_model_monitoring.py        # Endpoint health + metrics + alerts

configs/                          # YAML configs per model
lakehouse_app/                    # Streamlit UI (9 pages, full lifecycle)
```

## Quick Start

See **[GETTING_STARTED.md](GETTING_STARTED.md)** for the full setup guide. The short version:

```bash
# 1. Clone
git clone <repo-url> && cd Databricks_CV_ref

# 2. Pick a config, update paths
cp configs/classification_vit_config.yaml configs/my_config.yaml
# Edit data paths, MLflow experiment, checkpoint dirs

# 3. Train
python jobs/train.py --config_path configs/my_config.yaml

# 4. Evaluate
python jobs/evaluate.py --config_path configs/my_config.yaml --checkpoint_path /path/to/model

# 5. Deploy
python jobs/deploy.py --config_path configs/my_config.yaml --run_id <mlflow_run_id> \
    --model_name catalog.schema.my_model --endpoint_name my-endpoint
```

## Extending the Framework

### Add a New Model (Same Task)

For detection, just use a different HuggingFace model name in your config — if it works with `AutoModelForObjectDetection`, it works here. Same for classification with `AutoModelForImageClassification`.

For detection models with non-standard input/output formats, add an adapter in `src/tasks/detection/adapters.py`.

### Add a New Task

Create a new directory under `src/tasks/` and implement the task interface:

```python
from src.registry import TaskRegistry

@TaskRegistry.register("segmentation")
class SegmentationTask:
    def get_model(self, model_cfg): ...
    def get_train_dataset(self, config): ...
    def get_val_dataset(self, config): ...
    def get_collate_fn(self): ...
    def create_optimizer_and_scheduler(self, model, config, num_training_steps): ...
    def compute_loss(model, inputs, return_outputs=False): ...
    def get_eval_fn(self, model_cfg): ...
```

Then add `import src.tasks.segmentation` to `src/engine/engine.py` and `src/evaluation/engine.py`. The training engine, evaluation pipeline, and job scripts all work automatically.

## Multi-GPU

- **Single-node (default):** HF Trainer handles DDP natively. Pass `--num_gpus N` or auto-detect.
- **Multi-node (opt-in):** Pass `--distributed torchd` to use TorchDistributor across Spark workers.

```bash
# 4x A10G on a single node
python jobs/train.py --config_path configs/my_config.yaml --num_gpus 4

# Multi-node via TorchDistributor (only when needed)
python jobs/train.py --config_path configs/my_config.yaml --distributed torchd
```

## Lakehouse App

The `lakehouse_app/` directory contains a Streamlit UI covering the full lifecycle: config setup, data EDA, training, evaluation, model registration, deployment, inference testing, run history, and endpoint monitoring. Deploy as a Databricks App — see [`lakehouse_app/README.md`](lakehouse_app/README.md).

## Built With

[HuggingFace Transformers](https://huggingface.co/docs/transformers) | [MLflow](https://mlflow.org) | [Databricks](https://databricks.com) | [PyTorch](https://pytorch.org) | [torchmetrics](https://torchmetrics.readthedocs.io)

## License

MIT License — see [LICENSE](LICENSE).
