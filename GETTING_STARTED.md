# Getting Started

This guide walks through setting up the Databricks CV Accelerator — from data preparation to a deployed model endpoint. It covers both detection and classification tasks.

---

## Prerequisites

| Requirement | Details |
|---|---|
| Databricks workspace | Unity Catalog enabled, AI Runtime preview enabled, in an AI Runtime region |
| Compute | AI Runtime serverless GPUs — `GPU_1xA10` for dev, `GPU_8xH100` for production training |
| Environment | Databricks AI environment **v6** (`databricks_ai_v6`) |
| CLI | Databricks CLI 1.6.0+ (`databricks air`, `ai_runtime_task` in bundles) |
| Data | Images uploaded to a UC Volume |

### Python Dependencies

The Databricks AI environment v6 ships torch, torchvision, transformers 5,
MLflow and the rest of the stack. The three packages it lacks are listed in
`requirements_runtime.txt`; `air/*.yaml` and `databricks.yml` install them, and
the notebooks `%pip install` them.

If running locally:

```bash
pip install -e ".[dev]"
```

Or, to install the pieces by hand:

```bash
pip install torch torchvision "transformers>=5.0" datasets mlflow
pip install pycocotools torchmetrics  # for detection
pip install pydantic pyyaml pillow numpy
```

---

## Step 1: Prepare Your Data

### Detection (COCO Format)

Organize your data as images + a COCO-format JSON annotation file:

```
/Volumes/my_catalog/my_schema/my_volume/data/
├── train/
│   ├── 000001.jpg
│   ├── 000002.jpg
│   └── ...
├── val/
│   ├── 000100.jpg
│   └── ...
├── annotations/
│   ├── train.json       # COCO format: {"images": [...], "annotations": [...], "categories": [...]}
│   └── val.json
```

The annotation JSON must follow [COCO format](https://cocodataset.org/#format-data). Each annotation has `image_id`, `category_id`, `bbox` (x, y, w, h), and `area`.

### Classification (ImageFolder)

Organize images into subdirectories named by class:

```
/Volumes/my_catalog/my_schema/my_volume/data/
├── train/
│   ├── cat/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   ├── dog/
│   │   ├── img003.jpg
│   │   └── img004.jpg
│   └── bird/
│       └── ...
├── val/
│   ├── cat/
│   ├── dog/
│   └── bird/
```

Class names are derived from sorted subdirectory names. No annotation file needed.

---

## Step 2: Create a Config

Copy one of the starter configs and customize it:

```bash
# Detection
cp configs/detection_yolos_config.yaml configs/my_detection.yaml

# Classification
cp configs/classification_vit_config.yaml configs/my_classification.yaml
```

### Minimal Detection Config

```yaml
model:
  model_name: hustvl/yolos-base          # Any HuggingFace detection model
  task_type: detection
  num_classes: 80                         # Number of classes in your dataset
  image_size: 800
  learning_rate: 1e-4
  weight_decay: 1e-4
  scheduler: cosine
  epochs: 50
  confidence_threshold: 0.7
  iou_threshold: 0.5

data:
  train_data_path: /Volumes/my_catalog/my_schema/my_volume/data/train/
  train_annotation_file: /Volumes/my_catalog/my_schema/my_volume/data/annotations/train.json
  val_data_path: /Volumes/my_catalog/my_schema/my_volume/data/val/
  val_annotation_file: /Volumes/my_catalog/my_schema/my_volume/data/annotations/val.json
  batch_size: 8
  num_workers: 4
  model_name: hustvl/yolos-base           # Must match model.model_name
  image_size: [800, 800]

training:
  max_epochs: 50
  early_stopping_patience: 10
  monitor_metric: val_map
  monitor_mode: max
  checkpoint_dir: /tmp/checkpoints
  volume_checkpoint_dir: /Volumes/my_catalog/my_schema/my_volume/checkpoints
  save_top_k: 3

mlflow:
  experiment_name: /Users/you@company.com/cv_detection
  run_name: yolos_base_run1

output:
  results_dir: /tmp/results
```

### Minimal Classification Config

```yaml
model:
  model_name: google/vit-base-patch16-224
  task_type: classification
  num_classes: 3                          # Number of classes (subdirectories)
  image_size: 224
  learning_rate: 5e-5
  weight_decay: 1e-2
  scheduler: cosine
  epochs: 20
  dropout: 0.1

data:
  train_data_path: /Volumes/my_catalog/my_schema/my_volume/data/train/
  val_data_path: /Volumes/my_catalog/my_schema/my_volume/data/val/
  batch_size: 32
  num_workers: 4
  model_name: google/vit-base-patch16-224
  image_size: [224, 224]

training:
  max_epochs: 20
  early_stopping_patience: 5
  monitor_metric: val_accuracy
  monitor_mode: max
  checkpoint_dir: /tmp/checkpoints

mlflow:
  experiment_name: /Users/you@company.com/cv_classification
  run_name: vit_base_run1

output:
  results_dir: /tmp/results
```

### Key Config Fields

| Field | Notes |
|---|---|
| `model.model_name` | Any HuggingFace model ID. Must match `data.model_name`. |
| `model.task_type` | `"detection"` or `"classification"` |
| `model.num_classes` | Must match your dataset |
| `training.monitor_metric` | `val_map` for detection, `val_accuracy` for classification |
| `training.monitor_mode` | `max` for mAP/accuracy, `min` for loss |
| `training.volume_checkpoint_dir` | Optional. Syncs checkpoints to a Volume for persistence. |
| `training.resume_from_checkpoint` | Optional. `latest` resumes retries from `volume_checkpoint_dir`. |
| `output.results_dir` | A Volume path, so evaluate/deploy can find the run manifest. |

---

## Step 3: Train

### Option A — `databricks air run` (from your laptop)

```bash
# One A10 (the default in air/train.yaml)
databricks air run -f air/train.yaml \
    --override env_variables.CV_CONFIG_PATH=configs/my_config.yaml --watch

# A full H100 node: jobs/train.py runs one DDP process per GPU
databricks air run -f air/train.yaml \
    --override env_variables.CV_CONFIG_PATH=configs/my_config.yaml \
    --override compute.accelerator_type=GPU_8xH100 \
    --override compute.num_accelerators=8
```

The CLI snapshots `src/`, `jobs/` and `configs/`, so the config must be inside
the repo (or reference a `/Volumes` path). Inspect with `databricks air logs <run-id>`.

### Option B — Notebook (interactive development)

Clone the repo as a Git folder, open `notebooks/02_model_training.py`, attach it
to AI Runtime with the AI v6 environment, update the config path and run all
cells. On an 8xH100 attachment `engine.train()` uses all eight GPUs.

### Option C — Bundle (scheduled / CI)

Set the config in `jobs/air/pipeline.env` and `config_path` in
`databricks.yml`, then:

```bash
databricks bundle deploy --target dev
databricks bundle run cv_training_pipeline --target dev
```

This runs train → evaluate on AI Runtime and register → deploy on serverless CPU.

All metrics are logged to MLflow automatically, in the experiment AI Runtime
creates for the run. The model artifact is saved at the end of training.

---

## Step 4: Evaluate

Run standalone evaluation to get metrics, error analysis, and latency benchmarks.
With no model given, it evaluates the last training run's model from the run
manifest (`databricks air run -f air/evaluate.yaml` does exactly this):

```bash
python jobs/evaluate.py --config_path configs/my_config.yaml
```

Or point it at a specific checkpoint:

```bash
python jobs/evaluate.py \
    --config_path configs/my_config.yaml \
    --checkpoint_path /path/to/best/model \
    --output_dir /tmp/eval_results
```

Or use an MLflow run ID instead of a checkpoint path:

```bash
python jobs/evaluate.py \
    --config_path configs/my_config.yaml \
    --run_id abc123def456
```

This produces three JSON files in the output directory:

| File | Contents |
|---|---|
| `evaluation_metrics.json` | mAP (detection) or accuracy/F1 (classification) |
| `error_analysis.json` | FP/FN breakdown (detection) or confusion matrix (classification) |
| `benchmark.json` | FPS, latency p50/p95/p99, GPU memory |

For interactive analysis, use `notebooks/03_model_evaluation.py`.

---

## Step 5: Deploy

Register the model to Unity Catalog and deploy to Model Serving:

```bash
python jobs/deploy.py \
    --config_path configs/my_config.yaml \
    --run_id abc123def456 \
    --model_name my_catalog.my_schema.yolos_detection \
    --endpoint_name yolos-detection-endpoint \
    --test_image /path/to/test/image.jpg
```

This:
1. Wraps the model as a PyFunc (handles base64 images, batching, etc.)
2. Registers to Unity Catalog with `champion` and `latest` aliases
3. Creates a Model Serving endpoint
4. Waits for the endpoint to become READY
5. Runs a smoke test with the test image

For step-by-step deployment, use `notebooks/04_model_deployment.py`.

### Testing the Endpoint

Once deployed, send requests with base64-encoded images:

```python
import base64, requests

with open("test.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

# Via Databricks SDK
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
response = w.serving_endpoints.query(
    name="yolos-detection-endpoint",
    dataframe_records=[{"image": b64}],
)
```

**Detection response:**
```json
{"predictions": {"boxes": [[x1,y1,x2,y2], ...], "scores": [0.95, ...], "labels": [1, ...], "num_detections": 5, "status": "success"}}
```

**Classification response:**
```json
{"predictions": {"label": 2, "label_name": "cat", "confidence": 0.97, "top_k": [...], "status": "success"}}
```

---

## Step 6: Monitor

Generate a monitoring report for a deployed endpoint:

```bash
python jobs/monitor.py --endpoint_name yolos-detection-endpoint --hours 24
```

This queries `system.serving.served_model_requests` for request volume, error rates, latency distributions, and prediction class distributions. For dashboard-style monitoring, use `notebooks/05_model_monitoring.py`.

For production alerting, set up SQL Alerts on the system tables — examples are provided in the monitoring notebook.

---

## Notebooks

| Notebook | Purpose | GPU Required |
|---|---|---|
| `01_data_exploration.py` | Dataset stats, class distribution, bbox sizes, quality checks | No |
| `02_model_training.py` | Interactive training with TrainingEngine | Yes |
| `03_model_evaluation.py` | mAP/accuracy, per-class breakdown, error analysis, benchmarks | Yes |
| `04_model_deployment.py` | PyFunc test, UC registration, endpoint deployment, smoke test | Yes (for local test) |
| `05_model_monitoring.py` | Endpoint health, request metrics, prediction distribution, alerts | No |

---

## Lakehouse App

The `lakehouse_app/` directory is a Streamlit app with 9 pages covering the full lifecycle. Deploy it as a Databricks App:

1. Upload `lakehouse_app/` to your workspace
2. Create a Databricks App pointing to it
3. Use the UI to configure, train, evaluate, register, deploy, test, and monitor models

The app runs on CPU. Heavy compute (training, evaluation) is submitted as Databricks Jobs.

---

## Available Configs

| Config | Task | Model | Notes |
|---|---|---|---|
| `detection_detr_config.yaml` | Detection | DETR ResNet-50 | Full COCO training |
| `detection_yolos_config.yaml` | Detection | YOLOS Base | Full COCO training |
| `test_detection_yolos_sanity.yaml` | Detection | YOLOS Tiny | Quick sanity check |
| `classification_vit_config.yaml` | Classification | ViT Base | ImageFolder dataset |

---

## Common Patterns

### Switch Models (Same Task)

```yaml
# Just change model_name — everything else adapts
model:
  model_name: "facebook/detr-resnet-101"  # was detr-resnet-50
```

### Custom Learning Rate Schedule

```yaml
model:
  learning_rate: 3e-5
  weight_decay: 0.01
  scheduler: cosine   # cosine with 10% warmup (default)
```

### Checkpoint to Volume

```yaml
training:
  checkpoint_dir: /tmp/checkpoints                    # Fast local storage during training
  volume_checkpoint_dir: /Volumes/.../checkpoints     # Synced to Volume for persistence
```

### Resume from Checkpoint

Set `training.resume_from_checkpoint: latest` (with `volume_checkpoint_dir`) and
training resumes from the newest complete checkpoint on the Volume, or starts
fresh when there is none — so retried AI Runtime runs pick up where they
stopped. A path resumes from that checkpoint instead. Use a new
`volume_checkpoint_dir` for each training run.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `CUDA out of memory` | Reduce `batch_size` in config, or use a larger GPU |
| `KeyError: 'classification'` | Ensure `src/tasks/classification` is imported — check `src/engine/engine.py` |
| `NCCL timeout` on multi-GPU | This is normal on first run while NCCL initializes. Increase timeout or retry. |
| `No module named 'pycocotools'` | The job environment is missing `requirements_runtime.txt`; in a notebook, run the `%pip install` cell |
| `transformers >= 5` error | The task is not on the AI v6 environment — set `databricks_ai_v6` |
| `No run manifest` | `output.results_dir` must be a Volume path shared by the training and evaluate/deploy tasks |
| Retry resumed an old run | `resume_from_checkpoint: latest` found checkpoints from a previous run; use a fresh `volume_checkpoint_dir` |
| Metrics not in `mlflow.experiment_name` | AI Runtime runs log to their own experiment (`experiment_name` in `air/*.yaml`, `experiment` in the bundle) |
