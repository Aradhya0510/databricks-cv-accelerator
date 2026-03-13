# Databricks CV Pipeline — Agent Instructions

## Project Overview

A modular computer vision fine-tuning framework for Databricks. Supports object detection (DETR, YOLOS) and image classification (ViT, any HF `AutoModelForImageClassification` model) with HF Trainer backend. Extensible to new tasks via the TaskRegistry.

**Key architecture:** Config YAML → `PipelineConfig` (Pydantic) → `TrainingEngine` → `CVTrainer` (HF Trainer) with native DDP for multi-GPU.

## Repository Layout

```
src/
├── config/schema.py          # Pydantic v2 config models + YAML loader
├── registry.py               # TaskRegistry with @register decorator
├── engine/
│   ├── engine.py             # TrainingEngine: config → train → metrics
│   ├── trainer.py            # CVTrainer — generic HF Trainer with task-provided hooks
│   └── callbacks.py          # VolumeCheckpointCallback, EarlyStoppingCallback
├── tasks/
│   ├── detection/
│   │   ├── __init__.py       # DetectionTask: model, datasets, loss, eval hooks
│   │   ├── adapters.py       # DETR/YOLOS input/output adapters
│   │   ├── collate.py        # Standalone collate function
│   │   └── data.py           # COCODetectionDataset (plain torch Dataset)
│   └── classification/
│       ├── __init__.py       # ClassificationTask: model, datasets, loss, eval hooks
│       ├── collate.py        # Standalone collate function
│       └── data.py           # ImageFolderClassificationDataset (ImageFolder-style)
├── evaluation/
│   ├── __init__.py           # Exports EvaluationEngine
│   └── engine.py             # EvaluationEngine: evaluate, error_analysis, benchmark
├── serving/
│   ├── __init__.py           # Exports PyFunc wrappers + deploy/register functions
│   ├── pyfunc.py             # DetectionPyFuncModel, ClassificationPyFuncModel
│   ├── registration.py       # register_model() — task-aware UC registration
│   └── deployment.py         # deploy_endpoint(), wait_for_ready(), test_endpoint()
├── monitoring/
│   ├── __init__.py           # Exports EndpointMonitor
│   └── endpoint_monitor.py   # Health checks, request metrics, drift scoring
└── utils/
    └── environment.py        # is_databricks_job(), get_gpu_count(), stage_data_to_local()
jobs/
├── train.py                  # Training entry point: HF Trainer + native DDP
├── evaluate.py               # Evaluation entry point: metrics, error analysis, benchmark
├── deploy.py                 # Registration + deployment entry point
└── monitor.py                # Monitoring report entry point
configs/
├── detection_detr_config.yaml
├── detection_yolos_config.yaml
├── classification_vit_config.yaml
└── test_detection_yolos_sanity.yaml
```

## Running Training Locally

```bash
# Detection
python jobs/train.py --config_path configs/test_detection_yolos_sanity.yaml
python jobs/train.py --config_path configs/detection_yolos_config.yaml --num_gpus 4

# Classification
python jobs/train.py --config_path configs/classification_vit_config.yaml
python jobs/train.py --config_path configs/classification_vit_config.yaml --num_gpus 4
```

## Multi-GPU Strategy

- **Single-node multi-GPU (default):** HF Trainer handles DDP natively. No Spark/TorchDistributor overhead. Pass `--num_gpus N` or let it auto-detect.
- **Multi-node (opt-in only):** Pass `--distributed torchd` to use TorchDistributor across Spark workers. Only for scaling beyond a single machine.

Do NOT use TorchDistributor for single-machine multi-GPU. Native DDP is the Databricks-recommended path.

## Config Format

All YAML configs follow this structure (validated by `PipelineConfig` in `src/config/schema.py`):

```yaml
model:
  model_name: hustvl/yolos-base      # HuggingFace model ID
  task_type: detection                # Task registry key
  num_classes: 80
  image_size: 512                     # int or [H, W] list
  learning_rate: 1e-4
  weight_decay: 1e-4
  scheduler: cosine
  epochs: 50
  confidence_threshold: 0.7
  iou_threshold: 0.5
data:
  train_data_path: /Volumes/catalog/schema/volume/data/train/
  train_annotation_file: /Volumes/catalog/schema/volume/data/annotations/train.json
  val_data_path: /Volumes/catalog/schema/volume/data/val/
  val_annotation_file: /Volumes/catalog/schema/volume/data/annotations/val.json
  batch_size: 4
  num_workers: 4
  model_name: hustvl/yolos-base       # Used by adapters
  image_size: [512, 512]
training:
  max_epochs: 50
  early_stopping_patience: 10
  monitor_metric: val_map
  monitor_mode: max
  checkpoint_dir: /tmp/checkpoints
  volume_checkpoint_dir: /Volumes/catalog/schema/volume/checkpoints
  save_top_k: 3
  log_every_n_steps: 50
mlflow:
  experiment_name: /Users/user@databricks.com/my_experiment
  run_name: yolos_base_run1
  tags:
    framework: hf_trainer
    model: yolos
    dataset: coco
output:
  results_dir: /tmp/results
serving:                                      # Optional — for deployment
  registered_model_name: catalog.schema.model
  endpoint_name: my-endpoint
  workload_size: Small
  scale_to_zero: true
monitoring:                                   # Optional — for observability
  drift_threshold: 0.1
  error_rate_threshold: 0.05
  latency_p95_threshold_ms: 500
```

### Classification Config

Classification uses `task_type: classification` and ImageFolder-style data layout
(`root_dir/class_name/image.jpg`). No annotation files needed.

```yaml
model:
  model_name: google/vit-base-patch16-224
  task_type: classification
  num_classes: 10
  image_size: 224
  learning_rate: 5e-5
  weight_decay: 1e-2
  scheduler: cosine
  epochs: 20
  dropout: 0.1
data:
  train_data_path: /Volumes/catalog/schema/volume/data/classification/train/
  val_data_path: /Volumes/catalog/schema/volume/data/classification/val/
  batch_size: 32
  num_workers: 4
  model_name: google/vit-base-patch16-224
  image_size: [224, 224]
training:
  max_epochs: 20
  early_stopping_patience: 5
  monitor_metric: val_accuracy
  monitor_mode: max
```

---

# MCP-Based Fine-Tuning Workflow

Any agent with access to the Databricks MCP server can initiate, monitor, and complete a fine-tuning run using the workflow below.

## Step 1: Pre-flight Checks

### 1a. Get current user identity
```
mcp__databricks__get_current_user()
→ {username, home_path}
```
Use `username` for MLflow experiment paths: `/Users/{username}/cv_detection`.

### 1b. Validate training data exists on Volumes
```
mcp__databricks__list_volume_files(volume_path="/Volumes/catalog/schema/volume/data/train/")
→ {files: [...], truncated: bool}

mcp__databricks__get_volume_file_info(volume_path="/Volumes/catalog/schema/volume/data/annotations/train.json")
→ {name, path, file_size, ...}
```
Verify both image directory and annotation file exist. If missing, inform the user before proceeding.

### 1c. Find or provision a GPU cluster
```
mcp__databricks__list_clusters()
→ [{cluster_id, cluster_name, state, node_type_id, num_workers}, ...]
```
Look for a **RUNNING** cluster with a GPU node type (e.g., `g5.xlarge`, `g5.4xlarge`, `Standard_NC*`). If none running:
```
mcp__databricks__start_cluster(cluster_id="...")
→ {cluster_id, state: "PENDING", ...}
```
Then poll until RUNNING:
```
mcp__databricks__get_cluster_status(cluster_id="...")
→ {state: "RUNNING"}
```
**Always ask user confirmation before starting a cluster.**

## Step 2: Prepare Config and Upload Code

### 2a. Generate config YAML
Given the user's requirements (model, dataset paths, hyperparameters), generate a valid YAML config following the format above. Write it locally, e.g. `configs/my_run.yaml`.

### 2b. Upload project code to Workspace
```
mcp__databricks__upload_folder(
  local_folder=".",
  workspace_folder="/Workspace/Users/{username}/Databricks_CV_ref"
)
```

### 2c. Upload config to Volumes (for persistence)
```
mcp__databricks__upload_to_volume(
  local_path="configs/my_run.yaml",
  volume_path="/Volumes/catalog/schema/volume/configs/my_run.yaml"
)
```

## Step 3: Create and Run Training Job

### 3a. Create job (idempotent — returns existing if name matches)
```
mcp__databricks__manage_jobs(
  action="create",
  name="cv-finetune-{model}-{timestamp}",
  tasks=[{
    "task_key": "train",
    "description": "Fine-tune detection model",
    "spark_python_task": {
      "python_file": "/Workspace/Users/{username}/Databricks_CV_ref/jobs/train.py",
      "parameters": [
        "--config_path", "/Workspace/Users/{username}/Databricks_CV_ref/configs/my_run.yaml",
        "--num_gpus", "4"
      ]
    },
    "existing_cluster_id": "{cluster_id}"
  }],
  tags={"task": "detection", "model": "yolos", "framework": "hf_trainer"}
)
→ {job_id: 12345}
```

**Cluster selection in tasks:**
- `existing_cluster_id`: Use a running GPU cluster (simplest).
- `new_cluster`: Provision a fresh cluster for the job (use for production runs):
  ```json
  "new_cluster": {
    "spark_version": "16.4.x-gpu-ml-scala2.12",
    "node_type_id": "g5.4xlarge",
    "num_workers": 0,
    "data_security_mode": "SINGLE_USER"
  }
  ```

### 3b. Trigger the job
```
mcp__databricks__manage_job_runs(
  action="run_now",
  job_id=12345,
  python_params=["--config_path", "/Workspace/.../configs/my_run.yaml"]
)
→ {run_id: 67890}
```

### 3c. Monitor the run
**Option A — poll manually:**
```
mcp__databricks__manage_job_runs(action="get", run_id=67890)
→ {state: {life_cycle_state: "RUNNING", ...}, run_page_url: "..."}
```

**Option B — wait for completion (blocking):**
```
mcp__databricks__manage_job_runs(
  action="wait",
  run_id=67890,
  timeout=7200,
  poll_interval=30
)
→ {success: true, duration_seconds: 1234, run_page_url: "..."}
```

**Option C — get logs on failure:**
```
mcp__databricks__manage_job_runs(action="get_output", run_id=67890)
→ {output: "...", error: "..."}
```

## Step 4: Post-Training — Verify Metrics

### 4a. Check MLflow experiment
```
mcp__databricks__execute_sql(
  sql_query="SELECT run_id, status, end_time FROM system.ml.model_training_runs WHERE experiment_name = '/Users/{username}/cv_detection' ORDER BY end_time DESC LIMIT 5"
)
```

Or use the run page URL from step 3c to direct the user to MLflow UI.

### 4b. Verify metrics logged
Key metrics to check: `eval_map`, `eval_map_50`, `eval_map_75`, `eval_loss`.

## Step 5: Model Registration (Optional)

Register the best model to Unity Catalog:
```
mcp__databricks__execute_sql(
  sql_query="CREATE MODEL IF NOT EXISTS catalog.schema.yolos_detection COMMENT 'YOLOS object detection model'"
)
```

The HF Trainer logs the model artifact to MLflow automatically via `mlflow.transformers.log_model()`. Register a specific run's model via MLflow UI or:
```python
import mlflow
mlflow.register_model("runs:/{run_id}/model", "catalog.schema.yolos_detection")
```

## Step 6: Model Serving (Optional)

Check existing endpoints:
```
mcp__databricks__list_serving_endpoints()
```

The model can be served via Databricks Model Serving after registration.

## Step 7: App Deployment (Optional)

Deploy the Lakehouse App for a UI over the pipeline:
```
mcp__databricks__upload_folder(
  local_folder="lakehouse_app",
  workspace_folder="/Workspace/Users/{username}/Databricks_CV_ref/lakehouse_app"
)

mcp__databricks__create_or_update_app(
  name="cv-pipeline",
  source_code_path="/Workspace/Users/{username}/Databricks_CV_ref/lakehouse_app",
  description="CV fine-tuning pipeline dashboard"
)
```

---

## Quick Reference: Common Agent Scenarios

### "Fine-tune YOLOS on my COCO dataset"
1. `get_current_user` → get username
2. `list_volume_files` → verify data paths
3. Generate config YAML with user's paths
4. `upload_folder` → sync code to workspace
5. `manage_jobs(action="create")` → create job
6. `manage_job_runs(action="run_now")` → start training
7. `manage_job_runs(action="wait")` → wait for completion
8. Report metrics from run output

### "Check status of my training run"
1. `manage_job_runs(action="get", run_id=...)` → get state
2. If failed: `manage_job_runs(action="get_output")` → get error logs

### "Deploy my trained model"
1. `list_serving_endpoints` → check existing
2. Guide user through MLflow model registration
3. Create serving endpoint via Databricks UI (or SDK)

---

## Development Notes

- **Detection and classification** both use HF Trainer. Other tasks (segmentation) can be added via `TaskRegistry`.
- Classification uses `ImageFolderClassificationDataset` (ImageFolder-style directory layout, no annotation files).
- Detection uses `COCODetectionDataset` (COCO JSON annotations).
- `EvaluationEngine`, `register_model()`, and `deploy_endpoint()` are all task-aware — dispatch based on `config.model.task_type`.
- `ClassificationPyFuncModel` and `DetectionPyFuncModel` are separate PyFunc wrappers (different output schemas).
- Config YAML format is unchanged — `PipelineConfig` (Pydantic) accepts the same structure as the old `load_config()` dict-based loader. New `serving` and `monitoring` sections are optional with defaults.
