# Databricks notebook source
# MAGIC %md
# MAGIC # 02. Model Training (HF Trainer on AI Runtime)
# MAGIC
# MAGIC Trains an object detection model with `transformers.Trainer` on Databricks
# MAGIC AI Runtime. Attach to a **1xA10** or **1xH100** accelerator to train on one GPU,
# MAGIC or to **8xH100** / **8xB300** to train with DDP across all eight.
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC 1. Load a validated Pydantic config from YAML
# MAGIC 2. Create a `TrainingEngine` (one-liner)
# MAGIC 3. Call `engine.train()` — one GPU in this process, several through `@distributed`
# MAGIC 4. Review MLflow metrics
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Environment
# MAGIC
# MAGIC Run from a Git folder clone of this repo, attached to **AI Runtime**
# MAGIC (serverless GPU) with the **AI v6** base environment, which already ships
# MAGIC torch, transformers v5 and MLflow. This installs the few packages it lacks.

# COMMAND ----------

# MAGIC %pip install -q -r ../requirements_runtime.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

import sys
import os
import torch
from pathlib import Path

# The notebook runs from notebooks/ in the Git folder; only the repo root goes
# on the path, so `src` imports resolve the same way the job entry points do.
REPO_ROOT = os.path.dirname(os.getcwd())
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.config.schema import load_config
from src.engine import TrainingEngine

# Pipeline config: a Volume path, or a path relative to the repo root.
dbutils.widgets.text("config_path", "", "Config YAML")
CONFIG_PATH = dbutils.widgets.get("config_path")
if not CONFIG_PATH:
    raise ValueError("Set the config_path widget to your pipeline config YAML.")
if not os.path.isabs(CONFIG_PATH):
    CONFIG_PATH = os.path.join(REPO_ROOT, CONFIG_PATH)

# Load and validate configuration
config = load_config(CONFIG_PATH)

print(f"Model:       {config.model.model_name}")
print(f"Task:        {config.model.task_type}")
print(f"Classes:     {config.model.num_classes}")
print(f"Epochs:      {config.training.max_epochs}")
print(f"Batch size:  {config.data.batch_size}")
print(f"LR:          {config.model.learning_rate}")
print(f"Monitor:     {config.training.monitor_metric} ({config.training.monitor_mode})")
if torch.cuda.is_available():
    print(f"GPUs:        {torch.cuda.device_count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Training

# COMMAND ----------

engine = TrainingEngine(config)

# Train on every GPU of the attached accelerator. On 8xH100 this runs one
# process per GPU through serverless_gpu's @distributed; pass num_gpus=1 to
# debug on a single GPU first.
metrics = engine.train()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Results

# COMMAND ----------

print("=" * 60)
print("TRAINING RESULTS")
print("=" * 60)

if metrics:
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
else:
    print("  No metrics returned — check MLflow for details.")

print("=" * 60)
print("Check MLflow UI for detailed metrics, curves, and model artifacts.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Understanding Training
# MAGIC
# MAGIC ### Key Concepts
# MAGIC
# MAGIC **HF Trainer backend:**
# MAGIC - Handles gradient accumulation, mixed precision, and logging automatically
# MAGIC - Each task provides its own loss and eval hooks to the generic `CVTrainer`
# MAGIC - `report_to="mlflow"` logs all metrics to MLflow automatically
# MAGIC
# MAGIC **Multi-GPU via `@distributed`:**
# MAGIC - With more than one GPU attached, `TrainingEngine` launches one DDP process per GPU with `serverless_gpu`'s `@distributed`
# MAGIC - `/Volumes/` inputs are copied to local disk once, in parallel, before the workers start (`data.stage_to_local`)
# MAGIC - The MLflow run is created by `@distributed` in `mlflow.experiment_name` from the config
# MAGIC - Notebook sessions end after two days; for longer runs submit `air/train.yaml` with `databricks air run` and set `training.resume_from_checkpoint: latest`
# MAGIC
# MAGIC **Monitoring metrics:**
# MAGIC - `eval_map`: Mean Average Precision (primary metric)
# MAGIC - `eval_map_50`, `eval_map_75`: mAP at different IoU thresholds
# MAGIC - `eval_loss`: Total loss (classification + bbox)
# MAGIC - Per-class mAP and mAR metrics
