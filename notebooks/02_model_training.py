# Databricks notebook source
# MAGIC %md
# MAGIC # 02. Model Training (HF Trainer + TorchDistributor)
# MAGIC
# MAGIC This notebook trains an object detection model using the new HF Trainer backend.
# MAGIC It replaces PyTorch Lightning with `transformers.Trainer` and uses
# MAGIC `TorchDistributor` for multi-GPU DDP when needed.
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC 1. Load a validated Pydantic config from YAML
# MAGIC 2. Create a `TrainingEngine` (one-liner)
# MAGIC 3. Call `engine.train()` — handles single/multi-GPU automatically
# MAGIC 4. Review MLflow metrics
# MAGIC
# MAGIC ---

# COMMAND ----------

# (Databricks only) Install requirements and restart Python if running interactively
# %pip install -r "../requirements.txt"
# dbutils.library.restartPython()
# COMMAND ----------


# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

import sys
import os
import torch
from pathlib import Path

# Add the src directory to Python path
sys.path.append('/Workspace/Repos/your-repo/Databricks_CV_ref/src')
sys.path.append('/Workspace/Repos/your-repo/Databricks_CV_ref')

from src.config.schema import load_config
from src.engine import TrainingEngine

# --- Paths (customise for your workspace) ---
CATALOG = "your_catalog"
SCHEMA = "your_schema"
VOLUME = "your_volume"
PROJECT_PATH = "cv_detr_training"

BASE_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{PROJECT_PATH}"
CONFIG_PATH = f"{BASE_VOLUME_PATH}/configs/detection_detr_config.yaml"

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

# Train — auto-detects GPU count, uses TorchDistributor for multi-GPU
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
# MAGIC - Custom `DetectionTrainer` overrides the evaluation loop for mAP computation
# MAGIC - `report_to="mlflow"` logs all metrics to MLflow automatically
# MAGIC
# MAGIC **Multi-GPU via TorchDistributor:**
# MAGIC - When >1 GPU detected, `TrainingEngine` uses `TorchDistributor` for DDP
# MAGIC - Data is staged from `/Volumes/` to `/tmp/` before forking workers
# MAGIC - NCCL env vars are set automatically for Databricks networking
# MAGIC
# MAGIC **Monitoring metrics:**
# MAGIC - `eval_map`: Mean Average Precision (primary metric)
# MAGIC - `eval_map_50`, `eval_map_75`: mAP at different IoU thresholds
# MAGIC - `eval_loss`: Total loss (classification + bbox)
# MAGIC - Per-class mAP and mAR metrics
