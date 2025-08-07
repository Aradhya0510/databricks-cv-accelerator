# Databricks notebook source
# MAGIC %md
# MAGIC # 02. DETR Model Training (Simple MLflow Version)
# MAGIC 
# MAGIC This notebook trains a DETR model using Databricks' managed MLflow with autolog for simple integration.
# MAGIC 
# MAGIC ## Approach: Databricks Managed MLflow + Autolog
# MAGIC 
# MAGIC This version uses:
# MAGIC - **Databricks Managed MLflow**: Native integration
# MAGIC - **MLflow Autolog**: Automatic parameter and metric logging
# MAGIC - **Simple Lightning Trainer**: No complex logger setup
# MAGIC 
# MAGIC ---

# COMMAND ----------

import sys
import os
import torch
import lightning
import mlflow
import numpy as np
from pathlib import Path

# Enable MLflow autolog for automatic parameter and metric logging
mlflow.pytorch.autolog()

# Add the src directory to Python path
sys.path.append('/Workspace/Repos/your-repo/Databricks_CV_ref/src')

from config import load_config
from tasks.detection.model import DetectionModel
from tasks.detection.data import DetectionDataModule
from training.trainer import UnifiedTrainer

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Configuration

# COMMAND ----------

# Load configuration
CATALOG = "your_catalog"
SCHEMA = "your_schema"  
VOLUME = "your_volume"
PROJECT_PATH = "cv_detr_training"

BASE_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{PROJECT_PATH}"
CONFIG_PATH = f"{BASE_VOLUME_PATH}/configs/detection_detr_config.yaml"

# Set up directories
CHECKPOINT_DIR = f"{BASE_VOLUME_PATH}/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Load config
if os.path.exists(CONFIG_PATH):
    config = load_config(CONFIG_PATH)
    config['training']['checkpoint_dir'] = CHECKPOINT_DIR
    
    # Fix config structure
    if 'model' in config and 'epochs' in config['model']:
        if 'training' not in config:
            config['training'] = {}
        config['training']['max_epochs'] = config['model']['epochs']
else:
    from config import get_default_config
    config = get_default_config("detection")
    config['training']['checkpoint_dir'] = CHECKPOINT_DIR

print("✅ Configuration loaded successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Initialize Model and Data

# COMMAND ----------

# Initialize model
print("🔧 Initializing DETR model...")
model_config = config["model"].copy()
model_config["num_workers"] = config["data"]["num_workers"]
model = DetectionModel(model_config)
print(f"✅ Model initialized: {config['model']['model_name']}")

# Initialize data module
print("📊 Setting up data module...")
from tasks.detection.adapters import get_input_adapter

# Fix image size handling - handle both list and scalar formats
image_size = config["data"].get("image_size", 800)
if isinstance(image_size, list):
    image_size = image_size[0]  # Use first value if it's a list
elif isinstance(image_size, dict):
    image_size = image_size.get("height", 800)  # Use height if it's a dict

adapter = get_input_adapter(config["model"]["model_name"], image_size=image_size)
data_module = DetectionDataModule(config["data"])
data_module.adapter = adapter
data_module.setup('fit')
print(f"✅ Data module setup: {len(data_module.train_dataset)} train, {len(data_module.val_dataset)} val samples")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Setup Trainer

# COMMAND ----------

# Setup trainer with proper constructor parameters
print("🚀 Setting up trainer...")

# Create trainer config with required fields
trainer_config = {
    'task': config['model']['task_type'],
    'model_name': config['model']['model_name'],
    'max_epochs': config['training']['max_epochs'],
    'log_every_n_steps': config['training'].get('log_every_n_steps', 50),
    'monitor_metric': config['training'].get('monitor_metric', 'val_loss'),
    'monitor_mode': config['training'].get('monitor_mode', 'min'),
    'early_stopping_patience': config['training'].get('early_stopping_patience', 10),
    'checkpoint_dir': CHECKPOINT_DIR,
    'save_top_k': 3,
    'distributed': config['training'].get('distributed', False)
}

trainer = UnifiedTrainer(
    config=trainer_config,
    model=model,
    data_module=data_module
)
print("✅ Trainer setup complete!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Start Training with MLflow Autolog

# COMMAND ----------

print("🎯 Starting DETR training with MLflow autolog...")
print("=" * 60)

# Start MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        'model_name': config['model']['model_name'],
        'task_type': config['model']['task_type'],
        'num_classes': config['model']['num_classes'],
        'max_epochs': config['training']['max_epochs'],
        'learning_rate': config['training']['learning_rate'],
        'batch_size': config['data']['batch_size'],
        'num_workers': config['data']['num_workers'],
        'image_size': image_size
    })
    
    # Start training
    result = trainer.train()
    
    print("✅ Training completed successfully!")
    print(f"📊 MLflow run ID: {mlflow.active_run().info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Training Summary

# COMMAND ----------

print("=" * 60)
print("DETR TRAINING SUMMARY")
print("=" * 60)

print(f"✅ Model: {config['model']['model_name']}")
print(f"✅ Task: {config['model']['task_type']}")
print(f"✅ Classes: {config['model']['num_classes']}")
print(f"✅ Epochs: {config['training']['max_epochs']}")
print(f"✅ Learning Rate: {config['training']['learning_rate']}")
print(f"✅ Batch Size: {config['data']['batch_size']}")
print(f"✅ Image Size: {image_size}")
print(f"✅ Training Success: {'Yes' if result else 'No'}")

print(f"\n📁 Checkpoints: {CHECKPOINT_DIR}")
print("=" * 60) 