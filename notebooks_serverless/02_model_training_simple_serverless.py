# Databricks notebook source
# MAGIC %md
# MAGIC # 02. Serverless GPU Model Training (Simple MLflow Version)
# MAGIC 
# MAGIC This notebook trains computer vision models using **Databricks Serverless GPU compute** with Databricks' managed MLflow and autolog for simple integration.
# MAGIC 
# MAGIC ## Approach: Databricks Managed MLflow + Autolog + Serverless GPU
# MAGIC 
# MAGIC This version uses:
# MAGIC - **Databricks Managed MLflow**: Native integration
# MAGIC - **MLflow Autolog**: Automatic parameter and metric logging
# MAGIC - **Serverless GPU Training**: Multi-GPU training with @distributed decorator
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

from config_serverless import load_config
from tasks.detection.model import DetectionModel
from tasks.detection.data import DetectionDataModule
from tasks.detection.adapters import get_input_adapter
from training.trainer import UnifiedTrainer

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Configuration

# COMMAND ----------

# Load configuration
CATALOG = "your_catalog"
SCHEMA = "your_schema"  
VOLUME = "your_volume"
PROJECT_PATH = "cv_serverless_training"

BASE_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{PROJECT_PATH}"
CONFIG_PATH = f"{BASE_VOLUME_PATH}/configs_serverless/detection_detr_config.yaml"

# Set up volume directories
CHECKPOINT_DIR = f"{BASE_VOLUME_PATH}/checkpoints"
RESULTS_DIR = f"{BASE_VOLUME_PATH}/results"
LOGS_DIR = f"{BASE_VOLUME_PATH}/logs"

# Create directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Load configuration
if os.path.exists(CONFIG_PATH):
    config = load_config(CONFIG_PATH)
    config['training']['checkpoint_dir'] = CHECKPOINT_DIR
else:
    print("⚠️  Config file not found. Using default serverless config.")
    config = {
        'model': {
            'model_name': 'facebook/detr-resnet-50',
            'task_type': 'detection',
            'num_classes': 91,
            'image_size': 800
        },
        'data': {
            'batch_size': 2,
            'num_workers': 4,
            'image_size': 800,
            'train_data_path': f"{BASE_VOLUME_PATH}/data/train",
            'train_annotation_file': f"{BASE_VOLUME_PATH}/data/train_annotations.json",
            'val_data_path': f"{BASE_VOLUME_PATH}/data/val",
            'val_annotation_file': f"{BASE_VOLUME_PATH}/data/val_annotations.json"
        },
        'training': {
            'max_epochs': 10,  # Reduced for simple example
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'checkpoint_dir': CHECKPOINT_DIR,
            'distributed': True,
            'use_serverless_gpu': True,
            'serverless_gpu_type': 'A10',
            'serverless_gpu_count': 4,
            'monitor_metric': 'val_map',
            'monitor_mode': 'max',
            'early_stopping_patience': 5,
            'log_every_n_steps': 50
        },
        'output': {
            'results_dir': RESULTS_DIR
        }
    }

print("✅ Configuration loaded successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Setup Model and Data

# COMMAND ----------

# Prepare model config
model_config = config["model"].copy()
model_config["num_workers"] = config["data"]["num_workers"]

# Create model
model = DetectionModel(model_config)
print(f"✅ Model created: {config['model']['model_name']}")

# Setup adapter
from tasks.detection.adapters import get_input_adapter
adapter = get_input_adapter(
    config["model"]["model_name"], 
    image_size=config["data"].get("image_size", 800)
)

# Create data module
data_module = DetectionDataModule(config["data"])
data_module.adapter = adapter
data_module.setup()

print(f"✅ Data module created with {len(data_module.train_dataset)} train samples")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Start Serverless GPU Training

# COMMAND ----------

# Create trainer config
trainer_config = {
    'task': config['model']['task_type'],
    'model_name': config['model']['model_name'],
    'max_epochs': config['training']['max_epochs'],
    'log_every_n_steps': config['training']['log_every_n_steps'],
    'monitor_metric': config['training']['monitor_metric'],
    'monitor_mode': config['training']['monitor_mode'],
    'early_stopping_patience': config['training']['early_stopping_patience'],
    'checkpoint_dir': CHECKPOINT_DIR,
    'save_top_k': 3,
    'distributed': config['training']['distributed'],
    'use_serverless_gpu': config['training']['use_serverless_gpu'],
    'serverless_gpu_type': config['training']['serverless_gpu_type'],
    'serverless_gpu_count': config['training']['serverless_gpu_count']
}

print("✅ Serverless training components created")
print(f"   GPU Type: {config['training']['serverless_gpu_type']}")
print(f"   GPU Count: {config['training']['serverless_gpu_count']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Distributed Training Function

# COMMAND ----------

from serverless_gpu import distributed

@distributed(
    gpus=config['training']['serverless_gpu_count'], 
    gpu_type=config['training']['serverless_gpu_type'], 
    remote=True
)
def distributed_train(config_dict, DetectionModel, DetectionDataModule, UnifiedTrainer, get_input_adapter):
    """Distributed training function for Serverless GPU using UnifiedTrainer."""
    import lightning as pl
    from lightning.pytorch.loggers import MLFlowLogger
    
    # Setup model with proper configuration (like setup_model function)
    model_config = config_dict['model_config'].copy()
    model_config["num_workers"] = config_dict['data_config']["num_workers"]
    model = DetectionModel(model_config)
    
    # Setup data module with adapter (like setup_data_module function)
    adapter = get_input_adapter(
        config_dict['model_config']["model_name"], 
        image_size=config_dict['data_config'].get("image_size", 800)
    )
    
    if adapter is None:
        raise ValueError(f"Could not create adapter for model: {config_dict['model_config']['model_name']}")
    
    # Create data module with data config only
    data_module = DetectionDataModule(config_dict['data_config'])
    
    # Assign adapter to data module (CRITICAL!)
    data_module.adapter = adapter
    
    # Setup the data module to create datasets (CRITICAL!)
    data_module.setup()
    
    # Create logger
    logger = MLFlowLogger(
        experiment_name=config_dict.get('mlflow_experiment_name', 'serverless-gpu-simple-training'),
        run_name=config_dict.get('mlflow_run_name', 'simple-distributed-training'),
        tags=config_dict.get('mlflow_tags', {})
    )
    
    # Create UnifiedTrainer and run training
    trainer = UnifiedTrainer(
        config=config_dict,
        model=model,
        data_module=data_module,
        logger=logger
    )
    
    # Run training using the existing UnifiedTrainer
    return trainer.train()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Start Training

# COMMAND ----------

print("🎯 Starting simple serverless GPU training...")
print("=" * 50)

try:
    print("✅ Serverless GPU configuration ready.")
    print("🚀 Starting distributed training...")
    
    # Run distributed training directly - pass the class definitions and functions
    distributed_result = distributed_train.distributed(
        trainer_config,
        DetectionModel,
        DetectionDataModule, 
        UnifiedTrainer,
        get_input_adapter
    )
    
    print("\n✅ Distributed training completed successfully!")
    print(f"Final metrics: {distributed_result}")
    
except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Training Summary

# COMMAND ----------

print("=" * 50)
print("SIMPLE SERVERLESS GPU TRAINING SUMMARY")
print("=" * 50)

print(f"✅ Model: {config['model']['model_name']}")
print(f"✅ Task: {config['model']['task_type']}")
print(f"✅ Training completed successfully")
print(f"✅ Serverless GPU: {config['training']['serverless_gpu_type']} x {config['training']['serverless_gpu_count']}")

print(f"\n📁 Output Locations:")
print(f"   Checkpoints: {CHECKPOINT_DIR}")
print(f"   Results: {RESULTS_DIR}")

print("\n🎉 Simple serverless GPU training completed!")
print("Check MLflow UI for detailed metrics and logs.")

print("=" * 50)
