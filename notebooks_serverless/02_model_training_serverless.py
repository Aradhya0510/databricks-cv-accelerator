# Databricks notebook source
# MAGIC %md
# MAGIC # 02. Serverless GPU Model Training
# MAGIC 
# MAGIC This notebook trains computer vision models using **Databricks Serverless GPU compute**. We'll cover model initialization, training configuration, monitoring, and best practices for serverless GPU training.
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC **Serverless GPU Training Process:**
# MAGIC 1. **Model Initialization**: Load pre-trained models with proper configuration
# MAGIC 2. **Data Loading**: Efficient data loading with preprocessing
# MAGIC 3. **Serverless Training Setup**: Multi-GPU training with @distributed decorator
# MAGIC 4. **Monitoring**: Real-time metrics and MLflow tracking
# MAGIC 5. **Checkpointing**: Automatic model saving and early stopping
# MAGIC 
# MAGIC ### Key Serverless GPU Concepts:
# MAGIC - **@distributed Decorator**: Handles distributed training setup automatically
# MAGIC - **Automatic Scaling**: Scales based on configured GPU count
# MAGIC - **Environment Isolation**: Clean, reproducible training environments
# MAGIC - **Cost Optimization**: Pay only for actual GPU usage time
# MAGIC 
# MAGIC ## What This Notebook Does
# MAGIC 
# MAGIC 1. **Model Setup**: Initialize model with proper configuration
# MAGIC 2. **Training Configuration**: Set up serverless trainer with callbacks
# MAGIC 3. **Serverless GPU Training**: Configure distributed training with serverless GPU
# MAGIC 4. **Monitoring**: Real-time metrics tracking and visualization
# MAGIC 5. **Model Saving**: Automatic checkpointing and model registration
# MAGIC 
# MAGIC ---

# COMMAND ----------

# (Databricks only) Install requirements and restart Python if running interactively
# %pip install -r "../requirements.txt"
# dbutils.library.restartPython()
# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup Serverless GPU Environment
# MAGIC 
# MAGIC **Important**: Before running this notebook, you need to configure the Serverless GPU environment to include your workspace:
# MAGIC 
# MAGIC 1. **Select Serverless GPU**: From the notebook's compute selector, select **Serverless GPU**
# MAGIC 2. **Open Environment Panel**: Click the ⚙️ to open the **Environment** side panel
# MAGIC 3. **Add Workspace Path**: In the **Dependencies** section, add your workspace path:
# MAGIC    - Click **Add dependency**
# MAGIC    - Select **Workspace** 
# MAGIC    - Enter: `/Workspace/Users/your_username/Computer Vision/databricks-cv-accelerator`
# MAGIC 4. **Apply Changes**: Click **Apply** and then **Confirm**
# MAGIC 
# MAGIC This makes the `src/` directory available in the serverless GPU environment, allowing the distributed training to import your custom modules.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Import Dependencies and Load Configuration

# COMMAND ----------

import sys
import os
import torch
import lightning
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the workspace to Python path for serverless GPU environment
# This ensures the src/ directory is available for imports
workspace_path = "/Workspace/Users/<your_username>/Computer Vision/databricks-cv-accelerator"
if workspace_path not in sys.path:
    sys.path.append(workspace_path)
    print(f"✅ Added workspace to Python path: {workspace_path}")
else:
    print(f"✅ Workspace already in Python path: {workspace_path}")

from src.config_serverless import load_config
from src.tasks.detection.model import DetectionModel
from src.tasks.detection.data import DetectionDataModule
from src.training.trainer import UnifiedTrainer
from lightning.pytorch.loggers import MLFlowLogger

# Load configuration from previous notebooks
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

print(f"📁 Volume directories created:")
print(f"   Checkpoints: {CHECKPOINT_DIR}")
print(f"   Results: {RESULTS_DIR}")
print(f"   Logs: {LOGS_DIR}")

# Load configuration
if os.path.exists(CONFIG_PATH):
    config = load_config(CONFIG_PATH)
    # Update checkpoint directory to use volume path
    config['training']['checkpoint_dir'] = CHECKPOINT_DIR
    
    # Fix config structure - move epochs from model to training if needed
    if 'model' in config and 'epochs' in config['model']:
        if 'training' not in config:
            config['training'] = {}
        config['training']['max_epochs'] = config['model']['epochs']
        print(f"✅ Fixed config: moved epochs ({config['model']['epochs']}) from model to training.max_epochs")
else:
    print("⚠️  Config file not found. Using default detection config.")
    # Create a basic config for demonstration
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
            'max_epochs': 50,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'checkpoint_dir': CHECKPOINT_DIR,
            'distributed': True,
            'use_serverless_gpu': True,
            'serverless_gpu_type': 'A10',
            'serverless_gpu_count': 4,
            'monitor_metric': 'val_map',
            'monitor_mode': 'max',
            'early_stopping_patience': 20,
            'log_every_n_steps': 50
        },
        'output': {
            'results_dir': RESULTS_DIR
        }
    }

print("✅ Configuration loaded successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Model and Data Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initialize Model

# COMMAND ----------

def setup_model():
    """Initialize the detection model with proper configuration."""
    
    # Prepare model config with num_workers from data config
    model_config = config["model"].copy()
    model_config["num_workers"] = config["data"]["num_workers"]
    
    # Create model
    model = DetectionModel(model_config)
    
    print(f"✅ Model initialized: {config['model']['model_name']}")
    print(f"   Task: {config['model']['task_type']}")
    print(f"   Classes: {config['model']['num_classes']}")
    print(f"   Image size: {config['model'].get('image_size', 800)}")
    
    return model

model = setup_model()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup Data Module

# COMMAND ----------

def setup_data_module():
    """Initialize the data module with proper configuration."""
    
    # Setup adapter first
    from tasks.detection.adapters import get_input_adapter
    adapter = get_input_adapter(
        config["model"]["model_name"], 
        image_size=config["data"].get("image_size", 800)
    )
    
    if adapter is None:
        raise ValueError(f"Could not create adapter for model: {config['model']['model_name']}")
    
    # Create data module with data config only
    data_module = DetectionDataModule(config["data"])
    
    # Assign adapter to data module
    data_module.adapter = adapter
    
    # Setup the data module to create datasets
    data_module.setup()
    
    print(f"✅ Data module initialized")
    print(f"   Train samples: {len(data_module.train_dataset) if hasattr(data_module, 'train_dataset') else 'N/A'}")
    print(f"   Val samples: {len(data_module.val_dataset) if hasattr(data_module, 'val_dataset') else 'N/A'}")
    print(f"   Batch size: {config['data']['batch_size']}")
    print(f"   Image size: {config['data'].get('image_size', 800)}")
    
    return data_module

data_module = setup_data_module()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Serverless GPU Training Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initialize Serverless Trainer

# COMMAND ----------

def setup_serverless_trainer():
    """Initialize the serverless trainer with proper configuration."""
    
    # Create trainer config for the serverless approach
    trainer_config = {
        'task': config['model']['task_type'],
        'model_name': config['model']['model_name'],
        'max_epochs': config['training']['max_epochs'],
        'log_every_n_steps': config['training']['log_every_n_steps'],
        'monitor_metric': config['training']['monitor_metric'],
        'monitor_mode': config['training']['monitor_mode'],
        'early_stopping_patience': config['training']['early_stopping_patience'],
        'checkpoint_dir': f"{BASE_VOLUME_PATH}/checkpoints",
        'volume_checkpoint_dir': f"{BASE_VOLUME_PATH}/volume_checkpoints",
        'save_top_k': 3,
        'distributed': config['training']['distributed'],
        'use_serverless_gpu': config['training']['use_serverless_gpu'],
        'serverless_gpu_type': config['training']['serverless_gpu_type'],
        'serverless_gpu_count': config['training']['serverless_gpu_count'],
        'data_config': config['data'],
        'model_config': config['model']
    }
    
    # Create experiment name using Databricks user pattern
    username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()    
    experiment_name = f"/Users/{username}/{config['model']['task_type']}_serverless_pipeline"
    
    # Create run name
    run_name = f"{config['model']['model_name']}-{config['model']['task_type']}-serverless-training"
    
    # Set up logging
    from utils.logging import create_databricks_logger
    logger = create_databricks_logger(
        experiment_name=experiment_name,
        run_name=run_name,
        tags={
            "task": config['model']['task_type'], 
            "model": config['model']['model_name'],
            "compute": "serverless_gpu",
            "gpu_type": config['training']['serverless_gpu_type'],
            "gpu_count": str(config['training']['serverless_gpu_count'])
        }
    )
    
    print(f"✅ Serverless training components initialized")
    print(f"   Task: {config['model']['task_type']}")
    print(f"   Model: {config['model']['model_name']}")
    print(f"   Max epochs: {config['training']['max_epochs']}")
    print(f"   Monitor metric: {config['training']['monitor_metric']}")
    print(f"   Distributed: {config['training']['distributed']}")
    print(f"   Serverless GPU: {config['training']['use_serverless_gpu']}")
    print(f"   GPU Type: {config['training']['serverless_gpu_type']}")
    print(f"   GPU Count: {config['training']['serverless_gpu_count']}")
    
    return model, data_module, logger, trainer_config

model, data_module, logger, trainer_config = setup_serverless_trainer()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify Serverless GPU Environment

# COMMAND ----------

def verify_serverless_environment():
    """Verify that the serverless GPU environment is properly configured."""
    
    # Check for serverless_gpu module
    try:
        from serverless_gpu import distributed
        print("✅ serverless_gpu module available")
        serverless_available = True
    except ImportError:
        print("❌ serverless_gpu module not available")
        print("   This notebook requires Serverless GPU compute environment")
        serverless_available = False
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✅ CUDA available with {torch.cuda.device_count()} GPUs")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1e9
            print(f"   GPU {i}: {props.name} - {memory_gb:.1f} GB")
    else:
        print("❌ CUDA not available")
        serverless_available = False
    
    # Check configuration
    gpu_type = config['training']['serverless_gpu_type']
    gpu_count = config['training']['serverless_gpu_count']
    
    print(f"\nServerless GPU Configuration:")
    print(f"   GPU Type: {gpu_type}")
    print(f"   GPU Count: {gpu_count}")
    
    if gpu_type not in ['A10', 'H100']:
        print(f"❌ Invalid GPU type: {gpu_type}. Must be 'A10' or 'H100'")
        serverless_available = False
    
    if gpu_type == 'H100' and gpu_count > 1:
        print("⚠️  H100 GPUs only support single-node workflows")
        print("   Setting serverless_gpu_count to 1")
        config['training']['serverless_gpu_count'] = 1
    
    return serverless_available

serverless_ready = verify_serverless_environment()

if not serverless_ready:
    print("\n❌ Serverless GPU environment not ready. Please check the configuration.")
    dbutils.notebook.exit("Serverless GPU environment not ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Start Serverless GPU Training

# COMMAND ----------

# MAGIC %md
# MAGIC ### Clear GPU Memory and Start Training

# COMMAND ----------

def clear_gpu_memory():
    """Clear GPU memory before training."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 GPU memory cleared before training")
        
        # Show initial memory state
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"   GPU {i} - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

clear_gpu_memory()

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
def distributed_train(config_dict, DetectionModel, DetectionDataModule, UnifiedTrainer, MLFlowLogger):
    """Distributed training function for Serverless GPU using UnifiedTrainer."""
    import lightning as pl
    
    # Create model and data module inside distributed function using passed classes
    model = DetectionModel(config_dict['model_config'])
    data_module = DetectionDataModule(config_dict['data_config'])
    
    # Create logger
    logger = MLFlowLogger(
        experiment_name=config_dict.get('mlflow_experiment_name', 'serverless-gpu-training'),
        run_name=config_dict.get('mlflow_run_name', 'distributed-training'),
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
# MAGIC ### Launch Training

# COMMAND ----------

print("🎯 Starting Serverless GPU training...")
print("=" * 60)

try:
    print("✅ Serverless GPU configuration ready.")
    print("🚀 Starting distributed training...")
    
    # Run distributed training directly - pass the class definitions
    distributed_result = distributed_train.distributed(
        trainer_config,
        DetectionModel,
        DetectionDataModule, 
        UnifiedTrainer,
        MLFlowLogger
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
# MAGIC ## 5. Training Results and Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ### Get Training Metrics

# COMMAND ----------

def get_training_metrics():
    """Retrieve and display training metrics from MLflow."""
    
    try:
        # Get metrics from the trainer
        metrics = unified_trainer.get_metrics()
        
        print("📊 Training Metrics Summary:")
        print("=" * 40)
        
        # Display key metrics
        key_metrics = ['val_loss', 'val_map', 'train_loss']
        for metric in key_metrics:
            if metric in metrics:
                print(f"{metric}: {metrics[metric]:.4f}")
        
        # Display all available metrics
        print(f"\nAll available metrics ({len(metrics)} total):")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        return metrics
        
    except Exception as e:
        print(f"⚠️  Could not retrieve metrics: {e}")
        return {}

metrics = get_training_metrics()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Performance Analysis

# COMMAND ----------

def analyze_model_performance():
    """Analyze model performance and provide insights."""
    
    if not metrics:
        print("⚠️  No metrics available for analysis")
        return
    
    print("🔍 Model Performance Analysis:")
    print("=" * 40)
    
    # Check for key detection metrics
    if 'val_map' in metrics:
        val_map = metrics['val_map']
        print(f"Validation mAP: {val_map:.4f}")
        
        if val_map > 0.3:
            print("✅ Good performance (mAP > 0.3)")
        elif val_map > 0.2:
            print("⚠️  Moderate performance (mAP > 0.2)")
        else:
            print("❌ Poor performance (mAP < 0.2)")
    
    # Check for loss metrics
    if 'val_loss' in metrics:
        val_loss = metrics['val_loss']
        print(f"Validation Loss: {val_loss:.4f}")
        
        if val_loss < 1.0:
            print("✅ Low validation loss")
        elif val_loss < 2.0:
            print("⚠️  Moderate validation loss")
        else:
            print("❌ High validation loss")
    
    # Check for convergence
    if 'train_loss' in metrics and 'val_loss' in metrics:
        train_loss = metrics['train_loss']
        val_loss = metrics['val_loss']
        gap = abs(train_loss - val_loss)
        
        print(f"Train-Val Loss Gap: {gap:.4f}")
        if gap < 0.1:
            print("✅ Good convergence (small gap)")
        elif gap < 0.3:
            print("⚠️  Moderate convergence")
        else:
            print("❌ Poor convergence (large gap)")

analyze_model_performance()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Model Registration and Deployment

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register Model in MLflow

# COMMAND ----------

def register_model():
    """Register the trained model in MLflow Model Registry."""
    
    try:
        # Get the best checkpoint path
        checkpoint_dir = config['training']['checkpoint_dir']
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        
        if not checkpoint_files:
            print("⚠️  No checkpoint files found")
            return None
        
        # Find the best checkpoint (assuming it's the one with highest metric)
        best_checkpoint = None
        best_score = -1
        
        for checkpoint_file in checkpoint_files:
            # Extract score from filename if possible
            if 'val_map' in checkpoint_file:
                try:
                    score = float(checkpoint_file.split('val_map=')[1].split('-')[0])
                    if score > best_score:
                        best_score = score
                        best_checkpoint = checkpoint_file
                except:
                    pass
        
        if best_checkpoint is None:
            best_checkpoint = checkpoint_files[0]  # Use first checkpoint if no score found
        
        checkpoint_path = os.path.join(checkpoint_dir, best_checkpoint)
        
        print(f"📝 Registering model with checkpoint: {best_checkpoint}")
        
        # Register model
        model_name = f"{config['model']['model_name'].replace('/', '_')}_serverless"
        model_version = mlflow.register_model(
            model_uri=checkpoint_path,
            name=model_name
        )
        
        print(f"✅ Model registered successfully!")
        print(f"   Model name: {model_name}")
        print(f"   Version: {model_version.version}")
        print(f"   Stage: {model_version.current_stage}")
        
        return model_version
        
    except Exception as e:
        print(f"❌ Model registration failed: {e}")
        return None

model_version = register_model()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Summary and Next Steps

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training Summary

# COMMAND ----------

print("=" * 60)
print("SERVERLESS GPU TRAINING SUMMARY")
print("=" * 60)

print(f"✅ Model: {config['model']['model_name']}")
print(f"✅ Task: {config['model']['task_type']}")
print(f"✅ Training completed successfully")
print(f"✅ Serverless GPU: {config['training']['serverless_gpu_type']} x {config['training']['serverless_gpu_count']}")

if metrics:
    print(f"\n📊 Key Metrics:")
    if 'val_map' in metrics:
        print(f"   Validation mAP: {metrics['val_map']:.4f}")
    if 'val_loss' in metrics:
        print(f"   Validation Loss: {metrics['val_loss']:.4f}")

if model_version:
    print(f"\n📝 Model Registration:")
    print(f"   Model name: {model_version.name}")
    print(f"   Version: {model_version.version}")
    print(f"   Stage: {model_version.current_stage}")

print(f"\n📁 Output Locations:")
print(f"   Checkpoints: {CHECKPOINT_DIR}")
print(f"   Results: {RESULTS_DIR}")
print(f"   Logs: {LOGS_DIR}")

print(f"\n🔗 MLflow UI:")
print(f"   Experiment: {logger.experiment.name}")
print(f"   Run ID: {logger.run_id}")

print("\n🎉 Serverless GPU training completed successfully!")
print("\nNext steps:")
print("1. Run notebook 04_model_evaluation_serverless.py to evaluate the model")
print("2. Run notebook 05_model_registration_deployment.py to deploy the model")
print("3. Monitor model performance in MLflow UI")

print("=" * 60)
