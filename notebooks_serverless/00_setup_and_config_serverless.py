# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # 00. Setup and Configuration for Serverless GPU Training
# MAGIC 
# MAGIC This notebook sets up the environment and configuration for training computer vision models using **Databricks Serverless GPU compute**. This provides a specialized environment for deep learning workloads with automatic environment management and optimized GPU utilization.
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC **Databricks Serverless GPU** is a specialized compute option designed for deep learning workloads that provides:
# MAGIC - **Automatic Environment Management**: Pre-configured with ML frameworks
# MAGIC - **Optimized GPU Utilization**: Efficient resource allocation and scaling
# MAGIC - **Interactive Development**: Supports both interactive and batch workloads
# MAGIC - **Cost Optimization**: Pay only for actual GPU usage time
# MAGIC - **Multi-GPU Support**: Automatic distributed training setup
# MAGIC 
# MAGIC ### Key Features of Serverless GPU:
# MAGIC - **A10 and H100 Support**: Choose between different GPU types
# MAGIC - **Automatic Scaling**: Scale from single to multi-GPU seamlessly
# MAGIC - **Environment Isolation**: Clean, reproducible environments
# MAGIC - **Integrated MLflow**: Built-in experiment tracking and model management
# MAGIC - **Unity Catalog Integration**: Seamless data and model storage
# MAGIC 
# MAGIC ## Prerequisites
# MAGIC 
# MAGIC Before running this notebook, ensure you have:
# MAGIC - **Databricks Runtime ML 16.4** or higher with Serverless GPU support
# MAGIC - **Unity Catalog enabled** with access to a catalog, schema, and volume
# MAGIC - **Serverless GPU compute** configured (A10 or H100)
# MAGIC - **Dataset** already uploaded to Unity Catalog volume
# MAGIC - **serverless_gpu** module available (automatically provided)
# MAGIC 
# MAGIC ## What This Notebook Does
# MAGIC 
# MAGIC 1. **Environment Setup**: Verify Serverless GPU environment and dependencies
# MAGIC 2. **Unity Catalog Configuration**: Set up paths and permissions
# MAGIC 3. **Framework Initialization**: Import and configure the CV framework for serverless
# MAGIC 4. **Configuration Loading**: Load and validate the serverless config
# MAGIC 5. **System Verification**: Verify GPU availability and serverless capabilities
# MAGIC 
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Environment Setup and Dependencies

# COMMAND ----------

# (Databricks only) Install requirements and restart Python if running interactively
# %pip install -r "../requirements.txt"
# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify Serverless GPU Environment

# COMMAND ----------

import torch
import torchvision
import lightning

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"Lightning version: {lightning.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

# Check for serverless_gpu module
try:
    from serverless_gpu import distributed
    print("✅ serverless_gpu module available")
    _SERVERLESS_GPU_AVAILABLE = True
except ImportError:
    print("❌ serverless_gpu module not available")
    print("   This notebook requires Serverless GPU compute environment")
    _SERVERLESS_GPU_AVAILABLE = False

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"GPU {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Unity Catalog Configuration

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set Up Unity Catalog Paths
# MAGIC 
# MAGIC We need to configure the Unity Catalog paths for our project. Replace the placeholders with your actual catalog, schema, and volume names.

# COMMAND ----------

# Configuration for Unity Catalog
# Replace these with your actual Unity Catalog details
CATALOG = "your_catalog"  # Your Unity Catalog catalog name
SCHEMA = "your_schema"    # Your Unity Catalog schema name  
VOLUME = "your_volume"    # Your Unity Catalog volume name
PROJECT_PATH = "cv_serverless_training"  # Project subdirectory within the volume

# Base paths
BASE_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{PROJECT_PATH}"

# Data paths (assuming dataset is already uploaded)
DATA_PATH = f"{BASE_VOLUME_PATH}/data"
TRAIN_DATA_PATH = f"{DATA_PATH}/train"
TRAIN_ANNOTATION_FILE = f"{DATA_PATH}/train_annotations.json"
VAL_DATA_PATH = f"{DATA_PATH}/val" 
VAL_ANNOTATION_FILE = f"{DATA_PATH}/val_annotations.json"
TEST_DATA_PATH = f"{DATA_PATH}/test"
TEST_ANNOTATION_FILE = f"{DATA_PATH}/test_annotations.json"

# Output paths
CHECKPOINT_DIR = f"{BASE_VOLUME_PATH}/checkpoints"
RESULTS_DIR = f"{BASE_VOLUME_PATH}/results"
LOGS_DIR = f"{BASE_VOLUME_PATH}/logs"

# Config path
CONFIG_PATH = f"{BASE_VOLUME_PATH}/configs_serverless"

print("Unity Catalog Configuration:")
print(f"Base Volume Path: {BASE_VOLUME_PATH}")
print(f"Train Data Path: {TRAIN_DATA_PATH}")
print(f"Val Data Path: {VAL_DATA_PATH}")
print(f"Checkpoint Dir: {CHECKPOINT_DIR}")
print(f"Results Dir: {RESULTS_DIR}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verify Data Availability
# MAGIC 
# MAGIC Let's check if the dataset is properly uploaded to Unity Catalog.

# COMMAND ----------

import os
from pathlib import Path

def check_data_availability():
    """Check if dataset files are available in Unity Catalog."""
    paths_to_check = [
        TRAIN_DATA_PATH,
        TRAIN_ANNOTATION_FILE,
        VAL_DATA_PATH,
        VAL_ANNOTATION_FILE,
        TEST_DATA_PATH,
        TEST_ANNOTATION_FILE
    ]
    
    print("Checking data availability in Unity Catalog:")
    all_available = True
    
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"✅ {path}")
            # Count files if it's a directory
            if os.path.isdir(path):
                file_count = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))])
                print(f"   └── {file_count} image files found")
        else:
            print(f"❌ {path} - NOT FOUND")
            all_available = False
    
    return all_available

data_available = check_data_availability()

if not data_available:
    print("\n⚠️  WARNING: Some data files are missing!")
    print("Please ensure the dataset is properly uploaded to Unity Catalog.")
    print("Expected structure:")
    print(f"  {DATA_PATH}/")
    print("  ├── train/ (images)")
    print("  ├── val/ (images)")
    print("  ├── test/ (images)")
    print("  ├── train_annotations.json")
    print("  ├── val_annotations.json")
    print("  └── test_annotations.json")
else:
    print("\n✅ All data files are available!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Framework Initialization

# COMMAND ----------

# MAGIC %md
# MAGIC ### Import Framework Components

# COMMAND ----------

import sys
import yaml
from pathlib import Path

# Add the src directory to Python path
sys.path.append('/Workspace/Repos/your-repo/Databricks_CV_ref/src')

# Import framework components
from databricks_cv_accelerator.config import load_config, get_default_config
from tasks.detection.model import DetectionModel
from tasks.detection.data import DetectionDataModule
from training.trainer_serverless import UnifiedTrainerServerless
from utils.logging import create_databricks_logger

print("✅ Framework components imported successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load and Validate Configuration

# COMMAND ----------

def load_and_validate_config():
    """Load the serverless configuration and validate it."""
    
    # Try to load the config file
    config_file = f"{CONFIG_PATH}/detection_detr_config.yaml"
    if os.path.exists(config_file):
        print(f"Loading configuration from: {config_file}")
        config = load_config(config_file)
    else:
        print(f"Config file not found at {config_file}")
        print("Using default configuration for detection task...")
        config = get_default_config("detection")
    
    # Update config with Unity Catalog paths
    config['data']['train_data_path'] = TRAIN_DATA_PATH
    config['data']['train_annotation_file'] = TRAIN_ANNOTATION_FILE
    config['data']['val_data_path'] = VAL_DATA_PATH
    config['data']['val_annotation_file'] = VAL_ANNOTATION_FILE
    config['data']['test_data_path'] = TEST_DATA_PATH
    config['data']['test_annotation_file'] = TEST_ANNOTATION_FILE
    config['training']['checkpoint_dir'] = CHECKPOINT_DIR
    config['output']['results_dir'] = RESULTS_DIR
    
    # Enable serverless GPU settings
    config['training']['use_serverless_gpu'] = True
    config['training']['distributed'] = True
    config['training']['use_ray'] = False
    
    # Validate key configuration
    print("\nConfiguration Summary:")
    print(f"Model: {config['model']['model_name']}")
    print(f"Task Type: {config['model']['task_type']}")
    print(f"Number of Classes: {config['model']['num_classes']}")
    print(f"Batch Size: {config['data']['batch_size']}")
    print(f"Max Epochs: {config['training']['max_epochs']}")
    print(f"Learning Rate: {config['training']['learning_rate']}")
    print(f"Distributed Training: {config['training']['distributed']}")
    print(f"Serverless GPU: {config['training']['use_serverless_gpu']}")
    print(f"GPU Type: {config['training']['serverless_gpu_type']}")
    print(f"GPU Count: {config['training']['serverless_gpu_count']}")
    
    return config

config = load_and_validate_config()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initialize Logging

# COMMAND ----------

# Get task from config
task = config['model']['task_type']
model_name = config['model']['model_name']

# Create experiment name using Databricks user pattern
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()    
experiment_name = f"/Users/{username}/{task}_serverless_pipeline"

# Create run name
run_name = f"{model_name}-{task}-serverless-training"

# Set up logging
logger = create_databricks_logger(
    experiment_name=experiment_name,
    run_name=run_name,
    tags={"task": "detection", "model": config['model']['model_name'], "compute": "serverless_gpu"}
)

print("✅ Logging setup complete!")
print(f"Experiment: {experiment_name}")
print(f"Run name: {run_name}")
print(f"Serverless GPU: {config['training']['serverless_gpu_type']} x {config['training']['serverless_gpu_count']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. System Verification

# COMMAND ----------

# MAGIC %md
# MAGIC ### Serverless GPU Capabilities Check

# COMMAND ----------

def verify_serverless_gpu_setup():
    """Verify Serverless GPU setup and capabilities."""
    
    if not _SERVERLESS_GPU_AVAILABLE:
        print("❌ Serverless GPU module not available")
        print("   This notebook requires Serverless GPU compute environment")
        return False
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Serverless GPU training requires CUDA.")
        return False
    
    print(f"✅ Serverless GPU environment ready")
    print(f"✅ CUDA available with {torch.cuda.device_count()} GPUs")
    
    # Check memory per GPU
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1e9
        print(f"GPU {i}: {props.name} - {memory_gb:.1f} GB")
        
        # Check available memory
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(i) / 1e9
        reserved = torch.cuda.memory_reserved(i) / 1e9
        print(f"  └── Allocated: {allocated:.1f} GB, Reserved: {reserved:.1f} GB")
    
    # Validate serverless GPU configuration
    gpu_type = config['training']['serverless_gpu_type']
    gpu_count = config['training']['serverless_gpu_count']
    
    print(f"\nServerless GPU Configuration:")
    print(f"  GPU Type: {gpu_type}")
    print(f"  GPU Count: {gpu_count}")
    
    if gpu_type not in ['A10', 'H100']:
        print(f"❌ Invalid GPU type: {gpu_type}. Must be 'A10' or 'H100'")
        return False
    
    if gpu_type == 'H100' and gpu_count > 1:
        print("⚠️  H100 GPUs only support single-node workflows")
        print("   Setting serverless_gpu_count to 1")
        config['training']['serverless_gpu_count'] = 1
    
    return True

serverless_gpu_ready = verify_serverless_gpu_setup()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Framework Compatibility Check

# COMMAND ----------

def check_framework_compatibility():
    """Check if all framework components are compatible with serverless GPU."""
    
    try:
        # Prepare model config with num_workers from data config
        model_config = config["model"].copy()
        model_config["num_workers"] = config["data"]["num_workers"]
        
        # Test model creation
        model = DetectionModel(model_config)
        print("✅ DetectionModel created successfully")
        
        # Setup adapter first
        from tasks.detection.adapters import get_input_adapter
        adapter = get_input_adapter(config["model"]["model_name"], image_size=config["data"].get("image_size", 800))
        if adapter is None:
            print("❌ Failed to create adapter")
            return False
        
        # Create data module with data config only
        data_module = DetectionDataModule(config["data"])
        
        # Assign adapter to data module
        data_module.adapter = adapter
        
        # Setup the data module to create datasets
        data_module.setup()
        print("✅ DetectionDataModule created successfully")
        
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
            'serverless_gpu_count': config['training']['serverless_gpu_count']
        }

        # Test serverless trainer creation
        unified_trainer = UnifiedTrainerServerless(
            config=trainer_config,
            model=model,
            data_module=data_module,
            logger=logger
        )
        print("✅ UnifiedTrainerServerless created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Framework compatibility check failed: {e}")
        import traceback
        traceback.print_exc()
        return False

framework_ready = check_framework_compatibility()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Summary and Next Steps

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup Summary

# COMMAND ----------

print("=" * 60)
print("SERVERLESS GPU TRAINING SETUP SUMMARY")
print("=" * 60)

print(f"✅ Environment: PyTorch {torch.__version__}, Lightning {lightning.__version__}")
print(f"✅ Serverless GPU: {'Ready' if serverless_gpu_ready else 'Issues detected'}")
print(f"✅ Data Availability: {'Ready' if data_available else 'Missing files'}")
print(f"✅ Framework: {'Ready' if framework_ready else 'Issues detected'}")
print(f"✅ Configuration: {config['model']['model_name']} loaded")

print(f"\n📁 Unity Catalog Paths:")
print(f"   Data: {DATA_PATH}")
print(f"   Checkpoints: {CHECKPOINT_DIR}")
print(f"   Results: {RESULTS_DIR}")
print(f"   Logs: {LOGS_DIR}")

print(f"\n🔧 Training Configuration:")
print(f"   Model: {config['model']['model_name']}")
print(f"   Task: {config['model']['task_type']}")
print(f"   Classes: {config['model']['num_classes']}")
print(f"   Batch Size: {config['data']['batch_size']}")
print(f"   Max Epochs: {config['training']['max_epochs']}")
print(f"   Learning Rate: {config['training']['learning_rate']}")

print(f"\n🚀 Serverless GPU Configuration:")
print(f"   GPU Type: {config['training']['serverless_gpu_type']}")
print(f"   GPU Count: {config['training']['serverless_gpu_count']}")
print(f"   Distributed: {config['training']['distributed']}")

if serverless_gpu_ready and data_available and framework_ready:
    print("\n🎉 SETUP COMPLETE! Ready to proceed to serverless GPU training.")
    print("\nNext steps:")
    print("1. Run notebook 01_data_preparation.py to prepare the dataset")
    print("2. Run notebook 02_model_training_serverless.py to start serverless training")
    print("3. Monitor training progress in MLflow UI")
else:
    print("\n⚠️  SETUP INCOMPLETE! Please resolve the issues above before proceeding.")

print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Understanding Serverless GPU Training
# MAGIC 
# MAGIC ### How Serverless GPU Works:
# MAGIC 
# MAGIC 1. **Automatic Environment**: Pre-configured with ML frameworks and dependencies
# MAGIC 2. **Resource Management**: Automatic scaling and resource allocation
# MAGIC 3. **Distributed Training**: Seamless multi-GPU setup with @distributed decorator
# MAGIC 4. **Cost Optimization**: Pay only for actual GPU usage time
# MAGIC 5. **Integration**: Built-in MLflow and Unity Catalog support
# MAGIC 
# MAGIC ### Key Advantages:
# MAGIC - **No Infrastructure Management**: Automatic environment setup
# MAGIC - **Interactive Development**: Supports both interactive and batch workloads
# MAGIC - **Cost Effective**: Pay-per-use pricing model
# MAGIC - **Scalable**: Easy scaling from single to multi-GPU
# MAGIC - **Integrated**: Seamless integration with Databricks ecosystem
# MAGIC 
# MAGIC ### Training Strategy:
# MAGIC - **@distributed Decorator**: Handles distributed training setup automatically
# MAGIC - **Automatic Scaling**: Scales based on configured GPU count
# MAGIC - **Environment Isolation**: Clean, reproducible training environments
# MAGIC - **Built-in Monitoring**: Integrated with MLflow for experiment tracking
# MAGIC 
# MAGIC This setup provides everything needed to train models effectively using Databricks Serverless GPU compute!
