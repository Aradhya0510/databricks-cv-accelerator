# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # 00. Setup and Configuration for DETR Training
# MAGIC 
# MAGIC This notebook sets up the environment and configuration for training a DETR (DEtection TRansformer) model for object detection on the COCO dataset using our modular computer vision framework.
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC **DETR (DEtection TRansformer)** is a novel approach to object detection that views detection as a direct set prediction problem. Unlike traditional methods like Faster R-CNN that use complex pipelines with region proposals and non-maximum suppression, DETR uses a transformer encoder-decoder architecture to directly predict bounding boxes and class labels.
# MAGIC 
# MAGIC ### Key Features of DETR:
# MAGIC - **End-to-end training**: No need for hand-designed components like anchor generation or NMS
# MAGIC - **Set prediction**: Uses bipartite matching loss for unique predictions
# MAGIC - **Transformer architecture**: Leverages self-attention for global reasoning
# MAGIC - **Simple pipeline**: Streamlined detection without specialized libraries
# MAGIC 
# MAGIC ## Prerequisites
# MAGIC 
# MAGIC Before running this notebook, ensure you have:
# MAGIC - **Databricks Runtime ML 16.4** or higher
# MAGIC - **Unity Catalog enabled** with access to a catalog, schema, and volume
# MAGIC - **4 GPU single-node cluster** (recommended: g5.24xlarge or similar)
# MAGIC - **COCO dataset** already uploaded to Unity Catalog volume in the format specified in the config
# MAGIC 
# MAGIC ## What This Notebook Does
# MAGIC 
# MAGIC 1. **Environment Setup**: Install dependencies and configure the environment
# MAGIC 2. **Unity Catalog Configuration**: Set up paths and permissions
# MAGIC 3. **Framework Initialization**: Import and configure the CV framework
# MAGIC 4. **Configuration Loading**: Load and validate the DETR config
# MAGIC 5. **System Verification**: Verify GPU availability and memory
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
# MAGIC ### Verify PyTorch Installation and GPU Availability

# COMMAND ----------

import torch
import torchvision
import lightning

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"Lightning version: {lightning.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")

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
PROJECT_PATH = "cv_detr_training"  # Project subdirectory within the volume

# Base paths
BASE_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{PROJECT_PATH}"

# Data paths (assuming COCO dataset is already uploaded)
DATA_PATH = f"{BASE_VOLUME_PATH}/data"
TRAIN_DATA_PATH = f"{DATA_PATH}/train2017"
TRAIN_ANNOTATION_FILE = f"{DATA_PATH}/instances_train2017.json"
VAL_DATA_PATH = f"{DATA_PATH}/val2017" 
VAL_ANNOTATION_FILE = f"{DATA_PATH}/instances_val2017.json"
TEST_DATA_PATH = f"{DATA_PATH}/test2017"
TEST_ANNOTATION_FILE = f"{DATA_PATH}/instances_test2017.json"

# Output paths
CHECKPOINT_DIR = f"{BASE_VOLUME_PATH}/checkpoints/detection"
RESULTS_DIR = f"{BASE_VOLUME_PATH}/results/detection"
LOGS_DIR = f"{BASE_VOLUME_PATH}/logs"

# Config path
CONFIG_PATH = f"{BASE_VOLUME_PATH}/configs/detection_detr_config.yaml"

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
# MAGIC Let's check if the COCO dataset is properly uploaded to Unity Catalog.

# COMMAND ----------

import os
from pathlib import Path

def check_data_availability():
    """Check if COCO dataset files are available in Unity Catalog."""
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
    print("Please ensure the COCO dataset is properly uploaded to Unity Catalog.")
    print("Expected structure:")
    print(f"  {DATA_PATH}/")
    print("  ├── train2017/ (images)")
    print("  ├── val2017/ (images)")
    print("  ├── test2017/ (images)")
    print("  ├── instances_train2017.json")
    print("  ├── instances_val2017.json")
    print("  └── instances_test2017.json")
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
from config import load_config, get_default_config
from tasks.detection.model import DetectionModel
from tasks.detection.data import DetectionDataModule
from training.trainer import UnifiedTrainer
from utils.logging import create_databricks_logger

print("✅ Framework components imported successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load and Validate Configuration

# COMMAND ----------

def load_and_validate_config():
    """Load the DETR configuration and validate it."""
    
    # Try to load the config file
    if os.path.exists(CONFIG_PATH):
        print(f"Loading configuration from: {CONFIG_PATH}")
        config = load_config(CONFIG_PATH)
    else:
        print(f"Config file not found at {CONFIG_PATH}")
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
    
    # Validate key configuration
    print("\nConfiguration Summary:")
    print(f"Model: {config['model']['model_name']}")
    print(f"Task Type: {config['model']['task_type']}")
    print(f"Number of Classes: {config['model']['num_classes']}")
    print(f"Batch Size: {config['data']['batch_size']}")
    print(f"Max Epochs: {config['training']['max_epochs']}")
    print(f"Learning Rate: {config['training']['learning_rate']}")
    print(f"Distributed Training: {config['training']['distributed']}")
    
    return config

config = load_and_validate_config()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Initialize Logging

# COMMAND ----------

# Set up logging
logger = create_databricks_logger(
    name="detr_training",
    experiment_name=config['mlflow']['experiment_name'],
    run_name=config['mlflow']['run_name'],
    log_file=f"{LOGS_DIR}/training.log"
)

logger.info("DETR Training Setup Complete")
logger.info(f"Configuration loaded: {config['model']['model_name']}")
logger.info(f"Training on {torch.cuda.device_count()} GPUs")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. System Verification

# COMMAND ----------

# MAGIC %md
# MAGIC ### GPU Memory and Performance Check

# COMMAND ----------

def verify_gpu_setup():
    """Verify GPU setup and memory availability."""
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. Training will be very slow on CPU.")
        return False
    
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
    
    # Estimate batch size based on memory
    total_memory_gb = sum([torch.cuda.get_device_properties(i).total_memory for i in range(torch.cuda.device_count())]) / 1e9
    estimated_batch_size = int(total_memory_gb * 2)  # Rough estimate
    
    print(f"\nEstimated optimal batch size: {estimated_batch_size} (across all GPUs)")
    print(f"Configured batch size: {config['data']['batch_size']}")
    
    if config['data']['batch_size'] > estimated_batch_size:
        print("⚠️  Warning: Configured batch size might be too large for available GPU memory")
    
    return True

gpu_ready = verify_gpu_setup()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Framework Compatibility Check

# COMMAND ----------

def check_framework_compatibility():
    """Check if all framework components are compatible."""
    
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
        
        # Test trainer creation
        trainer = UnifiedTrainer(config)
        print("✅ UnifiedTrainer created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Framework compatibility check failed: {e}")
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
print("DETR TRAINING SETUP SUMMARY")
print("=" * 60)

print(f"✅ Environment: PyTorch {torch.__version__}, Lightning {lightning.__version__}")
print(f"✅ GPU Setup: {torch.cuda.device_count()} GPUs available" if gpu_ready else "❌ GPU Setup: Issues detected")
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

if gpu_ready and data_available and framework_ready:
    print("\n🎉 SETUP COMPLETE! Ready to proceed to data preparation.")
    print("\nNext steps:")
    print("1. Run notebook 01_data_preparation.py to prepare the dataset")
    print("2. Run notebook 02_model_training.py to start training")
    print("3. Monitor training progress in MLflow UI")
else:
    print("\n⚠️  SETUP INCOMPLETE! Please resolve the issues above before proceeding.")

print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Understanding DETR Architecture
# MAGIC 
# MAGIC ### How DETR Works:
# MAGIC 
# MAGIC 1. **Backbone**: A CNN (ResNet-50) extracts features from the input image
# MAGIC 2. **Transformer Encoder**: Processes the feature map to understand global context
# MAGIC 3. **Object Queries**: Learnable embeddings that "query" the image for objects
# MAGIC 4. **Transformer Decoder**: Updates object queries through self-attention and cross-attention
# MAGIC 5. **Prediction Heads**: Linear layers predict bounding boxes and class labels
# MAGIC 6. **Bipartite Matching Loss**: Ensures unique predictions through Hungarian algorithm
# MAGIC 
# MAGIC ### Key Advantages:
# MAGIC - **End-to-end**: No hand-designed components like NMS or anchor generation
# MAGIC - **Parallel**: All predictions made simultaneously (not autoregressive)
# MAGIC - **Global**: Uses attention to understand relationships between objects
# MAGIC - **Simple**: Clean, interpretable architecture
# MAGIC 
# MAGIC ### Training Strategy:
# MAGIC - **Bipartite Matching**: Matches predictions to ground truth using Hungarian algorithm
# MAGIC - **Set Loss**: Combines classification and bounding box losses
# MAGIC - **Auxiliary Losses**: Help with convergence (optional)
# MAGIC - **Position Embeddings**: Help the model understand spatial relationships
# MAGIC 
# MAGIC This setup provides everything needed to train DETR effectively on COCO dataset using our modular framework! 