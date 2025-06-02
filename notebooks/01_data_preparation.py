# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Data Preparation
# MAGIC 
# MAGIC This notebook handles data preparation for computer vision tasks.
# MAGIC 
# MAGIC ## Unity Catalog Setup
# MAGIC 
# MAGIC The notebook expects data to be organized in the following structure within your Unity Catalog volume:
# MAGIC ```
# MAGIC /Volumes/cv_ref/datasets/coco_mini/
# MAGIC ├── data/
# MAGIC │   ├── train/
# MAGIC │   │   ├── images/
# MAGIC │   │   └── annotations.json
# MAGIC │   ├── val/
# MAGIC │   │   ├── images/
# MAGIC │   │   └── annotations.json
# MAGIC │   └── test/
# MAGIC │       ├── images/
# MAGIC │       └── annotations.json
# MAGIC ├── configs/
# MAGIC │   └── {task}_config.yaml
# MAGIC └── logs/
# MAGIC     └── data_prep.log
# MAGIC ```

# COMMAND ----------

# DBTITLE 1,Import Dependencies
import sys
import os
from pathlib import Path
import mlflow
import torch
from torch.utils.data import DataLoader
import yaml

# Add the project root to Python path
project_root = "/Workspace/Repos/Databricks_CV_ref"
sys.path.append(project_root)

# Import project modules
from src.utils.coco_handler import COCOHandler
from src.utils.logging import setup_logger
from src.tasks.detection.data import DetectionDataModule, DetectionDataConfig
from src.tasks.classification.data import ClassificationDataModule, ClassificationDataConfig
from src.tasks.segmentation.data import SegmentationDataModule, SegmentationDataConfig

# COMMAND ----------

# DBTITLE 1,Initialize Logging
# Get the Unity Catalog volume path from environment or use default
volume_path = os.getenv("UNITY_CATALOG_VOLUME", "/Volumes/cv_ref/datasets/coco_mini")
log_dir = f"{volume_path}/logs"
os.makedirs(log_dir, exist_ok=True)

logger = setup_logger(
    name="data_preparation",
    log_file=f"{log_dir}/data_prep.log"
)

# COMMAND ----------

# DBTITLE 1,Load Configuration
def load_task_config(task: str):
    """Load task-specific configuration."""
    config_path = f"{volume_path}/configs/{task}_config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

# COMMAND ----------

# DBTITLE 1,Setup Data Module
def setup_data_module(task: str, config):
    """Initialize data module with configuration."""
    data_modules = {
        'detection': (DetectionDataModule, DetectionDataConfig),
        'classification': (ClassificationDataModule, ClassificationDataConfig),
        'segmentation': (SegmentationDataModule, SegmentationDataConfig)
    }
    
    if task not in data_modules:
        raise ValueError(f"Unsupported task: {task}")
    
    DataModule, DataConfig = data_modules[task]
    
    data_config = DataConfig(
        data_path=config['data']['train_path'],
        annotation_file=config['data']['annotation_file'],
        image_size=config['data']['image_size'][0],  # Use first dimension as size
        mean=tuple(config['data']['normalize_mean']),
        std=tuple(config['data']['normalize_std']),
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        horizontal_flip=config['data']['augment']['horizontal_flip'],
        vertical_flip=config['data']['augment']['vertical_flip'],
        rotation=config['data']['augment']['rotation'],
        brightness_contrast=config['data']['augment']['brightness_contrast'],
        hue_saturation=config['data']['augment']['hue_saturation'],
        model_name=config['model']['model_name']
    )
    
    return DataModule(data_config)

# COMMAND ----------

# DBTITLE 1,Main Data Preparation Function
def prepare_data(task: str):
    """Main function to prepare data for training."""
    # Load configuration
    config = load_task_config(task)
    
    # Setup data module
    data_module = setup_data_module(task, config)
    
    # Setup data module for training
    data_module.setup('fit')
    
    # Get dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Log dataset statistics
    stats = {
        'train_samples': len(data_module.train_dataset),
        'val_samples': len(data_module.val_dataset),
        'num_classes': len(data_module.train_dataset.class_names),
        'class_names': data_module.train_dataset.class_names
    }
    
    # Save statistics
    stats_path = f"{volume_path}/results/{task}_dataset_stats.yaml"
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, 'w') as f:
        yaml.dump(stats, f)
    
    logger.info(f"Dataset statistics: {stats}")
    
    return train_loader, val_loader

# COMMAND ----------

# DBTITLE 1,Example Usage
# Example: Prepare data for detection task
task = "detection"

train_loader, val_loader = prepare_data(task)

# Display a sample batch
sample_batch = next(iter(train_loader))
print("Sample batch keys:", sample_batch.keys())
print("Image shape:", sample_batch['pixel_values'].shape)
print("Target shape:", sample_batch['labels'].shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC 1. Review the prepared data
# MAGIC 2. Check data statistics and distributions
# MAGIC 3. Proceed to model training notebook 