# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Setup and Configuration
# MAGIC 
# MAGIC This notebook sets up the environment and loads configurations for the computer vision tasks.
# MAGIC 
# MAGIC ## Unity Catalog Setup
# MAGIC 
# MAGIC Before running this notebook, ensure you have:
# MAGIC 1. Created a Unity Catalog volume for your project
# MAGIC 2. Granted appropriate permissions to your user
# MAGIC 3. Mounted the volume in your workspace
# MAGIC 
# MAGIC Example Unity Catalog setup:
# MAGIC ```sql
# MAGIC -- Create catalog
# MAGIC CREATE CATALOG IF NOT EXISTS cv_ref;
# MAGIC 
# MAGIC -- Create schema
# MAGIC CREATE SCHEMA IF NOT EXISTS cv_ref.datasets;
# MAGIC 
# MAGIC -- Create volume
# MAGIC CREATE VOLUME IF NOT EXISTS cv_ref.datasets.coco_mini;
# MAGIC 
# MAGIC -- Grant permissions
# MAGIC GRANT ALL PRIVILEGES ON VOLUME cv_ref.datasets.coco_mini TO `aradhya.chouhan@databricks.com`;
# MAGIC ```

# COMMAND ----------

# MAGIC %pip install -r /Workspace/Repos/Databricks_CV_ref/requirements.txt

# COMMAND ----------

# DBTITLE 1,Import Dependencies
import sys
import os
from pathlib import Path
import yaml

# Add the project root to Python path
project_root = "/Volumes/<catalog>/<schema>/<volume>/<path>/<file_name>"
sys.path.append(project_root)

from src.config import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    get_default_config,
    save_config
)
from src.utils.logging import setup_logger, get_metric_logger
from src.tasks.detection.model import DetectionModel
from src.tasks.classification.model import ClassificationModel
from src.tasks.semantic_segmentation.model import SemanticSegmentationModel

# COMMAND ----------

# DBTITLE 1,Initialize Logging
# Get the Unity Catalog volume path from environment or use default
volume_path = os.getenv("UNITY_CATALOG_VOLUME",project_root)
log_dir = f"{volume_path}/logs"
os.makedirs(log_dir, exist_ok=True)

logger = setup_logger(
    name="cv_pipeline",
    log_file=f"{log_dir}/setup.log"
)

# COMMAND ----------

# DBTITLE 1,Load or Create Configuration
def setup_config(task: str, config_path: str = None):
    """Setup configuration for the specified task."""
    if config_path and os.path.exists(config_path):
        logger.info(f"Loading configuration from {config_path}")
        config = load_config(config_path)
    else:
        logger.info(f"Creating default configuration for {task}")
        config = get_default_config(task)
        
        # Update paths to use Unity Catalog volume
        config['training']['checkpoint_dir'] = f"{volume_path}/checkpoints/{task}"
        config['data']['train_path'] = f"{volume_path}/data/{task}/train"
        config['data']['val_path'] = f"{volume_path}/data/{task}/val"
        config['data']['test_path'] = f"{volume_path}/data/{task}/test"
        
        # Save default config
        if config_path:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            save_config(config, config_path)
            logger.info(f"Saved default configuration to {config_path}")
    
    return config

# COMMAND ----------

# DBTITLE 1,Setup MLflow
def setup_mlflow(experiment_name: str):
    """Setup MLflow experiment."""
    mlflow_logger = get_metric_logger(experiment_name)
    logger.info(f"Initialized MLflow experiment: {experiment_name}")
    return mlflow_logger

# COMMAND ----------

# DBTITLE 1,Main Setup Function
def setup_pipeline(task: str, experiment_name: str):
    """Main setup function for the pipeline."""
    # Create necessary directories in Unity Catalog volume
    os.makedirs(f"{volume_path}/logs", exist_ok=True)
    os.makedirs(f"{volume_path}/configs", exist_ok=True)
    os.makedirs(f"{volume_path}/checkpoints", exist_ok=True)
    os.makedirs(f"{volume_path}/results", exist_ok=True)
    
    # Setup configuration
    config_path = f"{volume_path}/configs/{task}_config.yaml"
    config = setup_config(task, config_path)
    
    # Setup MLflow
    mlflow_logger = setup_mlflow(experiment_name)
    
    # Log configuration
    mlflow_logger.log_params(config)
    
    return config, mlflow_logger

# COMMAND ----------

# DBTITLE 1,Example Usage
# Example: Setup detection pipeline
task = "detection"
experiment_name = "detection_pipeline"

config, mlflow_logger = setup_pipeline(task, experiment_name)

# Display configuration
print("Configuration:")
print(yaml.dump(config, default_flow_style=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC 1. Review and modify the configuration as needed
# MAGIC 2. Proceed to data preparation notebook
# MAGIC 3. Check MLflow experiment setup 