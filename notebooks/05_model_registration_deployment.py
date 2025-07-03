# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Model Registration and Deployment
# MAGIC 
# MAGIC This notebook demonstrates how to:
# MAGIC 1. Register trained models in Unity Catalog
# MAGIC 2. Create model serving endpoints
# MAGIC 3. Deploy models for real-time inference
# MAGIC 4. Test deployed endpoints
# MAGIC 5. Set up model versioning and staging

# COMMAND ----------

%pip install -r "../requirements.txt"
dbutils.library.restartPython()

# COMMAND ----------

%run ./01_data_preparation
%run ./02_model_training
%run ./04_model_evaluation

# COMMAND ----------

# DBTITLE 1,Import Dependencies
import sys
import os
from pathlib import Path
import yaml
import mlflow
import torch
import numpy as np
import pandas as pd
import requests
import json
from PIL import Image
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# Add the project root to Python path
project_root = "/Volumes/<catalog>/<schema>/<volume>/<path>"
sys.path.append(project_root)

# Import project modules
from src.utils.logging import setup_logger
from src.tasks.detection.model import DetectionModel
from src.tasks.detection.data import DetectionDataModule
from src.tasks.classification.model import ClassificationModel
from src.tasks.classification.data import ClassificationDataModule
from src.tasks.semantic_segmentation.model import SemanticSegmentationModel
from src.tasks.semantic_segmentation.data import SemanticSegmentationDataModule
from src.tasks.instance_segmentation.model import InstanceSegmentationModel
from src.tasks.instance_segmentation.data import InstanceSegmentationDataModule
from src.tasks.panoptic_segmentation.model import PanopticSegmentationModel
from src.tasks.panoptic_segmentation.data import PanopticSegmentationDataModule

# COMMAND ----------

# DBTITLE 1,Initialize Logging
volume_path = os.getenv("UNITY_CATALOG_VOLUME", project_root)
log_dir = f"{volume_path}/logs"
os.makedirs(log_dir, exist_ok=True)

logger = setup_logger(
    name="model_registration",
    log_file=f"{log_dir}/registration.log"
)

# COMMAND ----------

# DBTITLE 1,Load Configuration and Model
def load_task_config(task: str):
    """Load task-specific configuration."""
    config_path = f"{volume_path}/configs/{task}_config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def get_model_class(task: str):
    """Get the appropriate model class based on task."""
    model_classes = {
        'detection': DetectionModel,
        'classification': ClassificationModel,
        'semantic_segmentation': SemanticSegmentationModel,
        'instance_segmentation': InstanceSegmentationModel,
        'panoptic_segmentation': PanopticSegmentationModel
    }
    
    if task not in model_classes:
        raise ValueError(f"Unsupported task: {task}")
    
    return model_classes[task]

def load_best_model(task: str, model_path: str = None):
    """Load the best trained model for the task."""
    config = load_task_config(task)
    
    # Try to load best config first
    best_config_path = f"{volume_path}/configs/{task}_best_config.yaml"
    if os.path.exists(best_config_path):
        with open(best_config_path, 'r') as f:
            best_config = yaml.safe_load(f)
        config.update(best_config)
    
    # Load model from MLflow if no specific path provided
    if model_path is None:
        model_name = config['model']['model_name']
        model_uri = f"models:/{model_name}/Production"
        model = mlflow.pytorch.load_model(model_uri)
    else:
        model = torch.load(model_path, map_location='cpu')
    
    model.eval()
    return model, config

# COMMAND ----------

# DBTITLE 1,Register Model in Unity Catalog
def register_model_in_unity_catalog(task: str, model, config: dict):
    """Register model in Unity Catalog."""
    # Set up Unity Catalog paths
    catalog_name = os.getenv("UNITY_CATALOG_CATALOG", "hive_metastore")
    schema_name = os.getenv("UNITY_CATALOG_SCHEMA", "cv_models")
    model_name = f"{task}_{config['model']['model_name']}"
    
    # Register model in Unity Catalog
    with mlflow.start_run(run_name=f"{task}_model_registration") as run:
        # Log model to Unity Catalog
        mlflow.pytorch.log_model(
            model,
            "model",
            registered_model_name=f"{catalog_name}.{schema_name}.{model_name}"
        )
        
        # Log model configuration
        mlflow.log_dict(config, "config.yaml")
        
        # Log model metadata
        mlflow.log_param("task", task)
        mlflow.log_param("model_name", config['model']['model_name'])
        mlflow.log_param("model_version", config['model'].get('version', '1.0'))
        
        # Log evaluation metrics if available
        metrics_path = f"{volume_path}/results/{task}_evaluation_results.yaml"
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = yaml.safe_load(f)
            mlflow.log_metrics(metrics)
    
    logger.info(f"Model registered: {catalog_name}.{schema_name}.{model_name}")
    return f"{catalog_name}.{schema_name}.{model_name}"

# COMMAND ----------

# DBTITLE 1,Create Model Serving Endpoint
def create_serving_endpoint(task: str, model_name: str, config: dict):
    """Create a model serving endpoint."""
    # Define endpoint configuration
    endpoint_name = f"{task}-{config['model']['model_name']}-endpoint"
    
    endpoint_config = {
        "name": endpoint_name,
        "config": {
            "served_models": [{
                "name": f"{task}-model",
                "model_name": model_name,
                "model_version": "1",
                "workload_size": config.get('deployment', {}).get('workload_size', 'Small'),
                "scale_to_zero_enabled": config.get('deployment', {}).get('scale_to_zero', True)
            }]
        }
    }
    
    try:
        # Create endpoint
        endpoint = mlflow.deployments.create_endpoint(
            name=endpoint_config["name"],
            config=endpoint_config["config"]
        )
        
        logger.info(f"Endpoint created: {endpoint.name}")
        return endpoint
    except Exception as e:
        logger.error(f"Failed to create endpoint: {e}")
        return None

# COMMAND ----------

# DBTITLE 1,Prepare Input for Inference
def prepare_input_for_inference(image_path: str, task: str, config: dict):
    """Prepare image input for model inference."""
    image = Image.open(image_path).convert('RGB')
    
    # Resize image if needed
    image_size = config['data'].get('image_size', 640)
    if isinstance(image_size, list):
        image_size = image_size[0]
    
    image = image.resize((image_size, image_size))
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Normalize if specified
    if 'normalize_mean' in config['data'] and 'normalize_std' in config['data']:
        mean = np.array(config['data']['normalize_mean'])
        std = np.array(config['data']['normalize_std'])
        image_array = (image_array / 255.0 - mean) / std
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

# COMMAND ----------

# DBTITLE 1,Test Endpoint
def test_endpoint(endpoint_name: str, image_path: str, task: str, config: dict):
    """Test the deployed endpoint with a sample image."""
    try:
        # Prepare image
        image_array = prepare_input_for_inference(image_path, task, config)
        
        # Make prediction
        response = mlflow.deployments.predict(
            endpoint_name=endpoint_name,
            inputs={"instances": image_array.tolist()}
        )
        
        logger.info(f"Endpoint test successful: {response}")
        return response
    except Exception as e:
        logger.error(f"Endpoint test failed: {e}")
        return None

def visualize_endpoint_predictions(image_path: str, predictions: dict, task: str):
    """Visualize predictions from the deployed endpoint."""
    image = Image.open(image_path).convert('RGB')
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    if task == 'detection':
        # Draw detection predictions
        for pred in predictions.get('predictions', []):
            box = pred.get('bbox', [])
            score = pred.get('score', 0)
            label = pred.get('label', 'unknown')
            
            if len(box) == 4:
                x1, y1, x2, y2 = box
                plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "r-", linewidth=2)
                plt.text(x1, y1, f"{label}: {score:.2f}", 
                        color="white", bbox=dict(facecolor="red", alpha=0.5))
    
    elif task == 'classification':
        # Show classification prediction
        pred = predictions.get('predictions', [{}])[0]
        predicted_class = pred.get('predicted_class', 'unknown')
        confidence = pred.get('confidence', 0)
        
        plt.text(10, 10, f"Class: {predicted_class}\nConfidence: {confidence:.2f}", 
                color="white", fontsize=14, bbox=dict(facecolor="black", alpha=0.7))
    
    plt.axis("off")
    plt.show()

# COMMAND ----------

# DBTITLE 1,Set Up Model Versioning
def setup_model_versioning(model_name: str, stage: str = "Production"):
    """Set up model versioning and staging."""
    try:
        # Get latest model version
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(model_name)[0]
        
        # Update model version stage
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version.version,
            stage=stage
        )
        
        logger.info(f"Model version {latest_version.version} moved to {stage} stage")
        return latest_version.version
    except Exception as e:
        logger.error(f"Failed to set up model versioning: {e}")
        return None

# COMMAND ----------

# DBTITLE 1,Main Registration and Deployment Function
def register_and_deploy_model(task: str, model_path: str = None):
    """Main function to register and deploy a model."""
    # Load best model and configuration
    model, config = load_best_model(task, model_path)
    
    # Register model in Unity Catalog
    model_name = register_model_in_unity_catalog(task, model, config)
    
    # Create serving endpoint
    endpoint = create_serving_endpoint(task, model_name, config)
    
    if endpoint:
        # Set up model versioning
        version = setup_model_versioning(model_name)
        
        # Test endpoint with sample images
        test_images = [
            f"{volume_path}/data/test/images/sample1.jpg",
            f"{volume_path}/data/test/images/sample2.jpg",
            f"{volume_path}/data/test/images/sample3.jpg"
        ]
        
        for image_path in test_images:
            if os.path.exists(image_path):
                predictions = test_endpoint(endpoint.name, image_path, task, config)
                if predictions:
                    visualize_endpoint_predictions(image_path, predictions, task)
        
        return {
            'model_name': model_name,
            'endpoint_name': endpoint.name,
            'version': version,
            'status': 'success'
        }
    else:
        return {
            'model_name': model_name,
            'endpoint_name': None,
            'version': None,
            'status': 'failed'
        }

# COMMAND ----------

# DBTITLE 1,Example Usage
# Example: Register and deploy detection model
task = "detection"

# Register and deploy model
deployment_result = register_and_deploy_model(task)

# Display results
print("Deployment Results:")
print(f"Model Name: {deployment_result['model_name']}")
print(f"Endpoint Name: {deployment_result['endpoint_name']}")
print(f"Version: {deployment_result['version']}")
print(f"Status: {deployment_result['status']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC 
# MAGIC 1. Verify endpoint functionality
# MAGIC 2. Set up monitoring and alerting
# MAGIC 3. Configure auto-scaling policies
# MAGIC 4. Proceed to model monitoring notebook 