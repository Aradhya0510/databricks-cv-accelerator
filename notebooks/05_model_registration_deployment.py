# Databricks notebook source
# MAGIC %md
# MAGIC # 05. Model Registration and Deployment
# MAGIC 
# MAGIC This notebook demonstrates the complete model lifecycle management process:
# MAGIC 1. **Model Registration**: Register trained DETR model in Unity Catalog
# MAGIC 2. **Model Validation**: Validate model artifacts and dependencies
# MAGIC 3. **Model Serving**: Deploy model as a production endpoint
# MAGIC 4. **Endpoint Testing**: Test the deployed endpoint
# MAGIC 5. **Model Promotion**: Promote model across environments
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC **Model Lifecycle Management:**
# MAGIC - **Unity Catalog Integration**: Centralized model governance
# MAGIC - **MLflow Model Registry**: Version control and model tracking
# MAGIC - **Model Serving**: Production-grade API endpoints
# MAGIC - **Environment Promotion**: Staging → Production workflow
# MAGIC 
# MAGIC ### Key Concepts:
# MAGIC - **Model Registration**: Logging models with metadata and lineage
# MAGIC - **Model Validation**: Pre-deployment testing and validation
# MAGIC - **Model Serving**: REST API endpoints for real-time inference
# MAGIC - **Model Monitoring**: Performance tracking and drift detection
# MAGIC 
# MAGIC ## What This Notebook Does
# MAGIC 
# MAGIC 1. **Model Preparation**: Prepare trained model for registration
# MAGIC 2. **Unity Catalog Setup**: Configure model registry in Unity Catalog
# MAGIC 3. **Model Registration**: Log model with proper metadata
# MAGIC 4. **Model Validation**: Validate model artifacts and dependencies
# MAGIC 5. **Model Serving**: Deploy as production endpoint
# MAGIC 6. **Endpoint Testing**: Test deployed model functionality
# MAGIC 
# MAGIC ---

# COMMAND ----------

# (Databricks only) Install requirements and restart Python if running interactively
# %pip install -r "../requirements.txt"
# dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Import Dependencies and Load Configuration

# COMMAND ----------

import sys
import os
import yaml
import torch
import mlflow
import numpy as np
import pandas as pd
from PIL import Image
import json
import requests
from typing import Dict, List, Tuple, Any, Optional
import time
from datetime import datetime
from pathlib import Path

# Add the src directory to Python path
sys.path.append('/Workspace/Repos/your-repo/Databricks_CV_ref/src')

from config import load_config, get_default_config
from tasks.detection.model import DetectionModel
from tasks.detection.data import DetectionDataModule
from utils.logging import create_databricks_logger
from utils.coco_handler import COCOHandler

# Load configuration from previous notebooks
CATALOG = "your_catalog"
SCHEMA = "your_schema"  
VOLUME = "your_volume"
PROJECT_PATH = "cv_detr_training"

BASE_VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/{PROJECT_PATH}"
CONFIG_PATH = f"{BASE_VOLUME_PATH}/configs/detection_detr_config.yaml"

# Set up volume directories
DEPLOYMENT_RESULTS_DIR = f"{BASE_VOLUME_PATH}/deployment_results"
DEPLOYMENT_LOGS_DIR = f"{BASE_VOLUME_PATH}/logs"

# Create directories
os.makedirs(DEPLOYMENT_RESULTS_DIR, exist_ok=True)
os.makedirs(DEPLOYMENT_LOGS_DIR, exist_ok=True)

print(f"📁 Volume directories created:")
print(f"   Deployment Results: {DEPLOYMENT_RESULTS_DIR}")
print(f"   Deployment Logs: {DEPLOYMENT_LOGS_DIR}")

# Load configuration
if os.path.exists(CONFIG_PATH):
    config = load_config(CONFIG_PATH)
    # Update checkpoint directory to use volume path
    config['training']['checkpoint_dir'] = f"{BASE_VOLUME_PATH}/checkpoints"
    print(f"✅ Fixed config: updated checkpoint directory")
else:
    print("⚠️  Config file not found. Using default detection config.")
    config = get_default_config("detection")
    # Update checkpoint directory to use volume path
    config['training']['checkpoint_dir'] = f"{BASE_VOLUME_PATH}/checkpoints"

print("✅ Configuration loaded successfully!")
print(f"📁 Checkpoint directory: {config['training']['checkpoint_dir']}")

# Unity Catalog configuration
MODEL_NAME = config['model']['model_name']
MODEL_VERSION = "1.0.0"

# Sanitize model name for Unity Catalog (no forward slashes, special chars)
def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for Unity Catalog compatibility."""
    # Replace forward slashes and other invalid characters
    sanitized = model_name.replace('/', '_').replace('-', '_').replace('.', '_')
    # Remove any remaining special characters
    sanitized = ''.join(c for c in sanitized if c.isalnum() or c == '_')
    # Ensure it starts with a letter
    if sanitized and not sanitized[0].isalpha():
        sanitized = 'model_' + sanitized
    return sanitized

UNITY_CATALOG_MODEL_NAME = sanitize_model_name(MODEL_NAME)

# Initialize logging for deployment tracking

username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

deployment_logger = create_databricks_logger(
    experiment_name=f"/Users/{username}/deployment_pipeline",
    run_name="model_registration_deployment",
    log_model=False,  # No model logging for deployment
    tags={
        'task': 'deployment',
        'model': MODEL_NAME,
        'catalog': CATALOG,
        'schema': SCHEMA
    }
)

print(f"✅ Configuration loaded")
print(f"   Model: {MODEL_NAME}")
print(f"   Catalog: {CATALOG}")
print(f"   Schema: {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Model Preparation and Validation

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Trained Model for Registration

# COMMAND ----------

def load_trained_model():
    """Load the trained model for registration."""
    
    print("📦 Loading trained model for registration...")
    
    # Find best checkpoint
    checkpoint_dir = f"{BASE_VOLUME_PATH}/checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
            best_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[0])
            print(f"✅ Found checkpoint: {best_checkpoint}")
            
            # Load model
            # Prepare model config with num_workers from data config
            model_config = config["model"].copy()
            model_config["num_workers"] = config["data"]["num_workers"]
            
            model = DetectionModel.load_from_checkpoint(best_checkpoint, config=model_config)
            model.eval()
            
            print(f"✅ Model loaded successfully")
            print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            return model, best_checkpoint
        else:
            print("❌ No checkpoint files found")
            return None, None
    else:
        print("❌ Checkpoint directory not found")
        return None, None

model, checkpoint_path = load_trained_model()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Validate Model Artifacts

# COMMAND ----------

def validate_model_artifacts(model, config):
    """Validate model artifacts and dependencies."""
    
    print("🔍 Validating model artifacts...")
    
    validation_results = {
        'model_loaded': model is not None,
        'config_valid': config is not None,
        'model_parameters': sum(p.numel() for p in model.parameters()) if model else 0,
        'model_device': next(model.parameters()).device if model else None,
        'config_keys': list(config.keys()) if config else []
    }
    
    # Validate model structure
    if model:
        try:
            # Test forward pass with dummy input
            dummy_input = torch.randn(1, 3, 800, 800)
            with torch.no_grad():
                output = model(dummy_input)
            
            validation_results['forward_pass_success'] = True
            validation_results['output_shape'] = str(output.shape) if hasattr(output, 'shape') else str(type(output))
            
        except Exception as e:
            validation_results['forward_pass_success'] = False
            validation_results['forward_pass_error'] = str(e)
    
    # Validate configuration
    required_config_keys = ['model', 'data', 'training']
    validation_results['config_complete'] = all(key in config for key in required_config_keys)
    
    print("📊 Validation Results:")
    for key, value in validation_results.items():
        print(f"   {key}: {value}")
    
    return validation_results

validation_results = validate_model_artifacts(model, config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Unity Catalog Model Registry Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup Unity Catalog Model Registry

# COMMAND ----------

def setup_unity_catalog_model_registry():
    """Set up Unity Catalog model registry for model registration."""
    
    print("🏗️  Setting up Unity Catalog model registry...")
    
    try:
        # Set up Unity Catalog paths
        model_registry_path = f"{CATALOG}.{SCHEMA}.{UNITY_CATALOG_MODEL_NAME}"
        
        # Create model registry if it doesn't exist
        try:
            spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
            print(f"✅ Schema created/verified: {CATALOG}.{SCHEMA}")
        except Exception as e:
            print(f"⚠️  Schema creation failed: {e}")
        
        # Set up MLflow experiment for model registration
        experiment_name = f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/model_registry"
        mlflow.set_experiment(experiment_name)
        
        print(f"✅ Unity Catalog model registry setup complete")
        print(f"   Model registry path: {model_registry_path}")
        print(f"   Experiment: {experiment_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Unity Catalog setup failed: {e}")
        return False

registry_ready = setup_unity_catalog_model_registry()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Model Registry Entry

# COMMAND ----------

def create_model_registry_entry():
    """Create a model registry entry in Unity Catalog."""
    
    if not registry_ready:
        print("❌ Unity Catalog not ready")
        return False
    
    print("📝 Creating model registry entry...")
    
    try:
        # Create model registry entry with sanitized name
        model_registry_path = f"{CATALOG}.{SCHEMA}.{UNITY_CATALOG_MODEL_NAME}"
        
        # Set up MLflow model registry
        mlflow.set_registry_uri("databricks-uc")
        
        # Create model if it doesn't exist
        try:
            client = mlflow.tracking.MlflowClient()
            client.create_registered_model(
                name=model_registry_path,
                description=f"DETR model for object detection - {MODEL_NAME}"
            )
            print(f"✅ Model registry entry created: {model_registry_path}")
            print(f"   Original model name: {MODEL_NAME}")
            print(f"   Unity Catalog name: {UNITY_CATALOG_MODEL_NAME}")
        except Exception as e:
            print(f"⚠️  Model registry entry may already exist: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model registry entry creation failed: {e}")
        return False

registry_entry_created = create_model_registry_entry()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Model Registration

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare Model for MLflow Registration

# COMMAND ----------

def prepare_model_for_mlflow(model, config, checkpoint_path):
    """Prepare the model for MLflow registration."""
    
    if not model:
        print("❌ No model available for registration")
        return None, None, None
    
    print("📦 Preparing model for MLflow registration...")
    
    try:
        # Create model signature
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import Schema, TensorSpec
        
        # Define input schema (image tensor)
        input_schema = Schema([
            TensorSpec(np.dtype(np.float32), (-1, 3, 800, 800), name="image")
        ])
        
        # Define output schema (predictions) - matches standalone wrapper output
        output_schema = Schema([
            TensorSpec(np.dtype(np.float32), (-1, -1, 4), name="boxes"),
            TensorSpec(np.dtype(np.float32), (-1, -1), name="scores"),
            TensorSpec(np.dtype(np.int64), (-1, -1), name="labels")
        ])
        
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
        
        # Create input example
        input_example = {
            "image": torch.randn(1, 3, 800, 800).numpy()
        }
        
        # Create model metadata
        model_metadata = {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "framework": "pytorch",
            "task": "object_detection",
            "architecture": "DETR",
            "backbone": "resnet50",
            "num_classes": config['model']['num_classes'],
            "image_size": config['data'].get('image_size', 800),
            "confidence_threshold": config['model'].get('confidence_threshold', 0.7),
            "training_config": config,
            "checkpoint_path": checkpoint_path,
            "registration_date": datetime.now().isoformat()
        }
        
        print("✅ Model preparation completed")
        print(f"   Signature: {signature}")
        print(f"   Input example shape: {input_example['image'].shape}")
        print(f"   Metadata keys: {list(model_metadata.keys())}")
        
        return signature, input_example, model_metadata
        
    except Exception as e:
        print(f"❌ Model preparation failed: {e}")
        return None, None, None

signature, input_example, model_metadata = prepare_model_for_mlflow(model, config, checkpoint_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log Model to Registry

# COMMAND ----------

def log_model_to_registry(model, signature, input_example, model_metadata):
    """Log the model to Unity Catalog model registry."""
    
    if not all([model, signature, input_example, model_metadata]):
        print("❌ Missing required components for model registration")
        return None
    
    print("📤 Logging model to Unity Catalog registry...")
    
    try:
        # Set up MLflow experiment
        experiment_name = f"/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/model_registry"
        mlflow.set_experiment(experiment_name)
        
        # Create standalone model wrapper to avoid import issues
        print("🔧 Creating standalone model wrapper...")
        standalone_model = create_standalone_model_wrapper(model, config)
        
        # Create conda environment for serving (matching training environment)
        conda_env = {
            "name": "serving-env",
            "channels": ["conda-forge"],
            "dependencies": [
                "python=3.12",
                "pip",
                {
                    "pip": [
                        "mlflow==2.21.3",
                        "torch==2.6.0",
                        "torchvision==0.21.0",
                        "numpy==1.26.4",
                        "pillow==10.2.0",
                        "transformers==4.50.2",
                        "accelerate==1.5.2"
                    ]
                }
            ]
        }
        
        # Log model to MLflow with custom conda environment
        with mlflow.start_run(run_name=f"model_registration_{UNITY_CATALOG_MODEL_NAME}"):
            # Log standalone model with custom conda environment
            model_uri = mlflow.pytorch.log_model(
                pytorch_model=standalone_model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                conda_env=conda_env,
                registered_model_name=f"{CATALOG}.{SCHEMA}.{UNITY_CATALOG_MODEL_NAME}"
            )
            
            # Log metadata
            mlflow.log_params(model_metadata)
            
            # Log model artifacts
            mlflow.log_artifact(checkpoint_path, "checkpoint")
            
            print(f"✅ Model logged successfully")
            print(f"   Model URI: {model_uri}")
            print(f"   Registered model: {CATALOG}.{SCHEMA}.{UNITY_CATALOG_MODEL_NAME}")
            print(f"   Original model name: {MODEL_NAME}")
            print(f"   Conda environment: {conda_env['name']}")
            print(f"   Using standalone wrapper: ✅")
            
            return model_uri
            
    except Exception as e:
        print(f"❌ Model logging failed: {e}")
        return None

model_uri = log_model_to_registry(model, signature, input_example, model_metadata)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Validate Logged Model

# COMMAND ----------

def validate_logged_model(model_uri):
    """Validate the logged model in the registry."""
    
    if not model_uri:
        print("❌ No model URI available for validation")
        return False
    
    print("🔍 Validating logged model...")
    
    try:
        # Check if model_uri is a string or ModelInfo object
        if hasattr(model_uri, 'model_uri'):
            # It's a ModelInfo object, get the actual URI
            actual_uri = model_uri.model_uri
        else:
            # It's already a string URI
            actual_uri = model_uri
        
        print(f"   Using model URI: {actual_uri}")
        
        # Load model from registry
        loaded_model = mlflow.pytorch.load_model(actual_uri)
        
        # Move model to CPU for validation (to avoid device mismatch)
        loaded_model = loaded_model.cpu()
        loaded_model.eval()
        
        # Test inference on CPU
        test_input = torch.randn(1, 3, 800, 800)
        with torch.no_grad():
            test_output = loaded_model(test_input)
        
        print("✅ Model validation successful")
        print(f"   Model loaded from: {actual_uri}")
        print(f"   Model device: {next(loaded_model.parameters()).device}")
        print(f"   Input device: {test_input.device}")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output type: {type(test_output)}")
        
        # Check if output is a dictionary (standalone wrapper format)
        if isinstance(test_output, dict):
            print(f"   Output keys: {list(test_output.keys())}")
            if 'boxes' in test_output:
                print(f"   Boxes shape: {test_output['boxes'].shape}")
            if 'scores' in test_output:
                print(f"   Scores shape: {test_output['scores'].shape}")
            if 'labels' in test_output:
                print(f"   Labels shape: {test_output['labels'].shape}")
        else:
            print(f"   Output shape: {test_output.shape if hasattr(test_output, 'shape') else 'N/A'}")
        
        # Check model metadata
        client = mlflow.tracking.MlflowClient()
        model_info = client.get_registered_model(f"{CATALOG}.{SCHEMA}.{UNITY_CATALOG_MODEL_NAME}")
        
        print(f"   Model name: {model_info.name}")
        print(f"   Latest version: {model_info.latest_versions}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        print(f"   Model URI type: {type(model_uri)}")
        if hasattr(model_uri, '__dict__'):
            print(f"   Model URI attributes: {list(model_uri.__dict__.keys())}")
        return False

model_validated = validate_logged_model(model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Model Serving Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Model Serving Endpoint

# COMMAND ----------

def create_model_serving_endpoint():
    """Create a model serving endpoint for the registered model."""
    
    if not model_uri:
        print("❌ No model URI available for serving")
        return None
    
    print("🚀 Creating model serving endpoint...")
    
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput
        
        # Initialize workspace client
        client = WorkspaceClient()
        
        # Define endpoint configuration
        endpoint_name = f"detr-{UNITY_CATALOG_MODEL_NAME}"
        
        # Check if endpoint already exists
        try:
            existing_endpoint = client.serving_endpoints.get(endpoint_name)
            print(f"⚠️  Endpoint '{endpoint_name}' already exists")
            print(f"   Current state: {existing_endpoint.state}")
            
            # Ask user if they want to delete and recreate
            print(f"   Deleting existing endpoint to recreate with new model...")
            client.serving_endpoints.delete(endpoint_name)
            
            # Wait a bit for deletion to complete
            import time
            time.sleep(10)
            
        except Exception as e:
            print(f"   Endpoint doesn't exist or can't be accessed: {e}")
        
        # Get the latest model version automatically
        try:
            mlflow_client = mlflow.tracking.MlflowClient()
            model_info = mlflow_client.get_registered_model(f"{CATALOG}.{SCHEMA}.{UNITY_CATALOG_MODEL_NAME}")
            
            if model_info.latest_versions:
                latest_version = max([v.version for v in model_info.latest_versions])
                print(f"   Latest model version: {latest_version}")
            else:
                # If no versions found, try to get all versions
                all_versions = mlflow_client.search_model_versions(f"name='{CATALOG}.{SCHEMA}.{UNITY_CATALOG_MODEL_NAME}'")
                if all_versions:
                    latest_version = max([v.version for v in all_versions])
                    print(f"   Latest model version: {latest_version}")
                else:
                    print(f"   Warning: No model versions found, using version 1")
                    latest_version = "1"
        except Exception as e:
            print(f"   Warning: Could not get latest version, using version 1: {e}")
            latest_version = "1"
        
        # Try with minimal configuration first
        from databricks.sdk.service.serving import ServedModelInput, EndpointCoreConfigInput, ServedModelInputWorkloadSize
        
        served_model = ServedModelInput(
            model_name=f"{CATALOG}.{SCHEMA}.{UNITY_CATALOG_MODEL_NAME}",
            model_version=str(latest_version),
            workload_size=ServedModelInputWorkloadSize.SMALL,
            scale_to_zero_enabled=True
        )
        
        endpoint_config = EndpointCoreConfigInput(
            name=endpoint_name,
            served_models=[served_model]
        )
        
        print(f"   Endpoint config created:")
        print(f"   Name: {endpoint_name}")
        print(f"   Model: {CATALOG}.{SCHEMA}.{UNITY_CATALOG_MODEL_NAME}")
        print(f"   Workload size: Small")
        print(f"   Config type: {type(endpoint_config)}")
        print(f"   Config: {endpoint_config}")
        
        # Create endpoint - pass both name and config as required by API
        # Use WorkspaceClient for serving endpoints
        workspace_client = WorkspaceClient()
        endpoint = workspace_client.serving_endpoints.create(
            name=endpoint_name,
            config=endpoint_config
        )
        
        print(f"✅ Model serving endpoint created")
        print(f"   Endpoint name: {endpoint_name}")
        print(f"   Endpoint object: {endpoint}")
        print(f"   Endpoint attributes: {dir(endpoint)}")
        print(f"   Model: {CATALOG}.{SCHEMA}.{UNITY_CATALOG_MODEL_NAME}")
        print(f"   Original model name: {MODEL_NAME}")
        
        return endpoint_name
        
    except Exception as e:
        print(f"❌ Endpoint creation failed: {e}")
        return None

endpoint_name = create_model_serving_endpoint()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wait for Endpoint to be Ready

# COMMAND ----------

def wait_for_endpoint_ready(endpoint_name, timeout_minutes=15):
    """Wait for the endpoint to be ready for serving."""
    
    if not endpoint_name:
        print("❌ No endpoint name provided")
        return False
    
    print(f"⏳ Waiting for endpoint '{endpoint_name}' to be ready...")
    
    try:
        from databricks.sdk import WorkspaceClient
        
        client = WorkspaceClient()
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        while time.time() - start_time < timeout_seconds:
            try:
                endpoint = client.serving_endpoints.get(endpoint_name)
                
                # Check if endpoint is actually ready
                if hasattr(endpoint.state, 'ready') and endpoint.state.ready.value == 'READY':
                    print(f"✅ Endpoint '{endpoint_name}' is ready!")
                    print(f"   State: {endpoint.state}")
                    
                    # Try to get the inference URL safely
                    try:
                        # Debug endpoint structure
                        print(f"   Endpoint attributes: {dir(endpoint)}")
                        print(f"   State attributes: {dir(endpoint.state)}")
                        
                        # Try different ways to get the URL
                        if hasattr(endpoint.state, 'config') and hasattr(endpoint.state.config, 'inference_url'):
                            print(f"   URL: {endpoint.state.config.inference_url}")
                        elif hasattr(endpoint, 'config') and hasattr(endpoint.config, 'inference_url'):
                            print(f"   URL: {endpoint.config.inference_url}")
                        else:
                            print(f"   URL: Not available yet")
                    except Exception as e:
                        print(f"   URL: Error getting URL - {e}")
                    
                    return True
                else:
                    print(f"   Current state: {endpoint.state}")
                    print(f"   Ready status: {endpoint.state.ready if hasattr(endpoint.state, 'ready') else 'Unknown'}")
                    time.sleep(30)  # Wait 30 seconds before checking again
                    
            except Exception as e:
                print(f"   Checking endpoint status: {e}")
                time.sleep(30)
        
        print(f"❌ Endpoint '{endpoint_name}' did not become ready within {timeout_minutes} minutes")
        return False
        
    except Exception as e:
        print(f"❌ Error waiting for endpoint: {e}")
        return False

endpoint_ready = wait_for_endpoint_ready(endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Endpoint Testing

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Endpoint with Sample Data

# COMMAND ----------

def test_endpoint_with_sample_data(endpoint_name):
    """Test the deployed endpoint with sample data."""
    
    if not endpoint_name or not endpoint_ready:
        print("❌ Endpoint not ready for testing")
        return None
    
    print("🧪 Testing endpoint with sample data...")
    
    try:
        from databricks.sdk import WorkspaceClient
        
        client = WorkspaceClient()
        endpoint = client.serving_endpoints.get(endpoint_name)
        
        # Get endpoint URL
        endpoint_url = endpoint.state.config.inference_url
        
        # Create sample data
        sample_image = torch.randn(1, 3, 800, 800).numpy()
        
        # Prepare request payload
        payload = {
            "dataframe_records": [
                {
                    "image": sample_image.tolist()
                }
            ]
        }
        
        # Make prediction request
        headers = {
            "Authorization": f"Bearer {dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            f"{endpoint_url}/invocations",
            json=payload,
            headers=headers
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Endpoint test successful")
            print(f"   Response status: {response.status_code}")
            print(f"   Response keys: {list(result.keys())}")
            
            # Analyze predictions
            if 'predictions' in result:
                predictions = result['predictions']
                print(f"   Number of predictions: {len(predictions)}")
                
                if len(predictions) > 0:
                    first_pred = predictions[0]
                    print(f"   Prediction keys: {list(first_pred.keys())}")
                    
                    if 'boxes' in first_pred:
                        num_boxes = len(first_pred['boxes'])
                        print(f"   Number of detected objects: {num_boxes}")
            
            return result
        else:
            print(f"❌ Endpoint test failed")
            print(f"   Status code: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Endpoint testing failed: {e}")
        return None

test_results = test_endpoint_with_sample_data(endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Performance Test Endpoint

# COMMAND ----------

def performance_test_endpoint(endpoint_name, num_requests=10):
    """Perform performance testing on the endpoint."""
    
    if not endpoint_name or not endpoint_ready:
        print("❌ Endpoint not ready for performance testing")
        return None
    
    print(f"⚡ Performance testing endpoint with {num_requests} requests...")
    
    try:
        from databricks.sdk import WorkspaceClient
        
        client = WorkspaceClient()
        endpoint = client.serving_endpoints.get(endpoint_name)
        endpoint_url = endpoint.state.config.inference_url
        
        # Create sample data
        sample_image = torch.randn(1, 3, 800, 800).numpy()
        payload = {
            "dataframe_records": [
                {
                    "image": sample_image.tolist()
                }
            ]
        }
        
        headers = {
            "Authorization": f"Bearer {dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()}",
            "Content-Type": "application/json"
        }
        
        # Performance test
        response_times = []
        success_count = 0
        
        for i in range(num_requests):
            start_time = time.time()
            
            try:
                response = requests.post(
                    f"{endpoint_url}/invocations",
                    json=payload,
                    headers=headers,
                    timeout=30
                )
                
                end_time = time.time()
                response_time = end_time - start_time
                response_times.append(response_time)
                
                if response.status_code == 200:
                    success_count += 1
                    print(f"   Request {i+1}: {response_time:.3f}s ✅")
                else:
                    print(f"   Request {i+1}: {response_time:.3f}s ❌ ({response.status_code})")
                    
            except Exception as e:
                end_time = time.time()
                response_time = end_time - start_time
                response_times.append(response_time)
                print(f"   Request {i+1}: {response_time:.3f}s ❌ ({str(e)})")
        
        # Calculate statistics
        avg_response_time = np.mean(response_times)
        std_response_time = np.std(response_times)
        success_rate = success_count / num_requests * 100
        throughput = 1.0 / avg_response_time if avg_response_time > 0 else 0
        
        print(f"\n📊 Performance Test Results:")
        print(f"   Total requests: {num_requests}")
        print(f"   Successful requests: {success_count}")
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Average response time: {avg_response_time:.3f} ± {std_response_time:.3f}s")
        print(f"   Throughput: {throughput:.2f} requests/second")
        
        performance_results = {
            'total_requests': num_requests,
            'successful_requests': success_count,
            'success_rate': success_rate,
            'avg_response_time': avg_response_time,
            'std_response_time': std_response_time,
            'throughput': throughput,
            'response_times': response_times
        }
        
        return performance_results
        
    except Exception as e:
        print(f"❌ Performance testing failed: {e}")
        return None

performance_results = performance_test_endpoint(endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Model Promotion

# COMMAND ----------

# MAGIC %md
# MAGIC ### Promote Model to Production

# COMMAND ----------

def promote_model_to_production():
    """Promote the model to production stage."""
    
    if not model_uri:
        print("❌ No model URI available for promotion")
        return False
    
    print("🚀 Promoting model to production...")
    
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Transition model to production
        client.transition_model_version_stage(
            name=f"{CATALOG}.{SCHEMA}.{MODEL_NAME}",
            version="1",
            stage="Production"
        )
        
        print(f"✅ Model promoted to production")
        print(f"   Model: {CATALOG}.{SCHEMA}.{MODEL_NAME}")
        print(f"   Version: 1")
        print(f"   Stage: Production")
        
        return True
        
    except Exception as e:
        print(f"❌ Model promotion failed: {e}")
        return False

model_promoted = promote_model_to_production()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set Model Aliases

# COMMAND ----------

def set_model_aliases():
    """Set model aliases for easy access."""
    
    if not model_uri:
        print("❌ No model URI available for aliases")
        return False
    
    print("🏷️  Setting model aliases...")
    
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Set aliases
        aliases = {
            "latest": "1",
            "production": "1",
            "stable": "1"
        }
        
        for alias, version in aliases.items():
            try:
                client.set_registered_model_alias(
                    name=f"{CATALOG}.{SCHEMA}.{MODEL_NAME}",
                    alias=alias,
                    version=version
                )
                print(f"   ✅ Alias '{alias}' -> version '{version}'")
            except Exception as e:
                print(f"   ⚠️  Failed to set alias '{alias}': {e}")
        
        print(f"✅ Model aliases set successfully")
        return True
        
    except Exception as e:
        print(f"❌ Setting model aliases failed: {e}")
        return False

aliases_set = set_model_aliases()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Save Deployment Summary

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Deployment Summary

# COMMAND ----------

def save_deployment_summary():
    """Save a comprehensive deployment summary."""
    
    print("💾 Saving deployment summary...")
    
    # Create deployment summary
    deployment_summary = {
        'model_info': {
            'name': MODEL_NAME,
            'version': MODEL_VERSION,
            'uri': model_uri,
            'checkpoint_path': checkpoint_path
        },
        'registry_info': {
            'catalog': CATALOG,
            'schema': SCHEMA,
            'registered_model': f"{CATALOG}.{SCHEMA}.{MODEL_NAME}",
            'model_validated': model_validated
        },
        'serving_info': {
            'endpoint_name': endpoint_name,
            'endpoint_ready': endpoint_ready,
            'endpoint_url': f"https://<workspace-url>/serving-endpoints/{endpoint_name}/invocations" if endpoint_name else None
        },
        'test_results': {
            'test_successful': test_results is not None,
            'performance_results': performance_results
        },
        'promotion_info': {
            'model_promoted': model_promoted,
            'aliases_set': aliases_set
        },
        'deployment_date': datetime.now().isoformat(),
        'config': config
    }
    
    # Save summary
    summary_path = f"{DEPLOYMENT_RESULTS_DIR}/deployment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(deployment_summary, f, indent=2)
    
    print(f"✅ Deployment summary saved to: {summary_path}")
    
    # Save detailed results
    detailed_results = {
        'validation_results': validation_results,
        'test_results': test_results,
        'performance_results': performance_results,
        'model_metadata': model_metadata if 'model_metadata' in locals() else None
    }
    
    detailed_path = f"{DEPLOYMENT_RESULTS_DIR}/detailed_results.json"
    with open(detailed_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"✅ Detailed results saved to: {detailed_path}")
    
    return True

deployment_summary_saved = save_deployment_summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary and Next Steps

# COMMAND ----------

print("=" * 60)
print("MODEL REGISTRATION AND DEPLOYMENT SUMMARY")
print("=" * 60)

print(f"✅ Model Registration:")
print(f"   Model: {MODEL_NAME}")
print(f"   Version: {MODEL_VERSION}")
print(f"   Registry: {CATALOG}.{SCHEMA}.{MODEL_NAME}")
print(f"   Model validated: {'✅' if model_validated else '❌'}")

if endpoint_name:
    print(f"\n🚀 Model Serving:")
    print(f"   Endpoint: {endpoint_name}")
    print(f"   Status: {'✅ Ready' if endpoint_ready else '❌ Not ready'}")
    print(f"   Test successful: {'✅' if test_results else '❌'}")

if performance_results:
    print(f"\n⚡ Performance:")
    print(f"   Success rate: {performance_results.get('success_rate', 0):.1f}%")
    print(f"   Avg response time: {performance_results.get('avg_response_time', 0):.3f}s")
    print(f"   Throughput: {performance_results.get('throughput', 0):.2f} req/s")

print(f"\n📁 Results saved to:")
print(f"   Deployment results: {DEPLOYMENT_RESULTS_DIR}")
print(f"   Summary: {DEPLOYMENT_RESULTS_DIR}/deployment_summary.json")
print(f"   Detailed results: {DEPLOYMENT_RESULTS_DIR}/detailed_results.json")

print(f"\n🎯 Next steps:")
print(f"   1. Monitor endpoint performance and health")
print(f"   2. Set up model monitoring and alerting")
print(f"   3. Implement A/B testing if needed")
print(f"   4. Plan model updates and versioning strategy")

print("=" * 60) 

def create_standalone_model_wrapper(model, config):
    """Create a standalone model wrapper that doesn't depend on relative imports."""
    
    import torch
    import torch.nn as nn
    from transformers import AutoModelForObjectDetection, AutoConfig
    
    class StandaloneDetectionModel(nn.Module):
        """Standalone detection model wrapper for serving."""
        
        def __init__(self, model, config):
            super().__init__()
            self.model = model
            self.config = config
            self.confidence_threshold = config.get('confidence_threshold', 0.5)
            self.iou_threshold = config.get('iou_threshold', 0.5)
            self.max_detections = config.get('max_detections', 100)
            
        def forward(self, image):
            """
            Forward pass for serving.
            
            Args:
                image: Input image tensor of shape (batch_size, 3, height, width)
            
            Returns:
                dict: Dictionary containing predictions
                    - boxes: Bounding boxes (batch_size, num_detections, 4)
                    - scores: Confidence scores (batch_size, num_detections)
                    - labels: Class labels (batch_size, num_detections)
            """
            # Ensure model is in eval mode
            self.model.eval()
            
            # Move input to same device as model
            device = next(self.model.parameters()).device
            image = image.to(device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(pixel_values=image)
            
            # Extract predictions
            if hasattr(outputs, 'logits') and hasattr(outputs, 'pred_boxes'):
                # DETR-style outputs
                logits = outputs.logits
                pred_boxes = outputs.pred_boxes
                
                # Convert logits to probabilities
                probs = torch.softmax(logits, dim=-1)
                scores, labels = torch.max(probs, dim=-1)
                
                # Apply confidence threshold
                mask = scores > self.confidence_threshold
                
                # Filter predictions
                filtered_boxes = pred_boxes[mask]
                filtered_scores = scores[mask]
                filtered_labels = labels[mask]
                
                # Limit number of detections
                if len(filtered_boxes) > self.max_detections:
                    # Keep top detections by score
                    top_indices = torch.argsort(filtered_scores, descending=True)[:self.max_detections]
                    filtered_boxes = filtered_boxes[top_indices]
                    filtered_scores = filtered_scores[top_indices]
                    filtered_labels = filtered_labels[top_indices]
                
                return {
                    'boxes': filtered_boxes.unsqueeze(0),  # Add batch dimension
                    'scores': filtered_scores.unsqueeze(0),
                    'labels': filtered_labels.unsqueeze(0)
                }
            else:
                # Fallback for other model types
                return {
                    'boxes': torch.zeros(1, 0, 4),
                    'scores': torch.zeros(1, 0),
                    'labels': torch.zeros(1, 0, dtype=torch.long)
                }
    
    # Create the standalone wrapper
    standalone_model = StandaloneDetectionModel(model, config)
    
    return standalone_model 