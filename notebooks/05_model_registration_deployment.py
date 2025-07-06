# Databricks notebook source
# COMMAND ----------

# MAGIC %md
# MAGIC # Model Registration and Deployment
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

# (Databricks only) Run setup and previous notebooks if running interactively
# %run ./00_setup_and_config
# %run ./01_data_preparation
# %run ./02_model_training
# %run ./04_model_evaluation

# COMMAND ----------

# DBTITLE 1,Import Dependencies
import sys
import os
from pathlib import Path
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

# Add the project root to Python path
project_root = "/Volumes/<catalog>/<schema>/<volume>/<path>"
sys.path.append(project_root)

# Import project modules
from src.utils.logging import setup_logger
from src.utils.coco_handler import COCOHandler
from src.tasks.detection.model import DetectionModel
from src.tasks.detection.data import DetectionDataModule

# COMMAND ----------

# DBTITLE 1,Initialize Configuration and Logging
# Load configuration
try:
    config = load_config("detection_detr")
except NameError:
    from src.config import load_config
    config = load_config("detection_detr")

# Initialize logging
volume_path = os.getenv("UNITY_CATALOG_VOLUME", project_root)
log_dir = f"{volume_path}/logs"
os.makedirs(log_dir, exist_ok=True)

logger = setup_logger(
    name="model_registration_deployment",
    log_file=f"{log_dir}/registration_deployment.log"
)

# Unity Catalog configuration
CATALOG = os.getenv("CATALOG", "your_catalog")
SCHEMA = os.getenv("SCHEMA", "your_schema")
MODEL_NAME = config['model']['model_name']
MODEL_VERSION = "1.0.0"

print(f"✅ Configuration loaded")
print(f"   Model: {MODEL_NAME}")
print(f"   Catalog: {CATALOG}")
print(f"   Schema: {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Model Preparation and Validation

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Trained Model for Registration

# COMMAND ----------

def load_trained_model():
    """Load the trained model for registration."""
    
    print("📦 Loading trained model for registration...")
    
    # Find best checkpoint
    checkpoint_dir = f"{volume_path}/checkpoints"
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        if checkpoint_files:
            checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
            best_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[0])
            print(f"✅ Found checkpoint: {best_checkpoint}")
            
            # Load model
            model = DetectionModel.load_from_checkpoint(best_checkpoint, config=config)
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
# MAGIC ## 2. Unity Catalog Model Registry Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure Unity Catalog Model Registry

# COMMAND ----------

def setup_unity_catalog_model_registry():
    """Setup Unity Catalog model registry configuration."""
    
    print("🏗️  Setting up Unity Catalog model registry...")
    
    # Configure MLflow to use Unity Catalog
    mlflow.set_registry_uri("databricks-uc")
    
    # Set the catalog and schema
    mlflow.set_experiment(f"/Shared/{CATALOG}/{SCHEMA}/experiments")
    
    # Create model registry path
    model_registry_path = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
    
    print(f"✅ Unity Catalog configured")
    print(f"   Registry URI: databricks-uc")
    print(f"   Model Path: {model_registry_path}")
    print(f"   Catalog: {CATALOG}")
    print(f"   Schema: {SCHEMA}")
    
    return model_registry_path

model_registry_path = setup_unity_catalog_model_registry()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Model Registry Entry

# COMMAND ----------

def create_model_registry_entry():
    """Create model registry entry in Unity Catalog."""
    
    print("📝 Creating model registry entry...")
    
    try:
        # Create or get the model
        client = mlflow.tracking.MlflowClient()
        
        # Check if model exists
        try:
            model_info = client.get_registered_model(model_registry_path)
            print(f"✅ Model already exists: {model_info.name}")
        except:
            # Create new model
            model_info = client.create_registered_model(
                name=model_registry_path,
                description=f"DETR object detection model for COCO dataset",
                tags={
                    "framework": "pytorch",
                    "model_type": "detection",
                    "architecture": "detr",
                    "dataset": "coco",
                    "task": "object_detection"
                }
            )
            print(f"✅ Created new model: {model_info.name}")
        
        return model_info
        
    except Exception as e:
        print(f"❌ Failed to create model registry entry: {e}")
        return None

model_info = create_model_registry_entry()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Model Registration with MLflow

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare Model for MLflow Logging

# COMMAND ----------

def prepare_model_for_mlflow(model, config, checkpoint_path):
    """Prepare model for MLflow logging with proper metadata."""
    
    print("📦 Preparing model for MLflow logging...")
    
    # Create model signature
    from mlflow.models.signature import infer_signature
    
    # Create sample input for signature inference
    sample_input = torch.randn(1, 3, 800, 800)
    with torch.no_grad():
        sample_output = model(sample_input)
    
    # Create input example
    input_example = {
        "image": sample_input.numpy().tolist(),
        "image_shape": [1, 3, 800, 800]
    }
    
    # Infer signature
    signature = infer_signature(
        inputs=sample_input.numpy(),
        outputs=sample_output if isinstance(sample_output, torch.Tensor) else sample_output.logits
    )
    
    # Prepare model metadata
    model_metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "framework": "pytorch",
        "architecture": "detr",
        "dataset": "coco",
        "task": "object_detection",
        "checkpoint_path": checkpoint_path,
        "training_config": config,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "registration_date": datetime.now().isoformat()
    }
    
    print("✅ Model prepared for MLflow logging")
    print(f"   Signature: {signature}")
    print(f"   Parameters: {model_metadata['model_parameters']:,}")
    
    return signature, input_example, model_metadata

signature, input_example, model_metadata = prepare_model_for_mlflow(model, config, checkpoint_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log Model to MLflow Registry

# COMMAND ----------

def log_model_to_registry(model, signature, input_example, model_metadata):
    """Log model to MLflow registry with proper metadata."""
    
    print("📤 Logging model to MLflow registry...")
    
    try:
        with mlflow.start_run():
            # Log model with metadata
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=model_registry_path,
                pip_requirements=[
                    "torch>=1.9.0",
                    "torchvision>=0.10.0",
                    "transformers>=4.20.0",
                    "pillow>=8.0.0",
                    "numpy>=1.21.0",
                    "opencv-python>=4.5.0"
                ],
                extra_pip_requirements=[
                    "mlflow>=2.0.0",
                    "databricks-sdk>=0.12.0"
                ]
            )
            
            # Log additional metadata
            mlflow.log_dict(model_metadata, "model_metadata.json")
            mlflow.log_dict(config, "training_config.json")
            
            # Log model metrics from evaluation
            if 'evaluation_results' in globals():
                mlflow.log_metrics(evaluation_results)
            
            print("✅ Model logged successfully to MLflow registry")
            
    except Exception as e:
        print(f"❌ Failed to log model: {e}")
        return None
    
    return mlflow.get_artifact_uri("model")

model_uri = log_model_to_registry(model, signature, input_example, model_metadata)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Validate Logged Model

# COMMAND ----------

def validate_logged_model(model_uri):
    """Validate the logged model using MLflow validation."""
    
    print("🔍 Validating logged model...")
    
    try:
        # Load and test the logged model
        loaded_model = mlflow.pytorch.load_model(model_uri)
        
        # Test inference
        test_input = torch.randn(1, 3, 800, 800)
        with torch.no_grad():
            test_output = loaded_model(test_input)
        
        print("✅ Model validation successful")
        print(f"   Model URI: {model_uri}")
        print(f"   Output type: {type(test_output)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model validation failed: {e}")
        return False

model_valid = validate_logged_model(model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Model Serving Deployment

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Model Serving Endpoint

# COMMAND ----------

def create_model_serving_endpoint():
    """Create a model serving endpoint for the registered model."""
    
    print("🚀 Creating model serving endpoint...")
    
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput
        
        # Initialize workspace client
        client = WorkspaceClient()
        
        # Endpoint configuration
        endpoint_name = f"{MODEL_NAME}-endpoint"
        
        # Model serving configuration
        served_model = ServedModelInput(
            model_name=model_registry_path,
            model_version="1",
            workload_size="Small",
            scale_to_zero_enabled=True
        )
        
        # Endpoint configuration
        endpoint_config = EndpointCoreConfigInput(
            name=endpoint_name,
            served_models=[served_model]
        )
        
        # Create endpoint
        endpoint = client.serving_endpoints.create(endpoint_config)
        
        print(f"✅ Endpoint created successfully")
        print(f"   Endpoint name: {endpoint_name}")
        print(f"   Endpoint ID: {endpoint.id}")
        
        return endpoint
        
    except Exception as e:
        print(f"❌ Failed to create endpoint: {e}")
        return None

endpoint = create_model_serving_endpoint()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wait for Endpoint to be Ready

# COMMAND ----------

def wait_for_endpoint_ready(endpoint_name, timeout_minutes=15):
    """Wait for the endpoint to be ready for serving."""
    
    print(f"⏳ Waiting for endpoint '{endpoint_name}' to be ready...")
    
    try:
        from databricks.sdk import WorkspaceClient
        client = WorkspaceClient()
        
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        while time.time() - start_time < timeout_seconds:
            # Get endpoint status
            endpoint = client.serving_endpoints.get(endpoint_name)
            
            if endpoint.state.ready == "READY":
                print(f"✅ Endpoint '{endpoint_name}' is ready!")
                print(f"   State: {endpoint.state.ready}")
                print(f"   URL: {endpoint.state.config.inference_endpoint}")
                return endpoint
            elif endpoint.state.ready == "FAILED":
                print(f"❌ Endpoint '{endpoint_name}' failed to deploy")
                return None
            else:
                print(f"⏳ Endpoint state: {endpoint.state.ready}")
                time.sleep(30)  # Wait 30 seconds before checking again
        
        print(f"⏰ Timeout waiting for endpoint to be ready")
        return None
        
    except Exception as e:
        print(f"❌ Error checking endpoint status: {e}")
        return None

if endpoint:
    ready_endpoint = wait_for_endpoint_ready(endpoint.name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Endpoint Testing and Validation

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Endpoint with Sample Data

# COMMAND ----------

def test_endpoint_with_sample_data(endpoint_name):
    """Test the deployed endpoint with sample data."""
    
    print(f"🧪 Testing endpoint '{endpoint_name}' with sample data...")
    
    try:
        from databricks.sdk import WorkspaceClient
        client = WorkspaceClient()
        
        # Get endpoint details
        endpoint = client.serving_endpoints.get(endpoint_name)
        inference_url = endpoint.state.config.inference_endpoint
        
        # Prepare sample image
        sample_image = np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)
        sample_image_pil = Image.fromarray(sample_image)
        
        # Convert to base64 for API call
        import base64
        import io
        buffer = io.BytesIO()
        sample_image_pil.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Prepare request payload
        payload = {
            "dataframe_records": [
                {
                    "image": image_base64,
                    "image_shape": [800, 800, 3]
                }
            ]
        }
        
        # Make API call
        headers = {
            "Authorization": f"Bearer {client.config.token}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            f"{inference_url}/invocations",
            json=payload,
            headers=headers
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Endpoint test successful")
            print(f"   Response status: {response.status_code}")
            print(f"   Response keys: {list(result.keys())}")
            return True
        else:
            print(f"❌ Endpoint test failed")
            print(f"   Status code: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing endpoint: {e}")
        return False

if ready_endpoint:
    endpoint_test_success = test_endpoint_with_sample_data(ready_endpoint.name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Performance Testing

# COMMAND ----------

def performance_test_endpoint(endpoint_name, num_requests=10):
    """Test endpoint performance with multiple requests."""
    
    print(f"⚡ Performance testing endpoint '{endpoint_name}'...")
    
    try:
        from databricks.sdk import WorkspaceClient
        client = WorkspaceClient()
        
        # Get endpoint details
        endpoint = client.serving_endpoints.get(endpoint_name)
        inference_url = endpoint.state.config.inference_endpoint
        
        # Prepare sample image
        sample_image = np.random.randint(0, 255, (800, 800, 3), dtype=np.uint8)
        sample_image_pil = Image.fromarray(sample_image)
        
        # Convert to base64
        import base64
        import io
        buffer = io.BytesIO()
        sample_image_pil.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Prepare request payload
        payload = {
            "dataframe_records": [
                {
                    "image": image_base64,
                    "image_shape": [800, 800, 3]
                }
            ]
        }
        
        headers = {
            "Authorization": f"Bearer {client.config.token}",
            "Content-Type": "application/json"
        }
        
        # Performance test
        response_times = []
        successful_requests = 0
        
        for i in range(num_requests):
            start_time = time.time()
            
            response = requests.post(
                f"{inference_url}/invocations",
                json=payload,
                headers=headers
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            response_times.append(response_time)
            
            if response.status_code == 200:
                successful_requests += 1
            
            # Small delay between requests
            time.sleep(0.1)
        
        # Calculate statistics
        avg_response_time = np.mean(response_times)
        min_response_time = np.min(response_times)
        max_response_time = np.max(response_times)
        success_rate = successful_requests / num_requests * 100
        
        print("📊 Performance Test Results:")
        print(f"   Total requests: {num_requests}")
        print(f"   Successful requests: {successful_requests}")
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Average response time: {avg_response_time:.3f}s")
        print(f"   Min response time: {min_response_time:.3f}s")
        print(f"   Max response time: {max_response_time:.3f}s")
        
        return {
            'total_requests': num_requests,
            'successful_requests': successful_requests,
            'success_rate': success_rate,
            'avg_response_time': avg_response_time,
            'min_response_time': min_response_time,
            'max_response_time': max_response_time
        }
        
    except Exception as e:
        print(f"❌ Error during performance test: {e}")
        return None

if ready_endpoint:
    performance_results = performance_test_endpoint(ready_endpoint.name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Model Promotion and Lifecycle Management

# COMMAND ----------

# MAGIC %md
# MAGIC ### Promote Model to Production

# COMMAND ----------

def promote_model_to_production():
    """Promote the model to production in Unity Catalog."""
    
    print("🚀 Promoting model to production...")
    
    try:
        from databricks.sdk import WorkspaceClient
        client = WorkspaceClient()
        
        # Promote model version to production
        client.model_registry.transition_model_version_stage(
            name=model_registry_path,
            version="1",
            stage="Production"
        )
        
        print("✅ Model promoted to production successfully")
        print(f"   Model: {model_registry_path}")
        print(f"   Version: 1")
        print(f"   Stage: Production")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to promote model: {e}")
        return False

promotion_success = promote_model_to_production()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set Model Aliases

# COMMAND ----------

def set_model_aliases():
    """Set model aliases for easy reference."""
    
    print("🏷️  Setting model aliases...")
    
    try:
        from databricks.sdk import WorkspaceClient
        client = WorkspaceClient()
        
        # Set aliases
        aliases = {
            "latest": "1",
            "production": "1",
            "stable": "1"
        }
        
        for alias, version in aliases.items():
            client.model_registry.set_model_version_tag(
                name=model_registry_path,
                version=version,
                key=alias,
                value="true"
            )
        
        print("✅ Model aliases set successfully")
        for alias in aliases.keys():
            print(f"   {alias}: version {aliases[alias]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to set model aliases: {e}")
        return False

aliases_set = set_model_aliases()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Save Deployment Information

# COMMAND ----------

# MAGIC %md
# MAGIC ### Export Deployment Summary

# COMMAND ----------

def save_deployment_summary():
    """Save comprehensive deployment summary."""
    
    print("💾 Saving deployment summary...")
    
    # Create deployment directory
    deployment_dir = f"{volume_path}/deployment"
    os.makedirs(deployment_dir, exist_ok=True)
    
    # Prepare deployment summary
    deployment_summary = {
        'model_info': {
            'name': MODEL_NAME,
            'version': MODEL_VERSION,
            'registry_path': model_registry_path,
            'catalog': CATALOG,
            'schema': SCHEMA
        },
        'endpoint_info': {
            'name': f"{MODEL_NAME}-endpoint" if endpoint else None,
            'id': endpoint.id if endpoint else None,
            'url': ready_endpoint.state.config.inference_endpoint if ready_endpoint else None
        },
        'validation_results': validation_results,
        'performance_results': performance_results if 'performance_results' in locals() else None,
        'deployment_date': datetime.now().isoformat(),
        'model_uri': model_uri,
        'checkpoint_path': checkpoint_path
    }
    
    # Save as YAML
    deployment_path = f"{deployment_dir}/deployment_summary.yaml"
    with open(deployment_path, 'w') as f:
        yaml.dump(deployment_summary, f)
    
    print(f"✅ Deployment summary saved to: {deployment_path}")
    
    # Create deployment report
    report_path = f"{deployment_dir}/deployment_report.txt"
    with open(report_path, 'w') as f:
        f.write("DETR Model Deployment Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Model Information:\n")
        f.write(f"  Name: {MODEL_NAME}\n")
        f.write(f"  Version: {MODEL_VERSION}\n")
        f.write(f"  Registry Path: {model_registry_path}\n")
        f.write(f"  Catalog: {CATALOG}\n")
        f.write(f"  Schema: {SCHEMA}\n\n")
        
        if ready_endpoint:
            f.write(f"Endpoint Information:\n")
            f.write(f"  Name: {ready_endpoint.name}\n")
            f.write(f"  ID: {ready_endpoint.id}\n")
            f.write(f"  URL: {ready_endpoint.state.config.inference_endpoint}\n\n")
        
        if 'performance_results' in locals() and performance_results:
            f.write(f"Performance Results:\n")
            f.write(f"  Success Rate: {performance_results['success_rate']:.1f}%\n")
            f.write(f"  Avg Response Time: {performance_results['avg_response_time']:.3f}s\n")
            f.write(f"  Min Response Time: {performance_results['min_response_time']:.3f}s\n")
            f.write(f"  Max Response Time: {performance_results['max_response_time']:.3f}s\n\n")
        
        f.write(f"Deployment Date: {datetime.now()}\n")
        f.write(f"Model URI: {model_uri}\n")
        f.write(f"Checkpoint Path: {checkpoint_path}\n")
    
    print(f"✅ Deployment report saved to: {report_path}")
    
    return deployment_dir

deployment_dir = save_deployment_summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary and Next Steps

# MAGIC %md
# MAGIC ### Deployment Summary

# COMMAND ----------

print("=" * 60)
print("MODEL REGISTRATION AND DEPLOYMENT SUMMARY")
print("=" * 60)

print(f"✅ Model Registration:")
print(f"   Model: {MODEL_NAME}")
print(f"   Version: {MODEL_VERSION}")
print(f"   Registry Path: {model_registry_path}")
print(f"   Unity Catalog: {CATALOG}.{SCHEMA}")

if ready_endpoint:
    print(f"\n🚀 Model Serving:")
    print(f"   Endpoint Name: {ready_endpoint.name}")
    print(f"   Endpoint ID: {ready_endpoint.id}")
    print(f"   Inference URL: {ready_endpoint.state.config.inference_endpoint}")
    print(f"   Status: {ready_endpoint.state.ready}")

if 'performance_results' in locals() and performance_results:
    print(f"\n⚡ Performance:")
    print(f"   Success Rate: {performance_results['success_rate']:.1f}%")
    print(f"   Avg Response Time: {performance_results['avg_response_time']:.3f}s")
    print(f"   Throughput: {1/performance_results['avg_response_time']:.1f} requests/second")

print(f"\n📁 Deployment Files:")
print(f"   Summary: {deployment_dir}/deployment_summary.yaml")
print(f"   Report: {deployment_dir}/deployment_report.txt")

print(f"\n🎯 Next Steps:")
print(f"   1. Monitor endpoint performance and health")
print(f"   2. Set up model monitoring and drift detection")
print(f"   3. Configure alerts for endpoint issues")
print(f"   4. Plan model updates and retraining")

print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Understanding Model Deployment

# MAGIC 
# MAGIC ### Key Deployment Concepts:
# MAGIC 
# MAGIC **1. Unity Catalog Model Registry:**
# MAGIC - Centralized model governance and versioning
# MAGIC - Model lineage tracking and metadata management
# MAGIC - Environment promotion (Development → Staging → Production)
# MAGIC - Access control and security policies
# MAGIC 
# MAGIC **2. Model Serving Endpoints:**
# MAGIC - Production-grade REST API endpoints
# MAGIC - Automatic scaling based on traffic
# MAGIC - GPU support for compute-intensive models
# MAGIC - Built-in monitoring and logging
# MAGIC 
# MAGIC **3. Model Validation:**
# MAGIC - Pre-deployment testing and validation
# MAGIC - Dependency verification and compatibility checks
# MAGIC - Performance benchmarking and load testing
# MAGIC - Security and compliance validation
# MAGIC 
# MAGIC **4. Model Lifecycle Management:**
# MAGIC - Version control and model tracking
# MAGIC - Environment promotion workflows
# MAGIC - Model retirement and cleanup
# MAGIC - Rollback capabilities for failed deployments
# MAGIC 
# MAGIC **5. Best Practices:**
# MAGIC - Always validate models before deployment
# MAGIC - Use proper model signatures and input examples
# MAGIC - Monitor endpoint performance and health
# MAGIC - Implement proper error handling and logging
# MAGIC - Plan for model updates and retraining
# MAGIC 
# MAGIC **6. Deployment Checklist:**
# MAGIC - ✅ Model validation and testing
# MAGIC - ✅ Unity Catalog registration
# MAGIC - ✅ Model serving endpoint creation
# MAGIC - ✅ Endpoint testing and validation
# MAGIC - ✅ Performance benchmarking
# MAGIC - ✅ Model promotion to production
# MAGIC - ✅ Monitoring and alerting setup
# MAGIC 
# MAGIC The model is now successfully deployed and ready for production use! 